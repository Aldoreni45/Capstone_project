import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from config.settings import settings
from custom_logging.logger import app_logger
from custom_logging.error_handler import AppBaseException

from loaders.pdf_loader import ResearchPaperLoader
from chunking.recursive import RecursiveChunker
from chunking.semantic import SemanticTextChunker
from chunking.parent_child import ParentChildChunker, AdaptiveChunker, SectionAwareChunker
from embeddings import get_embeddings, EmbeddingBenchmarker
from vectordb.weaviate_client import WeaviateVectorClient
from retrievers.factory import get_retriever
from retrievers.context_compression import ContextualCompressionRetriever, RuleBasedContextCompressor
from rerankers.cross_encoder import CrossEncoderReranker
from memory.session_manager import SessionMemoryManager
from prompts.templates import RAG_PROMPT_TEMPLATE, SYSTEM_PROMPT
from llm.groq_client import GroqLLMClient
from pydantic_models.responses import StructuredAnswer, RetrievedChunk
from utils.metrics import track_latency, MetricsRegistry
from utils.langsmith_tracker import LangSmithTracker
from utils.citation_generator import CitationGenerator
from query.classifier import QueryClassifier
from query.general_chat import GeneralChatHandler
from query.general_knowledge import GeneralKnowledgeHandler
from crag.retrieval_evaluator import RetrievalEvaluator, RetrievalEvaluationResult
from crag.query_rewriter import QueryRewriter
from crag.web_search import WebSearcher
from crag.context_merger import ContextMerger
from query_understanding import QueryUnderstandingOrchestrator
from langsmith import traceable

# Load tracing variables
LangSmithTracker.init_tracing()

class RAGPipeline:
    """The master pipeline orchestrating documents ingestion, indexing, retrieval, reranking, and generation."""

    def __init__(self, enable_query_understanding: Optional[bool] = None):
        self.loader = ResearchPaperLoader()
        self.reranker = CrossEncoderReranker()
        self.context_compressor = RuleBasedContextCompressor()
        
        # Query routing components
        self.query_classifier = QueryClassifier()
        self.general_chat_handler = GeneralChatHandler()
        self.general_knowledge_handler = GeneralKnowledgeHandler()
        self.citation_generator = CitationGenerator()
        
        # CRAG components
        self.retrieval_evaluator = RetrievalEvaluator()
        self.query_rewriter = QueryRewriter()
        self.web_searcher = WebSearcher()
        self.context_merger = ContextMerger()
        
        # Query Understanding Layer - Read from config
        if enable_query_understanding is None:
            enable_query_understanding = settings.get("query_understanding", "enabled", default=True)
        
        self.enable_query_understanding = enable_query_understanding
        if enable_query_understanding:
            embedding_model_type = settings.get("embeddings", "default_model", default="huggingface")
            
            # Read individual step configurations
            enable_preprocessing = settings.get("query_understanding", "enable_preprocessing", default=True)
            enable_spell_correction = settings.get("query_understanding", "enable_spell_correction", default=True)
            enable_normalization = settings.get("query_understanding", "enable_normalization", default=True)
            enable_semantic_normalization = settings.get("query_understanding", "enable_semantic_normalization", default=True)
            enable_concept_extraction = settings.get("query_understanding", "enable_concept_extraction", default=True)
            enable_query_rewriting = settings.get("query_understanding", "enable_query_rewriting", default=True)
            
            self.query_understanding = QueryUnderstandingOrchestrator(
                enable_preprocessing=enable_preprocessing,
                enable_spell_correction=enable_spell_correction,
                enable_normalization=enable_normalization,
                enable_semantic_normalization=enable_semantic_normalization,
                enable_concept_extraction=enable_concept_extraction,
                enable_query_rewriting=enable_query_rewriting,
                embedding_model_type=embedding_model_type
            )
            app_logger.info("Query Understanding Layer enabled")
        else:
            self.query_understanding = None
            app_logger.info("Query Understanding Layer disabled")
        
        # Cache of all document chunks loaded in the current app context (used for local BM25 and parent retriever)
        self.session_corpus: List[Any] = []
        
        # Session Memory manager
        self.memory_manager = SessionMemoryManager(settings.groq_api_key)

    @traceable(name="Paper Ingestion")
    def ingest_paper(
        self,
        file_path: str,
        chunking_strategy: str = "recursive",
        embedding_model_type: str = "bge",
        namespace: str = "default"
    ) -> Dict[str, Any]:
        """Loads, chunks, embeds, and indexes a PDF paper."""
        app_logger.info(f"Ingesting paper '{file_path}' (strategy={chunking_strategy}, embeddings={embedding_model_type})")
        
        with track_latency("ingest_paper"):
            # 1. Load PDF
            pages = self.loader.load_paper(file_path)
            
            # 2. Get embeddings model
            emb_model = get_embeddings(embedding_model_type)
            
            # 3. Apply chunking strategy
            chunking_strategy_lower = chunking_strategy.lower()
            if chunking_strategy_lower == "semantic":
                chunker = SemanticTextChunker(emb_model)
            elif chunking_strategy_lower == "parent_child":
                chunker = ParentChildChunker()
            elif chunking_strategy_lower == "adaptive":
                chunker = AdaptiveChunker(emb_model)
            elif chunking_strategy_lower == "section_aware":
                chunker = SectionAwareChunker()
            else:
                chunker = RecursiveChunker()
                
            chunks = chunker.split_documents(pages)
            
            # 4. Generate embeddings with batching optimization
            texts = [c.page_content for c in chunks]
            
            # Check dimensions or run benchmark if needed
            start_emb = datetime.now()
            
            # Use batch processing for embeddings (optimized for performance)
            batch_size = 32  # Optimal batch size for most embedding models
            embeddings_list = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = emb_model.embed_documents(batch_texts)
                embeddings_list.extend(batch_embeddings)
                
            emb_latency = (datetime.now() - start_emb).total_seconds() * 1000
            app_logger.info(f"Generated {len(texts)} embeddings in {emb_latency:.2f}ms (batch_size={batch_size})")
            
            # Add embedding model info to metadata
            for c in chunks:
                c.metadata["embedding_model"] = embedding_model_type
                
            # 5. Index in Weaviate with proper connection management
            vector_client = None
            try:
                vector_client = WeaviateVectorClient(embedding_model_type)
                vector_client.upsert_documents(chunks, embeddings_list, namespace=namespace)
                
                # Keep tracks of the chunks in local memory corpus
                self.session_corpus.extend(chunks)
                
                return {
                    "chunk_count": len(chunks),
                    "embedding_latency_ms": emb_latency,
                    "metadata": pages[0].metadata if pages else {}
                }
            finally:
                if vector_client:
                    vector_client.close()

    @traceable(name="RAG Pipeline")
    def answer_query(
        self,
        query: str,
        session_id: str,
        retriever_type: str = "cosine",
        embedding_model_type: str = "bge",
        llm_model: str = "llama-3.3-70b-versatile",
        memory_type: str = "buffer",
        namespace: str = "default",
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> Tuple[StructuredAnswer, List[RetrievedChunk], Dict[str, Any]]:
        """Executes query routing and RAG pipeline with intelligent mode selection."""
        
        # 1. Setup metrics and tracker session
        MetricsRegistry.clear()
        
        # 2. Apply Query Understanding Layer if enabled
        original_query = query
        if self.enable_query_understanding and self.query_understanding:
            with track_latency("query_understanding"):
                qu_result = self.query_understanding.understand(query)
                query = qu_result.final_query
                app_logger.info(
                    f"Query Understanding: '{original_query}' -> '{query}' "
                    f"(steps: {len(qu_result.pipeline_steps)}, time: {qu_result.processing_time_ms:.2f}ms)"
                )
        
        # 3. Check if documents are available (both local corpus and Weaviate)
        has_documents = len(self.session_corpus) > 0
        if not has_documents:
            # Also check Weaviate for ingested documents
            try:
                vector_client = WeaviateVectorClient(embedding_model_type)
                ingested_papers = vector_client.get_ingested_papers()
                has_documents = len(ingested_papers) > 0
                vector_client.close()
                app_logger.info(f"Local corpus empty, Weaviate has {len(ingested_papers)} papers")
            except Exception as e:
                app_logger.warning(f"Could not check Weaviate for documents: {e}")
                has_documents = False
        
        # 4. Classify query and determine response mode
        response_mode = self.query_classifier.get_response_mode(query, has_documents)
        
        app_logger.info(
            f"Query: '{query}' | Mode: {response_mode} | Has Documents: {has_documents}"
        )
        
        # 4. Route to appropriate handler
        if response_mode == "general_chat":
            return self._handle_general_chat(query, session_id, memory_type)
        else:
            # RAG mode - use existing RAG pipeline (everything else)
            return self._execute_rag_pipeline(
                query, session_id, retriever_type, embedding_model_type,
                llm_model, memory_type, namespace, filter_dict
            )
    
    @traceable(name="General Chat Handler")
    def _handle_general_chat(
        self, query: str, session_id: str, memory_type: str
    ) -> Tuple[StructuredAnswer, List[RetrievedChunk], Dict[str, Any]]:
        """Handles general chat (greetings, small talk)."""
        response = self.general_chat_handler.handle_general_chat(query)
        
        # Update memory
        memory = self.memory_manager.get_session_memory(session_id, memory_type)
        memory.add_message("user", query)
        memory.add_message("assistant", response)
        
        # Return structured response
        structured_answer = StructuredAnswer(
            answer=response,
            citations=[],
            confidence_score=1.0
        )
        
        return structured_answer, [], MetricsRegistry.get_metrics()
    
    @traceable(name="General Knowledge Handler")
    def _handle_general_knowledge(
        self, query: str, session_id: str, memory_type: str
    ) -> Tuple[StructuredAnswer, List[RetrievedChunk], Dict[str, Any]]:
        """Handles general knowledge questions without RAG."""
        memory = self.memory_manager.get_session_memory(session_id, memory_type)
        chat_history = memory.get_history()
        
        response = self.general_knowledge_handler.handle_general_knowledge(query, chat_history)
        
        # Update memory
        memory.add_message("user", query)
        memory.add_message("assistant", response)
        
        # Return structured response
        structured_answer = StructuredAnswer(
            answer=response,
            citations=[],
            confidence_score=0.8
        )
        
        return structured_answer, [], MetricsRegistry.get_metrics()
    
    @traceable(name="CRAG RAG Pipeline")
    def _execute_rag_pipeline(
        self,
        query: str,
        session_id: str,
        retriever_type: str,
        embedding_model_type: str,
        llm_model: str,
        memory_type: str,
        namespace: str,
        filter_dict: Optional[Dict[str, Any]]
    ) -> Tuple[StructuredAnswer, List[RetrievedChunk], Dict[str, Any]]:
        """Executes the full CRAG-enhanced RAG pipeline for document queries."""
        
        app_logger.info(
            f"Executing CRAG-enhanced RAG pipeline: '{query}' (session={session_id}, retriever={retriever_type}, "
            f"embedding={embedding_model_type}, llm={llm_model}, memory={memory_type})"
        )

        with track_latency("total_pipeline"):
            # 1. Retrieve history and format context
            memory = self.memory_manager.get_session_memory(session_id, memory_type)
            chat_history = memory.get_history()

            # 2. Initial retrieval with CRAG evaluation loop
            vector_client = None
            emb_model = get_embeddings(embedding_model_type)
            current_query = query
            retrieval_attempts = 0
            max_attempts = 2
            web_results = None
            top_chunks = []
            
            try:
                vector_client = WeaviateVectorClient(embedding_model_type)
                retriever = get_retriever(
                    retriever_type=retriever_type,
                    vector_client=vector_client,
                    embedding_model=emb_model,
                    groq_api_key=settings.groq_api_key,
                    corpus=self.session_corpus
                )
                
                # CRAG Retrieval Loop with proper flow (conservative corrections)
                query_rewritten = False
                web_search_used = False
                final_retrieval_quality = "GOOD"
                
                while retrieval_attempts < max_attempts:
                    retrieval_attempts += 1
                    
                    # Step 1: Retrieve chunks (Top 20)
                    with track_latency("retrieval"):
                        retrieved_chunks = retriever.retrieve(
                            query=current_query,
                            top_k=20,
                            namespace=namespace,
                            filter_dict=filter_dict
                        )
                    
                    # Step 2: Rerank chunks (Top 20 -> Top 5) using Cross Encoder
                    if retrieved_chunks:
                        with track_latency("reranking"):
                            top_chunks = self.reranker.rerank(
                                query=current_query,
                                chunks=retrieved_chunks,
                                top_n=5,
                                score_threshold=None  # Uses config default (0.0)
                            )
                    else:
                        top_chunks = []
                    
                    # Step 3: Evaluate retrieval quality using Answerability Check + Weighted Confidence Scoring
                    evaluation = self.retrieval_evaluator.evaluate(top_chunks, current_query)
                    final_retrieval_quality = evaluation.quality
                    
                    app_logger.info(
                        f"CRAG Attempt {retrieval_attempts}: Quality={evaluation.quality}, "
                        f"Answerable={evaluation.answerable}, Reason={evaluation.reason}, "
                        f"AvgScore={evaluation.avg_score:.3f}, Chunks={len(top_chunks)}"
                    )
                    
                    # Check answerability - if not answerable, try web search fallback
                    if not evaluation.answerable:
                        app_logger.warning(
                            f"Retrieval is NOT answerable - trying web search fallback. "
                            f"Reason: {evaluation.reason}"
                        )
                        
                        # Try web search as fallback
                        try:
                            with track_latency("web_search"):
                                web_results = self.web_searcher.search(query, num_results=5)
                            
                            if web_results:
                                web_search_used = True
                                app_logger.info(f"Web search fallback found {len(web_results)} results")
                                break  # Exit loop and use web results
                            else:
                                app_logger.warning("Web search fallback returned no results")
                        except Exception as e:
                            app_logger.error(f"Web search fallback failed: {e}")
                        
                        # If web search failed, return failure message
                        if not web_results:
                            failure_message = (
                                "The uploaded documents do not contain sufficient information to answer your question.\n\n"
                                "The retrieved context cannot answer your query.\n\n"
                                f"Retrieval Quality: BAD\n"
                                f"Reason: {evaluation.reason}\n"
                                f"Confidence: {evaluation.confidence:.1%}\n"
                                f"Chunks Retrieved: {evaluation.chunk_count}"
                            )
                            
                            return {
                                "answer": failure_message,
                                "citations": [],
                                "confidence": evaluation.confidence,
                                "retrieval_quality": "BAD",
                                "retrieval_reason": evaluation.reason,
                                "query_rewritten": query_rewritten,
                                "web_search_used": False
                            }
                    
                    # Answerable - proceed based on quality classification
                    # GOOD: Proceed normally with pipeline
                    if evaluation.quality == "GOOD":
                        app_logger.info(f"Retrieval quality is GOOD - proceeding with pipeline")
                        break
                    
                    # PARTIAL: Context partially answers the query - proceed with pipeline
                    # PARTIAL is VALID and should continue (no validation needed since answerability is confirmed)
                    elif evaluation.quality == "PARTIAL":
                        app_logger.info(
                            f"Retrieval quality is PARTIAL - proceeding with pipeline. "
                            f"Context partially answers the query. Reason: {evaluation.reason}"
                        )
                        break
                    
                    # BAD (edge case): Should not reach here since answerability check handles BAD
                    # But handle it for safety - try web search fallback
                    if evaluation.quality == "BAD":
                        app_logger.warning(
                            f"Retrieval quality is BAD - trying web search fallback. "
                            f"Reason: {evaluation.reason}"
                        )
                        
                        # Try web search as fallback
                        try:
                            with track_latency("web_search"):
                                web_results = self.web_searcher.search(query, num_results=5)
                            
                            if web_results:
                                web_search_used = True
                                app_logger.info(f"Web search fallback found {len(web_results)} results")
                                break  # Exit loop and use web results
                            else:
                                app_logger.warning("Web search fallback returned no results")
                        except Exception as e:
                            app_logger.error(f"Web search fallback failed: {e}")
                        
                        # If web search failed, return failure message
                        if not web_results:
                            failure_message = (
                                "The uploaded documents do not contain sufficient information to answer your question.\n\n"
                                "No relevant context was found in the indexed corpus.\n\n"
                                f"Retrieval Quality: BAD\n"
                                f"Reason: {evaluation.reason}\n"
                                f"Confidence: {evaluation.confidence:.1%}\n"
                                f"Chunks Retrieved: {evaluation.chunk_count}"
                            )
                            
                            return {
                                "answer": failure_message,
                                "citations": [],
                                "confidence": evaluation.confidence,
                                "retrieval_quality": "BAD",
                                "retrieval_reason": evaluation.reason,
                                "query_rewritten": query_rewritten,
                                "web_search_used": False
                            }
                
                # 3. Handle no context case after all attempts
                if not top_chunks and not web_results:
                    app_logger.info("No documents or web results retrieved, falling back to general knowledge")
                    return self._handle_general_knowledge(query, session_id, memory_type)

                # 5. Merge document and web context if both available
                if web_results and top_chunks:
                    app_logger.info("Merging document and web search context")
                    # Use document priority strategy
                    merged_context_text = self.context_merger.merge_contexts(
                        retrieved_chunks=top_chunks,
                        web_results=web_results,
                        strategy="document_priority"
                    )
                    # For CRAG with merged context, we use the merged text directly
                    context_text = merged_context_text
                    compressed_chunks = top_chunks  # Still use document chunks for citations
                elif web_results and not top_chunks:
                    # Web search only
                    app_logger.info("Using web search context only")
                    context_text = self.web_searcher.format_results_for_context(web_results)
                    compressed_chunks = []
                else:
                    # Document chunks only - apply compression
                    with track_latency("context_compression"):
                        compressed_chunks = self.context_compressor.compress_context(
                            query=current_query,
                            chunks=top_chunks,
                            target_length=3000  # Target 3000 characters for context
                        )
                        app_logger.info(f"Context compression: {len(top_chunks)} -> {len(compressed_chunks)} chunks")

                    # Build prompt from compressed chunks
                    context_blocks = []
                    for idx, chunk in enumerate(compressed_chunks):
                        context_blocks.append(
                            f"--- Source {idx+1} ---\n"
                            f"Paper Title: {chunk.title}\n"
                            f"Page: {chunk.page}\n"
                            f"Chunk ID: {chunk.chunk_id}\n"
                            f"Author: {chunk.author}\n"
                            f"Passage: {chunk.content}\n"
                        )
                    context_text = "\n".join(context_blocks)
                
                # 6. Build prompt with merged or document context
                rag_prompt = RAG_PROMPT_TEMPLATE.format(
                    context_text=context_text,
                    question=current_query if current_query != query else query
                )

                # 7. Generate answer
                llm_client = GroqLLMClient(model=llm_model)
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": rag_prompt}
                ]
                
                with track_latency("llm_generation"):
                    structured_answer = llm_client.generate_structured(
                        messages=messages,
                        response_model=StructuredAnswer
                    )

                # 8. Generate citations from document chunks only (web sources don't have chunk metadata)
                if compressed_chunks:
                    auto_citations = self.citation_generator.generate_citations(compressed_chunks, max_citations=3)
                    structured_answer.citations = auto_citations

                # 9. Update memory with turn
                memory.add_message("user", query)
                memory.add_message("assistant", structured_answer.answer)
                
                # Fetch pipeline metrics and add CRAG status
                pipeline_metrics = MetricsRegistry.get_metrics()
                pipeline_metrics["query_rewritten"] = query_rewritten
                pipeline_metrics["web_search_used"] = web_search_used
                pipeline_metrics["retrieval_quality"] = final_retrieval_quality
                
                return structured_answer, compressed_chunks, pipeline_metrics
                
            finally:
                # Ensure Weaviate connection is closed after query completes
                if vector_client:
                    vector_client.close()
    
    def _get_no_context_response(self) -> str:
        """Returns a helpful response when no documents are retrieved."""
        return """No relevant information was found in the uploaded research papers.

You may ask:
- Questions about the uploaded documents
- General AI or Machine Learning questions
- Research-related questions

I can help you with both document-specific queries and general knowledge questions."""
