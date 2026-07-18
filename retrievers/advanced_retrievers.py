import json
import re
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.embeddings import Embeddings
from pydantic_models.responses import RetrievedChunk
from retrievers.base import BasePaperRetriever
from retrievers.vector_retrievers import CosineSimilarityRetriever
from custom_logging.logger import app_logger
import httpx
from config.settings import settings

class MultiQueryRetriever(BasePaperRetriever):
    """Generates multiple query formulations using LLM and blends search results."""

    def __init__(self, base_retriever: BasePaperRetriever, groq_api_key: str):
        self.base_retriever = base_retriever
        self.groq_api_key = groq_api_key
        self.default_model = settings.get("llm", "groq", "default_model", default="llama-3.3-70b-versatile")

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        namespace: str = "default",     
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        app_logger.info(f"Running MultiQueryRetriever for query: '{query}'")
        
        # 1. Generate query variants
        variants = self._generate_query_variants(query)
        variants.append(query)  # Ensure original is included
        
        # 2. Retrieve for all queries
        all_results = []
        for q_variant in set(variants):
            try:
                results = self.base_retriever.retrieve(
                    query=q_variant,
                    top_k=top_k,
                    namespace=namespace,
                    filter_dict=filter_dict
                )
                all_results.append(results)
            except Exception as e:
                app_logger.warning(f"Failed retrieval for query variant '{q_variant}': {str(e)}")
                
        # 3. Blend and deduplicate (highest score first)
        deduped: Dict[str, RetrievedChunk] = {}
        for sub_list in all_results:
            for chunk in sub_list:
                if chunk.chunk_id not in deduped:
                    deduped[chunk.chunk_id] = chunk
                else:
                    # Keep the higher score
                    if chunk.score > deduped[chunk.chunk_id].score:
                        deduped[chunk.chunk_id] = chunk
                        
        sorted_chunks = sorted(deduped.values(), key=lambda x: x.score, reverse=True)
        return sorted_chunks[:top_k]

    def _generate_query_variants(self, query: str) -> List[str]:
        if not self.groq_api_key:
            return []
        
        # Get number of variants from config
        num_variants = settings.get("retrieval", "multi_query", "num_variants", default=5)
            
        prompt = f"""
You are an AI assistant tasked with generating {num_variants} alternative formulations of a user search query.
The goal is to retrieve research paper chunks that cover various aspects of the user's intent.

Generate diverse query variants that:
1. Use different terminology and synonyms
2. Focus on different aspects of the query
3. Vary the level of specificity
4. Include related scientific concepts
5. Use different sentence structures

User Query: "{query}"

Output ONLY {num_variants} alternative queries, one per line. Do not include numbering, punctuation, or extra conversational text.
"""
        try:
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.default_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 150
            }
            with httpx.Client() as client:
                response = client.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=10.0)
                if response.status_code == 200:
                    text = response.json()["choices"][0]["message"]["content"].strip()
                    lines = [line.strip() for line in text.split("\n") if line.strip()]
                    return lines[:num_variants]
        except Exception as e:
            app_logger.warning(f"Error generating query variants: {str(e)}")
        return []

class ParentDocumentRetriever(BasePaperRetriever):
    """
    Retrieves small chunks but reconstructs larger context
    by blending neighboring chunks from the same page and paper.
    """

    def __init__(self, base_retriever: BasePaperRetriever, all_corpus_chunks: List[RetrievedChunk] = None):
        self.base_retriever = base_retriever
        self.all_corpus_chunks = all_corpus_chunks or []

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        namespace: str = "default",
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        # 1. Retrieve the top relevant small chunks
        child_chunks = self.base_retriever.retrieve(
            query=query,
            top_k=top_k,
            namespace=namespace,
            filter_dict=filter_dict
        )
        
        # 2. Expand context. If we have corpus cache, find all chunks in the same page and merge them
        expanded_chunks = []
        for child in child_chunks:
            page_text = self._get_full_page_context(child)
            expanded_chunks.append(RetrievedChunk(
                content=page_text or child.content,  # Fallback to child text if page text not resolved
                score=child.score,
                title=child.title,
                page=child.page,
                chunk_id=child.chunk_id,
                source=child.source,
                author=child.author
            ))
        return expanded_chunks

    def _get_full_page_context(self, chunk: RetrievedChunk) -> Optional[str]:
        """Finds all chunks sharing same paper and page, then joins them sequentially."""
        if not self.all_corpus_chunks:
            return None
            
        page_chunks = [
            c for c in self.all_corpus_chunks
            if c.title == chunk.title and c.page == chunk.page
        ]
        if not page_chunks:
            return None
            
        # Re-sort using chunk index (usually derived from chunk_id string or metadata sequence)
        def get_index_key(c: RetrievedChunk) -> int:
            try:
                # Expecting chunk_id format like "source_p1_c2"
                match = re.search(r'_c(\d+)$', c.chunk_id)
                if match:
                    return int(match.group(1))
            except Exception:
                pass
            return 0
            
        page_chunks.sort(key=get_index_key)
        return "\n\n...[Page Segment]...\n\n".join([c.content for c in page_chunks])

class ContextualCompressionRetriever(BasePaperRetriever):
    """Compresses chunks using an LLM to keep only query-relevant sentences."""

    def __init__(self, base_retriever: BasePaperRetriever, groq_api_key: str):
        self.base_retriever = base_retriever
        self.groq_api_key = groq_api_key
        self.default_model = settings.get("llm", "groq", "default_model", default="llama-3.3-70b-versatile")

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        namespace: str = "default",
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        raw_chunks = self.base_retriever.retrieve(
            query=query,
            top_k=top_k,
            namespace=namespace,
            filter_dict=filter_dict
        )
        
        compressed_chunks = []
        for chunk in raw_chunks:
            compressed_text = self._compress_text(query, chunk.content)
            if len(compressed_text.strip()) > 20:
                compressed_chunks.append(RetrievedChunk(
                    content=compressed_text,
                    score=chunk.score,
                    title=chunk.title,
                    page=chunk.page,
                    chunk_id=chunk.chunk_id,
                    source=chunk.source,
                    author=chunk.author
                ))
            else:
                # If compression fails/empty, keep original
                compressed_chunks.append(chunk)
                
        return compressed_chunks

    def _compress_text(self, query: str, context: str) -> str:
        if not self.groq_api_key:
            return context
            
        prompt = f"""
Given the following query and text snippet, extract ONLY the sentences from the text snippet that are directly relevant to answering the query.
Do not summarize, do not paraphrase, do not add explanation. Output only the exact relevant sentences.
If no sentences are relevant, output nothing.

Query: "{query}"

Text Snippet:
---
{context}
---
"""
        try:
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.default_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 300
            }
            with httpx.Client() as client:
                response = client.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=12.0)
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            app_logger.warning(f"Failed contextual compression: {str(e)}")
        return context

class EnsembleRetriever(BasePaperRetriever):
    """Blends multiple retrievers using reciprocal rank weights."""

    def __init__(self, retrievers: List[BasePaperRetriever], weights: List[float] = None):
        self.retrievers = retrievers
        self.weights = weights or [1.0 / len(retrievers)] * len(retrievers)

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        namespace: str = "default",
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        all_results = []
        for retriever in self.retrievers:
            try:
                results = retriever.retrieve(query, top_k=top_k * 2, namespace=namespace, filter_dict=filter_dict)
                all_results.append(results)
            except Exception as e:
                app_logger.warning(f"Retriever in ensemble failed: {str(e)}")
                all_results.append([])
                
        rrf_scores: Dict[str, Dict[str, Any]] = {}
        for r_idx, results in enumerate(all_results):
            weight = self.weights[r_idx]
            for rank, chunk in enumerate(results):
                if chunk.chunk_id not in rrf_scores:
                    rrf_scores[chunk.chunk_id] = {
                        "chunk": chunk,
                        "rrf_score": 0.0
                    }
                # Standard RRF formula weighted
                rrf_scores[chunk.chunk_id]["rrf_score"] += weight * (1.0 / (60.0 + rank))
                
        sorted_rrf = sorted(rrf_scores.values(), key=lambda x: x["rrf_score"], reverse=True)
        
        final_chunks = []
        for item in sorted_rrf[:top_k]:
            chunk = item["chunk"]
            # Update score to RRF score
            chunk.score = item["rrf_score"]
            final_chunks.append(chunk)
            
        return final_chunks

class SelfQueryRetriever(BasePaperRetriever):
    """Uses LLM to generate structured query and metadata filters from natural language."""

    def __init__(self, base_retriever: BasePaperRetriever, groq_api_key: str):
        self.base_retriever = base_retriever
        self.groq_api_key = groq_api_key
        self.default_model = settings.get("llm", "groq", "default_model", default="llama-3.3-70b-versatile")

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        namespace: str = "default",
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        app_logger.info(f"Running SelfQueryRetriever on query: '{query}'")
        
        # Merge manual filter with self-generated filter if it exists
        llm_query, generated_filter = self._parse_query(query)
        
        merged_filter = {}
        if filter_dict:
            merged_filter.update(filter_dict)
        if generated_filter:
            merged_filter.update(generated_filter)
            
        app_logger.info(f"Self-Query Extracted Query: '{llm_query}', Filters: {merged_filter}")
        
        return self.base_retriever.retrieve(
            query=llm_query or query,
            top_k=top_k,
            namespace=namespace,
            filter_dict=merged_filter or None
        )

    def _parse_query(self, query: str) -> Tuple[str, Dict[str, Any]]:
        if not self.groq_api_key:
            return query, {}
            
        prompt = f"""
You are an advanced search parsing agent. You have access to metadata fields:
1. "paper_title" (string)
2. "author" (string)
3. "source" (string - file name)

Your task is to analyze the user search query and output a JSON containing:
1. "query": The semantic text query (stripped of filter expressions).
2. "filter": A JSON metadata filtering query.

Supported filtering operators:
- Equality match. Example: {{"paper_title": "Attention Is All You Need"}}

If no filters are mentioned, return an empty object for "filter".

User Query: "{query}"

Output ONLY valid raw JSON in this format:
{{
  "query": "semantic query here",
  "filter": {{}}
}}
"""
        try:
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.default_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 150
            }
            with httpx.Client() as client:
                response = client.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=10.0)
                if response.status_code == 200:
                    text = response.json()["choices"][0]["message"]["content"].strip()
                    text_clean = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
                    text_clean = re.sub(r"\s*```$", "", text_clean, flags=re.IGNORECASE)
                    parsed = json.loads(text_clean)
                    return parsed.get("query", query), parsed.get("filter", {})
        except Exception as e:
            app_logger.warning(f"Self-Query parsing failed: {str(e)}")
        return query, {}
