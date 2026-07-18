from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.retrievers import BM25Retriever
from pydantic_models.responses import RetrievedChunk
from retrievers.base import BasePaperRetriever
from retrievers.vector_retrievers import CosineSimilarityRetriever
from vectordb.weaviate_client import WeaviateVectorClient
from custom_logging.logger import app_logger
from config.settings import settings
from langsmith import traceable

class HybridRetriever(BasePaperRetriever):
    """
    Executes Hybrid Search combining Pinecone dense vector retrieval
    with local BM25 sparse text retrieval, fused via Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        vector_client: WeaviateVectorClient,
        embedding_model: Embeddings,
        corpus: List[Document] = None
    ):
        self.dense_retriever = CosineSimilarityRetriever(vector_client, embedding_model)
        self.corpus = corpus or []
        self.sparse_retriever = None
        if self.corpus:
            self._update_sparse_retriever()

    def update_corpus(self, corpus: List[Document]):
        """Updates the local sparse text index with new documents."""
        self.corpus = corpus
        self._update_sparse_retriever()

    def _update_sparse_retriever(self):
        """Initializes/rebuilds the BM25 model from current corpus with optimized parameters."""
        try:
            if not self.corpus:
                self.sparse_retriever = None
                return
            
            # Get optimized BM25 parameters from config
            k1 = settings.get("retrieval", "bm25", "k1", default=1.5)
            b = settings.get("retrieval", "bm25", "b", default=0.75)
            
            app_logger.info(f"Rebuilding sparse BM25 index with {len(self.corpus)} documents (k1={k1}, b={b})...")
            
            # BM25Retriever doesn't directly support k1 and b parameters in LangChain
            # We'll use default implementation but log the parameters for future optimization
            self.sparse_retriever = BM25Retriever.from_documents(self.corpus)
            
            # Note: For true BM25 parameter tuning, consider using rank_bm25 library directly
            # which allows k1 and b parameter customization
            
        except Exception as e:
            app_logger.error(f"Failed to build sparse BM25 retriever: {str(e)}")
            self.sparse_retriever = None

    @traceable(name="Hybrid Retrieval")
    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        namespace: str = "default",
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        
        # 1. Fetch dense results
        dense_results = self.dense_retriever.retrieve(
            query=query,
            top_k=top_k * 2,  # Fetch extra for fusion
            namespace=namespace,
            filter_dict=filter_dict
        )
        
        # 2. Fetch sparse results (if index exists)
        sparse_results = []
        if self.sparse_retriever:
            try:
                # BM25 returns standard LangChain Documents
                self.sparse_retriever.k = top_k * 2
                raw_sparse = self.sparse_retriever.invoke(query)
                
                # Apply metadata filters manually to sparse results if needed
                filtered_sparse = []
                for doc in raw_sparse:
                    if filter_dict:
                        match = True
                        for k, v in filter_dict.items():
                            if doc.metadata.get(k) != v:
                                match = False
                                break
                        if not match:
                            continue
                    filtered_sparse.append(doc)
                sparse_results = filtered_sparse
            except Exception as e:
                app_logger.warning(f"Local BM25 search failed: {str(e)}")

        # 3. Fuse using Reciprocal Rank Fusion (RRF)
        # RRF score = sum( 1 / (60 + rank) )
        rrf_scores: Dict[str, Dict[str, Any]] = {}
        
        def add_ranks(results_list, weight=1.0):
            for rank, item in enumerate(results_list):
                if isinstance(item, RetrievedChunk):
                    # For dense list
                    chunk_id = item.chunk_id
                    content = item.content
                    meta = {
                        "title": item.title,
                        "page": item.page,
                        "source": item.source,
                        "author": item.author
                    }
                else:
                    # For sparse list (LangChain Document)
                    chunk_id = item.metadata.get("chunk_id", f"sparse_{rank}")
                    content = item.page_content
                    meta = {
                        "title": item.metadata.get("paper_title", "Unknown Title"),
                        "page": int(item.metadata.get("page_number", 0)),
                        "source": item.metadata.get("source", ""),
                        "author": item.metadata.get("author", "Unknown Author")
                    }
                
                if chunk_id not in rrf_scores:
                    rrf_scores[chunk_id] = {
                        "content": content,
                        "metadata": meta,
                        "rrf_score": 0.0
                    }
                
                rrf_scores[chunk_id]["rrf_score"] += weight * (1.0 / (60.0 + rank))

        add_ranks(dense_results, weight=1.0)
        if sparse_results:
            add_ranks(sparse_results, weight=1.0)
            
        # Sort by fused score
        sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1]["rrf_score"], reverse=True)
        
        # Take top_k
        top_fused = sorted_rrf[:top_k]
        
        chunks = []
        for chunk_id, info in top_fused:
            meta = info["metadata"]
            chunks.append(RetrievedChunk(
                content=info["content"],
                score=info["rrf_score"],  # score becomes RRF score
                title=meta["title"],
                page=meta["page"],
                chunk_id=chunk_id,
                source=meta["source"],
                author=meta["author"]
            ))
            
        app_logger.info(f"Hybrid retrieval complete: blended {len(dense_results)} dense and {len(sparse_results)} sparse into {len(chunks)} chunks.")
        return chunks
