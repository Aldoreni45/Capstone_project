import numpy as np
from typing import List, Dict, Any, Optional
from langchain_core.embeddings import Embeddings
from pydantic_models.responses import RetrievedChunk
from retrievers.base import BasePaperRetriever
from vectordb.weaviate_client import WeaviateVectorClient
from config.settings import settings
from custom_logging.logger import app_logger
from langsmith import traceable

class CosineSimilarityRetriever(BasePaperRetriever):
    """Retrieves chunks using simple cosine similarity search via Pinecone."""

    def __init__(self, vector_client: WeaviateVectorClient, embedding_model: Embeddings):
        self.vector_client = vector_client
        self.embedding_model = embedding_model

    @traceable(name="Cosine Similarity Retrieval")
    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        namespace: str = "default",
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        # Embed query
        query_vector = self.embedding_model.embed_query(query)
        
        # Search
        results = self.vector_client.vector_search(
            query_vector=query_vector,
            top_k=top_k,
            namespace=namespace,
            filter_dict=filter_dict
        )
        
        # Format output
        chunks = []
        for res in results:
            meta = res["metadata"]
            chunks.append(RetrievedChunk(
                content=meta.get("text", ""),
                score=res["score"],
                title=meta.get("paper_title", "Unknown Title"),
                page=int(meta.get("page_number", 0)),
                chunk_id=meta.get("chunk_id", ""),
                source=meta.get("source", ""),
                author=meta.get("author", "Unknown Author")
            ))
        return chunks

class MMRRetriever(BasePaperRetriever):
    """Retrieves chunks maximizing relevance and diversity (Maximal Marginal Relevance)."""

    def __init__(self, vector_client: WeaviateVectorClient, embedding_model: Embeddings, lambda_param: float = None):
        self.vector_client = vector_client
        self.embedding_model = embedding_model
        self.lambda_param = lambda_param or settings.get("retrieval", "mmr_lambda", default=0.5)

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        namespace: str = "default",
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        query_vector = self.embedding_model.embed_query(query)
        
        # Fetch larger pool for diversity selection (e.g. 2 * top_k, max 50)
        fetch_k = min(max(top_k * 2, 20), 50)
        results = self.vector_client.vector_search(
            query_vector=query_vector,
            top_k=fetch_k,
            namespace=namespace,
            filter_dict=filter_dict
        )
        
        if not results:
            return []

        # OPTIMIZATION: Use similarity scores from vector search instead of re-embedding
        # This avoids expensive re-embedding of all candidates
        # We'll use the cosine similarity scores as proxies for embedding similarity
        
        # Implement MMR selection algorithm using scores instead of embeddings
        selected_indices = self._maximal_marginal_relevance_optimized(
            query_vector=np.array(query_vector),
            results=results,
            lambda_param=self.lambda_param,
            top_k=min(top_k, len(results))
        )
        
        chunks = []
        for idx in selected_indices:
            res = results[idx]
            meta = res["metadata"]
            chunks.append(RetrievedChunk(
                content=meta.get("text", ""),
                score=res["score"],
                title=meta.get("paper_title", "Unknown Title"),
                page=int(meta.get("page_number", 0)),
                chunk_id=meta.get("chunk_id", ""),
                source=meta.get("source", ""),
                author=meta.get("author", "Unknown Author")
            ))
        return chunks

    def _maximal_marginal_relevance(
        self,
        query_vector: np.ndarray,
        candidate_embeddings: List[np.ndarray],
        lambda_param: float,
        top_k: int
    ) -> List[int]:
        """Classic MMR selection math on numpy vectors."""
        if not candidate_embeddings:
            return []

        # Normalize inputs
        query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-10)
        candidates_norm = [c / (np.linalg.norm(c) + 1e-10) for c in candidate_embeddings]
        
        # Sim to query
        sims_to_query = [np.dot(query_norm, c) for c in candidates_norm]
        
        selected = []
        unselected = list(range(len(candidate_embeddings)))
        
        # Select first element (highest similarity to query)
        first_selected = int(np.argmax(sims_to_query))
        selected.append(first_selected)
        unselected.remove(first_selected)
        
        while len(selected) < top_k and unselected:
            best_score = -float("inf")
            best_idx = -1
            
            for candidate in unselected:
                # Max similarity to already selected documents
                max_sim_to_selected = max([
                    np.dot(candidates_norm[candidate], candidates_norm[sel])
                    for sel in selected
                ])
                
                # MMR formula: lambda * sim(q, doc) - (1 - lambda) * max_sim(doc, selected_doc)
                score = lambda_param * sims_to_query[candidate] - (1.0 - lambda_param) * max_sim_to_selected
                
                if score > best_score:
                    best_score = score
                    best_idx = candidate
                    
            if best_idx == -1:
                break
                
            selected.append(best_idx)
            unselected.remove(best_idx)
            
        return selected

    def _maximal_marginal_relevance_optimized(
        self,
        query_vector: np.ndarray,
        results: List[Dict[str, Any]],
        lambda_param: float,
        top_k: int
    ) -> List[int]:
        """
        Optimized MMR selection using pre-computed similarity scores instead of re-embedding.
        This significantly reduces latency by avoiding expensive re-embedding of candidates.
        """
        if not results:
            return []

        # Use the pre-computed similarity scores from vector search
        sims_to_query = [res["score"] for res in results]
        
        selected = []
        unselected = list(range(len(results)))
        
        # Select first element (highest similarity to query)
        first_selected = int(np.argmax(sims_to_query))
        selected.append(first_selected)
        unselected.remove(first_selected)
        
        while len(selected) < top_k and unselected:
            best_score = -float("inf")
            best_idx = -1
            
            for candidate in unselected:
                # Estimate diversity using text overlap as a proxy for embedding similarity
                # This is much faster than computing actual embedding similarity
                candidate_text = list(results[candidate]["metadata"].get("text", "").lower().split())
                
                max_sim_to_selected = 0.0
                for sel in selected:
                    selected_text = list(results[sel]["metadata"].get("text", "").lower().split())
                    
                    # Jaccard similarity as proxy for embedding similarity
                    if candidate_text and selected_text:
                        candidate_set = set(candidate_text)
                        selected_set = set(selected_text)
                        intersection = len(candidate_set & selected_set)
                        union = len(candidate_set | selected_set)
                        jaccard_sim = intersection / union if union > 0 else 0.0
                        max_sim_to_selected = max(max_sim_to_selected, jaccard_sim)
                
                # MMR formula: lambda * sim(q, doc) - (1 - lambda) * max_sim(doc, selected_doc)
                score = lambda_param * sims_to_query[candidate] - (1.0 - lambda_param) * max_sim_to_selected
                
                if score > best_score:
                    best_score = score
                    best_idx = candidate
                    
            if best_idx == -1:
                break
                
            selected.append(best_idx)
            unselected.remove(best_idx)
            
        return selected

class SimilarityThresholdRetriever(BasePaperRetriever):
    """Retrieves chunks by cosine similarity but filters out results below score threshold."""

    def __init__(self, vector_client: WeaviateVectorClient, embedding_model: Embeddings, threshold: float = None):
        self.vector_client = vector_client
        self.embedding_model = embedding_model
        self.threshold = threshold or settings.get("retrieval", "similarity_threshold", default=0.6)

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        namespace: str = "default",
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        query_vector = self.embedding_model.embed_query(query)
        results = self.vector_client.vector_search(
            query_vector=query_vector,
            top_k=top_k,
            namespace=namespace,
            filter_dict=filter_dict
        )
        
        chunks = []
        for res in results:
            if res["score"] >= self.threshold:
                meta = res["metadata"]
                chunks.append(RetrievedChunk(
                    content=meta.get("text", ""),
                    score=res["score"],
                    title=meta.get("paper_title", "Unknown Title"),
                    page=int(meta.get("page_number", 0)),
                    chunk_id=meta.get("chunk_id", ""),
                    source=meta.get("source", ""),
                    author=meta.get("author", "Unknown Author")
                ))
        app_logger.info(f"Threshold retriever returned {len(chunks)}/{len(results)} chunks above threshold {self.threshold}")
        return chunks
