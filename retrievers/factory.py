from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pydantic_models.responses import RetrievedChunk
from vectordb.weaviate_client import WeaviateVectorClient

from retrievers.base import BasePaperRetriever
from retrievers.vector_retrievers import CosineSimilarityRetriever, MMRRetriever, SimilarityThresholdRetriever
from retrievers.hybrid_retrievers import HybridRetriever
from retrievers.advanced_retrievers import (
    MultiQueryRetriever,
    ParentDocumentRetriever,
    ContextualCompressionRetriever,
    EnsembleRetriever,
    SelfQueryRetriever
)
from custom_logging.logger import app_logger

def get_retriever(
    retriever_type: str,
    vector_client: WeaviateVectorClient,
    embedding_model: Embeddings,
    groq_api_key: str,
    corpus: List[Document] = None
) -> BasePaperRetriever:
    """
    Factory method to retrieve and initialize a specific retriever strategy.
    """
    retriever_type = retriever_type.lower().strip()
    app_logger.info(f"Instantiating retriever type: '{retriever_type}'")
    
    # 1. Cosine similarity
    cosine = CosineSimilarityRetriever(vector_client, embedding_model)
    
    if retriever_type == "cosine":
        return cosine
        
    # 2. MMR
    elif retriever_type == "mmr":
        return MMRRetriever(vector_client, embedding_model)
        
    # 3. Hybrid Search
    elif retriever_type == "hybrid":
        return HybridRetriever(vector_client, embedding_model, corpus)
        
    # 4. Multi Query
    elif retriever_type == "multiquery":
        return MultiQueryRetriever(cosine, groq_api_key)
        
    # 5. Parent Document
    elif retriever_type == "parent":
        # Convert corpus documents to RetrievedChunks for ParentDocument lookup caching
        retrieved_chunks_corpus = []
        if corpus:
            for idx, doc in enumerate(corpus):
                retrieved_chunks_corpus.append(RetrievedChunk(
                    content=doc.page_content,
                    score=1.0,
                    title=doc.metadata.get("paper_title", "Unknown Title"),
                    page=int(doc.metadata.get("page_number", 0)),
                    chunk_id=doc.metadata.get("chunk_id", f"corpus_{idx}"),
                    source=doc.metadata.get("source", ""),
                    author=doc.metadata.get("author", "Unknown Author")
                ))
        return ParentDocumentRetriever(cosine, retrieved_chunks_corpus)
        
    # 6. Contextual Compression
    elif retriever_type == "compression":
        return ContextualCompressionRetriever(cosine, groq_api_key)
        
    # 7. Ensemble Retriever (Cosine + Hybrid + MMR)
    elif retriever_type == "ensemble":
        hybrid = HybridRetriever(vector_client, embedding_model, corpus)
        mmr = MMRRetriever(vector_client, embedding_model)
        # Combine Cosine, MMR, Hybrid with equal weights
        return EnsembleRetriever([cosine, hybrid, mmr], weights=[0.33, 0.33, 0.34])
        
    # 8. Self Query
    elif retriever_type == "selfquery":
        return SelfQueryRetriever(cosine, groq_api_key)
        
    # 9. Similarity Threshold
    elif retriever_type == "threshold":
        return SimilarityThresholdRetriever(vector_client, embedding_model)
        
    else:
        app_logger.warning(f"Unknown retriever type '{retriever_type}'. Defaulting to Cosine Similarity.")
        return cosine
