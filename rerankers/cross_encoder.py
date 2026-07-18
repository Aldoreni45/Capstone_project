from typing import List
from sentence_transformers import CrossEncoder
from pydantic_models.responses import RetrievedChunk
from config.settings import settings
from custom_logging.logger import app_logger
from custom_logging.error_handler import RerankingError
from langsmith import traceable
import os

class CrossEncoderReranker:
    """Reranks candidate chunks using cross-encoder/ms-marco-MiniLM-L-6-v2."""

    _model_instance = None

    def __init__(self):
        self.model_name = settings.get("retrieval", "cross_encoder", "model_name", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.device = settings.get("retrieval", "cross_encoder", "device", default="cpu")
        self._load_model()

    def _load_model(self):
        """Loads and caches the CrossEncoder model instance."""
        if CrossEncoderReranker._model_instance is None:
            app_logger.info(f"Loading CrossEncoder model '{self.model_name}' on device={self.device}...")
            try:
                # Set HUGGINGFACE_HUB_TOKEN for model download if HF_TOKEN is available
                hf_token = os.getenv("HF_TOKEN", None)
                if hf_token and not os.getenv("HUGGINGFACE_HUB_TOKEN"):
                    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
                    app_logger.info("Using HF_TOKEN for Hugging Face Hub authentication")
                
                CrossEncoderReranker._model_instance = CrossEncoder(
                    self.model_name,
                    device=self.device
                )
                app_logger.info("CrossEncoder model loaded successfully.")
            except Exception as e:
                app_logger.error(f"Failed to load CrossEncoder model: {str(e)}")
                raise RerankingError(f"CrossEncoder model loading failed: {str(e)}")
        
        self.model = CrossEncoderReranker._model_instance

    @traceable(name="Cross Encoder Reranking")
    def rerank(self, query: str, chunks: List[RetrievedChunk], top_n: int = 5, score_threshold: float = None) -> List[RetrievedChunk]:
        """
        Reranks a list of retrieved chunks relative to the query.
        Returns the top_n chunks sorted by score, filtered by score threshold.
        
        Args:
            query: The search query
            chunks: List of retrieved chunks to rerank
            top_n: Maximum number of chunks to return
            score_threshold: Minimum score threshold (from config if not provided)
        """
        if not chunks:
            return []
        
        # Get score threshold from config if not provided
        if score_threshold is None:
            score_threshold = settings.get("retrieval", "rerank_score_threshold", default=0.3)
            
        app_logger.info(f"Reranking {len(chunks)} chunks down to top {top_n} with threshold {score_threshold} for query: '{query}'")
        
        try:
            # Pair query with each chunk content
            pairs = [[query, chunk.content] for chunk in chunks]
            
            # Predict similarity scores
            scores = self.model.predict(pairs)
            
            # Assign scores back to chunks
            for chunk, score in zip(chunks, scores):
                chunk.score = float(score)
                
            # Sort chunks in descending order of score
            reranked_chunks = sorted(chunks, key=lambda x: x.score, reverse=True)
            
            # Apply score threshold filtering (only if threshold > 0)
            if score_threshold > 0:
                thresholded_chunks = [chunk for chunk in reranked_chunks if chunk.score >= score_threshold]
                app_logger.info(f"Score threshold {score_threshold} filtered {len(reranked_chunks)} -> {len(thresholded_chunks)} chunks")
            else:
                # No threshold filtering - use all reranked chunks
                thresholded_chunks = reranked_chunks
                app_logger.info(f"Score threshold disabled (threshold={score_threshold}), using all {len(reranked_chunks)} chunks")
            
            # Return top N from thresholded chunks
            final_selection = thresholded_chunks[:top_n]
            
            # If we don't have enough chunks above threshold, log warning
            if score_threshold > 0 and len(final_selection) < top_n:
                app_logger.warning(
                    f"Only {len(final_selection)} chunks passed threshold {score_threshold}, "
                    f"requested top_n={top_n}. Consider lowering threshold."
                )
            
            app_logger.info(f"Reranking complete. Selected {len(final_selection)} chunks.")
            return final_selection
            
        except Exception as e:
            app_logger.error(f"Failed during cross-encoder reranking: {str(e)}")
            raise RerankingError(f"Reranking execution failed: {str(e)}")
