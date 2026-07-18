import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np
from langchain_core.embeddings import Embeddings
from custom_logging.logger import app_logger

class EmbeddingBenchmarker:
    """Helper to benchmark latency and similarity score distributions for different embedding models."""

    @staticmethod
    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        a = np.array(v1)
        b = np.array(v2)
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    @classmethod
    def benchmark_model(cls, model: Embeddings, name: str, test_texts: List[str], test_query: str) -> Dict[str, Any]:
        """Benchmarks embedding generation latency and basic retrieval query similarity."""
        app_logger.info(f"Starting benchmark for embedding model: {name}")
        
        # Latency benchmark
        start_time = time.perf_counter()
        doc_embeddings = model.embed_documents(test_texts)
        doc_latency = (time.perf_counter() - start_time) * 1000  # ms
        
        start_time = time.perf_counter()
        query_embedding = model.embed_query(test_query)
        query_latency = (time.perf_counter() - start_time) * 1000  # ms

        # Similarity metrics
        scores = [cls.cosine_similarity(query_embedding, doc_emb) for doc_emb in doc_embeddings]
        
        metrics = {
            "model_name": name,
            "dimension": len(query_embedding),
            "doc_batch_latency_ms": doc_latency,
            "avg_doc_latency_ms": doc_latency / len(test_texts) if test_texts else 0,
            "query_latency_ms": query_latency,
            "similarity_scores": scores,
            "avg_similarity": float(np.mean(scores)) if scores else 0.0,
            "max_similarity": float(np.max(scores)) if scores else 0.0,
            "min_similarity": float(np.min(scores)) if scores else 0.0
        }
        
        app_logger.info(
            f"Benchmark for {name} complete: Dimension={metrics['dimension']}, "
            f"Avg Doc Latency={metrics['avg_doc_latency_ms']:.2f}ms, Avg Similarity={metrics['avg_similarity']:.4f}"
        )
        return metrics
