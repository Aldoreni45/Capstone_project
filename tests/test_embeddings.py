import pytest
from embeddings.base import EmbeddingBenchmarker
from embeddings import get_embeddings

def test_cosine_similarity():
    """Validates cosine similarity computations on benchmark vectors."""
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    v3 = [0.0, 1.0, 0.0]
    
    assert EmbeddingBenchmarker.cosine_similarity(v1, v2) == pytest.approx(1.0)
    assert EmbeddingBenchmarker.cosine_similarity(v1, v3) == pytest.approx(0.0)

def test_embeddings_factory_fallback():
    """Ensures fallback values are respected by get_embeddings factory."""
    emb = get_embeddings("invalid-model-name")
    assert emb.__class__.__name__ == "HuggingFaceBgeEmbeddings"
