import numpy as np
import pytest
from retrievers.vector_retrievers import MMRRetriever
from pydantic_models.responses import RetrievedChunk

class DummyRetriever:
    pass

def test_mmr_selection():
    """Checks the core Maximal Marginal Relevance selection algorithm."""
    # Mocking query and candidates
    query = np.array([1.0, 0.0])
    candidates = [
        np.array([1.0, 0.0]),  # Highly similar to query
        np.array([0.9, 0.1]),  # Similar to query, redundant with candidate 0
        np.array([0.1, 0.9])   # Less similar to query, but highly diverse
    ]
    
    # Instantiate MMR retriever with dummy parameters
    mmr = MMRRetriever(vector_client=DummyRetriever(), embedding_model=DummyRetriever())
    
    # Run MMR with k=2, lambda=0.5
    selected = mmr._maximal_marginal_relevance(
        query_vector=query,
        candidate_embeddings=candidates,
        lambda_param=0.5,
        top_k=2
    )
    
    # 0 must be selected first. Then, 2 should be selected over 1 because 1 is redundant.
    assert selected == [0, 2]
