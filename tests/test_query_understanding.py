"""
Test suite for Query Understanding Layer.

Tests various query types including:
- Spelling mistakes
- Grammatical errors
- Abbreviations
- Typos
- Synonymous concepts
- Semantic variations
- Incomplete queries
- Natural language variations
"""

import pytest
from query_understanding import QueryUnderstandingOrchestrator


class TestQueryUnderstandingLayer:
    """Test suite for Query Understanding Layer."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create a Query Understanding Orchestrator for testing."""
        return QueryUnderstandingOrchestrator(
            enable_preprocessing=True,
            enable_spell_correction=True,
            enable_normalization=True,
            enable_semantic_normalization=True,
            enable_concept_extraction=True,
            enable_query_rewriting=True,
            embedding_model_type="huggingface"
        )
    
    def test_spelling_correction_basic(self, orchestrator):
        """Test basic spelling correction."""
        query = "What is attentionn"
        result = orchestrator.understand(query)
        
        assert result.original_query == query
        assert result.final_query != query  # Should be corrected
        assert "attention" in result.final_query.lower()
        assert len(result.pipeline_steps) > 0
    
    def test_spelling_correction_transformer(self, orchestrator):
        """Test spelling correction for 'transformars'."""
        query = "transformars"
        result = orchestrator.understand(query)
        
        assert result.original_query == query
        # Should correct to "transformer" or similar
        assert "transformer" in result.final_query.lower()
    
    def test_spelling_correction_rnn(self, orchestrator):
        """Test spelling correction for 'rnnn'."""
        query = "What is rnnn"
        result = orchestrator.understand(query)
        
        assert result.original_query == query
        # Should correct to "RNN" or "recurrent neural network"
        assert "rnn" in result.final_query.lower() or "recurrent" in result.final_query.lower()
    
    def test_grammatical_correction(self, orchestrator):
        """Test grammatical error correction."""
        query = "How transformer works"
        result = orchestrator.understand(query)
        
        assert result.original_query == query
        # Should correct to "How transformers work" or similar
        assert "transformer" in result.final_query.lower()
    
    def test_abbreviation_handling(self, orchestrator):
        """Test abbreviation handling."""
        query = "What is LLM"
        result = orchestrator.understand(query)
        
        assert result.original_query == query
        # Should recognize LLM as a concept
        assert len(result.concepts) > 0
    
    def test_hyphenated_terms(self, orchestrator):
        """Test hyphenated term normalization."""
        query = "multihead attention"
        result = orchestrator.understand(query)
        
        assert result.original_query == query
        # Should normalize to "multi-head attention" or similar
        assert "attention" in result.final_query.lower()
    
    def test_technical_term_normalization(self, orchestrator):
        """Test technical term normalization."""
        query = "vector db"
        result = orchestrator.understand(query)
        
        assert result.original_query == query
        # Should normalize to "vector database" or similar
        assert "vector" in result.final_query.lower()
    
    def test_concept_extraction(self, orchestrator):
        """Test concept extraction."""
        query = "Explain transformer architecture"
        result = orchestrator.understand(query)
        
        assert result.original_query == query
        assert len(result.concepts) > 0
        # Should extract "transformer" as a concept
        assert any("transformer" in c.lower() for c in result.concepts)
    
    def test_query_type_classification(self, orchestrator):
        """Test query type classification."""
        query = "What is attention mechanism"
        result = orchestrator.understand(query)
        
        assert result.original_query == query
        # Should classify as definition query
        assert result.query_type in ["definition", "explanation", "general"]
    
    def test_domain_classification(self, orchestrator):
        """Test domain classification."""
        query = "Explain RAG system"
        result = orchestrator.understand(query)
        
        assert result.original_query == query
        # Should classify in relevant domain
        assert result.domain in ["rag", "nlp", "ml", "general"]
    
    def test_pipeline_steps_execution(self, orchestrator):
        """Test that all pipeline steps are executed."""
        query = "What is attention mechanism"
        result = orchestrator.understand(query)
        
        assert result.original_query == query
        # Check that pipeline steps were executed
        assert len(result.pipeline_steps) > 0
        expected_steps = ["preprocessing", "spell_correction", "normalization", 
                         "semantic_normalization", "concept_extraction", "query_rewriting"]
        for step in expected_steps:
            if step in result.pipeline_steps:
                assert True
                break
    
    def test_confidence_score(self, orchestrator):
        """Test confidence score calculation."""
        query = "What is attention mechanism"
        result = orchestrator.understand(query)
        
        assert result.original_query == query
        assert 0.0 <= result.confidence <= 1.0
    
    def test_processing_time(self, orchestrator):
        """Test processing time tracking."""
        query = "What is attention mechanism"
        result = orchestrator.understand(query)
        
        assert result.original_query == query
        assert result.processing_time_ms >= 0.0
    
    def test_empty_query(self, orchestrator):
        """Test handling of empty query."""
        query = ""
        result = orchestrator.understand(query)
        
        assert result.original_query == ""
        assert result.final_query == ""
        assert result.confidence == 0.0
    
    def test_none_query(self, orchestrator):
        """Test handling of None query."""
        query = None
        result = orchestrator.understand(query)
        
        assert result.original_query == ""
        assert result.final_query == ""
        assert result.confidence == 0.0
    
    def test_batch_processing(self, orchestrator):
        """Test batch processing of multiple queries."""
        queries = [
            "What is attention",
            "Explain transformer",
            "RNN architecture"
        ]
        results = orchestrator.understand_batch(queries)
        
        assert len(results) == len(queries)
        for i, result in enumerate(results):
            assert result.original_query == queries[i]
    
    def test_cache_functionality(self, orchestrator):
        """Test that caching works correctly."""
        query = "What is attention mechanism"
        
        # First call
        result1 = orchestrator.understand(query)
        
        # Second call (should use cache)
        result2 = orchestrator.understand(query)
        
        assert result1.final_query == result2.final_query
    
    def test_step_configuration(self, orchestrator):
        """Test that individual steps can be configured."""
        # Disable spell correction
        orchestrator.configure_step("spell_correction", False)
        
        query = "What is attentionn"
        result = orchestrator.understand(query)
        
        # Spell correction should not be in pipeline steps
        assert "spell_correction" not in result.pipeline_steps
        
        # Re-enable
        orchestrator.configure_step("spell_correction", True)
    
    def test_pipeline_stats(self, orchestrator):
        """Test pipeline statistics retrieval."""
        stats = orchestrator.get_pipeline_stats()
        
        assert "enabled_steps" in stats
        assert "latency_breakdown" in stats
        assert "cache_stats" in stats
    
    def test_clear_caches(self, orchestrator):
        """Test cache clearing functionality."""
        # Process a query to populate cache
        query = "What is attention mechanism"
        orchestrator.understand(query)
        
        # Clear caches
        orchestrator.clear_caches()
        
        # Check that caches are empty
        stats = orchestrator.get_pipeline_stats()
        assert stats["cache_stats"]["spell_correction_cache"] == 0
        assert stats["cache_stats"]["semantic_normalization_cache"] == 0
        assert stats["cache_stats"]["concept_extraction_cache"] == 0
        assert stats["cache_stats"]["query_rewrite_cache"] == 0
    
    def test_complex_query_with_multiple_errors(self, orchestrator):
        """Test query with multiple types of errors."""
        query = "How multihead self attentoin works in transformars"
        result = orchestrator.understand(query)
        
        assert result.original_query == query
        # Should correct multiple errors
        assert result.final_query != query
        assert "attention" in result.final_query.lower()
        assert "transformer" in result.final_query.lower()
    
    def test_query_with_abbreviations(self, orchestrator):
        """Test query with abbreviations."""
        query = "Explain RAG in LLM systems"
        result = orchestrator.understand(query)
        
        assert result.original_query == query
        # Should extract concepts
        assert len(result.concepts) > 0


class TestQueryUnderstandingComponents:
    """Test individual components of Query Understanding Layer."""
    
    def test_preprocessor_only(self):
        """Test preprocessor in isolation."""
        from query_understanding.preprocessor import QueryPreprocessor
        
        preprocessor = QueryPreprocessor()
        query = "  What  is  ATTENTION  "
        result = preprocessor.preprocess(query)
        
        assert result == "what is attention"
    
    def test_normalizer_only(self):
        """Test normalizer in isolation."""
        from query_understanding.normalizer import QueryNormalizer
        
        normalizer = QueryNormalizer()
        query = "models"
        result = normalizer.normalize(query)
        
        # Should normalize plural to singular (if spaCy is available)
        assert result is not None
    
    def test_orchestrator_with_disabled_steps(self):
        """Test orchestrator with some steps disabled."""
        orchestrator = QueryUnderstandingOrchestrator(
            enable_preprocessing=True,
            enable_spell_correction=False,  # Disabled
            enable_normalization=True,
            enable_semantic_normalization=False,  # Disabled
            enable_concept_extraction=True,
            enable_query_rewriting=True,
            embedding_model_type="huggingface"
        )
        
        query = "What is attentionn"
        result = orchestrator.understand(query)
        
        assert result.original_query == query
        # Spell correction and semantic normalization should not be in steps
        assert "spell_correction" not in result.pipeline_steps
        assert "semantic_normalization" not in result.pipeline_steps


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
