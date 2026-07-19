"""
Query Understanding Orchestrator - Main pipeline coordinator.

Orchestrates the complete query understanding pipeline:
1. Query Preprocessor
2. Spell Corrector
3. Query Normalizer
4. Semantic Normalizer
5. Concept Extractor
6. Query Rewriter

This is the main entry point for the Query Understanding Layer.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from custom_logging.logger import app_logger
from query_understanding.preprocessor import QueryPreprocessor
from query_understanding.spell_corrector import SpellCorrector
from query_understanding.normalizer import QueryNormalizer
from query_understanding.semantic_normalizer import SemanticNormalizer
from query_understanding.concept_extractor import ConceptExtractor, ConceptExtractionResult
from query_understanding.enhanced_rewriter import EnhancedQueryRewriter, QueryRewriteResult
from utils.metrics import MetricsRegistry, track_latency


class QueryUnderstandingResult(BaseModel):
    """Result of the complete query understanding pipeline."""
    original_query: str = Field(..., description="Original user query")
    preprocessed_query: str = Field(..., description="Preprocessed query")
    spell_corrected_query: str = Field(..., description="Spell-corrected query")
    normalized_query: str = Field(..., description="Normalized query")
    semantically_normalized_query: str = Field(..., description="Semantically normalized query")
    final_query: str = Field(..., description="Final query for retrieval")
    concepts: list[str] = Field(default_factory=list, description="Extracted concepts")
    query_type: str = Field(default="general", description="Query type classification")
    domain: str = Field(default="general", description="Domain classification")
    pipeline_steps: list[str] = Field(default_factory=list, description="Pipeline steps executed")
    processing_time_ms: float = Field(default=0.0, description="Total processing time in milliseconds")
    confidence: float = Field(default=1.0, description="Overall confidence in the result")


class QueryUnderstandingOrchestrator:
    """Orchestrates the complete query understanding pipeline."""
    
    def __init__(
        self,
        enable_preprocessing: bool = True,
        enable_spell_correction: bool = True,
        enable_normalization: bool = True,
        enable_semantic_normalization: bool = True,
        enable_concept_extraction: bool = True,
        enable_query_rewriting: bool = True,
        embedding_model_type: str = "huggingface"
    ):
        """
        Initialize the query understanding orchestrator.
        
        Args:
            enable_preprocessing: Enable query preprocessing
            enable_spell_correction: Enable spell correction
            enable_normalization: Enable query normalization
            enable_semantic_normalization: Enable semantic normalization
            enable_concept_extraction: Enable concept extraction
            enable_query_rewriting: Enable query rewriting
            embedding_model_type: Embedding model type for semantic normalization
        """
        self.logger = app_logger
        
        # Configuration
        self.enable_preprocessing = enable_preprocessing
        self.enable_spell_correction = enable_spell_correction
        self.enable_normalization = enable_normalization
        self.enable_semantic_normalization = enable_semantic_normalization
        self.enable_concept_extraction = enable_concept_extraction
        self.enable_query_rewriting = enable_query_rewriting
        
        # Initialize components
        self.preprocessor = QueryPreprocessor()
        self.spell_corrector = SpellCorrector()
        self.normalizer = QueryNormalizer()
        self.semantic_normalizer = SemanticNormalizer(embedding_model_type)
        self.concept_extractor = ConceptExtractor()
        self.query_rewriter = EnhancedQueryRewriter()
        
        self.logger.info("Query Understanding Orchestrator initialized")
    
    @track_latency("query_understanding")
    def understand(self, query: str) -> QueryUnderstandingResult:
        """
        Apply the complete query understanding pipeline.
        
        Args:
            query: Original user query
            
        Returns:
            QueryUnderstandingResult with all pipeline outputs
        """
        if not query or not isinstance(query, str):
            return QueryUnderstandingResult(
                original_query="",
                preprocessed_query="",
                spell_corrected_query="",
                normalized_query="",
                semantically_normalized_query="",
                final_query="",
                confidence=0.0
            )
        
        self.logger.info(f"Starting query understanding for: '{query}'")
        
        # Initialize result
        result = QueryUnderstandingResult(
            original_query=query,
            preprocessed_query=query,
            spell_corrected_query=query,
            normalized_query=query,
            semantically_normalized_query=query,
            final_query=query
        )
        
        # Step 1: Query Preprocessing
        if self.enable_preprocessing:
            with track_latency("preprocessing"):
                result.preprocessed_query = self.preprocessor.preprocess(query)
                result.pipeline_steps.append("preprocessing")
                self.logger.debug(f"Preprocessed: '{result.preprocessed_query}'")
        
        # Step 2: Spell Correction
        if self.enable_spell_correction:
            with track_latency("spell_correction"):
                spell_result = self.spell_corrector.correct(result.preprocessed_query)
                result.spell_corrected_query = spell_result.corrected_query
                result.pipeline_steps.append("spell_correction")
                self.logger.debug(f"Spell corrected: '{result.spell_corrected_query}'")
        
        # Step 3: Query Normalization
        if self.enable_normalization:
            with track_latency("normalization"):
                result.normalized_query = self.normalizer.normalize(result.spell_corrected_query)
                result.pipeline_steps.append("normalization")
                self.logger.debug(f"Normalized: '{result.normalized_query}'")
        
        # Step 4: Semantic Normalization
        if self.enable_semantic_normalization:
            with track_latency("semantic_normalization"):
                result.semantically_normalized_query = self.semantic_normalizer.normalize(
                    result.normalized_query
                )
                result.pipeline_steps.append("semantic_normalization")
                self.logger.debug(f"Semantically normalized: '{result.semantically_normalized_query}'")
        
        # Step 5: Concept Extraction
        concept_result = None
        if self.enable_concept_extraction:
            with track_latency("concept_extraction"):
                concept_result = self.concept_extractor.extract(result.semantically_normalized_query)
                result.concepts = [c.text for c in concept_result.concepts]
                result.query_type = concept_result.query_type
                result.domain = concept_result.domain
                result.pipeline_steps.append("concept_extraction")
                self.logger.debug(f"Concepts extracted: {result.concepts}")
        
        # Step 6: Query Rewriting
        if self.enable_query_rewriting:
            with track_latency("query_rewriting"):
                rewrite_result = self.query_rewriter.rewrite(
                    query=result.original_query,
                    preprocessed_query=result.semantically_normalized_query,
                    concept_result=concept_result
                )
                result.final_query = rewrite_result.rewritten_query
                result.pipeline_steps.append("query_rewriting")
                self.logger.debug(f"Rewritten: '{result.final_query}'")
        else:
            result.final_query = result.semantically_normalized_query
        
        # Get processing time from metrics
        metrics = MetricsRegistry.get_metrics()
        result.processing_time_ms = metrics.get("avg_latency_ms", {}).get("query_understanding", 0.0)
        
        # Calculate overall confidence
        result.confidence = self._calculate_confidence(result)
        
        self.logger.info(
            f"Query understanding complete: '{query}' -> '{result.final_query}' "
            f"(steps: {len(result.pipeline_steps)}, time: {result.processing_time_ms:.2f}ms)"
        )
        
        return result
    
    def _calculate_confidence(self, result: QueryUnderstandingResult) -> float:
        """
        Calculate overall confidence based on pipeline execution.
        
        Args:
            result: Query understanding result
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = 1.0
        
        # Reduce confidence if query changed significantly
        if result.original_query != result.final_query:
            # More steps = more potential for error
            confidence -= 0.05 * len(result.pipeline_steps)
        
        # Ensure confidence doesn't go below 0.5
        return max(0.5, confidence)
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the query understanding pipeline.
        
        Returns:
            Dictionary with pipeline statistics
        """
        metrics = MetricsRegistry.get_metrics()
        latency_metrics = metrics.get("avg_latency_ms", {})
        
        return {
            "enabled_steps": {
                "preprocessing": self.enable_preprocessing,
                "spell_correction": self.enable_spell_correction,
                "normalization": self.enable_normalization,
                "semantic_normalization": self.enable_semantic_normalization,
                "concept_extraction": self.enable_concept_extraction,
                "query_rewriting": self.enable_query_rewriting,
            },
            "latency_breakdown": latency_metrics,
            "cache_stats": {
                "spell_correction_cache": len(self.spell_corrector.correction_cache),
                "semantic_normalization_cache": len(self.semantic_normalizer.semantic_cache),
                "concept_extraction_cache": len(self.concept_extractor.extraction_cache),
                "query_rewrite_cache": len(self.query_rewriter.rewrite_cache),
            }
        }
    
    def clear_caches(self):
        """Clear all caches in the pipeline components."""
        self.spell_corrector.clear_cache()
        self.semantic_normalizer.clear_cache()
        self.concept_extractor.clear_cache()
        self.query_rewriter.clear_cache()
        self.logger.info("All query understanding caches cleared")
    
    def configure_step(self, step: str, enabled: bool):
        """
        Enable or disable a specific pipeline step.
        
        Args:
            step: Name of the step (preprocessing, spell_correction, etc.)
            enabled: Whether to enable the step
        """
        step_map = {
            "preprocessing": "enable_preprocessing",
            "spell_correction": "enable_spell_correction",
            "normalization": "enable_normalization",
            "semantic_normalization": "enable_semantic_normalization",
            "concept_extraction": "enable_concept_extraction",
            "query_rewriting": "enable_query_rewriting",
        }
        
        if step in step_map:
            setattr(self, step_map[step], enabled)
            self.logger.info(f"Step '{step}' {'enabled' if enabled else 'disabled'}")
        else:
            self.logger.warning(f"Unknown step: '{step}'")
    
    def understand_batch(self, queries: list[str]) -> list[QueryUnderstandingResult]:
        """
        Apply query understanding to multiple queries in batch.
        
        Args:
            queries: List of queries to process
            
        Returns:
            List of QueryUnderstandingResults
        """
        results = []
        for query in queries:
            result = self.understand(query)
            results.append(result)
        return results
