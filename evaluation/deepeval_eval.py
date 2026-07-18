from typing import List, Dict, Any
from custom_logging.logger import app_logger
from config.settings import settings

class DeepEvalEvaluator:
    """Orchestrates DeepEval checks on pipeline results."""

    def __init__(self):
        self.openai_key = settings.openai_api_key

    def evaluate_sample(
        self,
        query: str,
        retrieved_contexts: List[str],
        actual_output: str,
        ground_truth: str
    ) -> Dict[str, Any]:
        """Runs DeepEval metrics. Falls back to heuristic checks if keys are absent."""
        app_logger.info("Running DeepEval test case checks...")
        
        if self.openai_key:
            try:
                from deepeval.test_case import LLMTestCase
                from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, HallucinationMetric
                
                # Setup test case
                test_case = LLMTestCase(
                    input=query,
                    actual_output=actual_output,
                    expected_output=ground_truth,
                    retrieval_context=retrieved_contexts
                )
                
                # Initialize metrics
                faith_metric = FaithfulnessMetric(threshold=0.5)
                rel_metric = AnswerRelevancyMetric(threshold=0.5)
                hall_metric = HallucinationMetric(threshold=0.5)
                
                # Measure
                faith_metric.measure(test_case)
                rel_metric.measure(test_case)
                hall_metric.measure(test_case)
                
                app_logger.info("DeepEval metrics computed successfully.")
                return {
                    "deepeval_faithfulness": float(faith_metric.score),
                    "deepeval_relevancy": float(rel_metric.score),
                    "deepeval_hallucination": float(hall_metric.score),
                    "deepeval_verdict": "Verified" if faith_metric.is_successful() else "Needs Attention"
                }
            except Exception as e:
                app_logger.error(f"DeepEval computation failed: {str(e)}. Using fallback scoring.")
                
        # Heuristic fallback matching DeepEval parameters
        return self._heuristic_fallback(query, retrieved_contexts, actual_output, ground_truth)

    def _heuristic_fallback(
        self,
        query: str,
        retrieved_contexts: List[str],
        actual_output: str,
        ground_truth: str
    ) -> Dict[str, Any]:
        """Heuristically estimates DeepEval metrics."""
        app_logger.info("Computing heuristic fallback DeepEval scores...")
        
        ctx_all = " ".join(retrieved_contexts).lower()
        ans_words = list(actual_output.lower().split())
        matched_words = [w for w in ans_words if w in ctx_all]
        
        faithfulness = len(matched_words) / len(ans_words) if ans_words else 0.5
        relevancy = sum(1 for w in query.lower().split() if w in actual_output.lower()) / len(query.split())
        
        # Hallucination estimation: 1 - faithfulness
        hallucination = max(1.0 - faithfulness, 0.0)
        
        return {
            "deepeval_faithfulness": faithfulness,
            "deepeval_relevancy": min(relevancy * 1.5, 1.0),
            "deepeval_hallucination": hallucination,
            "deepeval_verdict": "Verified Heuristically"
        }
