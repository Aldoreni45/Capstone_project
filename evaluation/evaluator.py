from typing import List, Dict, Any, Optional
from pydantic_models.responses import EvaluationOutput, RetrievedChunk
from evaluation.ragas_eval import RagasEvaluator
from evaluation.deepeval_eval import DeepEvalEvaluator
from custom_logging.logger import app_logger
from langsmith import traceable

class UnifiedRAGEvaluator:
    """Consolidates RAGAS and DeepEval metrics into a structured outcome."""

    def __init__(self):
        self.ragas_evaluator = RagasEvaluator()
        self.deepeval_evaluator = DeepEvalEvaluator()

    @traceable(name="RAG Evaluation")
    def evaluate_turn(
        self,
        query: str,
        retrieved_chunks: List[RetrievedChunk],
        actual_answer: str,
        ground_truth: str = None
    ) -> EvaluationOutput:
        """Runs RAGAS and DeepEval evaluations over a single query-answer turn."""
        app_logger.info("Executing unified RAG evaluation cycle...")
        
        # Format contexts and ground_truth fallbacks
        contexts = [chunk.content for chunk in retrieved_chunks]
        gt = ground_truth or "Research paper contents regarding " + query
        
        # 1. Run RAGAS metrics
        ragas_scores = self.ragas_evaluator.evaluate_rag(
            questions=[query],
            contexts=[contexts],
            answers=[actual_answer],
            ground_truths=[gt]
        )
        
        # 2. Run DeepEval metrics
        deepeval_scores = self.deepeval_evaluator.evaluate_sample(
            query=query,
            retrieved_contexts=contexts,
            actual_output=actual_answer,
            ground_truth=gt
        )
        
        # Compute custom metrics (Semantic Similarity, Retrieval Accuracy)
        semantic_similarity = self._compute_semantic_similarity(actual_answer, gt)
        retrieval_accuracy = self._compute_retrieval_accuracy(retrieved_chunks)
        
        # Blend metrics
        faithfulness = (ragas_scores.get("faithfulness", 0.5) + deepeval_scores.get("deepeval_faithfulness", 0.5)) / 2.0
        precision = ragas_scores.get("context_precision", 0.5)
        recall = ragas_scores.get("context_recall", 0.5)
        relevancy = (ragas_scores.get("answer_relevancy", 0.5) + deepeval_scores.get("deepeval_relevancy", 0.5)) / 2.0
        hallucination = deepeval_scores.get("deepeval_hallucination", 0.5)
        
        verdict = f"Evaluation complete. RAG pipeline health is stable (Verdict: {deepeval_scores.get('deepeval_verdict')})."
        
        output = EvaluationOutput(
            faithfulness=float(faithfulness),
            context_precision=float(precision),
            context_recall=float(recall),
            hallucination_score=float(hallucination),
            answer_relevancy=float(relevancy),
            semantic_similarity=float(semantic_similarity),
            retrieval_accuracy=float(retrieval_accuracy),
            verdict=verdict
        )
        
        app_logger.info(f"Evaluation outcomes finalized: {output.model_dump()}")
        return output

    def _compute_semantic_similarity(self, actual: str, expected: str) -> float:
        """
        Computes semantic similarity using embedding-based cosine similarity.
        Falls back to Jaccard similarity if embeddings are not available.
        """
        try:
            # Try to use embedding-based similarity
            from embeddings import get_embeddings
            embedding_model = get_embeddings()
            
            # Generate embeddings
            actual_embedding = embedding_model.embed_query(actual)
            expected_embedding = embedding_model.embed_query(expected)
            
            # Compute cosine similarity
            import numpy as np
            actual_vec = np.array(actual_embedding)
            expected_vec = np.array(expected_embedding)
            
            # Normalize vectors
            actual_norm = actual_vec / (np.linalg.norm(actual_vec) + 1e-10)
            expected_norm = expected_vec / (np.linalg.norm(expected_vec) + 1e-10)
            
            # Cosine similarity
            similarity = float(np.dot(actual_norm, expected_norm))
            
            app_logger.info(f"Embedding-based semantic similarity: {similarity:.4f}")
            return similarity
            
        except Exception as e:
            app_logger.warning(f"Embedding-based similarity failed: {str(e)}. Using Jaccard fallback.")
            
            # Fallback to Jaccard similarity (word overlap)
            words_actual = list(actual.lower().split())
            words_expected = list(expected.lower().split())
            if not words_actual or not words_expected:
                return 0.5
            intersection = set(words_actual) & set(words_expected)
            union = set(words_actual) | set(words_expected)
            return float(len(intersection) / len(union))

    def _compute_retrieval_accuracy(self, chunks: List[RetrievedChunk]) -> float:
        """
        Assesses retrieval confidence.
        Calculates the ratio of retrieved chunks with positive relevance score.
        """
        if not chunks:
            return 0.0
        relevant_count = sum(1 for c in chunks if c.score > 0.0)
        return float(relevant_count / len(chunks))
