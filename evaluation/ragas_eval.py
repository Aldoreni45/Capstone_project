from typing import List, Dict, Any
from datasets import Dataset
from custom_logging.logger import app_logger
from config.settings import settings

class RagasEvaluator:
    """Runs RAGAS evaluations over question-context-answer-ground_truth datasets."""

    def __init__(self):
        self.openai_key = settings.openai_api_key

    def evaluate_rag(
        self,
        questions: List[str],
        contexts: List[List[str]],
        answers: List[str],
        ground_truths: List[str]
    ) -> Dict[str, Any]:
        """Runs RAGAS metrics. Falls back to heuristic scores if OpenAI key is missing."""
        app_logger.info(f"Running RAGAS evaluation on {len(questions)} samples...")
        
        # Guard clause for empty lists
        if not questions:
            return {}

        # If OpenAI API Key is available, run real RAGAS
        if self.openai_key:
            try:
                from ragas import evaluate
                from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
                
                # Format dataset
                data = {
                    "question": questions,
                    "contexts": contexts,
                    "answer": answers,
                    "ground_truth": ground_truths
                }
                dataset = Dataset.from_dict(data)
                
                # Evaluate
                result = evaluate(
                    dataset=dataset,
                    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
                )
                app_logger.info("RAGAS evaluation completed successfully.")
                return dict(result)
            except Exception as e:
                app_logger.error(f"RAGAS evaluation failed: {str(e)}. Using fallback scoring.")

        # Fallback heuristic scorer
        return self._heuristic_fallback(questions, contexts, answers, ground_truths)

    def _heuristic_fallback(
        self,
        questions: List[str],
        contexts: List[List[str]],
        answers: List[str],
        ground_truths: List[str]
    ) -> Dict[str, Any]:
        """Provides heuristic fallback metrics when API evaluation tools fail or keys are absent."""
        app_logger.info("Computing heuristic fallback RAGAS scores...")
        
        avg_faithfulness = 0.0
        avg_precision = 0.0
        avg_recall = 0.0
        avg_relevancy = 0.0
        
        for q, ctx, ans, gt in zip(questions, contexts, answers, ground_truths):
            # 1. Faithfulness: overlap of answer words with context
            ctx_text = " ".join(ctx).lower()
            ans_words = list(ans.lower().split())
            if ans_words:
                matched_words = [w for w in ans_words if w in ctx_text]
                faithfulness_score = min(len(matched_words) / len(ans_words), 1.0)
            else:
                faithfulness_score = 0.5
                
            # 2. Context Precision: overlap of question words with context
            q_words = list(q.lower().split())
            if q_words:
                matched_q = [w for w in q_words if w in ctx_text]
                precision_score = min(len(matched_q) / len(q_words), 1.0)
            else:
                precision_score = 0.5
                
            # 3. Context Recall: context coverage of ground truth
            gt_words = list(gt.lower().split())
            if gt_words:
                matched_gt = [w for w in gt_words if w in ctx_text]
                recall_score = min(len(matched_gt) / len(gt_words), 1.0)
            else:
                recall_score = 0.5

            # 4. Answer Relevancy: semantic similarity of query to answer
            ans_lower = ans.lower()
            relevancy_score = sum(1 for w in q_words if w in ans_lower) / len(q_words) if q_words else 0.5
            
            avg_faithfulness += faithfulness_score
            avg_precision += precision_score
            avg_recall += recall_score
            avg_relevancy += relevancy_score
            
        n = len(questions)
        return {
            "faithfulness": avg_faithfulness / n,
            "context_precision": avg_precision / n,
            "context_recall": avg_recall / n,
            "answer_relevancy": avg_relevancy / n
        }
