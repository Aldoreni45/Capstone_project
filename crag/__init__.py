"""
CRAG (Corrective Retrieval-Augmented Generation) Module

This module provides production-grade CRAG capabilities for the RAG pipeline:
- Retrieval Evaluation
- Query Rewriting
- Corrective Retrieval
- Web Search Fallback
- Context Merging
"""

from crag.retrieval_evaluator import RetrievalEvaluator, RetrievalQuality
from crag.query_rewriter import QueryRewriter
from crag.context_merger import ContextMerger

__all__ = [
    "RetrievalEvaluator",
    "RetrievalQuality",
    "QueryRewriter",
    "ContextMerger"
]
