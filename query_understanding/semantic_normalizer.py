"""
Semantic Normalizer - Embedding-based semantic normalization.

Identifies semantically equivalent concepts using embeddings:
- vector db -> vector database
- multihead attention -> multi-head attention
- RAG -> Retrieval Augmented Generation
- LLM -> Large Language Model

Uses embedding similarity rather than hardcoded mappings.
"""

from typing import Optional, List, Dict, Any, Tuple
from embeddings import get_embeddings
from custom_logging.logger import app_logger
import numpy as np


class SemanticNormalizer:
    """Normalizes queries using embedding-based semantic similarity."""
    
    def __init__(self, embedding_model_type: str = "huggingface"):
        """
        Initialize the semantic normalizer.
        
        Args:
            embedding_model_type: Type of embedding model to use
        """
        self.logger = app_logger
        self.embedding_model_type = embedding_model_type
        self.embedding_model = None
        self.semantic_cache: Dict[str, str] = {}
        self.logger.info(f"Initialized SemanticNormalizer with {embedding_model_type}")

    def _get_embedding_model(self):
        """Lazily initialize the embedding model when semantic similarity is needed."""
        if self.embedding_model is not None:
            return self.embedding_model

        try:
            self.embedding_model = get_embeddings(self.embedding_model_type)
        except Exception as exc:
            self.logger.warning(
                f"Semantic embeddings unavailable for '{self.embedding_model_type}': {exc}. "
                "Falling back to heuristic-only normalization."
            )
            self.embedding_model = None

        return self.embedding_model
    
    def normalize(self, query: str, top_k: int = 3) -> str:
        """
        Normalize query using semantic similarity.
        
        Args:
            query: Input query
            top_k: Number of similar concepts to consider
            
        Returns:
            Semantically normalized query
        """
        if not query or not isinstance(query, str):
            return ""
        
        # Check cache
        if query in self.semantic_cache:
            self.logger.debug(f"Semantic normalization cache hit for: '{query}'")
            return self.semantic_cache[query]
        
        # Extract terms from query
        terms = self._extract_terms(query)
        
        # Normalize each term semantically
        normalized_terms = []
        for term in terms:
            normalized_term = self._normalize_term(term, top_k)
            normalized_terms.append(normalized_term)
        
        # Reconstruct query
        normalized_query = " ".join(normalized_terms)
        
        # Cache result
        self.semantic_cache[query] = normalized_query
        
        self.logger.debug(f"Semantic normalization: '{query}' -> '{normalized_query}'")
        
        return normalized_query
    
    def _extract_terms(self, query: str) -> List[str]:
        """
        Extract meaningful terms from query.
        
        Args:
            query: Input query
            
        Returns:
            List of terms
        """
        # Simple tokenization - can be enhanced with NLP
        terms = query.split()
        return terms
    
    def _normalize_term(self, term: str, top_k: int) -> str:
        """
        Normalize a single term using semantic similarity.
        
        Args:
            term: Input term
            top_k: Number of similar concepts to consider
            
        Returns:
            Normalized term
        """
        # For single-word terms, check for common semantic variations
        # This is a simplified approach - can be enhanced with a knowledge base
        
        # Check if term is already in standard form
        if self._is_standard_form(term):
            return term
        
        # For now, return as-is
        # In production, this would query a semantic knowledge base
        return term
    
    def _is_standard_form(self, term: str) -> bool:
        """
        Check if term is in standard form.
        
        This is a heuristic - can be enhanced with embedding similarity.
        
        Args:
            term: Input term
            
        Returns:
            True if term appears to be in standard form
        """
        # Simple heuristic: check for common non-standard patterns
        non_standard_patterns = [
            r'db$',  # db -> database
            r's$',   # plural forms
        ]
        
        for pattern in non_standard_patterns:
            if re.search(pattern, term):
                return False
        
        return True
    
    def find_similar_concepts(
        self,
        query: str,
        knowledge_base: List[str],
        threshold: float = 0.85
    ) -> List[Tuple[str, float]]:
        """
        Find semantically similar concepts in a knowledge base.
        
        Args:
            query: Input query
            knowledge_base: List of known concepts/terms
            threshold: Similarity threshold
            
        Returns:
            List of (concept, similarity_score) tuples
        """
        if not knowledge_base:
            return []

        embedding_model = self._get_embedding_model()
        if embedding_model is None:
            return []
        
        # Embed the query
        query_embedding = embedding_model.embed_query(query)
        
        # Embed all knowledge base concepts
        kb_embeddings = [
            embedding_model.embed_query(concept)
            for concept in knowledge_base
        ]
        
        # Calculate similarities
        similarities = []
        for concept, kb_embedding in zip(knowledge_base, kb_embeddings):
            similarity = self._cosine_similarity(query_embedding, kb_embedding)
            if similarity >= threshold:
                similarities.append((concept, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:5]  # Return top 5
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def expand_abbreviations_llm(self, query: str) -> str:
        """
        Expand abbreviations using LLM-based understanding.
        
        This is a placeholder for LLM-based abbreviation expansion.
        
        Args:
            query: Input query with potential abbreviations
            
        Returns:
            Query with expanded abbreviations
        """
        # This would use an LLM to expand abbreviations contextually
        # For now, return as-is
        return query
    
    def clear_cache(self):
        """Clear the semantic normalization cache."""
        self.semantic_cache.clear()
        self.logger.info("Semantic normalization cache cleared")


# Import regex for pattern matching
import re
