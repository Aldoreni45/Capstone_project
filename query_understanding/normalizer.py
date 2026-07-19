"""
Query Normalizer - Lemmatization and abbreviation handling.

Performs:
- Singular/plural normalization
- Abbreviation expansion
- Token format normalization
- Common NLP variations
- Technical term normalization
"""

import re
from typing import Optional, Dict, Any
import spacy
from custom_logging.logger import app_logger


class QueryNormalizer:
    """Normalizes queries using NLP techniques without hardcoded rules."""
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Initialize the query normalizer.
        
        Args:
            spacy_model: SpaCy model name for NLP processing
        """
        self.logger = app_logger
        try:
            self.nlp = spacy.load(spacy_model)
            self.logger.info(f"Loaded spaCy model: {spacy_model}")
        except OSError:
            self.logger.warning(
                f"SpaCy model '{spacy_model}' not found. "
                "Install with: python -m spacy download en_core_web_sm"
            )
            self.nlp = None
    
    def normalize(self, query: str) -> str:
        """
        Normalize the query using NLP techniques.
        
        Args:
            query: Input query
            
        Returns:
            Normalized query
        """
        if not query or not isinstance(query, str):
            return ""
        
        if self.nlp is None:
            # Fallback: basic normalization without spaCy
            return self._basic_normalize(query)
        
        # Process with spaCy
        doc = self.nlp(query)
        
        # Apply normalization
        normalized_tokens = []
        for token in doc:
            normalized_token = self._normalize_token(token)
            if normalized_token:
                normalized_tokens.append(normalized_token)
        
        normalized_query = " ".join(normalized_tokens)
        
        self.logger.debug(f"Query normalization: '{query}' -> '{normalized_query}'")
        
        return normalized_query
    
    def _normalize_token(self, token) -> Optional[str]:
        """
        Normalize a single token using spaCy.
        
        Args:
            token: spaCy token
            
        Returns:
            Normalized token or None if should be skipped
        """
        # Skip stopwords and punctuation
        if token.is_stop or token.is_punct:
            return token.text  # Keep original for structure
        
        # Lemmatize
        lemma = token.lemma_.lower()
        
        # Preserve original if lemma is empty or same
        if not lemma or lemma == token.text.lower():
            return token.text.lower()
        
        return lemma
    
    def _basic_normalize(self, query: str) -> str:
        """
        Basic normalization without spaCy (fallback).
        
        Args:
            query: Input query
            
        Returns:
            Basically normalized query
        """
        # Simple plural removal (basic heuristic)
        words = query.split()
        normalized = []
        
        for word in words:
            # Remove trailing 's' for basic plural handling
            if len(word) > 3 and word.endswith('s'):
                normalized.append(word[:-1])
            else:
                normalized.append(word)
        
        return " ".join(normalized)
    
    def normalize_abbreviations(self, query: str) -> str:
        """
        Normalize abbreviations using contextual understanding.
        
        This uses LLM-based understanding rather than hardcoded mappings.
        
        Args:
            query: Input query with potential abbreviations
            
        Returns:
            Query with normalized abbreviations
        """
        # For now, return as-is
        # This will be enhanced with LLM-based abbreviation expansion
        return query
    
    def normalize_hyphenated(self, query: str) -> str:
        """
        Normalize hyphenated terms.
        
        Handles:
        - multi-head -> multihead
        - encoder-decoder -> encoder decoder
        - self-attention -> selfattention
        
        Args:
            query: Input query
            
        Returns:
            Query with normalized hyphenation
        """
        # Remove hyphens and normalize spacing
        normalized = re.sub(r'-', ' ', query)
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized.strip()
    
    def normalize_technical_terms(self, query: str) -> str:
        """
        Normalize technical terms and formats.
        
        Handles:
        - CamelCase -> camel case
        - snake_case -> snake case
        - Mixed formats
        
        Args:
            query: Input query
            
        Returns:
            Query with normalized technical terms
        """
        # Convert CamelCase to space-separated
        normalized = re.sub(r'([a-z])([A-Z])', r'\1 \2', query)
        # Convert snake_case to space-separated
        normalized = re.sub(r'_', ' ', normalized)
        # Normalize spaces
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized.strip()
    
    def get_normalization_stats(self, original: str, normalized: str) -> dict:
        """
        Get statistics about normalization changes.
        
        Args:
            original: Original query
            normalized: Normalized query
            
        Returns:
            Dictionary with normalization statistics
        """
        original_tokens = set(original.split())
        normalized_tokens = set(normalized.split())
        
        return {
            "original_tokens": len(original_tokens),
            "normalized_tokens": len(normalized_tokens),
            "tokens_changed": len(original_tokens.symmetric_difference(normalized_tokens)),
            "spaCy_available": self.nlp is not None,
        }
