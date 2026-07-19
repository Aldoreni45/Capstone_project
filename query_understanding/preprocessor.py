"""
Query Preprocessor - Text normalization and cleaning.

Performs basic text preprocessing including:
- Lowercase normalization
- Punctuation removal
- Whitespace normalization
- Token cleaning
- Unicode normalization
"""

import re
import unicodedata
from typing import Optional
from custom_logging.logger import app_logger


class QueryPreprocessor:
    """Preprocesses user queries through text normalization and cleaning."""
    
    def __init__(self):
        """Initialize the query preprocessor."""
        self.logger = app_logger
    
    def preprocess(self, query: str) -> str:
        """
        Apply all preprocessing steps to the query.
        
        Args:
            query: Raw user query
            
        Returns:
            Preprocessed query string
        """
        if not query or not isinstance(query, str):
            return ""
        
        original_query = query
        processed = query
        
        # Apply preprocessing steps
        processed = self._normalize_unicode(processed)
        processed = self._normalize_case(processed)
        processed = self._normalize_whitespace(processed)
        processed = self._clean_punctuation(processed)
        processed = self._clean_special_chars(processed)
        processed = self._normalize_quotes(processed)
        
        self.logger.debug(
            f"Query preprocessing: '{original_query}' -> '{processed}'"
        )
        
        return processed
    
    def _normalize_unicode(self, text: str) -> str:
        """
        Normalize unicode characters to NFKC form.
        
        This handles:
        - Unicode normalization
        - Accented characters
        - Compatibility characters
        """
        return unicodedata.normalize('NFKC', text)
    
    def _normalize_case(self, text: str) -> str:
        """
        Normalize case to lowercase.
        
        Note: We preserve some capitalization for proper nouns in later stages.
        This is a basic normalization that can be enhanced.
        """
        return text.lower()
    
    def _normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace characters.
        
        Handles:
        - Multiple spaces to single space
        - Tabs to spaces
        - Newlines to spaces
        - Leading/trailing whitespace
        """
        # Replace all whitespace characters with space
        text = re.sub(r'\s+', ' ', text)
        # Strip leading/trailing whitespace
        text = text.strip()
        return text
    
    def _clean_punctuation(self, text: str) -> str:
        """
        Clean and normalize punctuation.
        
        Handles:
        - Multiple consecutive punctuation
        - Punctuation spacing
        - Common punctuation errors
        """
        # Remove multiple consecutive punctuation marks (keep last one)
        text = re.sub(r'([!?.,;:])\1+', r'\1', text)
        # Ensure proper spacing after punctuation
        text = re.sub(r'([!?.,;:])([a-zA-Z])', r'\1 \2', text)
        return text
    
    def _clean_special_chars(self, text: str) -> str:
        """
        Clean special characters while preserving meaningful ones.
        
        Preserves:
        - Alphanumeric characters
        - Basic punctuation (.,!?;:-)
        - Hyphens and underscores (common in technical terms)
        """
        # Keep only alphanumeric, spaces, and basic punctuation
        # This is conservative - can be adjusted based on needs
        allowed_chars = r'[a-zA-Z0-9\s.,!?;:\-_]'
        text = re.sub(allowed_chars, ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _normalize_quotes(self, text: str) -> str:
        """
        Normalize quote characters.
        
        Handles:
        - Smart quotes to straight quotes
        - Quote spacing
        """
        # Replace smart quotes with straight quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        return text
    
    def get_preprocessing_stats(self, original: str, processed: str) -> dict:
        """
        Get statistics about the preprocessing changes.
        
        Args:
            original: Original query
            processed: Processed query
            
        Returns:
            Dictionary with preprocessing statistics
        """
        return {
            "original_length": len(original),
            "processed_length": len(processed),
            "length_reduction": len(original) - len(processed),
            "whitespace_normalized": original != re.sub(r'\s+', ' ', original),
            "case_normalized": original.lower() != original,
        }
