"""
Spell Corrector - LLM-based dynamic spelling correction.

Uses LLM to correct spelling mistakes without hardcoded dictionaries.
Handles:
- Spelling mistakes
- Extra characters
- Missing characters
- Duplicated characters
- Grammatical variations
- Typing mistakes
"""

from typing import Optional, Dict, Any
from llm.groq_client import GroqLLMClient
from config.settings import settings
from custom_logging.logger import app_logger
from pydantic import BaseModel, Field


class SpellCorrectionResult(BaseModel):
    """Result of spell correction."""
    corrected_query: str = Field(..., description="Corrected query")
    corrections_made: list[str] = Field(default_factory=list, description="List of corrections made")
    confidence: float = Field(default=1.0, description="Confidence in correction")


class SpellCorrector:
    """LLM-based spell corrector for dynamic query correction."""
    
    def __init__(self, llm_client: Optional[GroqLLMClient] = None):
        """
        Initialize the spell corrector.
        
        Args:
            llm_client: Optional LLM client (creates default if not provided)
        """
        self.logger = app_logger
        self.llm_client = llm_client or GroqLLMClient()
        self.correction_cache: Dict[str, SpellCorrectionResult] = {}
    
    def correct(self, query: str) -> SpellCorrectionResult:
        """
        Correct spelling mistakes in the query using LLM.
        
        Args:
            query: Input query with potential spelling mistakes
            
        Returns:
            SpellCorrectionResult with corrected query and metadata
        """
        if not query or not isinstance(query, str):
            return SpellCorrectionResult(corrected_query="", corrections_made=[], confidence=0.0)
        
        # Check cache first
        if query in self.correction_cache:
            self.logger.debug(f"Spell correction cache hit for: '{query}'")
            return self.correction_cache[query]
        
        # Check if correction is needed (basic heuristic)
        if not self._needs_correction(query):
            return SpellCorrectionResult(
                corrected_query=query,
                corrections_made=[],
                confidence=1.0
            )
        
        # Use LLM for correction
        result = self._llm_correct(query)
        
        # Cache the result
        self.correction_cache[query] = result
        
        self.logger.info(
            f"Spell correction: '{query}' -> '{result.corrected_query}' "
            f"(corrections: {len(result.corrections_made)})"
        )
        
        return result
    
    def _needs_correction(self, query: str) -> bool:
        """
        Basic heuristic to check if correction might be needed.
        
        This is a fast pre-check to avoid unnecessary LLM calls.
        """
        # Check for common error patterns
        error_patterns = [
            r'(.)\1{2,}',  # Three or more repeated characters
            r'[aeiou]{3,}',  # Three or more consecutive vowels
            r'[^a-zA-Z0-9\s\-_]',  # Special characters that might be typos
        ]
        
        for pattern in error_patterns:
            if re.search(pattern, query):
                return True
        
        # Check for very short queries that might be typos
        if len(query.split()) == 1 and len(query) < 3:
            return True
        
        return False
    
    def _llm_correct(self, query: str) -> SpellCorrectionResult:
        """
        Use LLM to correct spelling mistakes.
        
        Args:
            query: Input query
            
        Returns:
            SpellCorrectionResult
        """
        prompt = self._build_correction_prompt(query)
        
        try:
            response = self.llm_client.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Low temperature for consistent corrections
                max_tokens=200
            )
            
            # Parse the LLM response
            return self._parse_correction_response(response, query)
            
        except Exception as e:
            self.logger.error(f"LLM spell correction failed: {e}")
            # Fallback: return original query
            return SpellCorrectionResult(
                corrected_query=query,
                corrections_made=[],
                confidence=0.0
            )
    
    def _build_correction_prompt(self, query: str) -> str:
        """
        Build the prompt for LLM-based spell correction.
        
        The prompt instructs the LLM to:
        - Correct spelling mistakes
        - Fix grammatical errors
        - Preserve technical terms
        - Maintain the original meaning
        """
        return f"""You are a spelling and grammar correction expert for technical and scientific queries.

Your task: Correct the following query while preserving technical terms, acronyms, and domain-specific vocabulary.

Rules:
1. Fix spelling mistakes (e.g., "attentionn" -> "attention", "transformars" -> "transformer")
2. Fix grammatical errors (e.g., "How transformer works" -> "How transformers work")
3. Preserve technical terms and acronyms (e.g., "RNN", "LLM", "BERT", "LangChain")
4. Preserve hyphenated terms (e.g., "multi-head attention")
5. Do not change the meaning of the query
6. Do not add or remove words unless necessary for correction
7. Return ONLY the corrected query, no explanations

Query to correct: "{query}"

Corrected query:"""
    
    def _parse_correction_response(self, response: str, original_query: str) -> SpellCorrectionResult:
        """
        Parse the LLM response into a SpellCorrectionResult.
        
        Args:
            response: LLM response
            original_query: Original query for comparison
            
        Returns:
            SpellCorrectionResult
        """
        # Clean the response
        corrected = response.strip().strip('"').strip("'")
        
        # Extract corrections made
        corrections = []
        if corrected != original_query:
            corrections.append("spelling_grammar_correction")
        
        # Calculate confidence based on changes
        confidence = 1.0
        if len(corrections) > 0:
            # Simple heuristic: more changes = lower confidence
            confidence = max(0.7, 1.0 - (len(corrections) * 0.1))
        
        return SpellCorrectionResult(
            corrected_query=corrected,
            corrections_made=corrections,
            confidence=confidence
        )
    
    def batch_correct(self, queries: list[str]) -> list[SpellCorrectionResult]:
        """
        Correct multiple queries in batch.
        
        Args:
            queries: List of queries to correct
            
        Returns:
            List of SpellCorrectionResults
        """
        results = []
        for query in queries:
            result = self.correct(query)
            results.append(result)
        return results
    
    def clear_cache(self):
        """Clear the correction cache."""
        self.correction_cache.clear()
        self.logger.info("Spell correction cache cleared")


# Import regex for pattern matching
import re
