"""
Enhanced Query Rewriter - Integrates preprocessing for optimized retrieval.

Converts noisy queries into optimized retrieval queries using:
- Preprocessed query
- Extracted concepts
- Query type
- Domain context
- LLM-based rewriting
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from llm.groq_client import GroqLLMClient
from custom_logging.logger import app_logger
from query_understanding.concept_extractor import ConceptExtractionResult


class QueryRewriteResult(BaseModel):
    """Result of query rewriting."""
    original_query: str = Field(..., description="Original user query")
    rewritten_query: str = Field(..., description="Rewritten query for retrieval")
    rewrite_type: str = Field(default="none", description="Type of rewrite performed")
    concepts_used: list[str] = Field(default_factory=list, description="Concepts used in rewrite")
    confidence: float = Field(default=1.0, description="Confidence in rewrite")


class EnhancedQueryRewriter:
    """Enhanced query rewriter with preprocessing integration."""
    
    def __init__(self, llm_client: Optional[GroqLLMClient] = None):
        """
        Initialize the enhanced query rewriter.
        
        Args:
            llm_client: Optional LLM client (creates default if not provided)
        """
        self.logger = app_logger
        self.llm_client = llm_client or GroqLLMClient()
        self.rewrite_cache: Dict[str, QueryRewriteResult] = {}
    
    def rewrite(
        self,
        query: str,
        preprocessed_query: str,
        concept_result: Optional[ConceptExtractionResult] = None
    ) -> QueryRewriteResult:
        """
        Rewrite the query for optimal retrieval.
        
        Args:
            query: Original query
            preprocessed_query: Preprocessed query
            concept_result: Optional concept extraction result
            
        Returns:
            QueryRewriteResult with rewritten query and metadata
        """
        if not query or not isinstance(query, str):
            return QueryRewriteResult(
                original_query="",
                rewritten_query="",
                rewrite_type="error",
                confidence=0.0
            )
        
        # Check cache
        cache_key = f"{query}:{preprocessed_query}"
        if cache_key in self.rewrite_cache:
            self.logger.debug(f"Query rewrite cache hit for: '{query}'")
            return self.rewrite_cache[cache_key]
        
        # Determine if rewrite is needed
        if not self._needs_rewrite(query, preprocessed_query):
            return QueryRewriteResult(
                original_query=query,
                rewritten_query=preprocessed_query,
                rewrite_type="none",
                confidence=1.0
            )
        
        # Use LLM for rewriting
        result = self._llm_rewrite(
            query,
            preprocessed_query,
            concept_result
        )
        
        # Cache result
        self.rewrite_cache[cache_key] = result
        
        self.logger.info(
            f"Query rewrite: '{query}' -> '{result.rewritten_query}' "
            f"(type: {result.rewrite_type})"
        )
        
        return result
    
    def _needs_rewrite(self, original: str, preprocessed: str) -> bool:
        """
        Determine if query needs rewriting.
        
        Args:
            original: Original query
            preprocessed: Preprocessed query
            
        Returns:
            True if rewrite is needed
        """
        # If preprocessing changed the query significantly, rewrite might help
        if len(original) != len(preprocessed):
            return True
        
        # Check for common patterns that benefit from rewriting
        rewrite_patterns = [
            r'how does',  # Explanation queries
            r'what is',   # Definition queries
            r'explain',   # Explanation queries
            r'tell me about',  # General queries
        ]
        
        for pattern in rewrite_patterns:
            if re.search(pattern, original.lower()):
                return True
        
        return False
    
    def _llm_rewrite(
        self,
        original: str,
        preprocessed: str,
        concept_result: Optional[ConceptExtractionResult]
    ) -> QueryRewriteResult:
        """
        Use LLM to rewrite the query for optimal retrieval.
        
        Args:
            original: Original query
            preprocessed: Preprocessed query
            concept_result: Concept extraction result
            
        Returns:
            QueryRewriteResult
        """
        prompt = self._build_rewrite_prompt(
            original,
            preprocessed,
            concept_result
        )
        
        try:
            response = self.llm_client.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,  # Low temperature for consistent rewrites
                max_tokens=150
            )
            
            # Parse the response
            return self._parse_rewrite_response(response, original)
            
        except Exception as e:
            self.logger.error(f"LLM query rewrite failed: {e}")
            # Fallback: return preprocessed query
            return QueryRewriteResult(
                original_query=original,
                rewritten_query=preprocessed,
                rewrite_type="fallback",
                confidence=0.5
            )
    
    def _build_rewrite_prompt(
        self,
        original: str,
        preprocessed: str,
        concept_result: Optional[ConceptExtractionResult]
    ) -> str:
        """
        Build the prompt for LLM-based query rewriting.
        
        Args:
            original: Original query
            preprocessed: Preprocessed query
            concept_result: Concept extraction result
            
        Returns:
            Prompt string
        """
        prompt = f"""You are an expert at rewriting queries for optimal information retrieval.

Your task: Rewrite the following query to improve retrieval performance while preserving the original intent.

Original query: "{original}"
Preprocessed query: "{preprocessed}"
"""
        
        if concept_result and concept_result.concepts:
            prompt += f"\nKey concepts identified: {', '.join([c.text for c in concept_result.concepts])}"
            prompt += f"\nQuery type: {concept_result.query_type}"
            prompt += f"\nDomain: {concept_result.domain}"
        
        prompt += """

Rewriting rules:
1. Remove filler words (e.g., "tell me about", "can you explain")
2. Focus on key technical terms and concepts
3. Preserve the core meaning and intent
4. Use standard terminology (e.g., "multi-head attention" not "multiheaded attention")
5. Keep the query concise but complete
6. For definition queries, use the format: "[concept]"
7. For explanation queries, use the format: "[concept] [aspect]"
8. Return ONLY the rewritten query, no explanations

Rewritten query:"""
        
        return prompt
    
    def _parse_rewrite_response(self, response: str, original: str) -> QueryRewriteResult:
        """
        Parse the LLM response into a QueryRewriteResult.
        
        Args:
            response: LLM response
            original: Original query
            
        Returns:
            QueryRewriteResult
        """
        # Clean the response
        rewritten = response.strip().strip('"').strip("'")
        
        # Determine rewrite type
        rewrite_type = "general"
        if len(rewritten) < len(original):
            rewrite_type = "simplified"
        elif len(rewritten) > len(original):
            rewrite_type = "expanded"
        
        # Calculate confidence
        confidence = 0.9  # High confidence for LLM rewrites
        
        return QueryRewriteResult(
            original_query=original,
            rewritten_query=rewritten,
            rewrite_type=rewrite_type,
            confidence=confidence
        )
    
    def optimize_for_retrieval(
        self,
        query: str,
        query_type: str = "general"
    ) -> str:
        """
        Optimize query specifically for retrieval based on query type.
        
        Args:
            query: Input query
            query_type: Type of query (definition, explanation, comparison, etc.)
            
        Returns:
            Optimized query for retrieval
        """
        # Apply type-specific optimizations
        if query_type == "definition":
            # For definitions, extract the core concept
            words = query.split()
            # Remove "what is", "define", etc.
            stop_phrases = ["what is", "define", "explain", "tell me about"]
            for phrase in stop_phrases:
                if query.lower().startswith(phrase):
                    query = query[len(phrase):].strip()
                    break
        
        elif query_type == "explanation":
            # For explanations, focus on the concept and aspect
            # Remove "how does", "how do", etc.
            if query.lower().startswith("how does"):
                query = query[8:].strip()
            elif query.lower().startswith("how do"):
                query = query[6:].strip()
        
        return query
    
    def clear_cache(self):
        """Clear the query rewrite cache."""
        self.rewrite_cache.clear()
        self.logger.info("Query rewrite cache cleared")


# Import regex for pattern matching
import re
