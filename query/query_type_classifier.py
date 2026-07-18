from typing import Literal
from custom_logging.logger import app_logger
import re

QueryType = Literal[
    "named_entity",
    "concept",
    "methodology",
    "architecture",
    "comparison",
    "numerical",
    "general_research"
]

class QueryTypeClassifier:
    """
    Classifies user queries into different types to determine
    appropriate retrieval evaluation strategies.
    
    This is a DYNAMIC classifier that does not rely on hardcoded entity lists.
    It uses linguistic patterns, query structure analysis, and capitalization rules
    to identify named entities dynamically.
    
    Query Types:
    - named_entity: Queries about specific people, models, or named entities
    - concept: Queries about concepts, definitions, or explanations
    - methodology: Queries about how something works
    - architecture: Queries about system or model architecture
    - comparison: Queries comparing multiple things
    - numerical: Queries asking for numbers, statistics, or counts
    - general_research: General research questions
    """
    
    def __init__(self):
        # Named entity patterns (strong indicators)
        self.named_entity_patterns = [
            "tell me about",
            "who is",
            "who was",
            "describe",
            "information about",
            "details about",
            "biography of",
            "background of",
        ]
        
        # Concept indicators (definition/explanation questions)
        self.concept_patterns = [
            "what is",
            "what are",
            "what does",
            "define",
            "explain",
            "meaning of",
            "definition of",
        ]
        
        # Methodology indicators (how questions)
        self.methodology_patterns = [
            "how does",
            "how do",
            "how to",
            "how can",
            "how would",
            "method",
            "approach",
            "technique",
            "process",
        ]
        
        # Architecture indicators (structure/design questions)
        self.architecture_patterns = [
            "architecture",
            "structure",
            "components",
            "layers",
            "design",
            "framework",
            "schema",
            "layout",
        ]
        
        # Comparison indicators
        self.comparison_patterns = [
            "compare",
            "difference",
            "vs",
            "versus",
            "better",
            "worse",
            "similar",
            "contrast",
        ]
        
        # Numerical indicators
        self.numerical_patterns = [
            "how many",
            "how much",
            "count",
            "number",
            "statistics",
            "percentage",
            "ratio",
            "quantity",
        ]
    
    def classify(self, query: str) -> QueryType:
        """
        Classifies the query type based on patterns and content.
        
        Args:
            query: The user query string
            
        Returns:
            QueryType: The classified query type
        """
        # Normalize query by removing extra spaces (preserve case for entity detection)
        query_normalized = ' '.join(query.split())
        query_lower = query_normalized.lower().strip()
        
        # Check for named entity queries (use normalized with preserved case)
        if self._is_named_entity_query(query_normalized):
            return "named_entity"
        
        # Check for comparison queries
        if self._is_comparison_query(query_lower):
            return "comparison"
        
        # Check for numerical queries
        if self._is_numerical_query(query_lower):
            return "numerical"
        
        # Check for methodology queries
        if self._is_methodology_query(query_lower):
            return "methodology"
        
        # Check for architecture queries
        if self._is_architecture_query(query_lower):
            return "architecture"
        
        # Check for concept queries
        if self._is_concept_query(query_lower):
            return "concept"
        
        # Default to general research
        return "general_research"
    
    def _is_named_entity_query(self, query: str) -> bool:
        """
        DYNAMIC named entity detection using linguistic patterns and capitalization.
        
        This method does NOT use hardcoded entity lists. It identifies named entities
        through:
        1. Strong linguistic patterns ("tell me about", "who is")
        2. Capitalization analysis (proper nouns)
        3. Query structure analysis
        
        Args:
            query: The normalized query string (extra spaces removed, case preserved)
        """
        query_lower = query.lower().strip()
        
        # STRONG INDICATORS: Named entity patterns
        for pattern in self.named_entity_patterns:
            if pattern in query_lower:
                app_logger.info(f"Named entity detected via pattern '{pattern}': {query}")
                return True
        
        # CAPITALIZATION ANALYSIS: Detect proper nouns
        words = query.split()
        capitalized_words = [word for word in words if word[0].isupper() and len(word) > 1]
        
        # If there are capitalized words, analyze the query structure
        if len(capitalized_words) >= 1:
            # Check if the query is NOT a concept question
            # Concept questions typically start with "what is", "define", etc.
            is_concept_question = any(pattern in query_lower for pattern in self.concept_patterns)
            
            # If it's not a concept question and has capitalized words, it's likely a named entity
            if not is_concept_question:
                app_logger.info(f"Named entity detected via capitalization (not a concept question): {query}")
                return True
            
            # If it IS a concept question but has capitalized words after the pattern,
            # it might still be a named entity (e.g., "What is BERT?")
            # Check if capitalized words appear after concept patterns
            for pattern in self.concept_patterns:
                if pattern in query_lower:
                    pattern_idx = query_lower.find(pattern)
                    after_pattern = query_lower[pattern_idx + len(pattern):].strip()
                    if after_pattern and any(word[0].isupper() for word in after_pattern.split() if len(word) > 1):
                        app_logger.info(f"Named entity detected via capitalized word after concept pattern: {query}")
                        return True
        
        return False
    
    def _is_concept_query(self, query: str) -> bool:
        """
        Determines if the query is about a concept or definition.
        """
        return any(pattern in query for pattern in self.concept_patterns)
    
    def _is_methodology_query(self, query: str) -> bool:
        """
        Determines if the query is about methodology or how something works.
        """
        return any(pattern in query for pattern in self.methodology_patterns)
    
    def _is_architecture_query(self, query: str) -> bool:
        """
        Determines if the query is about architecture or structure.
        """
        return any(pattern in query for pattern in self.architecture_patterns)
    
    def _is_comparison_query(self, query: str) -> bool:
        """
        Determines if the query is a comparison.
        """
        return any(pattern in query for pattern in self.comparison_patterns)
    
    def _is_numerical_query(self, query: str) -> bool:
        """
        Determines if the query asks for numerical information.
        """
        return any(pattern in query for pattern in self.numerical_patterns)
    
    def should_use_entity_matching(self, query_type: QueryType) -> bool:
        """
        Determines whether named entity matching should be used
        for the given query type.
        
        Args:
            query_type: The classified query type
            
        Returns:
            bool: True if entity matching should be used
        """
        return query_type == "named_entity"
