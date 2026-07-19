"""
Concept Extractor - LLM-based entity and concept extraction.

Extracts:
- Entities (technologies, frameworks, models)
- Research concepts
- Algorithms
- Methods
- Tools
- Databases
- Papers

Uses LLM-based NER rather than hardcoded lists.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from llm.groq_client import GroqLLMClient
from custom_logging.logger import app_logger


class ExtractedConcept(BaseModel):
    """Represents an extracted concept."""
    text: str = Field(..., description="The concept text")
    category: str = Field(..., description="Category (e.g., technology, algorithm, framework)")
    confidence: float = Field(default=1.0, description="Confidence score")
    start_pos: int = Field(default=0, description="Start position in original query")
    end_pos: int = Field(default=0, description="End position in original query")


class ConceptExtractionResult(BaseModel):
    """Result of concept extraction."""
    concepts: list[ExtractedConcept] = Field(default_factory=list, description="Extracted concepts")
    query_type: str = Field(default="general", description="Query type classification")
    domain: str = Field(default="general", description="Domain classification")


class ConceptExtractor:
    """LLM-based concept extractor for dynamic entity recognition."""
    
    def __init__(self, llm_client: Optional[GroqLLMClient] = None):
        """
        Initialize the concept extractor.
        
        Args:
            llm_client: Optional LLM client (creates default if not provided)
        """
        self.logger = app_logger
        self.llm_client = llm_client or GroqLLMClient()
        self.extraction_cache: Dict[str, ConceptExtractionResult] = {}
    
    def extract(self, query: str) -> ConceptExtractionResult:
        """
        Extract concepts from the query using LLM.
        
        Args:
            query: Input query
            
        Returns:
            ConceptExtractionResult with extracted concepts
        """
        if not query or not isinstance(query, str):
            return ConceptExtractionResult(
                concepts=[],
                query_type="general",
                domain="general"
            )
        
        # Check cache
        if query in self.extraction_cache:
            self.logger.debug(f"Concept extraction cache hit for: '{query}'")
            return self.extraction_cache[query]
        
        # Use LLM for extraction
        result = self._llm_extract(query)
        
        # Cache result
        self.extraction_cache[query] = result
        
        self.logger.info(
            f"Concept extraction: '{query}' -> {len(result.concepts)} concepts, "
            f"type: {result.query_type}, domain: {result.domain}"
        )
        
        return result
    
    def _llm_extract(self, query: str) -> ConceptExtractionResult:
        """
        Use LLM to extract concepts from the query.
        
        Args:
            query: Input query
            
        Returns:
            ConceptExtractionResult
        """
        prompt = self._build_extraction_prompt(query)
        
        try:
            response = self.llm_client.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Low temperature for consistent extraction
                max_tokens=300
            )
            
            # Parse the LLM response
            return self._parse_extraction_response(response, query)
            
        except Exception as e:
            self.logger.error(f"LLM concept extraction failed: {e}")
            # Fallback: return empty result
            return ConceptExtractionResult(
                concepts=[],
                query_type="general",
                domain="general"
            )
    
    def _build_extraction_prompt(self, query: str) -> str:
        """
        Build the prompt for LLM-based concept extraction.
        
        The prompt instructs the LLM to:
        - Extract technical concepts
        - Classify concept categories
        - Identify query type
        - Determine domain
        """
        return f"""You are an expert in extracting technical and scientific concepts from queries.

Your task: Extract key concepts from the following query and classify them.

Concept categories to identify:
- technology: Frameworks, libraries, tools (e.g., PyTorch, TensorFlow, LangChain)
- model: AI/ML models (e.g., BERT, GPT, RNN, Transformer)
- algorithm: Algorithms and methods (e.g., attention mechanism, backpropagation)
- database: Data storage systems (e.g., Weaviate, Pinecone, vector database)
- paper: Research papers (e.g., "Attention Is All You Need")
- method: Techniques and approaches (e.g., semantic chunking, retrieval)
- concept: General technical concepts (e.g., RAG, LLM, embedding)

Query types:
- definition: Asking "what is X"
- explanation: Asking "how X works"
- comparison: Asking "difference between X and Y"
- implementation: Asking "how to implement X"
- general: General question

Domains:
- nlp: Natural Language Processing
- ml: Machine Learning
- dl: Deep Learning
- rag: Retrieval Augmented Generation
- database: Database systems
- general: General knowledge

Query: "{query}"

Extract the concepts and provide the result in this format:
CONCEPTS:
- concept_text (category)

QUERY_TYPE: [type]

DOMAIN: [domain]

Provide ONLY the extraction, no explanations."""
    
    def _parse_extraction_response(self, response: str, original_query: str) -> ConceptExtractionResult:
        """
        Parse the LLM response into a ConceptExtractionResult.
        
        Args:
            response: LLM response
            original_query: Original query for position tracking
            
        Returns:
            ConceptExtractionResult
        """
        concepts = []
        query_type = "general"
        domain = "general"
        
        lines = response.strip().split('\n')
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('CONCEPTS:'):
                current_section = 'concepts'
            elif line.startswith('QUERY_TYPE:'):
                current_section = 'query_type'
                query_type = line.split(':', 1)[1].strip()
            elif line.startswith('DOMAIN:'):
                current_section = 'domain'
                domain = line.split(':', 1)[1].strip()
            elif line.startswith('-') and current_section == 'concepts':
                # Parse concept: "concept_text (category)"
                concept_line = line[1:].strip()
                if '(' in concept_line and concept_line.endswith(')'):
                    concept_text = concept_line[:concept_line.rfind('(')].strip()
                    category = concept_line[concept_line.rfind('(')+1:-1].strip()
                    
                    # Find position in original query
                    start_pos = original_query.lower().find(concept_text.lower())
                    end_pos = start_pos + len(concept_text) if start_pos >= 0 else 0
                    
                    concepts.append(ExtractedConcept(
                        text=concept_text,
                        category=category,
                        confidence=0.9,  # High confidence for LLM extraction
                        start_pos=start_pos,
                        end_pos=end_pos
                    ))
        
        return ConceptExtractionResult(
            concepts=concepts,
            query_type=query_type,
            domain=domain
        )
    
    def get_key_concepts(self, query: str, top_k: int = 5) -> List[str]:
        """
        Get the top key concepts from a query.
        
        Args:
            query: Input query
            top_k: Number of top concepts to return
            
        Returns:
            List of key concept texts
        """
        result = self.extract(query)
        return [c.text for c in result.concepts[:top_k]]
    
    def get_query_type(self, query: str) -> str:
        """
        Get the query type classification.
        
        Args:
            query: Input query
            
        Returns:
            Query type string
        """
        result = self.extract(query)
        return result.query_type
    
    def get_domain(self, query: str) -> str:
        """
        Get the domain classification.
        
        Args:
            query: Input query
            
        Returns:
            Domain string
        """
        result = self.extract(query)
        return result.domain
    
    def clear_cache(self):
        """Clear the concept extraction cache."""
        self.extraction_cache.clear()
        self.logger.info("Concept extraction cache cleared")
