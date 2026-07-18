from typing import List, Dict, Any, Optional
from llm.groq_client import GroqLLMClient
from config.settings import settings
from custom_logging.logger import app_logger
from langsmith import traceable

class QueryRewriter:
    """
    Rewrites user queries to improve retrieval quality using Groq LLM.
    
    Strategies:
    - Add relevant domain terms
    - Expand abbreviations
    - Add contextual information
    - Improve query specificity
    """
    
    def __init__(self, groq_api_key: str = None):
        self.groq_client = GroqLLMClient(
            api_key=groq_api_key or settings.groq_api_key
        )
    
    @traceable(name="Query Rewriting")
    def rewrite_query(
        self,
        original_query: str,
        retrieval_context: str = None,
        max_rewrites: int = 1
    ) -> str:
        """
        Rewrites a query to improve retrieval quality.
        
        Args:
            original_query: The original user query
            retrieval_context: Context from failed retrieval (optional)
            max_rewrites: Maximum number of rewrite attempts
            
        Returns:
            str: Rewritten query
        """
        system_prompt = """You are an expert query rewriter for a research paper retrieval system. 
Your task is to rewrite user queries to improve document retrieval while maintaining the original intent.

CRITICAL RULES:
1. PRESERVE named entities, proper nouns, and specific terms from the original query
2. DO NOT change document-specific names, authors, or entities to unrelated internet entities
3. Add relevant technical terms and domain-specific vocabulary ONLY if they enhance the original query
4. Expand abbreviations to their full forms if they are ambiguous
5. Add contextual information that might help retrieval WITHOUT changing the core meaning
6. Make the query more specific and precise while preserving the original entities
7. Output ONLY the rewritten query, no explanations
8. Keep the rewritten query concise (under 50 words)
9. If the query contains specific names/entities, keep them exactly as written

Examples:
- "Aldo Reni" → "Aldo Reni author research papers" (NOT "Aldo Rensi Italian philosopher")
- "transformer architecture" → "transformer neural network architecture attention mechanism"
- "BERT model" → "BERT bidirectional encoder representations transformers"
"""

        user_prompt = f"Original Query: {original_query}"
        
        if retrieval_context:
            user_prompt += f"\n\nRetrieval Context (failed to find relevant documents):\n{retrieval_context}"
        
        try:
            rewritten_query = self.groq_client.generate(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=100
            )
            
            app_logger.info(f"Query rewritten: '{original_query}' -> '{rewritten_query}'")
            return rewritten_query.strip()  
            
        except Exception as e:
            app_logger.error(f"Query rewriting failed: {str(e)}")
            return original_query  # Fallback to original query
    
    def expand_query(
        self,
        original_query: str,
        domain: str = "machine learning"
    ) -> List[str]:
        """
        Generates multiple query variations for broader retrieval.
        
        Args:
            original_query: The original user query
            domain: Domain context for query expansion
            
        Returns:
            List[str]: List of query variations
        """
        system_prompt = f"""You are an expert in {domain} research. 
Generate 3 different variations of the user's query to improve document retrieval.

Rules:
1. Each variation should focus on different aspects
2. Use different technical terms and phrasing
3. Keep each variation concise
4. Output each variation on a separate line
5. Number the variations (1., 2., 3.)"""

        user_prompt = f"Original Query: {original_query}"
        
        try:
            response = self.groq_client.generate(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,
                max_tokens=150
            )
            
            # Parse numbered variations
            variations = []
            for line in response.strip().split('\n'):
                line = line.strip()
                if line and (line.startswith(('1.', '2.', '3.'))):
                    # Remove numbering
                    variation = line.split('.', 1)[1].strip() if '.' in line else line
                    variations.append(variation)
            
            app_logger.info(f"Generated {len(variations)} query variations for: '{original_query}'")
            return variations if variations else [original_query]
            
        except Exception as e:
            app_logger.error(f"Query expansion failed: {str(e)}")
            return [original_query]
    
    def add_contextual_terms(
        self,
        query: str,
        chat_history: str = None
    ) -> str:
        """
        Adds contextual terms from conversation history to the query.
        
        Args:
            query: The current query
            chat_history: Recent conversation history
            
        Returns:
            str: Contextually enhanced query
        """
        if not chat_history:
            return query
        
        system_prompt = """You are a context-aware query enhancer. 
Add relevant contextual information from conversation history to the current query.

Rules:
1. Identify key topics, entities, or concepts from the conversation
2. Add 2-3 relevant terms to the current query
3. Maintain the original query's structure
4. Output ONLY the enhanced query
5. Keep it concise"""

        user_prompt = f"""Conversation History:
{chat_history}

Current Query: {query}

Enhanced Query:"""

        try:
            enhanced_query = self.groq_client.generate(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=80
            )
            
            app_logger.info(f"Query contextually enhanced: '{query}' -> '{enhanced_query}'")
            return enhanced_query.strip()
            
        except Exception as e:
            app_logger.error(f"Contextual query enhancement failed: {str(e)}")
            return query
