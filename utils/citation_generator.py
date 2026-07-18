from typing import List
from pydantic_models.responses import RetrievedChunk
from pydantic_models.citation import Citation
from custom_logging.logger import app_logger
from langsmith import traceable

class CitationGenerator:
    """
    Production-grade citation generator that extracts citations directly from 
    RetrievedChunk metadata. No LLM validation required.
    
    Source of truth: RetrievedChunk objects from the retriever.
    """
    
    def __init__(self, max_passage_length: int = 200):
        self.max_passage_length = max_passage_length
    
    @traceable(name="Citation Generation")
    def generate_citations(
        self,
        retrieved_chunks: List[RetrievedChunk],
        max_citations: int = 3
    ) -> List[Citation]:
        """
        Generates citations from retrieved chunks using metadata extraction.
        
        Args:
            retrieved_chunks: List of retrieved chunks (sorted by relevance)
            max_citations: Maximum number of citations to generate
            
        Returns:
            List[Citation]: Generated citation objects with metadata from chunks
        """
        if not retrieved_chunks:
            app_logger.debug("No chunks provided for citation generation")
            return []
        
        citations = []
        
        for chunk in retrieved_chunks[:max_citations]:
            try:
                citation = self._create_citation_from_chunk(chunk)
                citations.append(citation)
            except Exception as e:
                app_logger.warning(f"Failed to generate citation for chunk {chunk.chunk_id}: {str(e)}")
                continue
        
        app_logger.info(f"Generated {len(citations)} citations from {len(retrieved_chunks)} chunks")
        return citations
    
    def _create_citation_from_chunk(self, chunk: RetrievedChunk) -> Citation:
        """
        Creates a citation object from a retrieved chunk.
        
        Extracts metadata directly from the chunk object - no validation needed
        since the chunk comes from the retriever.
        """
        passage = self._extract_passage(chunk.content)
        
        return Citation(
            title=chunk.title,
            page=chunk.page,
            chunk_id=chunk.chunk_id,
            source=chunk.source,
            passage=passage
        )
    
    def _extract_passage(self, content: str) -> str:
        """
        Extracts a passage from chunk content with proper word boundary handling.
        
        Args:
            content: Full chunk content
            
        Returns:
            str: Extracted passage with proper truncation
        """
        if not content:
            return ""
        
        if len(content) <= self.max_passage_length:
            return content
        
        # Truncate at word boundary
        passage = content[:self.max_passage_length]
        last_space = passage.rfind(' ')
        
        if last_space > 0:
            passage = passage[:last_space]
        
        return passage + "..."
    
    def format_citations(self, citations: List[Citation]) -> str:
        """
        Formats citations into a readable string for display.
        
        Args:
            citations: List of citation objects
            
        Returns:
            str: Formatted citations string
        """
        if not citations:
            return ""
        
        formatted = []
        for idx, citation in enumerate(citations, 1):
            formatted.append(
                f"{idx}. {citation.title} (Page {citation.page}, Chunk ID: {citation.chunk_id})"
            )
        
        return "\n".join(formatted)
