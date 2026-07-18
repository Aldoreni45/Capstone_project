from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chunking.base import BaseChunker
from config.settings import settings
from custom_logging.logger import app_logger


class ParentChildChunker(BaseChunker):
    """
    Implements parent-child chunking strategy for research papers.
    
    Strategy:
    - Create small child chunks (200-400 chars) for precise retrieval
    - Create larger parent chunks (1000-1500 chars) for full context
    - Maintain parent-child mapping for context expansion
    - Optimize for research paper structure (sections, paragraphs)
    """

    def __init__(
        self,
        child_chunk_size: int = 300,
        child_chunk_overlap: int = 50,
        parent_chunk_size: int = 1200,
        parent_chunk_overlap: int = 200
    ):
        self.child_chunk_size = child_chunk_size
        self.child_chunk_overlap = child_chunk_overlap
        self.parent_chunk_size = parent_chunk_size
        self.parent_chunk_overlap = parent_chunk_overlap

        # Initialize splitters
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.child_chunk_size,
            chunk_overlap=self.child_chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )

        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_chunk_size,
            chunk_overlap=self.parent_chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits documents into parent-child chunks.
        
        Returns:
            List of child chunks with parent metadata for context expansion
        """
        app_logger.info(
            f"Running parent-child chunking "
            f"(child_size={self.child_chunk_size}, parent_size={self.parent_chunk_size})"
        )

        all_child_chunks = []

        for doc in documents:
            # Create parent chunks first
            parent_chunks = self.parent_splitter.split_documents([doc])
            
            # For each parent chunk, create child chunks
            for parent_idx, parent_chunk in enumerate(parent_chunks):
                # Create child chunks from this parent
                child_chunks = self.child_splitter.split_documents([parent_chunk])
                
                # Enrich child chunks with parent metadata
                for child_idx, child_chunk in enumerate(child_chunks):
                    # Preserve original metadata
                    child_chunk.metadata.update(doc.metadata)
                    
                    # Add parent-child relationship metadata
                    child_chunk.metadata["parent_chunk_id"] = f"{doc.metadata.get('source', 'unknown')}_p{doc.metadata.get('page_number', 0)}_parent{parent_idx}"
                    child_chunk.metadata["parent_chunk_index"] = parent_idx
                    child_chunk.metadata["child_chunk_index"] = child_idx
                    child_chunk.metadata["parent_content"] = parent_chunk.page_content
                    child_chunk.metadata["chunking_strategy"] = "parent_child"
                    child_chunk.metadata["child_chunk_size"] = len(child_chunk.page_content)
                    child_chunk.metadata["parent_chunk_size"] = len(parent_chunk.page_content)
                    
                    all_child_chunks.append(child_chunk)

        # Add chunk numbering
        numbered_chunks = self.add_chunk_numbering(all_child_chunks)
        
        app_logger.info(
            f"Parent-child chunking complete: "
            f"Created {len(numbered_chunks)} child chunks from {len(documents)} documents"
        )
        
        return numbered_chunks

    def get_parent_context(self, child_chunk: Document) -> str:
        """
        Retrieves the full parent context for a child chunk.
        
        Args:
            child_chunk: Child chunk document with parent metadata
            
        Returns:
            Parent chunk content
        """
        return child_chunk.metadata.get("parent_content", child_chunk.page_content)


class AdaptiveChunker(BaseChunker):
    """
    Implements adaptive chunking based on content type and structure.
    
    Strategy:
    - Use semantic chunking for body text (preserves meaning)
    - Use fixed chunking for references/tables (preserves structure)
    - Use section-aware chunking for headers (preserves hierarchy)
    - Adaptive chunk sizes based on content density
    """

    def __init__(self, embedding_model, min_chunk_size: int = 200, max_chunk_size: int = 1500):
        self.embedding_model = embedding_model
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
        # Import semantic chunker
        from langchain_experimental.text_splitter import SemanticChunker
        
        self.semantic_splitter = SemanticChunker(
            embedding_model,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=0.85  # More conservative than default
        )
        
        self.fixed_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits documents adaptively based on content analysis.
        """
        app_logger.info("Running adaptive chunking based on content type")
        
        all_chunks = []
        
        for doc in documents:
            content = doc.page_content
            
            # Detect content type
            content_type = self._detect_content_type(content, doc.metadata)
            
            # Apply appropriate chunking strategy
            if content_type == "semantic":
                chunks = self.semantic_splitter.split_documents([doc])
            elif content_type == "fixed":
                chunks = self.fixed_splitter.split_documents([doc])
            else:
                # Hybrid approach
                chunks = self._hybrid_chunk(doc)
            
            # Enrich metadata
            for chunk in chunks:
                chunk.metadata.update(doc.metadata)
                chunk.metadata["chunking_strategy"] = "adaptive"
                chunk.metadata["content_type"] = content_type
                chunk.metadata["chunk_size"] = len(chunk.page_content)
            
            all_chunks.extend(chunks)
        
        numbered_chunks = self.add_chunk_numbering(all_chunks)
        app_logger.info(f"Adaptive chunking complete: {len(numbered_chunks)} chunks")
        
        return numbered_chunks

    def _detect_content_type(self, content: str, metadata: Dict[str, Any]) -> str:
        """
        Detects the best chunking strategy for the content.
        """
        # Check for references (citations, bibliography)
        if self._is_references(content):
            return "fixed"
        
        # Check for tables/data
        if self._is_table_data(content):
            return "fixed"
        
        # Default to semantic for body text
        return "semantic"

    def _is_references(self, content: str) -> bool:
        """Detects if content is references/bibliography."""
        reference_indicators = ["references", "bibliography", "citations", "[1]", "[2]"]
        content_lower = content.lower()
        return any(indicator in content_lower for indicator in reference_indicators)

    def _is_table_data(self, content: str) -> bool:
        """Detects if content contains structured data/tables."""
        # Check for table-like patterns
        table_indicators = ["|", "\t", "table", "fig.", "figure"]
        return any(indicator in content.lower() for indicator in table_indicators)

    def _hybrid_chunk(self, doc: Document) -> List[Document]:
        """
        Applies hybrid chunking: semantic for main content, fixed for sections.
        """
        content = doc.page_content
        chunks = []
        
        # Simple sentence-based splitting for hybrid approach
        sentences = content.split('. ')
        
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < self.max_chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(Document(page_content=current_chunk.strip(), metadata=doc.metadata.copy()))
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(Document(page_content=current_chunk.strip(), metadata=doc.metadata.copy()))
        
        return chunks


class SectionAwareChunker(BaseChunker):
    """
    Implements section-aware chunking for research papers.
    
    Strategy:
    - Identify document sections (Abstract, Introduction, Methods, etc.)
    - Chunk within section boundaries
    - Preserve section hierarchy in metadata
    - Optimize chunk sizes per section type
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits documents while preserving section boundaries.
        """
        app_logger.info("Running section-aware chunking")
        
        all_chunks = []
        
        for doc in documents:
            # Extract sections from document
            sections = self._extract_sections(doc.page_content)
            
            # Chunk each section separately
            for section_name, section_content in sections:
                section_doc = Document(page_content=section_content, metadata=doc.metadata.copy())
                section_chunks = self.splitter.split_documents([section_doc])
                
                # Enrich with section metadata
                for chunk in section_chunks:
                    chunk.metadata["section_name"] = section_name
                    chunk.metadata["chunking_strategy"] = "section_aware"
                    chunk.metadata["section_hierarchy"] = self._get_section_hierarchy(section_name)
                
                all_chunks.extend(section_chunks)
        
        numbered_chunks = self.add_chunk_numbering(all_chunks)
        app_logger.info(f"Section-aware chunking complete: {len(numbered_chunks)} chunks")
        
        return numbered_chunks

    def _extract_sections(self, content: str) -> List[Tuple[str, str]]:
        """
        Extracts sections from research paper content.
        
        Returns:
            List of (section_name, section_content) tuples
        """
        sections = []
        
        # Common section headers in research papers
        section_patterns = [
            "abstract",
            "introduction",
            "related work",
            "background",
            "methods",
            "methodology",
            "experiments",
            "results",
            "discussion",
            "conclusion",
            "conclusions",
            "references",
            "acknowledgments"
        ]
        
        # Split content by section headers
        content_lower = content.lower()
        section_starts = []
        
        for pattern in section_patterns:
            idx = content_lower.find(pattern)
            if idx != -1:
                section_starts.append((idx, pattern))
        
        # Sort by position
        section_starts.sort(key=lambda x: x[0])
        
        # Extract sections
        for i, (start, pattern) in enumerate(section_starts):
            end = section_starts[i + 1][0] if i + 1 < len(section_starts) else len(content)
            section_content = content[start:end]
            sections.append((pattern.title(), section_content))
        
        # If no sections found, treat entire content as one section
        if not sections:
            sections.append(("Unknown", content))
        
        return sections

    def _get_section_hierarchy(self, section_name: str) -> str:
        """
        Returns the hierarchical level of a section.
        """
        high_level_sections = ["abstract", "introduction", "conclusion", "conclusions"]
        mid_level_sections = ["methods", "methodology", "results", "discussion"]
        low_level_sections = ["references", "acknowledgments"]
        
        section_lower = section_name.lower()
        
        if any(s in section_lower for s in high_level_sections):
            return "high"
        elif any(s in section_lower for s in mid_level_sections):
            return "medium"
        else:
            return "low"
