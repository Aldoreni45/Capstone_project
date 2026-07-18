from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np
from langchain_core.documents import Document

class BaseChunker(ABC):
    """Abstract base class for PDF chunkers."""

    @abstractmethod
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Splits a list of parent documents (pages) into smaller chunks."""
        pass

    def add_chunk_numbering(self, chunks: List[Document]) -> List[Document]:
        """Appends chunk tracking IDs and sequence numbers to the metadata."""
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_number"] = idx
            # Composite unique key: source_file + page + chunk_num
            source = chunk.metadata.get("source", "unknown")
            page = chunk.metadata.get("page_number", 0)
            chunk.metadata["chunk_id"] = f"{source}_p{page}_c{idx}"
        return chunks

def compare_chunkers(documents: List[Document], chunker_a: BaseChunker, chunker_b: BaseChunker, name_a: str, name_b: str) -> Dict[str, Dict[str, Any]]:
    """
    Splits documents using two chunking strategies and returns comparative statistics.
    """
    chunks_a = chunker_a.split_documents(documents)
    chunks_b = chunker_b.split_documents(documents)

    def calculate_stats(chunks: List[Document]) -> Dict[str, Any]:
        lengths = [len(c.page_content) for c in chunks]
        if not lengths:
            return {"count": 0, "avg": 0, "max": 0, "min": 0, "std": 0}
        return {
            "count": len(chunks),
            "avg": float(np.mean(lengths)),
            "max": int(np.max(lengths)),
            "min": int(np.min(lengths)),
            "std": float(np.std(lengths))
        }

    return {
        name_a: calculate_stats(chunks_a),
        name_b: calculate_stats(chunks_b)
    }
