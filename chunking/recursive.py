from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chunking.base import BaseChunker
from config.settings import settings
from custom_logging.logger import app_logger

class RecursiveChunker(BaseChunker):
    """Splits documents recursively based on structure characters (newlines, spaces)."""

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        cfg_size = settings.get("chunking", "recursive", "chunk_size", default=1000)
        cfg_overlap = settings.get("chunking", "recursive", "chunk_overlap", default=200)
        
        self.chunk_size = chunk_size or cfg_size
        self.chunk_overlap = chunk_overlap or cfg_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Splits loaded pages into recursive character chunks."""
        app_logger.info(f"Running recursive chunking (size={self.chunk_size}, overlap={self.chunk_overlap})")
        split_docs = self.splitter.split_documents(documents)
        return self.add_chunk_numbering(split_docs)
