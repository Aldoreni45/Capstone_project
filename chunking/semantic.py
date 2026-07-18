from typing import List
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from chunking.base import BaseChunker
from config.settings import settings
from custom_logging.logger import app_logger

class SemanticTextChunker(BaseChunker):
    """Splits documents based on semantic similarity between sentences using embeddings."""

    def __init__(self, embedding_model, breakpoint_threshold_type: str = None, breakpoint_threshold_amount: float = None):
        self.embedding_model = embedding_model
        
        cfg_type = settings.get("chunking", "semantic", "breakpoint_threshold_type", default="percentile")
        cfg_amount = settings.get("chunking", "semantic", "breakpoint_threshold_amount", default=0.95)
        
        self.breakpoint_threshold_type = breakpoint_threshold_type or cfg_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount or cfg_amount
        
        self.splitter = SemanticChunker(
            self.embedding_model,
            breakpoint_threshold_type=self.breakpoint_threshold_type,
            breakpoint_threshold_amount=self.breakpoint_threshold_amount
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Splits pages into semantic chunks using embeddings."""
        app_logger.info(
            f"Running semantic chunking with model {self.embedding_model.__class__.__name__} "
            f"(threshold_type={self.breakpoint_threshold_type}, amount={self.breakpoint_threshold_amount})"
        )
        split_docs = self.splitter.split_documents(documents)
        return self.add_chunk_numbering(split_docs)
