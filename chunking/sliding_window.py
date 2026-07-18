from typing import List
from langchain_core.documents import Document
from chunking.base import BaseChunker

class SlidingWindowChunker(BaseChunker):
    """Splits documents into overlapping chunks using a character-level sliding window approach."""

    def __init__(self, window_size: int = 1000, step_size: int = 500):
        self.window_size = window_size
        self.step_size = step_size

    def split_documents(self, documents: List[Document]) -> List[Document]:
        chunks = []
        for doc in documents:
            text = doc.page_content
            text_len = len(text)
            start = 0
            while start < text_len:
                end = min(start + self.window_size, text_len)
                chunk_text = text[start:end]
                
                metadata = doc.metadata.copy()
                metadata["sliding_window_start"] = start
                metadata["sliding_window_end"] = end
                
                chunks.append(Document(page_content=chunk_text, metadata=metadata))
                
                if end == text_len:
                    break
                start += self.step_size
                
        return self.add_chunk_numbering(chunks)
