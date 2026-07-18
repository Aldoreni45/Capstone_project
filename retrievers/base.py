from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic_models.responses import RetrievedChunk

class BasePaperRetriever(ABC):
    """Abstract interface for all document chunk retrievers."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        namespace: str = "default",
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        """Retrieves relevant paper chunks from the store matching query parameters."""
        pass
