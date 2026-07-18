from pydantic import BaseModel, Field

class Citation(BaseModel):
    """Represents a citation pointing to a specific source passage in a research paper."""
    title: str = Field(..., description="The title of the research paper.")
    page: int = Field(..., description="The 1-based page number where the passage is located.")
    chunk_id: str = Field(..., description="The identifier of the text chunk.")
    source: str = Field(..., description="The filename or source URI of the document.")
    passage: str = Field(..., description="The precise textual passage being cited.")
