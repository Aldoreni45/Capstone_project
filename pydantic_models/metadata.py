from pydantic import BaseModel, Field
from typing import Optional

class PaperMetadata(BaseModel):
    """Metadata schema for extracted and indexed research paper chunks."""
    paper_title: str = Field(..., description="The title of the research paper.")
    author: str = Field(default="Unknown", description="The author or authors of the paper.")
    page_number: int = Field(..., description="The page number where the chunk resides.")
    chunk_number: int = Field(..., description="The 0-indexed position of the chunk in the document.")
    chunk_id: str = Field(..., description="Unique generated chunk identifier (e.g. hash or composite key).")
    source: str = Field(..., description="The path or identifier of the source file.")
    upload_date: str = Field(..., description="The timestamp when the document was loaded.")
    embedding_model: str = Field(..., description="The model configuration used to generate embeddings (e.g., bge, openai).")
