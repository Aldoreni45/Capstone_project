from pathlib import Path
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from loaders.metadata_extractor import MetadataExtractor
from custom_logging.logger import app_logger
from custom_logging.error_handler import PDFParsingError

class ResearchPaperLoader:
    """Loads PDF research papers and enriches pages with extracted metadata."""

    def __init__(self):
        self.extractor = MetadataExtractor()

    def load_paper(self, file_path: str) -> List[Document]:
        """
        Loads a PDF file from a path, extracts paper metadata,
        and assigns normalized metadata to each page.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        app_logger.info(f"Loading research paper: {path.name}")
        
        try:
            # Extract common metadata (Title, Author)
            global_metadata = self.extractor.extract_from_pdf(str(path))
            
            # Load PDF pages using LangChain's PyPDFLoader
            loader = PyPDFLoader(str(path))
            pages = loader.load()
            
            enriched_docs = []
            for i, page in enumerate(pages):
                # We use 1-based page numbers
                page_num = i + 1
                
                # Combine standard details
                enriched_metadata = {
                    "paper_title": global_metadata["paper_title"],
                    "author": global_metadata["author"],
                    "page_number": page_num,
                    "source": path.name,
                    "upload_date": global_metadata["upload_date"]
                }
                
                # Create a new document with updated metadata
                doc = Document(
                    page_content=page.page_content,
                    metadata=enriched_metadata
                )
                enriched_docs.append(doc)
                
            app_logger.info(f"Successfully loaded {len(enriched_docs)} pages from {path.name}")
            return enriched_docs
            
        except Exception as e:
            app_logger.error(f"Failed to load paper {path.name}: {str(e)}")
            raise PDFParsingError(f"Failed to parse and load PDF: {str(e)}", details=file_path)
