import pytest
from loaders.metadata_extractor import MetadataExtractor
from loaders.pdf_loader import ResearchPaperLoader

def test_metadata_extractor_heuristics():
    """Verifies metadata title cleaning and fallback behaviors."""
    extractor = MetadataExtractor()
    
    # Test fallback naming convention from file paths
    meta = extractor.extract_from_pdf("sample-research_paper_on_rag.pdf")
    assert meta["paper_title"] == "Sample Research Paper On Rag"
    assert meta["author"] == "Unknown Author"
    assert "source" in meta
    assert "upload_date" in meta

def test_paper_loader_error():
    """Asserts that loading a non-existent PDF file correctly throws FileNotFoundError."""
    loader = ResearchPaperLoader()
    with pytest.raises(FileNotFoundError):
        loader.load_paper("non_existent_file.pdf")
