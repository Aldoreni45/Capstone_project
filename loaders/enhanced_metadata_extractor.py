import re
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from pypdf import PdfReader
from custom_logging.logger import app_logger
from custom_logging.error_handler import PDFParsingError
from config.settings import settings


class EnhancedMetadataExtractor:
    """
    Enhanced metadata extraction with section detection, document structure analysis,
    and comprehensive metadata enrichment for research papers.
    """

    def __init__(self):
        self.groq_api_key = settings.groq_api_key
        self.default_model = settings.get("llm", "groq", "default_model", default="llama-3.3-70b-versatile")

    def extract_from_pdf(self, file_path: str) -> Dict[str, Any]:
        """
        Extracts comprehensive metadata from PDF with section detection and structure analysis.
        """
        metadata = {
            "paper_title": "Unknown Title",
            "author": "Unknown Author",
            "source": file_path,
            "upload_date": datetime.now().isoformat(),
            "document_uuid": str(uuid.uuid4()),
            "embedding_model": "unknown",
            "embedding_version": "1.0",
            "language_code": "en",
            "section_names": [],
            "section_hierarchy": {},
            "total_pages": 0,
            "abstract": "",
            "keywords": [],
            "doi": "",
            "arxiv_id": ""
        }

        try:
            reader = PdfReader(file_path)
            metadata["total_pages"] = len(reader.pages)

            # Step 1: Extract basic PDF properties
            pdf_info = reader.metadata
            if pdf_info:
                title = pdf_info.get("/Title")
                author = pdf_info.get("/Author")
                
                if title and isinstance(title, str) and len(title.strip()) > 3:
                    metadata["paper_title"] = title.strip()
                if author and isinstance(author, str) and len(author.strip()) > 3:
                    metadata["author"] = author.strip()

            # Step 2: Extract text from first few pages for enhanced analysis
            first_pages_text = ""
            for i in range(min(3, len(reader.pages))):
                first_pages_text += reader.pages[i].extract_text() or ""

            # Step 3: Extract abstract
            metadata["abstract"] = self._extract_abstract(first_pages_text)

            # Step 4: Extract keywords
            metadata["keywords"] = self._extract_keywords(first_pages_text)

            # Step 5: Extract DOI and arXiv ID
            metadata["doi"] = self._extract_doi(first_pages_text)
            metadata["arxiv_id"] = self._extract_arxiv_id(first_pages_text)

            # Step 6: Detect language
            metadata["language_code"] = self._detect_language(first_pages_text)

            # Step 7: LLM-based metadata enhancement if available
            if first_pages_text and self.groq_api_key:
                try:
                    llm_metadata = self._extract_via_llm(first_pages_text[:2000])
                    if llm_metadata:
                        if metadata["paper_title"] == "Unknown Title" or metadata["paper_title"].lower().endswith(".pdf"):
                            metadata["paper_title"] = llm_metadata.get("title", metadata["paper_title"])
                        if metadata["author"] == "Unknown Author":
                            metadata["author"] = llm_metadata.get("author", metadata["author"])
                        metadata["keywords"].extend(llm_metadata.get("keywords", []))
                except Exception as e:
                    app_logger.warning(f"LLM metadata extraction failed: {str(e)}")

            # Step 8: Fallback to filename if title still unknown
            if metadata["paper_title"] == "Unknown Title" or metadata["paper_title"].lower().endswith(".pdf"):
                from pathlib import Path
                filename = re.sub(r'\.pdf$', '', re.sub(r'[\-_]', ' ', Path(file_path).name, flags=re.IGNORECASE))
                metadata["paper_title"] = filename.title().strip()

            # Step 9: Deduplicate keywords
            metadata["keywords"] = list(set(metadata["keywords"]))

        except Exception as e:
            app_logger.error(f"Error during enhanced metadata extraction for {file_path}: {str(e)}")
            raise PDFParsingError(f"Failed to parse enhanced metadata from PDF: {str(e)}", details=file_path)

        app_logger.info(f"Enhanced metadata extracted: {metadata['paper_title']}, {len(metadata['keywords'])} keywords")
        return metadata

    def extract_page_metadata(
        self,
        page_text: str,
        page_number: int,
        global_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extracts page-level metadata including section detection.
        """
        page_metadata = global_metadata.copy()
        page_metadata["page_number"] = page_number

        # Detect section on this page
        section_name = self._detect_section(page_text)
        page_metadata["section_name"] = section_name

        # Detect section hierarchy
        page_metadata["section_hierarchy"] = self._get_section_hierarchy(section_name)

        # Estimate token count (rough estimate: 1 word ≈ 1.3 tokens)
        word_count = len(page_text.split())
        page_metadata["estimated_tokens"] = int(word_count * 1.3)
        page_metadata["word_count"] = word_count

        # Detect if page contains references
        page_metadata["is_references"] = self._is_references_page(page_text)

        # Detect if page contains figures/tables
        page_metadata["has_figures"] = self._has_figures(page_text)
        page_metadata["has_tables"] = self._has_tables(page_text)

        return page_metadata

    def _extract_abstract(self, text: str) -> str:
        """Extracts abstract from text."""
        abstract_patterns = [
            r'abstract\s*:?\s*(.*?)(?=\n\s*(?:introduction|keywords|1\.|I\.))',
            r'ABSTRACT\s*:?\s*(.*?)(?=\n\s*(?:INTRODUCTION|KEYWORDS|1\.|I\.))'
        ]
        
        for pattern in abstract_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                abstract = match.group(1).strip()
                # Clean up abstract
                abstract = re.sub(r'\s+', ' ', abstract)
                return abstract[:500]  # Limit length
        
        return ""

    def _extract_keywords(self, text: str) -> List[str]:
        """Extracts keywords from text."""
        keywords = []
        
        # Pattern for keywords section
        keyword_patterns = [
            r'keywords?\s*:?\s*([^\n]+)',
            r'KEYWORDS?\s*:?\s*([^\n]+)'
        ]
        
        for pattern in keyword_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                keywords_str = match.group(1)
                # Split by common delimiters
                keywords.extend(re.split(r'[,;·•]', keywords_str))
        
        # Clean keywords
        keywords = [k.strip().lower() for k in keywords if len(k.strip()) > 2]
        
        # Also extract potential technical terms (capitalized words)
        technical_terms = re.findall(r'\b[A-Z][a-zA-Z]+\b', text)
        keywords.extend([t.lower() for t in technical_terms if len(t) > 3])
        
        return keywords[:20]  # Limit to top 20

    def _extract_doi(self, text: str) -> str:
        """Extracts DOI from text."""
        doi_pattern = r'doi\s*:?\s*(10\.\d{4,}/[^\s]+)'
        match = re.search(doi_pattern, text, re.IGNORECASE)
        return match.group(1) if match else ""

    def _extract_arxiv_id(self, text: str) -> str:
        """Extracts arXiv ID from text."""
        arxiv_patterns = [
            r'arxiv\s*:?\s*([a-z-\.]+/\d+)',
            r'arXiv\s*:?\s*([a-z-\.]+/\d+)'
        ]
        
        for pattern in arxiv_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return ""

    def _detect_language(self, text: str) -> str:
        """Detects language of text (simplified)."""
        # Check for common non-English indicators
        non_english_patterns = [
            r'[àáâãäå]',  # Nordic accents
            r'[èéêë]',    # French accents
            r'[ìíîï]',    # Italian accents
            r'[òóôõö]',   # More accents
            r'[ùúûü]',    # German umlauts
            r'[ñ]',       # Spanish
            r'[ç]',       # Portuguese/Catalan
            r'[αβγδεζηθικλμνξοπρστυφχψω]',  # Greek
            r'[一-龥]',    # Chinese
            r'[あ-んア-ン]',  # Japanese
            r'[가-힣]',    # Korean
        ]
        
        for pattern in non_english_patterns:
            if re.search(pattern, text):
                return "non-en"
        
        return "en"

    def _detect_section(self, page_text: str) -> str:
        """Detects the main section of a page."""
        section_patterns = {
            "abstract": r'\babstract\b',
            "introduction": r'\bintroduction\b',
            "related work": r'\brelated\s+work\b',
            "background": r'\bbackground\b',
            "methods": r'\bmethods?\b',
            "methodology": r'\bmethodology\b',
            "experiments": r'\bexperiments?\b',
            "experimental": r'\bexperimental\b',
            "results": r'\bresults?\b',
            "discussion": r'\bdiscussion\b',
            "conclusion": r'\bconclusions?\b',
            "references": r'\breferences?\b',
            "bibliography": r'\bbibliography\b',
            "acknowledgments": r'\backnowledgments?\b'
        }
        
        page_lower = page_text.lower()
        
        for section, pattern in section_patterns.items():
            if re.search(pattern, page_lower):
                return section
        
        return "unknown"

    def _get_section_hierarchy(self, section_name: str) -> str:
        """Returns hierarchical level of section."""
        high_level = ["abstract", "introduction", "conclusion", "conclusions"]
        mid_level = ["methods", "methodology", "results", "discussion", "related work", "background"]
        low_level = ["references", "bibliography", "acknowledgments"]
        
        section_lower = section_name.lower()
        
        if any(s in section_lower for s in high_level):
            return "high"
        elif any(s in section_lower for s in mid_level):
            return "medium"
        else:
            return "low"

    def _is_references_page(self, page_text: str) -> bool:
        """Detects if page contains references."""
        reference_indicators = ["references", "bibliography", "[1]", "[2]", "[3]"]
        page_lower = page_text.lower()
        return any(indicator in page_lower for indicator in reference_indicators)

    def _has_figures(self, page_text: str) -> bool:
        """Detects if page contains figures."""
        figure_patterns = [r'fig\.', r'figure', r'fig\s\d+', r'figure\s\d+']
        page_lower = page_text.lower()
        return any(re.search(pattern, page_lower) for pattern in figure_patterns)

    def _has_tables(self, page_text: str) -> bool:
        """Detects if page contains tables."""
        table_patterns = [r'table', r'tab\.', r'table\s\d+']
        page_lower = page_text.lower()
        return any(re.search(pattern, page_lower) for pattern in table_patterns)

    def _extract_via_llm(self, text_sample: str) -> Optional[Dict[str, Any]]:
        """Enhanced LLM-based metadata extraction."""
        if not self.groq_api_key:
            return None

        prompt = f"""
You are an expert metadata extraction system for research papers. Analyze the following text and extract:
1. The exact Title of the paper
2. The Author(s) of the paper
3. A list of 5-10 key topics/keywords

Respond ONLY with a JSON object in this format:
{{
  "title": "Extracted Paper Title",
  "author": "Author One, Author Two",
  "keywords": ["keyword1", "keyword2", "keyword3"]
}}

Do not include reasoning, markdown wrappers, or extra text.

Text:
---
{text_sample}
---
"""
        try:
            import httpx
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.default_model,
                "messages": [
                    {"role": "system", "content": "You are a precise JSON extractor. Output valid raw JSON only."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,
                "max_tokens": 200
            }

            with httpx.Client() as client:
                response = client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=15.0
                )
                if response.status_code == 200:
                    res_data = response.json()
                    content = res_data["choices"][0]["message"]["content"].strip()
                    
                    # Clean markdown
                    content = re.sub(r"^```json\s*", "", content, flags=re.IGNORECASE)
                    content = re.sub(r"\s*```$", "", content, flags=re.IGNORECASE)
                    
                    import json
                    parsed = json.loads(content)
                    return {
                        "title": parsed.get("title", "").strip(),
                        "author": parsed.get("author", "").strip(),
                        "keywords": parsed.get("keywords", [])
                    }
        except Exception as e:
            app_logger.warning(f"Enhanced LLM metadata extraction failed: {str(e)}")
        
        return None
