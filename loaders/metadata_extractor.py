import re
from datetime import datetime
from typing import Dict, Any, Optional
from pypdf import PdfReader
from custom_logging.logger import app_logger
from custom_logging.error_handler import PDFParsingError, retry_on_exception
from config.settings import settings
import httpx

class MetadataExtractor:
    """Extracts paper metadata (Title, Author) using heuristics and LLM processing."""

    def __init__(self):
        self.groq_api_key = settings.groq_api_key
        self.default_model = settings.get("llm", "groq", "default_model", default="llama-3.3-70b-versatile")

    def extract_from_pdf(self, file_path: str) -> Dict[str, Any]:
        """
        Extracts metadata from a PDF file using a multi-layered fallback strategy:
        1. PDF Properties
        2. LLM-based parsing of the first page
        3. File name heuristics
        """
        metadata = {
            "paper_title": "Unknown Title",
            "author": "Unknown Author",
            "source": file_path,
            "upload_date": datetime.now().isoformat()
        }

        try:
            reader = PdfReader(file_path)
            
            # Step 1: Try PDF Properties
            pdf_info = reader.metadata
            if pdf_info:
                title = pdf_info.get("/Title")
                author = pdf_info.get("/Author")
                
                if title and isinstance(title, str) and len(title.strip()) > 3:
                    metadata["paper_title"] = title.strip()
                if author and isinstance(author, str) and len(author.strip()) > 3:
                    metadata["author"] = author.strip()

            # Clean metadata checks
            has_good_title = metadata["paper_title"] != "Unknown Title" and not metadata["paper_title"].lower().endswith(".pdf")
            has_good_author = metadata["author"] != "Unknown Author"

            if has_good_title and has_good_author:
                app_logger.info(f"Metadata extracted successfully from PDF properties: {metadata}")
                return metadata

            # Step 2: Extract text from first page and use LLM fallback if Groq is available
            first_page_text = ""
            if len(reader.pages) > 0:
                first_page_text = reader.pages[0].extract_text() or ""

            if first_page_text and self.groq_api_key:
                try:
                    llm_metadata = self._extract_via_llm(first_page_text[:1500])
                    if llm_metadata:
                        if not has_good_title and llm_metadata.get("title"):
                            metadata["paper_title"] = llm_metadata["title"]
                        if not has_good_author and llm_metadata.get("author"):
                            metadata["author"] = llm_metadata["author"]
                except Exception as e:
                    app_logger.warning(f"LLM metadata extraction failed: {str(e)}. Falling back to heuristics.")

            # Step 3: Heuristic clean up from filename if title is still missing/invalid
            if metadata["paper_title"] == "Unknown Title" or metadata["paper_title"].lower().endswith(".pdf"):
                filename = re.sub(r'\.pdf$', '', re.sub(r'[\-_]', ' ', Path(file_path).name), flags=re.IGNORECASE)
                metadata["paper_title"] = filename.title().strip()

        except Exception as e:
            app_logger.error(f"Error during metadata extraction for {file_path}: {str(e)}")
            raise PDFParsingError(f"Failed to parse metadata from PDF: {str(e)}", details=file_path)

        app_logger.info(f"Final extracted metadata: {metadata}")
        return metadata

    def _extract_via_llm(self, text_sample: str) -> Optional[Dict[str, str]]:
        """Invokes Groq LLM to pull Title and Author from paper cover page."""
        if not self.groq_api_key:
            return None

        prompt = f"""
You are an expert metadata extraction system. Analyze the following text snippet from the beginning of a research paper and extract:
1. The exact Title of the paper.
2. The Author(s) of the paper.

Respond ONLY with a JSON object in this format:
{{
  "title": "Extracted Paper Title",
  "author": "Author One, Author Two"
}}

Do not include any reasoning, markdown wrappers (like ```json), or extra text.

Text Snippet:
---
{text_sample}
---
"""
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
            "max_tokens": 150
        }

        # Make request with timeout using httpx
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
                
                # Strip markdown code blocks if the LLM outputted them anyway
                content_clean = re.sub(r"^```json\s*", "", content, flags=re.IGNORECASE)
                content_clean = re.sub(r"\s*```$", "", content_clean, flags=re.IGNORECASE)
                
                import json
                parsed = json.loads(content_clean)
                return {
                    "title": parsed.get("title", "").strip(),
                    "author": parsed.get("author", "").strip()
                }
            else:
                app_logger.warning(f"Groq metadata API error: {response.status_code} - {response.text}")
                
        return None
from pathlib import Path
