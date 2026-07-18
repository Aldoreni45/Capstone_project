from typing import List, Dict, Any, Optional
import requests
from custom_logging.logger import app_logger
from config.settings import settings
from langsmith import traceable

class WebSearchResult:
    """Represents a web search result."""
    def __init__(
        self,
        title: str,
        url: str,
        snippet: str,
        source: str = "web"
    ):
        self.title = title
        self.url = url
        self.snippet = snippet
        self.source = source

class WebSearcher:
    """
    Web search fallback for when retrieval quality is BAD.
    
    Uses DuckDuckGo (free, no API key required) for web search.
    """
    
    def __init__(self, max_results: int = 5):
        self.max_results = max_results
        self.base_url = "https://api.duckduckgo.com/"
    
    @traceable(name="Web Search")
    def search(
        self,
        query: str,
        max_results: int = None
    ) -> List[WebSearchResult]:
        """
        Performs web search using DuckDuckGo.
        
        Args:
            query: Search query
            max_results: Maximum number of results (overrides default)
            
        Returns:
            List[WebSearchResult]: Search results
        """
        max_results = max_results or self.max_results
        
        try:
            # Use DuckDuckGo instant answer API
            params = {
                "q": query,
                "format": "json",
                "no_html": 1,
                "skip_disambig": 0
            }
            
            response = requests.get(
                "https://api.duckduckgo.com/",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            # Extract abstract if available
            if data.get("Abstract"):
                results.append(WebSearchResult(
                    title=data.get("Heading", query),
                    url=data.get("AbstractURL", ""),
                    snippet=data.get("Abstract", ""),
                    source="duckduckgo_abstract"
                ))
            
            # Extract related topics
            if data.get("RelatedTopics"):
                for topic in data.get("RelatedTopics", [])[:max_results]:
                    if isinstance(topic, dict) and topic.get("Text"):
                        results.append(WebSearchResult(
                            title=topic.get("FirstURL", "").split("/")[-1].replace("_", " "),
                            url=topic.get("FirstURL", ""),
                            snippet=topic.get("Text", ""),
                            source="duckduckgo_related"
                        ))
            
            # If no results from API, try HTML scraping as fallback
            if not results:
                results = self._html_search(query, max_results)
            
            app_logger.info(f"Web search returned {len(results)} results for query: '{query}'")
            return results[:max_results]
            
        except Exception as e:
            app_logger.error(f"Web search failed: {str(e)}")
            return []
    
    def _html_search(self, query: str, max_results: int) -> List[WebSearchResult]:
        """
        Fallback HTML-based search using DuckDuckGo HTML interface.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List[WebSearchResult]: Search results
        """
        try:
            params = {
                "q": query,
                "kl": "us-en"
            }
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            response = requests.get(
                "https://duckduckgo.com/html/",
                params=params,
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            
            # Parse HTML response (simplified parsing)
            import re
            results = []
            
            # Extract result snippets using regex (basic parsing)
            # Note: In production, use BeautifulSoup for robust parsing
            pattern = r'<a[^>]*class="result__a"[^>]*>(.*?)</a>.*?<a[^>]*class="result__url"[^>]*>(.*?)</a>.*?<a[^>]*class="result__snippet"[^>]*>(.*?)</a>'
            matches = re.findall(pattern, response.text, re.DOTALL)
            
            for title, url, snippet in matches[:max_results]:
                # Clean HTML tags
                clean_title = re.sub(r'<[^>]+>', '', title).strip()
                clean_url = re.sub(r'<[^>]+>', '', url).strip()
                clean_snippet = re.sub(r'<[^>]+>', '', snippet).strip()
                
                if clean_title and clean_snippet:
                    results.append(WebSearchResult(
                        title=clean_title,
                        url=clean_url,
                        snippet=clean_snippet,
                        source="duckduckgo_html"
                    ))
            
            return results
            
        except Exception as e:
            app_logger.warning(f"HTML search fallback failed: {str(e)}")
            return []
    
    def format_results_for_context(self, results: List[WebSearchResult]) -> str:
        """
        Formats web search results into context string for LLM.
        
        Args:
            results: List of web search results
            
        Returns:
            str: Formatted context string
        """
        if not results:
            return ""
        
        context_parts = []
        for idx, result in enumerate(results, 1):
            context_parts.append(
                f"Web Source {idx}: {result.title}\n"
                f"URL: {result.url}\n"
                f"Content: {result.snippet}\n"
            )
        
        return "\n".join(context_parts)
