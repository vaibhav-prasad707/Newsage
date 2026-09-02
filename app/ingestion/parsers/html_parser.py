from bs4 import BeautifulSoup
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class HTMLParser:
    """Service to extract article candidates from raw HTML."""

    def parse_links(self, html: str, base_url: str) -> List[Dict[str, Any]]:
        """
        Extract a list of potential article URLs and titles from the page.
        Uses multiple heuristics to find headlines across different site structures.
        """
        soup = BeautifulSoup(html, 'html.parser')
        candidates = []

        # Heuristic 1: Look for common headline tags (h1-h6) that contain links
        # or links that are inside headline tags.
        for header in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            a = header.find('a', href=True)
            if a:
                title = a.get_text(strip=True)
                url = a['href']
                if self._is_valid_candidate(title, url):
                    candidates.append({"title": title, "url": self._resolve_url(url, base_url)})
                    continue # Avoid double counting if we also find it in the generic search

        # Heuristic 2: Generic search for links with a reasonable title length
        for a in soup.find_all('a', href=True):
            title = a.get_text(strip=True)
            url = a['href']

            if self._is_valid_candidate(title, url):
                # Avoid duplicates from Heuristic 1
                if not any(c['url'] == self._resolve_url(url, base_url) for c in candidates):
                    candidates.append({"title": title, "url": self._resolve_url(url, base_url)})

        return candidates

    def _is_valid_candidate(self, title: str, url: str) -> bool:
        """Determine if a link is likely a news article."""
        # Minimum title length (lowered to 15 to be more inclusive)
        if len(title) < 15:
            return False

        # Filter out common navigation/meta links
        blacklist = [
            '/category/', '/tag/', '/author/', '/about/', '/contact/',
            '/privacy/', '/terms/', '/login/', '/register/', '/search/',
            '#', 'javascript:', 'mailto:', 'tel:'
        ]
        if any(x in url.lower() for x in blacklist):
            return False

        return True

    def _resolve_url(self, url: str, base_url: str) -> str:
        """Ensure URL is absolute."""
        if url.startswith('/'):
            return base_url.rstrip('/') + url
        if not url.startswith('http'):
            # Handle relative paths without leading slash
            return base_url.rstrip('/') + '/' + url
        return url
