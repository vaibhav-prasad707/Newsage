from bs4 import BeautifulSoup
from typing import List, Dict, Any
from app.ingestion.scrapers.base import BaseScraper

class HTMLParser:
    """Service to extract article candidates from raw HTML."""

    def parse_links(self, html: str, base_url: str) -> List[Dict[str, Any]]:
        """
        Extract a list of potential article URLs and titles from the page.
        This is a generic implementation; in a real system, this would be
        overridden per source.
        """
        soup = BeautifulSoup(html, 'html.parser')
        candidates = []

        # Look for <a> tags that look like articles (usually contain titles in the text)
        for a in soup.find_all('a', href=True):
            title = a.get_text(strip=True)
            url = a['href']

            # Basic heuristic: ignore very short titles, nav links, and fragments
            if len(title) < 20 or any(x in url.lower() for x in ['/category/', '/tag/', '/author/', '#']):
                continue

            # Ensure absolute URL
            if url.startswith('/'):
                url = base_url.rstrip('/') + url
            elif not url.startswith('http'):
                continue

            candidates.append({
                "title": title,
                "url": url
            })

        return candidates
