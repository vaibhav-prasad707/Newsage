import requests
from app.ingestion.scrapers.base import BaseScraper

class RequestsScraper(BaseScraper):
    """Lightweight scraper for static HTML pages."""

    def fetch(self, url: str) -> str:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"RequestsScraper error fetching {url}: {e}")
            return ""
