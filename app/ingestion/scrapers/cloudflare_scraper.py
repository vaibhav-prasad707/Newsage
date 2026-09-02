import cloudscraper
from app.ingestion.scrapers.base import BaseScraper

class CloudflareScraper(BaseScraper):
    """Scraper designed to bypass Cloudflare anti-bot protections."""

    def __init__(self, timeout: int = 30):
        super().__init__(timeout)
        # Create a scraper instance that can solve Cloudflare challenges
        self.scraper = cloudscraper.create_scraper(
            browser={
                'browser': 'chrome',
                'platform': 'windows',
                'desktop': True
            }
        )

    def fetch(self, url: str) -> str:
        try:
            response = self.scraper.get(url, timeout=self.timeout)
            response.raise_for_status()

            # Ensure we only process HTML content
            content_type = response.headers.get('Content-Type', '').lower()
            if 'text/html' not in content_type:
                return ""

            return response.text
        except Exception as e:
            print(f"CloudflareScraper error fetching {url}: {e}")
            return ""
