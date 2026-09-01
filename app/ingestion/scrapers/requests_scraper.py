import requests
from app.ingestion.scrapers.base import BaseScraper

class RequestsScraper(BaseScraper):
    """Lightweight scraper for static HTML pages."""

    def fetch(self, url: str) -> str:
        try:
            # Enhanced headers to mimic a real browser more closely
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
            }
            response = requests.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()

            # Check if the content is actually text/html to avoid binary files
            content_type = response.headers.get('Content-Type', '').lower()
            if 'text/html' not in content_type:
                return ""

            return response.text
        except requests.RequestException as e:
            print(f"RequestsScraper error fetching {url}: {e}")
            return ""
