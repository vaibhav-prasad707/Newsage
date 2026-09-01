import feedparser
from app.ingestion.scrapers.base import BaseScraper

class RSSScraper(BaseScraper):
    """Ultra-lightweight scraper for RSS feeds."""

    def fetch(self, url: str) -> str:
        """
        For RSS, 'fetch' returns the parsed feed object
        rather than raw HTML, as it's already structured.
        """
        try:
            feed = feedparser.parse(url)
            return str(feed) # Simple representation, the pipeline will handle the feed object
        except Exception as e:
            print(f"RSSScraper error fetching {url}: {e}")
            return ""

    def get_feed_entries(self, url: str):
        """Directly return feed entries for the pipeline."""
        feed = feedparser.parse(url)
        return feed.entries
