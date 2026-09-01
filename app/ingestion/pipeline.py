import logging
import feedparser
import requests
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from sqlalchemy.orm import Session

from app.config import config
from app.database.connection import SessionLocal
from app.database.models.pipeline import PipelineRun
from app.database.repositories.news_repository import NewsRepository
from app.ingestion.sources.config import SOURCE_REGISTRY, NewsSourceConfig
from app.ingestion.scrapers.rss_scraper import RSSScraper
from app.ingestion.scrapers.requests_scraper import RequestsScraper
from app.ingestion.scrapers.playwright_scraper import PlaywrightScraper
from app.ingestion.parsers.html_parser import HTMLParser
from app.ingestion.cleaners.content_cleaner import ContentCleaner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsPipeline:
    """Orchestrates the news ingestion process with strict priority fallback."""

    def __init__(self, db: Session):
        self.db = db
        self.repo = NewsRepository(db)
        self.parser = HTMLParser()
        self.cleaner = ContentCleaner()
        self.rss_scraper = RSSScraper()
        self.req_scraper = RequestsScraper()
        self.plwp_scraper = PlaywrightScraper()

    def _validate_rss(self, rss_url: str) -> Tuple[bool, Optional[List[Any]]]:
        """
        Implements the strict RSS Rule.
        Validates HTTP response, content type, and presence of items.
        """
        try:
            # 1. Validate HTTP Response
            response = requests.get(rss_url, timeout=self.req_scraper.timeout)
            if response.status_code != 200:
                return False, None

            # 2. Validate Content Type (should be XML/RSS)
            content_type = response.headers.get('Content-Type', '').lower()
            if 'xml' not in content_type and 'rss' not in content_type:
                # Some feeds don't set content-type correctly, so we attempt to parse anyway
                pass

            # 3. Parse and validate content
            feed = feedparser.parse(response.content)

            # Ensure it has entries and the entries have required fields
            if not feed.entries:
                return False, None

            first_entry = feed.entries[0]
            if not all(hasattr(first_entry, attr) for attr in ['title', 'link']):
                return False, None

            return True, feed.entries

        except Exception as e:
            logger.debug(f"RSS validation failed for {rss_url}: {e}")
            return False, None

    def _fetch_candidates(self, config: NewsSourceConfig) -> List[Dict[str, Any]]:
        """
        Implements Ingestion Priority: RSS -> Static HTML -> Playwright.
        """
        # Priority 1: RSS/Atom
        if config.rss_url:
            is_valid, entries = self._validate_rss(config.rss_url)
            if is_valid:
                logger.info(f"Using RSS for {config.name}")
                return [{"title": e.title, "url": e.link} for e in entries]

        # Priority 2 & 3: HTML (Static or JS-Rendered)
        url = config.scrape_url or config.base_url
        if not url:
            return []

        # Decide between Requests and Playwright
        if config.method == "playwright":
            logger.info(f"Using Playwright for {config.name}")
            html = self.plwp_scraper.fetch(url)
        else:
            logger.info(f"Using Requests for {config.name}")
            html = self.req_scraper.fetch(url)

        if not html:
            return []

        return self.parser.parse_links(html, config.base_url)

    def run_source(self, config: NewsSourceConfig):
        """Run ingestion for a single news source."""
        source = self.repo.get_or_create_source(config)

        run = PipelineRun(
            source=source.name,
            start_time=datetime.now(),
            status="running"
        )
        self.db.add(run)
        self.db.commit()

        articles_found = 0
        articles_processed = 0
        duplicates = 0
        failures = 0

        try:
            # 1. Discover Candidates using priority logic
            candidates = self._fetch_candidates(config)
            articles_found = len(candidates)
            logger.info(f"Found {articles_found} candidates for {source.name}")

            if not candidates:
                run.status = "success" # Found nothing, but process worked
                self.db.commit()
                return

            # 2. Process each candidate
            for cand in candidates:
                try:
                    url = cand['url']
                    title = cand['title']

                    # Fetch full article content
                    # Use the same priority logic for the article page itself
                    # (Most articles are static HTML, but some may need Playwright)
                    if config.method == "playwright":
                        full_html = self.plwp_scraper.fetch(url)
                    else:
                        full_html = self.req_scraper.fetch(url)

                    if not full_html:
                        failures += 1
                        continue

                    # Clean content
                    cleaned_text = self.cleaner.clean_html(full_html)
                    if not cleaned_text:
                        failures += 1
                        continue

                    # Store (with deduplication)
                    article = self.repo.add_article(
                        source_id=source.id,
                        url=url,
                        title=title,
                        content=cleaned_text
                    )

                    if article:
                        articles_processed += 1
                    else:
                        duplicates += 1

                except Exception as e:
                    logger.error(f"Error processing article {url}: {e}")
                    self.db.rollback() # CRITICAL: Clear the failed transaction
                    failures += 1

            run.status = "success" if failures == 0 else "partial_success"

        except Exception as e:
            logger.error(f"Pipeline failed for {source.name}: {e}")
            run.status = "failed"
            run.error_details = str(e)

        finally:
            run.end_time = datetime.now()
            run.articles_found = articles_found
            run.articles_processed = articles_processed
            run.duplicates = duplicates
            run.failures = failures
            self.db.commit()

    def run_all(self):
        """Run ingestion for all active sources in the registry."""
        for config in SOURCE_REGISTRY:
            if config.active:
                self.run_source(config)

if __name__ == "__main__":
    db = SessionLocal()
    try:
        pipeline = NewsPipeline(db)
        pipeline.run_all()
    finally:
        db.close()
