import logging
from datetime import datetime
from typing import List
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
    """Orchestrates the news ingestion process."""

    def __init__(self, db: Session):
        self.db = db
        self.repo = NewsRepository(db)
        self.parser = HTMLParser()
        self.cleaner = ContentCleaner()

    def _get_scraper(self, method: str):
        """Factory to return the correct scraper based on method."""
        if method == "rss":
            return RSSScraper()
        elif method == "playwright":
            return PlaywrightScraper()
        else:
            return RequestsScraper()

    def run_source(self, config: NewsSourceConfig):
        """Run ingestion for a single news source."""
        source = self.repo.get_or_create_source(config)

        # Initialize Pipeline Run tracking
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
            scraper = self._get_scraper(config.method)
            logger.info(f"Starting ingestion for {source.name} using {config.method}...")

            # 1. Fetch discovery content
            url = config.rss_url if config.method == "rss" else config.scrape_url
            if not url:
                logger.warning(f"No fetch URL provided for {source.name}. Skipping.")
                run.status = "failed"
                run.error_details = "Missing URL"
                self.db.commit()
                return

            raw_content = scraper.fetch(url)
            if not raw_content:
                raise Exception(f"Failed to fetch content from {url}")

            # 2. Extract Candidates
            if config.method == "rss":
                # Special handling for RSS entries
                candidates = scraper.get_feed_entries(url)
            else:
                candidates = self.parser.parse_links(raw_content, config.base_url)

            articles_found = len(candidates)
            logger.info(f"Found {articles_found} candidates for {source.name}")

            # 3. Process each candidate
            for cand in candidates:
                try:
                    # Resolve fields based on RSS vs HTML
                    if config.method == "rss":
                        url = cand.link
                        title = cand.title
                    else:
                        url = cand['url']
                        title = cand['title']

                    # Fetch full content
                    full_html = scraper.fetch(url)
                    if not full_html:
                        failures += 1
                        continue

                    # Clean content
                    cleaned_text = self.cleaner.clean_html(full_html)

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
    # Standalone execution for testing
    db = SessionLocal()
    try:
        pipeline = NewsPipeline(db)
        pipeline.run_all()
    finally:
        db.close()
