import hashlib
from typing import List, Optional
from sqlalchemy.orm import Session
from app.database.models.news import NewsSource, Article, ArticleEntity, Entity
from app.ingestion.sources.config import NewsSourceConfig

class NewsRepository:
    """Repository for managing news sources and articles."""

    def __init__(self, db: Session):
        self.db = db

    def get_or_create_source(self, config: NewsSourceConfig) -> NewsSource:
        """Ensure the news source exists in the database."""
        source = self.db.query(NewsSource).filter(NewsSource.name == config.name).first()
        if not source:
            source = NewsSource(
                name=config.name,
                base_url=config.base_url,
                rss_url=config.rss_url,
                scrape_url=config.scrape_url,
                category=config.category,
                country=config.country,
                scraping_method=config.method,
                active=config.active
            )
            self.db.add(source)
            self.db.commit()
            self.db.refresh(source)
        return source

    def compute_content_hash(self, content: str) -> str:
        """Generate a unique hash for the article content to prevent duplicates."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def add_article(self, source_id: int, url: str, title: str, content: str,
                    author: Optional[str] = None, published_at=None) -> Optional[Article]:
        """Add a new article if it doesn't already exist (deduplication)."""

        # 1. Check by URL first
        existing = self.db.query(Article).filter(Article.url == url).first()
        if existing:
            return existing

        # 2. Check by Content Hash
        content_hash = self.compute_content_hash(content)
        existing_hash = self.db.query(Article).filter(Article.content_hash == content_hash).first()
        if existing_hash:
            return existing_hash

        # Create new article
        article = Article(
            source_id=source_id,
            url=url,
            title=title,
            content=content,
            author=author,
            published_at=published_at,
            content_hash=content_hash
        )
        self.db.add(article)
        self.db.commit()
        self.db.refresh(article)
        return article

    def get_recent_articles(self, limit: int = 100):
        """Get most recently scraped articles."""
        return self.db.query(Article).order_by(Article.scraped_at.desc()).limit(limit).all()
