from sqlalchemy import Column, Integer, String, Text, DateTime, Float, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from app.database.connection import Base

class NewsSource(Base):
    __tablename__ = "news_sources"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    base_url = Column(String)
    rss_url = Column(String)
    scrape_url = Column(String)
    category = Column(String)
    country = Column(String)
    scraping_method = Column(String) # 'rss', 'requests', 'playwright'
    active = Column(Boolean, default=True)
    last_run_at = Column(DateTime(timezone=True))
    last_success_at = Column(DateTime(timezone=True))
    last_status = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    articles = relationship("Article", back_populates="source")

class Article(Base):
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True, index=True)
    source_id = Column(Integer, ForeignKey("news_sources.id"))
    url = Column(String, unique=True, index=True, nullable=False)
    title = Column(String, nullable=False)
    content = Column(Text)
    author = Column(String)
    published_at = Column(DateTime(timezone=True))
    scraped_at = Column(DateTime(timezone=True), server_default=func.now())
    language = Column(String, default="en")
    content_hash = Column(String, index=True)
    summary = Column(Text)
    embedding = Column(Vector(384)) # Dimension for all-MiniLM-L6-v2
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    source = relationship("NewsSource", back_populates="articles")
    sentiments = relationship("ArticleSentiment", back_populates="article", cascade="all, delete-orphan")
    topics = relationship("ArticleTopic", back_populates="article", cascade="all, delete-orphan")
    entities = relationship("ArticleEntity", back_populates="article", cascade="all, delete-orphan")

class ArticleSentiment(Base):
    __tablename__ = "article_sentiments"

    id = Column(Integer, primary_key=True, index=True)
    article_id = Column(Integer, ForeignKey("articles.id"))
    model = Column(String, nullable=False) # e.g., 'FinBERT', 'TextBlob'
    label = Column(String, nullable=False) # 'Positive', 'Negative', 'Neutral'
    score = Column(Float)
    confidence = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    article = relationship("Article", back_populates="sentiments")

class ArticleTopic(Base):
    __tablename__ = "article_topics"

    id = Column(Integer, primary_key=True, index=True)
    article_id = Column(Integer, ForeignKey("articles.id"))
    topic = Column(String, nullable=False)
    confidence = Column(Float)

    article = relationship("Article", back_populates="topics")

class Entity(Base):
    __tablename__ = "entities"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    entity_type = Column(String) # 'Company', 'Person', 'Organization'
    ticker = Column(String, index=True)
    exchange = Column(String)

    articles = relationship("ArticleEntity", back_populates="entity")

class ArticleEntity(Base):
    __tablename__ = "article_entities"

    article_id = Column(Integer, ForeignKey("articles.id"), primary_key=True)
    entity_id = Column(Integer, ForeignKey("entities.id"), primary_key=True)
    relevance_score = Column(Float)

    article = relationship("Article", back_populates="entities")
    entity = relationship("Entity", back_populates="articles")
