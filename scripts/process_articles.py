import logging
import concurrent.futures
from datetime import datetime
from typing import List, Tuple
from sqlalchemy.orm import Session

from app.database.connection import SessionLocal
from app.database.models.news import Article, ArticleSentiment, ArticleTopic, Entity, ArticleEntity
from app.ai.sentiment.service import SentimentService
from app.ai.entities.service import EntityExtractionService
from app.ai.summarization.service import SummarizationService
from app.ai.embeddings.service import EmbeddingService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AIProcessingPipeline:
    """
    High-performance batch pipeline to enrich scraped articles.
    Uses a ThreadPoolExecutor for I/O bound AI calls (Ollama)
    and batch encoding for CPU bound tasks (Embeddings).
    """

    def __init__(self, db: Session, batch_size: int = 10):
        self.db = db
        self.batch_size = batch_size
        logger.info("Loading AI models... (this may take a moment)")

        self.sentiment_service = SentimentService()
        self.entity_service = EntityExtractionService()
        self.summary_service = SummarizationService()
        self.embedding_service = EmbeddingService()
        logger.info("All AI services loaded successfully.")

    def get_unprocessed_articles(self, limit: int = 1000):
        """Fetch articles that are missing summaries or embeddings."""
        return self.db.query(Article).filter(
            (Article.summary == None) | (Article.embedding == None)
        ).limit(limit).all()

    def _process_single_ai_suite(self, article: Article) -> Tuple[Article, dict]:
        """
        Runs the AI suite for a single article.
        This method is designed to be run in a thread pool.
        """
        try:
            # 1. Sentiment (Local CPU/GPU)
            sentiment_data = self.sentiment_service.analyze(article.content)

            # 2. Entities (Ollama API)
            entities_data = self.entity_service.extract(article.content)

            # 3. Summary (Ollama API)
            summary = self.summary_service.summarize(article.content)

            return article, {
                "sentiment": sentiment_data['primary'],
                "entities": entities_data,
                "summary": summary,
                "success": True
            }
        except Exception as e:
            logger.error(f"AI Suite error for article {article.id}: {e}")
            return article, {"success": False, "error": str(e)}

    def process_batch(self, articles: List[Article]):
        """Processes a batch of articles using parallel threads for AI calls."""

        # 1. Parallel AI Inference (Ollama calls are I/O bound)
        # We use a ThreadPoolExecutor to call Ollama in parallel
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_article = {executor.submit(self._process_single_ai_suite, art): art for art in articles}
            for future in concurrent.futures.as_completed(future_to_article):
                results.append(future.result())

        # 2. Batch Embedding Generation (CPU bound - much faster as a batch)
        texts = [art.content for art, res in results if res['success']]
        if texts:
            embeddings = self.embedding_service.encode(texts)
            # Map embeddings back to the results
            success_idx = 0
            for art, res in results:
                if res['success']:
                    res['embedding'] = embeddings[success_idx]
                    success_idx += 1

        # 3. Sequential Database Update (SQLAlchemy sessions are not thread-safe)
        for article, res in results:
            if not res['success']:
                continue

            try:
                # Update Sentiment
                sent_record = ArticleSentiment(
                    article_id=article.id,
                    model=res['sentiment']['model'],
                    label=res['sentiment']['label'],
                    score=res['sentiment']['score'],
                    confidence=res['sentiment']['score']
                )
                self.db.add(sent_record)

                # Update Entities & Topics
                for company in res['entities'].get('companies', []):
                    entity = self.db.query(Entity).filter(Entity.name == company).first()
                    if not entity:
                        entity = Entity(name=company, entity_type="Company")
                        self.db.add(entity)
                        self.db.flush() # Get ID without committing full transaction

                    art_ent = ArticleEntity(article_id=article.id, entity_id=entity.id, relevance_score=1.0)
                    self.db.add(art_ent)

                for topic in res['entities'].get('topics', []):
                    topic_record = ArticleTopic(article_id=article.id, topic=topic, confidence=1.0)
                    self.db.add(topic_record)

                # Update Article metadata
                article.summary = res['summary']
                article.embedding = res.get('embedding')

            except Exception as e:
                logger.error(f"DB Update error for article {article.id}: {e}")

        self.db.commit()

    def run(self, limit: int = 1000):
        """Main entry point to process articles in optimized batches."""
        articles = self.get_unprocessed_articles(limit)
        if not articles:
            logger.info("No unprocessed articles found.")
            return

        total = len(articles)
        logger.info(f"Found {total} articles to process. Using batch size: {self.batch_size}")

        for i in range(0, total, self.batch_size):
            batch = articles[i : i + self.batch_size]
            logger.info(f"Processing batch { (i//self.batch_size)+1 } / { (total+self.batch_size-1)//self.batch_size }")
            self.process_batch(batch)
            logger.info(f"Batch completed. Progress: {min(i + self.batch_size, total)}/{total}")

if __name__ == "__main__":
    db = SessionLocal()
    try:
        # On 8GB RAM, a batch size of 5-10 is usually the sweet spot.
        pipeline = AIProcessingPipeline(db, batch_size=10)
        pipeline.run()
    finally:
        db.close()
