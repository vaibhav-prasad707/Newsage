import logging
from app.database.connection import engine, Base
from app.database.models.news import NewsSource, Article, ArticleSentiment, ArticleTopic, Entity, ArticleEntity
from app.database.models.market import Stock, StockPrice
from app.database.models.ml import Prediction, ModelRun
from app.database.models.pipeline import PipelineRun

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_database():
    """
    Creates all tables defined in the SQLAlchemy models.
    """
    try:
        logger.info("Connecting to database and creating tables...")

        # This creates all tables defined in the Base metadata
        Base.metadata.create_all(bind=engine)

        logger.info("Successfully initialized database schema!")
        logger.info("Tables created: news_sources, articles, article_sentiments, "
                    "article_topics, entities, article_entities, stocks, "
                    "stock_prices, predictions, model_runs, pipeline_runs")

    except Exception as e:
        logger.error(f"Error occurred while initializing database: {e}")
        raise e

if __name__ == "__main__":
    initialize_database()
