import logging
from sqlalchemy import func
from sqlalchemy.orm import Session
from app.database.connection import SessionLocal
from app.database.models.news import Article

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cleanup_duplicates():
    """
    Identifies and removes duplicate articles based on content_hash and URL.
    Keeps the first occurrence (oldest created_at).
    """
    db = SessionLocal()
    try:
        logger.info("Starting duplicate cleanup...")

        # 1. Find duplicates by content_hash
        # We group by content_hash and find those with count > 1
        duplicates_query = (
            db.query(Article.content_hash, func.count(Article.id))
            .group_by(Article.content_hash)
            .having(func.count(Article.id) > 1)
            .all()
        )

        total_deleted = 0
        for content_hash, count in duplicates_query:
            # Get all articles with this hash, ordered by created_at
            articles = db.query(Article).filter(Article.content_hash == content_hash).order_by(Article.created_at.asc()).all()

            # Keep the first one, delete the rest
            for dup in articles[1:]:
                db.delete(dup)
                total_deleted += 1

        db.commit()
        logger.info(f"Cleanup complete. Removed {total_deleted} duplicate articles based on content hash.")

    except Exception as e:
        db.rollback()
        logger.error(f"Error during cleanup: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    cleanup_duplicates()
