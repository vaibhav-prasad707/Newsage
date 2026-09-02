from sentence_transformers import SentenceTransformer
from typing import List, Union
import logging

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service to generate semantic embeddings for articles."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Embedding model {model_name} loaded.")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.model = None

    def encode(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Convert text or a list of texts into vector embeddings.
        Batch processing is significantly faster.
        """
        if not self.model:
            return [] if isinstance(texts, list) else []

        # model.encode handles both single strings and lists of strings efficiently
        embeddings = self.model.encode(texts)

        # If input was a single string, return a single list
        if isinstance(texts, str):
            return embeddings.tolist()

        # If input was a list, return a list of lists
        return embeddings.tolist()
