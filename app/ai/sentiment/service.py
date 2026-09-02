import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from textblob import TextBlob
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class SentimentService:
    """Service for financial sentiment analysis using FinBERT and TextBlob."""

    def __init__(self):
        # Model: ProsusAI/finbert is a standard for financial sentiment
        self.model_name = "ProsusAI/finbert"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.nlp = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("FinBERT model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load FinBERT: {e}. Falling back to TextBlob only.")
            self.nlp = None

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment using FinBERT (primary) and TextBlob (baseline).
        """
        # 1. FinBERT Analysis
        finbert_result = {"label": "Neutral", "score": 0.0, "model": "FinBERT"}
        if self.nlp:
            try:
                # FinBERT has a max length of 512 tokens
                result = self.nlp(text[:512])[0]
                finbert_result = {
                    "label": result['label'],
                    "score": result['score'],
                    "model": "FinBERT"
                }
            except Exception as e:
                logger.warning(f"FinBERT error: {e}")

        # 2. TextBlob Analysis (Baseline)
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        textblob_label = "Neutral"
        if polarity > 0.1: textblob_label = "Positive"
        elif polarity < -0.1: textblob_label = "Negative"

        return {
            "primary": finbert_result,
            "baseline": {
                "label": textblob_label,
                "score": polarity,
                "model": "TextBlob"
            }
        }
