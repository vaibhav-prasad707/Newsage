import json
import logging
from typing import List, Dict, Any
from langchain_ollama import OllamaLLM
from app.config import config

logger = logging.getLogger(__name__)

class EntityExtractionService:
    """Service to extract companies, tickers, and topics using LLMs."""

    def __init__(self):
        self.llm = OllamaLLM(model=config.OLLAMA_MODEL, timeout=60)

    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extract financial entities and topics from text.
        Returns a structured JSON response.
        """
        prompt = (
            f"Extract financial entities from the following text. "
            f"Return ONLY a JSON object with the keys: 'companies', 'tickers', 'topics'.\n"
            f"Example format: {{\"companies\": [\"Nvidia\"], \"tickers\": [\"NVDA\"], \"topics\": [\"AI\"]}}\n\n"
            f"Text: {text[:4000]}"
        )

        try:
            response = self.llm.invoke(prompt)
            # Basic cleanup to ensure it's valid JSON
            response_cleaned = response.strip()
            if "```json" in response_cleaned:
                response_cleaned = response_cleaned.split("```json")[1].split("```")[0].strip()
            elif "```" in response_cleaned:
                response_cleaned = response_cleaned.split("```")[1].split("```")[0].strip()

            return json.loads(response_cleaned)
        except Exception as e:
            logger.error(f"Entity extraction error: {e}")
            return {"companies": [], "tickers": [], "topics": []}
