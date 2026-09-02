import json
import logging
from langchain_ollama import OllamaLLM
from app.config import config

logger = logging.getLogger(__name__)

class SummarizationService:
    """Service to generate concise article summaries using LLMs."""

    def __init__(self):
        self.llm = OllamaLLM(model=config.OLLAMA_MODEL, timeout=60)

    def summarize(self, text: str) -> str:
        """Generate a concise 2-3 sentence summary of the article."""
        prompt = (
            f"Summarize the following financial news article into a concise "
            f"2-3 sentence summary. Focus on the key event and the impact. "
            f"Do not use introductory phrases like 'This article is about'.\n\n"
            f"Text: {text[:6000]}"
        )
        try:
            return self.llm.invoke(prompt).strip()
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            return "Summary unavailable."
