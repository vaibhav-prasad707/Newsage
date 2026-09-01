import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
# Find the project root (parent of the 'app' directory)
PROJECT_ROOT = Path(__file__).parent.parent
env_path = PROJECT_ROOT / '.env'
load_dotenv(dotenv_path=env_path)

class Config:
    """Application configuration."""

    # Basic Settings
    ENV = os.getenv("ENVIRONMENT", "development")

    # Database
    DATABASE_URL = os.getenv("DATABASE_URL")

    # AI/LLM
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Market Data
    MARKET_DATA_PROVIDER = os.getenv("MARKET_DATA_PROVIDER", "yfinance")

    # Pipeline
    NEWS_FETCH_TIMEOUT = int(os.getenv("NEWS_FETCH_TIMEOUT", "30"))
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

    @classmethod
    def validate(cls):
        """Ensure all critical configuration is present."""
        critical_vars = ["DATABASE_URL"]
        missing = [var for var in critical_vars if getattr(cls, var) is None]
        if missing:
            raise EnvironmentError(f"Missing critical environment variables: {', '.join(missing)}")

config = Config()
