from abc import ABC, abstractmethod
from typing import List, Dict, Any
import requests
from bs4 import BeautifulSoup

class BaseScraper(ABC):
    """Abstract base class for all scrapers."""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    @abstractmethod
    def fetch(self, url: str) -> str:
        """Fetch raw content from the given URL."""
        pass

    def get_soup(self, html: str) -> BeautifulSoup:
        """Helper to return a BeautifulSoup object."""
        return BeautifulSoup(html, 'html.parser')
