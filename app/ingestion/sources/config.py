from dataclasses import dataclass
from typing import Optional

@dataclass
class NewsSourceConfig:
    name: str
    base_url: str
    rss_url: Optional[str] = None
    scrape_url: Optional[str] = None
    category: str = "General"
    country: str = "Global"
    method: str = "requests"  # 'rss', 'requests', 'playwright'
    active: bool = True

# Initial Registry of Financial News Sources
SOURCE_REGISTRY = [
    NewsSourceConfig(
        name="Reuters",
        base_url="https://www.reuters.com",
        rss_url="https://www.reuters.com/arc.net/rss/topics/business",
        method="rss"
    ),
    NewsSourceConfig(
        name="CNBC",
        base_url="https://www.cnbc.com",
        rss_url="https://www.cnbc.com/id/100003114/device/rss/rss.html",
        method="rss"
    ),
    NewsSourceConfig(
        name="Yahoo Finance",
        base_url="https://finance.yahoo.com",
        scrape_url="https://finance.yahoo.com/news/",
        method="requests"
    ),
    NewsSourceConfig(
        name="MarketWatch",
        base_url="https://www.marketwatch.com",
        scrape_url="https://www.marketwatch.com/news",
        method="requests"
    ),
    NewsSourceConfig(
        name="Economic Times",
        base_url="https://economictimes.indiatimes.com",
        scrape_url="https://economictimes.indiatimes.com/markets",
        method="playwright"
    ),
]
