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
    method: str = "requests"  # Primary method: 'rss', 'requests', 'playwright'
    active: bool = True

# Comprehensive Registry of Financial News Sources
SOURCE_REGISTRY = [
    # --- Global / US Financial News ---
    NewsSourceConfig(
        name="Reuters",
        base_url="https://www.reuters.com",
        rss_url="https://www.reuters.com/arc.net/rss/topics/business",
        method="playwright", # Changed from cloudflare to playwright as Reuters is very strict
        category="Global Finance"
    ),
    NewsSourceConfig(
        name="CNBC",
        base_url="https://www.cnbc.com",
        rss_url="https://www.cnbc.com/id/100003114/device/rss/rss.html",
        method="rss",
        category="Global Finance"
    ),
    NewsSourceConfig(
        name="Bloomberg",
        base_url="https://www.bloomberg.com",
        scrape_url="https://www.bloomberg.com/markets",
        method="playwright",
        category="Global Finance"
    ),
    NewsSourceConfig(
        name="Financial Times",
        base_url="https://www.ft.com",
        scrape_url="https://www.ft.com/world",
        method="playwright", # Changed from requests to playwright
        category="Global Finance",
        country="United Kingdom"
    ),
    NewsSourceConfig(
        name="Wall Street Journal",
        base_url="https://www.wsj.com",
        scrape_url="https://www.wsj.com/news/markets",
        method="playwright", # Changed from requests to playwright
        category="Global Finance"
    ),
    NewsSourceConfig(
        name="Investing.com",
        base_url="https://www.investing.com",
        scrape_url="https://www.investing.com/news",
        method="playwright", # Changed from requests to playwright
        category="Global Markets"
    ),
    NewsSourceConfig(
        name="Seeking Alpha",
        base_url="https://seekingalpha.com",
        scrape_url="https://seekingalpha.com/market-news",
        method="requests",
        category="Investment Research"
    ),
    NewsSourceConfig(
        name="Barron's",
        base_url="https://www.barrons.com",
        scrape_url="https://www.barrons.com/market-news",
        method="playwright", # Changed from requests to playwright
        category="Investment Research"
    ),
    NewsSourceConfig(
        name="Forbes",
        base_url="https://www.forbes.com",
        rss_url="https://www.forbes.com/most-popular/rss",
        method="rss",
        category="Business Finance"
    ),
    NewsSourceConfig(
        name="Fortune",
        base_url="https://fortune.com",
        scrape_url="https://fortune.com/latest", # Corrected URL from /news to /latest
        method="requests",
        category="Business Finance"
    ),

    # --- Indian Financial News ---
    NewsSourceConfig(
        name="Yahoo Finance",
        base_url="https://finance.yahoo.com",
        scrape_url="https://finance.yahoo.com/news/",
        method="requests",
        category="Global Finance"
    ),
    NewsSourceConfig(
        name="MarketWatch",
        base_url="https://www.marketwatch.com",
        scrape_url="https://www.marketwatch.com/news",
        method="requests",
        category="Global Finance"
    ),
    NewsSourceConfig(
        name="Economic Times",
        base_url="https://economictimes.indiatimes.com",
        scrape_url="https://economictimes.indiatimes.com/markets",
        method="playwright",
        category="India Finance",
        country="India"
    ),
    NewsSourceConfig(
        name="Moneycontrol",
        base_url="https://www.moneycontrol.com",
        rss_url="https://www.moneycontrol.com/news/rss/",
        method="rss",
        category="India Finance",
        country="India"
    ),
    NewsSourceConfig(
        name="Business Standard",
        base_url="https://www.business-standard.com",
        rss_url="https://www.business-standard.com/rss-feeds/listing",
        method="rss",
        category="India Finance",
        country="India"
    ),
    NewsSourceConfig(
        name="Mint",
        base_url="https://www.livemint.com",
        scrape_url="https://www.livemint.com/market",
        method="requests",
        category="India Finance",
        country="India"
    ),
    NewsSourceConfig(
        name="BusinessLine",
        base_url="https://www.thehindubusinessline.com",
        rss_url="https://www.thehindubusinessline.com/feeder/default.rss",
        method="rss",
        category="India Finance",
        country="India"
    ),
    NewsSourceConfig(
        name="Financial Express",
        base_url="https://www.financialexpress.com",
        rss_url="https://www.financialexpress.com/syndication/",
        method="rss",
        category="India Finance",
        country="India"
    ),
    NewsSourceConfig(
        name="CNBC-TV18",
        base_url="https://www.cnbctv18.com",
        scrape_url="https://www.cnbctv18.com/market",
        method="requests",
        category="India Finance",
        country="India"
    ),
    NewsSourceConfig(
        name="NDTV Profit",
        base_url="https://www.ndtvprofit.com",
        scrape_url="https://www.ndtvprofit.com/news",
        method="requests",
        category="India Finance",
        country="India"
    ),
]
