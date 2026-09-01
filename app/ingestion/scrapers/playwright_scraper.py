import asyncio
from playwright.async_api import async_playwright
from app.ingestion.scrapers.base import BaseScraper

class PlaywrightScraper(BaseScraper):
    """Heavyweight scraper for JS-rendered pages."""

    async def _fetch_async(self, url: str) -> str:
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                )
                page = await context.new_page()
                await page.goto(url, wait_until="networkidle", timeout=self.timeout * 1000)
                content = await page.content()
                await browser.close()
                return content
        except Exception as e:
            print(f"PlaywrightScraper error fetching {url}: {e}")
            return ""

    def fetch(self, url: str) -> str:
        """Synchronous wrapper for the async fetch method."""
        return asyncio.run(self._fetch_async(url))
