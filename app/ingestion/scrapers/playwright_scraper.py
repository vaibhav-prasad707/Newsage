import asyncio
from playwright.async_api import async_playwright
from app.ingestion.scrapers.base import BaseScraper

class PlaywrightScraper(BaseScraper):
    """Heavyweight scraper for JS-rendered pages with stealth improvements."""

    async def _fetch_async(self, url: str) -> str:
        try:
            async with async_playwright() as p:
                # Launch browser with a few flags to reduce detectability
                browser = await p.chromium.launch(
                    headless=True,
                    args=[
                        "--disable-blink-features=AutomationControlled",
                        "--no-sandbox",
                        "--disable-setuid-sandbox"
                    ]
                )

                # Use a very realistic User-Agent and viewport
                context = await browser.new_context(
                    user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
                    viewport={'width': 1920, 'height': 1080}
                )

                page = await context.new_page()

                # Set extra headers to look more human
                await page.set_extra_http_headers({
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                    "Upgrade-Insecure-Requests": "1"
                })

                # Use a more flexible wait strategy. 'networkidle' is often too strict.
                # We wait for 'domcontentloaded' and then a small sleep.
                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=self.timeout * 1000)
                    await asyncio.sleep(2) # Allow some JS to execute
                except Exception as e:
                    logger.debug(f"Initial goto failed or timed out: {e}")

                content = await page.content()
                await browser.close()
                return content
        except Exception as e:
            print(f"PlaywrightScraper error fetching {url}: {e}")
            return ""

    def fetch(self, url: str) -> str:
        """Synchronous wrapper for the async fetch method."""
        return asyncio.run(self._fetch_async(url))
