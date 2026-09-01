from bs4 import BeautifulSoup
import re

class ContentCleaner:
    """Service to remove noise from HTML content."""

    @staticmethod
    def clean_html(html_content: str) -> str:
        """Remove scripts, styles, and unnecessary tags, then return clean text."""
        if not html_content:
            return ""

        soup = BeautifulSoup(html_content, 'html.parser')

        # 1. Remove noise tags
        for element in soup(["script", "style", "meta", "link", "noscript", "header", "footer", "nav"]):
            element.decompose()

        # 2. Handle specific common noise (ads, popups)
        for div in soup.find_all("div", class_=re.compile(r"ad-|popup|banner|social-share", re.I)):
            div.decompose()

        # 3. Get text with spacing
        text = soup.get_text(separator='\n', strip=True)

        # 4. Clean whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        cleaned_text = '\n'.join(lines)

        return cleaned_text
