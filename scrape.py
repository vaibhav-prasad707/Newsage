from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from tqdm import tqdm
import time
from bs4 import BeautifulSoup

def scrape_website(url):
    print("Launching browser...")

    options = Options()

    options.add_argument("--headless=new")  # Use 'new' headless mode (for Chrome 109+)
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    try:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        driver.get(url)

        # Add tqdm progress simulation
        for _ in tqdm(range(5), desc="Scraping in progress", ncols=70):
            time.sleep(0.4)

        html = driver.page_source
        return html
    except Exception as e:
        return f"Error: {e}"
    finally:
        try:
            driver.quit()
        except:
            pass

def extract_body_content(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    body_content = soup.body
    if body_content:
        return str(body_content)
    return "No body content found."

def clean_body_content(body_content):
    soup = BeautifulSoup(body_content, 'html.parser')
    for script in soup(["script", "style"]):
        script.extract  # Remove scripts and styles
    cleaned_content = soup.get_text(separator='\n', strip=True)
    cleaned_content = '\n'.join(line for line in cleaned_content.splitlines() if line.strip())
    return cleaned_content

#split the content into batches
def split_dom_content(dom_content, max_length=6000):
    return [dom_content[i:i + max_length] for i in range(0, len(dom_content), max_length)]
