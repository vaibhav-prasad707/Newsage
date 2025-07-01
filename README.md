# Newsage

**Your AI-powered platform to parse news, extract sentiment, and predict stock movements based on real-time headlines.**

---
![Screenshot 2025-07-01 at 13 43 40](https://github.com/user-attachments/assets/d7a71059-4ffb-45d2-a1bf-bb0d97790554)


##  Project Overview

**Newsage** leverages **AI models (Ollama)** and **web scraping techniques (Selenium)** to extract real-time headlines from any news website, analyze sentiment, and perform short-term stock prediction. Designed with an elegant **Streamlit interface**, it empowers users to track market moods and make informed decisions using cutting-edge AI tools.

---

##  Features

###  Web Scraping & Content Parsing
- Scrape any public news site using **headless Selenium**
- Clean and extract meaningful content from the HTML body
- Dynamically chunk long content for efficient parsing

###  AI-Powered Parsing (via Ollama)
- Parse scraped content using **Ollama** (LLaMA3 or compatible model)
- Extract key information like **headlines**, **timestamps**, and **stock symbols**
- Prompt-based: describe what you want to extract!

###  Sentiment Analysis
- Analyze extracted headlines using **TextBlob**
- Generate sentiment labels: Positive, Negative, Neutral
- Visualize distribution using **Plotly Pie Charts**
- Test your own headlines for sentiment on the fly

###  Stock Symbol Detection & Price Prediction
- Extract stock ticker symbols directly from news headlines (via AI)
- Choose between:
  -  **Polynomial Regression** for quick trend fitting
  -  **LSTM (Deep Learning)** for sequential forecasting (optional)
- Predict user-specified number of days ahead
- Visualize predicted stock prices using **Streamlit charts**


---

##  Screenshots

| Sentiment Analysis | Stock Prediction |
|--------------------|------------------|
| ![Screenshot 2025-07-01 at 13 50 00](https://github.com/user-attachments/assets/d0ac326e-38f3-4615-808b-e6e875306c48)
 | ![Screenshot 2025-07-01 at 13 51 05](https://github.com/user-attachments/assets/5bce823b-e34e-480d-b72f-7095712568c9)
 |

---

##  Tech Stack

| Tool | Purpose |
|------|---------|
| `Streamlit` | UI Framework |
| `Selenium` | Web scraping engine |
| `BeautifulSoup` | HTML parsing |
| `Ollama` | AI parsing and extraction |
| `TextBlob` | Sentiment analysis |
| `yfinance` | Real-time stock data |
| `scikit-learn` | Regression models |
| `TensorFlow` *(Optional)* | LSTM forecasting |
| `Plotly` | Visualizations |
| `LangChain` | Prompt-based AI orchestration |

---

##  Folder Structure

