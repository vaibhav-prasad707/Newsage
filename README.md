# Newsage

**Your AI-powered platform to parse news, extract sentiment, and predict stock movements based on real-time headlines.**

---

![Banner](https://your-banner-link-if-any.com)

## 📌 Project Overview

**Newsage** leverages **AI models (Ollama)** and **web scraping techniques (Selenium)** to extract real-time headlines from any news website, analyze sentiment, and perform short-term stock prediction. Designed with an elegant **Streamlit interface**, it empowers users to track market moods and make informed decisions using cutting-edge AI tools.

---

## 🎯 Features

### 🌐 Web Scraping & Content Parsing
- Scrape any public news site using **headless Selenium**
- Clean and extract meaningful content from the HTML body
- Dynamically chunk long content for efficient parsing

### 🧠 AI-Powered Parsing (via Ollama)
- Parse scraped content using **Ollama** (LLaMA3 or compatible model)
- Extract key information like **headlines**, **timestamps**, and **stock symbols**
- Prompt-based: describe what you want to extract!

### 📊 Sentiment Analysis
- Analyze extracted headlines using **TextBlob**
- Generate sentiment labels: Positive, Negative, Neutral
- Visualize distribution using **Plotly Pie Charts**
- Test your own headlines for sentiment on the fly

### 📈 Stock Symbol Detection & Price Prediction
- Extract stock ticker symbols directly from news headlines (via AI)
- Choose between:
  - 📉 **Polynomial Regression** for quick trend fitting
  - 🔮 **LSTM (Deep Learning)** for sequential forecasting (optional)
- Predict user-specified number of days ahead
- Visualize predicted stock prices using **Streamlit charts**

### 💅 Aesthetic UI
- Styled using advanced custom CSS & **Lottie animations**
- Glassmorphism interface for a premium user experience

---

## 🖼️ Screenshots

| Sentiment Analysis | Stock Prediction |
|--------------------|------------------|
| ![Sentiment](screenshots/sentiment.png) | ![Prediction](screenshots/prediction.png) |

---

## 🛠️ Tech Stack

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

## 🔄 Folder Structure

