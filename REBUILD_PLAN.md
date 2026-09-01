# Newsage V2 Rebuild Plan

## 1. Current State Audit

### Architecture
- **Pattern**: Monolithic Streamlit Application.
- **Data Flow**: Synchronous/Blocking $\rightarrow$ User Input $\rightarrow$ Scrape $\rightarrow$ Parse $\rightarrow$ Analyze $\rightarrow$ Visualize.
- **State**: Volatile (`st.session_state`), no persistent storage.
- **Components**:
    - `main.py`: UI, sentiment analysis, stock prediction, and orchestration.
    - `scrape.py`: Selenium-based scraping and BeautifulSoup cleaning.
    - `parse.py`: LangChain + Ollama for structured data extraction.

### Existing Functionality
- Generic web scraping of any URL.
- AI-powered information extraction from HTML.
- General-purpose sentiment analysis (TextBlob).
- Ticker extraction from headlines.
- Stock data retrieval (yfinance).
- Experimental forecasting (Polynomial Regression, LSTM).

### Critical Problems
- **No Persistence**: All data is lost on page refresh.
- **UI Blocking**: Heavy I/O (scraping, LLMs) runs on the main thread, freezing the dashboard.
- **Inefficient Ingestion**: Over-reliance on Selenium for all sources.
- **Naive ML**: Lack of proper time-series backtesting and validation.
- **Poor Sentiment Accuracy**: TextBlob is not domain-aware for financial news.
- **Deployment Blockers**: Local Ollama dependency prevents cloud deployment.

---

## 2. Proposed Architecture

We will move from a **Monolithic UI** to a **Service-Oriented Data Platform**.

### High-Level Flow
`Data Sources` $\rightarrow$ `Ingestion Pipeline` $\rightarrow$ `PostgreSQL (pgvector)` $\rightarrow$ `AI/ML Services` $\rightarrow$ `Streamlit Dashboard`.

### Modular Breakdown
1.  **Ingestion Layer**: Configurable Source Registry $\rightarrow$ Hybrid Scraper (RSS $\rightarrow$ Requests $\rightarrow$ Playwright).
2.  **Persistence Layer**: PostgreSQL + pgvector for articles, sentiment, entities, and embeddings.
3.  **AI/NLP Layer**: FinBERT (Sentiment) $\rightarrow$ LLM (Entity/Topic Extraction) $\rightarrow$ Sentence Transformers (Embeddings).
4.  **Market Layer**: yfinance abstraction for OHLCV data + caching.
5.  **ML Layer**: Chronological backtesting pipeline $\rightarrow$ XGBoost/RandomForest $\rightarrow$ Evaluation metrics.
6.  **Presentation Layer**: Multi-page Streamlit Dashboard reading from precomputed DB views.

---

## 3. Implementation Phases

### Phase 1: Foundation & Audit (Current)
- [x] Complete codebase audit.
- [x] Define target architecture and rebuild plan.

### Phase 2: Database & Configuration
- Setup PostgreSQL/Supabase.
- Define schema for `news_sources`, `articles`, `entities`, `stock_prices`, etc.
- Implement `.env` configuration system.
- Create database connection and repository patterns.

### Phase 3: News Ingestion Pipeline
- Implement `NewsSourceConfig` registry.
- Build the hybrid scraping engine (RSS $\rightarrow$ Request $\rightarrow$ Playwright).
- Implement deduplication using content hashing.
- Build the idempotent pipeline: `Fetch` $\rightarrow$ `Parse` $\rightarrow$ `Clean` $\rightarrow$ `Store`.

### Phase 4: AI/NLP Pipeline
- Integrate FinBERT for financial sentiment analysis.
- Build entity extraction service (Tickers, Companies, Topics).
- Implement article summarization and embedding generation.
- Store vectors in pgvector.

### Phase 5: Market Data Pipeline
- Build `MarketDataService` (yfinance).
- Implement incremental updates and caching for stock prices.
- Link stocks to extracted entities.

### Phase 6: Analytics & Impact Scoring
- Implement the "News Impact Score" algorithm.
- Build aggregators for sentiment trends and news volume.
- Create database views for dashboard KPIs.

### Phase 7: ML & Backtesting
- Implement chronological time-series splitting.
- Build a suite of models (Baseline $\rightarrow$ XGBoost $\rightarrow$ LSTM).
- Develop a backtesting framework comparing Price-only vs. Price+News.

### Phase 8: Dashboard Rebuild
- Implement multi-page layout:
    - Market Overview
    - News Explorer
    - Stock Intelligence
    - Predictions
    - AI Market Brief
    - System Monitors (Pipeline/Source).

### Phase 9: Semantic Search & AI Assistant
- Implement vector-based semantic search.
- Build a RAG-based AI Assistant querying the Newsage DB.

### Phase 10: Automation & Deployment
- Setup GitHub Actions for scheduled ingestion.
- Deploy to Streamlit Community Cloud.
- Finalize documentation and README.

---

## 4. Migration Strategy

- **Preserve**: The cleaning logic in `scrape.py` and the prompt templates in `parse.py`.
- **Replace**: `main.py`'s logic will be decomposed into services.
- **Incremental Transition**: The old `main.py` will be kept as a reference until the new Dashboard is fully operational.
