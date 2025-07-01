import streamlit as st
from scrape import (
    scrape_website,
    extract_body_content,
    clean_body_content,
    split_dom_content
)
from parse import parse_with_ollama
import pandas as pd
from textblob import TextBlob
import plotly.express as px
from streamlit_lottie import st_lottie
import json
import base64
from streamlit.components.v1 import html

# Configure page
st.set_page_config(
    page_title="TechWorld",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load background animation FIRST
with open("Background.json", "r") as f:
    background_lottie = json.load(f)

# Render background animation BEFORE any content
st_lottie(
    background_lottie,
    speed=1,
    width="100%",
    height="100%",
    key="background"
)

# Now apply CSS that targets Streamlit's main container
st.markdown(
    """
    <style>
    
    /* Make entire app background transparent */
    .stApp {
        background: transparent !important;
    }
    
    /* Position background animation */
    [data-testid="stVerticalBlock"] > div:first-child {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        z-index: -1 !important;
        opacity: 0.4 !important;
        pointer-events: none !important;
    }
    
    /* Style the main content container */
    .main .block-container {
        background: rgba(255, 255, 255, 0.15) !important;
        backdrop-filter: blur(15px) !important;
        -webkit-backdrop-filter: blur(15px) !important;
        border-radius: 20px !important;
        padding: 2rem !important;
        margin: 2rem auto !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2) !important;
        position: relative !important;
        z-index: 10 !important;
        max-width: 1200px !important;
    }
    
    /* Title styling */
    h1 {
        color: white !important;
        text-align: center !important;
        font-size: 3.5rem !important;
        font-weight: bold !important;
        text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.7) !important;
        margin-bottom: 2rem !important;
        -webkit-background-clip: text !important;
        background-clip: text !important;
        background-size: 300% 300% !important;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Style all text elements */
    .stMarkdown, .stMarkdown p, .stMarkdown div {
        color: rgba(255, 255, 255, 0.95) !important;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.5) !important;
    }
    
    /* Subheaders */
    h2, h3 {
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.6) !important;
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 20px !important;
        color: white !important;
        backdrop-filter: blur(10px) !important;
        font-weight: 500 !important;
    }
    
    .stTextInput > div > div > input::placeholder,
    .stTextArea > div > div > textarea::placeholder {
        color: rgba(255, 255, 255, 2.0) !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 0.75rem 2rem !important;
        font-weight: bold !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6) !important;
        background: linear-gradient(45deg, #764ba2 0%, #667eea 100%) !important;
    }
    
    /* Dataframes */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 15px !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        overflow: hidden !important;
    }
    
    /* Messages */
    .stSuccess, .stError, .stInfo, .stWarning {
        background: rgba(255, 255, 255, 0.15) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        color: white !important;
    }
    
    /* Code blocks */
    .stCode {
        background: rgba(0, 0, 0, 0.4) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(8px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: white !important;
        backdrop-filter: blur(5px) !important;
    }
    
    /* Plotly charts background */
    .js-plotly-plot {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(5px) !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Your app content starts here
st.title("TechWorld")

url = st.text_input("Enter the URL of the website:")

# Load second animation
with open("jRkNnOVNsC.json", "r") as f:
    lottie_data = json.load(f)

st_lottie(lottie_data, speed=1, width=700, height=400, key="initial_animation")

if st.button("Scrape Site"):
    if url:
        st.info("Launching browser and scraping the site...")
        html = scrape_website(url)

        if html.startswith("Error:"):
            st.error(html)
        else:
            st.success(f"Successfully scraped: {url}")
            st.subheader("Scraped HTML Content:")
            st.code(html[:5000], language='html')

            body_content = extract_body_content(html)
            cleaned_content = clean_body_content(body_content)
            st.session_state.cleaned_content = cleaned_content
            st.session_state.dom_content = cleaned_content

            with st.expander("View Cleaned Content"):
                st.text_area("Cleaned Body Content", cleaned_content, height=300)
    else:
        st.error("Please enter a valid URL.")

if "dom_content" in st.session_state:
    parse_description = st.text_area("Describe what you want to parse.")
    if st.button("Parse Content"):
        if parse_description:
            st.write("Parsing the content")
            dom_chunks = split_dom_content(st.session_state.dom_content)
            result = parse_with_ollama(dom_chunks, parse_description)
            parsed_lines = result.strip().split('\n')
            headlines = []
            timestamps = []
            for line in parsed_lines:
                line = line.strip()
                if "ago" in line or "hour" in line or "day" in line:
                    timestamps.append(line)
                elif line:
                    headlines.append(line)
            while len(timestamps) < len(headlines):
                timestamps.append("N/A")

            df = pd.DataFrame({
                "Headline": headlines,
                "Time published": timestamps
            })
            st.session_state.df = df

if "df" in st.session_state:
    df = st.session_state.df
    st.subheader("AI News Headlines")
    st.dataframe(df, use_container_width=True)

    search_term = st.text_input("Search for a specific headline:")
    if search_term:
        search_results = df[df['Headline'].str.contains(search_term, case=False)]
        if not search_results.empty:
            st.subheader("Search Results")
            st.dataframe(search_results, use_container_width=True)
        else:
            st.warning("No headlines found matching your search term.")

    if not df.empty:
        st.subheader("Choose Analysis Mode")
        mode = st.radio("Select an option:", ["Analyze Sentiment", "Stock Prediction"])

        if mode == "Analyze Sentiment":
            st.subheader("Sentiment Analysis")
            sentiments = []
            for headline in df["Headline"]:
                polarity = TextBlob(headline).sentiment.polarity
                if polarity > 0:
                    sentiments.append("Positive")
                elif polarity < 0:
                    sentiments.append("Negative")
                else:
                    sentiments.append("Neutral")

            df["Sentiment"] = sentiments
            overall_counts = df["Sentiment"].value_counts().reset_index()
            overall_counts.columns = ["Sentiment", "Count"]
            st.write("**Overall Sentiment Distribution**")
            st.dataframe(overall_counts)
            fig = px.pie(overall_counts, names="Sentiment", values="Count", title="Headline Sentiment Distribution")
            st.plotly_chart(fig)

            user_headline = st.text_input("Enter a headline to analyze sentiment:")
            if user_headline:
                blob = TextBlob(user_headline)
                polarity = blob.sentiment.polarity
                if polarity > 0:
                    user_sentiment = "Positive"
                elif polarity < 0:
                    user_sentiment = "Negative"
                else:
                    user_sentiment = "Neutral"
                st.write(f"Sentiment for your headline: **{user_sentiment}**")

        elif mode == "Stock Prediction":
            st.subheader("AI-Powered Stock Symbol Extraction & Prediction")

            # Prompt the user
            if "df" in st.session_state and not st.session_state.df.empty:
                headlines_text = "\n".join(st.session_state.df["Headline"].tolist())

                extract_symbols_prompt = (
                    "From the following list of headlines, extract **only** the **relevant stock ticker symbols** "
                    "(like AAPL, TSLA, AMZN, etc.) that relate to publicly traded companies mentioned or implied. "
                    "Return them as a comma-separated list, with no extra explanation or text.\n\n"
                    f"{headlines_text}"
                )

                dom_chunks = split_dom_content(headlines_text)
                result = parse_with_ollama(dom_chunks, extract_symbols_prompt)
                raw_symbols = result.strip().upper()
                extracted_symbols = set([s.strip() for s in raw_symbols.split(",") if s.strip().isalnum() and len(s.strip()) <= 6])

                if extracted_symbols:
                    st.success(f"Extracted stock symbols: {', '.join(extracted_symbols)}")
                    import yfinance as yf
                    import datetime
                    import numpy as np

                    prediction_model = st.radio("Choose prediction model:", ["Polynomial Regression", "LSTM (Deep Learning)"])
                    days = st.slider("Predict how many days ahead?", min_value=1, max_value=30, value=7)

                    for symbol in extracted_symbols:
                        st.write(f"### Stock Prediction for {symbol}")
                        stock = yf.Ticker(symbol)
                        df_hist = stock.history(period="6mo")

                        if len(df_hist) < 60:
                            st.warning(f"Not enough data for {symbol}")
                            continue

                        df_hist = df_hist.reset_index()

                        if prediction_model == "Polynomial Regression":
                            from sklearn.linear_model import LinearRegression
                            from sklearn.preprocessing import PolynomialFeatures
                            from sklearn.pipeline import make_pipeline

                            df_hist["Day"] = np.arange(len(df_hist))
                            model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
                            model.fit(df_hist[["Day"]], df_hist["Close"])
                            future_days = np.arange(len(df_hist), len(df_hist) + days)
                            predicted_prices = model.predict(future_days.reshape(-1, 1))

                            pred_df = pd.DataFrame({
                                "Date": [df_hist["Date"].iloc[-1] + datetime.timedelta(days=i+1) for i in range(days)],
                                "Predicted Close": predicted_prices
                            })

                        elif prediction_model == "LSTM (Deep Learning)":
                            tf = __import__("tensorflow")
                            from sklearn.preprocessing import MinMaxScaler

                            close_data = df_hist["Close"].values.reshape(-1, 1)
                            scaler = MinMaxScaler()
                            scaled_data = scaler.fit_transform(close_data)

                            sequence_length = 60
                            X, y = [], []
                            for i in range(sequence_length, len(scaled_data)):
                                X.append(scaled_data[i - sequence_length:i])
                                y.append(scaled_data[i])

                            X, y = np.array(X), np.array(y)

                            model = tf.keras.Sequential([
                                tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
                                tf.keras.layers.LSTM(50),
                                tf.keras.layers.Dense(1)
                            ])
                            model.compile(optimizer='adam', loss='mean_squared_error')
                            model.fit(X, y, epochs=10, batch_size=16, verbose=0)

                            last_sequence = scaled_data[-sequence_length:]
                            predictions = []

                            for _ in range(days):
                                input_seq = last_sequence.reshape(1, sequence_length, 1)
                                next_price_scaled = model.predict(input_seq, verbose=0)
                                predictions.append(next_price_scaled[0][0])
                                last_sequence = np.append(last_sequence[1:], next_price_scaled, axis=0)

                            predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
                            pred_df = pd.DataFrame({
                                "Date": [df_hist["Date"].iloc[-1] + datetime.timedelta(days=i+1) for i in range(days)],
                                "Predicted Close": predicted_prices
                            })

                        st.line_chart(pred_df.set_index("Date"))

                else:
                    st.error("No valid stock symbols found in the headlines.")

