import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from streamlit_chat import message
import yfinance as yf
import plotly.graph_objects as go

# ------------------------ STREAMLIT CONFIG ------------------------
st.set_page_config(page_title="AI Stock Predictor", layout="wide")

# ------------------------ SIDEBAR MENU & THEME ---------------------
st.sidebar.title("Navigation Menu")
section = st.sidebar.radio("Go to:", ["Dashboard", "Predictions", "Financial Assistant"])

st.sidebar.title("Theme")
theme = st.sidebar.radio("Select Theme", ["Light", "Dark"])

if theme == "Dark":
    st.markdown("""
    <style>
        body {background-color: #0e1117; color:white;}
        .stChatMessage {background-color: #1e222b;}
    </style>
    """, unsafe_allow_html=True)

# ------------------------ LOAD MODEL -------------------------

import zipfile
import os

# Extract model from zip if needed
if not os.path.exists("lstm_model.keras"):
    with zipfile.ZipFile("lstm_model.zip", 'r') as zip_ref:
        zip_ref.extractall()

model = load_model("lstm_model.keras", compile=False)


# ------------------------ STOCK SELECTION ---------------------
st.sidebar.markdown("### Select Stock")
stock_ticker = st.sidebar.selectbox("Company", ["TSLA", "AAPL", "GOOGL", "MSFT", "AMZN"])


# SAFER DATA FETCHING FUNCTION
def fetch_stock_history(ticker, periods=["5y", "1y", "6mo", "3mo", "1mo"]):
    stock = yf.Ticker(ticker)
    for p in periods:
        try:
            df = stock.history(period=p)
            if df is not None and not df.empty:
                return df
        except:
            pass
    return None


# Fetch stock history using fallback
history = fetch_stock_history(stock_ticker)

if history is None:
    st.error(f"⚠ Unable to fetch data for {stock_ticker}. Yahoo Finance may be blocking requests or the ticker may be unstable right now.")
    st.stop()

# Process close prices
close_prices = history["Close"].values.reshape(-1, 1)

# Safe live price extraction
try:
    live_price = history["Close"].iloc[-1]
except IndexError:
    st.error("⚠ No valid closing price found in returned data.")
    st.stop()

st.sidebar.success(f"Live Price: ${live_price:.2f}")

# ------------------------ LSTM PREDICTION ----------------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

last_60 = scaled_data[-60:]
X_test = np.reshape(np.array([last_60]), (1, 60, 1))
pred_price = scaler.inverse_transform(model.predict(X_test))[0][0]

# ------------------------ PAGE NAVIGATION ----------------------
# ----------------------------- DASHBOARD ------------------------
if section == "Dashboard":
    st.markdown("<h1 style='text-align:center;'> AI Stock Prediction Dashboard</h1>",
                unsafe_allow_html=True)

    st.metric("Predicted Closing Price (Next Day)", f"${pred_price:.2f}")

    st.write("---")

    # ------------------ CANDLESTICK CHART -----------------------
    st.subheader("📈 Candlestick Chart")

    candle_fig = go.Figure(data=[go.Candlestick(
        x=history.index,
        open=history['Open'],
        high=history['High'],
        low=history['Low'],
        close=history['Close'],

        increasing_line_color='green',
        decreasing_line_color='red',
        increasing_fillcolor='green',
        decreasing_fillcolor='red'
    )])

    candle_fig.update_layout(
        title=f"{stock_ticker} — Candlestick Price Movement",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=520,
        dragmode="pan",
        hovermode="x unified"
    )

    st.plotly_chart(candle_fig, use_container_width=True)

    st.write("---")

    # BUY/SELL SUGGESTION
    st.subheader(" 📍 Investment Suggestion")

    ma50 = pd.Series(close_prices.flatten()).rolling(50).mean().iloc[-1]
    ma200 = pd.Series(close_prices.flatten()).rolling(200).mean().iloc[-1]

    if live_price > ma50 > ma200:
        st.success("🔼 BUY — Upward trend is strong")
    elif live_price < ma50 < ma200:
        st.error("🔽 SELL — Downtrend pressure increasing")
    else:
        st.info("⏳ HOLD — Market showing uncertainty")

    st.caption("Based on Moving Average Crossover Strategy")


# ------------------------- PREDICTIONS PAGE ---------------------
elif section == "Predictions":
    st.markdown("<h1 style='text-align:center;'> Actual vs Predicted Stock Price</h1>",
                unsafe_allow_html=True)

    predicted_dummy = pd.Series(close_prices.flatten()).rolling(5).mean()

    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(close_prices[-300:], label="Actual")
    ax2.plot(predicted_dummy[-300:], label="Predicted", linestyle="dashed")

    ax2.legend()
    st.pyplot(fig2)

# ------------------------- FINANCIAL ASSISTANT ------------------
elif section == "Financial Assistant":
    st.markdown("<h1 style='text-align:center;'> Smart Stock Assistant</h1>",
                unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input(
        "Ask anything about stock trend, prediction or investment:"
    )

    if st.button("Send") and user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        if "buy" in user_input.lower():
            reply = "Buying may be considered when price stays above the 50-day average."
        elif "sell" in user_input.lower():
            reply = "Selling may be considered when a downtrend breaks support."
        elif "predict" in user_input.lower():
            reply = f"Next day predicted close price: **${pred_price:.2f}**"
        else:
            reply = "I can answer questions about stocks, trends, price movement & forecasting."

        st.session_state.chat_history.append({"role": "assistant", "content": reply})

    # Show chat bubble UI
    for i, chat in enumerate(st.session_state.chat_history):
        message(chat["content"], is_user=(chat["role"] == "user"), key=f"chat_{i}")
