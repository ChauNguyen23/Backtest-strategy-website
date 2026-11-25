import streamlit as st
import pandas as pd
import yfinance as yf


st.markdown("## Backtesting strategies with historical data")
tickers = ["SPY","QQQ","BTC-USD", "AAPL","NVDA"]
period_invl = {}
invl = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h",
             "1d", "5d", "1wk", "1mo", "3mo"]

strategies = ["SMA Crossover",        # Jason – Trend-following (50/200)
    "Bollinger Reversal",   # Jason – Mean reversion (20, 2)
    "ATR Breakout",         # Jason – Volatility (14)
    "MACD Crossover",       # Quyen – Momentum (12, 26, 9)
    "RSI Reversal",         # Quyen – Mean reversion (14; 30/70)
    "Donchian Breakout",    # Quyen – Trend-following (20)
    "ROC Momentum",         # Liem – Momentum (60 or 120)
    "Z-Score Reversion",    # Liem – Mean reversion (20)
    "Keltner Breakout",     # Liem – Volatility (20, 2×ATR)
    "Supertrend",           # Hoang – Trend-following (10, 3)
    "Stochastic",           # Hoang – Mean reversion (14, 3)
    "SMA200 + RSI"]         # Hoang – Hybrid
with st.sidebar:

    st.header("Data")
    tick = st.selectbox(label = "Stock", options = tickers,
                 accept_new_options = True, 
                 index = None,
                 placeholder = "Select a stock or enter a new one")
    
    l, r = st.columns(2)
    start_date = l.date_input("Start date")
    end_date = r.date_input("End date")
    data_interval = st.selectbox(label = "Interval", options = invl,
                            index = None,
                            placeholder = "Select an interval")
    
    st.divider()

    st.header("Strategy")
    strat = st.selectbox(label = "Choose your strategy", options = strategies,
                 index = None,
                 placeholder = "Select a strategy")



st.write("You select", tick)

@st.cache_data(ttl= "5h")
def load_data(t, s, e, i):
    df = yf.Ticker(t).history(start=s, end=e, interval=i, auto_adjust=True)
    return df


data = load_data(tick, start_date, end_date, data_interval)

if tick == "":
    st.error("Please select or enter a valid stock", icon="🚨")
    st.stop()
elif start_date > end_date:
    st.error("Please choose proper start and end date", icon="🚨")
    st.stop()
elif data.empty:
    st.error("Please enter your stock information", icon="🚨")
    st.stop()
st.success(f"Downloaded {len(data)} lines from {tick}.")

# @st.cache_data

