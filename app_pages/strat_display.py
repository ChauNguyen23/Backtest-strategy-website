from datetime import date, timedelta

import pandas as pd
import streamlit as st
import yfinance as yf

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backtest import compute_metrics, run_backtest
from strategies import STRATEGIES

st.title("Web-Based Trading Strategy Backtester")
st.caption("Position sizing by % account risk | ATR-based stop loss | "
           "Entry & exit visualization")

tickers = ["SPY", "QQQ", "BTC-USD", "AAPL", "NVDA"]

# Max days back from today that Yahoo keeps for each interval (None = full history)
MAX_DAYS = {
    "1m": 7, "2m": 59, "5m": 59, "15m": 59, "30m": 59, "60m": 729, "90m": 59,
    "1h": 729, "1d": None, "5d": None, "1wk": None, "1mo": None, "3mo": None,
}
invl = list(MAX_DAYS.keys())
strategies = list(STRATEGIES.keys())

today = date.today()

with st.sidebar:
    st.header("Data")
    tick = st.selectbox(label="Stock", options=tickers,
                        accept_new_options=True,
                        index=None,
                        placeholder="Select a stock or enter a new one")

    l, r = st.columns(2)
    start_date = l.date_input("Start date", value=today - timedelta(days=15 * 365),
                              max_value=today)
    end_date = r.date_input("End date", value=today, max_value=today)
    data_interval = st.selectbox(label="Interval", options=invl,
                                 index=None,
                                 placeholder="Select an interval")

    st.divider()

    st.header("Strategy")
    strat = st.selectbox(label="Choose your strategy", options=strategies,
                         index=None,
                         placeholder="Select a strategy")

    # Input boxes for the chosen strategy appear right under the selectbox
    params = {}
    if strat:
        params_ui, signal_fn, min_bars_fn = STRATEGIES[strat]
        params = params_ui()

    st.divider()

    st.header("Capital & Risk")
    capital = st.number_input("Initial capital ($)", min_value=100.0,
                              value=10000.0, step=1000.0, format="%.2f")
    risk_pct = st.number_input("Risk per trade Z (% of account)", min_value=0.1,
                               max_value=100.0, value=1.0, step=0.5,
                               help="How much of your account you are willing to lose "
                                    "if a trade hits its stop loss.")
    rl, rr_col = st.columns(2)
    atr_mult = rl.number_input("Stop loss = X × ATR", min_value=0.5,
                               max_value=10.0, value=2.0, step=0.5)
    atr_len = rr_col.number_input("ATR period (stop)", min_value=2,
                                  max_value=100, value=14)


# ---------------------------------------------------------------
# Input checks (before downloading anything)
# ---------------------------------------------------------------
errors = []
if not tick:
    errors.append("Please select or enter a stock.")
if not data_interval:
    errors.append("Please select an interval.")
if not strat:
    errors.append("Please select a strategy.")
if start_date >= end_date:
    errors.append("Start date must be before end date.")
if data_interval and MAX_DAYS[data_interval] is not None:
    limit = MAX_DAYS[data_interval]
    earliest = today - timedelta(days=limit)
    if start_date < earliest:
        errors.append(f"Interval {data_interval} only has data for the last {limit} days. "
                      f"Choose a start date on or after {earliest}.")
if strat == "SMA Crossover" and params["fast"] >= params["slow"]:
    errors.append("Fast SMA must be shorter than slow SMA.")
if strat == "MACD Crossover" and params["fast"] >= params["slow"]:
    errors.append("MACD fast period must be shorter than slow period.")

if errors:
    for msg in errors:
        st.error(msg, icon="🚨")
    st.stop()

tick = tick.strip().upper()


# ---------------------------------------------------------------
# Download data
# ---------------------------------------------------------------
@st.cache_data(ttl="5h")
def load_data(t, s, e, i):
    df = yf.Ticker(t).history(start=s, end=e, interval=i, auto_adjust=True)
    if df.empty:
        return df
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).as_unit("ns")
    return df.dropna(subset=["Open", "High", "Low", "Close"])


# yfinance does not include the end date -> add one day
data = load_data(tick, start_date, end_date + timedelta(days=1), data_interval)

if data.empty:
    st.error(f"No data found for {tick}. Check the ticker or the date range.", icon="🚨")
    st.stop()

min_bars = min_bars_fn(params)
if len(data) < min_bars:
    st.error(f"{strat} needs at least {min_bars} bars of data, but only {len(data)} "
             "were downloaded. Choose a longer date range.", icon="🚨")
    st.stop()

st.caption(f"✅ Downloaded {len(data):,} bars of {tick} "
           f"({data.index[0]:%Y-%m-%d} → {data.index[-1]:%Y-%m-%d}).")


# ---------------------------------------------------------------
# Run the strategy -> `result` has the indicator columns + "position" (1/0)
# ---------------------------------------------------------------
try:
    result = signal_fn(data, **params)
except Exception as err:
    st.error(f"Could not calculate {strat}: {err}", icon="🚨")
    st.stop()


# ---------------------------------------------------------------
# Backtest + statistics
# ---------------------------------------------------------------
equity, trades = run_backtest(result, capital, risk_pct, atr_mult, atr_len)
m = compute_metrics(equity, trades, capital)

# Save this run so the Compare page can reuse it without asking for inputs again
st.session_state["backtest"] = {
    "tick": tick, "start_date": start_date, "end_date": end_date,
    "interval": data_interval, "strat": strat, "params": params,
    "capital": capital, "risk_pct": risk_pct,
    "atr_mult": atr_mult, "atr_len": atr_len,
    "data": data,
}

STRATEGY_COLOR = "#1ABC9C"
BENCH_COLOR = "#6C7BF2"
DD_COLOR = "#E74C3C"

row1 = st.columns(3, gap="medium")
row1[0].metric("Final Equity", f"${m['final_equity']:,.2f}",
               f"{m['total_return']:.2f}%", border=True)
row1[1].metric("Benchmark (B&H)", f"{m['benchmark']:.2f}%", border=True)
row1[2].metric("Sharpe Ratio", f"{m['sharpe']:.2f}", border=True)

row2 = st.columns(3, gap="medium")
row2[0].metric("Max Drawdown", f"{m['max_dd']:.2f}%", border=True)
row2[1].metric("Win Rate", f"{m['win_rate']:.2f}%", border=True)
row2[2].metric("Actual R:R", "N/A" if pd.isna(m["rr"]) else f"1 : {m['rr']:.2f}",
               border=True)

if m["n_trades"] == 0:
    st.warning("This strategy made no trades in the selected period. "
               "Try a longer date range or different parameters.")


# ---------------------------------------------------------------
# Charts
# ---------------------------------------------------------------
# Indicator columns drawn on top of the price vs. in a panel below it
OVERLAYS = ["SMA_fast", "SMA_slow", "SMA_trend", "BB_lower", "BB_mid", "BB_upper",
            "DC_lower", "DC_upper", "KC_lower", "KC_mid", "KC_upper",
            "SUPERTREND", "ATR_upper", "ATR_lower"]
OSCILLATORS = ["RSI", "MACD", "MACD_signal", "MACD_hist", "ROC", "ZSCORE",
               "STOCH_k", "STOCH_d"]


def price_chart(df, trades):
    overlays = [col for col in OVERLAYS if col in df.columns]
    oscillators = [col for col in OSCILLATORS if col in df.columns]
    rows = 2 if oscillators else 1
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.72, 0.28] if rows == 2 else [1])

    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                                 low=df["Low"], close=df["Close"], name="Price",
                                 increasing_line_color="#26A69A",
                                 decreasing_line_color="#EF5350"), row=1, col=1)

    for col in overlays:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, mode="lines",
                                 line=dict(width=1.2)), row=1, col=1)

    if not trades.empty:
        fig.add_trace(go.Scatter(
            x=trades["Entry date"], y=trades["Entry price"], mode="markers", name="Buy",
            marker=dict(symbol="triangle-up", size=11, color="#00B386",
                        line=dict(width=1, color="white"))), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=trades["Exit date"], y=trades["Exit price"], mode="markers", name="Sell",
            marker=dict(symbol="triangle-down", size=11, color=DD_COLOR,
                        line=dict(width=1, color="white")),
            customdata=trades["Exit reason"],
            hovertemplate="Sell: %{y:.2f}<br>%{customdata}<extra></extra>"), row=1, col=1)

    for col in oscillators:
        if col == "MACD_hist":
            fig.add_trace(go.Bar(x=df.index, y=df[col], name=col,
                                 marker_color="#B0BEC5"), row=2, col=1)
        else:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, mode="lines",
                                     line=dict(width=1.2)), row=2, col=1)

    fig.update_layout(height=650 if rows == 2 else 520, xaxis_rangeslider_visible=False,
                      margin=dict(l=60, r=20, t=40, b=20), hovermode="x unified",
                      legend=dict(orientation="h", y=1.04, x=0))
    return fig


def equity_chart(eq):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        row_heights=[0.7, 0.3])
    fig.add_trace(go.Scatter(x=eq.index, y=eq["Strategy"], name="Strategy ($)",
                             line=dict(color=STRATEGY_COLOR, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=eq.index, y=eq["Buy & Hold"], name="Buy & Hold ($)",
                             line=dict(color=BENCH_COLOR, width=1.5, dash="dot")),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=eq.index, y=eq["Drawdown (%)"], name="Drawdown (%)",
                             fill="tozeroy", line=dict(color=DD_COLOR, width=1),
                             fillcolor="rgba(231, 76, 60, 0.45)"), row=2, col=1)
    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    fig.update_layout(height=600, margin=dict(l=60, r=20, t=40, b=20),
                      hovermode="x unified")
    return fig


tab1, tab2, tab3, tab4 = st.tabs(["Candlestick Chart & Entry/Exit Points",
                                  "Equity Curve", "Trade Log", "Backtest Data"])

with tab1:
    st.plotly_chart(price_chart(result, trades), width="stretch")

with tab2:
    st.plotly_chart(equity_chart(equity), width="stretch")

with tab3:
    if trades.empty:
        st.info("No trades to show.")
    else:
        st.caption(f"{m['n_trades']} trades")
        st.dataframe(
            trades,
            width="stretch", hide_index=True,
            column_config={
                "Entry date": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
                "Exit date": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
                "Entry price": st.column_config.NumberColumn(format="%.2f"),
                "Exit price": st.column_config.NumberColumn(format="%.2f"),
                "Shares": st.column_config.NumberColumn(format="%.4f"),
                "PnL ($)": st.column_config.NumberColumn(format="$%.2f"),
                "Return (%)": st.column_config.NumberColumn(format="%.2f%%"),
            })

with tab4:
    # Prices + strategy indicators + signal, joined with the equity curve
    bt_data = result.join(equity)
    bt_data = bt_data.rename(columns={"position": "Signal (1=long)",
                                      "Strategy": "Strategy Equity ($)",
                                      "Buy & Hold": "Buy & Hold Equity ($)"})

    st.caption(f"{len(bt_data):,} rows × {bt_data.shape[1]} columns, "
               "one row per bar")
    st.dataframe(bt_data.round(4), width="stretch", height=450)

    st.download_button(
        "⬇️ Download backtest data (CSV)",
        bt_data.to_csv().encode("utf-8"),
        file_name=f"{tick}_{strat.replace(' ', '_')}_backtest.csv",
        mime="text/csv",
    )
