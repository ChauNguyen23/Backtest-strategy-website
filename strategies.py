"""
strategies.py
-------------
All 12 group strategies, built on pandas-ta.

Each strategy has two parts:
  1. params_xxx()  -> draws its input boxes in the sidebar and returns a dict
  2. signal_xxx(df, **params) -> returns df with indicator columns + "position"

"position" convention (long-only):
    1 = holding the asset, 0 = in cash
The position is based on the bar's CLOSE, so the backtest engine must shift it
by one bar before computing returns (trade on the next bar -> no lookahead).
"""

import inspect

import numpy as np
import pandas as pd
import streamlit as st

try:                                    # works with either package name
    import pandas_ta as ta
except ImportError:
    import pandas_ta_classic as ta


# ===================================================================
# Helpers
# ===================================================================
def _col(result: pd.DataFrame, prefix: str) -> pd.Series:
    """Pick a column from a pandas-ta result by its name prefix.
    pandas-ta versions name columns slightly differently
    (e.g. 'BBL_20_2.0' vs 'BBL_20_2.0_2.0'), so we match the start only."""
    if result is None:
        raise ValueError("Not enough data to compute this indicator.")
    for c in result.columns:
        if c.startswith(prefix):
            return result[c]
    raise KeyError(f"Column starting with '{prefix}' not found in {list(result.columns)}")


def _check(series, name):
    if series is None:
        raise ValueError(f"Not enough data to compute {name}.")
    return series


def _hold(entry: pd.Series, exit_: pd.Series) -> pd.Series:
    """Turn entry/exit events into a 1/0 position that is held between them.
    If entry and exit happen on the same bar, exit wins (stay conservative)."""
    pos = pd.Series(np.nan, index=entry.index)
    pos[entry.fillna(False)] = 1
    pos[exit_.fillna(False)] = 0
    return pos.ffill().fillna(0).astype(int)


def _cross_above(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))


def _cross_below(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))


# ===================================================================
# 1. SMA Crossover (Jason) – Trend-following
# ===================================================================
def params_sma_cross():
    l, r = st.columns(2)
    fast = l.number_input("Fast SMA", 2, 300, 50, key="sma_fast")
    slow = r.number_input("Slow SMA", 5, 500, 200, key="sma_slow")
    return {"fast": fast, "slow": slow}


def signal_sma_cross(df, fast, slow):
    df = df.copy()
    df["SMA_fast"] = _check(ta.sma(df["Close"], length=fast), "SMA fast")
    df["SMA_slow"] = _check(ta.sma(df["Close"], length=slow), "SMA slow")
    df["position"] = (df["SMA_fast"] > df["SMA_slow"]).astype(int)
    return df


# ===================================================================
# 2. Bollinger Reversal (Jason) – Mean reversion
# ===================================================================
def params_bollinger():
    l, r = st.columns(2)
    length = l.number_input("Period", 5, 200, 20, key="bb_len")
    std = r.number_input("Std. deviation", 0.5, 5.0, 2.0, step=0.5, key="bb_std")
    return {"length": length, "std": std}


def signal_bollinger(df, length, std):
    df = df.copy()
    # newer pandas-ta uses lower_std/upper_std, older versions use std
    if "lower_std" in inspect.signature(ta.bbands).parameters:
        bb = ta.bbands(df["Close"], length=length, lower_std=std, upper_std=std)
    else:
        bb = ta.bbands(df["Close"], length=length, std=std)
    df["BB_lower"] = _col(bb, "BBL")
    df["BB_mid"] = _col(bb, "BBM")
    df["BB_upper"] = _col(bb, "BBU")
    entry = df["Close"] < df["BB_lower"]          # oversold -> buy
    exit_ = df["Close"] >= df["BB_mid"]           # back to the mean -> sell
    df["position"] = _hold(entry, exit_)
    return df


# ===================================================================
# 3. ATR Breakout (Jason) – Volatility
# ===================================================================
def params_atr_breakout():
    l, r = st.columns(2)
    length = l.number_input("ATR period", 2, 100, 14, key="atr_len")
    k = r.number_input("Multiplier (k × ATR)", 0.5, 5.0, 1.0, step=0.5, key="atr_k")
    return {"length": length, "k": k}


def signal_atr_breakout(df, length, k):
    df = df.copy()
    df["ATR"] = _check(ta.atr(df["High"], df["Low"], df["Close"], length=length), "ATR")
    prev_close, prev_atr = df["Close"].shift(1), df["ATR"].shift(1)
    df["ATR_upper"] = prev_close + k * prev_atr
    df["ATR_lower"] = prev_close - k * prev_atr
    entry = df["Close"] > df["ATR_upper"]         # unusually strong up-move
    exit_ = df["Close"] < df["ATR_lower"]         # unusually strong down-move
    df["position"] = _hold(entry, exit_)
    return df


# ===================================================================
# 4. MACD Crossover (Quyen) – Momentum
# ===================================================================
def params_macd():
    c1, c2, c3 = st.columns(3)
    fast = c1.number_input("Fast", 2, 100, 12, key="macd_fast")
    slow = c2.number_input("Slow", 5, 200, 26, key="macd_slow")
    signal = c3.number_input("Signal", 2, 50, 9, key="macd_sig")
    return {"fast": fast, "slow": slow, "signal": signal}


def signal_macd(df, fast, slow, signal):
    df = df.copy()
    m = ta.macd(df["Close"], fast=fast, slow=slow, signal=signal)
    df["MACD"] = _col(m, "MACD_")
    df["MACD_signal"] = _col(m, "MACDs")
    df["MACD_hist"] = _col(m, "MACDh")
    df["position"] = (df["MACD"] > df["MACD_signal"]).astype(int)
    return df


# ===================================================================
# 5. RSI Reversal (Quyen) – Mean reversion
# ===================================================================
def params_rsi():
    length = st.number_input("RSI period", 2, 100, 14, key="rsi_len")
    l, r = st.columns(2)
    low = l.number_input("Oversold (buy)", 5, 50, 30, key="rsi_low")
    high = r.number_input("Overbought (sell)", 50, 95, 70, key="rsi_high")
    return {"length": length, "low": low, "high": high}


def signal_rsi(df, length, low, high):
    df = df.copy()
    df["RSI"] = _check(ta.rsi(df["Close"], length=length), "RSI")
    df["position"] = _hold(df["RSI"] < low, df["RSI"] > high)
    return df


# ===================================================================
# 6. Donchian Breakout (Quyen) – Trend-following
# ===================================================================
def params_donchian():
    length = st.number_input("Donchian channel period", 5, 200, 20, key="dc_len")
    return {"length": length}


def signal_donchian(df, length):
    df = df.copy()
    dc = ta.donchian(df["High"], df["Low"], lower_length=length, upper_length=length)
    df["DC_lower"] = _col(dc, "DCL")
    df["DC_upper"] = _col(dc, "DCU")
    # compare with YESTERDAY's channel: today's channel already contains today's high
    entry = df["Close"] > df["DC_upper"].shift(1)
    exit_ = df["Close"] < df["DC_lower"].shift(1)
    df["position"] = _hold(entry, exit_)
    return df


# ===================================================================
# 7. ROC Momentum (Liem) – Momentum
# ===================================================================
def params_roc():
    length = st.selectbox("ROC period (days)", [60, 120], key="roc_len")
    return {"length": length}


def signal_roc(df, length):
    df = df.copy()
    df["ROC"] = _check(ta.roc(df["Close"], length=length), "ROC")
    df["position"] = (df["ROC"] > 0).astype(int)
    return df


# ===================================================================
# 8. Z-Score Reversion (Liem) – Mean reversion
# ===================================================================
def params_zscore():
    l, r = st.columns(2)
    length = l.number_input("Period", 5, 200, 20, key="z_len")
    thr = r.number_input("Z threshold", 0.5, 4.0, 2.0, step=0.5, key="z_thr")
    return {"length": length, "threshold": thr}


def signal_zscore(df, length, threshold):
    df = df.copy()
    df["ZSCORE"] = _check(ta.zscore(df["Close"], length=length), "Z-score")
    entry = df["ZSCORE"] < -threshold              # far below the mean
    exit_ = df["ZSCORE"] >= 0                       # back to the mean
    df["position"] = _hold(entry, exit_)
    return df


# ===================================================================
# 9. Keltner Breakout (Liem) – Volatility
# ===================================================================
def params_keltner():
    l, r = st.columns(2)
    length = l.number_input("Period", 5, 200, 20, key="kc_len")
    scalar = r.number_input("ATR multiplier", 0.5, 5.0, 2.0, step=0.5, key="kc_scalar")
    return {"length": length, "scalar": scalar}


def signal_keltner(df, length, scalar):
    df = df.copy()
    kc = ta.kc(df["High"], df["Low"], df["Close"], length=length, scalar=scalar)
    df["KC_lower"] = _col(kc, "KCL")
    df["KC_mid"] = _col(kc, "KCB")
    df["KC_upper"] = _col(kc, "KCU")
    entry = df["Close"] > df["KC_upper"].shift(1)  # breakout above yesterday's band
    exit_ = df["Close"] < df["KC_mid"]              # falls back below the middle
    df["position"] = _hold(entry, exit_)
    return df


# ===================================================================
# 10. Supertrend (Hoang) – Trend-following (ATR-based)
# ===================================================================
def params_supertrend():
    l, r = st.columns(2)
    length = l.number_input("ATR period", 2, 100, 10, key="st_len")
    mult = r.number_input("Multiplier", 0.5, 10.0, 3.0, step=0.5, key="st_mult")
    return {"length": length, "multiplier": mult}


def signal_supertrend(df, length, multiplier):
    df = df.copy()
    s = ta.supertrend(df["High"], df["Low"], df["Close"],
                      length=length, multiplier=multiplier)
    df["SUPERTREND"] = _col(s, "SUPERT_")
    df["ST_direction"] = _col(s, "SUPERTd")       # 1 = uptrend, -1 = downtrend
    df["position"] = (df["ST_direction"] == 1).astype(int)
    return df


# ===================================================================
# 11. Stochastic (Hoang) – Mean reversion
# ===================================================================
def params_stochastic():
    l, r = st.columns(2)
    k = l.number_input("%K", 2, 100, 14, key="sto_k")
    d = r.number_input("%D", 1, 50, 3, key="sto_d")
    l2, r2 = st.columns(2)
    low = l2.number_input("Oversold", 5, 50, 20, key="sto_low")
    high = r2.number_input("Overbought", 50, 95, 80, key="sto_high")
    return {"k": k, "d": d, "low": low, "high": high}


def signal_stochastic(df, k, d, low, high):
    df = df.copy()
    so = ta.stoch(df["High"], df["Low"], df["Close"], k=k, d=d)
    df["STOCH_k"] = _col(so, "STOCHk")
    df["STOCH_d"] = _col(so, "STOCHd")
    entry = _cross_above(df["STOCH_k"], df["STOCH_d"]) & (df["STOCH_k"] < low)
    exit_ = _cross_below(df["STOCH_k"], df["STOCH_d"]) & (df["STOCH_k"] > high)
    df["position"] = _hold(entry, exit_)
    return df


# ===================================================================
# 12. SMA200 + RSI pullback (Hoang) – Hybrid
# ===================================================================
def params_sma_rsi():
    l, r = st.columns(2)
    sma_len = l.number_input("Trend SMA", 20, 400, 200, key="sr_sma")
    rsi_len = r.number_input("RSI period", 2, 50, 14, key="sr_rsi")
    l2, r2 = st.columns(2)
    buy = l2.number_input("Buy when RSI below", 5, 60, 30, key="sr_buy")
    sell = r2.number_input("Sell when RSI above", 40, 95, 70, key="sr_sell")
    return {"sma_len": sma_len, "rsi_len": rsi_len, "buy": buy, "sell": sell}


def signal_sma_rsi(df, sma_len, rsi_len, buy, sell):
    df = df.copy()
    df["SMA_trend"] = _check(ta.sma(df["Close"], length=sma_len), "SMA")
    df["RSI"] = _check(ta.rsi(df["Close"], length=rsi_len), "RSI")
    uptrend = df["Close"] > df["SMA_trend"]
    entry = uptrend & (df["RSI"] < buy)           # dip inside an uptrend
    exit_ = (df["RSI"] > sell) | ~uptrend         # recovered, or trend broke
    df["position"] = _hold(entry, exit_)
    return df


# ===================================================================
# Registry: strategy name -> (sidebar params, signal function, min bars needed)
# min bars = longest lookback, used to check there is enough data
# ===================================================================
STRATEGIES = {
    "SMA Crossover":      (params_sma_cross,    signal_sma_cross,    lambda p: p["slow"]),
    "Bollinger Reversal": (params_bollinger,    signal_bollinger,    lambda p: p["length"]),
    "ATR Breakout":       (params_atr_breakout, signal_atr_breakout, lambda p: p["length"] + 1),
    "MACD Crossover":     (params_macd,         signal_macd,         lambda p: p["slow"] + p["signal"]),
    "RSI Reversal":       (params_rsi,          signal_rsi,          lambda p: p["length"] + 1),
    "Donchian Breakout":  (params_donchian,     signal_donchian,     lambda p: p["length"] + 1),
    "ROC Momentum":       (params_roc,          signal_roc,          lambda p: p["length"] + 1),
    "Z-Score Reversion":  (params_zscore,       signal_zscore,       lambda p: p["length"]),
    "Keltner Breakout":   (params_keltner,      signal_keltner,      lambda p: p["length"] + 1),
    "Supertrend":         (params_supertrend,   signal_supertrend,   lambda p: p["length"] + 1),
    "Stochastic":         (params_stochastic,   signal_stochastic,   lambda p: p["k"] + p["d"] + 3),
    "SMA200 + RSI":       (params_sma_rsi,      signal_sma_rsi,      lambda p: max(p["sma_len"], p["rsi_len"] + 1)),
}

# ===================================================================
# Default parameters (same defaults as the sidebar boxes above).
# Used by the comparison page for the strategies the user compares against.
# If you change a default in a params_xxx() function, change it here too.
# ===================================================================
DEFAULT_PARAMS = {
    "SMA Crossover":      {"fast": 50, "slow": 200},
    "Bollinger Reversal": {"length": 20, "std": 2.0},
    "ATR Breakout":       {"length": 14, "k": 1.0},
    "MACD Crossover":     {"fast": 12, "slow": 26, "signal": 9},
    "RSI Reversal":       {"length": 14, "low": 30, "high": 70},
    "Donchian Breakout":  {"length": 20},
    "ROC Momentum":       {"length": 60},
    "Z-Score Reversion":  {"length": 20, "threshold": 2.0},
    "Keltner Breakout":   {"length": 20, "scalar": 2.0},
    "Supertrend":         {"length": 10, "multiplier": 3.0},
    "Stochastic":         {"k": 14, "d": 3, "low": 20, "high": 80},
    "SMA200 + RSI":       {"sma_len": 200, "rsi_len": 14, "buy": 30, "sell": 70},
}