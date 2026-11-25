"""
backtest.py
-----------
Turns a strategy's "position" column (1 = want to be long, 0 = flat) into
actual trades, an equity curve and performance statistics.

Rules
- A signal from bar t's close is executed at bar t+1's OPEN (no lookahead).
- Position size: risk Z% of current equity per trade, where the risk is the
  distance from entry to the stop loss (X x ATR). Capped at available cash
  (no leverage).
- Stop loss: if a bar's Low touches the stop, exit at the stop price
  (or at the Open if the bar gaps below the stop).
- After a stop-out, wait for the signal to turn off and on again before
  re-entering (otherwise "always-on" signals would re-enter immediately).
"""

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
except ImportError:
    import pandas_ta_classic as ta


def run_backtest(df: pd.DataFrame, capital: float, risk_pct: float,
                 atr_mult: float, atr_len: int = 14):
    """Returns (equity DataFrame, trades DataFrame)."""
    df = df.copy()
    atr = ta.atr(df["High"], df["Low"], df["Close"], length=atr_len)
    df["ATR_risk"] = atr if atr is not None else np.nan

    o, h, l, c = (df[k].to_numpy() for k in ("Open", "High", "Low", "Close"))
    sig = df["position"].fillna(0).to_numpy()
    atr_v = df["ATR_risk"].to_numpy()
    dates = df.index

    cash, shares = float(capital), 0.0
    entry_px = stop = 0.0
    entry_date = None
    stopped_out = False
    equity = np.empty(len(df))
    equity[0] = cash
    trades = []

    def close_trade(i, px, reason):
        nonlocal cash, shares
        cash += shares * px
        trades.append({
            "Entry date": entry_date, "Entry price": entry_px,
            "Exit date": dates[i], "Exit price": px,
            "Shares": shares, "PnL ($)": shares * (px - entry_px),
            "Return (%)": (px / entry_px - 1) * 100, "Exit reason": reason,
        })
        shares = 0.0

    for i in range(1, len(df)):
        want_long = sig[i - 1] == 1          # yesterday's signal, today's open

        if not want_long:
            stopped_out = False

        # 1) act on the signal at the open
        if shares > 0 and not want_long:
            close_trade(i, o[i], "Signal")
        elif shares == 0 and want_long and not stopped_out and atr_v[i - 1] > 0:
            stop_dist = atr_mult * atr_v[i - 1]
            equity_now = cash
            size = (equity_now * risk_pct / 100) / stop_dist
            size = min(size, cash / o[i])                # no leverage
            if size > 0:
                shares, entry_px, entry_date = size, o[i], dates[i]
                cash -= shares * entry_px
                stop = entry_px - stop_dist

        # 2) stop loss during the bar
        if shares > 0 and l[i] <= stop:
            close_trade(i, min(o[i], stop), "Stop loss")
            stopped_out = True

        equity[i] = cash + shares * c[i]

    # close any open trade at the last price so it counts in the stats
    if shares > 0:
        close_trade(len(df) - 1, c[-1], "End of data")
        equity[-1] = cash

    eq = pd.DataFrame(index=dates)
    eq["Strategy"] = equity
    eq["Buy & Hold"] = capital * df["Close"] / df["Close"].iloc[0]
    eq["Drawdown (%)"] = (eq["Strategy"] / eq["Strategy"].cummax() - 1) * 100
    return eq, pd.DataFrame(trades)


def compute_metrics(eq: pd.DataFrame, trades: pd.DataFrame, capital: float) -> dict:
    final = eq["Strategy"].iloc[-1]

    # bars per year estimated from the data itself -> works for any interval
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    bars_per_year = len(eq) / years
    rets = eq["Strategy"].pct_change().dropna()
    sharpe = (rets.mean() / rets.std() * np.sqrt(bars_per_year)
              if rets.std() > 0 else 0.0)

    n = len(trades)
    if n:
        wins = trades.loc[trades["PnL ($)"] > 0, "PnL ($)"]
        losses = trades.loc[trades["PnL ($)"] <= 0, "PnL ($)"]
        win_rate = len(wins) / n * 100
        rr = (wins.mean() / abs(losses.mean())
              if len(wins) and len(losses) and losses.mean() != 0 else np.nan)
    else:
        win_rate, rr = 0.0, np.nan

    return {
        "final_equity": final,
        "total_return": (final / capital - 1) * 100,
        "benchmark": (eq["Buy & Hold"].iloc[-1] / capital - 1) * 100,
        "sharpe": sharpe,
        "max_dd": eq["Drawdown (%)"].min(),
        "win_rate": win_rate,
        "rr": rr,
        "n_trades": n,
    }
