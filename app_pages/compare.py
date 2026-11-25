"""
compare.py - Compare the strategy from the Backtest page with other strategies.
Uses the settings and data saved by the Backtest page in st.session_state,
so the user doesn't have to choose the inputs again.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtest import compute_metrics, run_backtest
from strategies import DEFAULT_PARAMS, STRATEGIES

# Path of the backtest page as registered in st.navigation (change if yours differs)
BACKTEST_PAGE = "app_pages/strat_display.py"

BASE_COLOR = "#1ABC9C"
BENCH_COLOR = "#6C7BF2"

st.title("Strategy Comparison")
st.caption("Compare your backtested strategy with other strategies on the same "
           "data, capital and risk settings.")

# ---------------------------------------------------------------
# 1. Get the saved backtest (or ask the user to run one first)
# ---------------------------------------------------------------
bt = st.session_state.get("backtest")
if bt is None:
    st.info("Run a backtest first. The comparison uses the ticker, dates, strategy "
            "and risk settings from the Backtest page.", icon="ℹ️")
    st.page_link(BACKTEST_PAGE, label="Go to Backtest page", icon="📈")
    st.stop()

base = bt["strat"]
base_label = f"{base} (yours)"
data = bt["data"]

params_text = ", ".join(f"{k}={v}" for k, v in bt["params"].items())
with st.container(border=True):
    st.markdown(
        f"**Ticker:** {bt['tick']} &nbsp;|&nbsp; "
        f"**Period:** {bt['start_date']} → {bt['end_date']} &nbsp;|&nbsp; "
        f"**Interval:** {bt['interval']}  \n"
        f"**Your strategy:** {base} ({params_text})  \n"
        f"**Capital:** ${bt['capital']:,.2f} &nbsp;|&nbsp; "
        f"**Risk per trade:** {bt['risk_pct']}% &nbsp;|&nbsp; "
        f"**Stop loss:** {bt['atr_mult']} × ATR({bt['atr_len']})"
    )
    st.caption("To change these settings, go back to the Backtest page and run again.")

# ---------------------------------------------------------------
# 2. Choose what to compare with
# ---------------------------------------------------------------
others = [s for s in STRATEGIES if s != base]

with st.sidebar:
    st.header("Compare with")
    mode = st.radio("Strategies", ["All strategies", "Choose strategies"],
                    label_visibility="collapsed")
    if mode == "All strategies":
        chosen = others
    else:
        chosen = st.multiselect("Select strategies", others,
                                placeholder="Pick one or more strategies")
    st.caption("Other strategies use their default parameters.")

    st.divider()
    rank_by = st.selectbox("Rank by", ["Sharpe Ratio", "Total Return (%)",
                                       "Max Drawdown (%)", "Win Rate (%)"])

if not chosen:
    st.warning("Select at least one strategy to compare with.")
    st.stop()


# ---------------------------------------------------------------
# 3. Run every strategy on the same data and settings
# ---------------------------------------------------------------
def run_one(name, params):
    _, signal_fn, min_bars_fn = STRATEGIES[name]
    if len(data) < min_bars_fn(params):
        return None, None, f"needs at least {min_bars_fn(params)} bars"
    result = signal_fn(data, **params)
    eq, trades = run_backtest(result, bt["capital"], bt["risk_pct"],
                              bt["atr_mult"], bt["atr_len"])
    return eq, compute_metrics(eq, trades, bt["capital"]), None


runs = [(base_label, base, bt["params"])] + [(n, n, DEFAULT_PARAMS[n]) for n in chosen]

rows, curves, skipped = [], {}, []
progress = st.progress(0.0, text="Running backtests...")
for i, (label, name, params) in enumerate(runs):
    try:
        eq, m, reason = run_one(name, params)
    except Exception as err:
        eq, m, reason = None, None, str(err)
    if eq is None:
        skipped.append(f"{label}: {reason}")
    else:
        curves[label] = eq
        rows.append({
            "Strategy": label,
            "Final Equity ($)": m["final_equity"],
            "Total Return (%)": m["total_return"],
            "vs. Buy & Hold (%)": m["total_return"] - m["benchmark"],
            "Sharpe Ratio": m["sharpe"],
            "Max Drawdown (%)": m["max_dd"],
            "Win Rate (%)": m["win_rate"],
            "Actual R:R": m["rr"],
            "Trades": m["n_trades"],
        })
    progress.progress((i + 1) / len(runs), text=f"Running backtests... {label}")
progress.empty()

if skipped:
    st.warning("Skipped (not enough data or an error):  \n- " + "  \n- ".join(skipped))
if base_label not in curves:
    st.error("Your strategy could not be run on this data. Go back to the Backtest page.")
    st.stop()

table = pd.DataFrame(rows)
# Max drawdown is negative: closer to 0 is better, so sort descending for every metric
table = table.sort_values(rank_by, ascending=False).reset_index(drop=True)
table.insert(0, "Rank", range(1, len(table) + 1))
benchmark = compute_metrics(curves[base_label], pd.DataFrame(), bt["capital"])["benchmark"]

# ---------------------------------------------------------------
# 4. Highlights
# ---------------------------------------------------------------
base_rank = int(table.loc[table["Strategy"] == base_label, "Rank"].iloc[0])
best = table.iloc[0]
best_return = table.loc[table["Total Return (%)"].idxmax()]
safest = table.loc[table["Max Drawdown (%)"].idxmax()]

row1 = st.columns(2, gap="medium")
row1[0].metric(f"Your Rank ({rank_by})", f"#{base_rank} of {len(table)}", border=True)
row1[1].metric(f"Best ({rank_by})", best["Strategy"].replace(" (yours)", ""),
               f"{best[rank_by]:.2f}", delta_color="off", border=True)

row2 = st.columns(2, gap="medium")
row2[0].metric("Highest Return", best_return["Strategy"].replace(" (yours)", ""),
               f"{best_return['Total Return (%)']:.2f}%", border=True)
row2[1].metric("Smallest Drawdown", safest["Strategy"].replace(" (yours)", ""),
               f"{safest['Max Drawdown (%)']:.2f}%", delta_color="off", border=True)
# ---------------------------------------------------------------
# 5. Table + charts
# ---------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Comparison Table", "Equity Curves", "Risk vs. Return"])

with tab1:
    st.dataframe(
        table, width="stretch", hide_index=True,
        column_config={
            "Final Equity ($)": st.column_config.NumberColumn(format="$%.2f"),
            "Total Return (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "vs. Buy & Hold (%)": st.column_config.NumberColumn(format="%+.2f%%"),
            "Sharpe Ratio": st.column_config.NumberColumn(format="%.2f"),
            "Max Drawdown (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "Win Rate (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "Actual R:R": st.column_config.NumberColumn(format="1 : %.2f"),
        })
    st.download_button("⬇️ Download comparison (CSV)",
                       table.to_csv(index=False).encode("utf-8"),
                       file_name=f"{bt['tick']}_strategy_comparison.csv",
                       mime="text/csv")

with tab2:
    fig = go.Figure()
    for label, eq in curves.items():
        is_base = label == base_label
        fig.add_trace(go.Scatter(
            x=eq.index, y=eq["Strategy"], name=label, mode="lines",
            line=dict(width=3 if is_base else 1.3,
                      color=BASE_COLOR if is_base else None),
            opacity=1 if is_base else 0.75))
    fig.add_trace(go.Scatter(x=curves[base_label].index,
                             y=curves[base_label]["Buy & Hold"],
                             name="Buy & Hold", mode="lines",
                             line=dict(color=BENCH_COLOR, width=1.5, dash="dot")))
    fig.update_layout(height=550, yaxis_title="Equity ($)", hovermode="x unified",
                      margin=dict(l=60, r=20, t=40, b=20),
                      legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, width="stretch")

with tab3:
    st.caption("Top-right is best: higher return with a smaller drawdown.")
    fig = go.Figure()
    for _, row in table.iterrows():
        is_base = row["Strategy"] == base_label
        fig.add_trace(go.Scatter(
            x=[row["Max Drawdown (%)"]], y=[row["Total Return (%)"]],
            mode="markers+text", name=row["Strategy"],
            text=[row["Strategy"].replace(" (yours)", " ★")], textposition="top center",
            marker=dict(size=16 if is_base else 11,
                        color=BASE_COLOR if is_base else "#90A4AE",
                        line=dict(width=1, color="white")),
            hovertemplate=(f"<b>{row['Strategy']}</b><br>Return: %{{y:.2f}}%<br>"
                           f"Max drawdown: %{{x:.2f}}%<br>"
                           f"Sharpe: {row['Sharpe Ratio']:.2f}<extra></extra>")))
    fig.add_hline(y=benchmark, line_dash="dot", line_color=BENCH_COLOR,
                  annotation_text="Buy & Hold return", annotation_position="bottom right")
    fig.update_layout(height=550, showlegend=False,
                      xaxis_title="Max Drawdown (%)", yaxis_title="Total Return (%)",
                      margin=dict(l=60, r=20, t=40, b=20))
    st.plotly_chart(fig, width="stretch")
