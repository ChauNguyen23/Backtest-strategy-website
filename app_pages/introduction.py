import streamlit as st
import pandas as pd
import numpy as np


st.title("📈 Trading Strategy Backtester & Live Simulator")

st.markdown("### Welcome to our web platform for backtesting trading strategies "
            "and simulating them in real time")

st.divider()

st.subheader("Team Members:")
members = ["Jason Nguyen", "Thi Quyen Vo", "Thanh Liem Huynh","Vo Hoang Dang"]
cols = st.columns(len(members))

for col,mem in zip(cols,members):
    with col.container():
        st.markdown(f"<p style='text-align:center'>{mem}</p>",
                    unsafe_allow_html=True)


st.subheader("Our goals")
c1, c2, c3 = st.columns(3)
with c1.container(border=True):
    st.markdown("#### 🔍 Backtest")
    st.write("Test strategies on historical Yahoo Finance data")
with c2.container(border=True):
    st.markdown("#### ⚡ Simulate")
    st.write("Follow strategies in a live market simulation, risk-free")
with c3.container(border=True):
    st.markdown("#### 🛡️ Manage Risk")
    st.write("Size positions by % risk and set ATR-based stop losses")


