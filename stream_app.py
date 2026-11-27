import streamlit as st

page = st.navigation(
    [
        st.Page("app_pages/introduction.py", title = "Introduction", icon = "👋"),
        st.Page("app_pages/strat_display.py",title = "Backtest", icon = "📈"),
        st.Page("app_pages/compare.py", title="Compare Strategies", icon="⚖️"),
    ]
)

page.run()