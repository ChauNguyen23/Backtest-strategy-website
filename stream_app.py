import streamlit as st

page = st.navigation(
    [
        st.Page("app_pages/introduction.py", title = "Introduction", icon = "👋"),
        st.Page("app_pages/app_test.py", title = "Uber in NY", icon = "👋"),
        # st.Page("app_pages/test_2.py",title = "Backtest", icon = "📈"),
        # st.Page("app_pages/test_3.py",title = "Risk Management", icon = "🌍"),
        st.Page("app_pages/strat_display.py",title = "Backtest", icon = "📈"),
        st.Page("app_pages/compare.py", title="Compare Strategies", icon="⚖️"),
    ]
)

page.run()