# %%
import streamlit as st
import pandas as pd
import numpy as np

# Create a title
st.title("Stategy mastermind")


DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
         'streamlit-demo-data/uber-raw-data-sep14.csv.gz')
# cache data so dont have to redownload everytime page reloads
@st.cache_data

def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
data = load_data(10000)
# Notify the reader that the data was successfully loaded.
data_load_state.text("Done! (using st.cache_data)")

# The sidebar is on the left side of the app
st.sidebar.header("Test multiple pages")
st.sidebar.selectbox("Strategies", ["SMA Crossover", "Moving average"])


# Print out the raw data to the app
# st.subheader('Raw data')
# st.write(data)      # Can render almost anything passing to it
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)


# Use st.columns when you know exactly which sections go side by side, 
# and st.grid when you have a list of similar items to lay out.
st.subheader("Use grid")
grid = st.grid(min_column_width=200)
for i in range(6):
    grid.metric(f"Metric {i + 1}", f"{i * 10}%", border=True)

grid = st.grid(3)
with grid.cell(column_span="all"):
    st.markdown("**Featured**")
    st.caption("Trailing 30 days")
grid.metric("Revenue", "$1.2M", "+8%")
grid.metric("Pipeline", "$4.8M", "+12%")
grid.metric("Retention", "96%", "-0.4%")

st.subheader("Use columns")
cols = st.columns(6, wrap=False)
for i, col in enumerate(cols, start=1):
    col.button(f"{i}")



# Draw a histogram
st.subheader('Number of pickups by hour')
hist_values = np.histogram(
    data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]

st.bar_chart(hist_values)

#plot data on a map
# st.subheader('Map of all pickups')
# st.map(data)
# hour_to_filter = 17
hour_to_filter = st.slider('hour', 0, 23, 17)  # min: 0h, max: 23h, default: 17h
filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
st.subheader(f'Map of all pickups at {hour_to_filter}:00')
st.map(filtered_data)

