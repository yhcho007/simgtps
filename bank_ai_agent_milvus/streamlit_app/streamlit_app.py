"""Streamlit dashboard for visualizing interaction metrics.
- Reads SQLite DB and displays daily counts, satisfaction histogram, and recent low-feedback items.
- Intended to run inside an internal network for staff monitoring.
"""
import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

DB = 'backend/data/metrics.db'

st.title('Bank AI Agent - Operations Dashboard')

@st.cache_data
def load_data():
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query('SELECT * FROM interactions', conn, parse_dates=['timestamp'])
    conn.close()
    return df

try:
    df = load_data()
    st.header('Overview')
    st.write('Total interactions:', len(df))
    # daily counts
    df['date'] = df['timestamp'].dt.date
    daily = df.groupby('date').size()
    st.line_chart(daily)

    st.header('Satisfaction distribution')
    fig, ax = plt.subplots()
    df['satisfaction'].dropna().astype(int).hist(bins=10, ax=ax)
    st.pyplot(fig)

    st.header('Low-rated interactions (<=3)')
    low = df[df['satisfaction']<=3].sort_values('timestamp', ascending=False).head(20)
    st.dataframe(low[['timestamp','session_id','user','query','reply','satisfaction','feedback']])
except Exception as e:
    st.error('Error loading DB: ' + str(e))
