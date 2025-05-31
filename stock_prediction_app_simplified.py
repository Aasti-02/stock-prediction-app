import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, YearLocator, MonthLocator

# Set page configuration for a wide, modern layout
st.set_page_config(page_title="Stock Price Predictor", layout="wide")

# Custom CSS for a pastel color theme and hamburger menu
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&family=Roboto:wght@300;400;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #f0e6ef 0%, #b8d8d8 100%);
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 0 15px rgba(135, 182, 194, 0.3);
    }
    h1 {
        font-family: 'Poppins', sans-serif;
        color: #7a9eb1;
        text-shadow: 0 0 8px rgba(135, 182, 194, 0.5);
        font-size: 2.5em;
        text-align: center;
    }
    h2 {
        font-family: 'Poppins', sans-serif;
        color: #ef959d;
        text-shadow: 0 0 6px rgba(239, 149, 157, 0.5);
        font-size: 1.7em;
    }
    .stSelectbox, .stNumberInput > div > div > input {
        background-color: #ffffff;
        border: 2px solid #b8d8d8;
        border-radius: 10px;
        color: #4a4a4a;
        font-family: 'Roboto', sans-serif;
        padding: 8px;
        transition: border-color 0.3s;
    }
    .stSelectbox:hover, .stNumberInput > div > div > input:hover {
        border-color: #ef959d;
    }
    .stButton > button {
        background: linear-gradient(45deg, #ef959d, #b8d8d8);
        color: #ffffff;
        border: none;
        border-radius: 10px;
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        padding: 10px 25px;
        transition: transform 0.3s, box-shadow 0.3s;
        box-shadow: 0 0 10px rgba(135, 182, 194, 0.4);
    }
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 20px rgba(239, 149, 157, 0.6);
    }
    .stSuccess {
        background: linear-gradient(45deg, #a8e6cf, #dcedc1);
        border-radius: 10px;
        padding: 12px;
        color: #4a4a4a;
        font-family: 'Roboto', sans-serif;
        box-shadow: 0 0 10px rgba(168, 230, 207, 0.4);
        text-align: center;
    }
    .stMarkdown p, .stMarkdown li {
        font-family: 'Roboto', sans-serif;
        color: #4a4a4a;
        font-size: 0.9em;
    }
    .stMarkdown strong {
        color: #7a9eb1;
        font-weight: 700;
    }
    /* Hamburger menu styling */
    .hamburger {
        cursor: pointer;
        padding: 8px;
        display: flex;
        flex-direction: column;
        justify-content: space-around;
        width: 25px;
        height: 20px;
        background: transparent;
        border: none;
    }
    .hamburger span {
        width: 100%;
        height: 3px;
        background: #ef959d;
        border-radius: 2px;
        transition: all 0.3s ease;
        box-shadow: 0 0 5px rgba(239, 149, 157, 0.5);
    }
    .hamburger:hover span {
        background: #b8d8d8;
        box-shadow: 0 0 8px rgba(184, 216, 216, 0.7);
    }
    .sidebar-content {
        background: #ffffff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 0 10px rgba(135, 182, 194, 0.3);
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Hamburger menu toggle
if 'show_stock_info' not in st.session_state:
    st.session_state.show_stock_info = False

# Hamburger menu button in sidebar
with st.sidebar:
    if st.button("☰", key="hamburger"):
        st.session_state.show_stock_info = not st.session_state.show_stock_info
    
    if st.session_state.show_stock_info:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.header("Stock Market Basics")
        st.markdown("""
            - **What is a stock?**  
              A stock represents ownership in a company, giving you a share of its profits and growth.
            - **What is the stock market?**  
              A marketplace (e.g., NYSE, Nasdaq) where stocks are bought and sold based on supply and demand.
            - **What determines stock prices?**  
              Prices are driven by company performance, market demand, and economic factors.
            - **What is a bull market?**  
              A period when stock prices rise, reflecting investor optimism.
            - **What is a bear market?**  
              A period when stock prices fall, indicating investor pessimism.
            - **What are dividends?**  
              Payments from a company’s profits to shareholders, often quarterly.
            - **What is trading?**  
              Buying and selling stocks to profit from price changes, including day trading for short-term gains.
            - **What are risk and return?**  
              Stocks offer high potential returns but carry risks; diversification helps manage risk.
            - **What are market indices?**  
              Metrics like the S&P 500 or Dow Jones track a group of stocks to show market trends.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

st.title("Stock Price Predictor")

@st.cache_data
def load_data():
    df = pd.read_csv('stocks.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df['MA10'] = df.groupby('Ticker')['Close'].rolling(window=10).mean().reset_index(0, drop=True)
    df['Return'] = df.groupby('Ticker')['Close'].pct_change()
    df = df.dropna()
    return df

def create_features(data, N):
    X, y = [], []
    for i in range(N, len(data)):
        X.append(np.concatenate([
            data['Close'].values[i-N:i],
            data['MA10'].values[i-N:i],
            data['Return'].values[i-N:i]
        ]))
        y.append(data['Close'].values[i])
    return np.array(X), np.array(y)

# Load data and train models
df = load_data()
models = {}
tickers = ['AAPL', 'MSFT', 'NFLX', 'GOOG']
for ticker in tickers:
    ticker_df = df[df['Ticker'] == ticker].copy()
    X, y = create_features(ticker_df, 5)
    model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=2, random_state=42)
    model.fit(X, y)
    models[ticker] = model

# Plotting section with a pastel-styled graph
st.header("Stock Price History")
ticker = st.selectbox("Choose a company to view historical prices:", tickers, key="plot_ticker")
ticker_df = df[df['Ticker'] == ticker].copy()

# Custom Matplotlib style for a pastel look
plt.rcParams.update({
    'axes.facecolor': '#f0e6ef',
    'figure.facecolor': '#ffffff',
    'axes.edgecolor': '#7a9eb1',
    'axes.labelcolor': '#7a9eb1',
    'xtick.color': '#7a9eb1',
    'ytick.color': '#7a9eb1',
    'text.color': '#7a9eb1',
    'grid.color': '#b8d8d8',
    'grid.linestyle': ':',
    'grid.linewidth': 0.7,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Roboto', 'Arial', 'sans-serif']
})

fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(ticker_df['Date'], ticker_df['Close'], label=f'{ticker} Closing Price', 
        color='#ef959d', linewidth=3.5)
ax.set_title(f'{ticker} Stock Price Over Time', fontsize=16, fontweight='bold', fontfamily='Poppins')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Closing Price ($)', fontsize=12)

# Dynamic date formatting for clear year visibility
date_range = (ticker_df['Date'].max() - ticker_df['Date'].min()).days
if date_range > 365 * 5:
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(MonthLocator())
elif date_range > 365:
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(MonthLocator())
    ax.xaxis.set_minor_formatter(DateFormatter('%b'))
else:
    ax.xaxis.set_major_locator(MonthLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%b %Y'))

ax.tick_params(axis='x', labelsize=10, rotation=45)
ax.tick_params(axis='y', labelsize=10)
ax.grid(True, which='both')
ax.legend(facecolor='#ffffff', edgecolor='#b8d8d8', fontsize=10, loc='upper left')

# Pastel-styled annotations
for i, (date, price) in enumerate(zip(ticker_df['Date'], ticker_df['Close'])):
    if i % (len(ticker_df) // 8) == 0:
        ax.annotate(f'${price:.2f}\n{date.strftime("%Y-%m-%d")}',
                    xy=(date, price), xytext=(0, 20),
                    textcoords='offset points', ha='center', fontsize=8,
                    color='#4a4a4a',
                    bbox=dict(boxstyle='round,pad=0.4', fc='#b8d8d8', ec='#7a9eb1', alpha=0.9))

plt.tight_layout()
st.pyplot(fig)

# Prediction section
st.header("Predict Next Day's Closing Price")
ticker = st.selectbox("Choose a company for prediction:", tickers, key="predict_ticker")
st.write(f"Enter the last 5 days of {ticker} closing prices.")

close_prices = []
for i in range(5):
    close = st.number_input(f"Day {i+1} Close Price", min_value=0.0, value=100.0, step=0.01, key=f"close_{ticker}_{i}")
    close_prices.append(close)

if len(close_prices) == 5 and st.button("Predict Next Day's Closing Price", key=f"predict_{ticker}"):
    close_series = pd.Series(close_prices)
    ma10_values = close_series.rolling(window=5).mean().iloc[-5:].tolist()
    returns = close_series.pct_change().iloc[-5:].fillna(0).tolist()
    input_data = np.concatenate([close_prices, ma10_values, returns]).reshape(1, -15)
    prediction = models[ticker].predict(input_data)
    st.success(f"Predicted {ticker} Closing Price for the next day: ${prediction[0]:.2f}")
