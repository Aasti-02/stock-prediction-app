import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, YearLocator, MonthLocator
import seaborn as sns

# Set page configuration for a modern layout
st.set_page_config(page_title="Stock Price Predictor", layout="wide")

# Custom CSS for a cool, modern look
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
    
    .main {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 20px;
        border-radius: 10px;
    }
    h1, h2, h3 {
        font-family: 'Poppins', sans-serif;
        color: #ffffff;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    .stSelectbox, .stNumberInput > div > div > input {
        background-color: #ffffff;
        border-radius: 8px;
        border: 1px solid #4CAF50;
        font-family: 'Poppins', sans-serif;
        color: #333;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        padding: 10px 20px;
        transition: background-color 0.3s;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .stSuccess {
        background-color: #2ecc71;
        border-radius: 8px;
        padding: 10px;
        color: white;
        font-family: 'Poppins', sans-serif;
    }
    .stMarkdown p {
        font-family: 'Poppins', sans-serif;
        color: #e0e0e0;
    }
    </style>
""", unsafe_allow_html=True)

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

# Plotting section with a cool graph
st.header("Stock Price History")
ticker = st.selectbox("Choose a company to view historical prices:", tickers, key="plot_ticker")
ticker_df = df[df['Ticker'] == ticker].copy()

# Use seaborn style for a modern look
plt.style.use('seaborn')
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(ticker_df['Date'], ticker_df['Close'], label=f'{ticker} Closing Price', color='#4CAF50', linewidth=2)
ax.set_title(f'{ticker} Stock Price Over Time', fontsize=16, fontweight='bold', color='#ffffff')
ax.set_xlabel('Date', fontsize=12, color='#ffffff')
ax.set_ylabel('Closing Price ($)', fontsize=12, color='#ffffff')

# Dynamic date formatting
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

# Customize axes and grid
ax.tick_params(axis='x', colors='#e0e0e0', labelsize=10, rotation=45)
ax.tick_params(axis='y', colors='#e0e0e0', labelsize=10)
ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='#e0e0e0')
ax.set_facecolor('#2a5298')
fig.patch.set_facecolor('#1e3c72')
ax.legend(facecolor='#ffffff', edgecolor='#4CAF50', fontsize=10)

# Add stylish annotations
for i, (date, price) in enumerate(zip(ticker_df['Date'], ticker_df['Close'])):
    if i % (len(ticker_df) // 8) == 0:
        ax.annotate(f'${price:.2f}\n{date.strftime("%Y-%m-%d")}',
                    xy=(date, price), xytext=(0, 15),
                    textcoords='offset points', ha='center', fontsize=9,
                    color='#ffffff',
                    bbox=dict(boxstyle='round,pad=0.3', fc='#4CAF50', ec='#ffffff', alpha=0.8))

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
