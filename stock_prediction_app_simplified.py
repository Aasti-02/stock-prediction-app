import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, YearLocator, MonthLocator

# Set page configuration for a wide, modern layout
st.set_page_config(page_title="Neon Stock Predictor", layout="wide")

# Custom CSS for a vibrant, cyberpunk look
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Roboto:wght@300;400;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #0a0e1a 0%, #1e2a47 100%);
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 0 25px rgba(0, 255, 255, 0.4);
    }
    h1 {
        font-family: 'Orbitron', sans-serif;
        color: #00f7ff;
        text-shadow: 0 0 12px #00f7ff, 0 0 24px #00f7ff;
        font-size: 2.8em;
        text-align: center;
    }
    h2 {
        font-family: 'Orbitron', sans-serif;
        color: #ff00a1;
        text-shadow: 0 0 10px #ff00a1;
        font-size: 1.9em;
    }
    .stSelectbox, .stNumberInput > div > div > input {
        background-color: #1e2a47;
        border: 2px solid #00f7ff;
        border-radius: 12px;
        color: #ffffff;
        font-family: 'Roboto', sans-serif;
        padding: 10px;
        transition: border-color 0.3s;
    }
    .stSelectbox:hover, .stNumberInput > div > div > input:hover {
        border-color: #ff00a1;
    }
    .stButton > button {
        background: linear-gradient(45deg, #ff00a1, #00f7ff);
        color: #ffffff;
        border: none;
        border-radius: 12px;
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        padding: 12px 30px;
        transition: transform 0.3s, box-shadow 0.3s;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.6);
    }
    .stButton > button:hover {
        transform: scale(1.1);
        box-shadow: 0 0 25px rgba(255, 0, 161, 0.8);
    }
    .stSuccess {
        background: linear-gradient(45deg, #2ecc71, #27ae60);
        border-radius: 12px;
        padding: 15px;
        color: #ffffff;
        font-family: 'Roboto', sans-serif;
        box-shadow: 0 0 15px rgba(46, 204, 113, 0.6);
        text-align: center;
    }
    .stMarkdown p {
        font-family: 'Roboto', sans-serif;
        color: #d0d8ff;
        font-size: 1.2em;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Neon Stock Price Predictor")

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

# Plotting section with a vibrant graph
st.header("Stock Price History")
ticker = st.selectbox("Choose a company to view historical prices:", tickers, key="plot_ticker")
ticker_df = df[df['Ticker'] == ticker].copy()

# Custom Matplotlib style for a neon, futuristic look
plt.rcParams.update({
    'axes.facecolor': '#0a0e1a',
    'figure.facecolor': '#1e2a47',
    'axes.edgecolor': '#00f7ff',
    'axes.labelcolor': '#00f7ff',
    'xtick.color': '#00f7ff',
    'ytick.color': '#00f7ff',
    'text.color': '#00f7ff',
    'grid.color': '#00f7ff',
    'grid.linestyle': ':',
    'grid.linewidth': 0.8,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Roboto', 'Arial', 'sans-serif']
})

fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(ticker_df['Date'], ticker_df['Close'], label=f'{ticker} Closing Price', 
        color='#ff00a1', linewidth=4)
ax.set_title(f'{ticker} Stock Price Over Time', fontsize=18, fontweight='bold', fontfamily='Orbitron')
ax.set_xlabel('Date', fontsize=14)
ax.set_ylabel('Closing Price ($)', fontsize=14)

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

ax.tick_params(axis='x', labelsize=12, rotation=45)
ax.tick_params(axis='y', labelsize=12)
ax.grid(True, which='both')
ax.legend(facecolor='#1e2a47', edgecolor='#00f7ff', fontsize=12, loc='upper left')

# Neon-styled annotations
for i, (date, price) in enumerate(zip(ticker_df['Date'], ticker_df['Close'])):
    if i % (len(ticker_df) // 8) == 0:
        ax.annotate(f'${price:.2f}\n{date.strftime("%Y-%m-%d")}',
                    xy=(date, price), xytext=(0, 25),
                    textcoords='offset points', ha='center', fontsize=10,
                    color='#ffffff',
                    bbox=dict(boxstyle='round,pad=0.4', fc='#ff00a1', ec='#00f7ff', alpha=0.9))

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
