import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, YearLocator, MonthLocator

# Set page configuration for a wide, professional layout
st.set_page_config(page_title="Stock Price Predictor 📈", layout="wide")

# Custom CSS for a neutral gray and white theme
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&family=Roboto:wght@300;400;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #f5f5f5 0%, #ffffff 100%);
        padding: 40px;
        border-radius: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
    }
    h1 {
        font-family: 'Poppins', sans-serif;
        color: #424242;
        font-size: 2.8em;
        font-weight: 600;
        text-align: center;
        margin-bottom: 30px;
    }
    h2 {
        font-family: 'Poppins', sans-serif;
        color: #424242;
        font-size: 1.8em;
        font-weight: 500;
        margin-top: 30px;
        margin-bottom: 15px;
    }
    .stSelectbox, .stNumberInput > div > div > input {
        background-color: #ffffff;
        border: 2px solid #e0e0e0;
        border-radius: 12px;
        color: #2e2e2e;
        font-family: 'Roboto', sans-serif;
        font-size: 0.95em;
        padding: 10px;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    .stSelectbox:hover, .stNumberInput > div > div > input:hover {
        border-color: #bdbdbd;
        box-shadow: 0 2px 10px rgba(189, 189, 189, 0.2);
    }
    .stButton > button {
        background: linear-gradient(45deg, #e0e0e0, #ffffff);
        color: #424242;
        border: none;
        border-radius: 12px;
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        font-size: 0.95em;
        padding: 12px 30px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(189, 189, 189, 0.3);
    }
    .stSuccess {
        background: linear-gradient(45deg, #90caf9, #bbdefb);
        border-radius: 12px;
        padding: 15px;
        color: #2e2e2e;
        font-family: 'Roboto', sans-serif;
        font-size: 0.95em;
        box-shadow: 0 2px 10px rgba(144, 202, 249, 0.3);
        text-align: center;
        margin-top: 20px;
    }
    .stMarkdown p, .stMarkdown li {
        font-family: 'Roboto', sans-serif;
        color: #ffffff;
        font-size: 0.9em;
        line-height: 1.5;
    }
    .stMarkdown strong {
        color: #ffffff;
        font-weight: 700;
    }
    .stock-basics-container {
        background: #f5f5f5;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        margin-top: 30px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Stock Price Predictor 📈")

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

# Plotting section with a serious, standard graph
st.header("Stock Price History 📅")
ticker = st.selectbox("Choose a company to view historical prices:", tickers, key="plot_ticker")
ticker_df = df[df['Ticker'] == ticker].copy()

# Standard Matplotlib style for a serious look
plt.rcParams.update({
    'axes.facecolor': '#ffffff',
    'figure.facecolor': '#ffffff',
    'axes.edgecolor': '#000000',
    'axes.labelcolor': '#000000',
    'xtick.color': '#000000',
    'ytick.color': '#000000',
    'text.color': '#000000',
    'grid.color': '#b0bec5',
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Roboto', 'Arial', 'sans-serif']
})

fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(ticker_df['Date'], ticker_df['Close'], label=f'{ticker} Closing Price', 
        color='#1565c0', linewidth=2.5)
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
ax.legend(facecolor='#ffffff', edgecolor='#b0bec5', fontsize=10, loc='upper left')

# Standard annotations for clarity
for i, (date, price) in enumerate(zip(ticker_df['Date'], ticker_df['Close'])):
    if i % (len(ticker_df) // 8) == 0:
        ax.annotate(f'${price:.2f}\n{date.strftime("%Y-%m-%d")}',
                    xy=(date, price), xytext=(0, 20),
                    textcoords='offset points', ha='center', fontsize=8,
                    color='#000000',
                    bbox=dict(boxstyle='round,pad=0.4', fc='#e3f2fd', ec='#1565c0', alpha=0.9))

plt.tight_layout()
st.pyplot(fig)

# Prediction section
st.header("Predict Next Day's Closing Price 🔮")
ticker = st.selectbox("Choose a company for prediction:", tickers, key="predict_ticker")
st.write(f"Enter the last 5 days of {ticker} closing prices.")

close_prices = []
for i in range(5):
    close = st.number_input(f"Day {i+1} Close Price", min_value=0.0, value=100.0, step=0.01, key=f"close_{ticker}_{i}")
    close_prices.append(close)

if len(close_prices) == 5 and st.button("Predict Next Day's Closing Price 🚀", key=f"predict_{ticker}"):
    close_series = pd.Series(close_prices)
    ma10_values = close_series.rolling(window=5).mean().iloc[-5:].tolist()
    returns = close_series.pct_change().iloc[-5:].fillna(0).tolist()
    input_data = np.concatenate([close_prices, ma10_values, returns]).reshape(1, -15)
    prediction = models[ticker].predict(input_data)
    st.success(f"Predicted {ticker} Closing Price for the next day: ${prediction[0]:.2f} 🎉")

# Stock Market Basics section moved to main page
st.header("Stock Market Basics 📚")
st.markdown('<div class="stock-basics-container">', unsafe_allow_html=True)
st.markdown("""
    - **What is a stock?**  
      A stock represents ownership in a company, giving you a share of its profits and growth.
    - **What is the stock market?**  
      A marketplace (e.g., NYSE, Nasdaq) where stocks are bought and sold based on supply and demand.
    - **What determines stock prices?**  
      Prices are driven by company performance, market demand, and economic factors.
    - **What is a bull market?**  
      A period when stock prices rise, reflecting investor optimism 🐂.
    - **What is a bear market?**  
      A period when stock prices fall, indicating investor pessimism 🐻.
    - **What are dividends?**  
      Payments from a company’s profits to shareholders, often quarterly.
    - **What is trading?**  
      Buying and selling stocks to profit from price changes, including day trading for short-term gains.
    - **What are risk and return?**  
      Stocks offer high potential returns but carry risks; diversification helps manage risk.
    - **What are market indices?**  
      Metrics like the S&P 500 or Dow Jones track a group of stocks to show market trends 📊.
""")
st.markdown('</div>', unsafe_allow_html=True)
