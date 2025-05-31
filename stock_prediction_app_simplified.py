import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, YearLocator, MonthLocator, AutoDateLocator

st.title("Stock Price Prediction for AAPL, MSFT, NFLX, GOOG")

@st.cache_data
def load_data():
    df = pd.read_csv('stocks.csv')
    df['Date'] = pd.to_datetime(df['Date'])  # Ensure 'Date' column is in datetime format
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

# Plotting the stock price with clear year and date information
st.header("Stock Price History")
ticker = st.selectbox("Choose a company to view historical prices:", tickers, key="plot_ticker")
ticker_df = df[df['Ticker'] == ticker].copy()

fig, ax = plt.subplots(figsize=(12, 6))  # Larger figure for better visibility
ax.plot(ticker_df['Date'], ticker_df['Close'], label=f'{ticker} Closing Price', color='blue')
ax.set_title(f'{ticker} Stock Price Over Time', fontsize=14)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Closing Price ($)', fontsize=12)

# Dynamic date formatting based on data range
date_range = (ticker_df['Date'].max() - ticker_df['Date'].min()).days
if date_range > 365 * 5:  # If data spans more than 5 years, show years
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(MonthLocator())
elif date_range > 365:  # If data spans 1-5 years, show year and month
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(MonthLocator())
    ax.xaxis.set_minor_formatter(DateFormatter('%b'))
else:  # If data spans less than a year, show months and days
    ax.xaxis.set_major_locator(AutoDateLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%b %Y'))

plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate and align labels
ax.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add grid
ax.legend()

# Optional: Add hover-like annotation in Streamlit (simulated with a tooltip-like text)
for i, (date, price) in enumerate(zip(ticker_df['Date'], ticker_df['Close'])):
    if i % (len(ticker_df) // 10) == 0:  # Annotate every 10th point to avoid clutter
        ax.annotate(f'{price:.2f}\n{date.strftime("%Y-%m-%d")}',
                    xy=(date, price), xytext=(0, 10),
                    textcoords='offset points', ha='center', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5))

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
