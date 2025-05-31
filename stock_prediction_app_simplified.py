import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
import os
import kaggle
import zipfile

st.title("Stock Price Prediction for AAPL, MSFT, NFLX, GOOG")

# Function to download and load Kaggle dataset
@st.cache_data
def load_kaggle_data():
    dataset = "jacksoncrow/stock-market-dataset"
    os.makedirs("kaggle_data", exist_ok=True)
    kaggle.api.dataset_download_files(dataset, path="kaggle_data", unzip=True)
    
    # Assuming dataset has a folder with stock CSV files
    data_files = os.listdir("kaggle_data/stocks")
    df_list = []
    tickers = ['AAPL', 'MSFT', 'NFLX', 'GOOG']
    
    for file in data_files:
        ticker = file.split('.')[0]
        if ticker in tickers:
            df = pd.read_csv(f"kaggle_data/stocks/{file}")
            df['Ticker'] = ticker
            df['Date'] = pd.to_datetime(df['Date'])
            df_list.append(df)
    
    df = pd.concat(df_list, ignore_index=True)
    
    # Add technical indicators
    for ticker in tickers:
        ticker_df = df[df['Ticker'] == ticker].copy()
        # RSI
        df.loc[df['Ticker'] == ticker, 'RSI'] = RSIIndicator(ticker_df['Close']).rsi()
        # MACD
        macd = MACD(ticker_df['Close'])
        df.loc[df['Ticker'] == ticker, 'MACD'] = macd.macd()
        df.loc[df['Ticker'] == ticker, 'MACD_Signal'] = macd.macd_signal()
        # Bollinger Bands
        bb = BollingerBands(ticker_df['Close'])
        df.loc[df['Ticker'] == ticker, 'BB_High'] = bb.bollinger_hband()
        df.loc[df['Ticker'] == ticker, 'BB_Low'] = bb.bollinger_lband()
        # Moving Average and Returns
        df.loc[df['Ticker'] == ticker, 'MA10'] = ticker_df['Close'].rolling(window=10).mean()
        df.loc[df['Ticker'] == ticker, 'Return'] = ticker_df['Close'].pct_change()
        df.loc[df['Ticker'] == ticker, 'Volatility'] = ticker_df['Close'].rolling(window=10).std()
    
    df = df.dropna()
    return df

# Feature creation with additional indicators
def create_features(data, N):
    X, y = [], []
    for i in range(N, len(data)):
        X.append(np.concatenate([
            data['Close'].values[i-N:i],
            data['MA10'].values[i-N:i],
            data['Return'].values[i-N:i],
            data['RSI'].values[i-N:i],
            data['MACD'].values[i-N:i],
            data['MACD_Signal'].values[i-N:i],
            data['BB_High'].values[i-N:i],
            data['BB_Low'].values[i-N:i],
            data['Volatility'].values[i-N:i],
            data['Volume'].values[i-N:i]  # Include volume
        ]))
        y.append(data['Close'].values[i])
    return np.array(X), np.array(y)

# Load data
df = load_kaggle_data()
models = {}
tickers = ['AAPL', 'MSFT', 'NFLX', 'GOOG']

# Train models
for ticker in tickers:
    ticker_df = df[df['Ticker'] == ticker].copy()
    X, y = create_features(ticker_df, 5)
    model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=2, random_state=42)
    model.fit(X, y)
    models[ticker] = model

# Plot historical closing prices
fig, ax = plt.subplots(figsize=(10, 5))
for ticker in tickers:
    ticker_df = df[df['Ticker'] == ticker]
    ax.plot(ticker_df['Date'], ticker_df['Close'], label=ticker)
ax.set_title("Historical Closing Prices")
ax.set_xlabel("Date")
ax.set_ylabel("Close Price")
ax.legend()
st.pyplot(fig)

# User input for prediction
st.header("Select a Company and Enter Data for Prediction")
ticker = st.selectbox("Choose a company:", tickers)
st.write(f"Enter the last 5 days of {ticker} data (Close Price, Volume).")

close_prices = []
volumes = []
for i in range(5):
    col1, col2 = st.columns(2)
    with col1:
        close = st.number_input(f"Day {i+1} Close Price", min_value=0.0, value=100.0, step=0.01, key=f"close_{ticker}_{i}")
    with col2:
        volume = st.number_input(f"Day {i+1} Volume", min_value=0, value=1000000, step=1000, key=f"volume_{ticker}_{i}")
    close_prices.append(close)
    volumes.append(volume)

# Calculate additional features for input
if len(close_prices) == 5 and st.button("Predict Next Day's Closing Price", key=f"predict_{ticker}"):
    close_series = pd.Series(close_prices)
    ma10_values = close_series.rolling(window=5).mean().iloc[-5:].tolist()
    returns = close_series.pct_change().iloc[-5:].fillna(0).tolist()
    rsi_values = RSIIndicator(pd.Series(close_prices)).rsi().iloc[-5:].fillna(0).tolist()
    macd = MACD(pd.Series(close_prices))
    macd_values = macd.macd().iloc[-5:].fillna(0).tolist()
    macd_signal = macd.macd_signal().iloc[-5:].fillna(0).tolist()
    bb = BollingerBands(pd.Series(close_prices))
    bb_high = bb.bollinger_hband().iloc[-5:].fillna(0).tolist()
    bb_low = bb.bollinger_lband().iloc[-5:].fillna(0).tolist()
    volatility = close_series.rolling(window=5).std().iloc[-5:].fillna(0).tolist()
    
    # Combine all features
    input_data = np.concatenate([
        close_prices,
        ma10_values,
        returns,
        rsi_values,
        macd_values,
        macd_signal,
        bb_high,
        bb_low,
        volatility,
        volumes
    ]).reshape(1, -1)
    
    # Predict
    prediction = models[ticker].predict(input_data)
    st.success(f"Predicted {ticker} Closing Price for the next day: ${prediction[0]:.2f}")
