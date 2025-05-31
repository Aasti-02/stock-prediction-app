import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.title("Stock Price Prediction for AAPL, MSFT, NFLX, GOOG")

@st.cache_data
def load_data():
    df = pd.read_csv('stocks.csv')
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

df = load_data()
models = {}
tickers = ['AAPL', 'MSFT', 'NFLX', 'GOOG']
for ticker in tickers:
    ticker_df = df[df['Ticker'] == ticker].copy()
    X, y = create_features(ticker_df, 5)
    model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=2, random_state=42)
    model.fit(X, y)
    models[ticker] = model

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ticker_df = df[df['Ticker'] == ticker]
ax.plot(ticker_df['Date'], ticker_df['Close'])
st.pyplot(fig)

st.header("Select a Company and Enter Closing Prices")
ticker = st.selectbox("Choose a company:", tickers)
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
