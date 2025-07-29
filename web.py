import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Load model and scaler
MODEL_PATH = "deploy_Stock/model.h5"
SCALER_PATH = "deploy_Stock/scaler.pkl"

@st.cache_resource
def load_lstm_model():
    return load_model(MODEL_PATH)

@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)

model = load_lstm_model()
scaler = load_scaler()

# App title
st.title("📈 Stock Price Predictor (LSTM Model)")
st.markdown("Enter a stock ticker symbol (e.g., `AAPL`, `GOOGL`, `TSLA`) to predict the next 30 days.")

# Input
ticker = st.text_input("Enter Stock Ticker:", value="AAPL").upper()

if st.button("Predict"):
    try:
        # Fetch 5 years data
        end_date = datetime.today()
        start_date = end_date - timedelta(days=5*365)
        df = yf.download(ticker, start=start_date, end=end_date)[["Close"]]

        if df.empty or len(df) < 60:
            st.error("Not enough data found. Please try a different ticker.")
        else:
            # Prepare input
            last_60 = df[-60:].values
            scaled = scaler.transform(last_60)
            X_input = np.reshape(scaled, (1, 60, 1))

            predictions = []
            current_input = X_input

            for _ in range(30):
                pred = model.predict(current_input, verbose=0)
                predictions.append(pred[0, 0])
                current_input = np.append(current_input[:, 1:, :], [[[pred[0, 0]]]], axis=1)

            predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
            future_dates = [df.index[-1] + timedelta(days=i) for i in range(1, 31)]

            # Plot
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df.index[-100:], df["Close"].tail(100), label='Past 100 Days')
            ax.plot(future_dates, predictions, color='red', label='Predicted Next 30 Days')
            ax.axhline(predictions[-1], color='black', linestyle='--', label='Hammer Line')
            ax.set_title(f"{ticker} - Price Prediction")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)

            # Save predictions
            prediction_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': predictions.flatten()})
            prediction_df.to_csv("predictions.csv", index=False)
            st.success("Prediction completed!")

            with open("predictions.csv", "rb") as file:
                st.download_button("📥 Download Predictions CSV", data=file, file_name="predictions.csv")

    except Exception as e:
        st.error(f"An error occurred: {e}")
