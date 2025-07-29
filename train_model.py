# train_model.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import joblib

TICKER = "AAPL"
MODEL_PATH = "model.h5"             # ✅ updated
SCALER_PATH = "scaler.pkl"

def load_data(ticker):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=5*365)
    df = yf.download(ticker, start=start_date, end=end_date)
    return df[["Close"]]

def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

def train():
    df = load_data(TICKER)
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    X, y = create_dataset(df_scaled)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, epochs=10, batch_size=64, verbose=1)

    model.save(MODEL_PATH)                 # ✅ use .h5
    joblib.dump(scaler, SCALER_PATH)
    print("✅ Model and scaler saved.")

if __name__ == '__main__':
    train()
