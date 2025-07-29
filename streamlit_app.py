# app.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from flask import Flask, render_template, request, send_file
import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

app = Flask(__name__)

MODEL_PATH = "model.h5"
SCALER_PATH = "scaler.pkl"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form['ticker'].upper()
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    end_date = datetime.today()
    start_date = end_date - timedelta(days=5*365)
    df = yf.download(ticker, start=start_date, end=end_date)[["Close"]]

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

    # Plotting with hammer line
    plt.figure(figsize=(10, 5))
    plt.plot(df.index[-100:], df["Close"].tail(100), label='Past 100 Days')
    plt.plot(future_dates, predictions, color='red', label='Predicted Next 30 Days')
    plt.axhline(predictions[-1], color='black', linestyle='--', label='Hammer Line')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'{ticker} - Price Prediction')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('static/prediction.png')
    plt.close()

    # Save prediction CSV
    prediction_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': predictions.flatten()})
    prediction_df.to_csv('static/predictions.csv', index=False)

    return render_template('index.html', prediction_image='static/prediction.png', ticker=ticker)

@app.route('/download')
def download():
    return send_file('static/predictions.csv', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
