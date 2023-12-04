# Name: João Pedro Correa Crozariolo
# Student ID: 2370557

# Import necessary libraries
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Global Definitions for ease of modification
START_DATE = '2015-01-01'                                               # Start date for historical data
END_DATE = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')    # End date set to yesterday by default
EPOCHS = 250                                                            # Number of epochs for training
BATCH_SIZE = 32                                                         # Batch size for training
SEQUENCE_LENGTH = 120                                                   # Number of days to use in prediction
SPLIT_RATIO = 0.8                                                       # Proportion to split data into training and testing sets

def print_last_week_prices(company_code):
    # Fetch stock market data for the last week
    data = yf.download(company_code, start=START_DATE, end=END_DATE)
    last_week_prices = data['Close'].tail(7)

    print(f"Prices for the last week for {company_code}:")
    for date, price in last_week_prices.iteritems():
        print(f"{date.date()}: {price:.2f}")

# Function to execute the prediction model
def run_prediction_model(company_code):
    # Print prices for the last week
    print_last_week_prices(company_code)

    # Fetch stock market data
    data = yf.download(company_code, start=START_DATE, end=END_DATE)
    data = data['Close'].values.reshape(-1, 1)

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Create a dataset with X as stock prices and Y as the price of the next day
    X, y = [], []
    for i in range(SEQUENCE_LENGTH, len(scaled_data)):
        X.append(scaled_data[i-SEQUENCE_LENGTH:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Split the data into training and testing sets
    split = int(X.shape[0] * SPLIT_RATIO)
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

    # Build the LSTM model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(SEQUENCE_LENGTH, 1)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(units=50))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(units=1))

    # Compile and train the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # Prepare the last sequence of known values as input for the model
    last_sequence = scaled_data[-SEQUENCE_LENGTH:]
    last_sequence = np.reshape(last_sequence, (1, SEQUENCE_LENGTH, 1))

    # Predict the price for the next 7 days
    print(f"Predictions for the next 7 days for {company_code}:")
    for i in range(7):
        next_day_price = model.predict(last_sequence)
        print(f"Day {i+1}: {scaler.inverse_transform(next_day_price)[0][0]}")

        # Update the last sequence with the most recent prediction
        last_sequence = np.append(last_sequence[:,1:,:], [next_day_price], axis=1)

    # Predict stock prices in the testing set
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # Get real prices from the testing set
    true_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate RMSE
    rmse = sqrt(mean_squared_error(true_prices, predicted_prices))
    print(f"RMSE for {company_code}: {rmse:.2f}")

    # Generate dates for the x-axis of the graph
    dates = pd.to_datetime(yf.download(company_code, start=START_DATE, end=END_DATE).index)
    test_dates = dates[-len(true_prices):]

    # Plot the graph of real prices vs. predicted prices
    plt.figure(figsize=(14, 5))
    plt.plot(test_dates, true_prices, color='blue', label=f'Real stock price of {company_code}')
    plt.plot(test_dates, predicted_prices, color='red', label=f'Predicted stock price of {company_code}')
    plt.title(f'Stock Price Prediction for {company_code}')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

# List of predefined companies
companyTicker = ['PBR', 'VALE', 'ITUB', 'BBD', 'ABEV']
companyName = ['Petrobras', 'Vale', 'Itaú Unibanco', 'Banco Bradesco', 'Ambev']

# Execute predictions for each predefined company
for i in range(len(companyTicker)):
    print(f"Running predictions for {companyName[i]} ({companyTicker[i]})...")
    run_prediction_model(companyTicker[i])

# Ask the user if they want to make a prediction for an additional company
user_choice = input("Would you like to make a prediction for another company? (yes/no): ").lower()
if user_choice == 'yes':
    extra_company_name = input("Enter the company name: ")
    extra_company_ticker = input("Enter the stock symbol of the company: ")
    print(f"Running predictions for {extra_company_name} ({extra_company_ticker})...")
    run_prediction_model(extra_company_ticker)