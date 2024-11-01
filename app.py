import streamlit as st
import pandas as pd
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.models import load_model

import datetime
import joblib

# Load the trained model and scalers
model = load_model('best_model.h5')
scaler_price = joblib.load('scaler_price.pkl')

# Load historical data
df = pd.read_csv(r"D:\courses\diploma\NTI\technical\projects\gold anf dollars\gold.csv")
df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
df = df.dropna().sort_values(by="Date")
df.reset_index(drop=True, inplace=True)

# Prepare Streamlit layout
st.title("Gold Price Prediction")
st.write("Enter a future date to predict the gold price:")

# Input for future date
date_input = st.date_input("Select a date:", min_value=datetime.date.today(), max_value=datetime.date.today() + datetime.timedelta(days=365))

# Convert date input to datetime for processing
date_input = pd.to_datetime(date_input)

# Display historical prices
st.subheader("Historical Gold Prices")
st.line_chart(df.set_index("Date")["Price"])

# Define function to create input sequence for the prediction model
def prepare_future_input(last_data, date):
    future_days = (date - last_data['Date'].iloc[-1]).days
    if future_days <= 0:
        st.warning("Please select a date in the future.")
        return None

    # Use the last 60 entries of the feature columns only (excluding 'Date' and 'Price')
    input_sequence = last_data.drop(columns=['Date', 'Price']).tail(60).values
    input_sequence = scaler_price.transform(input_sequence)  # Assume price scaling is enough for input

    predictions = []
    for _ in range(future_days):
        prediction = model.predict(input_sequence.reshape(1, 60, input_sequence.shape[1]))
        input_sequence = np.append(input_sequence[1:], prediction, axis=0)  # Slide window
        predictions.append(scaler_price.inverse_transform(prediction)[0][0])  # Collect predictions

    return predictions

# Run prediction when the button is clicked
if st.button("Predict Price"):
    predicted_prices = prepare_future_input(df, date_input)
    if predicted_prices:
        for i, price in enumerate(predicted_prices):
            prediction_date = date_input + datetime.timedelta(days=i)
            st.write(f"Predicted Gold Price on {prediction_date.strftime('%Y-%m-%d')}: ${price:.2f}")
