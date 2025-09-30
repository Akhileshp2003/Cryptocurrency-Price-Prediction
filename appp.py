import streamlit as st
import pandas as pd
import yfinance as yf
from autots import AutoTS
import matplotlib.pyplot as plt
from datetime import date, timedelta

st.title("📈 Cryptocurrency Price Prediction")

# Define date range
today = date.today()
start_date = (today - timedelta(days=768)).strftime("%Y-%m-%d")
end_date = today.strftime("%Y-%m-%d")

# Download BTC-USD data from Yahoo Finance
data = yf.download('BTC-USD', start=start_date, end=end_date, progress=False)
data["Date"] = data.index
data = data[["Date", "Open", "High", "Low", "Close", "Volume"]]
data.set_index("Date", inplace=True)
data = data.select_dtypes(include=["number"])  # Keep only numeric columns

# Display data preview
st.write("### Downloaded BTC-USD Data Preview:")
st.dataframe(data.tail())

# Train AutoTS Model
st.write("### Training AutoTS Model...")
model = AutoTS(forecast_length=30, frequency="infer", ensemble="simple")
model = model.fit(data)

# Generate Predictions
prediction = model.predict()
forecast = prediction.forecast

# Display Forecast
st.write("### Forecast for Next 30 Days:")
st.dataframe(forecast)

# Plot Forecast
st.write("### Price Trend Prediction:")
plt.figure(figsize=(10, 5))
plt.plot(forecast.index, forecast.values, marker='o', linestyle='-')
plt.xlabel("Date")
plt.ylabel("Predicted Price")
plt.title("Cryptocurrency Price Forecast")
plt.xticks(rotation=45)
st.pyplot(plt)

st.write("🔍 BTC-USD data is automatically downloaded. No file upload needed.")
