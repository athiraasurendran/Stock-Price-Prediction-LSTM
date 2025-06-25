import numpy as np
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import date, timedelta
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

# Page Configuration
st.set_page_config(page_title="AI Stock Forecaster", layout="wide")
st.title("📈 AI Stock Forecaster (LSTM Models)")

# Model Directory Setup
MODEL_DIR = "stock_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Available Stocks and Models
STOCKS = {
    "Google (GOOG)": "GOOG",
    "Tesla (TSLA)": "TSLA",
    "Microsoft (MSFT)": "MSFT",
    "Apple (AAPL)": "AAPL",
    "Amazon (AMZN)": "AMZN",
    "Netflix (NFLX)": "NFLX",
    "Starbucks (SBUX)": "SBUX",
    "Johnson & Johnson (JNJ)": "JNJ",
    "Costco (COST)": "COST",
    "Coca-Cola (KO)": "KO",
    "Nestle (NESN.SW)": "NESN.SW",
    "Toyota (TM)": "TM",
    "LOreal (OR.PA)": "OR.PA",
    "Walt Disney (DIS)": "DIS",
    "American Express (AXP)": "AXP",
    "PepsiCo (PEP)": "PEP",
    "Procter & Gamble (PG)": "PG",
    "Saudi Aramco (2222.SR)": "2222.SR",
    "Roche (ROG.SW)": "ROG.SW",
    "Tencent (0700.HK)": "0700.HK",
    "Unilever (UL)": "UL",
    "Colgate-Palmolive (CL)": "CL",
    "McCormick (MKC)": "MKC",
    "Estee Lauder (EL)": "EL",
    "Mondelez (MDLZ)": "MDLZ",
}

# Sidebar Controls
with st.sidebar:
    st.header("Settings")
    selected_stock_name = st.selectbox("Select Stock", list(STOCKS.keys()))
    selected_stock = STOCKS[selected_stock_name]
    forecast_days = st.slider("Forecast Days", 7, 10, 7)
    st.markdown("---")
    st.markdown("🔍 *Each stock uses its own trained LSTM model*")


# Data Loading
@st.cache_data(ttl=3600)
def load_stock_data(ticker):
    end_date = date.today()
    start_date = end_date - timedelta(days=5 * 365)
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)

    # Ensure Date column is properly formatted
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date")

    return data


# Data Preprocessing
def prepare_data(data, n_steps=60):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1, 1))

    X = []
    for i in range(n_steps, len(scaled_data)):
        X.append(scaled_data[i - n_steps : i, 0])

    return np.array(X), scaler


# Model Loading
@st.cache_resource
def load_stock_model(stock_ticker):
    model_path = os.path.join(MODEL_DIR, f"{stock_ticker}_model.h5")
    if os.path.exists(model_path):
        try:
            return load_model(model_path, compile=False)
        except Exception as e:
            st.error(f"Failed to load model for {stock_ticker}: {e}")
            return None
    else:
        st.error(
            f"Model not found for {stock_ticker}! Please check the model directory."
        )
        return None


# Forecast Generation
def make_predictions(model, last_sequence, days, scaler):
    predictions = []
    current_seq = last_sequence.copy()

    for _ in range(days):
        next_pred = model.predict(
            current_seq.reshape(1, len(current_seq), 1), verbose=0
        )
        predictions.append(next_pred[0, 0])
        current_seq = np.roll(current_seq, -1)
        current_seq[-1] = next_pred

    predictions = np.array(predictions).reshape(-1, 1)
    return scaler.inverse_transform(predictions).flatten()


# Trend Analysis
def analyze_trend(predictions):
    changes = np.diff(predictions)
    uptrend_days = sum(changes > 0)
    trend_strength = uptrend_days / len(changes)

    if trend_strength > 0.7:
        return "Strong Uptrend 🚀", "#2ecc71"
    elif trend_strength > 0.55:
        return "Mild Uptrend 📈", "#27ae60"
    elif trend_strength > 0.45:
        return "Neutral ↔️", "#f39c12"
    elif trend_strength > 0.3:
        return "Mild Downtrend 📉", "#e74c3c"
    else:
        return "Strong Downtrend 🔻", "#c0392b"


# Main App Logic
data_load_state = st.empty()
data_load_state.text(f"Loading {selected_stock} data...")
stock_data = load_stock_data(selected_stock)

# Check if data loaded properly
if stock_data.empty:
    st.error("No data loaded for this stock. Please check your internet connection.")
    st.stop()

# Check for required columns
required_columns = ["Date", "Open", "High", "Low", "Close"]
if not all(col in stock_data.columns for col in required_columns):
    st.error("Missing required columns in stock data")
    st.write("Available columns:", stock_data.columns.tolist())
    st.stop()

# Check for NaN values
if stock_data.isnull().values.any():
    st.warning("Data contains missing values - filling with previous values")
    stock_data.fillna(method="ffill", inplace=True)

data_load_state.empty()

# Display Historical Chart
st.subheader(f"{selected_stock_name} Historical Data")
col1, col2 = st.columns([3, 1])
with col1:
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=stock_data["Date"],
            open=stock_data["Open"],
            high=stock_data["High"],
            low=stock_data["Low"],
            close=stock_data["Close"],
            name="Price",
        )
    )
    fig.update_layout(
        height=500,
        xaxis_rangeslider_visible=False,
        title=f"{selected_stock_name} Historical Price",
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    latest_close = stock_data["Close"].iloc[-1]
    st.metric("Current Price", f"${float(latest_close):.2f}")
    st.metric("52W High", f"${float(stock_data['High'].max()):.2f}")
    st.metric("52W Low", f"${float(stock_data['Low'].min()):.2f}")

# Forecasting Section
st.subheader("AI Price Forecast")
model = load_stock_model(selected_stock)

if model:
    X, scaler = prepare_data(stock_data)
    last_sequence = X[-1].flatten()

    with st.spinner(f"Generating {forecast_days}-day forecast..."):
        predictions = make_predictions(model, last_sequence, forecast_days, scaler)
        forecast_dates = [
            stock_data["Date"].iloc[-1] + timedelta(days=i)
            for i in range(1, forecast_days + 1)
        ]

        # Trend Analysis
        trend, trend_color = analyze_trend(predictions)
        current_price = float(stock_data["Close"].iloc[-1])
        predicted_change = ((predictions[-1] - current_price) / current_price) * 100

        # Display Results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted Trend", trend)
        with col2:
            st.metric("Final Forecast", f"${predictions[-1]:.2f}")
        with col3:
            st.metric(
                "Projected Change",
                f"{predicted_change:.2f}%",
                delta_color="inverse" if predicted_change < 0 else "normal",
            )

        # Forecast Chart
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=stock_data["Date"],
                y=stock_data["Close"],
                name="Historical",
                line=dict(color="#3498db"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=predictions,
                name="Forecast",
                line=dict(color=trend_color, dash="dot", width=3),
            )
        )
        fig.update_layout(
            title=f"{selected_stock_name} {forecast_days}-Day Forecast", height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        # Daily Predictions Table
        st.subheader("Daily Close Forecast Details")
        forecast_df = pd.DataFrame(
            {
                "Date": forecast_dates,
                "Predicted Price": predictions,
                "Daily Change (%)": np.concatenate(
                    ([np.nan], np.diff(predictions) / predictions[:-1] * 100)
                ),
            }
        )
        st.dataframe(
            forecast_df.style.format(
                {"Predicted Price": "${:.2f}", "Daily Change (%)": "{:.2f}%"}
            ).applymap(
                lambda x: (
                    "color: #e74c3c"
                    if isinstance(x, str) and "-" in x
                    else "color: #2ecc71"
                ),
                subset=["Daily Change (%)"],
            ),
            hide_index=True,
        )

        # Display cleaned data sample
        # st.markdown("---")
        # st.write("Data Sample (cleaned):", stock_data.tail().reset_index(drop=True))
else:
    st.error("Model failed to load. Please check your model files.")

# Footer
st.markdown("---")
st.markdown(
    """
**Disclaimer**: This forecast is generated by AI models and should not be considered financial advice. 
Past performance is not indicative of future results.
"""
)
