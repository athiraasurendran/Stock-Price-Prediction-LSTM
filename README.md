# 📈 AI Stock Forecaster – Stock Price Prediction Using LSTM

This project uses Long Short-Term Memory (LSTM) deep learning models to forecast the **next 7 days of stock prices** for 25 leading companies. It integrates data collection, modeling, and real-time interactive visualization using **Streamlit**.

🌐 Try it Live: [AI Stock Forecaster App](https://ai-stock-forecaster.streamlit.app/)

---

##  Project Highlights

-  **7-Day Forecast** of stock closing prices (company-wise)
-  **Actual vs Predicted** price graphs
-  Detects **direction movement** (Up/Down)
-  **Data Source:** Yahoo Finance (2010–2025)
-  Trained individual **LSTM models for 25 companies**
-  **Sample Evaluation (Amazon):**
  - MSE: 7.81
  - MAE: 2.05
  - R² Score: 0.998

---

##  Tech Stack

- **Languages & Libraries:**
  - Python, Pandas, NumPy
  - TensorFlow & Keras (LSTM)
  - Scikit-learn
  - yfinance (data extraction)
  - Matplotlib, Seaborn, Plotly (visualization)

- **Deployment:**
  - Streamlit Cloud
  - GitHub for version control and hosting

---

## 📁 Directory Structure

 stock-price-prediction-lstm
├── 📁 stock_models/ # Trained LSTM models for 25 companies
├── 📁 streamlit_app/ # Streamlit front-end app
│ └── app.py # Main Streamlit script
├── 📁 notebooks/ # Training notebooks (1 per stock)
├── requirements.txt # Project dependencies
├── .gitignore # Git ignored files
├── LICENSE # License file
└── README.md # Project documentation
