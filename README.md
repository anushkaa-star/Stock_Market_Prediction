# Stock Market Prediction using Ensemble Learning

## Overview
This project is a Stock Market Trend Prediction System built using an ensemble of machine learning models. It analyzes historical stock data and predicts whether the stock price will go UP or DOWN for the next trading session. The system combines models like Random Forest, Gradient Boosting, AdaBoost, and Logistic Regression to improve prediction accuracy and stability.

## Features
- Real-time stock data fetching using Yahoo Finance  
- Technical indicators (RSI, MACD, Moving Averages, Bollinger Bands)  
- Multiple ML models with ensemble voting  
- Interactive dashboard using Streamlit  
- Model performance evaluation (Accuracy, Precision, Recall, ROC Curve)  

## 🛠️ Tech Stack
Python, Streamlit, Scikit-learn, Pandas, NumPy, Plotly, yFinance  

## ⚙️ How to Run the Project
1. Clone the repository  
git clone <your-repo-link>  

2. Go to project folder  
cd stock-market-prediction  

3. Install dependencies  
pip install -r requirements.txt  

4. Run the app  
streamlit run app.py  

## How It Works
User enters stock ticker → Data is fetched → Features like RSI, MACD are created → Models are trained → Ensemble combines predictions → Final output (BUY/SELL) is shown on dashboard.

## Models Used
Random Forest, Gradient Boosting, AdaBoost, Logistic Regression, Voting Classifier  

## Output
The system shows stock graphs, technical indicators, model comparison, and final prediction (Buy/Sell signal).

## Future Improvements
Add real-time data streaming, integrate news sentiment analysis, improve accuracy using advanced models.

