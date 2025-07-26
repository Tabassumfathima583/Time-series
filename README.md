# 📈 S&P 500 Volatility Prediction using Machine Learning

This project uses historical financial and macroeconomic time series data to **predict the 30-day volatility of the S&P 500 index** using a LightGBM model. We leverage SHAP values for **interpretable machine learning insights** and apply best practices in time series modeling.

---

## 📁 Dataset

- **Source**: [Kaggle - S&P 500 Volatility Time Series Dataset](https://www.kaggle.com/datasets/mathisjander/s-and-p500-volatility-prediction-time-series-data)
- **Target Variable**: `SP500 30 Day Volatility`
- **Features**: 30+ indicators including:
  - Equity market indices (SP500, DJIA, NASDAQ, RUSSELL)
  - Volatility index (VIX)
  - SPX options data (Put/Call Ratio, Volume)
  - Log returns & yield spreads
  - Macroeconomic indicators (Gold, Oil, Consumer Sentiment, USD Index)

---

## 🔧 Project Workflow

1. **Data Preprocessing**
   - Date parsing, sorting, and index setting
   - Handling missing values
   - Feature scaling (optional)

2. **Modeling**
   - Time-aware train-test split
   - LightGBM Regression Model
   - Evaluation metrics: MAE, RMSE, R²

3. **Model Evaluation**
   - Performance:
     - MAE: `~2.08`
     - RMSE: `~6.78`
     - R²: `~0.89`
   - Strong performance for a financial time series task

4. **Explainability**
   - SHAP visualizations: force plots, summary plots, and dependence plots
   - Insights into top drivers of volatility (e.g., VIX, Log Returns, Options data)

---

## 📊 Key Insights

```python
# VIX, SP500 log returns, and SPX Put/Call Ratio are top predictors.
# SHAP values confirm non-linear interactions (e.g., VIX > 25 has outsized impact).
# LightGBM handles multicollinearity and non-linearity well in financial data.
# Model can support decision-making in trading, risk management, or forecasting.
