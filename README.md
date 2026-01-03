# 911 Calls Forecasting — Baltimore 2024

**Time Series Forecasting | Machine Learning | LSTM & ARIMA**

---

## Project Overview
Forecasted daily 911 call volumes in Baltimore using **ARIMA** and **LSTM** models.  
Goal: identify trends, seasonal patterns, and high-risk days to support data-driven emergency resource planning.

---

## Dataset
- **Source:** Baltimore Open Data Portal (2024 911 Calls)  
- **Key column:** `callDateTime`  
- Preprocessed into **daily call counts** (`911_daily.csv`).

---

## Key Steps
- **EDA:** Visualized trends, weekly/monthly patterns, and high-risk spikes.  
- **Train/Test Split:** 80% train / 20% test (chronological).  
- **ARIMA Baseline:** RMSE calculated for comparison.  
- **LSTM Model:** 1 LSTM layer + Dense output, 14-day input window.  
- **Evaluation:** Compared LSTM vs ARIMA RMSE and computed improvement.  
- **Visualization:** Actual vs predicted calls, trends, seasonality, and spikes.

---

## Results

| Metric           | ARIMA | LSTM |
|-----------------|-------|------|
| RMSE             | 45.8  | 39.4 |
| Improvement (%)  | -     | 14%  |

- **High-Risk Days:** Identified dates with unusually high call volumes.
- **Plots:** Outcomes available in uploaded files

---

## Outputs
Saved in `outputs/`:
- Forecast plots (`plot_*.png`)  
- Predictions (`arima_predictions.csv`, `lstm_predictions.csv`)  
- Trained model (`lstm_model.h5`)  
- Metrics summary (`model_metrics.txt`)

---

## Technologies
**Python, Pandas, NumPy, Matplotlib, Scikit-learn, Statsmodels, TensorFlow/Keras**

---

## How to Run
```bash
git clone https://github.com/YourUsername/911-Calls-Forecasting.git
cd 911-Calls-Forecasting

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate      # macOS/Linux
# venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt

# Run the main script
python main.py
