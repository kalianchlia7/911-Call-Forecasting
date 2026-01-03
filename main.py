# ------------------------------------------------------------
# 1. DATA LOADING & PREPARATION
# ------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import os

#step 4
import warnings
warnings.filterwarnings("ignore")   # suppress statsmodels warnings
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

#step 5
from sklearn.preprocessing import MinMaxScaler

#step 6
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense 

#step 7
from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_csv("/Users/kalianchlia/Downloads/Coding Projects/ML Projects/Project 4/Data/911_Calls_For_Service_2024.csv")

#quick check of the first few rows to see what columns we have
print("Raw dataset preview:")
print(df.head())

# From the Baltimore schema--> correct column is 'callDateTime'
# get datetime
df['callDateTime'] = pd.to_datetime(df['callDateTime'])

# Extract date (drops the time portion)
df['date'] = df['callDateTime'].dt.date

# Group by date and count calls
daily_calls = df.groupby('date').size()
# Convert the Series to a DataFrame for convenience
daily_calls = daily_calls.reset_index(name='calls')
# Convert 'date' back to datetime type 
daily_calls['date'] = pd.to_datetime(daily_calls['date'])
#sort data by date chronologically 
daily_calls = daily_calls.sort_values('date')
# to make daily_calls.index = date, daily_calls['calls'] = number of calls.
daily_calls = daily_calls.set_index('date')

#quick preview
print("\nDaily 911 calls preview:")
print(daily_calls.head())

#statistics w/ .describe
print("\nSummary statistics for daily calls:")
print(daily_calls['calls'].describe())

#first plot
'''
plt.figure(figsize=(12, 5))
plt.plot(daily_calls.index, daily_calls['calls'])
plt.title("Daily 911 Calls — Baltimore 2024")
plt.xlabel("Date")
plt.ylabel("Number of 911 Calls")
plt.show()
'''

#save cleaned daily data
# daily_calls.to_csv("911_daily.csv")


# ------------------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ------------------------------------------------------------

#GOING TO PLOT AGAIN
#DIFFERENCE: USING NEW CLEANED DATA FILE

# This is the file  created in Step 1
daily_calls = pd.read_csv("/Users/kalianchlia/Downloads/Coding Projects/ML Projects/Project 4/Data/911_daily.csv", parse_dates=['date'], index_col='date')

# Visualize the overall trend, spikes, and seasonal patterns
plt.figure(figsize=(12, 5))
plt.plot(daily_calls.index, daily_calls['calls'], color='blue', linewidth=1)
plt.title("Daily 911 Calls — Baltimore 2024")
plt.xlabel("Date")
plt.ylabel("Number of Calls")
plt.grid(True)
plt.show()

# Summary of central tendency, dispersion, min/max, quartiles
print("\nBasic statistics for daily 911 calls:")
print(daily_calls['calls'].describe())

# Create a 'weekday' column to see patterns across days of the week
daily_calls['weekday'] = daily_calls.index.day_name()
weekly_avg = daily_calls.groupby('weekday')['calls'].mean().reindex([
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
])

# Plot average calls by weekday
plt.figure(figsize=(10, 4))
plt.bar(weekly_avg.index, weekly_avg.values, color='orange')
plt.title("Average Daily 911 Calls by Weekday")
plt.xlabel("Weekday")
plt.ylabel("Average Calls")
plt.show()

# Create a 'month' column to observe seasonality by month
daily_calls['month'] = daily_calls.index.month
monthly_avg = daily_calls.groupby('month')['calls'].mean()

# Plot average calls by month
plt.figure(figsize=(10, 4))
plt.plot(monthly_avg.index, monthly_avg.values, marker='o', linestyle='-', color='green')
plt.title("Average Daily 911 Calls by Month")
plt.xlabel("Month")
plt.ylabel("Average Calls")
plt.xticks(range(1, 13))
plt.show()


# Define spikes as calls above mean + 2*std
mean_calls = daily_calls['calls'].mean()
std_calls = daily_calls['calls'].std()
spikes = daily_calls[daily_calls['calls'] > mean_calls + 2 * std_calls]

print("\nPotential high-risk days (spikes):")
print(spikes)

# Plot spikes on the time-series
plt.figure(figsize=(12, 5))
plt.plot(daily_calls.index, daily_calls['calls'], color='blue', linewidth=1)
plt.scatter(spikes.index, spikes['calls'], color='red', label='High-Risk Spike')
plt.title("Daily 911 Calls with High-Risk Spikes Highlighted")
plt.xlabel("Date")
plt.ylabel("Number of Calls")
plt.legend()
plt.show()

# ------------------------------------------------------------
# 3. TRAIN / TEST SPLIT
# ------------------------------------------------------------

# We already have our cleaned time-series DataFrame:
# Index: date (daily)
# Column: call_count

# Determine the split point (80% train, 20% test)
split_point = int(len(daily_calls) * 0.8)   # convert 80% to an integer index

# Split chronologically 
train = daily_calls.iloc[:split_point]   # all rows from start → split point
test = daily_calls.iloc[split_point:]    # remaining rows

# Print shapes to confirm dataset sizes
print("Training set size:", train.shape)
print("Test set size:", test.shape)

# Visualize the split to confirm it makes sense
 #Trying to see how at what point (dates) training/testing begin/end bc date=index 
plt.figure(figsize=(12,6))
plt.plot(train.index, train['calls'], label='Training Data')
plt.plot(test.index,  test['calls'], label='Testing Data')
plt.title("Train/Test Split - 911 Daily Call Volume")
plt.xlabel("Date")
plt.ylabel("Call Count")
plt.legend()
plt.show()

# ------------------------------------------------------------
# 4. BASELINE MODEL — ARIMA
# ------------------------------------------------------------

# p = AR lags (auto-regressive)
# d = differencing (to remove trend)
# q = MA lags (moving average)
p, d, q = 5, 1, 2

#fitting arima model to training data
arima_model = ARIMA(train['calls'], order=(p, d, q))
arima_fit = arima_model.fit()

forecast_steps = len(test)               # number of future days to predict (in test set)
arima_forecast = arima_fit.forecast(steps=forecast_steps)
#.forecast produces predicted values, for #of specified steps=# days in test set

#evaluating arima performance by seeing error
rmse_arima = np.sqrt(mean_squared_error(test['calls'], arima_forecast))
print("ARIMA RMSE:", rmse_arima)

#plot arima predictions vs test
plt.figure(figsize=(12,6))
plt.plot(train.index, train['calls'], label='Training Data')
plt.plot(test.index, test['calls'], label='Actual Test Data')
plt.plot(test.index, arima_forecast, label='ARIMA Forecast')
plt.title("ARIMA Baseline Forecast")
plt.xlabel("Date")
plt.ylabel("Daily 911 Calls")
plt.legend()
plt.show()
#plt.savefig("arima_prediction_accuracy.png", dpi=300)

# ------------------------------------------------------------
# 5. LSTM DATA PREPARATION
# ------------------------------------------------------------

window = 14  # use past 14 days

# scale only training data
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train[['calls']])

# create sequences
def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_scaled, window)

# reshape for LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

# ------------------------------------------------------------
# 6. BUILD & TRAIN LSTM MODEL
# ------------------------------------------------------------

# Build model
model = Sequential()
model.add(LSTM(64, activation='tanh', return_sequences=False,
               input_shape=(window, 1)))
model.add(Dense(1))

# Compile
model.compile(optimizer='adam', loss='mse')

# Train
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=16,
    verbose=1
)

# ------------------------------------------------------------
# 7. LSTM PREDICTION & EVALUATION
# ------------------------------------------------------------

# Scale the test set using the same scaler as training
test_scaled = scaler.transform(test[['calls']])

# Create sequences for the LSTM (same window as training)
X_test, y_test = create_sequences(test_scaled, window)

# LSTM expects 3D input, so we reshape test data
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Predict using the trained model
lstm_pred_scaled = model.predict(X_test)

# y_test is 2D, reshape for inverse scaling
y_test_scaled = y_test.reshape(-1, 1)
y_test_unscaled = scaler.inverse_transform(y_test_scaled)

# LSTM predictions are also scaled, reshape to 2D
lstm_pred_unscaled = scaler.inverse_transform(lstm_pred_scaled)

# Align dates with LSTM predictions
lstm_dates = test.index[window:]  # skip first 'window' rows

# RMSE calculation
rmse_lstm = np.sqrt(mean_squared_error(y_test_unscaled, lstm_pred_unscaled))
print("\nLSTM RMSE:", rmse_lstm)

percent_improvement = ((rmse_arima - rmse_lstm) / rmse_arima) * 100
print(f"Improvement over ARIMA: {percent_improvement:.2f}%")


# ------------------------------------------------------------
# 8. VISUALIZATION & ANALYSIS
# ------------------------------------------------------------


# Ensure the output folder exists
output_folder = "/Users/kalianchlia/Downloads/Coding Projects/ML Projects/Project 4/OutcomePlots"
os.makedirs(output_folder, exist_ok=True)

# LSTM Actual vs Predicted
plt.figure(figsize=(10, 5))
plt.plot(test.index[window:], y_test_unscaled, label='Actual')
plt.plot(lstm_dates, lstm_pred_unscaled, label='LSTM Predicted')
plt.title("LSTM Prediction vs Actual Calls")
plt.xlabel("Date")
plt.ylabel("Call Volume")
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(output_folder, "lstm_vs_actual.png"), dpi=300)

# ARIMA vs LSTM Comparison
plt.figure(figsize=(10, 5))
plt.plot(test.index, test['calls'], label='Actual')
plt.plot(test.index, arima_forecast, label='ARIMA Forecast')
plt.plot(lstm_dates, lstm_pred_unscaled, label='LSTM Forecast')
plt.title("ARIMA vs LSTM Model Comparison")
plt.xlabel("Date")
plt.ylabel("Call Volume")
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(output_folder, "arima_vs_lstm.png"), dpi=300)

# Long-Term Trend Visualization
daily_calls['rolling_mean'] = daily_calls['calls'].rolling(window=30).mean()

plt.figure(figsize=(12, 6))
plt.plot(daily_calls.index, daily_calls['calls'], alpha=0.4, label="Daily Calls")
plt.plot(daily_calls.index, daily_calls['rolling_mean'], label="30-Day Trend")
plt.title("Long-Term Trend in Daily Emergency Calls")
plt.xlabel("Date")
plt.ylabel("Call Volume")
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(output_folder, "long_term_trend.png"), dpi=300)

# Seasonal Decomposition
decomposition = seasonal_decompose(daily_calls['calls'], model='additive', period=7)

plt.rcParams.update({'figure.figsize': (12, 8)})
decomposition.plot()
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(output_folder, "seasonal_decomposition.png"), dpi=300)

# Spike Detection (High-Risk Days)
threshold = daily_calls['calls'].mean() + daily_calls['calls'].std()
spike_days = daily_calls[daily_calls['calls'] > threshold]

plt.figure(figsize=(12, 6))
plt.plot(daily_calls.index, daily_calls['calls'], label="Daily Calls")
plt.axhline(threshold, linestyle='--', label="Spike Threshold")
plt.scatter(spike_days.index, spike_days['calls'], color='red', label="Spike Days")
plt.title("High-Risk Spike Detection")
plt.xlabel("Date")
plt.ylabel("Call Volume")
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(output_folder, "spike_detection.png"), dpi=300)


# ------------------------------------------------------------
# 9. SAVE PLOTS & PREDICTIONS
# ------------------------------------------------------------


# Save ARIMA predictions
arima_output = test.copy()
arima_output['arima_pred'] = arima_forecast
arima_output.to_csv(os.path.join(output_folder, "arima_predictions.csv"))
print("Saved:", os.path.join(output_folder, "arima_predictions.csv"))

# Save LSTM predictions
lstm_output = test.iloc[window:].copy()  # align with LSTM predictions
lstm_output['lstm_pred'] = lstm_pred_unscaled
lstm_output.to_csv(os.path.join(output_folder, "lstm_predictions.csv"))
print("Saved:", os.path.join(output_folder, "lstm_predictions.csv"))

# Save RMSE Metrics & Improvement
with open(os.path.join(output_folder, "model_metrics.txt"), "w") as f:
    f.write("MODEL PERFORMANCE METRICS\n")
    f.write("---------------------------------\n")
    f.write(f"ARIMA RMSE: {rmse_arima:.2f}\n")
    f.write(f"LSTM RMSE: {rmse_lstm:.2f}\n")
    f.write(f"LSTM Improvement (%): {percent_improvement:.2f}%\n")
print("Saved:", os.path.join(output_folder, "model_metrics.txt"))

# Save trained LSTM model
model.save(os.path.join(output_folder, "lstm_model.h5"))
print("Saved:", os.path.join(output_folder, "lstm_model.h5"))

