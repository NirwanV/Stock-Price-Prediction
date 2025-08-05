
# Importing relevent libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
import os
from datetime import datetime
import warnings
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

#Loading CSV file
data = pd.read_csv('all_stocks_5yr.csv', delimiter=',', on_bad_lines='skip') # read data from csv
print(data.shape)     
print(data.sample(7))

data.info()

# Convert date column
data['date'] = pd.to_datetime(data['date'])
data.info()

# Define the list of companies to visualize
companies = ['AAPL', 'AMD', 'FB', 'GOOGL', 'AMZN', 'NVDA', 'EBAY', 'CSCO', 'IBM']

# PLOT: Date vs Open and Close Prices 
plt.figure(figsize=(15, 8))
for index, company in enumerate(companies, 1):
    plt.subplot(3, 3, index)
    c = data[data['Name'] == company]
    plt.plot(c['date'], c['close'], c="r", label="Close", marker="+")
    plt.plot(c['date'], c['open'], c="g", label="Open", marker="^")
    plt.title(company)
    plt.legend()
plt.tight_layout()
plt.show()

# PLOT: Volume Over Time
plt.figure(figsize=(15, 8))
for index, company in enumerate(companies, 1):
    plt.subplot(3, 3, index)
    c = data[data['Name'] == company]
    plt.plot(c['date'], c['volume'], c='purple', marker='*')
    plt.title(f"{company} Volume")
plt.tight_layout()
plt.show()


# Select what company stock to be analyzed
Name = input('Name of the stock(AAPL, AMD, FB, GOOGL, AMZN, NVDA, EBAY, CSCO, IBM):')

#Prepare selected stock data
stock = data[data['Name'] == Name]

# Filter a specific prediction range (this variable is unused unless you want to plot it)
prediction_range = stock.loc[
    (stock['date'] > datetime(2013, 1, 1)) & (stock['date'] < datetime(2018, 1, 1))
]

# Plot selected stock prices
plt.figure(figsize=(12, 6))
plt.plot(stock['date'], stock['close'])
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.title(Name + " Stock Prices (Close)")
plt.grid()
plt.show()

# Prepare dataset
close_data = stock[['close']]
dataset = close_data.values
training = int(np.ceil(len(dataset) * 0.95))
print("Training data size:", training)

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[0:training, :]

# Create training features and labels
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

# Build LSTM model
model = keras.models.Sequential([
    keras.layers.LSTM(64, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    keras.layers.LSTM(64),
    keras.layers.Dense(32),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1)
])

# Compile and train model
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()
history = model.fit(x_train, y_train, epochs=10)

# Prepare test data
test_data = scaled_data[training - 60:, :]
x_test = []
y_test = dataset[training:, :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

# Make predictions
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Evaluation
mse = np.mean((predictions - y_test) ** 2)
rmse = np.sqrt(mse)
print("MSE:", mse)
print("RMSE:", rmse)

# Plot predictions vs actual
train = stock[:training]
test = stock[training:].copy()
test['Predictions'] = predictions

plt.figure(figsize=(12, 6))
plt.plot(train['date'], train['close'], label='Train')
plt.plot(test['date'], test['close'], label='Test')
plt.plot(test['date'], test['Predictions'], label='Predicted')
plt.title(Name + ' Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid()
plt.show()
