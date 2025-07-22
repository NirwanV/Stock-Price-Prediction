# 📈 Apple Stock Price Prediction Using LSTM

This project focuses on predicting the closing stock prices of Apple Inc. (AAPL) using an LSTM (Long Short-Term Memory) neural network. Utilizing historical stock data from the `all_stocks_5yr.csv` dataset, it begins with visualizing trends in open/close prices and trading volume for several major tech companies including AAPL, AMD, FB, GOOGL, AMZN, NVDA, EBAY, CSCO, and IBM. The data is preprocessed by converting date columns, filtering for the target company, and normalizing the closing prices using `MinMaxScaler`. A sequence-based dataset is then created for training the LSTM model, which features two stacked LSTM layers, dense layers, and dropout for regularization.

The model is trained on 95% of the Apple stock dataset and evaluated on the remaining data using Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) metrics. Finally, the actual and predicted stock prices are plotted to visually assess the model’s performance, showcasing how deep learning can be applied for time series forecasting in financial markets.

## 🛠 Technologies Used

- Python 3.x  
- Pandas  
- NumPy  
- Matplotlib & Seaborn  
- TensorFlow / Keras  
- Scikit-learn

## 📊 Features

- Company-wise visualization of open, close, and volume trends  
- Time series preparation using a 60-day window  
- Deep learning model for sequential data  
- Model training, prediction, and performance evaluation  
- Visualization of actual vs predicted Apple stock prices

## 📁 Dataset

- **File**: `all_stocks_5yr.csv`  
- **Content**: Historical stock data from 2013 to 2018  
- **Columns**: Date, Open, High, Low, Close, Volume, Company Name

## 📉 Evaluation Metrics

- **MSE (Mean Squared Error)**  
- **RMSE (Root Mean Squared Error)**

## 📷 Output

The final plot compares the model’s predicted prices with actual closing prices for Apple, clearly illustrating its forecasting capability.

---

> ⚠️ *Note*: This project is intended for educational and experimental purposes and should not be used for real-world financial trading decisions.
