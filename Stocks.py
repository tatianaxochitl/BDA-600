from tracemalloc import start
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np
import os
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from sktime.forecasting.arima import AutoARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

stockList = ['AAPL', 'TSLA', 'AMZN', 'FB', 'JPM', 'BTC-USD', 'ETH-USD', 'DOGE-USD', "ADA-USD", "SOL-USD"]

# Let us define our start and end time
startDate = '2018-01-01'
endDate = '2021-12-31'

aapl = yf.Ticker("AAPL").history(start=startDate, end=endDate)
tsla = yf.Ticker("TSLA").history(start=startDate, end=endDate)
amzn = yf.Ticker("AMZN").history(start=startDate, end=endDate)
fb = yf.Ticker("FB").history(start=startDate, end=endDate)
jpm = yf.Ticker("JPM").history(start=startDate, end=endDate)
btc = yf.Ticker("BTC-USD").history(start=startDate, end=endDate)
eth = yf.Ticker("ETH-USD").history(start=startDate, end=endDate)
doge = yf.Ticker("DOGE-USD").history(start=startDate, end=endDate)
ada = yf.Ticker("ADA-USD").history(start=startDate, end=endDate)
sol = yf.Ticker("SOL-USD").history(start=startDate, end=endDate)

stockNames = [aapl, tsla, amzn, fb, jpm, btc, eth, doge, ada, sol]

for x in stockNames:
    x['Adj Close'] = x['Close'] - x['Dividends']

# This plots the four year Daily Highs for each of the stocks

for i, x in enumerate(stockNames):
    plt.subplot(5, 2, i + 1)
    plt.ylabel('Price in US Dollar')
    x['High'].plot(figsize=(15, 25), title=f'Four Year Daily High History: {stockList[i]} ')
    plt.tight_layout()
plt.savefig('FourYearDailyHigh.png')

# We do the same for the lows
for i, x in enumerate(stockNames):
    plt.subplot(len(stockNames), 2, i + 1)
    plt.ylabel('Price in US Dollar')
    x['Low'].plot(figsize=(10, 20), title=f'Four Year Daily Low History: {stockList[i]} ')
    plt.tight_layout()
    break

# And again for the Open and Close
for i, x in enumerate(stockNames):
    plt.subplot(len(stockNames), 2, i + 1)
    plt.ylabel('Price in US Dollar')
    x['Open'].plot(figsize=(10, 20), title=f'Four Year Daily Open Price History:{stockList[i]} ')
    break

for i, x in enumerate(stockNames):
    plt.subplot(len(stockNames), 2, i + 1)
    plt.ylabel('Price in US Dollar')
    x['Close'].plot(figsize=(10, 20), title=f'Four Year Daily Closing Price History: {stockList[i]}')
    break

# Volume traded
for i, x in enumerate(stockNames):
    plt.subplot(len(stockNames), 2, i + 1)
    plt.ylabel('Volume Traded')
    x['Volume'].plot(figsize=(10, 20), title=f'Four Year History of Daily Volume Traded: {stockList[i]}')
    break

# High Minus Low
for i, x in enumerate(stockNames):
    plt.subplot(len(stockNames), 2, i + 1)
    plt.ylabel('Price in US Dollar')
    (x['High'] - x['Low']).plot(figsize=(10, 20), title=f'Four Year Daily High Minus Low Price History: {stockList[i]}')
    break

# Open-Close
for i, x in enumerate(stockNames):
    plt.subplot(len(stockNames), 2, i + 1)
    plt.ylabel('Price in US Dollar')
    (x['Open'] - x['Close']).plot(figsize=(10, 20),
                                  title=f'Four Year Daily Open-Close Difference History: {stockList[i]}')
    break

# Lets Start looking at the moving averages
# Set the moving average values
movingAverageDays = [7, 14, 21]

# This for loop should move through each moving average for each of the stocks and add three distinct columns for
# each stock.
for mov_avg in movingAverageDays:
    for stock in stockNames:
        c_name = f"Moving Avg for {mov_avg} days"
        stock[c_name] = stock['Close'].rolling(mov_avg).mean()
        c2_name = f"Standard Deviation for {mov_avg} days"
        stock[c2_name] = stock['Close'].rolling(mov_avg).std()

# Let's plot the moving averages (overlayed) for each stock
'''
aapl.plot(y=['Close', 'Moving Avg for 7 days', 'Moving Avg for 14 days', 'Moving Avg for 21 days'])
tsla.plot(y=['Close', 'Moving Avg for 7 days', 'Moving Avg for 14 days', 'Moving Avg for 21 days'])
amzn.plot(y=['Close', 'Moving Avg for 7 days', 'Moving Avg for 14 days', 'Moving Avg for 21 days'])
fb.plot(y=['Close', 'Moving Avg for 7 days', 'Moving Avg for 14 days', 'Moving Avg for 21 days'])
jpm.plot(y=['Close', 'Moving Avg for 7 days', 'Moving Avg for 14 days', 'Moving Avg for 21 days'])
eth.plot(y=['Close', 'Moving Avg for 7 days', 'Moving Avg for 14 days', 'Moving Avg for 21 days'])
btc.plot(y=['Close', 'Moving Avg for 7 days', 'Moving Avg for 14 days', 'Moving Avg for 21 days'])
doge.plot(y=['Close', 'Moving Avg for 7 days', 'Moving Avg for 14 days', 'Moving Avg for 21 days'])
'''


# let's look at the ARIMA Model
# Testing the a distribution of one of the stocks
aapl.plot(y=['Close'], figsize=(5, 5), kind='kde')
plt.close()
# plt.show()

'''
- The average value in the series is called the level.
- The increasing or falling value in the series is referred to as the trend.
- Seasonality is the series’ recurring short-term cycle.
- The random variance in the series is referred to as noise.
'''
# Using the stored information we have from the previous for loop, we can compare for stationarity
aapl.plot(y=['Adj Close', 'Moving Avg for 14 days', 'Standard Deviation for 14 days'])
plt.close()

# plt.show()


'''NOTE: An increasing rolling mean and standard deviation indicate that the series is not stationary.
The website that this model is based off of suggests that seasonality and trend may need to be removed to
undertake time series analysis. We will replicate that deconstruction for one of our stocks'''

# testing the function
# test_for_stationarity(aapl['Close'])

# We can remove any trend by using the log function
log_aapl = np.log(aapl['Adj Close'])

# Now we can train the ARIMA Model
# Obviously we must start by splitting the data into train and test
trainData = log_aapl[3:int(len(log_aapl) * 0.9)]
testData = log_aapl[int(len(log_aapl) * 0.9):]
# Plot it to make sure things look correct
plt.figure(figsize=(10, 6))
plt.grid(True)
plt.xlabel('Dates')
plt.ylabel('Closing Prices')
plt.plot(log_aapl, 'green', label='Train data')
plt.plot(testData, 'blue', label='Test data')
plt.legend()
# plt.show()

# Modeling
# Build Model
