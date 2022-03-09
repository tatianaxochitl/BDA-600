from tracemalloc import start
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np
import os
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

stockList = ['AAPL', 'TSLA', 'AMZN', 'FB', 'JPM', 'BTC-USD', 'ETH-USD', 'DOGE-USD']

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

# Summary Statistics
# print(aapl.describe())

stockNames = [aapl, tsla, amzn, fb, jpm, btc, eth, doge]

# This plots the four year Daily Highs for each of the stocks

for i, x in enumerate(stockNames):
    plt.subplot(len(stockNames), 2, i + 1)
    plt.ylabel('Price in US Dollar')
    x['High'].plot(figsize=(10, 20), title='Four Year Daily High History: ')
    break

# We do the same for the lows
for i, x in enumerate(stockNames):
    plt.subplot(len(stockNames), 2, i + 1)
    plt.ylabel('Price in US Dollar')
    x['Low'].plot(figsize=(10, 20), title='Four Year Daily Low History: ')
    break

# And again for the Open and Close
for i, x in enumerate(stockNames):
    plt.subplot(len(stockNames), 2, i + 1)
    plt.ylabel('Price in US Dollar')
    x['Open'].plot(figsize=(10, 20), title='Four Year Daily Open Price History: ')
    break

for i, x in enumerate(stockNames):
    plt.subplot(len(stockNames), 2, i + 1)
    plt.ylabel('Price in US Dollar')
    x['Close'].plot(figsize=(10, 20), title='Four Year Daily Closing Price History: ')
    break

# Volume traded
for i, x in enumerate(stockNames):
    plt.subplot(len(stockNames), 2, i + 1)
    plt.ylabel('Volume Traded')
    x['Volume'].plot(figsize=(10, 20), title='Four Year History of Daily Volume Traded: ')
    break

# High Minus Low
for i, x in enumerate(stockNames):
    plt.subplot(len(stockNames), 2, i + 1)
    plt.ylabel('Price in US Dollar')
    (x['High'] - x['Low']).plot(figsize=(10, 20), title='Four Year Daily High Minus Low Price History: ')
    break

# Open-Close
for i, x in enumerate(stockNames):
    plt.subplot(len(stockNames), 2, i + 1)
    plt.ylabel('Price in US Dollar')
    (x['Open'] - x['Close']).plot(figsize=(10, 20), title='Four Year Daily Open-Close Difference History: ')
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
aapl.plot(y=['Close', 'Moving Avg for 14 days', 'Standard Deviation for 14 days'])
plt.close()


# plt.show()

# Test for stationarity, this will output test statistics
def test_for_stationarity(closing_stock):
    print("Results of Dickey Fuller test")
    adft = adfuller(closing_stock, autolag='AIC')
    # output for dft will give us without defining what the values are.
    # hence we manually write what values does it explains using a for loop
    output = pd.Series(adft[0:4],
                       index=['Test Statistics', 'p-value', 'No. of lags used', 'Number of observations used'])
    for key, values in adft[4].items():
        output['critical value (%s)' % key] = values
    print(output)


'''NOTE: An increasing rolling mean and standard deviation indicate that the series is not stationary.
The website that this model is based off of suggests that seasonality and trend may need to be removed to
undertake time series analysis. We will replicate that deconstruction for one of our stocks'''

# testing the function
# test_for_stationarity(aapl['Close'])

# We can remove any trend by using the log function
log_aapl = np.log(aapl['Close'])

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
plt.show()

''' The auto ARIMA function returns a fitted ARIMA model after determining optimal parameters
The function performs these tests: Kwiatkowski–Phillips–Schmidt–Shin, Augmented Dickey-Fuller, or Phillips–Perron
in order to optimize starting values'''

Auto_ARIMA_Model = auto_arima(trainData, start_p=0, start_q=0,
                              test='adf',        # use adftest to find optimal 'd'
                              max_p=3, max_q=3,  # maximum p and q
                              m=1,               # frequency of series
                              d=None,            # let model determine 'd'
                              seasonal=False,    # No Seasonality
                              start_P=0,
                              D=0,
                              trace=True,
                              error_action='ignore',
                              suppress_warnings=True,
                              stepwise=True)
print(Auto_ARIMA_Model.summary())
Auto_ARIMA_Model.plot_diagnostics(figsize=(15, 8))
plt.show()

'''
# We can start by looking at the Daily Highs for the desired time period
for i, ticker in enumerate(stockList):
    current_ticker = yf.Ticker(ticker)
    plt.subplot(len(stockList), 2, i + 1)
    plt.ylabel('Price in USD')
    current_ticker.history(start=startDate, end=endDate)['High'].plot(
        figsize=(10, 20), title='Four Year History of Daily Highs: ' + ticker)
    break

# Code in somthing that will allow us to save all of the figures from the for loops
# plt.tight_layout(pad=3)
# plt.show()

# Now we can look at the lows
for i, ticker in enumerate(stockList):
    current_ticker = yf.Ticker(ticker)
    plt.subplot(len(stockList), 2, i + 1)
    plt.ylabel('Price in USD')
    current_ticker.history(start=startDate, end=endDate)['Low'].plot(
        figsize=(10, 20), title='Four Year History of Daily Lows: ' + ticker)
    break

# Open and Close Prices
for i, ticker in enumerate(stockList):
    current_ticker = yf.Ticker(ticker)
    plt.subplot(len(stockList), 2, i + 1)
    plt.ylabel('Price in USD')
    current_ticker.history(start=startDate, end=endDate)['Open'].plot(
        figsize=(10, 20), title='Four Year History of Open Prices: ' + ticker)
    break

for i, ticker in enumerate(stockList):
    current_ticker = yf.Ticker(ticker)
    plt.subplot(len(stockList), 2, i + 1)
    plt.ylabel('Price in USD')
    current_ticker.history(start=startDate, end=endDate)['Close'].plot(
        figsize=(10, 20), title='Four Year History of Closing Prices: ' + ticker)
    break

# Volume
for i, ticker in enumerate(stockList):
    current_ticker = yf.Ticker(ticker)
    plt.subplot(len(stockList), 2, i + 1)
    plt.ylabel('Market Volume')
    current_ticker.history(start=startDate, end=endDate)['Volume'].plot(
        figsize=(10, 20), title='Four Year History of Volume: ' + ticker)
    break

# The difference in highs and lows for each stock over the four year period
for i, ticker in enumerate(stockList):
    current_ticker = yf.Ticker(ticker)
    plt.subplot(len(stockList), 2, i + 1)
    plt.ylabel('Price in USD')
    (current_ticker.history(start=startDate, end=endDate)['High'] - current_ticker.history(
        start=startDate, end=endDate)['Low']).plot(figsize=(20, 60), title='Four Year History (High - Low): ' + ticker)
    break
'''
