# We start by importing all of the modules we think we might need.
from sklearn.metrics import mean_absolute_percentage_error
from tracemalloc import start
import matplotlib.pyplot as plt
#!pip install yfinance
import yfinance as yf
#!pip install pandas
import pandas as pd
import numpy as np
import os
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sktime.forecasting.arima import AutoARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

# Creating a list of the stock/cryptos we want to use
stockList = ['AAPL', 'TSLA', 'AMZN', 'FB', 'JPM', 'BTC-USD', 'ETH-USD', 'DOGE-USD', "ADA-USD", "SOL-USD"]

# Let us define our start and end time
startDate = '2018-01-01'
endDate = '2021-12-31'

# This actually pulls all of the historical data for our analysis
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

# Putting all of the stocks/cryotos in one object for reference later
stockNames = [aapl, tsla, amzn, fb, jpm, btc, eth, doge, ada, sol]

# Adding in the adjusted close column to all of the stocks/cryptos
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
    plt.subplot(5, 2, i + 1)
    plt.ylabel('Price in US Dollar')
    x['Low'].plot(figsize=(15, 25), title=f'Four Year Daily Low History: {stockList[i]} ')
    plt.tight_layout()
plt.savefig('FourYearDailyLow.png')

# And again for the Open and Close
for i, x in enumerate(stockNames):
    plt.subplot(5, 2, i + 1)
    plt.ylabel('Price in US Dollar')
    x['Open'].plot(figsize=(15, 25), title=f'Four Year Daily Open Price History:{stockList[i]} ')
    plt.tight_layout()
plt.savefig('FourYearDailyOpen.png')

for i, x in enumerate(stockNames):
    plt.subplot(5, 2, i + 1)
    plt.ylabel('Price in US Dollar')
    x['Close'].plot(figsize=(15, 25), title=f'Four Year Daily Closing Price History: {stockList[i]}')
    plt.tight_layout()
plt.savefig('FourYearDailyClose.png')

# Volume traded
for i, x in enumerate(stockNames):
    plt.subplot(5, 2, i + 1)
    plt.ylabel('Volume Traded')
    x['Volume'].plot(figsize=(15, 25), title=f'Four Year History of Daily Volume Traded: {stockList[i]}')
    plt.tight_layout()
plt.savefig('VolumeTraded.png')

# Open-Close
for i, x in enumerate(stockNames):
    plt.subplot(5, 2, i + 1)
    plt.ylabel('Price in US Dollar')
    (x['Open'] - x['Close']).plot(figsize=(15, 25),
                                  title=f'Four Year Daily Open-Close Difference History: {stockList[i]}')
    plt.tight_layout()
plt.savefig('OpenCloseDifference.png')

# Lets Start looking at the moving averages
# Set the moving average values
movingAverageDays = [7, 14, 21]

# This for loop should move through each moving average for each of the stocks and add three distinct columns for
# each stock.
for mov_avg in movingAverageDays:
    for stock in stockNames:
        c_name = f"Moving Avg for {mov_avg} days"
        stock[c_name] = stock['Adj Close'].rolling(mov_avg).mean()
        c2_name = f"Standard Deviation for {mov_avg} days"
        stock[c2_name] = stock['Adj Close'].rolling(mov_avg).std()

aapl.plot(y=['Adj Close', 'Moving Avg for 7 days', 'Moving Avg for 14 days', 'Moving Avg for 21 days'])
plt.savefig('AppleMA.png')

tsla.plot(y=['Adj Close', 'Moving Avg for 7 days', 'Moving Avg for 14 days', 'Moving Avg for 21 days'])
plt.savefig('TeslaMA.png')

amzn.plot(y=['Adj Close', 'Moving Avg for 7 days', 'Moving Avg for 14 days', 'Moving Avg for 21 days'])
plt.savefig('AmazonMA.png')

fb.plot(y=['Adj Close', 'Moving Avg for 7 days', 'Moving Avg for 14 days', 'Moving Avg for 21 days'])
plt.savefig('FacebookMA.png')

jpm.plot(y=['Adj Close', 'Moving Avg for 7 days', 'Moving Avg for 14 days', 'Moving Avg for 21 days'])
plt.savefig('JPMMA.png')

eth.plot(y=['Adj Close', 'Moving Avg for 7 days', 'Moving Avg for 14 days', 'Moving Avg for 21 days'])
plt.savefig('EtherumMA.png')

btc.plot(y=['Adj Close', 'Moving Avg for 7 days', 'Moving Avg for 14 days', 'Moving Avg for 21 days'])
plt.savefig('BitcoinMA.png')

doge.plot(y=['Adj Close', 'Moving Avg for 7 days', 'Moving Avg for 14 days', 'Moving Avg for 21 days'])
plt.savefig('DogeMA.png')

ada.plot(y=['Adj Close', 'Moving Avg for 7 days', 'Moving Avg for 14 days', 'Moving Avg for 21 days'])
plt.savefig('ADAMA.png')

sol.plot(y=['Adj Close', 'Moving Avg for 7 days', 'Moving Avg for 14 days', 'Moving Avg for 21 days'])
plt.savefig('SOLMA.png')

'''NOTE: An increasing rolling mean and standard deviation indicate that the series is not stationary.
The website that this model is based off of suggests that seasonality and trend may need to be removed to
undertake time series analysis. We will replicate that deconstruction for one of our stocks'''

# We can remove trend by using the log function
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
plt.show()

forecaster = AutoARIMA(start_p=8, max_p=9, suppress_warnings=True)
trainData.index = trainData.index.astype(int)
forecaster.fit(trainData)
forecaster.summary()


def plot_forecast(series_train, series_test, forecast, forecast_int=None):

    mae = mean_absolute_error(series_test, forecast)
    mape = mean_absolute_percentage_error(series_test, forecast)

    plt.figure(figsize=(12, 6))
    plt.title(f"MAE: {mae:.2f}, MAPE: {mape:.3f}", size=18)
    series_train.plot(label="Train", color="b")
    series_test.plot(label="Test", color="g")
    forecast.index = series_test.index
    forecast.plot(label="Forecast", color="r")
    if forecast_int is not None:
        plt.fill_between(
            series_test.index,
            forecast_int["lower"],
            forecast_int["upper"],
            alpha=0.2,
            color="dimgray",
        )
    plt.legend(prop={"size": 16})
    plt.show()

    return mae, mape


test_len = int(len(aapl) * 0.2)
fh = np.arange(test_len) + 1
forecast, forecast_int = forecaster.predict(fh=fh, return_pred_int=True, alpha=0.05)
sun_arima_mae, sun_arima_mape = plot_forecast(
    trainData, testData, forecast, forecast_int
)
