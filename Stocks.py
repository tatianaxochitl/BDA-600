
#!pip install statsmodels

#!pip install sktime

# We start by importing all of the modules we think we might need.
import statsmodels.api as sm
import pmdarima as pm
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
from statsmodels.tsa.stattools import kpss
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
#!pip install pmdarima

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

# We want to test for stationarity
fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
ax.plot(aapl['Adj Close'])

"""The clear upward trend poses the idea that there is not stationarity. We can try and reduce the variance to help, using the log function will aid in the process."""

# We can remove trend by using the log function
log_aapl = np.log(aapl['Adj Close'])
fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
ax.plot(log_aapl)

"""The stock still does not seem stationary but we can test for stationarity using the Augmented Dicky Fuller Test (ADF Test) and the Kwiatkowski-Phillips-Schmidt-Shin (KPSS). 

For ADF: The more negative the number is the more of a rejection of the null hypothesis that the time series is non-stationary.

For KPSS: The higher the test statistic, the more we can reject the null hypothesis that the series is stationary
"""

# Source for the two functions: https://medium.com/coinmonks/bitcoin-arima-model-f22456bd1fa9


def adf_test(series):
    result = adfuller(series)
    adf_stat = result[0]
    p_val = result[1]
    print("ADF Statistic: %f" % adf_stat)
    print("p-value: %f" % p_val)
    return adf_stat, p_val


def kpss_test(series):
    print("Results of KPSS Test:")
    kpssTest = kpss(series, regression="ct", nlags="auto")
    kpssOutput = pd.Series(kpssTest[0:3], index=["Test Stat", "p-value", "Lags Used"])
    for key, value in kpssTest[3].items():
        kpssOutput["Critical Value (%s)" % key] = value
    print(kpssOutput)


adf_statistic, p_value = adf_test(aapl['Adj Close'])

kpss_test(aapl['Adj Close'])

"""Since we can not get a conclusive answer on whether the series is stationary or not, we will difference the data"""


"""Determining the AR and MA. We use the Auto Correlation Function (ACF) and the Partial Auto Correlation Function (PACF) plot functions.

ACF: displays the correlation coefficients between a time series and its lag values. It explains how the present value iof a given time series is related to previous values.

PACF: explains the partial correlation between the series and its lags; however, it correlates the aspects of  γ(τ) and γ(τ-3) that are not predicted by γ(τ-1) γ(τ-2). 

When looking at both plots we look for significant points outside the shaded area and a geometric decay if we are dealing with a time series where ARIMA may be appropriate.
"""

# ACF
fig, ax = plt.subplots(1, figsize=(12, 8), dpi=100)
plot_acf(log_aapl, lags=20, ax=ax)
plt.ylim([-0.05, 0.25])
plt.yticks(np.arange(-0.10, 1.1, 0.1))
plt.show()

# PACF
fig, ax = plt.subplots(1, figsize=(12, 8), dpi=100)
plot_pacf(log_aapl, lags=20, ax=ax)
plt.ylim([-0.05, 0.25])
plt.yticks(np.arange(-0.10, 1.1, 0.1))
plt.show()

"""We can use the Auto ARIMA function to find the best fit for the model."""


def auto_arima(x):
    x = np.log(x['Adj Close'])
    model = pm.auto_arima(x, start_p=10, start_q=10, test="adf",
                          max_p=10, max_q=10, m=1, d=None,
                          seasonal=False, D=0, trace=True,
                          error_action="ignore", suppress_warnings=True,
                          stepwise=True)
    differenced_by_auto_arima = x.diff(model.order[1])
    return model.order, differenced_by_auto_arima, model.resid()


model_order, differenced_data, model_residuals = auto_arima(aapl)

"""Now that we have the best model to use, we will start to make predictions."""

# Fitting Model
aapl_model = sm.tsa.arima.ARIMA(np.log(aapl['Adj Close']), order=(5, 1, 7))
aapl_fitted = aapl_model.fit()

# Forecasting for a whole month
aapl_forecast = aapl_fitted.get_forecast(7)

# 95% confidence interval
aapl_forecast = (aapl_forecast.summary_frame(alpha=0.05))

# Need the axis for the prediction dates

future_7_days = ['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04',
                 '2022-01-05', '2022-01-06', '2022-01-07']
for x in future_7_days:
    pd.to_datetime(x)

aapl_forecast['Date'] = future_7_days
aapl_forecast['Date'] = aapl_forecast['Date'].astype('datetime64[ns]')
aapl_forecast.set_index('Date', inplace=True)

# Getting the mean forecast
aapl_forecast_mean = aapl_forecast['mean']

# Lower Confidence
aapl_forecast_lower = aapl_forecast['mean_ci_lower']

# Upper Confidence
aapl_forecast_upper = aapl_forecast['mean_ci_upper']

# Start plotting
plt.figure(figsize=(12, 8), dpi=100)

# Last 50 days
plt.plot(aapl['Adj Close'][-25:], label="Apple Price")


# Plot mean forecast
plt.plot(aapl_forecast.index, np.exp(aapl_forecast_mean), label="Mean Forecast", linewidth=1.5)

# Create confidence interval
plt.fill_between(aapl_forecast.index, np.exp(aapl_forecast_lower), np.exp(
    aapl_forecast_upper), color='b', alpha=0.1, label='95% Confidence')

# Set Title
plt.title('Apple 7 day forecast')

# Setting the legend
plt.legend(loc='upper left', fontsize=9)
plt.show()

"""We should now compare the forecasted results with what actually happened during the beginning of the year."""
