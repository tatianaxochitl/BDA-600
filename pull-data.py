import os
import sys

import yfinance as yf


def main():
    # Make folders to house the data
    newpath = r"data"
    if not os.path.exists(newpath):
        os.makedirs(newpath)
        os.makedirs(r"data/stocks")
        os.makedirs(r"data/crypto")

    # Download Stock CSV for past 4 years
    df = yf.download(
        "TSLA JPM FB AMZN AAPL", start="2018-01-01", end="2022-03-01", group_by="ticker"
    )
    df.TSLA.to_csv("data/stocks/TSLA.csv")
    df.JPM.to_csv("data/stocks/JPM.csv")
    df.FB.to_csv("data/stocks/FB.csv")
    df.AMZN.to_csv("data/stocks/AMZN.csv")
    df.AAPL.to_csv("data/stocks/AAPL.csv")

    # Download Crypto CSV for past 4 years
    df = yf.download(
        "BTC-USD DOGE-USD ETH-USD ADA-USD SOL-USD",
        start="2018-01-01",
        end="2022-03-01",
        group_by="ticker",
    )
    df["BTC-USD"].to_csv("data/crypto/BTC.csv")
    df["DOGE-USD"].to_csv("data/crypto/DOGE.csv")
    df["ETH-USD"].to_csv("data/crypto/ETH.csv")
    df["ADA-USD"].to_csv("data/crypto/ADA.csv")
    df["SOL-USD"].to_csv("data/crypto/SOL.csv")

    # Download NASDAQ100, S&P, and DOW for comparison
    df = yf.download(
        "^NDX ^GSPC ^DJI",
        start="2018-01-01",
        end="2022-03-01",
        group_by="ticker",
    )

    df["^NDX"].to_csv("data/NDX.csv")
    df["^GSPC"].to_csv("data/GSPC.csv")
    df["^DJI"].to_csv("data/DJI.csv")

    print("Data Downloaded!")


if __name__ == "__main__":
    sys.exit(main())
