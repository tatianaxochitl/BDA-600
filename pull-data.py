import os

import yfinance as yf


def main():
    # Make folders to house the data
    newpath = r"data"
    if not os.path.exists(newpath):
        os.makedirs(newpath)
        os.makedirs(r"data/stocks")
        os.makedirs(r"data/crypto")

    # Download Stock CSV for past 5 years
    df = yf.download(
        "NFLX JPM FB AMZN AAPL", start="2017-03-01", end="2022-03-01", group_by="ticker"
    )
    df.NFLX.to_csv("data/stocks/NFLX.csv")
    df.JPM.to_csv("data/stocks/JPM.csv")
    df.FB.to_csv("data/stocks/FB.csv")
    df.AMZN.to_csv("data/stocks/AMZN.csv")
    df.AAPL.to_csv("data/stocks/AAPL.csv")

    # Download Crypto CSV for past 5 years
    df = yf.download(
        "BTC-USD DOGE-USD ETH-USD ADA-USD SOL-USD",
        start="2017-03-01",
        end="2022-03-01",
        group_by="ticker",
    )
    df["BTC-USD"].to_csv("data/crypto/BTC.csv")
    df["DOGE-USD"].to_csv("data/crypto/DOGE.csv")
    df["ETH-USD"].to_csv("data/crypto/ETH.csv")
    df["ADA-USD"].to_csv("data/crypto/ADA.csv")
    df["SOL-USD"].to_csv("data/crypto/SOL.csv")

    print("Data Downloaded!")


if __name__ == "__main__":
    main()
