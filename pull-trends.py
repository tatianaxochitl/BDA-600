import sys

import pandas as pd
from pytrends import dailydata


def main():

    stock_crypto = [
        "stocks/AAPL",
        "stocks/AMZN",
        "stocks/FB",
        "stocks/JPM",
        "crypto/ADA",
        "crypto/BTC",
        "crypto/DOGE",
        "crypto/ETH",
        "crypto/SOL",
    ]
    # keyword codes can be taken from the end of the google trends URL you are interested in
    # unable to find good topics/terms for solana so it will be excluded
    keywords = [
        "/m/0k8z",
        "/m/0mgkg",
        "/m/0hmyfsv",
        "/m/01hlwv",
        "/m/0svqxy7",
        "/m/11gf2dcwbj",
        "/m/05p0rrx",
        "/m/0zmxk9t",
        "/m/0108bn2x",
    ]
    for x, y in zip(stock_crypto, keywords):
        daily = dailydata.get_daily_data(
            y, start_year=2018, start_mon=1, stop_year=2022, stop_mon=2
        )
        df = pd.DataFrame(daily)
        df.to_csv(f"data/{x}_trends.csv")


if __name__ == "__main__":
    sys.exit(main())
