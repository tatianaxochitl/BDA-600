import sys

import pandas as pd


def main():
    all_files = [
        "data/crypto/ADA.csv",
        "data/crypto/BTC.csv",
        "data/crypto/DOGE.csv",
        "data/crypto/ETH.csv",
        "data/crypto/SOL.csv",
        "data/stocks/AAPL.csv",
        "data/stocks/AMZN.csv",
        "data/stocks/FB.csv",
        "data/stocks/JPM.csv",
        "data/stocks/NFLX.csv",
    ]
    idx_files = ["data/DJI.csv", "data/GSPC.csv", "data/NDX.csv"]

    dji_df = pd.DataFrame()
    gspc_df = pd.DataFrame()
    ndx_df = pd.DataFrame()

    idx_dfs = [dji_df, gspc_df, ndx_df]
    for file, i in zip(idx_files, range(3)):
        df = pd.read_csv(file)
        idx_dfs[i] = create_features(df)
        save_features_file(file, idx_dfs[i])

    for file in all_files:
        df = pd.read_csv(file)
        new_df = create_features(df)
        new_df = compare_dfs(
            new_df,
            idx_dfs,
            [
                "DJI",
                "GSPC",
                "NDX",
            ],
        )
        save_features_file(file, new_df)


# create new features
def create_features(df: pd.DataFrame):

    # High Minus Low
    df["High Minus Low"] = df["High"] - df["Low"]

    # Close Minus Open
    df["Close Minus Open"] = df["Close"] - df["Open"]

    # 7 day Moving average for close
    df["7 day MA"] = df["Close"].rolling(7).mean()

    # 14 day Moving average for close
    df["14 day MA"] = df["Close"].rolling(14).mean()

    # 21 day Moving average for close
    df["21 day MA"] = df["Close"].rolling(21).mean()

    # Day Change in Close price
    df["Day Change"] = df["Close"] - df["Close"].shift(1)
    df["Day Change"] = (df["Day Change"] / (df["Close"].shift(1))) * 100

    # Day Change in Close price
    df["Volume Change"] = df["Volume"] - df["Volume"].shift(1)
    df["Volume Change"] = (df["Volume Change"] / (df["Volume"].shift(1))) * 100

    # 7 days standard dev
    arr_size = df["Close"].size
    df["7 Day STD DEV"] = ""
    for x in range(7, arr_size):
        # find previous index to look at
        y = x - 7
        mini_df = df["Close"][y:x]
        df.loc[x - 1, "7 Day STD DEV"] = mini_df.std()

    # 14 days standard dev
    df["14 Day STD DEV"] = ""
    for x in range(14, arr_size):
        # find previous index to look at
        y = x - 14
        mini_df = df["Close"][y:x]
        df.loc[x - 1, "7 Day STD DEV"] = mini_df.std()

    # 21 days standard dev
    df["21 Day STD DEV"] = ""

    for x in range(21, arr_size):
        # find previous index to look at
        y = x - 21
        mini_df = df["Close"][y:x]
        df.loc[x - 1, "7 Day STD DEV"] = mini_df.std()

    # shifting columns so that close is our result value
    # and everything else is an estimator that only uses
    # previous data
    df = shift_columns(df)
    return df


def shift_columns(df: pd.DataFrame):
    df["Date"] = df["Date"].shift(-1)
    df = df.rename(
        columns={
            "Close": "Previous Close",
            "Open": "Previous Open",
            "Volume": "Previous Volume",
            "High": "Previous High",
            "Low": "Previous Low",
        }
    )
    df["Close"] = df["Previous Close"].shift(1)
    df.drop(columns="Adj Close")
    return df


def save_features_file(file: str, new_df: pd.DataFrame):
    substr = ".csv"
    inserttxt = "_features"
    idx = file.index(substr)
    new_file = file[:idx] + inserttxt + file[idx:]
    new_df.to_csv(new_file)


def compare_dfs(first_df: pd.DataFrame, index_dfs: list, index_names: list):
    for idx_df, idx_name in zip(index_dfs, index_names):
        first_df[f"{idx_name} Change in Close Comparison"] = (
            first_df["Day Change"] - idx_df["Day Change"]
        )
        first_df[f"{idx_name} Change in Close Comparison"] = (
            first_df[f"{idx_name} Change in Close Comparison"] / idx_df["Day Change"]
        )
        first_df[f"{idx_name} Change in Volume Comparison"] = (
            first_df["Volume Change"] - idx_df["Volume Change"]
        ) / idx_df["Volume Change"]
        first_df[f"{idx_name} Change in Volume Comparison"] = (
            first_df[f"{idx_name} Change in Volume Comparison"]
            / idx_df["Volume Change"]
        )

    return first_df


if __name__ == "__main__":
    sys.exit(main())
