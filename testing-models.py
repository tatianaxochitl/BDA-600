# from sklearn.model_selection import GridSearchCV
import re
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error as mse
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

warnings.filterwarnings("ignore")


###############################################################
# SVR GARCH Testing
###############################################################
# previous code for testing different parameters
# svr_poly = SVR(kernel='poly', degree=2)
# svr_lin = SVR(kernel='linear')
# svr_rbf = SVR(kernel='rbf')

# parameters = {
#     "kernel": ["linear"],
#     "C": [1,10,10,100,1000],
#     "gamma": [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
#     }

# clf = GridSearchCV(SVR(), parameters, cv=5, verbose=0)
# print("best parameters are: ", clf.best_params_)
###############################################################
# ANN GARCH Testing
###############################################################
# previous code for testing different parameters

# parameters = {
#     'learning_rate_init': [0.001],
#     'random_state': [1],
#     'hidden_layer_sizes': [(100, 50), (50, 50), (10, 100)],
#     'max_iter': [500, 1000],
#     'alpha': [0.00005, 0.0005 ]
#     }

# clf = GridSearchCV(MLPRegressor(), parameters, cv=5, verbose=0)
# print("best parameters are: ", clf.best_params_)
###############################################################
all_files = [
    "data/crypto/ADA_features.csv",
    "data/crypto/BTC_features.csv",
    "data/crypto/DOGE_features.csv",
    "data/crypto/ETH_features.csv",
    "data/crypto/SOL_features.csv",
    "data/stocks/AAPL_features.csv",
    "data/stocks/AMZN_features.csv",
    "data/stocks/FB_features.csv",
    "data/stocks/JPM_features.csv",
    "data/stocks/TSLA_features.csv",
]

var_vol = [
    "Volatility (5 Day)",
    "Volatility (21 Day)",
    "Volatility (63 Day)",
]

rmse_df = pd.DataFrame(
    columns=["Financial Asset", "RMSE Value", "Volatility Span", "Model"]
)

for file in all_files:
    df = pd.read_csv(file)
    name = re.findall(r".+/(.+)\.csv", file)
    better_name = re.findall(r"(.+)_features", name[0])
    for v_stat in var_vol:
        mini_df = df[["Date", "Sq Returns", v_stat]]
        mini_df["Actual"] = mini_df[v_stat].shift(1)
        mini_df = mini_df.dropna()
        mini_df = mini_df.reset_index()
        n = round(mini_df.shape[0] / 5)
        svr_clf = SVR(kernel="linear", gamma=1e-08, C=1000, epsilon=0.001)
        ann_clf = MLPRegressor(
            learning_rate_init=0.001,
            random_state=1,
            hidden_layer_sizes=(10, 100),
            alpha=5e-05,
        )
        svr_clf.fit(
            mini_df[["Sq Returns", v_stat]].iloc[:-n].values,
            mini_df["Actual"].iloc[:-n].values,
        )
        ann_clf.fit(
            mini_df[["Sq Returns", v_stat]].iloc[:-n].values,
            mini_df["Actual"].iloc[:-n].values,
        )
        mini_df["SVR Prediction"] = ""
        mini_df["ANN Prediction"] = ""
        mini_df["SVR Prediction"].iloc[-n:] = svr_clf.predict(
            mini_df[["Sq Returns", v_stat]].iloc[-n:]
        )
        mini_df["ANN Prediction"].iloc[-n:] = ann_clf.predict(
            mini_df[["Sq Returns", v_stat]].iloc[-n:]
        )
        rmse_svr = np.sqrt(
            mse(
                mini_df["Actual"].iloc[-n:] / 100,
                mini_df["SVR Prediction"].iloc[-n:] / 100,
            )
        )
        rmse_df.loc[len(rmse_df.index)] = [better_name[0], rmse_svr, v_stat, "SVR"]
        rmse_ann = np.sqrt(
            mse(
                mini_df["Actual"].iloc[-n:] / 100,
                mini_df["ANN Prediction"].iloc[-n:] / 100,
            )
        )
        rmse_df.loc[len(rmse_df.index)] = [better_name[0], rmse_ann, v_stat, "ANN"]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=mini_df["Date"], y=mini_df["Actual"], mode="lines", name="Actual"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=mini_df["Date"],
                y=mini_df["SVR Prediction"],
                mode="lines",
                name="SVR (RMSE:{score:.2e})".format(score=rmse_svr),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=mini_df["Date"],
                y=mini_df["ANN Prediction"],
                mode="lines",
                name="ANN (RMSE:{score:.2e})".format(score=rmse_ann),
            )
        )
        fig.update_layout(
            title=f"{v_stat} Prediction Using SVR-GARCH vs ANN-GARCH",
            xaxis_title="Date",
            yaxis_title=v_stat,
        )
        fig.write_html(f"docs/plots/{name[0]}_{v_stat}_GARCH_Prediction.html")

fig = px.scatter(
    rmse_df,
    x="Financial Asset",
    y="RMSE Value",
    color="Model",
    hover_data=["Volatility Span"],
)
fig.write_html("docs/plots/RMSE.html")
