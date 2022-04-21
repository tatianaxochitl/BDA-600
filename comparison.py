import itertools
import os
import re
import sys

import pandas as pd
from plotly import express as px
from plotly.offline import plot
from scipy import stats

from cont_cont import linear_regression


def main():
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
        "data/DJI_features.csv",
        "data/GSPC_features.csv",
        "data/NDX_features.csv",
    ]

    adj_close = pd.DataFrame()
    log_returns = pd.DataFrame()
    volatility = pd.DataFrame()

    for file in all_files:
        m = re.findall(r"/([A-Z]+)_", file)
        fa_name = m[0]
        df = pd.read_csv(file, parse_dates=["Date"], index_col=["Date"])
        adj_close[fa_name] = df["Adj Close"]
        log_returns[fa_name] = df["Log Returns"]
        volatility[fa_name] = df["Volatility (5 Day)"]

    fac_list = ["adj_close", "log_returns", "volatility"]
    i = 0

    os.chdir("docs/plots")

    for df in [adj_close, log_returns, volatility]:
        df = df.dropna()
        corr_matrix = pd.DataFrame(columns=adj_close.columns, index=adj_close.columns)
        lr_matrix = pd.DataFrame(columns=adj_close.columns, index=adj_close.columns)
        fa_list = itertools.combinations(adj_close.columns, 2)
        for fa1, fa2 in fa_list:
            # Plotting
            filename = linear_regression(df, fa1, fa2)
            lr_matrix.at[fa1, fa2] = "plots/" + filename
            lr_matrix.at[fa2, fa1] = "plots/" + filename
            # Correlation Stats
            cont_cont_corr, p = stats.pearsonr(df[fa1], df[fa2])
            # Put value in correlation matrix
            corr_matrix.at[fa1, fa2] = cont_cont_corr
            corr_matrix.at[fa2, fa1] = cont_cont_corr

        corr_matrix = corr_matrix.fillna(value=1)
        fig = px.imshow(
            corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            zmin=0,
            zmax=1,
        )
        fig.update(data=[{"customdata": lr_matrix}])

        plot_div = plot(fig, output_type="div", include_plotlyjs=True)
        # Get id of html div element that looks like
        # <div id="301d22ab-bfba-4621-8f5d-dc4fd855bb33" ... >
        res = re.search('<div id="([^"]*)"', plot_div)
        div_id = res.groups()[0]

        # Build JavaScript callback for handling clicks
        # and opening the URL in the trace's customdata
        js_callback = """
        <script>
        var plot_element = document.getElementById("{div_id}");
        plot_element.on('plotly_click', function(data){{
            console.log(data);
            var point = data.points[0];
            if (point) {{
                console.log(point.customdata);
                window.open(point.customdata);
            }}
        }})
        </script>
        """.format(
            div_id=div_id
        )

        # Build HTML string
        html_str = """
        <html>
        <body>
        {plot_div}
        {js_callback}
        </body>
        </html>
        """.format(
            plot_div=plot_div, js_callback=js_callback
        )

        # Write out HTML file
        with open(f"{fac_list[i]}_corr_matrix.html", "w") as f:
            f.write(html_str)
        i = i + 1


if __name__ == "__main__":
    sys.exit(main())
