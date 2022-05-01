import os
import re
import sys

import numpy as np
import pandas as pd
from plotly import express as px
from plotly import graph_objects as go
from plotly.offline import plot
from plotly.subplots import make_subplots
from scipy import stats


def main():
    all_files = [
        "data/crypto/ADA_features.csv",
        "data/crypto/BTC_features.csv",
        "data/crypto/DOGE_features.csv",
        "data/crypto/ETH_features.csv",
        "data/crypto/SOL_features.csv",
        "data/DJI_features.csv",
        "data/GSPC_features.csv",
        "data/NDX_features.csv",
    ]

    adj_close = pd.DataFrame()
    log_returns = pd.DataFrame()
    volatility = pd.DataFrame()
    volume = pd.DataFrame()

    for file in all_files:
        m = re.findall(r"/([A-Z]+)_", file)
        fa_name = m[0]
        df = pd.read_csv(file, parse_dates=["Date"], index_col=["Date"])
        adj_close[fa_name] = df["Adj Close"]
        log_returns[fa_name] = df["Log Returns"]
        volatility[fa_name] = df["Volatility (5 Day)"]
        volume[fa_name] = df["Previous Volume"]

    fac_list = [
        "adj_close",
        "log_returns",
        "volatility",
        "volume",
    ]
    i = 0

    os.chdir("docs/plots")

    for df in [
        adj_close,
        log_returns,
        volatility,
        volume,
    ]:
        df = df.dropna()
        stock_ind = ["DJI", "GSPC", "NDX"]
        crypto = ["ADA", "BTC", "DOGE", "ETH", "SOL"]
        corr_matrix = pd.DataFrame(columns=crypto, index=stock_ind)
        p_corr = pd.DataFrame(columns=crypto, index=stock_ind)
        tlcc_matrix = pd.DataFrame(columns=crypto, index=stock_ind)
        off_matrix = pd.DataFrame(columns=crypto, index=stock_ind)
        for fa1 in stock_ind:
            for fa2 in crypto:
                # Plotting
                # making tlcc charts
                rs = [crosscorr(df[fa1], df[fa2], lag) for lag in range(-63, 64)]
                offset = np.argmax(rs) - 63
                tlcc_fig = px.line(
                    df,
                    x=range(-63, 64),
                    y=rs,
                    title=f"{fa1} vs {fa2} TLCC (Offset = {offset} days)",
                )
                tlcc_fig.add_vline(
                    x=offset, line_width=3, line_dash="dash", line_color="red"
                )
                tlcc_fig.add_vline(
                    x=0, line_width=3, line_dash="dash", line_color="black"
                )
                filename = f"{fac_list[i]}_{fa1}_{fa2}_tlcc.html"
                tlcc_fig.write_html(filename)
                off_matrix.at[fa1, fa2] = offset
                tlcc_matrix.at[fa1, fa2] = "plots/" + filename
                # Correlation Stats
                cont_cont_corr, p = stats.pearsonr(df[fa1], df[fa2])
                # Put value in correlation matrix
                corr_matrix.at[fa1, fa2] = abs(cont_cont_corr)
                p_corr.at[fa1, fa2] = p
        pearson_fig = make_subplots(1, 2, horizontal_spacing=0.15)
        pearson_fig.add_trace(
            go.Heatmap(
                z=corr_matrix.values.tolist(),
                x=corr_matrix.columns.tolist(),
                y=corr_matrix.index.tolist(),
                zmin=0,
                zmax=1,
                texttemplate="%{z:.2f}",
                colorscale="mint",
                colorbar_x=0.45,
            ),
            1,
            1,
        )
        pearson_fig.add_trace(
            go.Heatmap(
                z=p_corr.values.tolist(),
                x=corr_matrix.columns.tolist(),
                y=corr_matrix.index.tolist(),
                zmin=0,
                zmax=0.1,
                texttemplate="%{z:.2f}",
                colorscale="tealrose",
            ),
            1,
            2,
        )

        pearson_fig.write_html(f"{fac_list[i]}_corr_matrix.html")

        fig = go.Figure(
            data=go.Heatmap(
                z=off_matrix.values.tolist(),
                x=off_matrix.columns.tolist(),
                y=off_matrix.index.tolist(),
                zmin=-63,
                zmax=63,
                texttemplate="%{z}",
                colorscale="tealrose",
            )
        )
        fig.update(data=[{"customdata": tlcc_matrix}])

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
        with open(f"{fac_list[i]}_tlcc_matrix.html", "w") as f:
            f.write(html_str)
        i = i + 1


def crosscorr(datax, datay, lag=0, wrap=False):
    """Lag-N cross correlation.
    Shifted data filled with NaNs

    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : float
    """
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else:
        return datax.corr(datay.shift(lag))


if __name__ == "__main__":
    sys.exit(main())
