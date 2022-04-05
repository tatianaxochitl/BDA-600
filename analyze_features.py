import itertools
import os
import re

import numpy as np
import pandas as pd
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.ensemble import RandomForestClassifier

from cont_cont import cont_cont_dwm, linear_regression


def process_dataframe(
    pandas_df: pd.DataFrame, predictor_columns: list, response_column: str
):
    # set up things create dir for plots
    path = os.path.join(os.getcwd(), "docs/plots")
    if not os.path.exists(path):
        os.mkdir(path)

    cont_list = predictor_columns

    # Continuous/Continuous
    cont_cont_df = pd.DataFrame(
        columns=[
            "Predictors",
            "Pearson's r",
            "Absolute Value of Correlation",
            "Linear Regression Plot",
        ]
    )
    pred_cont_list_comb = itertools.combinations(cont_list, 2)
    cont_cont_corr_matrix = pd.DataFrame(columns=cont_list, index=cont_list)
    cont_cont_diff = pd.DataFrame(
        columns=[
            "Predictor 1",
            "Predictor 2",
            "Difference of Mean Response",
            "Weighted Difference of Mean Response",
            "Bin Plot",
            "Residual Plot",
        ]
    )
    for pred1, pred2 in pred_cont_list_comb:
        # Plotting
        file = linear_regression(pandas_df, pred1, pred2)
        # Correlation Stats
        cont_cont_corr, p = stats.pearsonr(pandas_df[pred1], pandas_df[pred2])
        # Put value in correlation matrix
        cont_cont_corr_matrix.at[pred1, pred2] = cont_cont_corr
        cont_cont_corr_matrix.at[pred2, pred1] = cont_cont_corr
        # Put correlation value and plot into table
        new_row = {
            "Predictors": f"{pred1} and {pred2}",
            "Pearson's r": cont_cont_corr,
            "Absolute Value of Correlation": abs(cont_cont_corr),
            "Linear Regression Plot": file,
        }
        cont_cont_df = cont_cont_df.append(new_row, ignore_index=True)
        # Brute Force
        file1, file2, diff, w_diff = cont_cont_dwm(
            pandas_df, pred1, pred2, response_column
        )
        new_row = {
            "Predictor 1": pred1,
            "Predictor 2": pred2,
            "Difference of Mean Response": diff,
            "Weighted Difference of Mean Response": w_diff,
            "Bin Plot": file1,
            "Residual Plot": file2,
        }
        cont_cont_diff = cont_cont_diff.append(new_row, ignore_index=True)

    # sort dataframe by abs val of corr
    cont_cont_df = cont_cont_df.sort_values(
        by="Absolute Value of Correlation", ascending=False
    )

    # fill empty with 1
    cont_cont_corr_matrix = cont_cont_corr_matrix.fillna(value=1)

    # sort dataframes by weighted dwmor
    cont_cont_diff = cont_cont_diff.sort_values(
        by="Weighted Difference of Mean Response", ascending=False
    )

    # sort all by Random Forest Importance
    # all_pred_df = all_pred_df.sort_values(
    #     by="Random Forest Importance", ascending=False
    # ) this is a note to add rf to cont_cont !!!

    # make clickable links for all plots

    if len(cont_list) != 0:
        cont_cont_df["Linear Regression Plot"] = make_html_link(
            cont_cont_df["Linear Regression Plot"]
        )
        cont_cont_diff["Bin Plot"] = make_html_link(cont_cont_diff["Bin Plot"])
        cont_cont_diff["Residual Plot"] = make_html_link(
            cont_cont_diff["Residual Plot"]
        )  # noqa: E80

    # create html file
    page = open("stock_crypto_comparison.html", "w")
    page.write("<h2>Predictor Ranking</h2>")
    # page.write(
    #     all_pred_df.to_html(escape=False, index=False, justify="center")
    # )  # noqa: E501
    if len(cont_list) != 0:
        page.write("<h3>Correlation Table</h3>")
        page.write(
            cont_cont_df.to_html(escape=False, index=False, justify="center")
        )  # noqa: E501
        page.write("<h3>Correlation Matrix</h3>")
        page.write(make_heatmap_html(cont_cont_corr_matrix))
        page.write('<h3>"Brute Force" Table</h3>')
        page.write(
            cont_cont_diff.to_html(escape=False, index=False, justify="center")
        )  # noqa: E501
    return


def make_heatmap_html(matrix: pd.DataFrame):
    fig = go.Figure(
        data=go.Heatmap(
            x=matrix.columns,
            y=matrix.index,
            z=matrix.values,
            zmin=0,
            zmax=1,
            colorscale="curl",
        )
    )
    matrix_html = fig.to_html()
    return matrix_html


def cont_dwm(pandas_df, predictor, response):
    mean, edges, bin_number = stats.binned_statistic(
        pandas_df[predictor], pandas_df[response], statistic="mean", bins=10
    )
    count, edges, bin_number = stats.binned_statistic(
        pandas_df[predictor], pandas_df[response], statistic="count", bins=10
    )
    pop_mean = np.mean(pandas_df[response])
    edge_centers = (edges[:-1] + edges[1:]) / 2
    mean_diff = mean - pop_mean
    mdsq = mean_diff ** 2
    pop_prop = count / len(pandas_df[response])
    wmdsq = pop_prop * mdsq
    msd = np.nansum(mdsq) / 10
    wmsd = np.sum(wmdsq)
    pop_mean_list = [pop_mean] * 10

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=edge_centers,
            y=mean_diff,
            name="$\\mu_{i}$ - $\\mu_{pop}$",
            mode="lines+markers",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Bar(x=edge_centers, y=count, name="Population"), secondary_y=True
    )  # noqa: E501
    fig.add_trace(
        go.Scatter(
            y=pop_mean_list,
            x=edge_centers,
            mode="lines",
            name="$\\mu_{pop}$",
        )
    )

    filename = f"plots/{predictor}_{response}_dwm.html"

    fig.write_html(
        file=filename,
        include_plotlyjs="cdn",
    )

    return filename, msd, wmsd


def rf_importance(df, predictors, response):
    df_X = df[predictors]
    df_y = df[response]
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(df_X, df_y)
    importances = clf.feature_importances_
    return importances.tolist()


def make_html_link(plot_col: pd.Series):
    # regex for making link text
    regex = ".+/([^/]+).html$"
    for x in range(len(plot_col)):
        text = re.findall(regex, plot_col[x])
        link_html = (
            f'<a target="_blank" href="{plot_col[x]}">{text[0]}</a>'  # noqa: E501
        )
        plot_col[x] = link_html
    return plot_col
