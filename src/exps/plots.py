# ruff: noqa: E501, PD010, PD013, PD901
from __future__ import annotations

import json
import operator
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from more_itertools import batched, first_true

_colors = iter(plt.get_cmap("tab10").colors)  # type: ignore
MARKER_SIZE = 10
LEGEND_MAX_COLS = 5
LEGEND_FONTSIZE = "xx-large"
TICK_FONTSIZE = "large"
AXIS_LABEL_FONTSIZE = "xx-large"
FIG_TITLE_FONTSIZE = "xx-large"
AXIS_TITLE_FONTSIZE = "x-large"
COLORS = {
    "disabled": next(_colors),
    "current_average_worse_than_mean_best": next(_colors),
    "current_average_worse_than_best_worst_split": next(_colors),
    "robust_std_top_3": next(_colors),
    "robust_std_top_5": next(_colors),
    "both": next(_colors),
}
COLORS.update(
    {
        "smac_early_stop_as_failed-disabled": COLORS["disabled"],
        "smac_early_stop_as_failed-current_average_worse_than_mean_best": COLORS[
            "current_average_worse_than_mean_best"
        ],
        "smac_early_stop_as_failed-current_average_worse_than_best_worst_split": COLORS[
            "current_average_worse_than_best_worst_split"
        ],
        "smac_early_stop_as_failed-robust_std_top_3": COLORS["robust_std_top_3"],
        "smac_early_stop_as_failed-robust_std_top_5": COLORS["robust_std_top_5"],
        # --- #
        "smac_early_stop_with_fold_mean-disabled": COLORS["disabled"],
        "smac_early_stop_with_fold_mean-current_average_worse_than_mean_best": COLORS[
            "current_average_worse_than_mean_best"
        ],
        "smac_early_stop_with_fold_mean-current_average_worse_than_best_worst_split": COLORS[
            "current_average_worse_than_best_worst_split"
        ],
        "smac_early_stop_with_fold_mean-robust_std_top_3": COLORS["robust_std_top_3"],
        "smac_early_stop_with_fold_mean-robust_std_top_5": COLORS["robust_std_top_5"],
    },
)
# https://matplotlib.org/stable/gallery/lines_bars_and_markers/marker_reference.html#filled-markers
MARKERS = {
    "disabled": "o",
    "current_average_worse_than_mean_best": "h",
    "current_average_worse_than_best_worst_split": "H",
    "robust_std_top_3": "X",
    "robust_std_top_5": "P",
    "both": "s",
    # --- #
}
MARKERS.update(
    {
        "smac_early_stop_as_failed-disabled": MARKERS["disabled"],
        "smac_early_stop_as_failed-current_average_worse_than_mean_best": MARKERS[
            "current_average_worse_than_mean_best"
        ],
        "smac_early_stop_as_failed-current_average_worse_than_best_worst_split": MARKERS[
            "current_average_worse_than_best_worst_split"
        ],
        "smac_early_stop_as_failed-robust_std_top_3": MARKERS["robust_std_top_3"],
        "smac_early_stop_as_failed-robust_std_top_5": MARKERS["robust_std_top_5"],
        # --- #
        "smac_early_stop_with_fold_mean-disabled": MARKERS["disabled"],
        "smac_early_stop_with_fold_mean-current_average_worse_than_mean_best": MARKERS[
            "current_average_worse_than_mean_best"
        ],
        "smac_early_stop_with_fold_mean-current_average_worse_than_best_worst_split": MARKERS[
            "current_average_worse_than_best_worst_split"
        ],
        "smac_early_stop_with_fold_mean-robust_std_top_3": MARKERS["robust_std_top_3"],
        "smac_early_stop_with_fold_mean-robust_std_top_5": MARKERS["robust_std_top_5"],
    },
)
LINE_STYLES = {
    "smac_early_stop_as_failed-disabled": "solid",
    "smac_early_stop_as_failed-current_average_worse_than_mean_best": "solid",
    "smac_early_stop_as_failed-current_average_worse_than_best_worst_split": "solid",
    "smac_early_stop_as_failed-robust_std_top_3": "solid",
    "smac_early_stop_as_failed-robust_std_top_5": "solid",
    # --- #
    "smac_early_stop_with_fold_mean-disabled": "dashed",
    "smac_early_stop_with_fold_mean-current_average_worse_than_mean_best": "dashed",
    "smac_early_stop_with_fold_mean-current_average_worse_than_best_worst_split": "dashed",
    "smac_early_stop_with_fold_mean-robust_std_top_3": "dashed",
    "smac_early_stop_with_fold_mean-robust_std_top_5": "dashed",
}

X_TICKS = {
    (0, 3600): [0, 600, 1200, 1800, 2400, 3000, 3600],
}
RENAMES: dict[str, str] = {
    "disabled": "No Early Stopping",
    "current_average_worse_than_mean_best": "Mean Of Best",
    "current_average_worse_than_best_worst_split": "Worst Fold of Best",
    "robust_std_top_3": "Robust 3",
    "robust_std_top_5": "Robust 5",
    # --- #
}
RENAMES.update(
    {
        "smac_early_stop_as_failed-disabled": RENAMES["disabled"] + " (failed)",
        "smac_early_stop_as_failed-current_average_worse_than_mean_best": RENAMES[
            "current_average_worse_than_mean_best"
        ]
        + " (failed)",
        "smac_early_stop_as_failed-current_average_worse_than_best_worst_split": RENAMES[
            "current_average_worse_than_best_worst_split"
        ]
        + " (failed)",
        "smac_early_stop_as_failed-robust_std_top_3": RENAMES["robust_std_top_3"]
        + " (failed)",
        "smac_early_stop_as_failed-robust_std_top_5": RENAMES["robust_std_top_5"]
        + " (failed)",
        # --- #
        "smac_early_stop_with_fold_mean-disabled": RENAMES["disabled"] + " (mean)",
        "smac_early_stop_with_fold_mean-current_average_worse_than_mean_best": RENAMES[
            "current_average_worse_than_mean_best"
        ]
        + " (mean)",
        "smac_early_stop_with_fold_mean-current_average_worse_than_best_worst_split": RENAMES[
            "current_average_worse_than_best_worst_split"
        ]
        + " (mean)",
        "smac_early_stop_with_fold_mean-robust_std_top_3": RENAMES["robust_std_top_3"]
        + " (mean)",
        "smac_early_stop_with_fold_mean-robust_std_top_5": RENAMES["robust_std_top_5"]
        + " (mean)",
    },
)


def markup_speedup_summary_table(
    df: pd.DataFrame,
    *,
    bigger_is_better: bool,
    n_datasets: int,
) -> pd.DataFrame:
    df.index.name = "Method"
    df.index.map(RENAMES.get)
    df = df.sort_index()

    # Mean \\pm Std with bolding of highest speedup
    mean_std = df["mean"].round(0).astype(str) + "\\pm" + df["std"].round(0).astype(str)
    idx = df["mean"].idxmax() if bigger_is_better else df["mean"].idxmin()
    mean_std.loc[idx] = "\\textbf{" + mean_std.loc[idx] + "}"
    mean_std = "$" + mean_std + "$"
    mean_std.name = "Average Speedup \\%"

    # Count failed with highlighting of least failed
    count_failed = df["Datasets Failed"].astype(int).astype(str)
    idx = count_failed.idxmin()
    count_failed.loc[idx] = "\\textbf{" + count_failed.loc[idx] + "}"

    count_failed = "$" + count_failed + f" /{n_datasets}$"
    return pd.concat([mean_std, count_failed], axis=1)


def markup_speedup_table_full(df: pd.DataFrame) -> pd.DataFrame:
    df.index.name = "Dataset (OpenML Task ID)"
    df = df.rename(columns=RENAMES)
    df = df.sort_index()

    # Sort the columns by method names
    df = df[sorted(df.columns)]

    # Mean \\pm Std with bolding of highest speedup
    df.columns.name = "Method"

    # Convert all to percentage
    df = df.round(0).astype(str) + r"\%"

    # Replace NaN with r"\textbf{---}"
    return df.where(df != r"nan\%", r"\textbf{---}")


def _dataset_stats(path: Path | str | None = None) -> pd.DataFrame:
    if path is None:
        path = Path("openml_suite_271.json")
    elif isinstance(path, str):
        path = Path(path)

    with path.open("r") as f:
        data = json.load(f)

    _df = pd.DataFrame(data)
    _df.index = _df.index.astype(int)
    _df.astype(int)
    return _df


def _inc_trace(
    df: pd.DataFrame,
    *,
    x_start_col: str,
    x_col: str,
    y_col: str,
    minimize: bool,
    test_y_col: str,
) -> pd.Series:
    # We now have each individual std group to plot, i.e. the fold
    _start = df[x_start_col].min()
    _x = (df[x_col] - _start).dt.total_seconds()

    ind = pd.Index(_x, name="time (s)")
    _df = (
        df[[y_col, test_y_col]]
        .rename(columns={y_col: "y", test_y_col: "test_y"})
        .set_index(ind)
        .sort_index()
        .dropna()
    )

    # Transform everything
    match minimize:
        case True:
            _df["cumulative"] = _df["y"].cummin()
        case False:
            _df["cumulative"] = _df["y"].cummax()

    _df = _df.drop_duplicates(subset="cumulative", keep="first").drop(
        columns="cumulative",
    )

    return pd.Series(
        # We do this concat operation so that data is contiguous for later opertions
        data=np.concatenate(
            [_df["y"].to_numpy(), _df["test_y"].to_numpy()],
        ),
        index=pd.MultiIndex.from_product([["val", "test"], _df.index]),
    )


def ranking_plots_aggregated(  # noqa: PLR0913
    df: pd.DataFrame,
    y: str,
    test_y: str,
    x: str,
    x_start: str,
    fold: str,
    method: str,
    dataset: str,  # TODO: list
    title: str,
    *,
    minimize: bool = False,
    x_bounds: tuple[int | float | None, int | float | None] | None = None,
    fig_ax: tuple[plt.Figure, plt.Axes] | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    markevery: int | float | None = None,
) -> plt.Axes:
    def incumbent_trace(_x: pd.DataFrame) -> pd.Series:
        return _inc_trace(
            _x,
            x_start_col=x_start,
            x_col=x,
            y_col=y,
            test_y_col=test_y,
            minimize=minimize,
        )

    """
    setting:task                                                    146818  146820  168350  168757  168784  168910  168911  189922  ...  359968  359969  359970  359971  359972  359973  359975  359979
    setting:cv_early_stop_strategy              values time (s)                                                                     ...
    current_average_worse_than_best_worst_split test   21.595531       2.0     5.0     4.0     2.0     5.0     4.0     1.0     1.0  ...     3.0     2.0     2.0     1.0     2.0     1.0     1.0     2.0
                                                       23.308698       2.0     5.0     4.0     2.0     5.0     4.0     1.0     1.0  ...     3.0     2.0     2.0     1.0     2.0     1.0     1.0     2.0
                                                       23.412293       2.0     5.0     4.0     2.0     5.0     4.0     1.0     1.0  ...     3.0     2.0     2.0     1.0     2.0     1.0     1.0     2.0
                                                       27.131684       2.0     5.0     4.0     2.0     5.0     4.0     1.0     1.0  ...     3.0     2.0     2.0     1.0     2.0     1.0     1.0     2.0
                                                       29.640223       2.0     5.0     4.0     2.0     5.0     4.0     1.0     1.0  ...     3.0     2.0     2.0     1.0     2.0     1.0     1.0     2.0
    ...                                                                ...     ...     ...     ...     ...     ...     ...     ...  ...     ...     ...     ...     ...     ...     ...     ...     ...
    robust_std_top_5                            val    3576.123217     1.0     1.0     2.5     1.0     1.0     3.0     2.0     1.0  ...     1.0     5.0     4.0     2.0     1.0     4.0     3.0     3.0
                                                       3577.107590     1.0     1.0     2.5     1.0     1.0     3.0     2.0     1.0  ...     1.0     1.0     4.0     2.0     1.0     4.0     3.0     3.0
                                                       3583.088503     1.0     1.0     2.5     1.0     1.0     3.0     2.0     1.0  ...     1.0     5.0     4.0     2.0     1.0     4.0     3.0     3.0
                                                       3589.155218     1.0     1.0     2.5     1.0     1.0     3.0     2.0     1.0  ...     1.0     5.0     4.0     2.0     1.0     4.0     3.0     3.0
                                                       3591.590766     1.0     1.0     2.5     1.0     1.0     3.0     2.0     1.0  ...     1.0     5.0     4.0     2.0     1.0     4.0     3.0     3.0
    """
    ranks_per_dataset = (
        df.groupby([dataset, method, fold], observed=True)
        .apply(incumbent_trace)
        .rename_axis(index={None: "values"})
        .swaplevel(fold, "values")
        .unstack([dataset, method, "values", fold])
        .ffill()
        .dropna()
        .stack([dataset, method, "values"], future_stack=True)
        .mean(axis=1)
        .unstack(method)
        .reorder_levels([dataset, "values", "time (s)"])
        .sort_index()
        .rank(axis=1, ascending=False)
        .unstack(["values", "time (s)"])
        .T
    )

    if fig_ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    else:
        fig, axes = fig_ax

    val_ax = axes[0]
    test_ax = axes[1]

    def extend_to_x_bound(s: pd.Series) -> pd.Series:
        if x_bounds is not None and s.index[-1] < x_bounds[1]:  # type: ignore
            return pd.concat([s, pd.Series([s.iloc[-1]], index=[x_bounds[1]])])
        return s

    legend_lines = []

    groups = ranks_per_dataset.groupby(method, observed=True)
    groups = sorted(groups, key=lambda x: x[0])
    for method_name, _df in groups:
        means = _df.mean(axis=1)
        sems = _df.sem(axis=1)
        _color = COLORS[method_name]  # type: ignore
        _marker = MARKERS[method_name]  # type: ignore
        _style = {
            "marker": _marker,  # type: ignore
            "markersize": MARKER_SIZE,
            "markerfacecolor": "white",
            "markeredgecolor": _color,  # type: ignore
            "color": _color,  # type: ignore
        }
        legend_lines.append(
            Line2D(
                [0],
                [0],
                label=RENAMES.get(method_name, method_name),  # type: ignore
                linestyle=LINE_STYLES.get(method_name, "solid"),  # type: ignore
                linewidth=3,
                **_style,
            ),
        )
        for _y, _ax in zip(("val", "test"), (val_ax, test_ax), strict=True):
            _means = means.loc[method_name, _y]
            _stds = sems.loc[method_name, _y]

            _means = extend_to_x_bound(_means)
            _stds = extend_to_x_bound(_stds)

            label_name = RENAMES.get(method_name, method_name)  # type: ignore

            _means.plot(  # type: ignore
                drawstyle="steps-post",
                label=f"{label_name}",
                ax=_ax,
                linestyle=LINE_STYLES.get(method_name, "solid"),  # type: ignore
                markevery=markevery,
                linewidth=3,
                **_style,
            )
            _ax.fill_between(
                _means.index,  # type: ignore
                _means - _stds,
                _means + _stds,
                alpha=0.2,
                color=_color,
                edgecolor=_color,
                linewidth=2,
                step="post",
            )

    for _ax, _y_set in zip((val_ax, test_ax), ("Validation", "Test"), strict=True):
        if x_bounds:
            _ax.set_xlim(*x_bounds)

            if x_bounds in X_TICKS:
                _ax.set_xticks(X_TICKS[x_bounds])
                _ax.set_xticklabels(
                    [str(x) for x in X_TICKS[x_bounds]],
                    fontsize=TICK_FONTSIZE,
                )

        _ax.set_ylim(1, df[method].nunique())

        _ax.set_xlabel(
            x_label if x_label is not None else x,
            fontsize=AXIS_LABEL_FONTSIZE,
        )
        _ax.set_ylabel(
            y_label if y_label is not None else y,
            fontsize=AXIS_LABEL_FONTSIZE,
        )
        _ax.set_title(_y_set, fontsize=AXIS_TITLE_FONTSIZE)

    fig.suptitle(title, fontsize=FIG_TITLE_FONTSIZE)
    fig.legend(
        loc="upper center",
        handles=legend_lines,
        bbox_to_anchor=(0.5, 0),
        fontsize=LEGEND_FONTSIZE,
        ncols=LEGEND_MAX_COLS,
    )
    fig.tight_layout()
    return fig, axes


def incumbent_traces_aggregated_with_test(  # noqa: PLR0913, C901
    dfs: dict[str, pd.DataFrame],
    y: str,
    test_y: str,
    x: str,
    x_start: str,
    fold: str,
    method: str,
    dataset: str,  # TODO: list
    title: str,
    *,
    minimize: bool = False,
    log_y: bool = False,
    ncols_legend: int | None = None,
    figsize_per_ax: tuple[int, int] = (6, 5),
    x_bounds: tuple[int | float | None, int | float | None] | None = None,
    y_bounds: tuple[int | float | None, int | float | None] | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    invert: bool = False,
    markevery: int | float | None = None,
) -> None:
    nrows = len(dfs)
    ncols = 2
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * figsize_per_ax[0], nrows * figsize_per_ax[1]),
        sharex=True,
        sharey=False,
    )
    legend_lines = []

    for i, ((ax_title, df), (ax_val, ax_test)) in enumerate(
        zip(
            dfs.items(),
            batched(axs.flatten(), n=2, strict=True),
            strict=True,
        ),
    ):

        def incumbent_trace(_x: pd.DataFrame) -> pd.Series:
            return _inc_trace(
                _x,
                x_start_col=x_start,
                x_col=x,
                y_col=y,
                test_y_col=test_y,
                minimize=minimize,
            )

        """
        setting:task                                                      146818    146820    168350    168757    168784    168910  ...    359970    359971    359972    359973    359975    359979
        setting:cv_early_stop_strategy              values time (s)                                                                 ...
        current_average_worse_than_best_worst_split test   274.880387   0.725997  0.467660  0.387187  0.469796  0.840555  0.927984  ...  0.952019  0.445896  0.821756  0.679336  0.605856  0.630766
                                                           275.085706   0.725997  0.467660  0.387187  0.469796  0.840555  0.927984  ...  0.952019  0.445896  0.821756  0.934713  0.605856  0.630766
                                                           275.405484   0.725997  0.467660  0.387187  0.469796  0.840555  0.927984  ...  0.952404  0.445896  0.821756  0.934713  0.605856  0.630766
                                                           275.600901   0.725997  0.467660  0.387187  0.469796  0.840555  0.927984  ...  0.952404  0.445896  0.821756  0.934713  0.617228  0.630766
                                                           275.976421   0.725997  0.467660  0.387187  0.469796  0.840555  0.927984  ...  0.952404  0.445896  0.756635  0.934713  0.617228  0.630766
        ...                                                                  ...       ...       ...       ...       ...       ...  ...       ...       ...       ...       ...       ...       ...
        disabled                                    val    3552.753273  0.034523  0.193127  0.156118  0.010731  0.225250  0.027016  ...  0.315990  0.261195  0.170653  0.187368  0.040620  0.253996
                                                           3559.086472  0.034523  0.193127  0.156118  0.010731  0.225250  0.027016  ...  0.315990  0.261195  0.170653  0.187368  0.040620  0.253996
                                                           3571.056051  0.034523  0.193127  0.156118  0.010731  0.225250  0.027016  ...  0.258106  0.261195  0.170653  0.187368  0.040620  0.253996
                                                           3584.430381  0.034523  0.193127  0.156118  0.010731  0.187467  0.027016  ...  0.258106  0.261195  0.170653  0.187368  0.040620  0.253996
                                                           3598.315723  0.034523  0.193127  0.156118  0.010731  0.187467  0.027016  ...  0.258106  0.244365  0.170653  0.187368  0.040620  0.253996
        """
        inc_traces_per_dataset = (
            df.groupby([dataset, method, fold], observed=True)
            .apply(incumbent_trace)
            .rename_axis(index={None: "values"})
            .swaplevel(fold, "values")  # type: ignore
            .unstack(fold)
            .groupby([dataset, method, "values"], observed=True)
            .ffill()
            .mean(axis=1)
            .unstack(dataset)
            .groupby([method, "values"], observed=True)
            .ffill()
            .dropna()  # We can only being aggregating once we have a value for each dataset
            .unstack("values")
            .transform(lambda x: (x - x.min()) / (x.max() - x.min()))
            .transform(lambda x: 1 - x if invert else x)
            .stack("values")
            .swaplevel("time (s)", "values")
            .sort_index()
        )

        # Needed for y axis scaling
        all_means = inc_traces_per_dataset.mean(axis=1)
        val_mean_min = all_means.loc[pd.IndexSlice[:, "val"]].min()
        test_mean_min = all_means.loc[pd.IndexSlice[:, "test"]].min()

        def extend_to_x_bound(s: pd.Series) -> pd.Series:
            if x_bounds is not None and s.index[-1] < x_bounds[1]:  # type: ignore
                return pd.concat([s, pd.Series([s.iloc[-1]], index=[x_bounds[1]])])
            return s

        groups = inc_traces_per_dataset.groupby(method, observed=True)
        groups = sorted(groups, key=lambda x: x[0])
        for method_name, _df in groups:
            # We dropna across the dataframe so that when we take mean/std, it's only
            # once we have a datapoint for each dataset (~40s)
            means = _df.mean(axis=1)
            sems = _df.sem(axis=1)
            _color = COLORS[method_name]
            _marker = MARKERS[method_name]
            _style = {
                "marker": _marker,
                "markersize": MARKER_SIZE,
                "markerfacecolor": "white",
                "markeredgecolor": _color,
                "color": _color,
            }

            if i == 0:
                legend_lines.append(
                    Line2D(
                        [0],
                        [0],
                        label=RENAMES.get(method_name, method_name),
                        linestyle=LINE_STYLES.get(method_name, "solid"),  # type: ignore
                        linewidth=3,
                        **_style,
                    ),
                )

            for _y, _ax in zip(("val", "test"), (ax_val, ax_test), strict=True):
                _means = means.loc[method_name, _y]
                _stds = sems.loc[method_name, _y]

                _means = extend_to_x_bound(_means)
                _stds = extend_to_x_bound(_stds)

                label_name = RENAMES.get(method_name, method_name)

                _means.plot(  # type: ignore
                    drawstyle="steps-post",
                    label=f"{label_name}",
                    ax=_ax,
                    linestyle=LINE_STYLES.get(method_name, "solid"),  # type: ignore
                    markevery=markevery,
                    linewidth=3,
                    **_style,
                )
                _ax.fill_between(
                    _means.index,  # type: ignore
                    _means - _stds,
                    _means + _stds,
                    alpha=0.2,
                    color=_color,
                    edgecolor=_color,
                    linewidth=2,
                    step="post",
                )

                if x_bounds:
                    _ax.set_xlim(*x_bounds)

                    if x_bounds in X_TICKS:
                        _ax.set_xticks(X_TICKS[x_bounds])
                        _ax.set_xticklabels(
                            [str(x) for x in X_TICKS[x_bounds]],
                            fontsize=TICK_FONTSIZE,
                        )

                if y_bounds:
                    _ax.set_ylim(*y_bounds)
                else:
                    _min = test_mean_min if _y == "test" else val_mean_min
                    lower = first_true(
                        [1 / (10**i) for i in range(10)],
                        pred=lambda low: low < _min,  # type: ignore
                    )
                    _ax.set_ylim(lower, 1)

                if log_y:
                    _ax.set_yscale("log")

                # Define custom formatter function to format tick labels as decimals
                def format_func(value: float, _: int):
                    return f"{value:.2f}"

                # Apply the custom formatter to the y-axis
                _ax.yaxis.set_major_formatter(
                    matplotlib.ticker.FuncFormatter(format_func),
                )
                from matplotlib.ticker import (
                    LogFormatter,
                    NullFormatter,
                )

                _ax.yaxis.set_minor_formatter(LogFormatter(labelOnlyBase=False))
                _ax.yaxis.set_minor_formatter(NullFormatter())
                _ax.tick_params(axis="y", which="both", labelsize=TICK_FONTSIZE)
                _ax.set_title(f"{ax_title} ({_y})", fontsize=AXIS_TITLE_FONTSIZE)

    fig.suptitle(title, fontsize=FIG_TITLE_FONTSIZE)
    fig.supxlabel(
        x_label if x_label is not None else x,
        fontsize=AXIS_LABEL_FONTSIZE,
    )
    fig.supylabel(
        y_label if y_label is not None else y,
        fontsize=AXIS_LABEL_FONTSIZE,
    )
    ncols_legend = ncols_legend if ncols_legend is not None else LEGEND_MAX_COLS

    # Hacky way to sort legend, where baseline is always last
    legend_lines = sorted(
        legend_lines,
        key=lambda line: label
        if (label := line.get_label())
        not in (
            RENAMES.get("disabled"),
            RENAMES.get("smac_early_stop_as_failed-disabled"),
        )
        else "ZZZ",
    )
    fig.legend(
        loc="upper center",
        handles=legend_lines,
        bbox_to_anchor=(0.5, 0),
        fontsize=LEGEND_FONTSIZE,
        ncols=ncols_legend,
    )
    fig.tight_layout()


def incumbent_traces_aggregated_no_test(  # noqa: PLR0913, C901
    dfs: dict[str, pd.DataFrame],
    y: str,
    test_y: str,
    x: str,
    x_start: str,
    fold: str,
    method: str,
    dataset: str,  # TODO: list
    title: str,
    *,
    minimize: bool = False,
    log_y: bool = False,
    nrows: int = 2,
    ncols_legend: int | None = None,
    figsize_per_ax: tuple[int, int] = (6, 5),
    x_bounds: tuple[int | float | None, int | float | None] | None = None,
    y_bounds: tuple[int | float | None, int | float | None] | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    invert: bool = False,
    markevery: int | float | None = None,
) -> None:
    assert len(dfs) % nrows == 0, "Number of datasets must be divisible by nrows"
    ncols = len(dfs) // nrows
    fig, axs = plt.subplots(
        nrows,
        len(dfs) // nrows,
        figsize=(ncols * figsize_per_ax[0], nrows * figsize_per_ax[1]),
        sharex=True,
        sharey=True,
    )
    legend_lines = []

    for i, ((ax_title, df), ax) in enumerate(
        zip(dfs.items(), axs.flatten(), strict=True),
    ):

        def incumbent_trace(_x: pd.DataFrame) -> pd.Series:
            return _inc_trace(
                _x,
                x_start_col=x_start,
                x_col=x,
                y_col=y,
                test_y_col=test_y,
                minimize=minimize,
            )

        inc_traces_per_dataset = (
            df.groupby([dataset, method, fold], observed=True)
            .apply(incumbent_trace)
            .rename_axis(index={None: "values"})
            .swaplevel(fold, "values")  # type: ignore
            .unstack(fold)
            .groupby([dataset, method, "values"], observed=True)
            .ffill()
            .mean(axis=1)
            .unstack(dataset)
            .groupby([method, "values"], observed=True)
            .ffill()
            .dropna()  # We can only being aggregating once we have a value for each dataset
            .unstack("values")
            .transform(lambda x: (x - x.min()) / (x.max() - x.min()))
            .transform(lambda x: 1 - x if invert else x)
            .stack("values")
            .swaplevel("time (s)", "values")
            .sort_index()
        )

        def extend_to_x_bound(s: pd.Series) -> pd.Series:
            if x_bounds is not None and s.index[-1] < x_bounds[1]:  # type: ignore
                return pd.concat([s, pd.Series([s.iloc[-1]], index=[x_bounds[1]])])
            return s

        groups = inc_traces_per_dataset.groupby(method, observed=True)
        groups = sorted(groups, key=lambda x: x[0])
        for method_name, _df in groups:
            # We dropna across the dataframe so that when we take mean/std, it's only
            # once we have a datapoint for each dataset (~40s)
            means = _df.mean(axis=1)
            sems = _df.sem(axis=1)
            _color = COLORS[method_name]
            _marker = MARKERS[method_name]
            _style = {
                "marker": _marker,
                "markersize": MARKER_SIZE,
                "markerfacecolor": "white",
                "markeredgecolor": _color,
                "color": _color,
            }

            if i == 0:
                legend_lines.append(
                    Line2D(
                        [0],
                        [0],
                        label=RENAMES.get(method_name, method_name),
                        linestyle=LINE_STYLES.get(method_name, "solid"),  # type: ignore
                        linewidth=3,
                        **_style,
                    ),
                )
            _means = means.loc[method_name, "val"]
            _stds = sems.loc[method_name, "val"]

            _means = extend_to_x_bound(_means)
            _stds = extend_to_x_bound(_stds)

            label_name = RENAMES.get(method_name, method_name)

            _means.plot(  # type: ignore
                drawstyle="steps-post",
                label=f"{label_name}",
                ax=ax,
                linestyle=LINE_STYLES.get(method_name, "solid"),  # type: ignore
                markevery=markevery,
                linewidth=3,
                **_style,
            )
            ax.fill_between(
                _means.index,  # type: ignore
                _means - _stds,
                _means + _stds,
                alpha=0.2,
                color=_color,
                edgecolor=_color,
                linewidth=2,
                step="post",
            )

        if x_bounds:
            ax.set_xlim(*x_bounds)

            if x_bounds in X_TICKS:
                ax.set_xticks(X_TICKS[x_bounds])
                ax.set_xticklabels(
                    [str(x) for x in X_TICKS[x_bounds]],
                    fontsize=TICK_FONTSIZE,
                )

        if y_bounds:
            ax.set_ylim(*y_bounds)

        if log_y:
            ax.set_yscale("log")

        # Define custom formatter function to format tick labels as decimals
        def format_func(value: float, _: int):
            return f"{value:.2f}"

        # Apply the custom formatter to the y-axis
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_func))
        ax.tick_params(axis="y", which="both", labelsize=TICK_FONTSIZE)

        # ax.set_xlabel(x_label if x_label is not None else x, fontsize=AXIS_LABEL_FONTSIZE)

        ax.set_title(ax_title, fontsize=AXIS_TITLE_FONTSIZE)

    fig.suptitle(title, fontsize=FIG_TITLE_FONTSIZE)
    fig.supxlabel(
        x_label if x_label is not None else x,
        fontsize=AXIS_LABEL_FONTSIZE,
    )
    fig.supylabel(
        y_label if y_label is not None else y,
        fontsize=AXIS_LABEL_FONTSIZE,
    )
    ncols_legend = ncols_legend if ncols_legend is not None else LEGEND_MAX_COLS

    # Hacky way to sort legend, where baseline is always last
    legend_lines = sorted(
        legend_lines,
        key=lambda line: label
        if (label := line.get_label())
        not in (
            RENAMES.get("disabled"),
            RENAMES.get("smac_early_stop_as_failed-disabled"),
        )
        else "ZZZ",
    )
    fig.legend(
        loc="upper center",
        handles=legend_lines,
        bbox_to_anchor=(0.5, 0),
        fontsize=LEGEND_FONTSIZE,
        ncols=ncols_legend,
    )
    fig.tight_layout()


def incumbent_traces_aggregated(  # noqa: PLR0913, C901
    df: pd.DataFrame,
    y: str,
    test_y: str,
    x: str,
    x_start: str,
    fold: str,
    method: str,
    dataset: str,  # TODO: list
    title: str,
    *,
    figsize_per_ax: tuple[int, int] = (6, 5),
    minimize: bool = False,
    log_y: bool = False,
    x_bounds: tuple[int | float | None, int | float | None] | None = None,
    y_bounds: tuple[int | float | None, int | float | None] | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    invert: bool = False,
    markevery: int | float | None = None,
) -> plt.Axes:
    def incumbent_trace(_x: pd.DataFrame) -> pd.Series:
        return _inc_trace(
            _x,
            x_start_col=x_start,
            x_col=x,
            y_col=y,
            test_y_col=test_y,
            minimize=minimize,
        )

    inc_traces_per_dataset = (
        df.groupby([dataset, method, fold], observed=True)
        .apply(incumbent_trace)
        .rename_axis(index={None: "values"})
        .swaplevel(fold, "values")  # type: ignore
        .unstack(fold)
        .groupby([dataset, method, "values"], observed=True)
        .ffill()
        .mean(axis=1)
        .unstack(dataset)
        .groupby([method, "values"], observed=True)
        .ffill()
        .dropna()  # We can only being aggregating once we have a value for each dataset
        .unstack("values")
        .transform(lambda x: (x - x.min()) / (x.max() - x.min()))
        .transform(lambda x: 1 - x if invert else x)
        .stack("values")
        .swaplevel("time (s)", "values")
        .sort_index()
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(figsize_per_ax[0] * 2, figsize_per_ax[1] * 1),
    )

    val_ax = axes[0]
    test_ax = axes[1]

    def extend_to_x_bound(s: pd.Series) -> pd.Series:
        if x_bounds is not None and s.index[-1] < x_bounds[1]:  # type: ignore
            return pd.concat([s, pd.Series([s.iloc[-1]], index=[x_bounds[1]])])
        return s

    legend_lines = []

    groups = inc_traces_per_dataset.groupby(method, observed=True)
    groups = sorted(groups, key=lambda x: x[0])
    for method_name, _df in groups:
        # We dropna across the dataframe so that when we take mean/std, it's only
        # once we have a datapoint for each dataset (~40s)
        means = _df.mean(axis=1)
        sems = _df.sem(axis=1)
        _color = COLORS[method_name]
        _marker = MARKERS[method_name]
        _style = {
            "marker": _marker,
            "markersize": MARKER_SIZE,
            "markerfacecolor": "white",
            "markeredgecolor": _color,
            "color": _color,
        }
        legend_lines.append(
            Line2D(
                [0],
                [0],
                label=RENAMES.get(method_name, method_name),
                linestyle=LINE_STYLES.get(method_name, "solid"),  # type: ignore
                linewidth=3,
                **_style,
            ),
        )
        for _y, _ax in zip(("val", "test"), (val_ax, test_ax), strict=True):
            _means = means.loc[method_name, _y]
            _stds = sems.loc[method_name, _y]

            _means = extend_to_x_bound(_means)
            _stds = extend_to_x_bound(_stds)

            label_name = RENAMES.get(method_name, method_name)

            _means.plot(  # type: ignore
                drawstyle="steps-post",
                label=f"{label_name}",
                ax=_ax,
                linestyle=LINE_STYLES.get(method_name, "solid"),  # type: ignore
                markevery=markevery,
                linewidth=3,
                **_style,
            )
            _ax.fill_between(
                _means.index,  # type: ignore
                _means - _stds,
                _means + _stds,
                alpha=0.2,
                color=_color,
                edgecolor=_color,
                linewidth=2,
                step="post",
            )

    for _ax, _y_set in zip((val_ax, test_ax), ("Validation", "Test"), strict=True):
        if x_bounds:
            _ax.set_xlim(*x_bounds)

            if x_bounds in X_TICKS:
                _ax.set_xticks(X_TICKS[x_bounds])
                _ax.set_xticklabels([str(x) for x in X_TICKS[x_bounds]])

        if y_bounds:
            _ax.set_ylim(*y_bounds)

        if log_y:
            _ax.set_yscale("log")

        # Define custom formatter function to format tick labels as decimals
        def format_func(value: float, _: int):
            return f"{value:.2f}"

        # Apply the custom formatter to the y-axis
        _ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_func))

        _ax.set_xlabel(
            x_label if x_label is not None else x,
            fontsize=AXIS_LABEL_FONTSIZE,
        )
        _ax.set_ylabel(
            y_label if y_label is not None else y,
            fontsize=AXIS_LABEL_FONTSIZE,
        )
        _ax.set_title(_y_set, fontsize=AXIS_TITLE_FONTSIZE)

    fig.suptitle(title, fontsize=FIG_TITLE_FONTSIZE)
    fig.legend(
        loc="upper center",
        handles=legend_lines,
        bbox_to_anchor=(0.5, 0),
        fontsize=LEGEND_FONTSIZE,
        ncols=LEGEND_MAX_COLS,
    )
    fig.tight_layout()
    return fig, axes


def speedup_plots(  # noqa: PLR0913
    df: pd.DataFrame,
    y: str,
    test_y: str,
    x: str,
    x_start: str,
    fold: str,
    method: str,
    dataset: str,  # TODO: list
    baseline: str,
    *,
    x_label: str | None = None,
    x_bounds: tuple[int, int] = (0, 3600),
    minimize: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    def incumbent_trace(_x: pd.DataFrame) -> pd.Series:
        return _inc_trace(
            _x,
            x_start_col=x_start,
            x_col=x,
            y_col=y,
            test_y_col=test_y,
            minimize=minimize,
        )

    """
    setting:cv_early_stop_strategy               setting:task  time (s)
    current_average_worse_than_best_worst_split  146818        47.404515      0.930459
                                                               47.442476      0.930459
                                                               47.457400      0.930459
                                                               47.537788      0.930459
                                                               47.569711      0.930459
                                                                                ...
    robust_std_top_5                             359979        3586.225907    0.706072
                                                               3588.827892    0.706072
                                                               3589.155218    0.706072
                                                               3591.590766    0.706072
                                                               3595.131776    0.706072
    """
    mean_incumbent_traces = (
        df.groupby([dataset, method, fold], observed=True)
        .apply(incumbent_trace)
        .rename_axis(index={None: "values"})
        .drop(index="test", level="values")
        .droplevel("values")
        .unstack([dataset, method, fold])
        .ffill()
        .dropna()
        .stack([dataset, method, fold], future_stack=True)
        .reorder_levels([method, dataset, fold, "time (s)"])
        .sort_index()
        .unstack(fold)
        .mean(axis=1)
    )

    """
    setting:task  time (s)
    146818        3137.506805    0.935946
    146820        3376.317769    0.996346
                    ...
    359979        3485.686726    0.694450
    """
    baseline_mean_incumbent_trace = mean_incumbent_traces.loc[baseline]
    baseline_time_to_best_with_values = baseline_mean_incumbent_trace.groupby(
        dataset,
        observed=True,
        group_keys=False,
    ).apply(
        lambda x: x.loc[[x.idxmin()]] if minimize else x.loc[[x.idxmax()]],
    )

    # Same thing but with the value column being the time
    """
                     time (s)
    setting:task
    146818        3137.506805
    146820        3376.317769
            ...
    359979        3485.686726
    """
    baseline_time_to_best = (
        baseline_time_to_best_with_values.reset_index("time (s)")
        .drop(columns=0)
        .squeeze()
    )

    def first_row_where_method_reaches_baseline_final_score(
        _x: pd.DataFrame,
    ) -> pd.Series:
        # Get the group that this is being applied to
        (_method, _dataset, _) = _x.index[0]

        baseline_score = baseline_time_to_best_with_values.loc[_dataset].item()
        op = operator.le if minimize else operator.ge
        index_where_better = _x[op(_x, baseline_score)].first_valid_index()

        if index_where_better is not None:
            return _x.loc[[index_where_better]]

        # There was no row in the method that beat the baseline score for that dataset.
        # Hence we return an "empty" series to indicate that.
        return pd.Series(
            [np.nan],
            index=pd.MultiIndex.from_tuples(
                [(_method, _dataset, np.nan)],
                names=[method, dataset, "time (s)"],
            ),
        )

    # For each method, dataset, we get the time at which the method reaches the
    """
    setting:cv_early_stop_strategy              setting:task
    current_average_worse_than_best_worst_split 146818        2816.735736
                                                146820        2986.699438
                                                168350        1921.299560
                                                168757        1480.095332
                                                168784        2532.242832
    ...                                                               ...
    robust_std_top_5                            359971        1245.261844
                                                359972        1496.863776
                                                359973        1532.384214
                                                359975        1560.436368
                                                359979        2621.005973
    """
    method_time_to_beat_baseline = (
        mean_incumbent_traces.groupby(
            [method, dataset],
            observed=True,
            group_keys=False,
        )
        .apply(first_row_where_method_reaches_baseline_final_score)
        .reset_index("time (s)")
        .drop(columns=0)
        .sort_index()
        .squeeze()
        .drop(index=baseline, level=method)
    )

    # Produces the following table, note that disabled (baseline) is dropped but put here for verification.
    """
    setting:cv_early_stop_strategy current_average_worse_than_best_worst_split current_average_worse_than_mean_best disabled robust_std_top_3 robust_std_top_5
    setting:task
    146818                                                            1.113880                                  NaN      1.0         2.746096         2.600318
    146820                                                            1.130451                                  NaN      1.0         2.380920         2.563339
    168350                                                            1.622759                             2.192712      1.0         2.135020         1.799044
    168757                                                            2.285091                                  NaN      1.0         1.138267         1.727029
    ...
    359979                                                            2.677162                             3.856739      1.0         1.814912         1.329904
    """
    speedups = (
        method_time_to_beat_baseline.groupby(method, observed=True, group_keys=False)
        .apply(lambda x: (baseline_time_to_best / x) * 100)
        .unstack(method)
    )

    def count_performed_worse(x: pd.Series) -> int:
        return x.isna().sum()

    _speedup_stats = speedups.agg(["mean", "std", count_performed_worse]).T
    _speedup_stats = _speedup_stats.rename(index=RENAMES).rename(
        columns={"count_performed_worse": "Datasets Failed"},
    )

    """
                          min          max
    setting:task
    359954        2296.842680  2548.263483
            ...
    168910         950.159503  1329.614514
    359960         159.711339   257.199337
    """

    _speedup_stats = markup_speedup_summary_table(
        _speedup_stats,
        bigger_is_better=True,  # Bigger speedup is better
        n_datasets=df[dataset].nunique(),  # type: ignore
    )
    speedups = markup_speedup_table_full(speedups)

    # We sort datasets by the time it took for the baseline to reach the best score
    sort_order = baseline_time_to_best.sort_values(ascending=False).index

    minmaxes = method_time_to_beat_baseline.groupby(dataset).agg(["min", "max"])

    sections = 3
    block = len(minmaxes) // sections
    fig, axes = plt.subplots(
        1,
        sections,
        figsize=(5 * sections, len(minmaxes) / sections / 1.75),
    )

    for section, ax in zip(reversed(range(sections)), axes, strict=True):
        datasets = sort_order[block * section : block * (section + 1)]
        bttb_ = baseline_time_to_best.loc[datasets]
        mm_ = minmaxes.loc[datasets]

        y_ticks = np.arange(len(mm_))
        ax.set_xlim(*x_bounds)
        if x_bounds in X_TICKS:
            ax.set_xticks(X_TICKS[x_bounds])
            ax.set_xticklabels(
                [str(x) for x in X_TICKS[x_bounds]],
                fontsize=TICK_FONTSIZE,
            )

        if section == sections - 1:  # Since we reverse it
            # Fig supylabel was overlapping, we do this hack instead
            ax.set_ylabel("OpenML Task ID", fontsize=AXIS_LABEL_FONTSIZE)

        ax.hlines(
            y_ticks,
            xmin=np.zeros(len(y_ticks)),
            xmax=mm_["max"],
            linestyle="dashed",
            linewidth=1,
            color="grey",
            zorder=1,
        )
        ax.hlines(
            y_ticks,
            xmin=mm_["max"],
            xmax=bttb_,
            linestyle="solid",
            linewidth=1,
            color="black",
            zorder=1,
        )
        ax.scatter(
            x=bttb_,
            y=y_ticks,
            s=MARKER_SIZE**2,
            marker="<",
            c="black",
            edgecolor="black",
            zorder=2,
        )

        for _i, (_method, method_results) in enumerate(
            method_time_to_beat_baseline.groupby(
                method,
                observed=True,
            ),
        ):
            _mttbb = method_results.loc[_method].loc[datasets]
            _color = COLORS[_method]  # type: ignore
            _marker = MARKERS[_method]  # type: ignore
            _style = {
                "s": MARKER_SIZE**2,
                "marker": _marker,
                "color": _color,  # "white"
                "edgecolors": _color,
            }
            ax.scatter(x=_mttbb, y=y_ticks, zorder=2, **_style)

            rng = np.random.default_rng(0)
            jitter_y = rng.uniform(-0.25, 0.25)

            # We also mark where a method failed, bit manual but it works
            for _tick, mttbb_single in zip(y_ticks, _mttbb, strict=True):
                # We place a slight jitter on the x-axis so that we can see the points
                # if they overlap
                if np.isnan(mttbb_single):
                    ax.scatter(
                        x=[x_bounds[1]],
                        y=[_tick + jitter_y],
                        zorder=2,
                        clip_on=False,
                        **{**_style, "marker": "x", "s": MARKER_SIZE**2 / 2},
                    )

        ax.set_yticks(y_ticks, labels=datasets, fontsize=TICK_FONTSIZE)

    methods = sorted(
        method_time_to_beat_baseline.index.get_level_values(method).unique(),
    )
    legend_lines = [
        Line2D(
            [0],
            [0],
            label=RENAMES.get(m, m),
            linestyle="",
            markersize=MARKER_SIZE,
            marker=MARKERS[m],
            color=COLORS[m],
        )
        for m in methods
    ]
    legend_lines.append(
        Line2D(
            [0],
            [0],
            label="Failed to beat baseline",
            markersize=MARKER_SIZE * 3 / 4,
            linestyle="",
            marker="x",
            color="black",
        ),
    )

    fig.supxlabel(x_label if x_label is not None else x, fontsize=AXIS_LABEL_FONTSIZE)

    # Maptlotlib doesn't like this supylabel, so we'll just use the the first axis
    # fig.supylabel("OpenML Task ID", fontsize="x-large")
    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
        fontsize=LEGEND_FONTSIZE,
        handles=legend_lines,
        ncols=LEGEND_MAX_COLS,
    )
    fig.tight_layout()
    return speedups, _speedup_stats
