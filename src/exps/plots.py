# ruff: noqa: E501, PD010, PD013
from __future__ import annotations

import json
import operator
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

_colors = iter(plt.get_cmap("tab10").colors)  # type: ignore
MARKER_SIZE = 10
COLORS = {
    "disabled": next(_colors),
    "current_average_worse_than_mean_best": next(_colors),
    "current_average_worse_than_best_worst_split": next(_colors),
    "robust_std_top_3": next(_colors),
    "robust_std_top_5": next(_colors),
}
MARKERS = {
    "disabled": ".",
    "current_average_worse_than_mean_best": "v",
    "current_average_worse_than_best_worst_split": ">",
    "robust_std_top_3": "*",
    "robust_std_top_5": "o",
    #    "^",
    #    "<",
    #    "1",
    #    "2",
    #    "3",
    #    "4",
    #    "s",
    #    "p",
    #    "h",
    #    "H",
    #    "D",
    #    "+",
    #    "x",
    #    "d",
    #    "|",
    #    "_",
    #    ",",
}
X_TICKS = {
    (0, 3600): [0, 600, 1200, 1800, 2400, 3000, 3600],
}
RENAMES: dict[str, str] = {
    "disabled": "no-cv-es",
    "current_average_worse_than_mean_best": "fold-worse-best-mean",
    "current_average_worse_than_best_worst_split": "fold-worse-best-worst",
    "robust_std_top_3": "robust-3",
    "robust_std_top_5": "robust-5",
}


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

    for method_name, _df in ranks_per_dataset.groupby(method, observed=True):
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
                linestyle="solid",
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
                linestyle="solid",
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

        _ax.set_ylim(1, df[method].nunique())

        _ax.set_xlabel(x_label if x_label is not None else x, fontsize="x-large")
        _ax.set_ylabel(y_label if y_label is not None else y, fontsize="x-large")
        _ax.set_title(_y_set, fontsize="x-large")

    fig.suptitle(title, fontsize="xx-large")
    fig.legend(
        loc="upper center",
        handles=legend_lines,
        bbox_to_anchor=(0.5, 0),
        fontsize="xx-large",
        ncols=len(legend_lines),
    )
    fig.tight_layout()
    return fig, axes


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
    minimize: bool = False,
    log_y: bool = False,
    x_bounds: tuple[int | float | None, int | float | None] | None = None,
    y_bounds: tuple[int | float | None, int | float | None] | None = None,
    fig_ax: tuple[plt.Figure, plt.Axes] | None = None,
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

    for method_name, _df in inc_traces_per_dataset.groupby(method, observed=True):
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
                linestyle="solid",
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
                linestyle="solid",
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
            return f"{value:.1f}"

        # Apply the custom formatter to the y-axis
        _ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_func))

        _ax.set_xlabel(x_label if x_label is not None else x, fontsize="x-large")
        _ax.set_ylabel(y_label if y_label is not None else y, fontsize="x-large")
        _ax.set_title({_y_set}, fontsize="x-large")

    fig.suptitle(title, fontsize="xx-large")
    fig.legend(
        loc="upper center",
        handles=legend_lines,
        bbox_to_anchor=(0.5, 0),
        fontsize="xx-large",
        ncols=len(legend_lines),
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
    title: str,
    baseline: str,
    *,
    ax: plt.Axes | None = None,
    x_label: str | None = None,
    x_bounds: tuple[int, int] = (0, 3600),
    minimize: bool = False,
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

    # TODO: Need to export this to a table somewhere
    _speedup_stats = speedups.agg(["mean", "std", count_performed_worse]).T
    print("TODO: Write these somewhere")
    print(_speedup_stats)

    # Didn't go with this pairwise plot because it didn't have any more information than a boxplot would
    # NOTABLY: baseline convergence time (color) and dataset size (cell count) showed no clear patterns in relation
    # to speedups.
    """
    In [697]: g = sns.PairGrid(s, hue="baseline_convergence_time", corner=True, vars=XS); g.map_diag(sns.histplot, hue=None, color=".3"); g.map_offdiag(sns.scatterplot, size=s["dataset_size"]); g.add_legend(adjust_subtitles=True)
     ...: for ax in g.axes.flatten():
     ...:     if ax is not None:
     ...:         ax.set_ylim(*bs)
     ...:         ax.set_xlim(*bs)
     ...: plt.show()
    """
    if ax is None:
        _, _ax = plt.subplots(1, 1, figsize=(10, 20))
    else:
        _ax = ax

    # We sort datasets by the time it took for the baseline to reach the best score
    sort_order = baseline_time_to_best.sort_values(ascending=False).index

    ys = np.arange(len(sort_order)) + 1

    _ax.set_xlim(*x_bounds)
    _ax.hlines(
        ys,
        xmin=np.zeros(len(ys)),
        xmax=baseline_time_to_best.loc[sort_order],
        linestyle="dashed",
        linewidth=2,
    )
    _ax.scatter(
        x=baseline_time_to_best.loc[sort_order],
        y=ys,
        # https://stackoverflow.com/questions/14827650/pyplot-scatter-plot-marker-size#comment113131391_14827650
        s=MARKER_SIZE**2,
        marker="<",
        c="white",
        edgecolor="black",
    )

    minmaxes = (
        method_time_to_beat_baseline.groupby(dataset)
        .agg(["min", "max"])
        .loc[sort_order]
    )
    _ax.hlines(
        ys,
        xmin=minmaxes["min"],
        xmax=minmaxes["max"],
        linestyle="solid",
        linewidth=3,
    )

    for _method, _x in method_time_to_beat_baseline.groupby(
        method,
        observed=True,
    ):
        print(f"Method: {_method}")
        print(_x)
        _color = COLORS[_method]  # type: ignore
        _marker = MARKERS[_method]  # type: ignore
        _style = {
            "s": MARKER_SIZE**2,
            "marker": _marker,
            "color": "white",
            "edgecolors": _color,
        }
        _ax.scatter(
            x=_x.loc[_method].loc[sort_order],
            y=ys,
            label=_method,
            **_style,
        )

    _ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
        fontsize="xx-large",
        ncols=(
            method_time_to_beat_baseline.index.get_level_values(method).nunique() // 2
        ),
    )
    return ax
