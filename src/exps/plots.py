# ruff: noqa: E501, PD010, PD013
from __future__ import annotations

from collections.abc import Hashable
from functools import partial

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

MARKERS = [
    ".",
    "v",
    ">",
    "*",
    "o",
    "^",
    "<",
    "1",
    "2",
    "3",
    "4",
    "s",
    "p",
    "h",
    "H",
    "D",
    "+",
    "x",
    "d",
    "|",
    "_",
    ",",
]
X_TICKS = {
    (0, 3600): [0, 600, 1200, 1800, 2400, 3000, 3600],
}
RENAMES: dict[str, str] = {
    "disabled": "no-cv-es",
    "current_average_worse_than_mean_best": "fold-worse-best-mean",
    "current_average_worse_than_best_worst_split": "fold-worse-best-worst",
}


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


def rank_plots(  # noqa: PLR0913
    df: pd.DataFrame,
    y: str,
    x: str,
    x_start: str,
    std: str,
    hue: str,
    subplot: str,  # TODO: list
    *,
    minimize: bool = False,
    log_y: bool = False,
    x_bounds: tuple[int | float | None, int | float | None] | None = None,
    ax: list[plt.Axes] | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    markevery: int | float | None = None,
    ncols: int = 3,
) -> list[plt.Axes]:
    perplot = list(df.groupby(subplot))

    # Get the colors to use
    colors = plt.get_cmap("tab10").colors  # type: ignore
    markers = MARKERS
    list(zip(colors, markers, strict=False))

    if axes is None:
        nrows = (len(perplot) + ncols - 1) // ncols
        _, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
        axes = list(axes.flatten())  # type: ignore
    else:
        assert len(axes) >= len(perplot)

    for (plot_name, plot_group), ax in zip(perplot, axes, strict=False):
        ranks_per_fold = []
        for _std_name, std_group in plot_group.groupby(std):
            inc_traces_per_fold = [
                _inc_trace(
                    str(hue_name),
                    hue_group,
                    x_start_col=x_start,
                    x_col=x,
                    y_col=y,
                    minimize=minimize,
                    x_bounds=x_bounds,
                )
                for hue_name, hue_group in std_group.groupby(hue)
            ]
            ranks_for_fold = (
                pd.concat(inc_traces_per_fold, axis=1, ignore_index=False)
                .sort_index()
                .ffill()
                .dropna()
                .rank(axis=1, method="average", ascending=minimize)
            )
            ranks_per_fold.append(ranks_for_fold)

        means_stds: dict[str, tuple[pd.Series, pd.Series]] = {}

        for hue_name in ranks_per_fold[0].columns:
            ranks_for_hue = [fold_ranks[hue_name] for fold_ranks in ranks_per_fold]
            merged_ranks_for_hue = (
                pd.concat(ranks_for_hue, axis=1, ignore_index=False)
                .sort_index()
                .ffill()
                .dropna()
            )
            mean_ranks = merged_ranks_for_hue.mean(axis=1)
            std_ranks = merged_ranks_for_hue.std(axis=1)
            means_stds[hue_name] = (mean_ranks, std_ranks)

        for (hue_name, (mean_ranks, std_ranks)), color, marker in zip(
            means_stds.items(),
            colors,
            markers,
            strict=False,
        ):
            legend_name = RENAMES.get(hue_name, hue_name)
            mean_ranks.plot(  # type: ignore
                ax=ax,
                drawstyle="steps-post",
                color=color,
                marker=marker,
                label=f"{legend_name}",
                markevery=markevery,
            )
            ax.fill_between(
                mean_ranks.index,  # type: ignore
                mean_ranks - std_ranks,
                mean_ranks + std_ranks,
                alpha=0.3,
                color=color,
                step="post",
            )

        if x_bounds:
            ax.set_xlim(*x_bounds)

        if log_y:
            ax.set_yscale("log")

        ax.set_xlabel(x_label if x_label is not None else x)
        ax.set_ylabel(y_label if y_label is not None else y)
        ax.set_title(str(plot_name))
        ax.legend()

    return axes


def baseline_improvement_aggregated(  # noqa: PLR0913
    df: pd.DataFrame,
    y: str,
    x: str,
    x_start: str,
    fold: str,
    hue: str,
    baseline: str,
    dataset: str,  # TODO: list
    *,
    minimize: bool = False,
    log_y: bool = False,
    x_bounds: tuple[int | float | None, int | float | None] | None = None,
    ax: plt.Axes | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    markevery: int | float | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    # Get the colors to use
    colors = plt.get_cmap("tab10").colors  # type: ignore
    markers = MARKERS
    list(zip(colors, markers, strict=False))

    # TODO: Might have to change here if we also consider other scores such as test
    # scores
    baseline_data = df[df[hue] == baseline]
    baseline_traces = dict(baseline_data.groupby([dataset, fold])[y])
    print(baseline_traces)
    mins = per_dataset_bounds["min"]
    maxs = per_dataset_bounds["max"]
    assert isinstance(mins, pd.Series)
    assert isinstance(maxs, pd.Series)

    def _normalized_incumbent_trace(
        _name: Hashable,
        _df: pd.DataFrame,
        _dataset: Hashable,
        _fold: Hashable,
    ) -> pd.Series:
        return _inc_trace(
            _name,
            _df,
            x_start_col=x_start,
            x_col=x,
            y_col=y,
            minimize=minimize,
            x_bounds=x_bounds,
            invert=not minimize,
            normalize=(mins.loc[_normalize_index], maxs.loc[_normalize_index]),
        )

    # dict from dataset name to methods mean and std for that dataset
    #
    # Innermost series looks lik this, represents mean of normalized cost of that
    # dataset across folds.
    # METH| mean
    #  t1 |  .
    #  t2 |  .
    #  t3 |  .
    dfs = {
        dataset_name: pd.DataFrame(
            [
                pd.concat(
                    [
                        _normalized_incumbent_trace(
                            _name=fold_name,
                            _df=_df,
                            _dataset=dataset_name,
                            _fold=fold_name,
                        )
                        for fold_name, _df in method_group.groupby(fold, observed=True)
                    ],
                    axis=1,
                    ignore_index=False,
                )
                # Ensure index is sorted according to time
                .sort_index()
                # Ffill to fill in NaNs, indicating the score for what was found
                # previously is carried forward
                .ffill()
                # Drop rows with NaNs, indicating there wasn't a score for at least 1
                # fold at that time point
                .dropna(axis=0)
                # Take mean across incumbent traces (columns)
                .mean(axis=1)
                .rename(method_name)  # type: ignore
                for method_name, method_group in dataset_group.groupby(
                    hue,
                    observed=True,
                )
            ],
        )
        .T.sort_index()
        .ffill()
        .dropna(axis=0)
        for dataset_name, dataset_group in df.groupby(dataset, observed=True)
    }

    if ax is None:
        _, _ax = plt.subplots(1, 1, figsize=(10, 10))
    else:
        _ax = ax

    method_names = sorted(set.union(*[set(dfs[dataset]) for dataset in dfs]))
    dataset_names = list(dfs)
    for method in method_names:
        # Just veryify all datasets have the same methods
        for dname in dataset_names:
            if method not in dfs[dname].columns:
                raise ValueError(f"Method {method} not found in dataset {dname}")

        method_df = (
            pd.concat(  # type: ignore
                [dfs[dname][method].rename(dataset) for dname in dataset_names],  # type: ignore
                axis=1,
                ignore_index=False,
            )
            .sort_index()
            .ffill()
            .dropna()
        )

        _means = method_df.mean(axis=1)
        _stds = method_df.std(axis=1)

        method_name = RENAMES.get(method, method)
        _means.plot(  # type: ignore
            drawstyle="steps-post",
            label=f"{method_name}",
            markevery=markevery,
            ax=_ax,
        )
        _ax.fill_between(
            _means.index,  # type: ignore
            _means - _stds,
            _means + _stds,
            alpha=0.3,
            step="post",
        )

    if x_bounds:
        _ax.set_xlim(*x_bounds)

    if log_y:
        _ax.set_yscale("log")

    _ax.set_xlabel(x_label if x_label is not None else x)
    _ax.set_ylabel(y_label if y_label is not None else y)
    _ax.set_title("Aggregated")
    _ax.legend()

    return _ax


def ranking_plots_aggregated(  # noqa: PLR0913
    df: pd.DataFrame,
    y: str,
    test_y: str,
    x: str,
    x_start: str,
    fold: str,
    method: str,
    dataset: str,  # TODO: list
    *,
    minimize: bool = False,
    x_bounds: tuple[int | float | None, int | float | None] | None = None,
    fig_ax: tuple[plt.Figure, plt.Axes] | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    markevery: int | float | None = None,
) -> plt.Axes:
    # Get the colors to use
    colors = plt.get_cmap("tab10").colors  # type: ignore
    markers = MARKERS
    list(zip(colors, markers, strict=False))

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

    for (method_name, _df), _color, _marker in zip(
        ranks_per_dataset.groupby(method, observed=True),
        colors,
        markers,
        strict=False,
    ):
        means = _df.mean(axis=1)
        sems = _df.sem(axis=1)
        _style = {
            "marker": _marker,
            "markersize": 10,
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

        _ax.set_ylim(1, df[method].nunique())

        _ax.set_xlabel(x_label if x_label is not None else x, fontsize="large")
        _ax.set_ylabel(y_label if y_label is not None else y, fontsize="large")
        N_DATASETS = df[dataset].nunique()
        _ax.set_title(
            f"Rank Aggregation over {N_DATASETS} Datasets ({_y_set})",
            fontsize="x-large",
        )

    fig.legend(
        loc="upper center",
        handles=legend_lines,
        bbox_to_anchor=(0.5, 0),
        fontsize="large",
        ncols=len(legend_lines),
        title_fontproperties={"size": "x-large"},
    )
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
    # Get the colors to use
    colors = plt.get_cmap("tab10").colors  # type: ignore
    markers = MARKERS
    list(zip(colors, markers, strict=False))

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

    for (method_name, _df), _color, _marker in zip(
        inc_traces_per_dataset.groupby(method, observed=True),
        colors,
        markers,
        strict=False,
    ):
        # We dropna across the dataframe so that when we take mean/std, it's only
        # once we have a datapoint for each dataset (~40s)
        means = _df.mean(axis=1)
        sems = _df.sem(axis=1)
        _style = {
            "marker": _marker,
            "markersize": 10,
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

        _ax.set_xlabel(x_label if x_label is not None else x, fontsize="large")
        _ax.set_ylabel(y_label if y_label is not None else y, fontsize="large")
        N_DATASETS = df[dataset].nunique()
        _ax.set_title(
            f"Aggregation over {N_DATASETS} Datasets ({_y_set})",
            fontsize="x-large",
        )

    fig.legend(
        loc="upper center",
        handles=legend_lines,
        bbox_to_anchor=(0.5, 0),
        fontsize="large",
        ncols=len(legend_lines),
        title_fontproperties={"size": "x-large"},
    )
    return fig, axes


def incumbent_traces(  # noqa: PLR0913
    df: pd.DataFrame,
    y: str,
    x: str,
    x_start: str,
    std: str,
    hue: str,
    subplot: str,  # TODO: list
    *,
    minimize: bool = False,
    log_y: bool = False,
    x_bounds: tuple[int | float | None, int | float | None] | None = None,
    axes: list[plt.Axes] | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    markevery: int | float | None = None,
    ncols: int = 3,
) -> list[plt.Axes]:
    perplot = list(df.groupby(subplot, observed=True))

    # Get the colors to use
    colors = plt.get_cmap("tab10").colors  # type: ignore
    markers = MARKERS
    list(zip(colors, markers, strict=False))

    if axes is None:
        nrows = (len(perplot) + ncols - 1) // ncols
        _, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
        axes = list(axes.flatten())  # type: ignore
    else:
        assert len(axes) >= len(perplot)

    _incumbent_trace = partial(
        _inc_trace,
        x_start_col=x_start,
        x_col=x,
        y_col=y,
        minimize=minimize,
        x_bounds=x_bounds,
    )

    for (plot_name, plot_group), ax in zip(perplot, axes, strict=False):
        print(plot_name)
        # We now have group as a dataframe for a given plot, i.e. the task

        # Key is the hue name, value is the below dataframe
        #
        #    | 0 | 1 | 2 | 3 | ...  <- fold index
        # t1 | . | . | . | . | ...  <- time 1
        # t2 | . | . | . | . | ...  <- time 2
        # t3 | . | . | . | . | ...  <- time 3
        hue_dfs: dict[Hashable, pd.DataFrame] = {
            hue_name: pd.concat(
                [
                    _incumbent_trace(_std_name, _std_group)
                    for _std_name, _std_group in hue_group.groupby(std, observed=True)
                ],
                axis=1,
                ignore_index=False,
            )
            .sort_index()
            .ffill()
            .dropna()
            for hue_name, hue_group in plot_group.groupby(hue, observed=True)
        }

        # For each dataframe, we want to normalize each fold column by the min/max ever
        # seen for an incumbent trace on that fold.
        n_folds = len(next(iter(hue_dfs.values())).columns)
        fold_mins = [
            min(float(df[f].min()) for df in hue_dfs.values()) for f in range(n_folds)
        ]
        fold_maxs = [
            max(float(df[f].max()) for df in hue_dfs.values()) for f in range(n_folds)
        ]

        inverted = not minimize  # Make it so that lower is better
        normalized_hue_dfs = {
            hue_name: min_max_normalize(
                df,
                mins=fold_mins,
                maxs=fold_maxs,
                invert=inverted,
            )
            for hue_name, df in hue_dfs.items()
        }

        for (hue_name, _df), color, marker in zip(
            normalized_hue_dfs.items(),
            colors,
            markers,
            strict=False,
        ):
            legend_name = RENAMES.get(hue_name, hue_name)
            _means = _df.mean(axis=1)
            _stds = _df.std(axis=1)

            _means.plot(  # type: ignore
                ax=ax,
                drawstyle="steps-post",
                color=color,
                marker=marker,
                label=f"{legend_name}",
                markevery=markevery,
            )
            ax.fill_between(
                _means.index,  # type: ignore
                _means - _stds,
                _means + _stds,
                alpha=0.3,
                color=color,
                step="post",
            )

        if x_bounds:
            ax.set_xlim(*x_bounds)

        if log_y:
            ax.set_yscale("symlog")

        ax.set_xlabel(x_label if x_label is not None else x)
        ax.set_ylabel(y_label if y_label is not None else y)
        ax.set_title(str(plot_name))
        ax.legend()

    return axes
