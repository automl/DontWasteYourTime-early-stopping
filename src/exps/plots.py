from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MARKERS = [
    ".",
    "v",
    ">",
    ",",
    "o",
    "^",
    "<",
    "1",
    "2",
    "3",
    "4",
    "s",
    "p",
    "*",
    "h",
    "H",
    "+",
    "x",
    "D",
    "d",
    "|",
    "_",
]
RENAMES: dict[str, str] = {
    "disabled": "no-cv-es",
    "current_average_worse_than_mean_best": "fold-worse-best-mean",
    "current_average_worse_than_best_worst_split": "fold-worse-best-worst",
}


def _inc_trace(
    name: str,
    df: pd.DataFrame,
    *,
    x_start_col: str,
    x_col: str,
    y_col: str,
    minimize: bool,
    x_bounds: tuple[int | float | None, int | float | None] | None,
) -> pd.Series:
    # We now have each individual std group to plot, i.e. the fold
    _start = df[x_start_col].min()
    _x = (df[x_col] - _start).dt.total_seconds()
    _y = df[y_col]
    _s = pd.Series(_y.to_numpy(), index=_x, name=name).sort_index()

    # Transform everything
    match minimize:
        # Bounded metrics, 0-1 normalize
        case True:
            _s = _s.cummin()
        case False:
            _s = _s.cummax()

    if x_bounds is not None and x_bounds[1] is not None and _s.index[-1] < x_bounds[1]:
        _new_index = np.concatenate([_s.index, [x_bounds[1]]])
        _new_values = np.concatenate([_s.to_numpy(), [_s.iloc[-1]]])
        _s = pd.Series(_new_values, _new_index, name=name)

    return _s


def min_max_normalize(
    df: pd.DataFrame,
    *,
    mins: list[float],
    maxs: list[float],
    invert: bool = False,
) -> pd.DataFrame:
    assert len(mins) == len(maxs)
    assert len(df.columns) == len(mins)
    normalized = (df - mins) / (np.asarray(maxs) - np.asarray(mins))
    if invert:
        return 1 - normalized

    return normalized


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
    axes: list[plt.Axes] | None = None,
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
        # We now have group as a dataframe for a given plot, i.e. the task

        # Collect a dataframe of incumbent traces for each hue group
        # Key is the hue name, value is the below dataframe
        #
        #    | 0 | 1 | 2 | 3 | ...  <- fold index
        # t1 | . | . | . | . | ...  <- time 1
        # t2 | . | . | . | . | ...  <- time 2
        # t3 | . | . | . | . | ...  <- time 3
        hue_dfs: dict[str, pd.DataFrame] = {}
        for hue_name, hue_group in plot_group.groupby(hue):
            hue_inc_traces = []
            for _std_name, _std_group in hue_group.groupby(std):
                inc_trace = _inc_trace(
                    _std_name,
                    _std_group,
                    x_start_col=x_start,
                    x_col=x,
                    y_col=y,
                    minimize=minimize,
                    x_bounds=x_bounds,
                )
                hue_inc_traces.append(inc_trace)

            hue_df = (
                pd.concat(hue_inc_traces, axis=1, ignore_index=False)
                .sort_index()
                .ffill()
                .dropna()  # TODO: drop?
            )
            assert isinstance(hue_name, str)
            hue_dfs[hue_name] = hue_df

        # For each dataframe, we want to normalize each fold column by the min/max ever seen
        # for an incumbent trace on that fold.
        n_folds = len(next(iter(hue_dfs.values())).columns)
        mins = [
            min(float(df[f].min()) for df in hue_dfs.values()) for f in range(n_folds)
        ]
        maxs = [
            max(float(df[f].max()) for df in hue_dfs.values()) for f in range(n_folds)
        ]

        normalized_hue_dfs = {
            hue_name: min_max_normalize(df, mins=mins, maxs=maxs, invert=not minimize)
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
            ax.set_yscale("log")

        ax.set_xlabel(x_label if x_label is not None else x)
        ax.set_ylabel(y_label if y_label is not None else y)
        ax.set_title(str(plot_name))
        ax.legend()

    return axes


def baseline_normalized_over_time(  # noqa: PLR0913
    df: pd.DataFrame,
    y: str,
    x: str,
    x_start: str,
    std: str,
    hue: str,
    baseline: str,
    metric_bounds: tuple[float, float],
    subplot: str,  # TODO: list
    *,
    minimize: bool = False,
    log_y: bool = False,
    x_bounds: tuple[int | float | None, int | float | None] | None = None,
    y_bounds: tuple[int | float | None, int | float | None] | None = None,
    axes: list[plt.Axes] | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    markevery: int | float | None = None,
    ncols: int = 3,
) -> list[plt.Axes]:
    metric_worst = metric_bounds[0] if minimize else metric_bounds[1]
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
        # We now have group as a dataframe for a given plot, i.e. the task
        # We now group everything by the task fold

        # Contains a mapping from the hue group (e.g. method) to a list
        # of fold results
        fold_results: dict[str, list[pd.Series]] = defaultdict(list)

        for _, std_group in plot_group.groupby(std):
            # We select out the baseline results for this fold, we will use this to
            # center the scores of the fold
            hue_groups = dict(iter(std_group.groupby(hue)))
            baseline_trace = hue_groups[baseline]
            baseline_inc_trace = _inc_normalized(
                baseline_trace,
                x_start_col=x_start,
                x_col=x,
                y_col=y,
                minimize=minimize,
                x_bounds=x_bounds,
            )

            for hue_name, hue_group in hue_groups.items():
                assert isinstance(hue_name, str)
                hue_inc_trace = _inc_normalized(
                    hue_group,
                    x_start_col=x_start,
                    x_col=x,
                    y_col=y,
                    minimize=minimize,
                    x_bounds=x_bounds,
                )

                # Create a joint index that covers both the baseline and the hue
                merged_index = baseline_inc_trace.index.union(
                    hue_inc_trace.index,
                    sort=True,
                )
                # Reindex the baseline and hue to the merged index, ffilling where
                # possible and just filling with worst score otherwise.
                # TODO: Maybe consider just dropping?
                reindexed_baseline = baseline_inc_trace.reindex(
                    merged_index,
                    method="ffill",
                ).fillna(metric_worst)
                reindexed_hue = hue_inc_trace.reindex(
                    merged_index,
                    method="ffill",
                ).fillna(metric_worst)

                assert len(reindexed_baseline) == len(reindexed_hue)
                fold_results[hue_name].append(reindexed_hue - reindexed_baseline)

        for hue_name, traces in fold_results.items():
            _hue_df = (
                pd.concat(traces, axis=1, ignore_index=False)
                .sort_index()
                .ffill()
                .dropna()
            )
            _means = _hue_df.mean(axis=1)
            _stds = _hue_df.std(axis=1)

            assert isinstance(hue_name, str)
            legend_name = RENAMES.get(hue_name, hue_name)

            _means.plot(  # type: ignore
                ax=ax,
                drawstyle="steps-post",
                label=legend_name,
                markevery=markevery,
            )
            ax.fill_between(
                _means.index,  # type: ignore
                _means - _stds,
                _means + _stds,
                alpha=0.3,
                step="post",
            )

        if x_bounds:
            ax.set_xlim(*x_bounds)
        if y_bounds:
            ax.set_ylim(*y_bounds)

        if log_y:
            ax.set_yscale("log")

        ax.set_xlabel(x_label if x_label is not None else x)
        ax.set_ylabel(y_label if y_label is not None else y)
        ax.set_title(str(plot_name))
        ax.legend()

    return axes
