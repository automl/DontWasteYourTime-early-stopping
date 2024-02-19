# ruff: noqa: E501
from __future__ import annotations

from collections.abc import Hashable
from functools import partial

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
    ind = pd.Index(_x, name="time (s)")
    _s = pd.Series(_y.to_numpy(), index=ind, name="y").sort_index().dropna()

    # Transform everything
    match minimize:
        case True:
            _s = _s.cummin().drop_duplicates(keep="first")
        case False:
            _s = _s.cummax().drop_duplicates(keep="first")

    if x_bounds is not None and x_bounds[1] is not None and _s.index[-1] < x_bounds[1]:
        _new_index = np.concatenate([_s.index, [x_bounds[1]]])
        _new_values = np.concatenate([_s.to_numpy(), [_s.iloc[-1]]])
        ind = pd.Index(_new_index, name="time (s)")
        _s = pd.Series(_new_values, index=ind, name="y")

    return _s


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
) -> plt.Axes:
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


def incumbent_traces_aggregated(  # noqa: PLR0913
    df: pd.DataFrame,
    y: str,
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
    ax: plt.Axes | None = None,
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
            minimize=minimize,
            x_bounds=x_bounds,
        )

    data = (
        # Get the incumbent trace for each (dataset, method, fold)
        # Gives: index=(dataset, method, fold, time) col=(,) values=(inc)
        df.groupby([dataset, method, fold], observed=True)  # noqa: PD010
        .apply(incumbent_trace)
        # Make method and fold columns
        # Gives: index=(dataset, time), col=(method, fold) values=(inc)
        .unstack([method, fold])
        # Groupby together each (dataset,) and apply a ffill to each (method, fold) over (time,)
        # Gives: index=(dataset, time), col=(method, fold) values=(inc-ffilled)
        .groupby(dataset, observed=True)
        .ffill()
        # Take mean across folds for the method
        # Gives: index=(dataset, time), col=(method,) values=(mean_inc_across_folds)
        .groupby(level=method, axis=1, observed=True)
        .mean()
        # Min/max each mean-inc-trace by the min/max of all observed values for the dataset
        # Gives: index=(dataset, time), col=(method,) values=(normalized-mean-inc-across-folds)
        .groupby(dataset, observed=True, group_keys=False)
        .apply(lambda x: (x - x.min().min()) / (x.max().max() - x.min().min()))
        # Pop off the dataset index, and ffill so we can then mean across it
        # Gives: index=(time), col=(method,dataset) values=(normalized-mean-inc-across-folds)
        .unstack(dataset)
        .ffill()
    )

    # Our data now looks like this
    """
    setting:cv_early_stop_strategy                                  current_average_worse_than_best_worst_split                                                                                                                                           ... robust_std_top_5
    setting:task                                                        146818    146820    168350    168757    168784   168910    168911    189922 190137 190146    190392    190410 190411    190412    359953 359954  ...           359959 359960    359961    359962 359963 359964    359965 359967    359968    359969    359970    359971    359972 359973 359975    359979
    time (s)                                                                                                                                                                                                             ...
    1.011305                                                               NaN       NaN       NaN       NaN       NaN      NaN       NaN       NaN    NaN    NaN       NaN       NaN    NaN       NaN       NaN    NaN  ...              NaN    NaN       NaN       NaN    NaN    NaN       NaN    NaN       NaN       NaN       NaN       NaN       NaN    NaN    NaN       NaN
    1.016992                                                               NaN       NaN       NaN       NaN       NaN      NaN       NaN       NaN    NaN    NaN       NaN       NaN    NaN       NaN       NaN    NaN  ...              NaN    NaN       NaN       NaN    NaN    NaN       NaN    NaN       NaN       NaN       NaN       NaN       NaN    NaN    NaN       NaN
    1.017846                                                               NaN       NaN       NaN       NaN       NaN      NaN       NaN       NaN    NaN    NaN       NaN       NaN    NaN       NaN       NaN    NaN  ...              NaN    NaN       NaN       NaN    NaN    NaN       NaN    NaN       NaN       NaN       NaN       NaN       NaN    NaN    NaN       NaN
    1.022772                                                               NaN       NaN       NaN       NaN       NaN      NaN       NaN       NaN    NaN    NaN       NaN       NaN    NaN       NaN       NaN    NaN  ...              NaN    NaN       NaN       NaN    NaN    NaN       NaN    NaN       NaN       NaN       NaN       NaN       NaN    NaN    NaN       NaN
    1.024474                                                               NaN       NaN       NaN       NaN       NaN      NaN       NaN       NaN    NaN    NaN       NaN       NaN    NaN       NaN       NaN    NaN  ...              NaN    NaN       NaN       NaN    NaN    NaN       NaN    NaN       NaN       NaN       NaN       NaN       NaN    NaN    NaN       NaN
    ...                                                                    ...       ...       ...       ...       ...      ...       ...       ...    ...    ...       ...       ...    ...       ...       ...    ...  ...              ...    ...       ...       ...    ...    ...       ...    ...       ...       ...       ...       ...       ...    ...    ...       ...
    3587.761974                                                       0.966475  0.999956  0.991856  0.966388  0.985812  0.97294  0.952077  0.968142    1.0    1.0  0.753457  0.998171    1.0  0.956321  0.989934    1.0  ...         0.983291    1.0  0.992377  0.885627    1.0    1.0  0.966052    1.0  0.995941  0.986114  0.981243  0.995812  0.985233    1.0    1.0  0.861138
    3587.914040                                                       0.966475  0.999956  0.991856  0.966388  0.985812  0.97294  0.952077  0.968142    1.0    1.0  0.753457  0.998171    1.0  0.956321  0.989934    1.0  ...         0.983291    1.0  0.992377  0.885627    1.0    1.0  0.966052    1.0  0.995941  0.986114  0.981243  0.995812  0.985233    1.0    1.0  0.861138
    3587.954826                                                       0.966475  0.999956  0.991856  0.966388  0.985812  0.97294  0.952077  0.968142    1.0    1.0  0.753457  0.998171    1.0  0.956321  0.989934    1.0  ...         0.983291    1.0  0.992377  0.885627    1.0    1.0  0.966052    1.0  0.995941  0.986114  0.981243  0.995812  0.985233    1.0    1.0  0.861138
    3589.325142                                                       0.966475  0.999956  0.991856  0.971617  0.985812  0.97294  0.952077  0.968142    1.0    1.0  0.753457  0.998171    1.0  0.956321  0.989934    1.0  ...         0.983291    1.0  0.992377  0.885627    1.0    1.0  0.966052    1.0  0.995941  0.986114  0.981243  0.995812  0.985233    1.0    1.0  0.861138
    3600.000000                                                       0.966475  0.999956  0.991856  0.971617  0.985812  0.97294  0.952077  0.968142    1.0    1.0  0.753457  0.998171    1.0  0.956321  0.989934    1.0  ...         0.983291    1.0  0.992377  0.885627    1.0    1.0  0.966052    1.0  0.995941  0.986114  0.981243  0.995812  0.985233    1.0    1.0  0.861138
    """

    if ax is None:
        _, _ax = plt.subplots(1, 1, figsize=(10, 10))
    else:
        _ax = ax

    for method_name, method_group in data.T.groupby(level=method, observed=True):
        method_group = method_group.T  # noqa: PLW2901
        _means = method_group.mean(axis=1)
        _stds = method_group.sem(axis=1)

        if invert:
            _means = 1 - _means

        label_name = RENAMES.get(method_name, method_name)
        _means.plot(  # type: ignore
            drawstyle="steps-post",
            label=f"{label_name}",
            markevery=markevery,
            ax=_ax,
        )
        _ax.fill_between(
            _means.index,  # type: ignore
            _means - _stds,
            _means + _stds,
            alpha=0.2,
            step="post",
        )

    if x_bounds:
        _ax.set_xlim(*x_bounds)

    if y_bounds:
        _ax.set_ylim(*y_bounds)

    if log_y:
        _ax.set_yscale("log")

    _ax.set_xlabel(x_label if x_label is not None else x)
    _ax.set_ylabel(y_label if y_label is not None else y)
    _ax.set_title("Aggregated")
    _ax.legend()

    return _ax


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
