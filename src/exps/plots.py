from __future__ import annotations

import matplotlib.pyplot as plt
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


def incumbent_traces(
    df: pd.DataFrame,
    y: str,
    x: str,
    min_x: str,
    std: str,
    hue: str | list[str],
    subplot: str,  # TODO: list
    *,
    minimize: bool = False,
    axes: list[plt.Axes] | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    markevery: int | float | None = None,
    ncols: int = 3,
) -> list[plt.Axes]:
    perplot = list(df.groupby(subplot))

    # Get the colors to use
    colors = plt.get_cmap("tab10").colors
    markers = MARKERS
    cms = list(zip(colors, markers, strict=False))

    if axes is None:
        nrows = (len(perplot) + ncols - 1) // ncols
        _, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
        axes = list(axes.flatten())
    else:
        assert len(axes) >= len(perplot)

    for (plot_name, plot_group), ax in zip(perplot, axes, strict=False):
        # We now have group as a dataframe for a given plot, i.e. the task
        hue_groups = list(plot_group.groupby(hue))

        for (hue_name, hue_group), (color, marker) in zip(
            hue_groups,
            cms,
            strict=False,
        ):
            # We now have each individual hue group to plot, i.e. the pipeline

            fold_traces: list[pd.Series] = []
            for _, std_group in hue_group.groupby(std):
                # We now have each individual std group to plot, i.e. the fold
                _start = std_group[min_x].min()
                _x = (std_group[x] - _start).dt.total_seconds()
                _y = std_group[y]
                _s = pd.Series(_y.to_numpy(), index=_x).sort_index()
                _s = _s.cummin() if minimize else _s.cummax()
                fold_traces.append(_s)

            _hue_df = pd.concat(fold_traces, axis=1, sort=True).ffill().dropna()
            _means = _hue_df.mean(axis=1)
            _stds = _hue_df.std(axis=1)

            _means.plot(
                ax=ax,
                drawstyle="steps-post",
                color=color,
                marker=marker,
                label=hue_name,
                markevery=5,
            )
            ax.fill_between(
                _means.index,
                _means - _stds,
                _means + _stds,
                alpha=0.3,
                color=color,
                step="post",
            )

        ax.set_xlabel(x_label if x_label is not None else x)
        ax.set_ylabel(y_label if y_label is not None else y)
        ax.set_title(str(plot_name))
        ax.legend()

    return axes
