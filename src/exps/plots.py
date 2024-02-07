from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from copy import copy

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
    x_start: str,
    std: str,
    hue: str | list[str],
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
                _start = std_group[x_start].min()
                _x = (std_group[x] - _start).dt.total_seconds()
                _y = std_group[y]
                _s = pd.Series(_y.to_numpy(), index=_x).sort_index()
                lower, upper = metric_bounds

                bounded = not np.isinf(lower) and not np.isinf(upper)
                normalized: bool
                inverted: bool

                # Transform everything
                match minimize, bounded:
                    # Bounded metrics, 0-1 normalize
                    case True, True:
                        _s = _s.cummin()
                        _s = (_s - lower) / (upper - lower)
                        normalized = True
                        inverted = False
                    case True, False:
                        _s = _s.cummin()
                        normalized = False
                        inverted = False
                    case False, True:
                        _s = _s.cummax()
                        _s = (_s - lower) / (upper - lower)
                        _s = 1 - _s
                        normalized = True
                        inverted = True
                    case False, False:
                        _s = _s.cummax()
                        _s = -_s
                        normalized = False
                        inverted = True
                    case _:
                        raise RuntimeError("Shouldn't get here")

                if x_bounds is not None and x_bounds[1] is not None:
                    _new_index = np.concatenate([_s.index, [x_bounds[1]]])
                    _new_values = np.concatenate([_s.to_numpy(), [_s.iloc[-1]]])
                    _s = pd.Series(_new_values, _new_index)

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
                markevery=markevery,
            )
            ax.fill_between(
                _means.index,
                _means - _stds,
                _means + _stds,
                alpha=0.3,
                color=color,
                step="post",
            )

        if x_bounds:
            ax.set_xlim(*x_bounds)
        if y_bounds:
            ax.set_ylim(*y_bounds)

        if log_y:
            ax.set_yscale("log")

        _y_label = copy(y_label)
        if _y_label is not None:
            if inverted:  # type: ignore
                _y_label = f"{_y_label} | inverted"
            if normalized:  # type: ignore
                _y_label = f"{_y_label} | normalized"

        ax.set_xlabel(x_label if x_label is not None else x)
        ax.set_ylabel(_y_label if _y_label is not None else y)
        ax.set_title(str(plot_name))
        ax.legend()

    return axes
