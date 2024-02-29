# ruff: noqa: PD901, PD015
from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from itertools import pairwise, product
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import seaborn as sns
from ConfigSpace import (
    CategoricalHyperparameter,
    Constant,
    OrdinalHyperparameter,
)
from ConfigSpace.util import deactivate_inactive_hyperparameters
from matplotlib.lines import Line2D
from scipy.spatial.distance import pdist, squareform
from sklearn.ensemble import RandomForestRegressor
from sklearn.manifold import MDS

from exps.methods import METHODS
from exps.pipelines import PIPELINES
from exps.plots import (
    COLORS,
    MARKER_SIZE,
    MARKERS,
    RENAMES,
)
from exps.seaborn2fig import SeabornFig2Grid

LEG_MARKER = 25

if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace
    from ConfigSpace.hyperparameters import Hyperparameter

logger = logging.getLogger(__name__)


def leg(**kwargs: Any) -> Line2D:
    return Line2D([], [], linestyle="", **kwargs)


def bin():
    # Bin any unscored configs if requested as there may be too many configs which
    # makes the MDS computation slow
    return
    """
    x               y               z               cat    other  num_nan
    (-0.01, 3.333]  (-0.01, 3.333]  (-0.01, 3.333]  True   a      (16.0, 20.0]     9
    (6.667, 10.0]   (6.667, 10.0]   (6.667, 10.0]   False  c      (7.988, 12.0]    6
    (3.333, 6.667]  (3.333, 6.667]  (3.333, 6.667]  True   b      (16.0, 20.0]     5
                                                    False  b      (16.0, 20.0]     3
                                                           c      (16.0, 20.0]     2
    (6.667, 10.0]   (6.667, 10.0]   (6.667, 10.0]   False  <NA>   (7.988, 12.0]    2
    (-0.01, 3.333]  (-0.01, 3.333]  (-0.01, 3.333]  True   b      (16.0, 20.0]     1
    (6.667, 10.0]   (6.667, 10.0]   (6.667, 10.0]   False  c      NaN              1
                                                           <NA>   NaN              1
    """
    bin_counts = (
        df[df[metric].isna()][list(self.config_cols)]
        .apply(  # type: ignore
            lambda x: pd.cut(x, bins=bins) if self.config_cols[x.name] is False else x,
        )
        .value_counts(dropna=False)
    )

    # Convert the binned configs to the center of the bin
    # and renamed the index of both to match
    """
                x       y       z      cat  other  num_nan
    binned_0  1.6615  1.6615  1.6615   True     a  18.000
    binned_1  8.3335  8.3335  8.3335  False     c   9.994
    binned_2  5.0000  5.0000  5.0000   True     b  18.000
    binned_3  5.0000  5.0000  5.0000  False     b  18.000
    binned_4  5.0000  5.0000  5.0000  False     c  18.000
    binned_5  8.3335  8.3335  8.3335  False  <NA>   9.994
    binned_6  1.6615  1.6615  1.6615   True     b  18.000
    binned_7  8.3335  8.3335  8.3335  False     c     NaN
    binned_8  8.3335  8.3335  8.3335  False  <NA>     NaN5
    """
    binned_center_configs = (
        bin_counts.index.to_frame()
        .map(lambda x: x.mid if isinstance(x, pd.Interval) else x)
        .reset_index(drop=True)
        .rename(lambda i: f"binned_{i}")
    )
    bin_counts.index = binned_center_configs.index


@dataclass
class MDSSurface:
    # All original data with emb-x and emb-y
    data: pd.DataFrame

    def select(self, index: pd.Index) -> pd.DataFrame:
        return self.data.loc[index]

    def area_model(
        self,
        *,
        model: Any | None = None,
        random_state: int = 0,
        granularity: int = 30,
        expand_axis: float = 1.0,
    ) -> Any:
        if model is None:
            _model = RandomForestRegressor(random_state=random_state, n_estimators=1000)
        else:
            _model = model

        grid_x, grid_y = self.embedding_grid(
            granularity=granularity,
            expand_axis=expand_axis,
        )

        # Array of squares = [square_1= [x1, x2, y1, y2], ..., square_n= [...]]
        # Same as pairwise(grid_x, grid_y) but a little more efficient
        squares = product(pairwise(grid_x), pairwise(grid_y))

        embedding_cols = self.data[["emb-x", "emb-y"]]
        assert isinstance(embedding_cols, pd.DataFrame)

        N = (granularity - 1) ** 2
        x = np.zeros((N, 2), dtype=float)
        y = np.zeros(N, dtype=float)

        for _i, ((x1, x2), (y1, y2)) in enumerate(squares):
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            x[_i] = center_x, center_y

            # Find if any row satisfies the condition
            valid_rows = embedding_cols.transform(
                {
                    # Check each row to see it's in the squares range
                    "emb-x": lambda x: (x >= x1) & (x <= x2),  # noqa: B023
                    "emb-y": lambda y: (y >= y1) & (y <= y2),  # noqa: B023
                },
            ).all(axis=1)

            if valid_rows.any():  # type: ignore
                y[_i] = 1

        return _model.fit(x, y)

    def performance_model(
        self,
        metric_col: str,
        fillna: float,
        *,
        model: Any | None = None,
        normalize: bool = True,
        random_state: int = 0,
    ) -> Any:
        training_data = self.data[["emb-x", "emb-y", metric_col]].copy()
        training_data = training_data.drop_duplicates()

        X = training_data[["emb-x", "emb-y"]]
        y = training_data[metric_col].copy()
        y = y.fillna(fillna)
        if normalize:
            y = (y - y.min()) / (y.max() - y.min())

        if model is None:
            _model = RandomForestRegressor(random_state=random_state, n_estimators=1000)
        else:
            _model = model

        return _model.fit(X=X, y=y)

    def heatmap(
        self,
        model: Any,
        *,
        granularity: int = 30,
        expand_axis: float = 1.0,
        threshold: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # returns x: 1d, y: 1d, z: 2d
        grid_x, grid_y = self.embedding_grid(
            granularity=granularity,
            expand_axis=expand_axis,
        )
        x_mesh, y_mesh = np.meshgrid(grid_x, grid_y)
        conc = np.c_[x_mesh.ravel(), y_mesh.ravel()]
        X = pd.DataFrame({"emb-x": conc[:, 0], "emb-y": conc[:, 1]})
        z = model.predict(X).reshape(x_mesh.shape)

        if threshold is not None:
            z[z < threshold] = 0

        return grid_x, grid_y, z

    def embedding_grid(
        self,
        *,
        granularity: int = 30,
        expand_axis: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        emb_x_min, emb_x_max = self.data["emb-x"].min(), self.data["emb-x"].max()
        emb_y_min, emb_y_max = self.data["emb-y"].min(), self.data["emb-y"].max()
        return (
            np.linspace(
                start=emb_x_min - expand_axis,
                stop=emb_x_max + expand_axis,
                num=granularity,
                endpoint=True,
            ),
            np.linspace(
                start=emb_y_min - expand_axis,
                stop=emb_y_max + expand_axis,
                num=granularity,
                endpoint=True,
            ),
        )

    @classmethod
    def generate(
        cls,
        X: pd.DataFrame,  # noqa: N803
        space: ConfigurationSpace,
        *,
        scaling_factors: Mapping[str, float] | None = None,
        random_state: int | None = None,
        distance_metric: (
            str | Callable[[np.ndarray, np.ndarray], np.floating | float]
        ) = "cityblock",
        n_jobs: int = 1,
        mds_kwargs: dict[str, Any] | None = None,
    ) -> MDSSurface:
        mds_kwargs = mds_kwargs or {}
        hps = list(space.get_hyperparameters())
        hp_names = [hp.name for hp in hps]

        x = normalizer(X=X[hp_names].copy(), hps=hps)  # type: ignore

        if scaling_factors is None:
            scaling_factors = {hp.name: 1 for hp in hps}

        mds = MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=random_state,
            n_jobs=n_jobs,
            **mds_kwargs,
        )

        # TODO: scikit-learn has a njobs version of distance calc, maybe its faster
        # but it would end up doing pairwise both ways
        print("Prepping data")
        xx = ready_for_l1_dist(
            df=x,  # type: ignore
            scaling_factors=scaling_factors,
            categoricals=[
                hp.name for hp in hps if isinstance(hp, CategoricalHyperparameter)
            ],
        )
        print("Calculating distances")
        dists = squareform(pdist(xx, metric=distance_metric))  # type: ignore

        print("Generating embedding")
        embedding = mds.fit_transform(dists)  # type: ignore
        assert isinstance(embedding, np.ndarray)

        embed_data = pd.DataFrame(embedding, columns=["emb-x", "emb-y"], index=X.index)  # type: ignore
        data = pd.concat([X, embed_data], axis=1)
        return MDSSurface(data)


def ready_for_l1_dist(
    df: pd.DataFrame,
    *,
    scaling_factors: Mapping[str, float],
    categoricals: Sequence[str],
) -> np.ndarray:
    r"""# NOTE: Assumes it can mutate x, please copy if not True.

    Original distance used this (also with a nan check which doesn't happen).

    ```python
    # Normal l1 distances
    d = np.abs(x - y)

    # Anywhere where one of the values was missing, set the distance for the element
    # to 1
    # This is already handled by the fact there is no nans as done by the encoding...
    # so we can ignore this line
    d[np.isnan(d)] = 1


    # Categorical distance is always 1
    category_differnt = np.logical_and(category_col_indicator, d != 0)
    d[category_different] = 1

    # Elementwise scale down by depth of the HP
    return np.sum(d / depth)
    ```

    We wan't to instead leverage scikit-learns optimized version of l1 distance
    to speed up computation. To do this, we can reform our data such that
    when you compute the l1 distance, you get the same result.

    Recall l1 is defined as:

        \sum_{i=1}^{n} |x_i - y_i|

    So the strategy is to invert the opterations such that we can
    directly apply `np.abs(converted_x - converted_y)` and get the same result.

    We can do this by:
    a) Assuming we have no NaNs (which is true by encoding that happens before this
    b) Scaling down values by the depth, column wise
    c) One hot encoding the categorical variables such that the distance is 1 if the
         categories are different
    """
    # We create a copy as we're going to start modifying inplace
    df = df.copy()

    # First we apply one hot encoding to the categorical variables,
    # this ensures that the distance is 1 if the categories are different
    numerical_cols = df.columns.difference(categoricals)
    for ncol in numerical_cols:
        df[ncol] = df[ncol] * scaling_factors[ncol]

    dummies = []
    for cat in categoricals:
        ohe = pd.get_dummies(
            df[cat],
            columns=cat,
            drop_first=False,
            dummy_na=False,
            dtype=float,
        )
        dummies.append(ohe * scaling_factors[cat])

    # apply pairwise l1 norm to this should be equivalent
    return pd.concat([df[numerical_cols], *dummies], axis=1).to_numpy()


def border_configs(
    configspace: ConfigurationSpace,
    *,
    n: int | None,
    random_state: int | None = None,
) -> list[dict[str, Any]]:
    def product_dict(**kwargs: Any):
        # https://stackoverflow.com/a/5228294/5332072
        keys = kwargs.keys()
        for instance in product(*kwargs.values()):
            yield dict(zip(keys, instance, strict=True))

    def boundaries(hp: Hyperparameter) -> Sequence[Any]:
        match hp:
            case CategoricalHyperparameter():
                return list(hp.choices)
            case OrdinalHyperparameter():
                return (hp.sequence[0], hp.sequence[-1])
            case Constant():
                return (hp.value,)
            case _:
                return (hp.lower, hp.upper)

    def valid_config(config: dict[str, Any]) -> bool:
        try:
            deactivate_inactive_hyperparameters(
                config,
                configspace,
            ).is_valid_configuration()
            return True
        except:  # noqa: E722
            return False

    # We can end up with non-sensical configurations due to conditionals
    # so we iterate and skip those
    borders = [
        c
        for c in product_dict(
            **{hp.name: boundaries(hp) for hp in configspace.values()},
        )
        if valid_config(c)
    ]
    if n is not None:
        rs = np.random.default_rng(random_state)
        borders = rs.choice(
            np.array(borders),
            size=min(n, len(borders)),
            replace=False,
        )
        return borders.tolist()

    return borders


def hp_depth(hp: Hyperparameter, space: ConfigurationSpace) -> int:
    parents = space.get_parents_of(hp)
    if not parents:
        return 1

    new_parents = parents
    d = 1
    while new_parents:
        d += 1
        old_parents = new_parents
        new_parents = []
        for p in old_parents:
            if (pp := space.get_parents_of(p)) is not None:
                new_parents.extend(pp)
            else:
                return d

    return d


def normalizer(
    X: pd.DataFrame,  # noqa: N803
    hps: Iterable[Hyperparameter],
    *,
    # TODO: Not sure what the default should be
    fillna: float = -0.5,  # https://github.com/automl/DeepCAVE/issues/120 not sure what default should be (-0.2 or -0.5)  # noqa: E501
    constant_fill: float = 1.0,  # Default according to DeepCave
) -> pd.DataFrame:
    def _normalized(s: pd.Series, hp: Hyperparameter) -> pd.Series:
        # Normalization for MDS according to DeepCave
        # NOTE: Unfortunatly, not all `_inverse_transform` methods can operate on vector
        match hp:
            case CategoricalHyperparameter():
                # https://github.com/automl/ConfigSpace/blob/ce27ba26e1cf27a5b9fe242519cb622327c82420/ConfigSpace/hyperparameters/categorical.pyx#L282-L285
                # Replace the categories with their index
                values = np.fromiter(
                    [hp.choices.index(v) if not pd.isna(v) else v for v in s],
                    dtype=int,
                    count=len(s),
                )
                return pd.Series(
                    values / (len(hp.choices) - 1),
                    index=s.index,
                )
            case OrdinalHyperparameter():
                raise NotImplementedError("OrdinalHyperparameter not implemented")
            case Constant():
                return pd.Series(np.full(len(s), constant_fill), index=s.index)
            case _:
                return s.transform(hp._inverse_transform)  # type: ignore

    hps = {hp.name: hp for hp in hps}
    return X.apply(
        lambda s: _normalized(s, hps[s.name]).fillna(fillna) if s.name in hps else s,
        axis=0,
    )  # type: ignore


def main():
    import argparse

    import matplotlib.pyplot as plt
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--njobs", type=int, default=-1)
    parser.add_argument("--figsize", type=int, default=7)
    parser.add_argument("--outpath", type=Path, required=True)
    parser.add_argument("--pipeline", type=str, choices=["mlp", "rf"], default="mlp")
    parser.add_argument("--granularity", type=int, default=30)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--ignore-too-many-configs", action="store_true")
    parser.add_argument(
        "--metric",
        type=str,
        default="metric:roc_auc_ovr [0.0, 1.0] (maximize)",
    )
    parser.add_argument("--dataset", type=int, default=168350)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--baseline",
        choices=list(METHODS.keys()),
        default="disabled",
        type=str,
    )
    parser.add_argument("--support", type=int, default=100)
    parser.add_argument("--borders", type=int, default=100)
    parser.add_argument(
        "--methods",
        choices=list(METHODS.keys()),
        default="current_average_worse_than_best_worst_split",
        nargs="+",
        type=str,
    )
    parser.add_argument("input", type=Path)
    args = parser.parse_args()

    method_col = "setting:cv_early_stop_strategy"
    fold_col = "setting:fold"
    space = PIPELINES[f"{args.pipeline}_classifier"].search_space(parser="configspace")

    df = pd.read_parquet(args.input)

    # Select out the dataset
    df = df[df["setting:task"] == args.dataset]  # type: ignore
    df = df[df["setting:fold"] == args.fold]  # type: ignore
    df = df[df["setting:cv_early_stop_strategy"].isin([args.baseline, *args.methods])]  # type: ignore
    assert isinstance(df, pd.DataFrame)
    df.index = df.index.astype(str)
    index_name = df.index.name

    config_cols = {
        hp.name: isinstance(hp, CategoricalHyperparameter)
        for hp in space.get_hyperparameters()
    }
    df = (
        df.rename(lambda c: c.removeprefix("config:"), axis=1)
        .astype({c: "category" for c, is_cat in config_cols.items() if is_cat})
        .convert_dtypes()
        .reset_index()
    )
    df["method"] = df[method_col].copy()
    df = df.set_index(keys=[method_col, fold_col, index_name]).sort_index()

    # TODO: Will break if we need to operate across folds... (maybe)
    border_df = pd.DataFrame.from_records(
        border_configs(space, n=args.borders, random_state=args.seed),
        index=[("border", 0, f"trial_{i}") for i in range(args.borders)],
    )

    cs = deepcopy(space)
    cs.seed(args.seed)
    if args.support > 0:
        sampled_configs = cs.sample_configuration(args.support)
        configs = [c.get_dictionary() for c in sampled_configs]
        support_df = pd.DataFrame.from_records(
            configs,
            index=[("support", 0, f"trial_{i}") for i in range(args.support)],
        )
    else:
        support_df = pd.DataFrame()

    baseline_data = df.xs(args.baseline, drop_level=False)
    method_datas = [df.xs(m, drop_level=False) for m in args.methods]

    X = pd.concat(
        [baseline_data, support_df, *method_datas, border_df],
        names=["method", "fold", "trial"],
    )
    if len(X) > 4000 and not args.ignore_too_many_configs:  # noqa: PLR2004
        raise ValueError(
            f"Too many configs {len(X)} for {args}, please choose another dataset"
            "or use --ignore-too-many-configs",
        )

    # We can't drop duplicates as we need to keep indices for later indexing,
    # this will make MDS work a little harder to produce it's embedding as a consequence.
    surface = MDSSurface.generate(
        X=X,
        space=space,
        scaling_factors={hp.name: 1 for hp in space.get_hyperparameters()},
        random_state=args.seed,
        distance_metric="cityblock",
        n_jobs=args.njobs,
        mds_kwargs={
            "max_iter": args.max_iter,
        },
    )
    real_data = surface.data.loc[df.index]
    best_point = real_data[
        real_data[args.metric] == real_data[args.metric].max()
    ].drop_duplicates()
    worst_score = real_data[args.metric].min()

    # Create performance heatmap
    perf_model = surface.performance_model(
        metric_col=args.metric,
        fillna=worst_score,
        normalize=True,
    )
    perf_x, perf_y, perf_z = surface.heatmap(
        perf_model,
        granularity=args.granularity,
        expand_axis=0,
    )
    # Create valid area heatmap
    area_model = surface.area_model(granularity=args.granularity, expand_axis=0.0)
    area_x, area_y, area_z = surface.heatmap(
        area_model,
        granularity=args.granularity,
        expand_axis=0,
    )
    area_z[area_z < 0.8] = 0  # noqa: PLR2004

    legend_markers = [
        leg(marker="x", color="black", label="Border", markersize=LEG_MARKER * 3 / 4),
        leg(
            markeredgecolor="black",
            marker=MARKERS[args.baseline],
            color=COLORS[args.baseline],
            label=RENAMES.get(args.baseline, args.baseline),
            markersize=LEG_MARKER,
        ),
        # leg(
        # markeredgecolor="black",
        # marker=MARKERS["both"],
        # color=COLORS["both"],
        # label="Both",
        # markersize=MARKER_SIZE / 2,
        # ),
    ]
    for method in args.methods:
        METHOD_NAME = RENAMES.get(method, method)
        EARLY_STOPPED_LABEL = f"{METHOD_NAME} (Stopped)"
        legend_markers.extend(
            [
                leg(
                    marker=".",
                    color=COLORS[method],
                    label=EARLY_STOPPED_LABEL,
                    markersize=LEG_MARKER,
                ),
                leg(
                    markeredgecolor="black",
                    marker=MARKERS[method],
                    color=COLORS[method],
                    label=METHOD_NAME,
                    markersize=LEG_MARKER,
                ),
            ],
        )

    grids = []
    things_to_plot = [args.baseline, *args.methods]
    for i_method, _method in enumerate(things_to_plot):
        # General plot figure
        g = sns.JointGrid(
            height=args.figsize,
            data=real_data,
            x="emb-x",
            y="emb-y",
            hue="method",
            palette=COLORS,
            xlim=(surface.data["emb-x"].min(), surface.data["emb-x"].max()),
            ylim=(surface.data["emb-y"].min(), surface.data["emb-y"].max()),
            space=0,
            marginal_ticks=False,
        )
        g.ax_joint.contourf(perf_x, perf_y, perf_z, cmap="Purples", alpha=0.5)
        g.ax_joint.contour(
            area_x,
            area_y,
            area_z,
            levels=1,
            colors="black",
            alpha=0.5,
            linestyles="dotted",
        )

        # Plot marginals on x-axis with a kde
        # We make all methods that are not this one transparent
        modified_pallette = deepcopy(COLORS)
        for k, _v in COLORS.items():
            if k != _method:
                modified_pallette[k] = (1, 1, 1, 1)

        sns.kdeplot(
            data=real_data,
            x="emb-x",
            hue="method",
            fill=True,
            common_norm=True,
            common_grid=True,
            palette=modified_pallette,
            ax=g.ax_marg_x,
            legend=False,
        )

        if i_method == len(things_to_plot) - 1:
            # Plot marginals on y-axis with a kde
            sns.kdeplot(
                data=real_data,
                x="emb-y",
                hue="method",
                fill=True,
                common_norm=True,
                common_grid=True,
                palette=COLORS,
                ax=g.ax_marg_y,
                vertical=True,
                legend=False,
            )
        else:
            g.ax_marg_y.remove()
            # Monkey patch
            g.ax_marg_y.monkey_deactivated = True

        # Now using the real data, we need to seperate it into different categories:
        # * Configs which both baseline and method fully scored (both)
        # * Configs which only the baseline scored (left_only)
        # * Configs which only the method scored (right_only)
        # * Configs which method early stopped
        _data = real_data[real_data["method"].isin([args.baseline, _method])]
        _data["is_scored"] = _data[args.metric].notna()

        config_cols = list(config_cols)
        merge_cols = [*config_cols]
        extra_cols = ["emb-x", "emb-y", "is_scored"]
        select_cols = merge_cols + extra_cols

        baseline_data = _data.xs(args.baseline, drop_level=False)[select_cols]
        method_data = _data.xs(_method, drop_level=False)[select_cols]

        print(best_point)
        print(best_point[[*config_cols, args.metric]])

        if _method == args.baseline:
            baseline_scored = baseline_data[baseline_data["is_scored"]]
            g.ax_joint.scatter(
                data=baseline_scored,
                x="emb-x",
                y="emb-y",
                color=COLORS[_method],
                marker=MARKERS[_method],
                edgecolors="black",
                label=RENAMES.get(_method, _method),
                s=MARKER_SIZE**2,
            )
        else:
            merged_data = pd.merge(
                baseline_data,
                method_data,
                how="outer",
                on=merge_cols,
                suffixes=("_baseline", "_method"),
                indicator=True,
            )
            both = merged_data[merged_data["_merge"] == "both"]

            # Baseline will have scored it
            both[both["is_scored_baseline"] & both["is_scored_method"]]
            method_scored = method_data[method_data["is_scored"]]
            method_only = merged_data[merged_data["_merge"] == "right_only"]
            method_only[method_only["is_scored_method"]]
            method_only_not_scored = method_only[~method_only["is_scored_method"]]

            g.ax_joint.scatter(
                data=method_only_not_scored,
                x="emb-x_method",
                y="emb-y_method",
                color=COLORS[_method],
                marker=".",
                # edgecolors="black",
                label=f"{RENAMES.get(_method, _method)} (Stopped)",
                alpha=0.7,
                s=MARKER_SIZE**2 * (3 / 4),
            )
            g.ax_joint.scatter(
                data=method_scored,
                x="emb-x",
                y="emb-y",
                color=COLORS[_method],
                marker=MARKERS[_method],
                edgecolors="black",
                label=RENAMES.get(_method, _method),
                s=MARKER_SIZE**2,
            )
            """
            g.ax_joint.scatter(
                data=both_scored,
                x="emb-x_baseline",
                y="emb-y_baseline",
                color=COLORS["both"],
                marker=MARKERS["both"],
                s=MARKER_SIZE**2,
                edgecolors="black",
            )
            g.ax_joint.scatter(
                data=method_only_scored,
                x="emb-x_method",
                y="emb-y_method",
                color=COLORS[_method],
                marker=MARKERS[_method],
                edgecolors="black",
                s=MARKER_SIZE**2,
            )
            """

        g.ax_joint.scatter(
            data=surface.data.loc[border_df.index],
            x="emb-x",
            y="emb-y",
            color="grey",
            marker="x",
            s=MARKER_SIZE,
        )
        g.ax_joint.scatter(
            data=best_point,
            x="emb-x",
            y="emb-y",
            color="none",
            marker="o",
            linewidths=2,
            label="Best",
            edgecolors="red",
            s=MARKER_SIZE**3,
            zorder=1,
        )
        g.ax_joint.scatter(
            data=best_point,
            x="emb-x",
            y="emb-y",
            color="red",
            marker="X",
            label="Best",
            edgecolors="black",
            s=MARKER_SIZE,
            zorder=1,
        )

        g.ax_joint.set_xlabel("")
        g.ax_joint.set_ylabel("")
        g.ax_joint.set_xticks([])
        g.ax_joint.set_xticklabels([])
        g.ax_joint.set_yticks([])
        g.ax_joint.set_yticklabels([])

        for ax in [g.ax_marg_x, g.ax_marg_y, g.ax_joint]:
            ax.tick_params(
                axis="both",  # changes apply to the x-axis
                which="both",  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                left=False,  # ticks along the left edge are off
                right=False,  # ticks along the right edge are off
                labelbottom=False,
            )  # labels along the bottom edge are off

        g.figure.tight_layout(h_pad=0, w_pad=0, pad=1.00)
        grids.append(g)

    # Place legend in top right of figure
    fig = plt.figure(
        figsize=(args.figsize * (len(args.methods) + 1), args.figsize),
        constrained_layout=False,
    )
    gridsp = fig.add_gridspec(
        1,
        len(args.methods) + 1,
        hspace=0,
        wspace=0,
        left=0,
        right=1,
        top=1,
        bottom=0,
    )

    [SeabornFig2Grid(g, fig, gridsp[i]) for i, g in enumerate(grids)]
    # Place legends below and center
    # Not sure why this is just so much bigger than everything else
    legend_markers.append(
        leg(
            markeredgecolor="red",
            marker="o",
            color="none",
            label="Best",
            markersize=LEG_MARKER,
        ),
    )
    fig.tight_layout(pad=1, w_pad=0, h_pad=0)
    fig.legend(
        loc="lower left",
        bbox_to_anchor=(0.011, 1.36),
        fontsize=LEG_MARKER,
        handles=legend_markers,
        ncol=len(legend_markers),
    )
    for ext in ["png", "pdf"]:
        fig.savefig(f"{args.outpath}.{ext}", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()
