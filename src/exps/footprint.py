# TODO: Metric should be converted to worst for Y, seems normalized cost
# TODO Seems that X values need to be imputed.
# ruff: noqa: PD901
from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from itertools import pairwise, product
from typing import TYPE_CHECKING, Any

import numpy as np
from ConfigSpace import (
    CategoricalHyperparameter,
    Constant,
    OrdinalHyperparameter,
)
from ConfigSpace.util import deactivate_inactive_hyperparameters
from pandas.core.dtypes.missing import partial
from scipy.spatial.distance import pdist, squareform
from sklearn.ensemble import RandomForestRegressor
from sklearn.manifold import MDS

from exps.pipelines import PIPELINES

if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace
    from ConfigSpace.hyperparameters import Hyperparameter

logger = logging.getLogger(__name__)


@dataclass
class MDSSurface:
    # All original data with emb-x and emb-y
    data: pd.DataFrame
    metric_col: str
    embedding_cols: list[str]

    def area_model(
        self,
        *,
        model: Any | None = None,
        random_state: int = 0,
        granularity: int = 30,
        expand_axis: float = 1.0,
    ) -> Any:
        if model is None:
            _model = RandomForestRegressor(random_state=random_state)
        else:
            _model = model

        grid_x, grid_y = self.embedding_grid(
            granularity=granularity,
            expand_axis=expand_axis,
        )

        # Array of squares = [square_1= [x1, x2, y1, y2], ..., square_n= [...]]
        # Same as pairwise(grid_x, grid_y) but a little more efficient
        squares = product(pairwise(grid_x), pairwise(grid_y))

        embedding_cols = df[["emb-x", "emb-y"]]
        assert isinstance(embedding_cols, pd.DataFrame)

        N = (granularity - 1) ** 2
        x = np.zeros((N, 2), dtype=int)
        y = np.zeros(N, dtype=int)

        for _i, ((x1, x2), (y1, y2)) in enumerate(squares):
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            x[_i] = center_x, center_y

            # Find if any row satisfies the condition
            valid_rows = embedding_cols.transform(
                {
                    # Check each row to see it's in the squares range
                    "emb-x": lambda x: x >= x1 & x <= x2,  # noqa: B023
                    "emb-y": lambda y: y >= y1 & y <= y2,  # noqa: B023
                },
            ).any(axis=1)

            if valid_rows.any():  # type: ignore
                y[_i] = 1

        return _model.fit(x, y)

    def performance_model(
        self,
        *,
        model: Any | None = None,
        random_state: int = 0,
    ) -> Any:
        valid_index = self.data[self.metric_col].notna()

        Xy = self.data[["emb-x", "emb-y", self.metric_col]].loc[valid_index]
        if model is None:
            _model = RandomForestRegressor(random_state=random_state)
        else:
            _model = model

        return _model.fit(X=Xy[["emb-x", "emb-y"]], y=Xy[self.metric_col])

    def heatmap(
        self,
        model: Any,
        *,
        granularity: int = 30,
        expand_axis: float = 1.0,
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
        embedding_cols: list[str],
        metric_col: str,
        distance_metric: Callable[[np.ndarray, np.ndarray], float | np.floating],
        *,
        random_state: int | None = None,
        n_jobs: int = 1,
    ) -> MDSSurface:
        distances = squareform(pdist(X[embedding_cols], metric=distance_metric))
        mds = MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=random_state,
            n_jobs=n_jobs,
        )
        embedding = mds.fit_transform(distances)
        assert isinstance(embedding, np.ndarray)

        embed_data = pd.DataFrame(embedding, columns=["emb-x", "emb-y"], index=X.index)  # type: ignore
        data = pd.concat([X, embed_data], axis=1)
        return MDSSurface(
            data=data,
            metric_col=metric_col,
            embedding_cols=embedding_cols,
        )


def config_distance(
    x: np.ndarray,
    y: np.ndarray,
    *,
    depth: np.ndarray,
    categories: np.ndarray,
) -> np.floating:
    d = np.abs(x - y)
    # category_and_not_zero = np.logical_and(categories, d != 0)
    # d[category_and_not_zero] = 1
    return np.sum(d / depth)


def border_configs(configspace: ConfigurationSpace) -> list[dict[str, Any]]:
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
    return [
        c
        for c in product_dict(
            **{hp.name: boundaries(hp) for hp in configspace.values()},
        )
        if valid_config(c)
    ]


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
    fillna: float = -0.2,  # https://github.com/automl/DeepCAVE/issues/120 not sure what default should be (-0.2 or -0.5)  # noqa: E501
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


def create_surface(
    df: pd.DataFrame,
    configspace: ConfigurationSpace,
    metric_col: str,
    *,
    n_jobs: int = 1,
    random_state: int | None = None,
    max_border_configs: int | None = 100,
    n_random_configs_in_embedding: int | None = 100,
) -> MDSSurface:
    if metric_col not in df.columns:
        raise ValueError(f"Metric column '{metric_col}' not found in DataFrame")

    cs = deepcopy(configspace)
    cs.seed(0)
    hps = list(cs.get_hyperparameters())
    hp_names = {hp.name for hp in hps}

    if not (set(df.columns) >= hp_names):
        raise ValueError(
            "DataFrame columns must be a superset of the hyperparameters"
            f"\nDataFrame columns: {sorted(df.columns)}"
            f"\nHyperparameters: {sorted(hp_names)}",
        )

    dfs = [df]

    if max_border_configs is not None:
        borders = np.array(border_configs(configspace))
        if max_border_configs > 0:
            rs = np.random.default_rng(random_state)
            borders = rs.choice(
                borders,
                size=min(max_border_configs, len(borders)),
                replace=False,
            )
        _borders = pd.DataFrame.from_records(
            data=list(borders),
            index=[f"border_{i}" for i in range(len(borders))],
        )
        dfs.append(_borders)

    if n_random_configs_in_embedding is not None:
        random_configs = [
            config.get_dictionary()
            for config in cs.sample_configuration(n_random_configs_in_embedding)
        ]
        _random = pd.DataFrame.from_records(
            data=list(random_configs),
            index=[f"random_{i}" for i in range(len(random_configs))],
        )
        dfs.append(_random)

    # Defaults to -0.5 and 1.0 according to DeepCave
    X = pd.concat(dfs)
    assert isinstance(X, pd.DataFrame)
    X = normalizer(X=X, hps=hps)  # type: ignore
    assert isinstance(X, pd.DataFrame)

    _metric = partial(
        config_distance,
        depth=np.array([hp_depth(hp, configspace) for hp in hps]),
        categories=np.array([isinstance(hp, CategoricalHyperparameter) for hp in hps]),
    )

    return MDSSurface.generate(
        X,
        metric_col=metric_col,
        distance_metric=_metric,
        n_jobs=n_jobs,
        random_state=random_state,
        embedding_cols=[c for c in df.columns if c in sorted(hp_names)],
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_parquet("mlp-nsplits-10.parquet")
    df = df.reset_index(drop=True)
    df.index = df.index.astype(str).map("scored_{}".format)

    pipeline = PIPELINES["mlp_classifier"]

    df = df.rename(lambda c: c.removeprefix("config:"), axis=1)
    metric = "metric:roc_auc_ovr [0.0, 1.0] (maximize)"
    surface = create_surface(
        df=df,
        metric_col=metric,
        configspace=pipeline.search_space(parser="configspace"),
        random_state=0,
        max_border_configs=100,
        n_random_configs_in_embedding=None,
    )
    perf_model = surface.performance_model()

    heatmap = surface.heatmap(perf_model)

    fig, ax = plt.subplots()
    countours = ax.contourf(*heatmap, cmap=plt.cm.bone_r)
    colorbar_heatmap = fig.colorbar(countours, ax=ax)
    ax.grid(visible=False)

    scored_data = surface.data[
        surface.data[metric].notna() & surface.data.index.str.startswith("scored_")
    ]
    unscored_data = surface.data[
        surface.data[metric].isna() & surface.data.index.str.startswith("scored_")
    ]
    border_data = surface.data[surface.data.index.str.startswith("border_")]
    random_data = surface.data[surface.data.index.str.startswith("random_")]
    scatter_cb_data = ax.scatter(
        x="emb-x",
        y="emb-y",
        c="white",
        data=unscored_data,
        label="early-stopped",
        marker=".",
    )
    scatter_cb_data = ax.scatter(
        x="emb-x",
        y="emb-y",
        c=metric,
        data=scored_data,
        label="evaluated",
        cmap="viridis",
    )
    colorbar_scatter = fig.colorbar(scatter_cb_data, ax=ax)
    ax.scatter(
        x="emb-x",
        y="emb-y",
        c="black",
        data=border_data,
        marker="x",
        label="border",
    )
    ax.scatter(
        x="emb-x",
        y="emb-y",
        c="black",
        data=random_data,
        marker="*",
        label="support",
    )
    ax.axis("off")
    ax.legend()
    plt.show()
