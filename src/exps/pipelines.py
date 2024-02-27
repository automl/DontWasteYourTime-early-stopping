from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from amltk.pipeline import Choice, Component, Sequential, Split, request
from ConfigSpace import Categorical, Float, Integer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def mlp_config_transform(config: Mapping[str, Any], _: Any) -> Mapping[str, Any]:
    new_config = dict(config)
    num_nodes_per_layer = new_config.pop("num_nodes_per_layer")
    hidden_layer_depth = new_config.pop("hidden_layer_depth")

    hidden_layer_sizes = tuple([int(num_nodes_per_layer)] * int(hidden_layer_depth))
    new_config["hidden_layer_sizes"] = hidden_layer_sizes
    return new_config


mlp_classifier = Sequential(
    Split(
        {
            "categorical": [
                Component(
                    OrdinalEncoder,
                    space={"min_frequency": (0.01, 0.5)},
                    config={
                        "categories": "auto",
                        "handle_unknown": "use_encoded_value",
                        "unknown_value": -1,
                        "encoded_missing_value": -2,
                    },
                ),
                Choice(
                    "passthrough",
                    Component(
                        OneHotEncoder,
                        space={"max_categories": (2, 20)},
                        config={
                            "categories": request("categories", default="auto"),
                            "drop": None,
                            "sparse_output": False,
                            "handle_unknown": "infrequent_if_exist",
                        },
                    ),
                    name="one_hot",
                ),
            ],
            "numerical": Component(
                SimpleImputer,
                space={"strategy": ["mean", "median"]},
            ),
        },
        name="encoding",
    ),
    Component(StandardScaler, name="standarization"),
    Component(
        # NOTE: This space should not be used for evaluating how good this MLP is
        # vs other algorithms
        item=MLPClassifier,
        config={
            "random_state": request("random_state", default=None),
            "warm_start": False,
            "n_iter_no_change": 32,
            "validation_fraction": 0.1,
            "tol": 1e-4,
            "solver": "adam",
            "batch_size": "auto",
            "shuffle": True,
            "beta_1": 0.9,
            "beta_2": 0.999,
            "epsilon": 1e-8,
            "max_iter": 512,
        },
        config_transform=mlp_config_transform,
        space={
            "hidden_layer_depth": Integer("hidden_layer_depth", (1, 3), default=1),
            "num_nodes_per_layer": Integer(
                "num_nodes_per_layer",
                (16, 264),
                default=32,
                log=True,
            ),
            "momentum": (0.8, 1),
            "activation": ["relu", "tanh"],
            "alpha": Float("alpha", (1e-7, 1e-1), default=1e-4, log=True),
            "learning_rate": ["constant", "invscaling", "adaptive"],
            "learning_rate_init": Float(
                "learning_rate_init",
                (1e-4, 0.5),
                default=1e-3,
                log=True,
            ),
            "early_stopping": [True, False],
        },
    ),
    name="mlp_classifier",
)


def rf_config_transform(config: Mapping[str, Any], _: Any) -> Mapping[str, Any]:
    new_config = dict(config)
    if new_config["class_weight"] == "None":
        new_config["class_weight"] = None
    return new_config


rf_classifier = Sequential(
    Split(
        {
            "categorical": [
                Component(
                    OrdinalEncoder,
                    config={
                        "categories": "auto",
                        "handle_unknown": "use_encoded_value",
                        "unknown_value": -1,
                        "encoded_missing_value": -2,
                    },
                ),
                Choice(
                    "passthrough",
                    Component(
                        OneHotEncoder,
                        space={"max_categories": (2, 20)},
                        config={
                            "categories": request("categories", default="auto"),
                            "drop": None,
                            "sparse_output": False,
                            "handle_unknown": "infrequent_if_exist",
                        },
                    ),
                    name="one_hot",
                ),
            ],
            "numerical": Component(
                SimpleImputer,
                space={"strategy": ["mean", "median"]},
            ),
        },
        name="encoding",
    ),
    Component(
        # NOTE: This space should not be used for evaluating how good this RF is
        # vs other algorithms
        RandomForestClassifier,
        config_transform=rf_config_transform,
        space={
            "criterion": ["gini", "entropy"],
            "max_features": Categorical(
                "max_features",
                list(np.logspace(0.1, 1, base=10, num=10) / 10),
                ordered=True,
            ),
            "min_samples_split": Integer(
                "min_samples_split",
                bounds=(2, 20),
                default=2,
            ),
            "min_samples_leaf": Integer("min_samples_leaf", bounds=(1, 20), default=1),
            "bootstrap": Categorical("bootstrap", [True, False], default=True),
            "class_weight": ["balanced", "balanced_subsample", "None"],
            "min_impurity_decrease": (1e-9, 1e-1),
        },
        config={
            "random_state": request("random_state", default=None),
            "n_estimators": 512,
            "max_depth": None,
            "min_weight_fraction_leaf": 0.0,
            "max_leaf_nodes": None,
            "warm_start": False,  # False due to no iterative fit used here
            "n_jobs": 1,
        },
    ),
    # Whoops, should have added `name="rf_classifier"` here
    # We can circumvent this by renaming these columns such that these columns overlap as
    # needed. This is only relevant for footprint plots.
)

PIPELINES = {
    "mlp_classifier": mlp_classifier,
    "rf_classifier": rf_classifier,
}
