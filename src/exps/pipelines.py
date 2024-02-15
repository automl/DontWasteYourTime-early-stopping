from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from amltk.pipeline import Component, Sequential, Split, request, Choice
from ConfigSpace import Float, Integer
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
                    name="one_hot"
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

PIPELINES = {
    "mlp_classifier": mlp_classifier,
}
