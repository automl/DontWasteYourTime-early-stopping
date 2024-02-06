from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Any

from amltk import Choice, Node
from amltk.pipeline import Component, Sequential, Split, request
from ConfigSpace import (
    ConfigurationSpace,
    EqualsCondition,
    Float,
    ForbiddenAndConjunction,
    ForbiddenEqualsClause,
    InCondition,
    Integer,
)
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier


def config_transform_convert_string_values(
    config: Mapping[str, Any],
    _: Any,
) -> dict[str, Any]:
    return {
        k: True
        if v == "True"
        else False
        if v == "False"
        else None
        if v == "None"
        else v
        for k, v in config.items()
    }


def sgd_classifier_create_space() -> ConfigurationSpace:
    params = {
        "alpha": Float("alpha", (1e-7, 1e-1), default=1e-4, log=True),
        "average": [False, True],
        "epsilon": Float("epsilon", (1e-5, 1e-1), default=1e-4, log=True),
        "eta0": Float("eta0", (1e-7, 1e-1), default=1e-2, log=True),
        "l1_ratio": Float("l1_ratio", (1e-9, 1), default=0.15, log=True),
        "learning_rate": ["invscaling", "optimal", "constant"],
        "loss": [
            "log_loss",
            "hinge",
            "modified_huber",
            # "squared_hinge",  # No probability with "squared_hinge"
            "perceptron",
        ],
        "penalty": ["l2", "l1", "elasticnet"],
        "power_t": Float("power_t", (1e-5, 1), default=0.5, log=True),
        "tol": Float("tol", (1e-5, 1e-1), default=1e-4, log=True),
        "class_weight": ["None", "balanced"],
    }
    cs = ConfigurationSpace(params)
    cs.add_conditions(
        [
            EqualsCondition(cs["l1_ratio"], cs["penalty"], "elasticnet"),
            EqualsCondition(cs["epsilon"], cs["loss"], "modified_huber"),
            EqualsCondition(cs["power_t"], cs["learning_rate"], "invscaling"),
            InCondition(
                cs["eta0"],
                cs["learning_rate"],
                ["invscaling", "constant"],
            ),
        ],
    )
    return cs


sgd_classifier_component = Component(
    name="sgd_classifier",
    item=SGDClassifier,
    config={"fit_intercept": True, "warm_start": True, "shuffle": True},
    config_transform=config_transform_convert_string_values,
    space=sgd_classifier_create_space(),
)

rf_classifier = Sequential(
    Split(
        {
            "categorical": Component(
                OrdinalEncoder,
                space={"min_frequency": (0.001, 0.5), "max_categories": (2, 50)},
                config={"handle_unknown": "use_encoded_value", "unknown_value": -1},
            ),
            "numerical": Component(
                SimpleImputer,
                space={"strategy": ["mean", "median"]},
            ),
        },
        name="categorical_encoding",
    ),
    Component(
        item=RandomForestClassifier,
        config={
            "max_depth": None,
            "min_weight_fraction_leaf": 0.0,
            "max_leaf_nodes": None,
            "min_impurity_decrease": 0.0,
            "random_state": request("random_state", default=None),
            "warm_start": True,
        },
        space={
            "criterion": ["gini", "entropy"],
            "bootstrap": [True, False],
            "min_samples_split": Integer(
                "min_samples_split",
                bounds=(2, 20),
                default=2,
            ),
            "min_samples_leaf": Integer("min_samples_leaf", bounds=(1, 20), default=1),
            "max_features": Float("max_features", bounds=(0.1, 1.0), default=0.5),
        },
    ),
    name="random_forest_classifier",
)

qda_component = Component(
    name="qda",
    item=QuadraticDiscriminantAnalysis,
    space={"reg_param": Float("reg_param", bounds=(0, 1), default=0.0)},
)

passive_aggresive_classifier_component = Component(
    name="passive_aggressive",
    item=PassiveAggressiveClassifier,
    config={
        "warm_start": True,
        "random_state": request("random_state", default=None),
        "fit_intercept": True,
    },
    space={
        "C": Float("C", bounds=(1e-5, 10), default=1.0, log=True),
        "loss": ["hinge", "squared_hinge"],
        "tol": Float("tol", bounds=(1e-5, 1e-1), default=1e-4, log=True),
        "average": [False, True],
    },
)

multinomial_nb_component = Component(
    name="multinomial_nb",
    item=MultinomialNB,
    space={
        "alpha": Float("alpha", bounds=(1e-2, 100.0), default=1.0, log=True),
        "fit_prior": [True, False],
    },
)


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
                    },
                ),
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
            ],
            "numerical": Component(
                SimpleImputer,
                space={"strategy": ["mean", "median"]},
            ),
        },
        name="encoding",
    ),
    Split(
        Component(MinMaxScaler, name="numerical"),
        name="scaling",
    ),
    Component(
        item=MLPClassifier,
        config={
            "random_state": request("random_state", default=None),
            "warm_start": True,
            "n_iter_no_change": 32,
            "validation_fraction": 0.1,
            "tol": 1e-4,
            "solver": "adam",
            "batch_size": "auto",
            "shuffle": True,
            "beta_1": 0.9,
            "beta_2": 0.999,
            "epsilon": 1e-8,
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
            "activation": ["relu", "tanh"],
            "alpha": Float("alpha", (1e-7, 1e-1), default=1e-4, log=True),
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


def svc_create_space() -> ConfigurationSpace:
    params = {
        "C": Float("C", bounds=(0.03125, 32768), default=1.0, log=True),
        "kernel": ["rbf", "poly", "sigmoid"],
        "class_weight": ["balanced", "None"],
        "coef0": Float("coef0", bounds=(-1.0, 1.0), default=0.0),
        "degree": Integer("degree", bounds=(2, 5), default=3),
        "gamma": Float(
            "gamma",
            bounds=(3.0517578125e-05, 8),
            default=0.1,
            log=True,
        ),
        "shrinking": [True, False],
        "tol": Float("tol", bounds=(1e-5, 1e-1), default=1e-3, log=True),
    }
    cs = ConfigurationSpace(params)
    cs.add_conditions(
        [
            EqualsCondition(cs["degree"], cs["kernel"], "poly"),
            InCondition(cs["coef0"], cs["kernel"], ["poly", "sigmoid"]),
        ],
    )
    return cs


def svc_config_transform(config: Mapping[str, Any], _: Any) -> Mapping[str, Any]:
    new_config = dict(config)
    if new_config.get("class_weight") == "None":
        new_config["class_weight"] = None

    try:
        import resource

        soft, _ = resource.getrlimit(resource.RLIMIT_AS)

        if soft > 0:
            soft //= 1024 * 1024
            maxrss = resource.getrusage(resource.RUSAGE_SELF)[2] / 1024

            if sys.platform == "darwin":
                maxrss /= 1024

            cache_size = (soft - maxrss) / 2

            if cache_size < 0:
                cache_size = 200
        else:
            cache_size = 200
    except Exception:  # noqa: BLE001
        cache_size = 200

    new_config["cache_size"] = int(cache_size)
    return new_config


svc_component = Component(
    name="svc",
    item=SVC,
    space=svc_create_space(),
    config={
        "random_state": request("random_state", default=None),
        "decision_function_shape": "ovr",
        "probability": True,
        "max_iter": -1,
        "cache_size": 200,
    },
    config_transform=svc_config_transform,
)


def liblinear_svc_create_space() -> ConfigurationSpace:
    cs = ConfigurationSpace(
        {
            "penalty": ["l2", "l1"],
            "loss": ["squared_hinge", "hinge"],
            "tol": Float("tol", bounds=(1e-5, 1e-1), default=1e-4, log=True),
            "C": Float("C", bounds=(0.03125, 32768), default=1.0, log=True),
            "class_weight": ["balanced", "None"],
        },
    )
    cs.add_forbidden_clause(
        ForbiddenAndConjunction(
            ForbiddenEqualsClause(cs["penalty"], "l1"),
            ForbiddenEqualsClause(cs["loss"], "hinge"),
        ),
    )
    return cs


liblinear_svc_component = Component(
    name="liblinear_svc",
    item=LinearSVC,
    space=liblinear_svc_create_space(),
    config={
        "dual": "auto",
        "multi_class": "ovr",
        "random_state": request("random_state", default=None),
        "fit_intercept": True,
        "intercept_scaling": 1,
    },
    config_transform=config_transform_convert_string_values,
)


def lda_config_transform(config: Mapping[str, Any], _: Any) -> Mapping[str, Any]:
    new_config = dict(config)
    shrinkage = new_config.pop("shrinkage", None)

    if shrinkage == "None":
        new_config.update({"shrinkage": None, "solver": "svd"})
        return new_config

    if shrinkage == "auto":
        new_config.update({"shrinkage": "auto", "solver": "lsqr"})
        return new_config

    if shrinkage == "manual":
        shrinkage_factor = new_config.pop("shrinkage_factor")
        new_config.update({"shrinkage": float(shrinkage_factor), "solver": "lsqr"})
        return new_config

    raise ValueError(shrinkage)


def lda_space() -> ConfigurationSpace:
    cs = ConfigurationSpace(
        {
            "shrinkage": ["None", "auto", "manual"],
            "shrinkage_factor": Float(
                "shrinkage_factor",
                bounds=(0, 1),
                default=0.5,
            ),
            "tol": Float("tol", bounds=(1e-5, 1e-1), default=1e-4, log=True),
        },
    )
    cs.add_condition(EqualsCondition(cs["shrinkage_factor"], cs["shrinkage"], "manual"))
    return cs


lda_component = Component(
    name="lda",
    item=LinearDiscriminantAnalysis,
    space=lda_space(),
    config_transform=lda_config_transform,
)

knn_classifier = Sequential(
    Split(
        {
            "categorical": Choice(
                Component(
                    OrdinalEncoder,
                    space={"min_frequency": (0.001, 0.5), "max_categories": (2, 50)},
                ),
                Component(
                    OneHotEncoder,
                    space={"min_frequency": (0.001, 0.5), "max_categories": (2, 50)},
                    config={"drop": "if_binary", "sparse_output": False},
                ),
                name="categorical_encoder",
            ),
            "numerical": Component(
                SimpleImputer,
                space={"strategy": ["mean", "median"]},
            ),
        },
        name="categorical_encoding",
    ),
    Split(
        {
            "categorical": "passthrough",
            "numerical": Component(MinMaxScaler),
        },
        name="numerical_scaler",
    ),
    Component(
        KNeighborsClassifier,
        space={
            "n_neighbors": Integer("n_neighbors", (1, 100), log=True, default=1),
            "weights": ["uniform", "distance"],
            "p": [2, 1],
        },
    ),
    name="knn_classifier",
)


def hist_gradient_boosting_space() -> ConfigurationSpace:
    params = {
        "learning_rate": Float("learning_rate", (0.01, 1), default=0.1, log=True),
        "min_samples_leaf": Integer(
            "min_samples_leaf",
            (1, 200),
            default=20,
            log=False,
        ),
        "max_leaf_nodes": Integer(
            "max_leaf_nodes",
            (3, 2047),
            default=31,
            log=True,
        ),
        "l2_regularization": Float(
            "l2_regularization",
            (1e-10, 1),
            default=1e-10,
            log=True,
        ),
        "early_stopping": [True, False],
        "n_iter_no_change": Integer("n_iter_no_change", (1, 20), default=10),
        "validation_fraction": Float(
            "validation_fraction",
            (0.01, 0.4),
            default=0.1,
        ),
    }
    cs = ConfigurationSpace(params)
    cs.add_conditions(
        [
            EqualsCondition(
                cs["validation_fraction"],
                cs["early_stopping"],
                True,  # noqa: FBT003
            ),
            EqualsCondition(
                cs["n_iter_no_change"],
                cs["early_stopping"],
                False,  # noqa: FBT003
            ),
        ],
    )
    return cs


def hist_gradient_boosting_classifier_config_transform(
    config: Mapping[str, Any],
    _: Any,
) -> Mapping[str, Any]:
    new_config = dict(config)
    early_stopping = new_config.pop("early_stopping")

    if early_stopping:
        return new_config

    new_config.pop("n_iter_no_change", None)
    new_config.pop("validation_fraction", None)
    return new_config


hist_gradient_boosting_classifier_component = Component(
    name="hist_gradient_boosting_classifier",
    item=HistGradientBoostingClassifier,
    config={
        "loss": "log_loss",
        "max_depth": None,
        "max_bins": 255,
        "tol": 1e-7,
        "scoring": "loss",
        "random_state": request("random_state", default=None),
        "warm_start": True,
    },
    config_transform=hist_gradient_boosting_classifier_config_transform,
    space=hist_gradient_boosting_space(),
)

gaussian_nb_component = Component(
    name="gaussian_nb",
    item=GaussianNB,
)

extra_trees_classifier_component = Component(
    name="extra_trees_classifier",
    item=ExtraTreesClassifier,
    config={
        "random_state": request("random_state", default=None),
        "min_weight_fraction_leaf": 0.0,
        "max_leaf_nodes": None,
        "min_impurity_decrease": 0.0,
        "max_depth": None,
        "oob_score": False,
        "n_jobs": 1,
        "verbose": 0,
        "warm_start": True,
    },
    space={
        "bootstrap": [True, False],
        "class_weight": ["balanced", "balanced_subsample", "None"],
        "criterion": ["gini", "entropy"],
        "min_samples_split": Integer("min_samples_split", bounds=(2, 20), default=2),
        "min_samples_leaf": Integer("min_samples_leaf", bounds=(1, 20), default=1),
        "max_features": Float("max_features", bounds=(0.0, 1.0), default=0.5),
    },
    config_transform=config_transform_convert_string_values,
)

decision_tree_classifier_component = Component(
    name="decision_tree_classifier",
    item=DecisionTreeClassifier,
    config={
        "random_state": request("random_state", default=None),
        "min_weight_fraction_leaf": 0.0,
        "max_features": 1.0,
        "max_leaf_nodes": None,
        "max_depth": None,
        "min_impurity_decrease": 0.0,
    },
    config_transform=config_transform_convert_string_values,
    space={
        "criterion": ["gini", "entropy"],
        "min_samples_split": Integer("min_samples_split", bounds=(2, 20), default=5),
        "min_samples_leaf": Integer("min_samples_leaf", bounds=(1, 20), default=1),
        "class_weight": ["balanced", "None"],
    },
)

bernoulli_component = Component(
    name="bernoulli_nb",
    item=BernoulliNB,
    space={
        "alpha": Float("alpha", bounds=(1e-2, 100), default=1, log=True),
        "fit_prior": [True, False],
    },
)


def create_adaboost_estimator_with_decision_tree(
    *_: Any,
    **kwargs: Any,
) -> AdaBoostClassifier:
    return AdaBoostClassifier(
        estimator=DecisionTreeClassifier(
            max_depth=kwargs["max_depth"],
            random_state=kwargs["random_state"],
        ),
        learning_rate=kwargs["learning_rate"],
        algorithm=kwargs["algorithm"],
        n_estimators=kwargs["n_estimators"],
        random_state=kwargs["random_state"],
    )


adaboost_classifier_component = Component(
    item=create_adaboost_estimator_with_decision_tree,
    name="adaboost_classifier",
    config={
        "random_state": request("random_state", default=None),
    },
    space={
        "n_estimators": Integer("n_estimators", bounds=(50, 500), default=50),
        "learning_rate": Float(
            "learning_rate",
            bounds=(0.01, 2),
            default=0.1,
            log=True,
        ),
        "algorithm": ["SAMME.R", "SAMME"],
        "max_depth": Integer("max_depth", bounds=(1, 10), default=1),
    },
)

CLASSIFIERS: tuple[Node, ...] = (
    adaboost_classifier_component,
    bernoulli_component,
    decision_tree_classifier_component,
    extra_trees_classifier_component,
    gaussian_nb_component,
    hist_gradient_boosting_classifier_component,
    knn_classifier,
    lda_component,
    liblinear_svc_component,
    mlp_classifier,
    multinomial_nb_component,
    passive_aggresive_classifier_component,
    qda_component,
    rf_classifier,
    sgd_classifier_component,
    svc_component,
)
