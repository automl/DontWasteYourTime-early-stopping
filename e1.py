from __future__ import annotations

import sys
from collections import defaultdict
from itertools import chain, product
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias

import pandas as pd
from rich import print

from exps.experiments.exp1 import E1, PIPELINES
from exps.methods import METHODS
from exps.metrics import METRICS
from exps.plots import (
    incumbent_traces_aggregated,
    incumbent_traces_aggregated_no_test,
    incumbent_traces_aggregated_with_test,
    ranking_plots_aggregated,
    speedup_plots,
)
from exps.slurm import seconds_to_slurm_time
from exps.tasks import TASKS
from exps.util import shrink_dataframe

if TYPE_CHECKING:
    from amltk.optimization import Metric


EXP_NAME: TypeAlias = Literal[
    "debug",
    "reproduce",
    "time-analysis",
    "category3-nsplits-2-5",
    "category3-nsplits-20",
    "category3-nsplits-10",
    "category3-nsplits-5",
    "category3-nsplits-3",
    "category4-nsplits-2-5",
    "category4-nsplits-20",
    "category4-nsplits-10",
    "category4-nsplits-5",
    "category4-nsplits-3",
    "category5-nsplits-10",
    "category5-nsplits-20",
    "category6-nsplits-10",
    "category6-nsplits-20",
    "category7-nsplits-20-unseeded",
    "category8-nsplits-20-unseeded",
]
EXP_CHOICES = [
    "debug",
    "reproduce",
    "time-analysis",
    # -------
    "category3-nsplits-2-5",  # MLP pipeline (2 repeat, 5 fold)
    "category3-nsplits-20",  # MLP pipeline (2 repeat, 10 fold)
    "category3-nsplits-10",  # MLP pipeline
    "category3-nsplits-5",  # MLP pipeline
    "category3-nsplits-3",  # MLP pipeline
    # -------
    "category4-nsplits-2-5",  # RF pipeline (2 repeat, 5 fold)
    "category4-nsplits-20",  # RF pipeline (2 repeat, 10 fold)
    "category4-nsplits-10",  # RF pipeline
    "category4-nsplits-5",  # RF pipeline
    "category4-nsplits-3",  # RF pipeline
    # -------
    "category5-nsplits-10",  # {Default SMAC, SMAC w/ early stop mean report} MLP
    "category5-nsplits-20",  # {Default SMAC, SMAC w/ early stop mean report} MLP
    # -------
    "category6-nsplits-10",  # {Default SMAC, SMAC w/ early stop mean report} RF
    "category6-nsplits-20",  # {Default SMAC, SMAC w/ early stop mean report} RF
    # ---
    "category7-nsplits-20-unseeded",  # MLP pipeline (2 repeat, 10 fold) (unseeded inner)
    "category8-nsplits-20-unseeded",  # RF pipeline (2 repeat, 10 fold) (unseeded inner)
]


def cols_needed_for_plotting(
    metric: Metric | str,
    n_splits: int | None = None,
) -> list[str]:
    if isinstance(metric, str):
        metric = METRICS[metric]

    CORE = [
        "created_at",
        "reported_at",
        "status",
        "setting:fold",
        "setting:metric",
        "setting:optimizer",
        "setting:task",
        "setting:n_splits",
        "setting:pipeline",
        "setting:cv_early_stop_strategy",
    ]
    METRIC_COLS = [
        f"metric:{metric}",
        f"summary:val_mean_{metric.name}",
        f"summary:val_std_{metric.name}",
        f"summary:test_mean_{metric.name}",
        f"summary:test_std_{metric.name}",
        f"summary:test_bagged_{metric.name}",
    ]
    if n_splits is not None:
        SPLIT_COLS = [
            f"summary:split_{split}:{kind}_{metric.name}"
            for kind, split in product(("val", "test"), range(n_splits))
        ]
    else:
        SPLIT_COLS = []
    return CORE + METRIC_COLS + SPLIT_COLS


def exp_name_to_result_dir(exp_name: EXP_NAME) -> Path:
    # Unfortunatly I bundled some experiments into the same directory, this just maps
    # them to where the results for a specific experiment are actually stored.
    match exp_name:
        case "reproduce":
            return Path("results-reproduce").resolve()
        case "time-analysis":
            return Path("results-time-analysis").resolve()
        case (
            "category3-nsplits-20"
            | "category3-nsplits-10"
            | "category3-nsplits-5"
            | "category3-nsplits-3"
            | "category3-nsplits-2-5"
        ):
            return Path("results-category3").resolve()
        case (
            "category4-nsplits-10"
            | "category4-nsplits-5"
            | "category4-nsplits-3"
            | "category4-nsplits-20"
            | "category4-nsplits-2-5"
        ):
            return Path("results-category4").resolve()
        case "category5-nsplits-10" | "category5-nsplits-20":
            return Path("results-category5").resolve()
        case "category6-nsplits-10" | "category6-nsplits-20":
            return Path("results-category6").resolve()
        case "category7-nsplits-20-unseeded":
            return Path("results-category7").resolve()
        case "category8-nsplits-20-unseeded":
            return Path("results-category8").resolve()
        case "debug":
            return Path("results-debug").resolve()
        case _:
            raise ValueError(f"Unknown experiment set: {exp_name}")


def experiment_set(name: EXP_NAME) -> list[E1]:
    # Defaults unless overwritten by an experiment set below
    time_seconds = 1 * 60 * 60
    folds = list(range(10))
    mem_per_cpu_gb = 5
    n_cpu = 4
    seeded_inner_cv = True
    metric = "roc_auc_ovr"
    pipeline = "mlp_classifier"
    experiment_fixed_seed = 42
    optimizers = ["random_search"]
    suite = TASKS["amlb_4hr_10cv_more_than_50_trials"]
    result_dir = exp_name_to_result_dir(name)

    # Must be set
    n_splits: list[int]
    methods: list[str]

    match name:
        case "reproduce":
            n_splits = [10]
            n_cpu = 4
            suite = [146818, 146820]
            time_seconds = 30
            folds = [0, 1]
            methods = [
                "disabled",
                "current_average_worse_than_best_worst_split",
                "current_average_worse_than_mean_best",
            ]
        case "time-analysis":
            # This suite runs the full automlbenchmark with our maximum cv splits, such
            # that we can see how many trials can be evaluated in a given time frame.
            # This will allow use to bin the datasets according to "size" where size
            # dictates how much time we run optimization for, effectively giving us
            # different categories of datasets with which to compare our results.
            n_splits = [10]
            folds = [0]  # We only need 1 outfold from openml to analise
            n_cpu = 1
            time_seconds = 4 * 60 * 60
            suite = TASKS["amlb_classification_full"]
            methods = ["disabled"]
        case "category3-nsplits-2-5":
            n_splits = [-5]  # This is a hack to run 2 repeats of 5 fold cv (sorry)
            methods = [
                "disabled",
                "current_average_worse_than_best_worst_split",
                "current_average_worse_than_mean_best",
            ]
        case "category7-nsplits-20-unseeded":
            n_splits = [20]
            seeded_inner_cv = False
            methods = [
                "disabled",
                "current_average_worse_than_best_worst_split",
                "current_average_worse_than_mean_best",
            ]
        case "category3-nsplits-20":
            n_splits = [20]
            methods = [
                "disabled",
                "current_average_worse_than_best_worst_split",
                "current_average_worse_than_mean_best",
            ]
        case "category3-nsplits-10":
            n_splits = [10]
            methods = [
                "disabled",
                "current_average_worse_than_best_worst_split",
                "current_average_worse_than_mean_best",
            ]
        case "category3-nsplits-5":
            n_splits = [5]
            methods = [
                "disabled",
                "current_average_worse_than_best_worst_split",
                "current_average_worse_than_mean_best",
            ]
        case "category3-nsplits-3":
            # This suite is running everything that had more
            # than 50 trials after 4 hours of 10 fold cross-validation
            n_splits = [3]
            methods = [
                "disabled",
                "current_average_worse_than_best_worst_split",
                "current_average_worse_than_mean_best",
            ]
        case "category4-nsplits-2-5":
            n_splits = [-5]  # This is a hack to run 2 repeats of 5 fold cv (sorry)
            pipeline = "rf_classifier"
            methods = [
                "disabled",
                "current_average_worse_than_best_worst_split",
                "current_average_worse_than_mean_best",
            ]
        case "category8-nsplits-20-unseeded":
            n_splits = [20]
            pipeline = "rf_classifier"
            seeded_inner_cv = False
            methods = [
                "disabled",
                "current_average_worse_than_best_worst_split",
                "current_average_worse_than_mean_best",
            ]
        case "category4-nsplits-20":
            n_splits = [20]
            pipeline = "rf_classifier"
            methods = [
                "disabled",
                "current_average_worse_than_best_worst_split",
                "current_average_worse_than_mean_best",
            ]
        case "category4-nsplits-10":
            # This suite is running everything that had more
            # than 50 trials after 4 hours of 10 fold cross-validation
            n_splits = [10]
            pipeline = "rf_classifier"
            methods = [
                "disabled",
                "current_average_worse_than_best_worst_split",
                "current_average_worse_than_mean_best",
            ]
        case "category4-nsplits-5":
            # This suite is running everything that had more
            # than 50 trials after 4 hours of 10 fold cross-validation
            n_splits = [5]
            pipeline = "rf_classifier"
            methods = [
                "disabled",
                "current_average_worse_than_best_worst_split",
                "current_average_worse_than_mean_best",
            ]
        case "category4-nsplits-3":
            # This suite is running everything that had more
            # than 50 trials after 4 hours of 10 fold cross-validation
            n_splits = [3]
            pipeline = "rf_classifier"
            methods = [
                "disabled",
                "current_average_worse_than_best_worst_split",
                "current_average_worse_than_mean_best",
            ]
        case "category5-nsplits-10":
            # We have to return specifically for this as we don't want a full product of
            # experiments. This is because the "smac_early_stop_with_fold_mean" makes no
            # sense to run when early stopping is disabled, it will behave the same as
            # "smac_early_stop_as_failed" which is default SMAC
            n_splits = [10]
            pipeline = "mlp_classifier"
            opt_method_set_1 = product(
                ["smac_early_stop_as_failed"],
                [
                    "disabled",
                    "current_average_worse_than_best_worst_split",
                    "current_average_worse_than_mean_best",
                ],
            )
            opt_method_set_2 = product(
                ["smac_early_stop_with_fold_mean"],
                [
                    # No "disabled" as it makes no sense to run with early stopping
                    "current_average_worse_than_best_worst_split",
                    "current_average_worse_than_mean_best",
                ],
            )
            opt_methods = chain(opt_method_set_1, opt_method_set_2)
            return [
                E1(
                    # Parameters defining experiments
                    task=task,
                    fold=fold,
                    n_splits=n_splits,
                    seeded_inner_cv=seeded_inner_cv,
                    # Constants for now
                    pipeline=pipeline,
                    n_cpus=n_cpu,
                    optimizer=optimizer,
                    memory_gb=mem_per_cpu_gb * n_cpu,
                    time_seconds=time_seconds,
                    experiment_seed=experiment_fixed_seed,
                    minimum_trials=1,  # Takes no effect...
                    metric=metric,
                    # Extra
                    wait=False,
                    root=result_dir,
                    cv_early_stop_strategy=method,
                    openml_cache_directory=(Path() / ".openml-cache").resolve(),
                )
                for task, fold, n_splits, (optimizer, method) in product(
                    suite,
                    folds,
                    n_splits,
                    opt_methods,
                )
            ]
        case "category5-nsplits-20":
            # We have to return specifically for this as we don't want a full product of
            # experiments. This is because the "smac_early_stop_with_fold_mean" makes no
            # sense to run when early stopping is disabled, it will behave the same as
            # "smac_early_stop_as_failed" which is default SMAC
            n_splits = [20]
            pipeline = "mlp_classifier"
            opt_method_set_1 = product(
                ["smac_early_stop_as_failed"],
                [
                    "disabled",
                    "current_average_worse_than_best_worst_split",
                    "current_average_worse_than_mean_best",
                    "robust_std_top_3",
                    "robust_std_top_5",
                ],
            )
            opt_method_set_2 = product(
                ["smac_early_stop_with_fold_mean"],
                [
                    # No "disabled" as it makes no sense to run with early stopping
                    "current_average_worse_than_best_worst_split",
                    "current_average_worse_than_mean_best",
                    "robust_std_top_3",
                    "robust_std_top_5",
                ],
            )
            opt_methods = chain(opt_method_set_1, opt_method_set_2)
            return [
                E1(
                    # Parameters defining experiments
                    task=task,
                    fold=fold,
                    seeded_inner_cv=seeded_inner_cv,
                    n_splits=n_splits,
                    # Constants for now
                    pipeline=pipeline,
                    n_cpus=n_cpu,
                    optimizer=optimizer,
                    memory_gb=mem_per_cpu_gb * n_cpu,
                    time_seconds=time_seconds,
                    experiment_seed=experiment_fixed_seed,
                    minimum_trials=1,  # Takes no effect...
                    metric=metric,
                    # Extra
                    wait=False,
                    root=result_dir,
                    cv_early_stop_strategy=method,
                    openml_cache_directory=(Path() / ".openml-cache").resolve(),
                )
                for task, fold, n_splits, (optimizer, method) in product(
                    suite,
                    folds,
                    n_splits,
                    opt_methods,
                )
            ]
        case "category6-nsplits-20":
            n_splits = [20]
            pipeline = "rf_classifier"
            opt_method_set_1 = product(
                ["smac_early_stop_as_failed"],
                [
                    "disabled",
                    "current_average_worse_than_best_worst_split",
                    "current_average_worse_than_mean_best",
                    # "robust_std_top_3",
                    # "robust_std_top_5",
                ],
            )
            opt_method_set_2 = product(
                ["smac_early_stop_with_fold_mean"],
                [
                    # No "disabled" as it makes no sense to run with early stopping
                    "current_average_worse_than_best_worst_split",
                    "current_average_worse_than_mean_best",
                    # "robust_std_top_3",
                    # "robust_std_top_5",
                ],
            )
            opt_methods = chain(opt_method_set_1, opt_method_set_2)
            return [
                E1(
                    # Parameters defining experiments
                    task=task,
                    fold=fold,
                    n_splits=n_splits,
                    seeded_inner_cv=seeded_inner_cv,
                    # Constants for now
                    pipeline=pipeline,
                    n_cpus=n_cpu,
                    optimizer=optimizer,
                    memory_gb=mem_per_cpu_gb * n_cpu,
                    time_seconds=time_seconds,
                    experiment_seed=experiment_fixed_seed,
                    minimum_trials=1,  # Takes no effect...
                    metric=metric,
                    # Extra
                    wait=False,
                    root=result_dir,
                    cv_early_stop_strategy=method,
                    openml_cache_directory=(Path() / ".openml-cache").resolve(),
                )
                for task, fold, n_splits, (optimizer, method) in product(
                    suite,
                    folds,
                    n_splits,
                    opt_methods,
                )
            ]
        case "category6-nsplits-10":
            # We have to return specifically for this as we don't want a full product of
            # experiments. This is because the "smac_early_stop_with_fold_mean" makes no
            # sense to run when early stopping is disabled, it will behave the same as
            # "smac_early_stop_as_failed" which is default SMAC
            n_splits = [10]
            pipeline = "rf_classifier"
            opt_method_set_1 = product(
                ["smac_early_stop_as_failed"],
                [
                    "disabled",
                    "current_average_worse_than_best_worst_split",
                    "current_average_worse_than_mean_best",
                    "robust_std_top_3",
                    "robust_std_top_5",
                ],
            )
            opt_method_set_2 = product(
                ["smac_early_stop_with_fold_mean"],
                [
                    # No "disabled" as it makes no sense to run with early stopping
                    "current_average_worse_than_best_worst_split",
                    "current_average_worse_than_mean_best",
                    "robust_std_top_3",
                    "robust_std_top_5",
                ],
            )
            opt_methods = chain(opt_method_set_1, opt_method_set_2)
            return [
                E1(
                    # Parameters defining experiments
                    task=task,
                    fold=fold,
                    n_splits=n_splits,
                    seeded_inner_cv=seeded_inner_cv,
                    # Constants for now
                    pipeline=pipeline,
                    n_cpus=n_cpu,
                    optimizer=optimizer,
                    memory_gb=mem_per_cpu_gb * n_cpu,
                    time_seconds=time_seconds,
                    experiment_seed=experiment_fixed_seed,
                    minimum_trials=1,  # Takes no effect...
                    metric=metric,
                    # Extra
                    wait=False,
                    root=result_dir,
                    cv_early_stop_strategy=method,
                    openml_cache_directory=(Path() / ".openml-cache").resolve(),
                )
                for task, fold, n_splits, (optimizer, method) in product(
                    suite,
                    folds,
                    n_splits,
                    opt_methods,
                )
            ]
        case "debug":
            n_splits = [10]
            folds = [0]
            time_seconds = 300
            methods = ["disabled"]
            n_cpu = 1
            suite = [359993]
        case _:
            raise ValueError(f"Unknown experiment set: {name}")

    return [
        E1(
            # Parameters defining experiments
            task=task,
            fold=fold,
            n_splits=n_splits,
            seeded_inner_cv=seeded_inner_cv,
            # Constants for now
            pipeline=pipeline,
            n_cpus=n_cpu,
            optimizer=optimizer,
            memory_gb=mem_per_cpu_gb * n_cpu,
            time_seconds=time_seconds,
            experiment_seed=experiment_fixed_seed,
            minimum_trials=1,  # Takes no effect...
            metric=metric,
            # Extra
            wait=False,
            root=result_dir,
            cv_early_stop_strategy=method,
            openml_cache_directory=(Path() / ".openml-cache").resolve(),
        )
        for task, fold, n_splits, optimizer, method in product(
            suite,
            folds,
            n_splits,
            optimizers,
            methods,
        )
    ]


def main():  # noqa: C901, PLR0915, PLR0912
    parser, cmds = E1.parser(["status", "run", "submit"])

    with cmds("run") as p:
        p.add_argument("--expname", choices=EXP_CHOICES, type=str, required=True)
        p.add_argument("--dry", action="store_true")
        p.add_argument(
            "--overwrite-by",
            choices=["failed", "running", "pending", "success", "submitted"],
            nargs="*",
            default=["pending"],
            type=str,
        )

    with cmds("submit") as p:
        p.add_argument("--expname", choices=EXP_CHOICES, type=str, required=True)
        p.add_argument("--dry", action="store_true")
        p.add_argument(
            "--overwrite-by",
            choices=["failed", "running", "pending", "success", "submitted"],
            nargs="*",
            default=["pending"],
            type=str,
        )
        p.add_argument("--overwrite-all", action="store_true")

    with cmds("status") as p:
        p.add_argument("--expname", choices=EXP_CHOICES, type=str, required=True)
        p.add_argument("--count", type=str, nargs="+", default=None)
        p.add_argument("--out", type=Path, default=None)

    with cmds("collect") as p:
        p.add_argument("--expname", choices=EXP_CHOICES, type=str, required=True)
        p.add_argument("--fail-early", action="store_true")
        p.add_argument("--ignore", nargs="+", type=str)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--no-config", action="store_true")

    with cmds("plot-stacked") as p:
        p.add_argument("--outpath", type=Path, default=Path("./plots"))
        p.add_argument("--prefix", type=str, required=True)
        p.add_argument("--metric", type=str, choices=METRICS.keys(), required=True)
        p.add_argument("--n-splits", nargs="+", type=int, required=True)
        p.add_argument("--time-limit", type=int, required=True)
        p.add_argument("--ncols-legend", type=int, default=None)
        p.add_argument("--ax-width", type=float, default=6)
        p.add_argument("--ax-height", type=float, default=5)
        p.add_argument("--with-test", action="store_true")
        p.add_argument(
            "--methods",
            nargs="+",
            type=str,
            choices=list(METHODS.keys()),
            default=list(METHODS),
        )
        p.add_argument(
            "--model",
            type=str,
            choices=["mlp", "rf"],
            nargs="+",
            required=True,
        )
        p.add_argument("--merge-opt-into-method", action="store_true")
        p.add_argument(
            "input",
            type=Path,
            nargs="+",
            help="The path to the `collect`'ed experiment name, one per `--n-split`",
        )

    with cmds("plot") as p:
        p.add_argument("--outpath", type=Path, default=Path("./plots"))
        p.add_argument("--prefix", type=str, required=True)
        p.add_argument("--ax-width", type=float, default=6)
        p.add_argument("--ax-height", type=float, default=5)
        p.add_argument("--metric", type=str, choices=METRICS.keys(), required=True)
        p.add_argument("--n-splits", type=int, required=True)
        p.add_argument("--time-limit", type=int, required=True)
        p.add_argument(
            "--methods",
            nargs="+",
            type=str,
            choices=list(METHODS.keys()),
            default=list(METHODS),
        )
        p.add_argument(
            "--model",
            type=str,
            choices=["mlp", "rf"],
            required=True,
        )
        p.add_argument("--merge-opt-into-method", action="store_true")
        p.add_argument(
            "--kind",
            type=str,
            choices=[
                "speedups",
                "ranks-aggregated",
                "incumbent-aggregated",
            ],
        )
        p.add_argument(
            "input",
            type=Path,
            help="The path to the `collect`'ed experiment name",
        )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "plot-stacked":
        import matplotlib.pyplot as plt

        args.outpath.mkdir(parents=True, exist_ok=True)

        _n_splits_titles = {
            -5: "2 Repeated 5-Fold",
            20: "2 Repeated 10-Fold",
        }
        nsplits_title = lambda n_splits: _n_splits_titles.get(
            n_splits,
            f"{n_splits}-Fold",
        )

        metric = METRICS[args.metric]
        time_limit = args.time_limit
        _dfs = {
            f"{model.upper()}, {nsplits_title(n_splits)} CV": pd.read_parquet(
                path,
                columns=cols_needed_for_plotting(metric, n_splits),
            )
            for path, (model, n_splits) in zip(
                args.input,
                product(args.model, args.n_splits),
                strict=True,
            )
        }

        first = next(iter(_dfs.values()))
        N_DATASETS = first["setting:task"].nunique()

        # Quick sanity check...
        for key, _df in _dfs.items():
            model = "rf_classifier" if "RF" in key else "mlp_classifier"
            assert _df["setting:task"].nunique() == N_DATASETS
            assert list(_df["setting:pipeline"].unique()) == [model]

        _dfs = {
            axis_title: _df[_df["setting:cv_early_stop_strategy"].isin(args.methods)]
            for axis_title, _df in _dfs.items()
        }

        title = f"Incumbent Traces, {N_DATASETS} Datasets"
        if args.merge_opt_into_method:
            for _, _df in _dfs.items():
                _df.loc[:, "setting:opt-method"] = _df["setting:optimizer"].str.cat(
                    _df["setting:cv_early_stop_strategy"],
                    sep="-",
                )
            _dfs = {
                k.replace("MLP", "Optimized MLP").replace("RF", "Optimized RF"): v
                for k, v in _dfs.items()
            }
            method_col = "setting:opt-method"
            baseline = "smac_early_stop_as_failed-disabled"  # Used for speedups
            title = f"Optimized {title}"
        else:
            method_col = "setting:cv_early_stop_strategy"
            baseline = "disabled"  # Used for speedups

        match len(_dfs):
            case 1 | 2 | 3:
                nrows = 1
            case 4 | 5 | 6:
                nrows = 2
            case _:
                raise ValueError("Not handled.")

        if args.with_test:
            incumbent_traces_aggregated_with_test(
                _dfs,
                y=f"metric:{metric}",
                test_y=f"summary:test_bagged_{metric.name}",
                method=method_col,
                fold="setting:fold",
                dataset="setting:task",
                x="reported_at",
                x_start="created_at",
                x_label="Time (s)",
                y_label="1 - Normalized ROC AUC [OVR]",
                title=title,
                figsize_per_ax=(args.ax_width, args.ax_height),
                x_bounds=(0, time_limit),
                ncols_legend=args.ncols_legend,
                minimize=metric.minimize,
                invert=True,
                log_y=True,
                markevery=0.1,
            )
        else:
            incumbent_traces_aggregated_no_test(
                _dfs,
                y=f"metric:{metric}",
                test_y=f"summary:test_bagged_{metric.name}",
                method=method_col,
                fold="setting:fold",
                dataset="setting:task",
                x="reported_at",
                x_start="created_at",
                x_label="Time (s)",
                y_label="1 - Normalized ROC AUC [OVR]",
                title=title,
                figsize_per_ax=(args.ax_width, args.ax_height),
                x_bounds=(0, time_limit),
                ncols_legend=args.ncols_legend,
                minimize=metric.minimize,
                nrows=nrows,
                invert=True,
                log_y=True,
                markevery=0.1,
            )
        for ext in ["pdf", "png"]:
            filepath = args.outpath / f"{args.prefix}-inc-stacked.{ext}"
            plt.savefig(filepath, bbox_inches="tight")

        return

    if args.command == "plot":
        import matplotlib.pyplot as plt

        args.outpath.mkdir(parents=True, exist_ok=True)

        metric = METRICS[args.metric]
        time_limit = args.time_limit
        cols = cols_needed_for_plotting(metric, args.n_splits)
        _df = pd.read_parquet(args.input, columns=cols)
        N_DATASETS = _df["setting:task"].nunique()

        _df = _df[_df["setting:cv_early_stop_strategy"].isin(args.methods)]

        if args.merge_opt_into_method:
            _df.loc[:, "setting:opt-method"] = _df["setting:optimizer"].str.cat(
                _df["setting:cv_early_stop_strategy"],
                sep="-",
            )
            method_col = "setting:opt-method"
            baseline = "smac_early_stop_as_failed-disabled"  # Used for speedups
            method_title = f"Optimized {args.model.upper()}"
        else:
            method_col = "setting:cv_early_stop_strategy"
            baseline = "disabled"  # Used for speedups
            method_title = args.model.upper()

        match args.kind:
            case "incumbent-aggregated":
                title = f"Normalized Cost of {method_title} with {args.n_splits} CV splits, {N_DATASETS} Datasets"  # noqa: E501
                incumbent_traces_aggregated(
                    _df,
                    y=f"metric:{metric}",
                    test_y=f"summary:test_bagged_{metric.name}",
                    method=method_col,
                    fold="setting:fold",
                    dataset="setting:task",
                    x="reported_at",
                    figsize_per_ax=(args.ax_width, args.ax_height),
                    x_start="created_at",
                    x_label="Time (s)",
                    y_label="1 - Normalized ROC AUC [OVR]",
                    title=title,
                    x_bounds=(0, time_limit),
                    minimize=metric.minimize,
                    invert=True,
                    log_y=True,
                    markevery=0.1,
                )
            case "ranks-aggregated":
                title = f"Rank Aggregation of {method_title} with {args.n_splits} CV splits, {N_DATASETS} Datasets"  # noqa: E501
                ranking_plots_aggregated(
                    _df,
                    y=f"metric:{metric}",
                    test_y=f"summary:test_bagged_{metric.name}",
                    method=method_col,
                    fold="setting:fold",
                    dataset="setting:task",
                    x="reported_at",
                    x_start="created_at",
                    x_label="Time (s)",
                    y_label="Rank",
                    title=title,
                    x_bounds=(0, time_limit),
                    minimize=metric.minimize,
                    markevery=0.1,
                )
            case "speedups":
                title = f"Speedups for {method_title} with {args.n_splits} CV splits"
                table_full, table_summary = speedup_plots(
                    _df,
                    y=f"metric:{metric}",
                    baseline=baseline,
                    test_y=f"summary:test_bagged_{metric.name}",
                    method=method_col,
                    fold="setting:fold",
                    dataset="setting:task",
                    x="reported_at",
                    x_start="created_at",
                    x_label="Time (s)",
                )
                table_full.to_latex(args.outpath / f"{args.prefix}-speedups-full.tex")
                table_summary.to_parquet(
                    args.outpath / f"{args.prefix}-speedups-aggregated.parquet",
                )
                table_summary.to_latex(
                    args.outpath / f"{args.prefix}-speedups-aggregated.tex",
                )
            case _:
                print(f"Unknown kind {args.kind}")

        for ext in ["pdf", "png"]:
            filepath = args.outpath / f"{args.prefix}-{args.kind}.{ext}"
            plt.savefig(filepath, bbox_inches="tight")
        return

    experiments = experiment_set(args.expname)
    result_dir = exp_name_to_result_dir(args.expname)
    log_dir = result_dir / "slurm-logs"
    script_dir = result_dir / "slurm-scripts"

    script_dir.mkdir(exist_ok=True, parents=True)
    result_dir.mkdir(exist_ok=True, parents=True)
    log_dir.mkdir(exist_ok=True, parents=True)

    match args.command:
        case "status":
            pd.set_option("display.max_colwidth", None)
            pd.set_option("display.max_rows", None)
            pd.set_option("display.max_columns", None)
            array = E1.as_array(experiments)
            status = array.status(
                exclude=["root", "openml_cache_directory"],
                count=args.count,
            )
            print(status)
            if args.out:
                shrink_dataframe(status).to_parquet(args.out)
        case "collect":
            array = E1.as_array(experiments)
            if args.fail_early:
                non_success_exps = [e for e in array if e.status() != "success"]
                if any(non_success_exps):
                    print(non_success_exps)

            first = experiments[0]
            columns_to_load = cols_needed_for_plotting(
                metric=METRICS[first.metric],
                n_splits=first.n_splits,
            )

            if not args.no_config:
                config_cols = [
                    f"config:{k}"
                    for k in PIPELINES[first.pipeline].search_space(
                        parser="configspace",
                    )
                ]
                if first.pipeline == "rf_classifier":
                    config_cols = [
                        c.replace(c.split(":")[1], "rf_classifier")
                        if c.startswith("config:Seq-")
                        else c
                        for c in config_cols
                    ]
                columns_to_load += config_cols

            print(f"Columns to load: {columns_to_load}")

            print(f"Collecting {len(array)} histories.")
            _df = pd.concat([exp.history(columns=columns_to_load) for exp in array])

            print(f"Collected {len(_df)} rows")
            print(f"Size: {round(_df.memory_usage().sum() / 1e6, 2)} MB")
            print("Shrinking...")
            _df = shrink_dataframe(_df)
            print(
                f"Size after shrinking: {round(_df.memory_usage().sum() / 1e6, 2)} MB",
            )
            print(f"Writing parquet to {args.out}")
            _df.to_parquet(args.out)

        case "submit" | "run":
            exps_by_status = defaultdict(list)
            for e in experiments:
                exps_by_status[e.status()].append(e)

            for status, exps in exps_by_status.items():
                print(f"{status}: {len(exps)}")

            to_submit = list(
                chain.from_iterable(exps_by_status[s] for s in args.overwrite_by),
            )
            if not any(to_submit):
                print(f"Nothing to run from {len(experiments)} experiments.")
                sys.exit(0)

            if args.dry:
                print(f"Would reset: {args.overwrite_by}")
                sys.exit(0)

            for exp in to_submit:
                exp.reset()

            match args.command:
                case "submit":
                    array = E1.as_array(to_submit)
                    first = array[0]
                    array.submit(
                        name=args.expname,
                        slurm_headers={
                            "partition": "ANON REPLACE ME",
                            "mem": f"{first.memory_gb}G",
                            "time": seconds_to_slurm_time(
                                int(5 * 60 + first.time_seconds * 1.5),
                            ),
                            "cpus-per-task": first.n_cpus,
                            "output": str(log_dir / "%j-%a.out"),
                            "error": str(log_dir / "%j-%a.err"),
                        },
                        python=None,  # Set explicitly if required.
                        script_dir=result_dir / "slurm-scripts",
                        sbatch=["sbatch"],
                        limit=1,
                    )
                case "run":
                    for exp in to_submit:
                        exp.run()
                case _:
                    raise RuntimeError("Something wasn't accounted for")
        case _:
            print("Unknown command")
            parser.print_help()


if __name__ == "__main__":
    main()
