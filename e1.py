from __future__ import annotations

import sys
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias

import matplotlib.pyplot as plt
import pandas as pd
from rich import print

from exps.experiments.exp1 import E1
from exps.metrics import METRICS
from exps.plots import (
    incumbent_traces_aggregated,
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
    "time-analysis",
    "category3-nsplits-10",
    "category3-nsplits-5",
    "category3-nsplits-3",
    "category4-nsplits-10",
    "category4-nsplits-5",
    "category4-nsplits-3",
]
EXP_CHOICES = [
    "debug",
    "time-analysis",
    "category3-nsplits-10",  # MLP pipeline
    "category3-nsplits-5",  # MLP pipeline
    "category3-nsplits-3",  # MLP pipeline
    "category4-nsplits-10",  # RF pipeline
    "category4-nsplits-5",  # RF pipeline
    "category4-nsplits-3",  # RF pipeline
]


def cols_needed_for_plotting(metric: Metric | str, n_splits: int) -> list[str]:
    if isinstance(metric, str):
        metric = METRICS[metric]

    CORE = [
        "created_at",
        "reported_at",
        "setting:fold",
        "setting:metric",
        "setting:optimizer",
        "setting:task",
        "setting:n_splits",
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
    SPLIT_COLS = [
        f"summary:split_{split}:{kind}_{metric.name}"
        for kind, split in product(("val", "test"), range(n_splits))
    ]
    return CORE + METRIC_COLS + SPLIT_COLS


def exp_name_to_result_dir(exp_name: EXP_NAME) -> Path:
    # Unfortunatly I bundled some experiments into the same directory, this just maps
    # them to where the results for a specific experiment are actually stored.
    match exp_name:
        case "time-analysis":
            return Path("results-time-analysis").resolve()
        case "category3-nsplits-10" | "category3-nsplits-5" | "category3-nsplits-3":
            return Path("results-category3").resolve()
        case "category4-nsplits-10" | "category4-nsplits-5" | "category4-nsplits-3":
            return Path("results-category4").resolve()
        case "debug":
            return Path("results-debug").resolve()
        case _:
            raise ValueError(f"Unknown experiment set: {exp_name}")


def experiment_set(name: EXP_NAME) -> list[E1]:  # noqa: PLR0915
    match name:
        case "time-analysis":
            # This suite runs the full automlbenchmark with our maximum cv splits, such
            # that we can see how many trials can be evaluated in a given time frame.
            # This will allow use to bin the datasets according to "size" where size
            # dictates how much time we run optimization for, effectively giving us
            # different categories of datasets with which to compare our results.
            n_splits = [10]
            folds = [0]  # We only need 1 outfold from openml to analise
            n_cpu = 1
            mem_per_cpu_gb = 5
            time_seconds = 4 * 60 * 60
            optimizers = ["random_search"]
            suite = TASKS["amlb_classification_full"]
            pipeline = "mlp_classifier"
            metric = "roc_auc_ovr"
            methods = ["disabled"]
            experiment_fixed_seed = 42
        case "category3-nsplits-10":
            # This suite is running everything that had more
            # than 50 trials after 4 hours of 10 fold cross-validation
            n_splits = [10]
            folds = list(range(10))
            n_cpu = 4
            mem_per_cpu_gb = 5
            time_seconds = 1 * 60 * 60
            optimizers = ["random_search"]  # TODO: SMAC
            methods = [
                "disabled",
                "current_average_worse_than_best_worst_split",
                "current_average_worse_than_mean_best",
                "robust_std_top_3",
                "robust_std_top_5",
            ]
            suite = TASKS["amlb_4hr_10cv_more_than_50_trials"]
            pipeline = "mlp_classifier"
            metric = "roc_auc_ovr"
            experiment_fixed_seed = 42
        case "category3-nsplits-5":
            # This suite is running everything that had more
            # than 50 trials after 4 hours of 10 fold cross-validation
            n_splits = [5]
            folds = list(range(10))
            n_cpu = 4
            mem_per_cpu_gb = 5
            time_seconds = 1 * 60 * 60
            optimizers = ["random_search"]  # TODO: SMAC
            methods = [
                "disabled",
                "current_average_worse_than_best_worst_split",
                "current_average_worse_than_mean_best",
                "robust_std_top_3",
                "robust_std_top_5",
            ]
            suite = TASKS["amlb_4hr_10cv_more_than_50_trials"]
            pipeline = "mlp_classifier"
            metric = "roc_auc_ovr"
            experiment_fixed_seed = 42
        case "category3-nsplits-3":
            # This suite is running everything that had more
            # than 50 trials after 4 hours of 10 fold cross-validation
            n_splits = [3]
            folds = list(range(10))
            n_cpu = 4
            mem_per_cpu_gb = 5
            time_seconds = 1 * 60 * 60
            optimizers = ["random_search"]  # TODO: SMAC
            methods = [
                "disabled",
                "current_average_worse_than_best_worst_split",
                "current_average_worse_than_mean_best",
                "robust_std_top_3",
                "robust_std_top_5",
            ]
            suite = TASKS["amlb_4hr_10cv_more_than_50_trials"]
            pipeline = "mlp_classifier"
            metric = "roc_auc_ovr"
            experiment_fixed_seed = 42
        case "category4-nsplits-10":
            # This suite is running everything that had more
            # than 50 trials after 4 hours of 10 fold cross-validation
            n_splits = [10]  # TODO: Add in 5 and 3 fold
            folds = list(range(10))
            n_cpu = 4
            mem_per_cpu_gb = 5
            time_seconds = 1 * 60 * 60
            optimizers = ["random_search"]  # TODO: SMAC
            methods = [
                "disabled",
                "current_average_worse_than_best_worst_split",
                "current_average_worse_than_mean_best",
                "robust_std_top_3",
                "robust_std_top_5",
            ]
            suite = TASKS["amlb_4hr_10cv_more_than_50_trials"]
            pipeline = "rf_classifier"
            metric = "roc_auc_ovr"
            experiment_fixed_seed = 42
        case "debug":
            n_splits = [3]
            folds = [0]
            n_cpu = 4
            mem_per_cpu_gb = 5
            time_seconds = 30
            optimizers = ["random_search"]
            suite = TASKS["debug"]
            pipeline = "rf_classifier"
            metric = "roc_auc_ovr"
            methods = ["disabled"]
            experiment_fixed_seed = 42
        case _:
            raise ValueError(f"Unknown experiment set: {name}")

    result_dir = exp_name_to_result_dir(name)
    return [
        E1(
            # Parameters defining experiments
            task=task,
            fold=fold,
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
        p.add_argument("--overwrite-all", action="store_true")
        p.add_argument("--overwrite-failed-only", action="store_true")

    with cmds("submit") as p:
        p.add_argument("--expname", choices=EXP_CHOICES, type=str, required=True)
        p.add_argument("--overwrite-all", action="store_true")
        p.add_argument("--overwrite-failed-only", action="store_true")

    with cmds("status") as p:
        p.add_argument("--expname", choices=EXP_CHOICES, type=str, required=True)
        p.add_argument("--count", type=str, nargs="+", default=None)
        p.add_argument("--out", type=Path, default=None)

    with cmds("collect") as p:
        p.add_argument("--expname", choices=EXP_CHOICES, type=str, required=True)
        p.add_argument("--fail-early", action="store_true")
        p.add_argument("--ignore", nargs="+", type=str)
        p.add_argument("--out", type=Path, required=True)

    with cmds("plot") as p:
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--metric", type=str, choices=METRICS.keys(), required=True)
        p.add_argument("--n-splits", type=int, required=True)
        p.add_argument("--time-limit", type=int, required=True)
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

    with cmds("plot-normalized-baseline") as p:
        p.add_argument("--out", type=Path, required=True)
        p.add_argument(
            "input",
            type=Path,
            help="The path to the `collect`'ed experiment name",
        )

    args = parser.parse_args()
    if args.command == "plot":
        metric = METRICS[args.metric]
        n_splits = args.n_splits
        time_limit = args.time_limit
        cols = cols_needed_for_plotting(metric, n_splits)
        _df = pd.read_parquet(args.input, columns=cols)
        _df = _df[_df["setting:n_splits"] == n_splits]
        N_DATASETS = _df["setting:task"].nunique()
        match args.kind:
            case "incumbent-aggregated":
                title = f"Normalized Cost Over {N_DATASETS} Datasets"
                incumbent_traces_aggregated(
                    _df,
                    y=f"metric:{metric}",
                    test_y=f"summary:test_bagged_{metric.name}",
                    method="setting:cv_early_stop_strategy",
                    fold="setting:fold",
                    dataset="setting:task",
                    x="reported_at",
                    x_start="created_at",
                    x_label="Time (s)",
                    y_label="1 - (normalized) ROC AUC [OVR]",
                    title=title,
                    x_bounds=(0, time_limit),
                    # y_bounds=(1e-1, 1),
                    minimize=metric.minimize,
                    invert=True,
                    log_y=True,
                    markevery=0.1,
                )
            case "ranks-aggregated":
                title = f"Rank Aggregation Over {N_DATASETS} Datasets"
                ranking_plots_aggregated(
                    _df,
                    y=f"metric:{metric}",
                    test_y=f"summary:test_bagged_{metric.name}",
                    method="setting:cv_early_stop_strategy",
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
                speedup_plots(
                    _df,
                    y=f"metric:{metric}",
                    baseline="disabled",
                    test_y=f"summary:test_bagged_{metric.name}",
                    method="setting:cv_early_stop_strategy",
                    fold="setting:fold",
                    dataset="setting:task",
                    x="reported_at",
                    x_start="created_at",
                    x_label="Time (s)",
                    title="TODO",
                )
            case _:
                print(f"Unknown kind {args.kind}")
        plt.savefig(args.out, bbox_inches="tight")
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
        case "run":
            if args.overwrite_all:
                exps = experiments
                for exp in exps:
                    exp.reset()
                array = E1.as_array(experiments)
            elif args.overwrite_failed_only:
                exps = [e for e in experiments if e.status() == "failed"]
                if not any(exps):
                    print(f"No failed experiments from {len(experiments)} to reset.")
                    sys.exit(0)

                for exp in exps:
                    if exp.status() == "failed":
                        exp.reset()
                array = E1.as_array(exps)
            else:
                pending = [exp for exp in experiments if exp.status() == "pending"]
                if len(pending) == 0:
                    print(f"Nothing to run from {len(experiments)} experiments.")
                    sys.exit(0)

                array = E1.as_array(pending)

            for exp in array:
                exp.run()
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

        case "submit":
            if args.overwrite_all:
                exps = experiments
                for exp in exps:
                    exp.reset()
                array = E1.as_array(experiments)
            elif args.overwrite_failed_only:
                exps = [e for e in experiments if e.status() == "failed"]
                if not any(exps):
                    print(f"No failed experiments from {len(experiments)} to reset.")
                    sys.exit(0)

                for exp in exps:
                    if exp.status() == "failed":
                        exp.reset()
                array = E1.as_array(exps)
            else:
                pending = [exp for exp in experiments if exp.status() == "pending"]
                array = E1.as_array(pending)
                if len(pending) == 0:
                    print(f"Nothing to run from {len(experiments)} experiments.")
                    sys.exit(0)

            first = experiments[0]
            array.submit(
                name=args.expname,
                slurm_headers={
                    "partition": "bosch_cpu-cascadelake",
                    "mem": f"{first.memory_gb}G",
                    "time": seconds_to_slurm_time(
                        int(5 * 60 + first.time_seconds * 1.5),
                    ),
                    "cpus-per-task": first.n_cpus,
                    "output": str(log_dir / "%j-%a.out"),
                    "error": str(log_dir / "%j-%a.err"),
                },
                python="/work/dlclarge2/bergmane-pipeline-exps/exps/.eddie-venv/bin/python",
                script_dir=result_dir / "slurm-scripts",
                sbatch=["sbatch", "--bosch"],
                limit=1,
            )
        case _:
            print("Unknown command")
            parser.print_help()


if __name__ == "__main__":
    main()
