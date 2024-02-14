from __future__ import annotations

import sys
from itertools import product
from pathlib import Path
from typing import Literal, TypeAlias

import matplotlib.pyplot as plt
import pandas as pd
from rich import print

from exps.experiments.exp1 import E1
from exps.methods import METHODS
from exps.plots import baseline_normalized_over_time, incumbent_traces, rank_plots
from exps.slurm import seconds_to_slurm_time
from exps.tasks import TASKS


def path_col_to_str(_df: pd.DataFrame) -> pd.DataFrame:
    path_dtypes = _df.select_dtypes(Path).columns
    return _df.astype({k: pd.StringDtype() for k in path_dtypes})


EXP_NAME: TypeAlias = Literal["debug", "small", "full"]
EXP_CHOICES = ["debug", "small", "full"]


def experiment_set(name: EXP_NAME) -> list[E1]:
    match name:
        case "debug":
            n_splits = [5]
            folds = [0]
            n_cpu = 1
            mem_per_cpu_gb = 4
            time_seconds = 30
            minimum_trials = 1
            optimizers = ["random_search"]
            suite = TASKS["debug"]
            pipeline = "mlp_classifier"
            metric = "roc_auc_ovr"
            methods = [
                "disabled",
                "current_average_worse_than_mean_best",
                "current_average_worse_than_best_worst_split",
            ]
            experiment_fixed_seed = 42
        case "small":
            n_splits = [5]
            folds = list(range(10))
            n_cpu = 1
            mem_per_cpu_gb = 4
            time_seconds = 10 * 60
            minimum_trials = 1
            optimizers = ["random_search"]
            suite = TASKS["amlb_classification_less_than_50k"]
            pipeline = "mlp_classifier"
            metric = "roc_auc_ovr"
            methods = [
                "disabled",
                "current_average_worse_than_mean_best",
                "current_average_worse_than_best_worst_split",
            ]
            experiment_fixed_seed = 42
        case "full":
            pipeline = "mlp_classifier"
            folds = list(range(10))
            n_splits = [10]  # Set to 10, see if we can even evaluate a full model
            suite = TASKS["amlb_classification_full"]
            optimizers = ["random_search"]  # , "smac"]
            methods = list(METHODS.keys())
            mem_per_cpu_gb = 4
            time_seconds = 10 * 60
            n_cpu = 1
            minimum_trials = 1
            metric = "roc_auc_ovr"
            experiment_fixed_seed = 42
        case _:
            raise ValueError(f"Unknown experiment set: {name}")

    result_dir = Path(f"results-{name}").resolve()

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
            minimum_trials=minimum_trials,
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


def main():
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
        p.add_argument(
            "--kind",
            type=str,
            choices=["incumbent", "baseline-normalized", "ranks"],
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
        _df = pd.read_parquet(args.input)
        match args.kind:
            case "incumbent":
                incumbent_traces(
                    _df,
                    y="metric:roc_auc_ovr [0.0, 1.0] (maximize)",
                    hue="setting:cv_early_stop_strategy",
                    std="setting:fold",
                    subplot="setting:task",
                    x="reported_at",
                    x_start="created_at",
                    x_label="Time (s)",
                    y_label="ROC AUC (OVR)",
                    x_bounds=(0, 10 * 60),
                    minimize=False,
                    log_y=False,
                    markevery=0.1,
                )
            case "ranks":
                rank_plots(
                    _df,
                    y="metric:roc_auc_ovr [0.0, 1.0] (maximize)",
                    hue="setting:cv_early_stop_strategy",
                    std="setting:fold",
                    subplot="setting:task",
                    x="reported_at",
                    x_start="created_at",
                    x_label="Time (s)",
                    y_label="Mean Rank",
                    x_bounds=(0, 10 * 60),
                    minimize=False,
                    log_y=False,
                    markevery=0.1,
                )
            case "baseline-normalized":
                baseline_normalized_over_time(
                    _df,
                    y="metric:roc_auc_ovr [0.0, 1.0] (maximize)",
                    baseline="disabled",
                    hue="setting:cv_early_stop_strategy",
                    std="setting:fold",
                    subplot="setting:task",
                    x="reported_at",
                    x_start="created_at",
                    x_label="Time (s)",
                    y_label="ROC AUC (shifted)",
                    x_bounds=(0, 10 * 60),
                    minimize=False,
                    metric_bounds=(0, 1),
                    log_y=False,
                    markevery=0.1,
                )
            case _:
                print(f"Unknown kind {args.kind}")
        plt.savefig(args.out)
        return

    experiments = experiment_set(args.expname)
    result_dir = Path(f"results-{args.expname}").resolve()
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
                status = path_col_to_str(status)
                status.convert_dtypes().to_parquet(args.out)
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

            _df = pd.concat([exp.history() for exp in array])
            if args.ignore:
                _df = _df.drop(columns=args.ignore)

            _df = path_col_to_str(_df)
            _df.convert_dtypes().to_parquet(args.out)
            print(f"Concatenated {len(array)} histories to {args.out}")

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
                    "mem-per-cpu": f"{first.memory_gb}G",
                    "time": seconds_to_slurm_time(
                        int(5 * 60 + first.time_seconds * 1.2),
                    ),
                    "ntasks": 1,
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
