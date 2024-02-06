from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from rich import print

from exps.experiments.exp1 import E1
from exps.plots import incumbent_traces
from exps.slurm import seconds_to_slurm_time

EXP_NAME = "pipelines-exp1"
amlb_classification = [
    2073,
    3945,
    7593,
    10090,
    146818,
    146820,
    167120,
    168350,
    168757,
    168784,
    168868,
    168909,
    168910,
    168911,
    189354,
    189355,
    189356,
    189922,
    190137,
    190146,
    190392,
    190410,
    190411,
    190412,
    211979,
    211986,
    359953,
    359954,
    359955,
    359956,
    359957,
    359958,
    359959,
    359960,
    359961,
    359962,
    359963,
    359964,
    359965,
    359966,
    359967,
    359968,
    359969,
    359970,
    359971,
    359972,
    359973,
    359974,
    359975,
    359976,
    359977,
    359979,
    359980,
    359981,
    359982,
    359983,
    359984,
    359985,
    359986,
    359987,
    359988,
    359989,
    359990,
    359991,
    359992,
    359993,
    359994,
    360112,
    360113,
    360114,
    360975,
]

if __name__ == "__main__":
    tasks = amlb_classification
    pipelines = ["rf_classifier", "mlp_classifier", "knn_classifier"]
    result_dir = Path(f"results-{EXP_NAME}").resolve()
    log_dir = result_dir / "slurm-logs"
    script_dir = result_dir / "slurm-scripts"

    script_dir.mkdir(exist_ok=True, parents=True)
    result_dir.mkdir(exist_ok=True, parents=True)
    log_dir.mkdir(exist_ok=True, parents=True)

    folds = list(range(10))
    MEM_PER_CPU_GB = 4
    TIME_SECONDS = 10 * 60
    N_CPU = 2

    tasks = tasks[:3]
    folds = folds[:3]

    experiments = [
        E1(
            task=task,
            fold=fold,
            pipeline=pipeline,
            n_splits=3,
            n_cpus=N_CPU,
            memory_gb=MEM_PER_CPU_GB * N_CPU,
            time_seconds=TIME_SECONDS,
            minimum_trials=5,
            wait=False,
            root=result_dir,
            cv_early_stop_strategy="disabled",
            openml_cache_directory=(Path() / ".openml-cache").resolve(),
        )
        for task, fold, pipeline in product(tasks, folds, pipelines)
    ]

    parser, cmds = E1.parser(["status", "run", "submit"])
    experiments = experiments[:1]

    with cmds("run") as p:
        p.add_argument("--overwrite", action="store_true")

    with cmds("submit") as p:
        p.add_argument("--overwrite", action="store_true")

    with cmds("status") as p:
        p.add_argument("--count", type=str, nargs="+", default=None)

    with cmds("plot") as p:
        pass

    args = parser.parse_args()

    match args.command:
        case "status":
            array = E1.as_array(experiments)
            print(array.status(exclude=["openml_cache_directory"], count=args.count))
        case "run":
            if args.overwrite:
                array = E1.as_array(experiments)
                for exp in array:
                    exp.reset()
            else:
                pending = [exp for exp in experiments if exp.status() == "pending"]
                if len(pending) == 0:
                    print(f"Nothing to run from {len(experiments)} experiments.")
                    sys.exit(0)

                array = E1.as_array(pending)

            for exp in array:
                exp.run()

        case "submit":
            if args.overwrite:
                array = E1.as_array(experiments)
                for exp in array:
                    exp.reset()
            else:
                pending = [exp for exp in experiments if exp.status() == "pending"]
                array = E1.as_array(pending)
                if len(pending) == 0:
                    print(f"Nothing to run from {len(experiments)} experiments.")
                    sys.exit(0)

            array.submit(
                name=EXP_NAME,
                slurm_headers={
                    "partition": "gki_cpu-cascadelake",
                    "mem-per-cpu": f"{MEM_PER_CPU_GB}G",
                    "time": seconds_to_slurm_time(int(5 * 60 + TIME_SECONDS * 1.2)),
                    "ntasks": 1,
                    "cpus-per-task": N_CPU,
                    "output": str(log_dir / "%j-%a.out"),
                    "error": str(log_dir / "%j-%a.err"),
                },
                python="/work/dlclarge2/bergmane-pipeline-exps/exps/.eddie-venv/bin/python",
                script_dir=result_dir / "slurm-scripts",
                #sbatch=["sbatch",  "--bosch"],
                limit=1,
            )
        case "plot":
            array = E1.as_array(experiments)
            if any(exp.status() != "success" for exp in array):
                print("Not all experiments are successful")
                sys.exit(1)
            _df = pd.concat([exp.history() for exp in array])
            incumbent_traces(
                _df,
                y="metric:accuracy [0.0, 1.0] (maximize)",
                hue="setting:pipeline",
                std="setting:fold",
                subplot="setting:task",
                x="reported_at",
                min_x="created_at",
                x_label="Time (s)",
                y_label="Accuracy",
                markevery=0.1,
            )
            plt.show()
        case _:
            print("Unknown command")
            parser.print_help()
