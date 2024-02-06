from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from rich import print

from exps.experiments.exp1 import E1
from exps.plots import incumbent_traces

if __name__ == "__main__":
    tasks = [31]
    pipelines = ["mlp_classifier", "rf_classifier", "knn_classifier"]
    # pipelines = ["mlp_classifier"]
    folds = list(range(5))
    experiments = [
        E1(
            task=task,
            fold=fold,
            pipeline=pipeline,
            n_splits=5,
            n_cpus=2,
            memory_gb=4,
            time_seconds=30,
            minimum_trials=3,
            wait=False,
            root=Path("test-results"),
            cv_early_stop_strategy="disabled",
        )
        for task, fold, pipeline in product(tasks, folds, pipelines)
    ]

    parser, cmds = E1.parser(["status", "run", "submit"])

    with cmds("run") as p:
        p.add_argument("--overwrite", action="store_true")

    with cmds("submit") as p:
        p.add_argument("--overwrite", action="store_true")
        E1.add_submission_arguments(p)

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
                E1.as_array(pending)
                if len(pending) == 0:
                    print(f"Nothing to run from {len(experiments)} experiments.")
                    sys.exit(0)

            array.submit()
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
