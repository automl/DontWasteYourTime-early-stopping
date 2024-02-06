from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import openml
import pandas as pd
import sklearn
from amltk import Metric
from amltk.sklearn.evaluation import CVEvaluation

from exps.pipelines import knn_classifier, mlp_classifier, rf_classifier
from exps.slurm import Arg, Slurmable

if TYPE_CHECKING:
    from amltk.sklearn.evaluation import TaskTypeName


PIPELINES = {
    "rf_classifier": rf_classifier,
    "mlp_classifier": mlp_classifier,
    "knn_classifier": knn_classifier,
}
OPENML_CACHE_DIRECTORY = Path(openml.config.get_cache_directory())
EXPERIMENT_FIXED_SEED = 42


@dataclass(kw_only=True)
class E1(Slurmable):
    EXP_NAME: ClassVar[str] = "experiment_cv_es"
    GROUPS_FOR_PATH: ClassVar[tuple[str, ...]] = ("task", "resources")

    EXPERIMENT_SEED: int = field(
        default=EXPERIMENT_FIXED_SEED,
        metadata=Arg(help="The seed for the experiment", group="z-extra"),
    )
    # -- task
    task: int = field(metadata=Arg(help="OpenML task id", group="task"))
    fold: int = field(metadata=Arg(help="Fold number", group="task"))
    pipeline: str = field(
        metadata=Arg(
            help="Name of the pipeline to use for the experiment",
            choices=PIPELINES.keys(),
            group="task",
        ),
    )
    n_splits: int = field(
        metadata=Arg(help="Number of splits to use for cross validation", group="task"),
    )
    cv_early_stop_strategy: str = field(
        metadata=Arg(help="Early stop strategy", group="task", choices=["disabled"]),
    )

    # -- resources
    n_cpus: int = field(metadata=Arg(help="Number of cpus to use", group="resources"))
    memory_gb: int = field(metadata=Arg(help="Memory in GB to use", group="resources"))
    time_seconds: int = field(
        metadata=Arg(help="Time in seconds to use", group="resources"),
    )
    minimum_trials: int = field(
        metadata=Arg(
            help="Minimum trials squeezed into --time_seconds",
            group="resources",
        ),
    )
    wait: bool = field(
        metadata=Arg(
            help="Wait for workers at --time_seconds elapsed",
            group="resources",
        ),
    )
    openml_cache_directory: Path = field(
        default=OPENML_CACHE_DIRECTORY,
        metadata=Arg(help="OpenML cache directory", group="paths"),
    )

    def get_data(
        self,
    ) -> tuple[
        TaskTypeName,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame | pd.Series,
        pd.DataFrame | pd.Series,
    ]:
        from exps.data import get_fold

        return get_fold(self.task, self.fold)

    def history(self) -> pd.DataFrame:
        _df = pd.read_parquet(self.unique_path / "history.parquet")
        return _df.assign(**{f"setting:{k}": v for k, v in self._values().items()})

    def python_path(self) -> Path:
        return Path(__file__)


def run_it(run: E1) -> None:
    print(run)
    sklearn.set_config(enable_metadata_routing=False, transform_output="pandas")
    try:
        tt, X, _, y, _ = run.get_data()
        pipeline = PIPELINES[run.pipeline]

        if run.cv_early_stop_strategy.lower() not in ("disabled",):
            raise NotImplementedError(
                "Only 'none' is supported for cv_early_stop_strategy",
            )
        history = pipeline.optimize(
            target=CVEvaluation(
                X,
                y,
                splitter="cv",
                n_splits=run.n_splits,
                random_state=run.EXPERIMENT_SEED,
                working_dir=run.unique_path / "evaluator",
                on_error="fail",
                task_hint=tt,
            ),
            metric=Metric("accuracy", minimize=False, bounds=(0, 1)),
            timeout=run.time_seconds,
            max_trials=None,
            n_workers=run.n_cpus,
            working_dir=run.unique_path / "optimizer",
            on_trial_exception="continue",
            display=False,
            wait=run.wait,
            seed=run.EXPERIMENT_SEED,
            threadpool_limit_ctl=True,
            process_memory_limit=(run.memory_gb / run.n_cpus, "GB"),  # type: ignore
            process_walltime_limit=(run.time_seconds // run.minimum_trials, "m"),
        )
        history.df().to_parquet(run.unique_path / "history.parquet")
    except Exception as e:
        tb = traceback.format_exc()
        with (run.unique_path / "error.txt").open("w") as f:
            f.write(f"{tb}\n{e}")

        raise e


if __name__ == "__main__":
    parser, _ = E1.parser()
    E1.add_creation_arugments(parser)
    args = parser.parse_args()
    e1 = E1.parse(args)

    run_it(e1)
