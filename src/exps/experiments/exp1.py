from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import openml
import pandas as pd
import sklearn
from amltk.sklearn.evaluation import CVEvaluation

from exps.methods import METHODS
from exps.metrics import METRICS
from exps.optimizers import OPTIMIZERS
from exps.pipelines import PIPELINES
from exps.slurm import Arg, Slurmable

from amltk.store import PathBucket

if TYPE_CHECKING:
    from amltk.sklearn.evaluation import TaskTypeName


OPENML_CACHE_DIRECTORY = Path(openml.config.get_cache_directory())


@dataclass(kw_only=True)
class E1(Slurmable):
    EXP_NAME: ClassVar[str] = "experiment_cv_es"
    GROUPS_FOR_PATH: ClassVar[tuple[str, ...]] = ("task", "resources")

    experiment_seed: int = field(
        metadata=Arg(
            help="The seed for the experiment",
            group="z-extra",
        ),
    )
    # -- task
    task: int = field(
        metadata=Arg(
            help="OpenML task id",
            group="task",
        ),
    )
    fold: int = field(
        metadata=Arg(
            help="Fold number",
            group="task",
        ),
    )
    metric: str = field(
        metadata=Arg(
            help="Metric to optimize",
            group="task",
        ),
    )
    pipeline: str = field(
        metadata=Arg(
            help="Name of the pipeline to use for the experiment",
            choices=list(PIPELINES),
            group="task",
        ),
    )
    optimizer: str = field(
        metadata=Arg(
            help="Name of the optimizer to use for the experiment",
            choices=list(OPTIMIZERS),
            group="task",
        ),
    )
    n_splits: int = field(
        metadata=Arg(
            help="Number of splits to use for cross validation",
            group="task",
        ),
    )
    cv_early_stop_strategy: str = field(
        metadata=Arg(
            help="Early stop strategy",
            group="task",
            choices=list(METHODS),
        ),
    )

    # -- resources
    n_cpus: int = field(
        metadata=Arg(
            help="Number of cpus to use",
            group="resources",
        ),
    )
    memory_gb: int = field(
        metadata=Arg(
            help="Memory in GB to use",
            group="resources",
        ),
    )
    time_seconds: int = field(
        metadata=Arg(
            help="Time in seconds to use",
            group="resources",
        ),
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
        metadata=Arg(
            help="OpenML cache directory",
            group="paths",
        ),
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

        openml.config.set_root_cache_directory(self.openml_cache_directory)

        return get_fold(self.task, self.fold)

    def history(self) -> pd.DataFrame:
        _df = pd.read_parquet(self.unique_path / "history.parquet")
        return _df.assign(**{f"setting:{k}": v for k, v in self._values().items()})

    def python_path(self) -> Path:
        return Path(__file__)


def run_it(run: E1) -> None:
    print(run)
    sklearn.set_config(enable_metadata_routing=False, transform_output="pandas")
    tt, X, _, y, _ = run.get_data()
    pipeline = PIPELINES[run.pipeline]
    metric = METRICS[run.metric]
    cv_early_stopping_method = METHODS[run.cv_early_stop_strategy]

    evaluator = CVEvaluation(
        X,
        y,
        splitter="cv",
        n_splits=run.n_splits,
        random_state=run.experiment_seed,
        working_dir=run.unique_path / "evaluator",
        on_error="fail",
        task_hint=tt,
    )
    try:

        if cv_early_stopping_method is not None:
            plugins = [
                evaluator.cv_early_stopping_plugin(
                    strategy=cv_early_stopping_method(metric),
                ),
            ]
        else:
            plugins = []

        history = pipeline.optimize(
            working_dir=run.unique_path / "optimizer",
            target=evaluator.fn,
            metric=metric,
            timeout=run.time_seconds,
            n_workers=run.n_cpus,
            wait=run.wait,
            optimizer=OPTIMIZERS[run.optimizer],
            seed=run.experiment_seed,
            plugins=plugins,  # CV early stopping passed in here
            process_memory_limit=None,
            process_walltime_limit=None,
            threadpool_limit_ctl=False,
            max_trials=None,
            display=False,
            on_trial_exception="continue",  # Continue if a trial errors
            on_scheduler_exception="raise",  # End if the scheduler throws an exception
        )
        _df = history.df()
        _df.to_parquet(run.unique_path / "history.parquet")
        if len(_df) == 0:
            raise RuntimeError(f"No trial finished in time for {run}!")
        if (_df["status"] == "fail").all():
            raise RuntimeError(f"All configurations failed for {run}")
    except Exception as e:
        tb = traceback.format_exc()
        with (run.unique_path / "error.txt").open("w") as f:
            f.write(f"{tb}\n{e}")

        raise e
    finally:
        try:
            evaluator.bucket.rmdir()
            PathBucket(run.unique_path / "optimizer").rmdir()
        except Exception as e:  # noqa: BLE001
            print(e)


if __name__ == "__main__":
    parser, _ = E1.parser()
    E1.add_creation_arugments(parser)
    args = parser.parse_args()
    e1 = E1.parse(args)

    run_it(e1)
