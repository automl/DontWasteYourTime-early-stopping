from __future__ import annotations

import traceback
from asyncio import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import openml
import pandas as pd
import pyarrow.parquet as pq
import sklearn
from amltk.optimization import Trial
from amltk.sklearn.evaluation import CVEvaluation
from amltk.sklearn.voting import voting_with_preffited_estimators
from amltk.store import PathBucket
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.metrics._scorer import _MultimetricScorer
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from exps.methods import METHODS
from exps.metrics import METRICS
from exps.optimizers import OPTIMIZERS
from exps.pipelines import PIPELINES
from exps.slurm import Arg, Slurmable
from exps.util import shrink_dataframe

if TYPE_CHECKING:
    from amltk.pipeline import Node
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
    seeded_inner_cv: bool = field(
        default=True,
        metadata=Arg(
            help="Seed the inner CV",
            group="task-extra",
        ),
    )
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

        return get_fold(
            self.task,
            self.fold,
            seed=self.experiment_seed + self.fold,
            n_splits=self.n_splits,
        )

    def history(self, columns: list[str] | None = None) -> pd.DataFrame:
        cs = (
            None
            if columns is None
            else [c for c in columns if not c.startswith("setting:")]
        )
        history_path = self.unique_path / "history.parquet"

        # Unfortunatly, if we are loading config columns, it could be the case
        # that some column is not present in the parquet file. To alleviate this,
        # we directly read the schema to got the column that are available.
        if cs is not None:
            parquet_file = pq.ParquetFile(history_path)
            present_columns = parquet_file.schema.names
            cs = [c for c in cs if c in present_columns or c.startswith("config:Seq-")]

        _df = pd.read_parquet(history_path, columns=cs)
        _df = _df.assign(**{f"setting:{k}": v for k, v in self._values().items()})

        if self.pipeline == "rf_classifier":
            # NOTE: we accidentally didn't name the rf pipeline which means we
            # have randuids isntead of `"rf_classifier". We rename these columns
            # accordingly
            seq_columns = {
                c: c.split(":")[1] for c in _df.columns if c.startswith("config:Seq-")
            }
            _df = _df.rename(
                columns={
                    c: c.replace(v, "rf_classifier") for c, v in seq_columns.items()
                },
            )

        return shrink_dataframe(_df)

    def python_path(self) -> Path:
        return Path(__file__)


def test_score_bagged_ensemble(
    report: Trial.Report,
    pipeline: Node,  # noqa: ARG001
    info: CVEvaluation.CompleteEvalInfo,
) -> Trial.Report:
    # Only calculate bagged test scores if the CV evaluation was successful
    if report.status is not Trial.Status.SUCCESS:
        return report

    assert info.models is not None
    # Just a proxy heuristic, this will break if we expand the scope of experiments
    assert len(info.scorers) == 1
    scorer_name = next(iter(info.scorers))
    if scorer_name in ("accuracy", "roc_auc_ovr"):
        voter = VotingClassifier
        # TODO: Probably only makes sense to do voting:soft if we're using
        # MLP which tends to be a little better calibrated than something like
        # a shallow RF
        kwargs = {"voting": "soft", "n_jobs": 1}
    elif scorer_name in ("r2", "neg_mean_squared_error", "root_mean_squared_error"):
        voter = VotingRegressor
        kwargs = {"n_jobs": 1}
    else:
        raise ValueError("Assumption broken")

    bagged_model = voting_with_preffited_estimators(
        info.models,
        voter=voter,
        **kwargs,  # type: ignore
    )
    multimetric = _MultimetricScorer(scorers=info.scorers, raise_exc=True)
    scores = multimetric(bagged_model, info.X_test, info.y_test)
    report.summary.update({f"test_bagged_{k}": v for k, v in scores.items()})
    return report


def run_it(run: E1) -> None:
    print(run)
    sklearn.set_config(enable_metadata_routing=False, transform_output="pandas")
    task_type, X, X_test, y, y_test = run.get_data()
    pipeline = PIPELINES[run.pipeline]
    metric = METRICS[run.metric]
    cv_early_stopping_method = METHODS[run.cv_early_stop_strategy]

    inner_cv_seed = run.experiment_seed + run.fold if run.seeded_inner_cv else None

    # This is a bit of hack but we don't do 20 CV, we do it
    # as 2 cross 10
    if run.n_splits == 20:  # noqa: PLR2004
        splitter = RepeatedStratifiedKFold(
            n_splits=2,
            n_repeats=10,
            random_state=inner_cv_seed,
        )
    elif run.n_splits == -5:  # noqa: PLR2004
        # This is a hack to do 2 CV 5 times
        splitter = RepeatedStratifiedKFold(
            n_splits=2,
            n_repeats=5,
            random_state=inner_cv_seed,
        )
    else:
        splitter = StratifiedKFold(
            n_splits=run.n_splits,
            random_state=inner_cv_seed,
            shuffle=True,
        )

    evaluator = CVEvaluation(
        X,
        y,
        splitter=splitter,  # type: ignore
        n_splits=run.n_splits,
        random_state=run.experiment_seed + run.fold,
        working_dir=run.unique_path / "evaluator",
        on_error="fail",
        X_test=X_test,
        y_test=y_test,
        task_hint=task_type,
        # We don't need train scores.
        train_score=False,
        # We don't need to store models, they will be handed to post processing
        # at which point we can discard them to save disk space
        store_models=False,
        # Calculate the test score for all fold validation models bagged together
        post_processing=test_score_bagged_ensemble,
        post_processing_requires_models=True,
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

        scheduler, task, history = pipeline.register_optimization_loop(
            working_dir=run.unique_path / "optimizer",
            target=evaluator.fn,
            metric=metric,
            n_workers=run.n_cpus,
            optimizer=OPTIMIZERS[run.optimizer],
            seed=run.experiment_seed + run.fold,
            plugins=plugins,  # CV early stopping passed in here
            process_memory_limit=None,
            process_walltime_limit=None,
            threadpool_limit_ctl=True,
            max_trials=None,
            on_trial_exception="continue",  # Continue if a trial errors
        )

        # Make sure to cleanup after trials if we do not have anything stored
        # (most often the case) such that we do not blow up the filesystem with
        # empty folders.
        @task.on_result
        def _cleanup_after_trial(_: Future, report: Trial.Report) -> None:
            report.bucket.rmdir()

        scheduler.run(
            timeout=run.time_seconds,
            wait=run.wait,
            on_exception="raise",
            display=False,
        )
        _df = history.df()
        _df.to_parquet(run.unique_path / "history.parquet")
        if len(_df) == 0:
            raise RuntimeError(f"No trial finished in time for {run}!")
        if (_df["status"] == "fail").all():
            exceptions = _df["exception"]
            raise RuntimeError(
                f"All configurations failed for {run}\n"
                + "\n".join(map(str, exceptions)),
            )
    except Exception as e:
        tb = traceback.format_exc()
        with (run.unique_path / "error.txt").open("w") as f:
            f.write(f"{tb}\n{e}")

        raise e
    finally:
        try:
            # Remove the data
            evaluator.bucket.rmdir()
            PathBucket(run.unique_path / "optimizer").rmdir()
        except Exception as e:  # noqa: BLE001
            # We do not want to raise here as it's just cleanup
            print(e)


if __name__ == "__main__":
    parser, _ = E1.parser()
    E1.add_creation_arugments(parser)
    args = parser.parse_args()
    e1 = E1.parse(args)

    run_it(e1)
