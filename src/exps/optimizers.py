from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Concatenate, Literal, overload
from typing_extensions import Self, override

import amltk.randomness
import numpy as np
from amltk.exceptions import CVEarlyStoppedError
from amltk.optimization import Metric, Trial
from amltk.optimization.optimizer import Optimizer
from amltk.optimization.optimizers.smac import SMACOptimizer, SMACTrialInfo
from amltk.pipeline import Node
from amltk.randomness import as_int
from amltk.store import PathBucket
from smac.facade import AbstractFacade, HyperparameterOptimizationFacade
from smac.scenario import Scenario

if TYPE_CHECKING:
    from amltk.types import FidT, Seed
    from ConfigSpace import Configuration, ConfigurationSpace


class SMACOptimizerWithIncreasesRetries(SMACOptimizer):
    @override
    @classmethod
    def create(
        cls,
        *,
        space: ConfigurationSpace | Node,
        metrics: Metric | Sequence[Metric],
        bucket: PathBucket | str | Path | None = None,
        time_profile: str | None = None,
        deterministic: bool = True,
        seed: Seed | None = None,
        fidelities: Mapping[str, FidT] | None = None,
        continue_from_last_run: bool = False,
        logging_level: int | Path | Literal[False] | None = False,
    ) -> Self:
        """Overwrites the default implementation of create to also increase
        the number of retries with the ConfigSelector as on small datasets, the
        default number of retries is not enough to find a unique configurations,
        causing the whole process to fail.
        """
        seed = as_int(seed)
        match bucket:
            case None:
                bucket = PathBucket(
                    f"{cls.__name__}-{datetime.now().isoformat()}",
                )
            case str() | Path():
                bucket = PathBucket(bucket)
            case bucket:
                bucket = bucket  # noqa: PLW0127

        # NOTE SMAC always minimizes! Hence we make it a minimization problem
        metric_names: str | list[str]
        if isinstance(metrics, Sequence):
            metric_names = [metric.name for metric in metrics]
        else:
            metric_names = metrics.name

        if isinstance(space, Node):
            space = space.search_space(parser=cls.preferred_parser())

        facade_cls: type[AbstractFacade]
        if fidelities:
            raise NotImplementedError("Fidelities are not used for these experiments")
        scenario = Scenario(
            configspace=space,
            seed=seed,
            output_directory=bucket.path / "smac3_output",
            deterministic=deterministic,
            objectives=metric_names,
            crash_cost=list(cls.crash_costs(metrics).values()),
        )
        facade_cls = HyperparameterOptimizationFacade

        # NOTE: This is the important part.
        # As sometimes the optimizer raises StopIteration, it seems that a good
        # strategy is to increase the number of retries. I suspect this is because
        # the acquisition function fails to find new configurations within the default
        # of 16 retries due to hitting an optimum. Given we have a non-finite search
        # space, this is kind of surprising but it's hard to debug with SMAC otherwise.
        # If this doesn't solve the 74/3200 failed runs, then the solution will be to
        # consider SMAC as having converged at this point and to not treat the
        # `StopIteration` as an error and instead gracefully halt at this point.
        # We will try this retry strategy first
        config_selector = HyperparameterOptimizationFacade.get_config_selector(
            scenario,
            retries=128,  # Default is 16
        )

        facade = facade_cls(
            scenario=scenario,
            target_function="dummy",  # NOTE: https://github.com/automl/SMAC3/issues/946
            overwrite=not continue_from_last_run,
            logging_level=logging_level,
            config_selector=config_selector,
            multi_objective_algorithm=facade_cls.get_multi_objective_algorithm(
                scenario=scenario,
            ),
        )
        return cls(
            facade=facade,
            fidelities=fidelities,
            bucket=bucket,
            metrics=metrics,
            time_profile=time_profile,
        )


class SMACReportEarlyStopAsFailed(SMACOptimizerWithIncreasesRetries):
    @override
    def tell(self, report: Trial.Report[SMACTrialInfo]) -> None:
        """We don't really need to overwrite anything as this is default behaviour."""
        return super().tell(report)


class SMACReportEarlyStopWithFoldMean(SMACOptimizerWithIncreasesRetries):
    @override
    def tell(self, report: Trial.Report[SMACTrialInfo]) -> None:
        """If a trial was report as failed, we convert it to a success by taking
        the mean of the fold means and using that as the performance to report to SMAC.
        """
        print(report)
        print(report.status)
        print(report.exception)
        print(type(report.exception))
        match report.status:
            # In the case of success or crash, nothing to change
            case Trial.Status.SUCCESS | Trial.Status.CRASHED | Trial.Status.UNKNOWN:
                return super().tell(report)
            case Trial.Status.FAIL if isinstance(report.exception, CVEarlyStoppedError):
                print("IN HERE")
                # Howeever when a trial fails due to early stopping, we create a
                # success report instead, using the mean of fold scores as the value.
                trial = report.trial
                metric = next(iter(self.metrics.values()))
                fold_scores = [
                    v
                    for k, v in trial.summary.items()
                    if k.startswith("split_") and k.endswith(f"val_{metric.name}")
                ]
                mean_fold_score = float(np.mean(fold_scores))

                success_report = Trial.Report(
                    trial=trial,
                    status=Trial.Status.SUCCESS,
                    exception=report.exception,
                    traceback=report.traceback,
                    reported_at=report.reported_at,
                    values={metric.name: mean_fold_score},
                )
                return super().tell(success_report)
            case _:
                # In all other cases, we just pass the report as is.
                return super().tell(report)


class RSOptimizer(Optimizer):
    def __init__(
        self,
        space: Node,
        metrics: Metric | Sequence[Metric],
        bucket: str | Path | PathBucket | None = None,
        seed: Seed | None = None,
        retries: int = 10_000,
    ) -> None:
        from amltk.pipeline.parsers.configspace import parser

        bucket = PathBucket(bucket) if bucket is not None else None

        self.space = space
        self.seed = amltk.randomness.as_int(seed)
        self.seen: set[Configuration] = set()
        self.n = 0
        self.retries = retries
        self._space = parser(space, seed=self.seed)
        super().__init__([metrics] if isinstance(metrics, Metric) else metrics, bucket)

    def _sample_one_next(self) -> Trial:
        if self.n == 0:
            default = self._space.get_default_configuration()
            self.seen.add(default)
            name = f"trial-{self.n}"
            trial = Trial.create(
                name=name,
                config=dict(default),
                info=None,
                seed=self.seed,
                bucket=self.bucket / name,
                metrics=self.metrics,
            )
            self.n = self.n + 1
            return trial

        for _ in range(self.retries):
            config = self._space.sample_configuration()
            if config not in self.seen:
                self.seen.add(config)
                name = f"trial-{self.n}"
                trial = Trial.create(
                    name=name,
                    config=dict(config),
                    info=None,
                    seed=self.seed,
                    bucket=self.bucket / name,
                    metrics=self.metrics,
                )
                self.n = self.n + 1
                return trial

        raise RuntimeError(f"Could not sample a unique configuration in {self.retries}")

    @overload
    def ask(self, n: int) -> Iterable[Trial]:
        ...

    @overload
    def ask(self, n: None = None) -> Trial:
        ...

    @override
    def ask(self, n: int | None = None) -> Trial | Iterable[Trial]:
        if n is None:
            return self._sample_one_next()

        return [self._sample_one_next() for _ in range(n)]

    @override
    def tell(self, report: Trial.Report) -> None:
        return

    @override
    @classmethod
    def preferred_parser(
        cls,
    ) -> str | Callable[Concatenate[Node, ...], Any] | Callable[[Node], Any] | None:
        from amltk.pipeline.parsers.configspace import parser

        return parser

    @override
    @classmethod
    def create(
        cls,
        *,
        space: Node,
        metrics: Metric | Sequence[Metric],
        bucket: str | Path | PathBucket | None = None,
        seed: Seed | None = None,
    ) -> Self:
        return cls(space=space, metrics=metrics, bucket=bucket, seed=seed)


OPTIMIZERS = {
    "smac": SMACOptimizer,
    "random_search": RSOptimizer,
    "smac_early_stop_as_failed": SMACReportEarlyStopAsFailed,
    "smac_early_stop_with_fold_mean": SMACReportEarlyStopWithFoldMean,
}
