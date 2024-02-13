from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Concatenate, overload
from typing_extensions import Self, override

import amltk.randomness
from amltk.optimization import Metric, Trial
from amltk.optimization.optimizer import Optimizer
from amltk.optimization.optimizers.smac import SMACOptimizer
from amltk.store import PathBucket

if TYPE_CHECKING:
    from amltk.pipeline import Node
    from amltk.types import Seed
    from ConfigSpace import Configuration


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
            default = self._space.default_configuration()
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
}
