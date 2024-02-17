from __future__ import annotations

from collections.abc import Iterator
from functools import partial
from itertools import chain
from typing import TYPE_CHECKING, TypeVar

import numpy as np
from amltk.optimization import Metric, Trial

if TYPE_CHECKING:
    from amltk.sklearn import CVEvaluation

T = TypeVar("T")


def with_replacement(lst: list[T], replacement: T) -> Iterator[list[T]]:
    for i in range(len(lst)):
        yield lst[:i] + [replacement] + lst[i + 1 :]


def line_to_beat(
    reports: list[Trial.Report],
    metric: Metric,
    sigma: float = 1.0,
) -> float:
    mean = np.mean([r.values[metric.name] for r in reports])
    std = np.std([r.values[metric.name] for r in reports])
    offset = sigma * std
    return float(mean + offset) if metric.minimize else float(mean - offset)


class CVEarlyStopRobustStdOfTopN:
    def __init__(self, metric: Metric, n: int, sigma: float = 1.0):
        super().__init__()
        # List of top_n fold scores
        self.top_n: list[np.ndarray] = []
        self.metric = metric
        self.n = n
        self.sigma = sigma

    def update(self, report: Trial.Report) -> None:
        if report.status is not Trial.Status.SUCCESS:
            return

        if len(self.top_n) <= self.n:
            challenger_fold_Scores = self.scores_from_report(report)
            self.top_n.append(challenger_fold_Scores)
            return

        # Create all possible replacements of the top_n by this new challenger
        challenger_fold_scores = self.scores_from_report(report)

        # Include the current top_n as a possibility as well as all possibilities of
        # replacing one of the current top_n with the challenger
        possible_top_ns = chain(
            [self.top_n],
            with_replacement(self.top_n, challenger_fold_scores),
        )

        # Now select the top_n by the one that minimizes/maximizes the value to beat
        if self.metric.minimize:
            self.top_n = min(possible_top_ns, key=self.value_to_beat)
        else:
            self.top_n = max(possible_top_ns, key=self.value_to_beat)

    def scores_from_report(self, report: Trial.Report) -> np.ndarray:
        suffix = f"val_{self.metric.name}"
        return np.array([v for k, v in report.summary.items() if k.endswith(suffix)])

    def value_to_beat(self, fold_scores: list[np.ndarray]) -> float:
        all_scores = np.concatenate(fold_scores)
        mean = np.mean(all_scores)
        std = self.sigma * np.std(all_scores)
        return float(mean + std) if self.metric.minimize else float(mean - std)

    def should_stop(
        self,
        trial: Trial,  # noqa: ARG002
        scores: CVEvaluation.SplitScores,
    ) -> bool:
        if len(self.top_n) <= self.n:
            return False

        # (mean +/- sigma * std) OF top_n
        value_to_beat = self.value_to_beat(self.top_n)

        fold_scores = np.asarray(scores.val[self.metric.name])

        # (mean +/- sigma * std) OF top_n if challenger replaced one of top_n
        values_if_challeneger_replaced_one_of_top_n = (
            self.value_to_beat(possible_top_n)
            for possible_top_n in with_replacement(self.top_n, fold_scores)
        )

        possible_replacement_is_better = (
            self.metric.compare(v, value_to_beat) is Metric.Comparison.BETTER
            for v in values_if_challeneger_replaced_one_of_top_n
        )

        # Don't stop if it's possible we could get a better top_n
        if any(possible_replacement_is_better):
            return False

        return True


class CVEarlyStopCurrentAverageWorseThanMeanBest:
    def __init__(self, metric: Metric):
        super().__init__()
        self.value_to_beat: float | None = None
        self.metric = metric

    def update(self, report: Trial.Report) -> None:
        if report.status is not Trial.Status.SUCCESS:
            return

        if self.value_to_beat is None:
            self.value_to_beat = self.mean_fold_value(report)
            return

        match self.metric.compare(
            v1=report.values[self.metric.name],
            v2=self.value_to_beat,
        ):
            case Metric.Comparison.BETTER:
                self.value_to_beat = self.mean_fold_value(report)
            case _:
                pass

    def mean_fold_value(self, report: Trial.Report) -> float:
        suffix = f"val_{self.metric.name}"
        scores = [v for k, v in report.summary.items() if k.endswith(suffix)]
        return float(np.mean(scores))

    def should_stop(
        self,
        trial: Trial,  # noqa: ARG002
        scores: CVEvaluation.SplitScores,
    ) -> bool:
        if self.value_to_beat is None:
            return False

        challenger = float(np.mean(scores.val[self.metric.name]))

        match self.metric.compare(v1=challenger, v2=self.value_to_beat):
            case Metric.Comparison.WORSE | Metric.Comparison.EQUAL:
                return True
            case Metric.Comparison.BETTER:
                return False
            case _:
                raise ValueError("Invalid comparison")


class CVEarlyStopCurrentAverageWorseThanBestWorstSplit:
    def __init__(self, metric: Metric):
        super().__init__()
        self.metric = metric
        self.value_to_beat: float | None = None

    def update(self, report: Trial.Report) -> None:
        if report.status is not Trial.Status.SUCCESS:
            return

        if self.value_to_beat is None:
            self.value_to_beat = self.worst_fold_value(report)
            return

        match self.metric.compare(
            v1=report.values[self.metric.name],
            v2=self.value_to_beat,
        ):
            case Metric.Comparison.BETTER:
                self.value_to_beat = self.worst_fold_value(report)
            case _:
                pass

    def worst_fold_value(self, report: Trial.Report) -> float:
        suffix = f"val_{self.metric.name}"
        scores = (v for k, v in report.summary.items() if k.endswith(suffix))
        worst = max(scores) if self.metric.minimize else min(scores)
        return float(worst)

    def should_stop(
        self,
        trial: Trial,  # noqa: ARG002
        scores: CVEvaluation.SplitScores,
    ) -> bool:
        if self.value_to_beat is None:
            return False

        challenger = float(np.mean(scores.val[self.metric.name]))

        match self.metric.compare(v1=challenger, v2=self.value_to_beat):
            case Metric.Comparison.WORSE | Metric.Comparison.EQUAL:
                return True
            case Metric.Comparison.BETTER:
                return False
            case _:
                raise ValueError("Invalid comparison")


METHODS = {
    "disabled": None,
    "current_average_worse_than_mean_best": CVEarlyStopCurrentAverageWorseThanMeanBest,
    "current_average_worse_than_best_worst_split": CVEarlyStopCurrentAverageWorseThanBestWorstSplit,  # noqa: E501
    "robust_std_top_3": partial(
        CVEarlyStopRobustStdOfTopN,
        n=3,
        sigma=1,
    ),
    "robust_std_top_5": partial(
        CVEarlyStopRobustStdOfTopN,
        n=5,
        sigma=1,
    ),
}
