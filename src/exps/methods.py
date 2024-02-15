from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, TypeVar

import numpy as np
from amltk.optimization import Metric, Trial

if TYPE_CHECKING:
    from amltk.sklearn import CVEvaluation

T = TypeVar("T")


def without_i(lst: list[T], i: int) -> list[T]:
    return lst[:i] + lst[i + 1 :]


def line_to_beat(
    reports: list[Trial.Report],
    metric: Metric,
    sigma: float = 1.0,
) -> float:
    mean = np.mean([r.values[metric.name] for r in reports])
    std = np.std([r.values[metric.name] for r in reports])
    offset = sigma * std
    return float(mean + offset) if metric.minimize else float(mean - offset)


class CVEarlyStopMeanOutsideSigmaStdOfTopN:
    def __init__(self, metric: Metric, n: int, sigma: float = 1.0):
        super().__init__()
        self.top_n: list[Trial.Report] = []
        self.metric = metric
        self.n = n
        self.sigma = sigma

    def update(self, report: Trial.Report) -> None:
        if report.status is not Trial.Status.SUCCESS:
            return

        if len(self.top_n) <= self.n:
            self.top_n.append(report)
            return

        # Possible new lists of top_n
        possible_top_ns = (
            without_i([*self.top_n, report], i=i) for i in range(self.n + 1)
        )

        # Select the new top_n which makes the line to beat more difficult,
        # i.e. higher if it's a maximizing metric or lower if it's a minimizing one
        select = min if self.metric.minimize else max
        selected_reports = select(
            possible_top_ns,
            key=lambda reports: line_to_beat(
                reports,
                metric=self.metric,
                sigma=self.sigma,
            ),
        )
        self.top_n = selected_reports

    def should_stop(self, info: CVEvaluation.SplitInfo) -> bool:
        if len(self.top_n) <= self.n:
            return False

        challenger = float(np.mean(info.scores[self.metric.name]))
        bar = line_to_beat(self.top_n, metric=self.metric, sigma=self.sigma)
        match self.metric.compare(challenger, bar):
            case Metric.Comparison.WORSE | Metric.Comparison.EQUAL:
                return True
            case Metric.Comparison.BETTER:
                return False
            case _:
                raise ValueError("Invalid comparison")


class CVEarlyStopCurrentAverageWorseThanMeanBest:
    def __init__(self, metric: Metric):
        super().__init__()
        self.best: Trial.Report | None = None
        self.metric = metric

    def update(self, report: Trial.Report) -> None:
        if report.status is not Trial.Status.SUCCESS:
            return

        if self.best is None:
            self.best = report
            return

        v_challenger = report.values[self.metric.name]
        v_best = self.best.values[self.metric.name]
        match self.metric.compare(v1=v_challenger, v2=v_best):
            case Metric.Comparison.BETTER:
                self.best = report
            case _:
                pass

    def should_stop(self, info: CVEvaluation.SplitInfo) -> bool:
        if self.best is None:
            return False

        challenger = float(np.mean(info.scores[self.metric.name]))
        best = self.best.values[self.metric.name]
        match self.metric.compare(challenger, best):
            case Metric.Comparison.WORSE | Metric.Comparison.EQUAL:
                return True
            case Metric.Comparison.BETTER:
                return False
            case _:
                raise ValueError("Invalid comparison")


class CVEarlyStopCurrentAverageWorseThanBestWorstSplit:
    def __init__(self, metric: Metric):
        super().__init__()
        self.best: Trial.Report | None = None
        self.metric = metric

    def update(self, report: Trial.Report) -> None:
        if report.status is not Trial.Status.SUCCESS:
            return

        if self.best is None:
            self.best = report
            return

        v_challenger = report.values[self.metric.name]
        v_best = self.best.values[self.metric.name]
        match self.metric.compare(v1=v_challenger, v2=v_best):
            case Metric.Comparison.BETTER:
                self.best = report
            case _:
                pass

    def should_stop(self, info: CVEvaluation.SplitInfo) -> bool:
        if self.best is None:
            return False

        challenger = float(np.mean(info.scores[self.metric.name]))
        split_scores = [
            float(v)
            for k, v in self.best.summary.items()
            if k.startswith("split_") and "train" not in k and "test" not in k
        ]
        worst_fold_score_of_best = min(split_scores)
        match self.metric.compare(challenger, worst_fold_score_of_best):
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
    "mean_outside_1_std_of_top_3": partial(
        CVEarlyStopMeanOutsideSigmaStdOfTopN,
        n=3,
        sigma=1,
    ),
    "mean_outside_1_std_of_top_5": partial(
        CVEarlyStopMeanOutsideSigmaStdOfTopN,
        n=5,
        sigma=1,
    ),
}
