from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from amltk.optimization import Metric, Trial

if TYPE_CHECKING:
    from amltk.sklearn import CVEvaluation


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
            if k.startswith("split_") and "train" not in k
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
}
