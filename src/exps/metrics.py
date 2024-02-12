from __future__ import annotations

from amltk.optimization import Metric

METRICS = {
    "accuracy": Metric("accuracy", minimize=False, bounds=(0, 1)),
    "roc_auc_ovr": Metric("roc_auc_ovr", minimize=False, bounds=(0, 1)),
}
