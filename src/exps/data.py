from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import openml
from openml.tasks.task import TaskType

if TYPE_CHECKING:
    import pandas as pd
    from amltk.sklearn.evaluation import TaskTypeName


def get_fold(
    openml_task_id: int,
    fold: int,
) -> tuple[
    TaskTypeName,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame | pd.Series,
    pd.DataFrame | pd.Series,
]:
    task = openml.tasks.get_task(
        openml_task_id,
        download_splits=True,
        download_data=True,
        download_qualities=True,
        download_features_meta_data=True,
    )
    train_idx, test_idx = task.get_train_test_split_indices(fold=fold)
    X, y = task.get_X_and_y(dataset_format="dataframe")  # type: ignore
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    match task.task_type_id:
        case TaskType.SUPERVISED_CLASSIFICATION:
            if np.ndim(y_train) > 1:
                return "multilabel-indicator", X_train, X_test, y_train, y_test

            if np.unique(y_train).size == 2:  # noqa: PLR2004
                return "binary", X_train, X_test, y_train, y_test

            return "multiclass", X_train, X_test, y_train, y_test

        case TaskType.SUPERVISED_REGRESSION:
            if np.ndim(y_train) > 1:
                return "continuous-multioutput", X_train, X_test, y_train, y_test

            return "continuous", X_train, X_test, y_train, y_test
        case _:
            raise ValueError(f"Task type {task.task_type_id} not supported")
