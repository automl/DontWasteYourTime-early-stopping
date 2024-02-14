from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import openml
import pandas as pd
from openml.tasks.task import TaskType

if TYPE_CHECKING:
    from amltk.sklearn.evaluation import TaskTypeName


def get_fold(
    openml_task_id: int,
    fold: int,
    n_splits: int,
    seed: int | None = None,
) -> tuple[
    TaskTypeName,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame | pd.Series,
    pd.DataFrame | pd.Series,
]:
    """Get the data for a specific fold of an OpenML task.

    Args:
        openml_task_id: The OpenML task id.
        fold: The fold number.
        n_splits: The number of splits that will be applied. This is used
            to resample training data such that enough at least instance for each class
            is present for every stratified split.
        seed: The random seed to use for reproducibility of resampling if necessary.
    """
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

    # If we are in binary/multilclass setting and there is not enough instances
    # with a given label to perform stratified sampling with `n_splits`, we first
    # find these labels, take the first N instances which have these labels and allows
    # us to reach `n_splits` instances for each label.
    indices_to_resample = None
    if y_train.ndim == 1:
        label_counts = y_train.value_counts()
        under_represented_labels = label_counts[label_counts < n_splits]

        collected_indices = []
        if any(under_represented_labels):
            under_rep_instances = y_train[y_train.isin(under_represented_labels.index)]

            grouped_by_label = under_rep_instances.to_frame("label").groupby(
                "label",
                observed=True,  # Handles categoricals
            )
            for _label, instances_with_label in grouped_by_label:
                n_to_take = n_splits - len(instances_with_label)

                need_to_sample_repeatedly = n_to_take > len(instances_with_label)
                resampled_instances = instances_with_label.sample(
                    n=n_to_take,
                    random_state=seed,
                    # It could be that we have to repeat sample if there are not enough
                    # instances to hit `n_splits` for a given label.
                    replace=need_to_sample_repeatedly,
                )
                collected_indices.append(np.asarray(resampled_instances.index))

            indices_to_resample = np.concatenate(collected_indices)

    if indices_to_resample is not None:
        # Give the new samples a new index to not overlap with the original data.
        new_start_idx = X_train.index.max() + 1
        new_end_idx = new_start_idx + len(indices_to_resample)
        new_idx = pd.RangeIndex(start=new_start_idx, stop=new_end_idx)
        resampled_X = X_train.loc[indices_to_resample].set_index(new_idx)
        resampled_y = y_train.loc[indices_to_resample].set_axis(new_idx)
        X_train = pd.concat([X_train, resampled_X])
        y_train = pd.concat([y_train, resampled_y])

    if y_train.value_counts().min() < n_splits:
        raise RuntimeError(
            "Not enough instances for stratified sampling, something went wrong"
            "\n"
            f"y_train.value_counts():\n{y_train.value_counts()}",
        )

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
