"""This file is only used for timing purposes."""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

import openml
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

if TYPE_CHECKING:
    import pandas as pd


def _load_openml(tid: int):
    oml_task = openml.tasks.get_task(
        tid,
        download_splits=True,
        download_data=True,
        download_qualities=False,
        download_features_meta_data=False,
    )
    print("Task", oml_task, "target", oml_task.target_name, flush=True)
    X, *_ = oml_task.get_dataset().get_data(dataset_format="dataframe")
    y = X[oml_task.target_name]
    X = X.drop(columns=oml_task.target_name)

    # Leaky preprocessing, however not used to measure performances.
    X = OrdinalEncoder().fit_transform(X)
    X = SimpleImputer(strategy="mean").fit_transform(X)
    y = LabelEncoder().fit_transform(y=y)
    return X, y


def time_holdout(X: pd.DataFrame, y: pd.Series) -> float:
    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        random_state=42,
        test_size=0.10,
        stratify=y,
        shuffle=True,
    )
    mlp = MLPClassifier(random_state=42, n_iter_no_change=10_000)

    start = time.time()
    mlp.fit(X_train, y_train)
    end = time.time()
    return end - start


def time_cv(X: pd.DataFrame, y: pd.Series) -> float:
    splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    total_time = 0
    for i, (train, _) in enumerate(splitter.split(X, y)):
        X_train, y_train = X[train], y[train]
        mlp = MLPClassifier(random_state=42, n_iter_no_change=10_000)

        start = time.time()
        mlp.fit(X_train, y_train)
        end = time.time()
        duration = end - start
        print("duration fold", i, duration, flush=True)
        total_time += end - start

    return total_time


if __name__ == "__main__":
    task_id = 359993
    X, y = _load_openml(task_id)
    holdout = time_holdout(X, y)
    print("holdout", holdout, flush=True)
    cv = time_cv(X, y)
    print("cv", cv, flush=True)
    print("increase", cv / holdout)
