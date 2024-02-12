from __future__ import annotations

import json

import openml

OPENML_AMLB_CLASSIFICATION_SUITE = 271

if __name__ == "__main__":
    suite = openml.study.get_suite(OPENML_AMLB_CLASSIFICATION_SUITE)
    tasks = suite.tasks

    data = {}
    for task_i in suite.tasks:
        task = openml.tasks.get_task(task_i)
        dataset_id = openml.datasets.get_dataset(task.dataset_id)
        n_features = dataset_id.qualities["NumberOfFeatures"]
        n_instances = dataset_id.qualities["NumberOfInstances"]
        cell_size = n_features * n_instances
        data[task_i] = {
            "dataset_id": task.dataset_id,
            "n_features": n_features,
            "n_instances": n_instances,
            "cell_size": cell_size,
        }

    with open(f"openml_suite_{OPENML_AMLB_CLASSIFICATION_SUITE}.json", "w") as f:
        json.dump(data, f, indent=4)
