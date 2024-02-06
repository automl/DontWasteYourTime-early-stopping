from __future__ import annotations

import argparse
import logging
from collections.abc import Iterator
from pathlib import Path

import openml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def as_task_list(tasks: int | str | list[int | str]) -> Iterator[int]:
    match tasks:
        case int():
            yield tasks
        case str():
            suite_id = int(tasks.split("-")[1])
            _suite = openml.study.get_suite(suite_id)
            assert _suite.tasks is not None
            yield from _suite.tasks
        case list():
            for t in tasks:
                yield from as_task_list(t)


def download(tasks: list[int | str], *, openml_cache_dir: Path | None = None) -> None:
    if openml_cache_dir is not None:
        openml.config.set_root_cache_directory(openml_cache_dir)

    for task in as_task_list(tasks):
        logger.info(f"Downloading {task=}")
        # Don't need return value, this causes the download
        openml.tasks.get_task(
            task_id=task,
            download_splits=True,
            download_data=True,
            download_qualities=True,
            download_features_meta_data=True,
        )
        logger.info("- Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks", nargs="+", type=str)
    parser.add_argument("--openml-cache-dir", type=Path, required=False)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list is True:
        tasks = as_task_list(args.tasks)
        for task in tasks:
            print(task)
    else:
        download(args.tasks, openml_cache_dir=args.openml_cache_dir)
