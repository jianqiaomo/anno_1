#!/usr/bin/env python3

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BUILD_EQ_CSV = (
    SCRIPT_DIR
    / "msm_commit_wrapper_runs"
    / "gpt2-small"
    / "gpt2-small_build_eq_info.csv"
)


@lru_cache(maxsize=None)
def _load_rows(csv_path: str) -> list[dict[str, str]]:
    with Path(csv_path).open(newline="") as f:
        return list(csv.DictReader(f))


def get_build_eq_rows(csv_path: Path = DEFAULT_BUILD_EQ_CSV) -> list[dict[str, int | str]]:
    return [
        {
            "workload": row["workload"],
            "build_eq_length": int(row["build_eq_length"]),
            "count": int(row["count"]),
        }
        for row in _load_rows(str(csv_path))
    ]


def get_build_eq_length_counts(csv_path: Path = DEFAULT_BUILD_EQ_CSV) -> dict[int, int]:
    return {
        row["build_eq_length"]: row["count"]
        for row in get_build_eq_rows(csv_path)
    }


if __name__ == "__main__":
    for row in get_build_eq_rows():
        print(
            f"workload={row['workload']} "
            f"build_eq_length={row['build_eq_length']} "
            f"count={row['count']}"
        )
