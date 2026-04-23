#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ESTIMATE_CSV = (
    SCRIPT_DIR
    / "msm_commit_wrapper_runs"
    / "gpt2-small"
    / "gpt2-small_msm_latency_estimate_sweep.csv"
)
BITS_PER_WORD = 255
BITS_PER_GIB = (2**30) * 8


def _parse_float(value: str) -> float:
    if value == "":
        return 0.0
    return float(value)


def _bandwidth_limited_latency_ms(base_latency_ms: float, transfer_words: float, bw_gibps: float) -> float:
    if bw_gibps <= 0:
        raise ValueError("bw must be positive")
    transfer_time_ms = (transfer_words * BITS_PER_WORD * 1000.0) / (bw_gibps * BITS_PER_GIB)
    return max(base_latency_ms, transfer_time_ms)


@lru_cache(maxsize=None)
def _load_rows(csv_path: str) -> list[dict[str, str]]:
    with Path(csv_path).open(newline="") as f:
        return list(csv.DictReader(f))


def _find_row(num_msm_pes: int, csv_path: Path) -> dict[str, str]:
    for row in _load_rows(str(csv_path)):
        if int(row["num_pes"]) != num_msm_pes:
            continue
        return row
    raise KeyError(f"no estimator row found for num_pes={num_msm_pes} in {csv_path}")


def get_commit_and_open_latency_breakdown(
    num_MSM_PEs: int,
    num_modmul_dp: int,
    num_modmul_matmul: int,
    bw: float,
    csv_path: Path = DEFAULT_ESTIMATE_CSV,
) -> dict[str, float]:
    if num_modmul_dp <= 0:
        raise ValueError("num_modmul_dp must be positive")
    if num_modmul_matmul <= 0:
        raise ValueError("num_modmul_matmul must be positive")

    row = _find_row(num_MSM_PEs, csv_path)

    commit_and_open_ms = sum(
        _parse_float(row[column])
        for column in [
            "msm_redundant_latency_ms",
            "msm_commit_latency_ms",
            "msm_classic_latency_ms",
            "msm_last_latency_ms",
            "one_point_shift_add_latency_ms",
            "two_point_shift_add_latency_ms",
            "other_trace_latency_ms",
        ]
    )

    dotprod_reduction_parallel_ms = _parse_float(row["dotprod_reduction_compute_only_ms"]) / num_modmul_dp
    matmul_parallel_ms = _parse_float(row["matmul_compute_only_ms"]) / num_modmul_matmul

    dotprod_reduction_ms = _bandwidth_limited_latency_ms(
        base_latency_ms=dotprod_reduction_parallel_ms,
        transfer_words=_parse_float(row["dotprod_reduction_transfer_words"]),
        bw_gibps=bw,
    )
    matmul_ms = _bandwidth_limited_latency_ms(
        base_latency_ms=matmul_parallel_ms,
        transfer_words=_parse_float(row["matmul_transfer_words"]),
        bw_gibps=bw,
    )

    total_ms = commit_and_open_ms + dotprod_reduction_ms + matmul_ms
    return {
        "commit_and_open_ms": commit_and_open_ms,
        "dotprod_reduction_ms": dotprod_reduction_ms,
        "matmul_ms": matmul_ms,
        "total_ms": total_ms,
        "total_area_mm2": _parse_float(row["total_area_mm2"]),
        "padd_area_mm2": _parse_float(row["padd_area_mm2"]),
        "aggregate_mem_area_mm2": _parse_float(row["aggregate_mem_area_mm2"]),
    }


def get_commit_and_open_area_breakdown(
    num_MSM_PEs: int,
    num_modmul_dp: int,
    num_modmul_matmul: int,
    bw: float,
    csv_path: Path = DEFAULT_ESTIMATE_CSV,
) -> dict[str, float]:
    breakdown = get_commit_and_open_latency_breakdown(
        num_MSM_PEs=num_MSM_PEs,
        num_modmul_dp=num_modmul_dp,
        num_modmul_matmul=num_modmul_matmul,
        bw=bw,
        csv_path=csv_path,
    )
    return {
        "total_area_mm2": breakdown["total_area_mm2"],
        "padd_area_mm2": breakdown["padd_area_mm2"],
        "aggregate_mem_area_mm2": breakdown["aggregate_mem_area_mm2"],
    }


def get_commit_and_open_latency(
    num_MSM_PEs: int,
    num_modmul_dp: int,
    num_modmul_matmul: int,
    bw: float,
    csv_path: Path = DEFAULT_ESTIMATE_CSV,
) -> float:
    return get_commit_and_open_latency_breakdown(
        num_MSM_PEs=num_MSM_PEs,
        num_modmul_dp=num_modmul_dp,
        num_modmul_matmul=num_modmul_matmul,
        bw=bw,
        csv_path=csv_path,
    )["total_ms"]


if __name__ == "__main__":
    num_MSM_PEs_list = [1, 2, 4, 8, 16, 32, 64]
    num_modmul_list = [64, 256, 1024, 2048]
    bw_list = [512, 1024, 2048]
    model_hyper_param = [
        ("gpt2-small", "SqueezeMerge_1"),
        ("gpt2-medium", "SqueezeMerge_1"),
        ("opt-125m", "SqueezeMerge_0"),
    ]

    for model_name, squeeze_merge in model_hyper_param:
        csv_path = (
            SCRIPT_DIR
            / "msm_commit_wrapper_runs"
            / model_name
            / f"{model_name}_msm_latency_estimate_sweep.csv"
        )
        if not csv_path.exists():
            print(f"Skipping {model_name}: missing CSV {csv_path}")
            continue

        dump_dir = SCRIPT_DIR / "sim_data" / model_name / squeeze_merge / "commit_open"
        dump_dir.mkdir(parents=True, exist_ok=True)

        for num_MSM_PEs in num_MSM_PEs_list:
            for num_modmul in num_modmul_list:
                for bw in bw_list:
                    breakdown = get_commit_and_open_latency_breakdown(
                        num_MSM_PEs=num_MSM_PEs,
                        num_modmul_dp=num_modmul,
                        num_modmul_matmul=num_modmul,
                        bw=bw,
                        csv_path=csv_path,
                    )

                    output = {
                        "model_name": model_name,
                        "squeeze_merge": squeeze_merge,
                        "csv_path": str(csv_path),
                        "config": {
                            "num_MSM_PEs": num_MSM_PEs,
                            "num_modmul_dp": num_modmul,
                            "num_modmul_matmul": num_modmul,
                            "bw": bw,
                        },
                        "commit_open_latency_breakdown": breakdown,
                    }

                    dump_path = (
                        dump_dir
                        / f"MSM_pe_{num_MSM_PEs}_modmul_{num_modmul}_bw_{bw}.json"
                    )
                    with dump_path.open("w", encoding="utf-8") as f:
                        json.dump(output, f, indent=2)
                    print(f"Dumped {dump_path}")

    print("commit_open_latency_helpers.py end.")
