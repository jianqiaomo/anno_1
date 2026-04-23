import math
import json
import itertools
import time
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
from GKR_model_wrap import (
    phase_last_sumcheck_all_rounds_latency, 
    model_gkr_gates_phase12_all_layers_latency,
    model_matmul_all_layers_latency,
)
from logup_model import model_logup_proof_core_all_columns
import argparse


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hard-window-sizes",
        type=int,
        nargs="+",
        default=[256, 512, 1024, 2048],  # [256, 1024]
        help="One or more hard window sizes to evaluate. E.g., --hard-window-sizes 256 512 1024",
    )
    parser.add_argument(
        "--top-ns-exp",
        type=int,
        nargs="+",
        default=[13, 14, 15, 16, 17, 18],  # [13, 14, 15, 16, 17]
        help="One or more top_n values to evaluate. E.g., --top-ns 13 14 15 16 17 (these are exponents, so the actual top_n will be 2^13, 2^14, etc.)",
    )
    parser.add_argument(
        "--onchip-mle-sizes",
        type=int,
        nargs="+",
        default=[2**10, 2**11, 2**12, 2**13, 2**14],  # [2**10, 2**11, 2**12, 2**13, 2**14],
        help="Onchip MLE sizes to sweep.",
    )
    parser.add_argument(
        "--dram-bandwidth",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024, 2048],  # [512, 1024, 2048, 4096],
        help="DRAM bandwidth values to sweep.",
    )
    parser.add_argument(
        "--inv-units",
        type=int,
        nargs="+",
        default=[1, 2, 4],  # [1, 4, 16],
        help="Inverse unit counts to sweep.",
    )
    parser.add_argument(
        "--sumcheck-pes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64],  # [1, 2, 4, 8, 16, 32],
        help="Sumcheck PE counts to sweep.",
    )
    parser.add_argument(
        "--eval-engines",
        type=int,
        nargs="+",
        default=[2, 3],  # [2, 3],
        help="Eval engine counts to sweep.",
    )
    parser.add_argument(
        "--product-lanes",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5], # [2, 3, 4, 5],
        help="Product lane counts to sweep.",
    )
    parser.add_argument(
        "--elements-per-sram",
        type=int,
        nargs="+",
        default=[512, 1024, 2048],
        help="Elements per SRAM feed to DP to sweep.",
    )
    parser.add_argument(
        "--sc-vg-sizes",
        type=int,
        nargs="+",
        default=[128],
        help="Sumcheck Vg sizes to sweep.",
    )
    parser.add_argument(
        "--compiler-options",
        type=int,
        nargs="+",
        default=[0, 1],
        help="Compiler options to sweep.",
    )
    parser.add_argument(
        "--multiprocess",
        action="store_true",
        default=False,
        help="Use multiprocessing across entries in model_squeeze_list.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=70,
        help="Maximum number of workers.",
    )
    return parser.parse_args()


def _stringify_shape_key(obj):
    if isinstance(obj, dict):
        return {str(key): _stringify_shape_key(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_stringify_shape_key(value) for value in obj]
    return obj


NOISY_DUMP_KEYS = {"compute_queue", "load_store_queue", "range_tasks"}
NOISY_DUMP_PATTERNS = [
    re.compile(r"^[A-Za-z0-9]+_easy_range_list$"),
    re.compile(r"^[A-Za-z0-9]+_easy_avg_unique_[A-Za-z0-9]+_per_[A-Za-z0-9]+$"),
    re.compile(r"^[A-Za-z0-9]+_hard_window_list$"),
    re.compile(r"^[A-Za-z0-9]+_hard_window_captured_[A-Za-z0-9]+_count$"),
    re.compile(r"^[A-Za-z0-9]+_hard_window_uncaptured_[A-Za-z0-9]+_count$"),
    re.compile(r"^[A-Za-z0-9]+_hard_window_captured_unique_[A-Za-z0-9]+_count$"),
    re.compile(r"^[A-Za-z0-9_]+_info_range_sort$"),
]


def _should_drop_dump_key(key):
    if key in NOISY_DUMP_KEYS:
        return True
    return any(pattern.match(key) for pattern in NOISY_DUMP_PATTERNS)


def _strip_large_dump_fields(obj):
    if isinstance(obj, dict):
        cleaned = {}
        for key, value in obj.items():
            if _should_drop_dump_key(key):
                continue
            cleaned[key] = _strip_large_dump_fields(value)
        return cleaned
    if isinstance(obj, list):
        return [_strip_large_dump_fields(value) for value in obj]
    return obj


def _run_single_sweep(task):
    model_name = task["model_name"]
    squeeze_merge = task["squeeze_merge"]
    layers = task["layers"]
    repeat_layers = task["repeat_layers"]
    repeat_times = task["repeat_times"]
    logup_l = task["logup_l"]
    logup_m = task["logup_m"]
    logup_n = task["logup_n"]
    hard_window_size = task["hard_window_size"]
    capture_y_sram_size_exp = task["capture_y_sram_size_exp"]
    num_onchip_mle_sizes = task["num_onchip_mle_sizes"]
    DRAM_bandwidth_B_cycle = task["DRAM_bandwidth_B_cycle"]
    num_inv_unit = task["num_inv_unit"]
    sumcheck_pes = task["sumcheck_pes"]
    eval_engines = task["eval_engines"]
    product_lanes = task["product_lanes"]
    num_elements_per_sram_feed_to_dp = task["num_elements_per_sram_feed_to_dp"]
    num_sc_Vg_sizes = task["num_sc_Vg_sizes"]
    compiler_option = task["compiler_option"]
    build_mle_throughput_per_cycle = task["build_mle_throughput_per_cycle"]
    num_dp_mul = task["num_dp_mul"]
    script_dir = Path(task["script_dir"])
    repo_root = Path(task["repo_root"])

    logup_result = model_logup_proof_core_all_columns(
        l=logup_l,
        m=logup_m,
        n=logup_n,
        DRAM_bandwidth_B_cycle=DRAM_bandwidth_B_cycle,
        num_inv_unit=num_inv_unit,
        num_sumcheck_pe=sumcheck_pes,
        num_eval_engines=eval_engines,
        num_product_lanes=product_lanes,
        num_onchip_mle_sizes=num_onchip_mle_sizes,
    )

    gkr_gates_result = model_gkr_gates_phase12_all_layers_latency(
        model_name=model_name,
        squeeze_merge=squeeze_merge,
        layers=layers,
        repeat_layers=repeat_layers,
        repeat_times=repeat_times,
        hard_window_sizes=hard_window_size,
        capture_y_sram_size_exp=capture_y_sram_size_exp,
        num_onchip_mle_sizes=num_onchip_mle_sizes,
        num_sc_Vg_sizes=num_sc_Vg_sizes,
        compiler_option=compiler_option,
        build_mle_throughput_per_cycle=build_mle_throughput_per_cycle,
        DRAM_bandwidth_B_cycle=DRAM_bandwidth_B_cycle,
        num_dp_mul=num_dp_mul,
        num_elements_per_sram_feed_to_dp=num_elements_per_sram_feed_to_dp,
        sumcheck_pes=sumcheck_pes,
        eval_engines=eval_engines,
        product_lanes=product_lanes,
        data_root=str(script_dir / "comp_data"),
    )

    matmul_result = model_matmul_all_layers_latency(
        model_name=model_name,
        squeeze_merge=squeeze_merge,
        build_mle_throughput_per_cycle=build_mle_throughput_per_cycle,
        DRAM_bandwidth_B_cycle=DRAM_bandwidth_B_cycle,
        num_onchip_mle_sizes=num_onchip_mle_sizes,
        num_dp_mul=num_dp_mul,
        num_elements_per_sram_feed_to_dp=num_elements_per_sram_feed_to_dp,
        sumcheck_pes=sumcheck_pes,
        eval_engines=eval_engines,
        product_lanes=product_lanes,
        output_root=str(repo_root / "output"),
    )

    phase_last_result = phase_last_sumcheck_all_rounds_latency(
        comp_data_dir=script_dir / "comp_data" / model_name / squeeze_merge,
        layers=layers,
        repeat_layers=repeat_layers,
        repeat_times=repeat_times,
        hard_window_sizes=hard_window_size,
        top_n_exp=capture_y_sram_size_exp,
        num_onchip_mle_sizes=num_onchip_mle_sizes,
        num_sc_Vg_sizes=num_sc_Vg_sizes,
        compiler_option=compiler_option,
        build_mle_throughput_per_cycle=build_mle_throughput_per_cycle,
        DRAM_bandwidth_B_cycle=DRAM_bandwidth_B_cycle,
        num_dp_mul=num_dp_mul,
        num_elements_per_sram_feed_to_dp=num_elements_per_sram_feed_to_dp,
        sumcheck_pes=sumcheck_pes,
        eval_engines=eval_engines,
        product_lanes=product_lanes,
    )

    latency_breakdown = {
        "logup_latency": logup_result["Total_latency_cycle"],
        "phase12_all_layers_latency": gkr_gates_result["Total_latency_cycle"],
        "matmul_all_layers_latency": matmul_result["Total_latency_cycle"],
        "last_latency": phase_last_result["Total_latency_cycle"],
    }
    latency_breakdown["total_latency_proof_part"] = sum(latency_breakdown.values())

    combined_result = {
        "model_name": model_name,
        "squeeze_merge": squeeze_merge,
        "sweep_config": {
            "hard_window_size": hard_window_size,
            "capture_y_sram_size_exp": capture_y_sram_size_exp,
            "num_onchip_mle_sizes": num_onchip_mle_sizes,
            "DRAM_bandwidth_B_cycle": DRAM_bandwidth_B_cycle,
            "num_inv_unit": num_inv_unit,
            "sumcheck_pes": sumcheck_pes,
            "eval_engines": eval_engines,
            "product_lanes": product_lanes,
            "num_elements_per_sram_feed_to_dp": num_elements_per_sram_feed_to_dp,
            "num_sc_Vg_sizes": num_sc_Vg_sizes,
            "compiler_option": compiler_option,
            "build_mle_throughput_per_cycle": build_mle_throughput_per_cycle,
            "num_dp_mul": num_dp_mul,
        },
        "latency_breakdown": latency_breakdown,
        "logup": _stringify_shape_key(logup_result),
        "gkr_gates": _stringify_shape_key(gkr_gates_result),
        "matmul": _stringify_shape_key(matmul_result),
        "phase_last": _stringify_shape_key(phase_last_result),
    }
    combined_result = _strip_large_dump_fields(combined_result)

    dump_dir = (
        script_dir
        / "sim_data"
        / model_name
        / squeeze_merge
        / f"compile_{compiler_option}"
        / f"bw_{DRAM_bandwidth_B_cycle}"
        / f"build_thput_{build_mle_throughput_per_cycle}_dp_mul_{num_dp_mul}"
    )
    dump_dir.mkdir(parents=True, exist_ok=True)
    dump_filename = (
        f"hardwins_{hard_window_size}"
        f"_topn_{capture_y_sram_size_exp}"
        f"_onchip_{num_onchip_mle_sizes}"
        f"_inv_{num_inv_unit}"
        f"_sumcheck_pes_{sumcheck_pes}"
        f"_eval_{eval_engines}"
        f"_product_{product_lanes}"
        f"_sramfeed_{num_elements_per_sram_feed_to_dp}"
        f"_scvg_{num_sc_Vg_sizes}.json"
    )
    dump_path = dump_dir / dump_filename
    with dump_path.open("w", encoding="utf-8") as f:
        json.dump(combined_result, f, indent=2)

    return {
        "model_name": model_name,
        "dump_path": str(dump_path),
        "latency_breakdown": latency_breakdown,
    }


def _get_dump_path(task):
    script_dir = Path(task["script_dir"])
    model_name = task["model_name"]
    squeeze_merge = task["squeeze_merge"]
    compiler_option = task["compiler_option"]
    DRAM_bandwidth_B_cycle = task["DRAM_bandwidth_B_cycle"]
    build_mle_throughput_per_cycle = task["build_mle_throughput_per_cycle"]
    num_dp_mul = task["num_dp_mul"]
    hard_window_size = task["hard_window_size"]
    capture_y_sram_size_exp = task["capture_y_sram_size_exp"]
    num_onchip_mle_sizes = task["num_onchip_mle_sizes"]
    num_inv_unit = task["num_inv_unit"]
    sumcheck_pes = task["sumcheck_pes"]
    eval_engines = task["eval_engines"]
    product_lanes = task["product_lanes"]
    num_elements_per_sram_feed_to_dp = task["num_elements_per_sram_feed_to_dp"]
    num_sc_Vg_sizes = task["num_sc_Vg_sizes"]

    dump_dir = (
        script_dir
        / "sim_data"
        / model_name
        / squeeze_merge
        / f"compile_{compiler_option}"
        / f"bw_{DRAM_bandwidth_B_cycle}"
        / f"build_thput_{build_mle_throughput_per_cycle}_dp_mul_{num_dp_mul}"
    )
    dump_filename = (
        f"hardwins_{hard_window_size}"
        f"_topn_{capture_y_sram_size_exp}"
        f"_onchip_{num_onchip_mle_sizes}"
        f"_inv_{num_inv_unit}"
        f"_sumcheck_pes_{sumcheck_pes}"
        f"_eval_{eval_engines}"
        f"_product_{product_lanes}"
        f"_sramfeed_{num_elements_per_sram_feed_to_dp}"
        f"_scvg_{num_sc_Vg_sizes}.json"
    )
    return dump_dir / dump_filename


"""
python3 model_sweep.py \
--hard-window-sizes 512 \
--top-ns-exp 15 \
--onchip-mle-sizes 1024 \
--dram-bandwidth 1024 \
--inv-units 2 \
--sumcheck-pes 4 \
--eval-engines 3 \
--product-lanes 4 \
--elements-per-sram 1024 \
--sc-vg-sizes 128 \
--compiler-options 0
"""
if __name__ == "__main__":
    args = _parse_args()
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    sweep_start_time = time.time()

    # sweep arguments from parsed args
    hard_window_sizes = args.hard_window_sizes
    capture_y_sram_size_exp_list = args.top_ns_exp
    onchip_mle_sizes_list = args.onchip_mle_sizes
    DRAM_bandwidth_B_cycle_list = args.dram_bandwidth
    inv_unit_list = args.inv_units
    sumcheck_pes_list = args.sumcheck_pes
    eval_engines_list = args.eval_engines
    product_lanes_list = args.product_lanes
    elements_per_sram_feed_to_dp_list = args.elements_per_sram
    sc_Vg_sizes_list = args.sc_vg_sizes
    compiler_option_list = args.compiler_options
    
    # build_mle_throughput_per_cycle_dp_mul_list is hardcoded for now as it contains tuples
    build_mle_throughput_per_cycle_dp_mul_list = [(32, 64), (128, 256), (512, 1024), (1024, 2048)]  # [(32, 64), (64, 128), (128, 256), (256, 512), (512, 1024), (1024, 2048)]
    # build_mle_throughput_per_cycle_dp_mul_list = [(128, 256)]
    # sumcheck_pes_build_mle_throughput_per_cycle_dp_mul_list = [(1, 32, 64), (4, 128, 256), (16, 512, 1024), (32, 1024, 2048)]

    # model, model, layer, repeat layer, repeat times, l, m, n,
    model_hyper_param = [
        ("gpt2-small", "SqueezeMerge_1", [1, 2, 3, 4, 6, 8], [6, 8], 24, 7, 2**21, 2**7),
        ("gpt2-medium", "SqueezeMerge_1", [1, 2, 3, 4, 6, 8], [6, 8], 48, 7, 2**23, 2**7),
        ("opt-125m", "SqueezeMerge_0", [1,2,3,5,6,7,8,9,11,12,13,14,16,17,19,20,21,22,24,25,26,27,28,30,31,32,33,35,36,38], [20,21,22,24,25,26,27,28,30,31,32,33,35,36,38], 11, 7, 2**21, 2**7),
    ]

    # print all sweep parameters
    print("Sweep parameters:")
    print(f"Model hyperparameters (model_name, squeeze_merge, layers, repeat_layers, repeat_times, logup_l, logup_m, logup_n): {model_hyper_param}")
    print(f"Hard window sizes: {hard_window_sizes}")
    print(f"Top n exponents: {capture_y_sram_size_exp_list}")
    print(f"Onchip MLE sizes: {onchip_mle_sizes_list}")
    print(f"DRAM bandwidth (B/cycle): {DRAM_bandwidth_B_cycle_list}")
    print(f"Inverse unit counts: {inv_unit_list}")
    print(f"Sumcheck PE counts: {sumcheck_pes_list}")
    print(f"Eval engine counts: {eval_engines_list}")
    print(f"Product lane counts: {product_lanes_list}")
    print(f"Elements per SRAM feed to DP: {elements_per_sram_feed_to_dp_list}")
    print(f"Sumcheck Vg sizes: {sc_Vg_sizes_list}")
    print(f"Compiler options: {compiler_option_list}")
    print(f"Build MLE throughput per cycle and DP mul pairs: {build_mle_throughput_per_cycle_dp_mul_list}")
    
    config_iter = list(
        itertools.product(
            hard_window_sizes,
            capture_y_sram_size_exp_list,
            onchip_mle_sizes_list,
            DRAM_bandwidth_B_cycle_list,
            inv_unit_list,
            sumcheck_pes_list,
            eval_engines_list,
            product_lanes_list,
            elements_per_sram_feed_to_dp_list,
            sc_Vg_sizes_list,
            compiler_option_list,
            build_mle_throughput_per_cycle_dp_mul_list,
        )
    )
    print(f"Config combinations per model: {len(config_iter)}")
    print(f"Total planned tasks before skip check: {len(config_iter) * len(model_hyper_param)}")

    total_planned_tasks = 0
    total_skipped_tasks = 0
    total_submitted_tasks = 0
    total_completed_tasks = 0

    for (
        model_name,
        squeeze_merge,
        layers,
        repeat_layers,
        repeat_times,
        logup_l,
        logup_m,
        logup_n,
    ) in model_hyper_param:
        model_tasks = []
        for (
            hard_window_size,
            capture_y_sram_size_exp,
            num_onchip_mle_sizes,
            DRAM_bandwidth_B_cycle,
            num_inv_unit,
            sumcheck_pes,
            eval_engines,
            product_lanes,
            num_elements_per_sram_feed_to_dp,
            num_sc_Vg_sizes,
            compiler_option,
            build_dp_pair,
        ) in config_iter:
            build_mle_throughput_per_cycle, num_dp_mul = build_dp_pair
            model_tasks.append(
                {
                    "model_name": model_name,
                    "squeeze_merge": squeeze_merge,
                    "layers": layers,
                    "repeat_layers": repeat_layers,
                    "repeat_times": repeat_times,
                    "logup_l": logup_l,
                    "logup_m": logup_m,
                    "logup_n": logup_n,
                    "hard_window_size": hard_window_size,
                    "capture_y_sram_size_exp": capture_y_sram_size_exp,
                    "num_onchip_mle_sizes": num_onchip_mle_sizes,
                    "DRAM_bandwidth_B_cycle": DRAM_bandwidth_B_cycle,
                    "num_inv_unit": num_inv_unit,
                    "sumcheck_pes": sumcheck_pes,
                    "eval_engines": eval_engines,
                    "product_lanes": product_lanes,
                    "num_elements_per_sram_feed_to_dp": num_elements_per_sram_feed_to_dp,
                    "num_sc_Vg_sizes": num_sc_Vg_sizes,
                    "compiler_option": compiler_option,
                    "build_mle_throughput_per_cycle": build_mle_throughput_per_cycle,
                    "num_dp_mul": num_dp_mul,
                    "script_dir": str(script_dir),
                    "repo_root": str(repo_root),
                }
            )

        total_planned_tasks += len(model_tasks)
        print(f"Prepared {len(model_tasks)} tasks for model {model_name} with squeeze_merge {squeeze_merge}.")
        if args.multiprocess:
            pending_tasks = []
            for task in model_tasks:
                dump_path = _get_dump_path(task)
                if dump_path.exists():
                    print(f"Skipping existing sweep result {dump_path}")
                    total_skipped_tasks += 1
                    continue
                pending_tasks.append(task)
            total_submitted_tasks += len(pending_tasks)
            print(
                f"{model_name}: planned={len(model_tasks)}, "
                f"skip_existing={len(model_tasks) - len(pending_tasks)}, "
                f"to_run={len(pending_tasks)}"
            )

            with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
                futures = [executor.submit(_run_single_sweep, task) for task in pending_tasks]
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Sweeping {model_name}"):
                    result = future.result()
                    total_completed_tasks += 1
                    print(f"Dumped sweep result to {result['dump_path']}")
        else:
            pending_count = 0
            for task in tqdm(model_tasks, desc=f"Sweeping {model_name}"):
                dump_path = _get_dump_path(task)
                if dump_path.exists():
                    print(f"Skipping existing sweep result {dump_path}")
                    total_skipped_tasks += 1
                    continue
                pending_count += 1
                total_submitted_tasks += 1
                result = _run_single_sweep(task)
                total_completed_tasks += 1
                print(f"Dumped sweep result to {result['dump_path']}")
            print(
                f"{model_name}: planned={len(model_tasks)}, "
                f"skip_existing={len(model_tasks) - pending_count}, "
                f"to_run={pending_count}"
            )

    total_elapsed_sec = time.time() - sweep_start_time
    print(
        "Sweep summary: "
        f"planned={total_planned_tasks}, "
        f"skipped={total_skipped_tasks}, "
        f"submitted={total_submitted_tasks}, "
        f"completed={total_completed_tasks}, "
        f"elapsed_sec={total_elapsed_sec:.2f}, "
        f"elapsed_min={total_elapsed_sec / 60:.2f}, "
        f"elapsed_hr={total_elapsed_sec / 3600:.2f}"
    )
    print("model_sweep.py end.")
