import math
import json
from hardware_experiments.params import modmul_latency, extensions_latency
from hardware_experiments.sumcheck_NTT_sweep import sweep_sumcheck_configs_wo_fz
from matmul_model import build_mle_cost


def build_mle_cost_input_not_2_pow(
    total_elements,
    build_mle_throughput_per_cycle,
):
    if total_elements <= 0:
        return {"total_cycles": 0}
    elif total_elements == 1:
        return {"total_cycles": modmul_latency}
    # elif total_elements < 2**15:
    # elif total_elements < 2**10:
    #     return {"total_cycles": round(total_elements / build_mle_throughput_per_cycle)}
    num_vars = int(math.ceil(math.log2(total_elements)))
    diff = 2 ** num_vars - total_elements
    cost_in_num_vars = build_mle_cost(
        num_vars=num_vars,
        build_mle_throughput_per_cycle=build_mle_throughput_per_cycle,
    )
    actual_cycles = cost_in_num_vars["total_cycles"] - math.ceil(diff / build_mle_throughput_per_cycle)
    if actual_cycles > 0:
        cost_in_num_vars["total_cycles"] = actual_cycles
    return cost_in_num_vars


def append_task(queue, name, start_cycle, duration_cycle, metadata=None):
    end_cycle = start_cycle + round(duration_cycle)
    task = {
        "name": name,
        "start_cycle": start_cycle,
        "duration_cycle": round(duration_cycle),
        "end_cycle": end_cycle,
    }
    if metadata:
        task.update(metadata)
    queue.append(task)
    return task


def merge_continuous_easy_range_tasks(range_tasks):
    """
    Merge consecutive `easy` tasks in task order.

    For any consecutive run of easy tasks, even if the original ranges are not
    contiguous:
    - new range size = sum of member range sizes
    - new end = last easy task's end
    - new start = new end - new range size

    Hard tasks are preserved as-is.
    """
    merged_tasks = []
    pending_easy = None

    for task in range_tasks:
        if task.get("kind") != "easy":
            if pending_easy is not None:
                pending_easy["start"] = pending_easy["end"] - pending_easy["range_size"]
                pending_easy["range"] = (pending_easy["start"], pending_easy["end"])
                merged_tasks.append(pending_easy)
                pending_easy = None
            merged_tasks.append(task)
            continue

        if pending_easy is None:
            pending_easy = dict(task)
            continue

        pending_easy["range_size"] += task["range_size"]
        pending_easy["end"] = task["end"]

    if pending_easy is not None:
        pending_easy["start"] = pending_easy["end"] - pending_easy["range_size"]
        pending_easy["range"] = (pending_easy["start"], pending_easy["end"])
        merged_tasks.append(pending_easy)

    return merged_tasks


def phase1_round_1_sumcheck_latency(
    x_y_data_dict,
    lv,
    num_onchip_mle_sizes,
    num_sc_Vg_sizes,
    compiler_option,
    build_mle_throughput_per_cycle,
    DRAM_bandwidth_B_cycle,
    num_dp_mul,
    num_elements_per_sram_feed_to_dp,
    sumcheck_pes,
    eval_engines,
    product_lanes,
    x_name="u",
    y_name="g",
):
    """
    Scheduling model for GKR phase-1 round-1 sumcheck latency.

    Returns:
        A dict containing scheduled compute/load-store queues and the final
        round-1 total latency.
    """
    x_y_data_dict = x_y_data_dict[f"{lv}"]
    x_easy_range_list = x_y_data_dict.get(f"{x_name}_easy")
    x_hard_window_list = x_y_data_dict.get(f"{x_name}_hard_window")
    x_hard_window_captured_y_count = x_y_data_dict.get(
        f"{x_name}_hard_window_captured_{y_name}_count"
    )
    x_hard_window_uncaptured_y_count = x_y_data_dict.get(
        f"{x_name}_hard_window_uncaptured_{y_name}_count"
    )
    x_hard_window_captured_unique_y_count = x_y_data_dict.get(
        f"{x_name}_hard_window_captured_unique_{y_name}_count"
    )
    x_hard_window_final_size = x_y_data_dict.get(f"{x_name}_hard_window_final_size")
    total_easy_y_count = x_y_data_dict.get(f"total_{x_name}_easy_{y_name}_count", 0)
    average_easy_y_count_per_x = total_easy_y_count / sum(end - start for start, end in x_easy_range_list) if x_easy_range_list else 0
    if average_easy_y_count_per_x > 2:
        raise ValueError(f"average_easy_y_count_per_x seems too high: {average_easy_y_count_per_x}")
    total_g_count = x_y_data_dict[f"total_{y_name}_count"]

    all_hard_uncapture_unq_y = x_y_data_dict.get(
        f"total_{x_name}_hard_uncaptured_unique_{y_name}_count", 0
    )
    if total_g_count == 0:
        return {
            "inputs": {
                "x_name": x_name,
                "y_name": y_name,
                "x_easy_range_list": x_easy_range_list,
                "x_hard_window_list": x_hard_window_list,
                "x_hard_window_captured_y_count": x_hard_window_captured_y_count,
                "x_hard_window_uncaptured_y_count": x_hard_window_uncaptured_y_count,
                "x_hard_window_captured_unique_y_count": x_hard_window_captured_unique_y_count,
                "x_hard_window_final_size": x_hard_window_final_size,
                "total_easy_y_count": total_easy_y_count,
                "total_g_count": total_g_count,
                "num_onchip_mle_sizes": num_onchip_mle_sizes,
                "compiler_option": compiler_option,
                "all_hard_uncapture_unq_y": all_hard_uncapture_unq_y,
                "build_mle_throughput_per_cycle": build_mle_throughput_per_cycle,
                "build_bg_mle_throughput_per_cycle": build_mle_throughput_per_cycle / 2,
                "DRAM_bandwidth_B_cycle": DRAM_bandwidth_B_cycle,
                "num_dp_mul": num_dp_mul,
                "num_elements_per_sram_feed_to_dp": num_elements_per_sram_feed_to_dp,
                "sumcheck_pes": sumcheck_pes,
                "eval_engines": eval_engines,
                "product_lanes": product_lanes,
            },
            "effective_num_dp_mul": min(num_elements_per_sram_feed_to_dp, num_dp_mul),
            "range_tasks": [],
            "compute_queue": [],
            "load_store_queue": [],
            "Total_latency_cycle": 0,
        }
    # filter out "1 point len" hard window
    for idx, hard_window in enumerate(x_hard_window_list):
        len_hard_window = hard_window[1] - hard_window[0]
        if len_hard_window == 1:
            this_hard_window_uncaptured_y_count = x_hard_window_uncaptured_y_count[idx]
            actual_all_hard_uncapture_y = all_hard_uncapture_unq_y - this_hard_window_uncaptured_y_count
            if actual_all_hard_uncapture_y >= 0:
                all_hard_uncapture_unq_y = actual_all_hard_uncapture_y

    if x_hard_window_final_size is None:
        raise ValueError(f"Missing key: {x_name}_hard_window_final_size")
    if len(x_hard_window_list) != len(x_hard_window_captured_y_count):
        raise ValueError("x_hard_window_list and x_hard_window_captured_y_count must have the same length")
    if len(x_hard_window_list) != len(x_hard_window_uncaptured_y_count):
        raise ValueError("x_hard_window_list and x_hard_window_uncaptured_y_count must have the same length")
    if len(x_hard_window_list) != len(x_hard_window_captured_unique_y_count):
        raise ValueError("x_hard_window_list and x_hard_window_captured_unique_y_count must have the same length")
    if build_mle_throughput_per_cycle <= 0:
        raise ValueError("build_mle_throughput_per_cycle must be positive")
    if DRAM_bandwidth_B_cycle <= 0:
        raise ValueError("DRAM_bandwidth_B_cycle must be positive")
    if num_dp_mul <= 0:
        raise ValueError("num_dp_mul must be positive")
    if num_elements_per_sram_feed_to_dp <= 0:
        raise ValueError("num_elements_per_sram_feed_to_dp must be positive")
    if num_onchip_mle_sizes <= 0:
        raise ValueError("num_onchip_mle_sizes must be positive")
    if compiler_option not in (0, 1):
        raise ValueError("compiler_option must be 0 or 1")

    effective_num_dp_mul = min(num_elements_per_sram_feed_to_dp, num_dp_mul)

    hard_window_info = []
    for window, captured_count, uncaptured_count, captured_unique_count in zip(
        x_hard_window_list,
        x_hard_window_captured_y_count,
        x_hard_window_uncaptured_y_count,
        x_hard_window_captured_unique_y_count,
    ):
        start, end = window
        hard_window_info.append(
            {
                "kind": "hard",
                "range": (start, end),
                "start": start,
                "end": end,
                "range_size": end - start,
                "captured_y_count": captured_count,
                "uncaptured_y_count": uncaptured_count,
                "captured_unique_y_count": captured_unique_count,
            }
        )

    easy_range_info = [
        {
            "kind": "easy",
            "range": (start, end),
            "start": start,
            "end": end,
            "range_size": end - start,
        }
        for start, end in x_easy_range_list
    ]

    range_tasks0 = sorted(easy_range_info + hard_window_info, key=lambda task: task["start"])
    range_tasks = merge_continuous_easy_range_tasks(range_tasks0)

    compute_queue = []
    load_store_queue = []

    all_hard_uncapture_build_count = all_hard_uncapture_unq_y
    if compiler_option == 1:
        total_min_y = x_y_data_dict.get(f"total_min_{y_name}")
        total_max_y = x_y_data_dict.get(f"total_max_{y_name}")
        if total_min_y is None or total_max_y is None:
            raise ValueError(f"Missing total_min_{y_name} or total_max_{y_name} for compiler_option=1")
        all_hard_uncapture_build_count = max(total_max_y - total_min_y, 0)

    if all_hard_uncapture_build_count > 0:
        build_all_hard_uncaptured_y_cycle = build_mle_cost_input_not_2_pow(
            total_elements=all_hard_uncapture_build_count,
            build_mle_throughput_per_cycle=build_mle_throughput_per_cycle / 2,  # for βg
        )["total_cycles"]
    else:
        build_all_hard_uncaptured_y_cycle = 0
    store_all_hard_uncaptured_y_cycle = math.ceil(all_hard_uncapture_build_count * 32 / DRAM_bandwidth_B_cycle)

    if lv == 0:
        append_task(
            compute_queue,
            "build_all_hard_uncaptured_y",
            0,
            build_all_hard_uncaptured_y_cycle,
        )
        append_task(
            load_store_queue,
            "store_all_hard_uncaptured_y",
            0,
            store_all_hard_uncaptured_y_cycle,
        )

    for idx, task in enumerate(range_tasks):
        if task["kind"] == "hard" and task["range_size"] > 1:
            if compiler_option == 1:
                task["uncaptured_y_count"] += task["captured_y_count"]
                task["captured_y_count"] = 0
                task["captured_unique_y_count"] = 0
            captured_unique_y_count = task["captured_unique_y_count"]
            if captured_unique_y_count > 0:
                build_captured_bg_cycle = build_mle_cost_input_not_2_pow(
                    total_elements=captured_unique_y_count,
                    build_mle_throughput_per_cycle=build_mle_throughput_per_cycle / 2,  # for βg
                )["total_cycles"]
            else:
                build_captured_bg_cycle = 0
            compute_start = compute_queue[-1]["end_cycle"] if compute_queue else 0
            build_task = append_task(
                compute_queue,
                f"{task['start']}_{task['end']}_build_captured_bg_full_hard_window",
                compute_start,
                build_captured_bg_cycle,
                metadata={"range": task["range"]},
            )

            avg_hard_window_total_y = int((
                task["captured_y_count"] + task["uncaptured_y_count"]
            ) / task["range_size"])
            avg_hard_window_uncaptured_y = int(task["uncaptured_y_count"] / task["range_size"])

            # sc: 4 Bytes per element, vg: 32 bytes per element
            load_sc_vg_uncap_y_cycle = (
                avg_hard_window_total_y * (4 + 32)
                + avg_hard_window_uncaptured_y * 32
            ) / DRAM_bandwidth_B_cycle
            load_sc_vg_uncap_y_full_window_cycle = int(
                load_sc_vg_uncap_y_cycle * task["range_size"]
            )

            shared_start = max(
                build_task["end_cycle"],
                load_store_queue[-1]["end_cycle"] if load_store_queue else 0,
            )
            lookahead_cycle = (
                num_sc_Vg_sizes * (4 + 32) / DRAM_bandwidth_B_cycle
            )
            load_start = max(
                shared_start - lookahead_cycle,
                load_store_queue[-1]["end_cycle"] if load_store_queue else 0,
            )
            load_task = append_task(
                load_store_queue,
                f"{task['start']}_{task['end']}_load_sc_vg_uncap_y_full_hard_window",
                load_start,
                load_sc_vg_uncap_y_full_window_cycle,
                metadata={
                    "range": task["range"],
                    "avg_hard_window_total_y": avg_hard_window_total_y,
                    "avg_hard_window_uncaptured_y": avg_hard_window_uncaptured_y,
                    "num_sc_Vg_sizes": num_sc_Vg_sizes,
                },
            )

            dot_prod_cycle = avg_hard_window_total_y / effective_num_dp_mul
            dot_prod_full_window_cycle = int(dot_prod_cycle * task["range_size"])
            dot_prod_start = shared_start
            dot_prod_task = append_task(
                compute_queue,
                f"{task['start']}_{task['end']}_dot_product",
                dot_prod_start,
                dot_prod_full_window_cycle,
                metadata={"range": task["range"]},
            )

            load_Vmul_store_hard_window_mle_cycle = math.ceil(
                2 * task["range_size"] * 32 / DRAM_bandwidth_B_cycle
            )
            append_task(
                load_store_queue,
                f"{task['start']}_{task['end']}_load_Vmul_store_hard_window_mle",
                load_store_queue[-1]["end_cycle"] if load_store_queue else 0,
                load_Vmul_store_hard_window_mle_cycle,
                metadata={"range": task["range"]},
            )

            sumcheck_sweep_df = sweep_sumcheck_configs_wo_fz(
                num_var_list=[task["range_size"].bit_length() - 1],
                available_bw_list=[1e7],
                polynomial_list=[[["g1", "g2"]]],
                sweep_sumcheck_pes_range=[sumcheck_pes],
                sweep_eval_engines_range=[eval_engines],
                sweep_product_lanes_range=[product_lanes],
                sweep_onchip_mle_sizes_range=[num_onchip_mle_sizes],
                no_rd1_prefetch=True,
            )
            if sumcheck_sweep_df.empty:
                raise ValueError("sweep_sumcheck_configs_wo_fz returned an empty DataFrame")
            sumcheck_r1_latency_cycle = sumcheck_sweep_df.iloc[0]["round_latencies"][0]

            sumcheck_start = max(load_task["end_cycle"], dot_prod_task["end_cycle"])
            append_task(
                compute_queue,
                f"{task['start']}_{task['end']}_sumcheck_r1",
                sumcheck_start,
                sumcheck_r1_latency_cycle,
                metadata={
                    "range": task["range"],
                    "sumcheck_r1_latency_cycle": sumcheck_r1_latency_cycle,
                },
            )
        elif task["kind"] == "hard" and task["range_size"] == 1:
            total_y_count = task["captured_y_count"] + task["uncaptured_y_count"]
            if total_y_count > 0 and compiler_option != 1:
                build_ez_bg_cycle = build_mle_cost_input_not_2_pow(
                    total_elements=total_y_count,
                    build_mle_throughput_per_cycle=build_mle_throughput_per_cycle / 2,  # for βg
                )["total_cycles"]
            else:
                build_ez_bg_cycle = 0

            compute_start = compute_queue[-1]["end_cycle"] if compute_queue else 0
            build_task = append_task(
                compute_queue,
                f"{task['start']}_{task['end']}_build_ez_bg_one_point_hard_window",
                compute_start,
                build_ez_bg_cycle,
                metadata={"range": task["range"]},
            )

            load_start = load_store_queue[-1]["end_cycle"] if load_store_queue else 0
            if compiler_option == 1:
                load_sc_vg_cycle = math.ceil(
                    total_y_count * (4 + 32 * 2) / DRAM_bandwidth_B_cycle
                )
            else:
                load_sc_vg_cycle = math.ceil(
                    total_y_count * (4 + 32) / DRAM_bandwidth_B_cycle
                )
            load_task = append_task(
                load_store_queue,
                f"{task['start']}_{task['end']}_load_sc_vg_one_point_hard_window",
                load_start,
                load_sc_vg_cycle,
                metadata={
                    "range": task["range"],
                    "total_y_count": total_y_count,
                },
            )

            # Build MLE immediately feed to DP, not SRAM.
            dot_prod_cycle = int(math.ceil(total_y_count / num_dp_mul))
            append_task(
                compute_queue,
                f"{task['start']}_{task['end']}_dot_product_one_point_hard_window",
                max(build_task["end_cycle"], load_task["end_cycle"]),
                dot_prod_cycle,
                metadata={
                    "range": task["range"],
                    "total_y_count": total_y_count,
                },
            )
        elif task["kind"] == "easy":
            total_easy_y_count_in_range = int(average_easy_y_count_per_x * task["range_size"])
            if total_easy_y_count_in_range > 0 and compiler_option != 1:
                build_ez_bg_cycle = build_mle_cost_input_not_2_pow(
                    total_elements=total_easy_y_count_in_range,
                    build_mle_throughput_per_cycle=build_mle_throughput_per_cycle / 2,  # for βg
                )["total_cycles"]
            else:
                build_ez_bg_cycle = 0

            compute_start = compute_queue[-1]["end_cycle"] if compute_queue else 0
            build_task = append_task(
                compute_queue,
                f"{task['start']}_{task['end']}_build_ez_bg_easy_window",
                compute_start,
                build_ez_bg_cycle,
                metadata={
                    "range": task["range"],
                    "total_easy_y_count_in_range": total_easy_y_count_in_range,
                },
            )

            load_start = load_store_queue[-1]["end_cycle"] if load_store_queue else 0
            if compiler_option == 1:
                load_sc_vg_cycle = math.ceil(
                    total_easy_y_count_in_range * (4 + 32 * 2) / DRAM_bandwidth_B_cycle
                )
            else:
                load_sc_vg_cycle = math.ceil(
                    total_easy_y_count_in_range * (4 + 32) / DRAM_bandwidth_B_cycle
                )
            load_task = append_task(
                load_store_queue,
                f"{task['start']}_{task['end']}_load_sc_vg_easy_window",
                load_start,
                load_sc_vg_cycle,
                metadata={
                    "range": task["range"],
                    "total_easy_y_count_in_range": total_easy_y_count_in_range,
                },
            )

            sumcheck_sweep_df = sweep_sumcheck_configs_wo_fz(
                num_var_list=[task["range_size"].bit_length() - 1],
                available_bw_list=[1e7],
                polynomial_list=[[["g1", "g2"]]],
                sweep_sumcheck_pes_range=[sumcheck_pes],
                sweep_eval_engines_range=[eval_engines],
                sweep_product_lanes_range=[product_lanes],
                sweep_onchip_mle_sizes_range=[num_onchip_mle_sizes],
                no_rd1_prefetch=True,
            )
            if sumcheck_sweep_df.empty:
                raise ValueError("sweep_sumcheck_configs_wo_fz returned an empty DataFrame")
            sumcheck_r1_latency_cycle = sumcheck_sweep_df.iloc[0]["round_latencies"][0]

            append_task(
                compute_queue,
                f"{task['start']}_{task['end']}_sumcheck_r1_easy_window",
                max(build_task["end_cycle"], load_task["end_cycle"]),
                sumcheck_r1_latency_cycle,
                metadata={
                    "range": task["range"],
                    "sumcheck_r1_latency_cycle": sumcheck_r1_latency_cycle,
                },
            )

            load_Vmul_store_easy_window_mle_cycle = math.ceil(
                2 * task["range_size"] * 32 / DRAM_bandwidth_B_cycle
            )
            append_task(
                load_store_queue,
                f"{task['start']}_{task['end']}_load_Vmul_store_easy_window_mle",
                load_store_queue[-1]["end_cycle"] if load_store_queue else 0,
                load_Vmul_store_easy_window_mle_cycle,
                metadata={"range": task["range"]},
            )
        else:
            raise ValueError(f"Unknown task kind: {task['kind']}")

    total_latency_cycle = max(
        compute_queue[-1]["end_cycle"] if compute_queue else 0,
        load_store_queue[-1]["end_cycle"] if load_store_queue else 0,
    )

    return_result = {
        "inputs": {
            "x_name": x_name,
            "y_name": y_name,
            "x_easy_range_list": x_easy_range_list,
            "x_hard_window_list": x_hard_window_list,
            "x_hard_window_captured_y_count": x_hard_window_captured_y_count,
            "x_hard_window_uncaptured_y_count": x_hard_window_uncaptured_y_count,
            "x_hard_window_captured_unique_y_count": x_hard_window_captured_unique_y_count,
            "x_hard_window_final_size": x_hard_window_final_size,
            "total_easy_y_count": total_easy_y_count,
            "num_onchip_mle_sizes": num_onchip_mle_sizes,
            "compiler_option": compiler_option,
            "all_hard_uncapture_unq_y": all_hard_uncapture_unq_y,
            "all_hard_uncapture_build_count": all_hard_uncapture_build_count,
            "build_mle_throughput_per_cycle": build_mle_throughput_per_cycle,
            "build_bg_mle_throughput_per_cycle": build_mle_throughput_per_cycle / 2,
            "DRAM_bandwidth_B_cycle": DRAM_bandwidth_B_cycle,
            "num_dp_mul": num_dp_mul,
            "num_elements_per_sram_feed_to_dp": num_elements_per_sram_feed_to_dp,
            "sumcheck_pes": sumcheck_pes,
            "eval_engines": eval_engines,
            "product_lanes": product_lanes,
        },
        "effective_num_dp_mul": effective_num_dp_mul,
        "range_tasks": range_tasks,
        "compute_queue": compute_queue,
        "load_store_queue": load_store_queue,
        "Total_latency_cycle": total_latency_cycle,
    }
    return return_result


def phase1_rest_round_sumcheck_latency(
    x_y_data_dict,
    lv,
    DRAM_bandwidth_B_cycle,
    num_onchip_mle_sizes,
    sumcheck_pes,
    eval_engines,
    product_lanes,
    x_name="u",
    y_name="g",
):
    """
    Estimate phase-1 rest-round sumcheck latency from dumped easy/hard ranges.

    `num_var` is computed as:
        round(log2(sum(range lengths in x_hard and x_easy)))
    """
    x_y_data_dict = x_y_data_dict[f"{lv}"]
    x_easy_range_list = x_y_data_dict.get(f"{x_name}_easy", [])
    x_hard_range_list = x_y_data_dict.get(f"{x_name}_hard", [])

    total_easy_range_length = sum(end - start for start, end in x_easy_range_list)
    total_hard_range_length = sum(end - start for start, end in x_hard_range_list)
    total_range_length = total_easy_range_length + total_hard_range_length
    if total_range_length < 0:
        raise ValueError("total range length must be non-negative")
    if DRAM_bandwidth_B_cycle <= 0:
        raise ValueError("DRAM_bandwidth_B_cycle must be positive")
    if num_onchip_mle_sizes <= 0:
        raise ValueError("num_onchip_mle_sizes must be positive")
    if total_range_length == 0:
        return {
            "inputs": {
                "lv": lv,
                "x_name": x_name,
                "y_name": y_name,
                "DRAM_bandwidth_B_cycle": DRAM_bandwidth_B_cycle,
                "num_onchip_mle_sizes": num_onchip_mle_sizes,
                "sumcheck_pes": sumcheck_pes,
                "eval_engines": eval_engines,
                "product_lanes": product_lanes,
            },
            "total_easy_range_length": total_easy_range_length,
            "total_hard_range_length": total_hard_range_length,
            "total_range_length": total_range_length,
            "num_var": 0,
            "rest_round_latency_cycle": 0,
            "sumcheck_cost": {
                "round_latencies": [],
                "total_latency": 0,
                "area": None,
                "area_with_hbm": None,
                "modmul_count": 0,
                "design_modmul_area": None,
                "total_onchip_memory_MB": 0,
                "utilization": None,
                "per_round_utilization": None,
                "hardware_config": None,
            },
        }

    num_var = int(math.log2(total_range_length))

    sumcheck_sweep_df = sweep_sumcheck_configs_wo_fz(
        num_var_list=[num_var],
        available_bw_list=[DRAM_bandwidth_B_cycle],
        polynomial_list=[[["g1", "g2"]]],
        sweep_sumcheck_pes_range=[sumcheck_pes],
        sweep_eval_engines_range=[eval_engines],
        sweep_product_lanes_range=[product_lanes],
        sweep_onchip_mle_sizes_range=[num_onchip_mle_sizes],
        no_rd1_prefetch=True,
    )
    if sumcheck_sweep_df.empty:
        raise ValueError("sweep_sumcheck_configs_wo_fz returned an empty DataFrame")

    sumcheck_row = sumcheck_sweep_df.iloc[0].to_dict()
    round_latencies = sumcheck_row["round_latencies"]
    rest_round_latency_cycle = sum(round_latencies[1:]) if len(round_latencies) >= 2 else 0

    return {
        "inputs": {
            "lv": lv,
            "x_name": x_name,
            "y_name": y_name,
            "DRAM_bandwidth_B_cycle": DRAM_bandwidth_B_cycle,
            "num_onchip_mle_sizes": num_onchip_mle_sizes,
            "sumcheck_pes": sumcheck_pes,
            "eval_engines": eval_engines,
            "product_lanes": product_lanes,
        },
        "total_easy_range_length": total_easy_range_length,
        "total_hard_range_length": total_hard_range_length,
        "total_range_length": total_range_length,
        "num_var": num_var,
        "rest_round_latency_cycle": rest_round_latency_cycle,
        "sumcheck_cost": {
            "round_latencies": round_latencies,
            "total_latency": sumcheck_row.get("total_latency"),
            "area": sumcheck_row.get("area"),
            "area_with_hbm": sumcheck_row.get("area_with_hbm"),
            "modmul_count": sumcheck_row.get("modmul_count"),
            "design_modmul_area": sumcheck_row.get("design_modmul_area"),
            "total_onchip_memory_MB": sumcheck_row.get("total_onchip_memory_MB"),
            "utilization": sumcheck_row.get("utilization"),
            "per_round_utilization": sumcheck_row.get("per_round_utilization"),
            "hardware_config": sumcheck_row.get("hardware_config"),
        },
    }


def _phase1_sumcheck_all_rounds_latency(
    x_y_data_dict,
    lv,
    num_onchip_mle_sizes,
    num_sc_Vg_sizes,
    compiler_option,
    build_mle_throughput_per_cycle,
    DRAM_bandwidth_B_cycle,
    num_dp_mul,
    num_elements_per_sram_feed_to_dp,
    sumcheck_pes,
    eval_engines,
    product_lanes,
    x_name="u",
    y_name="g",
):
    """
    Merge phase-1 round-1 and rest-round sumcheck models.
    """
    round_1_result = phase1_round_1_sumcheck_latency(
        x_y_data_dict=x_y_data_dict,
        lv=lv,
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
        x_name=x_name,
        y_name=y_name,
    )
    rest_round_result = phase1_rest_round_sumcheck_latency(
        x_y_data_dict=x_y_data_dict,
        lv=lv,
        DRAM_bandwidth_B_cycle=DRAM_bandwidth_B_cycle,
        num_onchip_mle_sizes=num_onchip_mle_sizes,
        sumcheck_pes=sumcheck_pes,
        eval_engines=eval_engines,
        product_lanes=product_lanes,
        x_name=x_name,
        y_name=y_name,
    )

    sumcheck_cost = rest_round_result["sumcheck_cost"]

    return {
        "inputs": {
            "lv": lv,
            "x_name": x_name,
            "y_name": y_name,
            "num_onchip_mle_sizes": num_onchip_mle_sizes,
            "num_sc_Vg_sizes": num_sc_Vg_sizes,
            "compiler_option": compiler_option,
            "build_mle_throughput_per_cycle": build_mle_throughput_per_cycle,
            "DRAM_bandwidth_B_cycle": DRAM_bandwidth_B_cycle,
            "num_dp_mul": num_dp_mul,
            "num_elements_per_sram_feed_to_dp": num_elements_per_sram_feed_to_dp,
            "sumcheck_pes": sumcheck_pes,
            "eval_engines": eval_engines,
            "product_lanes": product_lanes,
        },
        "Total_latency_cycle": (
            round_1_result["Total_latency_cycle"] + rest_round_result["rest_round_latency_cycle"]
        ),
        "sumcheck_cost": sumcheck_cost,
        "round_1_result": round_1_result,
        "rest_round_result": rest_round_result,
    }


def phase1_sumcheck_all_rounds_all_lv_latency(
    json_data_dict,
    num_onchip_mle_sizes,
    num_sc_Vg_sizes,
    compiler_option,
    build_mle_throughput_per_cycle,
    DRAM_bandwidth_B_cycle,
    num_dp_mul,
    num_elements_per_sram_feed_to_dp,
    sumcheck_pes,
    eval_engines,
    product_lanes,
):
    """
    Sum phase-1 all-rounds latency across both `lv=0` and `lv=1`.

    Args:
        json_data_dict: Complete JSON data dictionary loaded from `compile_gate_range.py`.
        num_onchip_mle_sizes: On-chip MLE capacity used by the sumcheck sweep.
        num_sc_Vg_sizes: Number of `sc/Vg` elements that can be prefetched for the hard-window load lookahead in round 1.
        compiler_option: 0 for using compiler, 1 for no compiler.
        build_mle_throughput_per_cycle: Build-MLE throughput in elements/cycle used by the round-1 scheduling model. (βg can only achieve half)
        DRAM_bandwidth_B_cycle: Effective DRAM bandwidth in bytes/cycle used by both round-1 and rest-round models.
        num_dp_mul: Number of available datapath multipliers used by the round-1 scheduling model.
        num_elements_per_sram_feed_to_dp: SRAM feed rate into the DP array, in elements/cycle, used by the round-1 scheduling model.
        sumcheck_pes: Number of sumcheck PEs used by both round-1 and rest-round models.
        eval_engines: Number of sumcheck evaluation engines used by both round-1 and rest-round models.
        product_lanes: Number of sumcheck product lanes used by both round-1 and rest-round models.
    """
    x_y_data_dict = json_data_dict["phase1"]['unary_u_g']
    x_name = "u"
    y_name = "g"
    lv0_result = _phase1_sumcheck_all_rounds_latency(
        x_y_data_dict=x_y_data_dict,
        lv=0,
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
        x_name=x_name,
        y_name=y_name,
    )
    lv1_result = _phase1_sumcheck_all_rounds_latency(
        x_y_data_dict=x_y_data_dict,
        lv=1,
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
        x_name=x_name,
        y_name=y_name,
    )

    return {
        "inputs": {
            "x_name": x_name,
            "y_name": y_name,
            "num_onchip_mle_sizes": num_onchip_mle_sizes,
            "num_sc_Vg_sizes": num_sc_Vg_sizes,
            "compiler_option": compiler_option,
            "build_mle_throughput_per_cycle": build_mle_throughput_per_cycle,
            "DRAM_bandwidth_B_cycle": DRAM_bandwidth_B_cycle,
            "num_dp_mul": num_dp_mul,
            "num_elements_per_sram_feed_to_dp": num_elements_per_sram_feed_to_dp,
            "sumcheck_pes": sumcheck_pes,
            "eval_engines": eval_engines,
            "product_lanes": product_lanes,
        },
        "Total_latency_cycle": (
            lv0_result["Total_latency_cycle"] + lv1_result["Total_latency_cycle"]
        ),
        "lv0_result": lv0_result,
        "lv1_result": lv1_result,
    }


if __name__ == "__main__":
    hard_window_sizes = 512
    capture_y_sram_size_exp = 15

    # model, layer and repeat times.
    # ModelSqueeze=[
    #     ("gpt2-small", "SqueezeMerge_1", [1, 2, 3, 4, 6, 8], [6, 8], 24),
    #     ("gpt2-medium", "SqueezeMerge_1", [1, 2, 3, 4, 6, 8], [6, 8], 48),
    #     ("opt-125m", "SqueezeMerge_0", [1,2,3,5,6,7,8,9,11,12,13,14,16,17,19,20,21,22,24,25,26,27,28,30,31,32,33,35,36,38], [20,21,22,24,25,26,27,28,30,31,32,33,35,36,38], 11),
    # ]
    ModelSqueeze=[
        ("gpt2-small", "SqueezeMerge_1", 6),
    ]

    layer_json_path = f"./comp_data/{ModelSqueeze[0][0]}/{ModelSqueeze[0][1]}/{ModelSqueeze[0][0]}_layer_{ModelSqueeze[0][2]}_hardwins_{hard_window_sizes}_topn_{capture_y_sram_size_exp}.json"
    with open(layer_json_path, "r") as f:
        layer_data = json.load(f)

    phase1_all_0 = phase1_sumcheck_all_rounds_all_lv_latency(
        json_data_dict=layer_data,
        num_onchip_mle_sizes=1024,
        num_sc_Vg_sizes=128,
        compiler_option=0,
        build_mle_throughput_per_cycle=128,
        DRAM_bandwidth_B_cycle=512,
        num_dp_mul=128,
        num_elements_per_sram_feed_to_dp=128,
        sumcheck_pes=8,
        eval_engines=5,
        product_lanes=5,
    )

    phase1_all_1 = phase1_sumcheck_all_rounds_all_lv_latency(
        json_data_dict=layer_data,
        num_onchip_mle_sizes=1024,
        num_sc_Vg_sizes=128,
        compiler_option=1,
        build_mle_throughput_per_cycle=128,
        DRAM_bandwidth_B_cycle=512,
        num_dp_mul=128,
        num_elements_per_sram_feed_to_dp=128,
        sumcheck_pes=8,
        eval_engines=5,
        product_lanes=5,
    )

    print("GKR_gates_phase1_model.py end.")
