import math
import json
from hardware_experiments.params import modmul_latency, extensions_latency
from hardware_experiments.sumcheck_NTT_sweep import sweep_sumcheck_configs_wo_fz
from matmul_model import build_mle_cost
from GKR_gates_phase1_model import build_mle_cost_input_not_2_pow, append_task, merge_continuous_easy_range_tasks


def build_easy_hard_range_info(relation_dict, x_name, y_name):
    x_easy_range_list = relation_dict.get(f"{x_name}_easy", [])
    x_easy_avg_unique_y_per_x = relation_dict.get(
        f"{x_name}_easy_avg_unique_{y_name}_per_{x_name}",
        [],
    )
    x_hard_window_list = relation_dict.get(f"{x_name}_hard_window", [])
    x_hard_window_captured_y_count = relation_dict.get(
        f"{x_name}_hard_window_captured_{y_name}_count", []
    )
    x_hard_window_uncaptured_y_count = relation_dict.get(
        f"{x_name}_hard_window_uncaptured_{y_name}_count", []
    )
    x_hard_window_captured_unique_y_count = relation_dict.get(
        f"{x_name}_hard_window_captured_unique_{y_name}_count", []
    )

    if len(x_hard_window_list) != len(x_hard_window_captured_y_count):
        raise ValueError(f"{x_name}_hard_window and captured count length mismatch")
    if len(x_hard_window_list) != len(x_hard_window_uncaptured_y_count):
        raise ValueError(f"{x_name}_hard_window and uncaptured count length mismatch")
    if len(x_hard_window_list) != len(x_hard_window_captured_unique_y_count):
        raise ValueError(f"{x_name}_hard_window and captured unique count length mismatch")
    if len(x_easy_range_list) != len(x_easy_avg_unique_y_per_x):
        raise ValueError(f"{x_name}_easy and easy avg unique count length mismatch")

    easy_range_info = [
        {
            "kind": "easy",
            "range": (start, end),
            "start": start,
            "end": end,
            "range_size": end - start,
            "avg_unique_y_per_x": avg_unique_y_per_x,
        }
        for (start, end), avg_unique_y_per_x in zip(
            x_easy_range_list,
            x_easy_avg_unique_y_per_x,
        )
    ]
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

    return sorted(easy_range_info + hard_window_info, key=lambda task: task["start"])


def _split_range_task(task, split_size):
    if split_size <= 0 or split_size >= task["range_size"]:
        raise ValueError("split_size must be between 1 and range_size-1")
    start = task["start"]
    mid = start + split_size
    end = task["end"]

    left = dict(task)
    left["range"] = (start, mid)
    left["start"] = start
    left["end"] = mid
    left["range_size"] = split_size

    right = dict(task)
    right["range"] = (mid, end)
    right["start"] = mid
    right["end"] = end
    right["range_size"] = end - mid

    if task["kind"] == "hard" and task["range_size"] > 0:
        left_fraction = split_size / task["range_size"]
        left["captured_y_count"] = int(task["captured_y_count"] * left_fraction)
        left["uncaptured_y_count"] = int(task["uncaptured_y_count"] * left_fraction)
        left["captured_unique_y_count"] = int(task["captured_unique_y_count"] * left_fraction)
        right["captured_y_count"] = task["captured_y_count"] - left["captured_y_count"]
        right["uncaptured_y_count"] = task["uncaptured_y_count"] - left["uncaptured_y_count"]
        right["captured_unique_y_count"] = task["captured_unique_y_count"] - left["captured_unique_y_count"]

    return left, right


def _merge_continuous_easy_range_tasks(range_tasks):
    merged_tasks = []
    pending_easy = None

    for task in range_tasks:
        v_u_task = task["v_u_task"]
        if v_u_task["kind"] != "easy":
            if pending_easy is not None:
                pending_easy_v_u = pending_easy["v_u_task"]
                pending_easy_v_u["start"] = pending_easy_v_u["end"] - pending_easy_v_u["range_size"]
                pending_easy_v_u["range"] = (
                    pending_easy_v_u["start"],
                    pending_easy_v_u["end"],
                )
                merged_tasks.append(pending_easy)
                pending_easy = None
            merged_tasks.append(task)
            continue

        if pending_easy is None:
            pending_easy = {
                "v_u_task": dict(v_u_task),
                "v_g_tasks": list(task["v_g_tasks"]),
            }
            continue

        pending_easy["v_u_task"]["range_size"] += v_u_task["range_size"]
        pending_easy["v_u_task"]["end"] = v_u_task["end"]
        pending_easy["v_g_tasks"].extend(task["v_g_tasks"])

    if pending_easy is not None:
        pending_easy_v_u = pending_easy["v_u_task"]
        pending_easy_v_u["start"] = pending_easy_v_u["end"] - pending_easy_v_u["range_size"]
        pending_easy_v_u["range"] = (
            pending_easy_v_u["start"],
            pending_easy_v_u["end"],
        )
        merged_tasks.append(pending_easy)

    return merged_tasks


def phase2_v_gu_round_1_sumcheck_latency(
    v_gu_data_dict,
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
):
    """
    Build phase-2 scheduling tasks for the `v-g/u` relation pair.
    `compiler_option=1` enables the compiler-aware compressed loading/build path.

    Input structure expectation:
    - `v_gu_data_dict["binary_v_g"][str(lv)]`
    - `v_gu_data_dict["binary_v_u"][str(lv)]`

    The actual compute/load-store scheduling logic is intentionally left as
    `pass` for later implementation.
    """
    if build_mle_throughput_per_cycle <= 0:
        raise ValueError("build_mle_throughput_per_cycle must be positive")
    if DRAM_bandwidth_B_cycle <= 0:
        raise ValueError("DRAM_bandwidth_B_cycle must be positive")
    if num_dp_mul <= 0:
        raise ValueError("num_dp_mul must be positive")
    if num_sc_Vg_sizes < 0:
        raise ValueError("num_sc_Vg_sizes must be non-negative")
    if num_elements_per_sram_feed_to_dp <= 0:
        raise ValueError("num_elements_per_sram_feed_to_dp must be positive")
    if compiler_option not in (0, 1):
        raise ValueError("compiler_option must be 0 or 1")

    effective_num_dp_mul = min(num_elements_per_sram_feed_to_dp, num_dp_mul)

    binary_v_g = v_gu_data_dict["binary_v_g"][f"{lv}"]
    binary_v_u = v_gu_data_dict["binary_v_u"][f"{lv}"]
    total_g_count = binary_v_g.get("total_g_count", binary_v_g.get("total_g", 0))
    total_u_count = binary_v_u.get("total_u_count", binary_v_u.get("total_u", 0))

    if total_g_count == 0 or total_u_count == 0:
        return {
            "inputs": {
                "lv": lv,
                "compiler_option": compiler_option,
                "build_mle_throughput_per_cycle": build_mle_throughput_per_cycle,
                "DRAM_bandwidth_B_cycle": DRAM_bandwidth_B_cycle,
                "num_dp_mul": num_dp_mul,
                "num_elements_per_sram_feed_to_dp": num_elements_per_sram_feed_to_dp,
                "sumcheck_pes": sumcheck_pes,
                "eval_engines": eval_engines,
                "product_lanes": product_lanes,
                "num_onchip_mle_sizes": num_onchip_mle_sizes,
                "num_sc_Vg_sizes": num_sc_Vg_sizes,
                "total_g_count": total_g_count,
                "total_u_count": total_u_count,
            },
            "effective_num_dp_mul": effective_num_dp_mul,
            "v_u_info_range_sort": [],
            "v_g_info_range_sort": [],
            "range_tasks": [],
            "compute_queue": [],
            "load_store_queue": [],
            "Total_latency_cycle": 0,
        }

    all_hard_uncapture_uniq_u = binary_v_u["total_v_hard_uncaptured_unique_u_count"]
    all_hard_uncapture_build_u = all_hard_uncapture_uniq_u
    if compiler_option == 1:
        total_min_u = binary_v_u.get("total_min_u")
        total_max_u = binary_v_u.get("total_max_u")
        if total_min_u is None or total_max_u is None:
            raise ValueError("Missing total_min_u or total_max_u for compiler_option=1")
        all_hard_uncapture_build_u = max(total_max_u - total_min_u, 0)

    v_g_info_range_sort = build_easy_hard_range_info(binary_v_g, "v", "g")
    v_u_info_range_sort = build_easy_hard_range_info(binary_v_u, "v", "u")

    range_tasks = []
    v_g_cursor = 0
    v_g_remainder = None

    # build tasks based on v (v-u range)
    for v_u_task in v_u_info_range_sort:
        target_size = v_u_task["range_size"]
        matched_v_g_ranges = []
        matched_size = 0

        while matched_size < target_size:
            using_v_g_remainder = v_g_remainder is not None
            current_v_g_task = v_g_remainder if using_v_g_remainder else (
                v_g_info_range_sort[v_g_cursor] if v_g_cursor < len(v_g_info_range_sort) else None
            )
            if current_v_g_task is None:
                raise ValueError("Not enough v_g ranges to match v_u ranges")

            remaining_needed = target_size - matched_size
            if current_v_g_task["range_size"] <= remaining_needed:
                matched_v_g_ranges.append(current_v_g_task)
                matched_size += current_v_g_task["range_size"]
                if using_v_g_remainder:
                    v_g_remainder = None
                else:
                    v_g_cursor += 1
            else:
                if current_v_g_task["kind"] == "hard":
                    left_task, right_task = _split_range_task(
                        current_v_g_task,
                        remaining_needed,
                    )
                    original_range_size = current_v_g_task["range_size"]
                    left_task["captured_y_count"] = int(
                        current_v_g_task["captured_y_count"]
                        * left_task["range_size"]
                        / original_range_size
                    )
                    left_task["uncaptured_y_count"] = int(
                        current_v_g_task["uncaptured_y_count"]
                        * left_task["range_size"]
                        / original_range_size
                    )
                    left_task["captured_unique_y_count"] = current_v_g_task[
                        "captured_unique_y_count"
                    ]
                    right_task["captured_y_count"] = int(
                        current_v_g_task["captured_y_count"]
                        * right_task["range_size"]
                        / original_range_size
                    )
                    right_task["uncaptured_y_count"] = int(
                        current_v_g_task["uncaptured_y_count"]
                        * right_task["range_size"]
                        / original_range_size
                    )
                    right_task["captured_unique_y_count"] = 0
                else:
                    left_task, right_task = _split_range_task(
                        current_v_g_task,
                        remaining_needed,
                    )
                matched_v_g_ranges.append(left_task)
                matched_size += left_task["range_size"]
                v_g_remainder = right_task
                if not using_v_g_remainder:
                    v_g_cursor += 1

        range_tasks.append(
            {
                "v_u_task": v_u_task,
                "v_g_tasks": matched_v_g_ranges,
            }
        )
    # sanity check for the generated range tasks
    for idx, task in enumerate(range_tasks):
        v_u_task = task["v_u_task"]
        v_g_tasks = task["v_g_tasks"]
        if not v_g_tasks:
            raise ValueError(f"No matched v_g tasks for v_u task {v_u_task}")

        v_g_start = v_g_tasks[0]["start"]
        v_g_end = v_g_tasks[-1]["end"]
        if v_u_task["start"] != v_g_start or v_u_task["end"] != v_g_end:
            raise ValueError(
                "Mismatched v_u/v_g task boundaries: "
                f"v_u=({v_u_task['start']}, {v_u_task['end']}), "
                f"v_g_combined=({v_g_start}, {v_g_end})"
            )
    range_tasks = _merge_continuous_easy_range_tasks(range_tasks)

    compute_queue = []
    load_store_queue = []

    build_all_hard_uncaptured_u_cycle = round(
        all_hard_uncapture_build_u / build_mle_throughput_per_cycle
        if all_hard_uncapture_build_u > 0 else 0
    )
    store_all_hard_uncaptured_u_cycle = round(
        all_hard_uncapture_build_u * 32 / DRAM_bandwidth_B_cycle
        if all_hard_uncapture_build_u > 0 else 0
    )
    append_task(
        compute_queue,
        "build_all_hard_uncaptured_u",
        0,
        build_all_hard_uncaptured_u_cycle,
        metadata={
            "all_hard_uncapture_uniq_u": all_hard_uncapture_uniq_u,
            "all_hard_uncapture_build_u": all_hard_uncapture_build_u,
        },
    )
    append_task(
        load_store_queue,
        "store_all_hard_uncaptured_u",
        0,
        store_all_hard_uncaptured_u_cycle,
        metadata={
            "all_hard_uncapture_uniq_u": all_hard_uncapture_uniq_u,
            "all_hard_uncapture_build_u": all_hard_uncapture_build_u,
        },
    )

    # process tasks based on v-u ranges, and their corresponding v-g ranges
    for task in range_tasks:
        v_u_task = task["v_u_task"]
        v_g_tasks = task["v_g_tasks"]

        if v_u_task["kind"] == "hard" and v_u_task["range_size"] > 1:
            if compiler_option == 1:
                v_u_task["uncaptured_y_count"] += v_u_task["captured_y_count"]
                v_u_task["captured_y_count"] = 0
                v_u_task["captured_unique_y_count"] = 0
            captured_unique_y_count = v_u_task.get("captured_unique_y_count")
            build_captured_bu_cycle = (
                captured_unique_y_count / build_mle_throughput_per_cycle
                if captured_unique_y_count > 0 else 0
            )
            build_captured_bu_task = append_task(
                compute_queue,
                f"{v_u_task['start']}_{v_u_task['end']}_build_captured_bu",
                compute_queue[-1]["end_cycle"] if compute_queue else 0,
                build_captured_bu_cycle,
                metadata={"range": v_u_task["range"]},
            )

            load_all_hard_g_subtasks_cap_bg_cycle = round(
                sum(
                    subtask.get("captured_unique_y_count")
                    for subtask in v_g_tasks
                    if subtask["kind"] == "hard"
                ) * 32 / DRAM_bandwidth_B_cycle
            )
            append_task(
                load_store_queue,
                f"{v_u_task['start']}_{v_u_task['end']}_load_all_hard_g_subtasks_cap_bg",
                load_store_queue[-1]["end_cycle"] if load_store_queue else 0,
                load_all_hard_g_subtasks_cap_bg_cycle,
                metadata={"range": v_u_task["range"]},
            )

            load_all_hard_g_subtasks_uncap_bg_cycle = (
                sum(
                    subtask.get("uncaptured_y_count")
                    for subtask in v_g_tasks
                    if subtask["kind"] == "hard"
                ) * 32 / DRAM_bandwidth_B_cycle
            )
            load_all_easy_g_subtasks_bg_cycle = (
                sum(
                    subtask["range_size"] * subtask.get("avg_unique_y_per_x", 0)
                    for subtask in v_g_tasks
                    if subtask["kind"] == "easy"
                ) * 32 / DRAM_bandwidth_B_cycle
            )
            load_uncap_bu_cycle = (
                v_u_task.get("uncaptured_y_count") * 32 / DRAM_bandwidth_B_cycle
            )
            total_v_u_y_count = (
                v_u_task.get("captured_y_count")
                + v_u_task.get("uncaptured_y_count")
            )
            load_sc_cycle = total_v_u_y_count * 4 / DRAM_bandwidth_B_cycle
            load_all_other_inputs_cycle = (
                load_all_hard_g_subtasks_uncap_bg_cycle
                + load_all_easy_g_subtasks_bg_cycle
                + load_uncap_bu_cycle
                + load_sc_cycle
            )
            shared_start = max(
                load_store_queue[-1]["end_cycle"] if load_store_queue else 0,
                build_captured_bu_task["end_cycle"],
            )
            lookahead_cycle = num_sc_Vg_sizes * (4 + 32) / DRAM_bandwidth_B_cycle
            load_inputs_start = max(
                shared_start - lookahead_cycle,
                load_store_queue[-1]["end_cycle"] if load_store_queue else 0,
            )
            load_inputs_task = append_task(
                load_store_queue,
                f"{v_u_task['start']}_{v_u_task['end']}_load_uncap_bg_easy_bg_uncap_bu_sc",
                load_inputs_start,
                load_all_other_inputs_cycle,
                metadata={
                    "range": v_u_task["range"],
                    "load_all_hard_g_subtasks_uncap_bg_cycle": load_all_hard_g_subtasks_uncap_bg_cycle,
                    "load_all_easy_g_subtasks_bg_cycle": load_all_easy_g_subtasks_bg_cycle,
                    "load_uncap_bu_cycle": load_uncap_bu_cycle,
                    "load_sc_cycle": load_sc_cycle,
                    "num_sc_Vg_sizes": num_sc_Vg_sizes,
                    "lookahead_cycle": lookahead_cycle,
                },
            )

            dot_prod_cycle = total_v_u_y_count / effective_num_dp_mul
            dot_prod_task = append_task(
                compute_queue,
                f"{v_u_task['start']}_{v_u_task['end']}_dot_product",
                compute_queue[-1]["end_cycle"] if compute_queue else 0,
                dot_prod_cycle,
                metadata={"range": v_u_task["range"]},
            )

            sumcheck_sweep_df = sweep_sumcheck_configs_wo_fz(
                num_var_list=[v_u_task["range_size"].bit_length() - 1],
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
            sumcheck_task = append_task(
                compute_queue,
                f"{v_u_task['start']}_{v_u_task['end']}_sumcheck_r1",
                max(
                    compute_queue[-1]["end_cycle"] if compute_queue else 0,
                    load_inputs_task["end_cycle"],
                ),
                sumcheck_r1_latency_cycle,
                metadata={
                    "range": v_u_task["range"],
                    "sumcheck_r1_latency_cycle": sumcheck_r1_latency_cycle,
                },
            )

            load_Vmul_store_hard_window_mle_cycle = (
                2 * v_u_task["range_size"] * 32 / DRAM_bandwidth_B_cycle
            )
            append_task(
                load_store_queue,
                f"{v_u_task['start']}_{v_u_task['end']}_load_Vmul_store_hard_window_mle",
                load_store_queue[-1]["end_cycle"] if load_store_queue else 0,
                load_Vmul_store_hard_window_mle_cycle,
                metadata={"range": v_u_task["range"]},
            )
        elif v_u_task["kind"] == "hard" and v_u_task["range_size"] == 1:
            total_v_u_y_count = (
                v_u_task.get("captured_y_count")
                + v_u_task.get("uncaptured_y_count")
            )
            build_ez_bu_cycle = (
                build_mle_cost_input_not_2_pow(
                    total_elements=total_v_u_y_count,
                    build_mle_throughput_per_cycle=build_mle_throughput_per_cycle,
                )["total_cycles"]
                if total_v_u_y_count > 0 and compiler_option != 1 else 0
            )

            build_task = append_task(
                compute_queue,
                f"{v_u_task['start']}_{v_u_task['end']}_build_ez_bu_one_point_hard_window",
                compute_queue[-1]["end_cycle"] if compute_queue else 0,
                build_ez_bu_cycle,
                metadata={
                    "range": v_u_task["range"],
                    "total_v_u_y_count": total_v_u_y_count,
                },
            )

            if len(v_g_tasks) != 1:
                raise ValueError(
                    "Expected exactly one matched v_g task for one-point hard v_u task, "
                    f"got {len(v_g_tasks)}"
                )
            v_g_task = v_g_tasks[0]
            if v_g_task["kind"] == "easy":
                bu_count = 2
            elif v_g_task["kind"] == "hard":
                bu_count = v_g_task.get("uncaptured_y_count")
            else:
                raise ValueError(f"Unknown v_g task kind: {v_g_task['kind']}")

            if compiler_option == 1:
                load_sc_bu_cycle = (
                    total_v_u_y_count * (4 + 32) + bu_count * 32
                ) / DRAM_bandwidth_B_cycle
            else:
                load_sc_bu_cycle = (
                    total_v_u_y_count * 4 + bu_count * 32
                ) / DRAM_bandwidth_B_cycle
            load_task = append_task(
                load_store_queue,
                f"{v_u_task['start']}_{v_u_task['end']}_load_sc_bu_one_point_hard_window",
                load_store_queue[-1]["end_cycle"] if load_store_queue else 0,
                load_sc_bu_cycle,
                metadata={
                    "range": v_u_task["range"],
                    "bu_count": bu_count,
                    "total_v_u_y_count": total_v_u_y_count,
                    "v_g_kind": v_g_task["kind"],
                },
            )

            dot_prod_cycle = total_v_u_y_count / num_dp_mul
            append_task(
                compute_queue,
                f"{v_u_task['start']}_{v_u_task['end']}_dot_product_one_point_hard_window",
                max(build_task["end_cycle"], load_task["end_cycle"]),
                dot_prod_cycle,
                metadata={
                    "range": v_u_task["range"],
                    "total_v_u_y_count": total_v_u_y_count,
                },
            )
        elif v_u_task["kind"] == "easy":
            total_easy_u_count_in_range = int(
                v_u_task.get("avg_unique_y_per_x", 0) * v_u_task["range_size"]
            )
            build_ez_bu_cycle = (
                build_mle_cost_input_not_2_pow(
                    total_elements=total_easy_u_count_in_range,
                    build_mle_throughput_per_cycle=build_mle_throughput_per_cycle,
                )["total_cycles"]
                if total_easy_u_count_in_range > 0 and compiler_option != 1 else 0
            )

            build_ez_bu_task = append_task(
                compute_queue,
                f"{v_u_task['start']}_{v_u_task['end']}_build_ez_bu_easy_window",
                compute_queue[-1]["end_cycle"] if compute_queue else 0,
                build_ez_bu_cycle,
                metadata={
                    "range": v_u_task["range"],
                    "total_easy_u_count_in_range": total_easy_u_count_in_range,
                },
            )

            load_all_hard_g_subtasks_cap_bg_cycle = round(
                sum(
                    subtask.get("captured_unique_y_count")
                    for subtask in v_g_tasks
                    if subtask["kind"] == "hard"
                ) * 32 / DRAM_bandwidth_B_cycle
            )
            append_task(
                load_store_queue,
                f"{v_u_task['start']}_{v_u_task['end']}_load_all_hard_g_subtasks_cap_bg_easy_window",
                load_store_queue[-1]["end_cycle"] if load_store_queue else 0,
                load_all_hard_g_subtasks_cap_bg_cycle,
                metadata={"range": v_u_task["range"]},
            )

            load_all_hard_g_subtasks_uncap_bg_cycle = (
                sum(
                    subtask.get("uncaptured_y_count")
                    for subtask in v_g_tasks
                    if subtask["kind"] == "hard"
                ) * 32 / DRAM_bandwidth_B_cycle
            )
            load_all_easy_g_subtasks_bg_cycle = (
                sum(
                    subtask["range_size"] * subtask.get("avg_unique_y_per_x", 0)
                    for subtask in v_g_tasks
                    if subtask["kind"] == "easy"
                ) * 32 / DRAM_bandwidth_B_cycle
            )
            if compiler_option == 1:
                load_uncap_bu_cycle = total_easy_u_count_in_range * 32 / DRAM_bandwidth_B_cycle
            else:
                load_uncap_bu_cycle = 0
            load_sc_cycle = total_easy_u_count_in_range * 4 / DRAM_bandwidth_B_cycle
            load_all_other_inputs_cycle = (
                load_all_hard_g_subtasks_uncap_bg_cycle
                + load_all_easy_g_subtasks_bg_cycle
                + load_uncap_bu_cycle
                + load_sc_cycle
            )
            load_inputs_start = load_store_queue[-1]["end_cycle"] if load_store_queue else 0
            load_inputs_task = append_task(
                load_store_queue,
                f"{v_u_task['start']}_{v_u_task['end']}_load_uncap_bg_easy_bg_uncap_bu_sc_easy_window",
                load_inputs_start,
                load_all_other_inputs_cycle,
                metadata={
                    "range": v_u_task["range"],
                    "load_all_hard_g_subtasks_uncap_bg_cycle": load_all_hard_g_subtasks_uncap_bg_cycle,
                    "load_all_easy_g_subtasks_bg_cycle": load_all_easy_g_subtasks_bg_cycle,
                    "load_uncap_bu_cycle": load_uncap_bu_cycle,
                    "load_sc_cycle": load_sc_cycle,
                },
            )

            sumcheck_sweep_df = sweep_sumcheck_configs_wo_fz(
                num_var_list=[v_u_task["range_size"].bit_length() - 1],
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
                f"{v_u_task['start']}_{v_u_task['end']}_sumcheck_r1_easy_window",
                max(
                    compute_queue[-1]["end_cycle"] if compute_queue else 0,
                    load_inputs_task["end_cycle"],
                ),
                sumcheck_r1_latency_cycle,
                metadata={
                    "range": v_u_task["range"],
                    "sumcheck_r1_latency_cycle": sumcheck_r1_latency_cycle,
                },
            )

            load_Vmul_store_easy_window_mle_cycle = (
                2 * v_u_task["range_size"] * 32 / DRAM_bandwidth_B_cycle
            )
            append_task(
                load_store_queue,
                f"{v_u_task['start']}_{v_u_task['end']}_load_Vmul_store_easy_window_mle",
                load_store_queue[-1]["end_cycle"] if load_store_queue else 0,
                load_Vmul_store_easy_window_mle_cycle,
                metadata={"range": v_u_task["range"]},
            )
        else:
            raise ValueError(f"Unknown v_u task kind: {v_u_task['kind']}")

    total_latency_cycle = max(
        compute_queue[-1]["end_cycle"] if compute_queue else 0,
        load_store_queue[-1]["end_cycle"] if load_store_queue else 0,
    )

    return {
        "inputs": {
            "lv": lv,
            "compiler_option": compiler_option,
            "build_mle_throughput_per_cycle": build_mle_throughput_per_cycle,
            "DRAM_bandwidth_B_cycle": DRAM_bandwidth_B_cycle,
            "num_dp_mul": num_dp_mul,
            "num_elements_per_sram_feed_to_dp": num_elements_per_sram_feed_to_dp,
            "sumcheck_pes": sumcheck_pes,
            "eval_engines": eval_engines,
            "product_lanes": product_lanes,
            "v_easy_avg_unique_u_per_v": binary_v_u.get("v_easy_avg_unique_u_per_v", []),
            "v_easy_avg_unique_g_per_v": binary_v_g.get("v_easy_avg_unique_g_per_v", []),
            "all_hard_uncapture_uniq_u": all_hard_uncapture_uniq_u,
            "all_hard_uncapture_build_u": all_hard_uncapture_build_u,
        },
        "effective_num_dp_mul": effective_num_dp_mul,
        "v_u_info_range_sort": v_u_info_range_sort,
        "v_g_info_range_sort": v_g_info_range_sort,
        "range_tasks": range_tasks,
        "compute_queue": compute_queue,
        "load_store_queue": load_store_queue,
        "Total_latency_cycle": total_latency_cycle,
    }


def phase2_u_v_round_adder_latency(
    json_data_dict,
    lv,
    DRAM_bandwidth_B_cycle,
    num_dp_mul,
    num_sc_Vg_sizes,
    build_mle_throughput_per_cycle,
):
    """
    Reduced unary `u-g` round scheduler for phase 2.

    This mirrors the phase-1 unary `u/g` scheduling flow, but removes the
    round-1 sumcheck work and the `load_Vmul_store_*_mle` tasks.
    """
    x_y_data_dict = json_data_dict["phase2"]["unary_u_g"][f"{lv}"]
    x_name = "u"
    y_name = "g"

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
    total_g_count = x_y_data_dict[f"total_{y_name}_count"]
    average_easy_y_count_per_x = (
        total_easy_y_count / sum(end - start for start, end in x_easy_range_list)
        if x_easy_range_list else 0
    )
    all_hard_uncapture_unq_y = x_y_data_dict.get(
        f"total_{x_name}_hard_uncaptured_unique_{y_name}_count",
        0,
    )

    if total_g_count == 0:
        return {
            "inputs": {
                "lv": lv,
                "x_name": x_name,
                "y_name": y_name,
                "DRAM_bandwidth_B_cycle": DRAM_bandwidth_B_cycle,
                "num_dp_mul": num_dp_mul,
                "num_sc_Vg_sizes": num_sc_Vg_sizes,
                "build_mle_throughput_per_cycle": build_mle_throughput_per_cycle,
                "x_easy_range_list": x_easy_range_list,
                "x_hard_window_list": x_hard_window_list,
                "all_hard_uncapture_unq_y": all_hard_uncapture_unq_y,
                "total_g_count": total_g_count,
            },
            "range_tasks": [],
            "compute_queue": [],
            "load_store_queue": [],
            "total_dot_prod_cycle": 0,
            "Total_latency_cycle": 0,
        }

    for idx, hard_window in enumerate(x_hard_window_list):
        if hard_window[1] - hard_window[0] == 1:
            actual_all_hard_uncapture_y = (
                all_hard_uncapture_unq_y - x_hard_window_uncaptured_y_count[idx]
            )
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

    build_all_hard_uncaptured_y_cycle = (
        build_mle_cost_input_not_2_pow(
            total_elements=all_hard_uncapture_unq_y,
            build_mle_throughput_per_cycle=build_mle_throughput_per_cycle,
        )["total_cycles"]
        if all_hard_uncapture_unq_y > 0 else 0
    )
    store_all_hard_uncaptured_y_cycle = math.ceil(
        all_hard_uncapture_unq_y * 32 / DRAM_bandwidth_B_cycle
    )
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

    for task in range_tasks:
        if task["kind"] == "hard" and task["range_size"] > 1:
            captured_unique_y_count = task["captured_unique_y_count"]
            build_captured_bg_cycle = (
                build_mle_cost_input_not_2_pow(
                    total_elements=captured_unique_y_count,
                    build_mle_throughput_per_cycle=build_mle_throughput_per_cycle,
                )["total_cycles"]
                if captured_unique_y_count > 0 else 0
            )
            build_task = append_task(
                compute_queue,
                f"{task['start']}_{task['end']}_build_captured_bg_full_hard_window",
                compute_queue[-1]["end_cycle"] if compute_queue else 0,
                build_captured_bg_cycle,
                metadata={"range": task["range"]},
            )

            avg_hard_window_total_y = int(
                (task["captured_y_count"] + task["uncaptured_y_count"]) / task["range_size"]
            )
            avg_hard_window_uncaptured_y = int(task["uncaptured_y_count"] / task["range_size"])
            load_cycle = int(
                (task["uncaptured_y_count"] * 32 + task["range_size"] * (4 + 32)) / DRAM_bandwidth_B_cycle
            )
            shared_start = max(
                build_task["end_cycle"],
                load_store_queue[-1]["end_cycle"] if load_store_queue else 0,
            )
            lookahead_cycle = num_sc_Vg_sizes * (4 + 32) / DRAM_bandwidth_B_cycle
            load_start = max(
                shared_start - lookahead_cycle,
                load_store_queue[-1]["end_cycle"] if load_store_queue else 0,
            )
            load_task = append_task(
                load_store_queue,
                f"{task['start']}_{task['end']}_load_sc_bu_uncap_y_full_hard_window",
                load_start,
                load_cycle,
                metadata={
                    "range": task["range"],
                    "avg_hard_window_total_y": avg_hard_window_total_y,
                    "avg_hard_window_uncaptured_y": avg_hard_window_uncaptured_y,
                    "num_sc_Vg_sizes": num_sc_Vg_sizes,
                },
            )
            dp_task = append_task(
                compute_queue,
                f"{task['start']}_{task['end']}_dot_product",
                max(shared_start, compute_queue[-1]["end_cycle"] if compute_queue else 0),
                int(avg_hard_window_total_y / num_dp_mul * task["range_size"]),
                metadata={
                    "range": task["range"],
                    "load_end_cycle": load_task["end_cycle"],
                    "dot_prod_cycle": int(avg_hard_window_total_y / num_dp_mul * task["range_size"]),
                },
            )
        elif task["kind"] == "hard" and task["range_size"] == 1:
            total_y_count = task["captured_y_count"] + task["uncaptured_y_count"]
            build_task = append_task(
                compute_queue,
                f"{task['start']}_{task['end']}_build_ez_bg_one_point_hard_window",
                compute_queue[-1]["end_cycle"] if compute_queue else 0,
                build_mle_cost_input_not_2_pow(
                    total_elements=total_y_count,
                    build_mle_throughput_per_cycle=build_mle_throughput_per_cycle,
                )["total_cycles"] if total_y_count > 0 else 0,
                metadata={"range": task["range"], "total_y_count": total_y_count},
            )
            load_task = append_task(
                load_store_queue,
                f"{task['start']}_{task['end']}_load_sc_bu_one_point_hard_window",
                load_store_queue[-1]["end_cycle"] if load_store_queue else 0,
                round((total_y_count * 4 + 32) / DRAM_bandwidth_B_cycle),
                metadata={"range": task["range"], "total_y_count": total_y_count},
            )
            dp_task = append_task(
                compute_queue,
                f"{task['start']}_{task['end']}_dot_product_one_point_hard_window",
                max(build_task["end_cycle"], load_task["end_cycle"]),
                round(total_y_count / num_dp_mul),
                metadata={
                    "range": task["range"],
                    "total_y_count": total_y_count,
                    "dot_prod_cycle": int(math.ceil(total_y_count / num_dp_mul)),
                },
            )
        elif task["kind"] == "easy":
            total_easy_y_count_in_range = int(average_easy_y_count_per_x * task["range_size"])
            build_task = append_task(
                compute_queue,
                f"{task['start']}_{task['end']}_build_ez_bg_easy_window",
                compute_queue[-1]["end_cycle"] if compute_queue else 0,
                build_mle_cost_input_not_2_pow(
                    total_elements=total_easy_y_count_in_range,
                    build_mle_throughput_per_cycle=build_mle_throughput_per_cycle,
                )["total_cycles"] if total_easy_y_count_in_range > 0 else 0,
                metadata={
                    "range": task["range"],
                    "total_easy_y_count_in_range": total_easy_y_count_in_range,
                },
            )
            load_task = append_task(
                load_store_queue,
                f"{task['start']}_{task['end']}_load_sc_bu_easy_window",
                load_store_queue[-1]["end_cycle"] if load_store_queue else 0,
                round((total_easy_y_count_in_range * 4 + task["range_size"] * 32) / DRAM_bandwidth_B_cycle),
                metadata={
                    "range": task["range"],
                    "total_easy_y_count_in_range": total_easy_y_count_in_range,
                },
            )
            dp_task = append_task(
                compute_queue,
                f"{task['start']}_{task['end']}_dot_product_easy_window",
                max(build_task["end_cycle"], load_task["end_cycle"]),
                round(total_easy_y_count_in_range / num_dp_mul),
                metadata={
                    "range": task["range"],
                    "total_easy_y_count_in_range": total_easy_y_count_in_range,
                    "dot_prod_cycle": round(total_easy_y_count_in_range / num_dp_mul),
                },
            )
        else:
            raise ValueError(f"Unknown task kind: {task['kind']}")

    total_latency_cycle = max(
        compute_queue[-1]["end_cycle"] if compute_queue else 0,
        load_store_queue[-1]["end_cycle"] if load_store_queue else 0,
    )

    return {
        "inputs": {
            "lv": lv,
            "x_name": x_name,
            "y_name": y_name,
            "DRAM_bandwidth_B_cycle": DRAM_bandwidth_B_cycle,
            "num_dp_mul": num_dp_mul,
            "num_sc_Vg_sizes": num_sc_Vg_sizes,
            "build_mle_throughput_per_cycle": build_mle_throughput_per_cycle,
            "x_easy_range_list": x_easy_range_list,
            "x_hard_window_list": x_hard_window_list,
            "all_hard_uncapture_unq_y": all_hard_uncapture_unq_y,
        },
        "range_tasks": range_tasks,
        "compute_queue": compute_queue,
        "load_store_queue": load_store_queue,
        "total_dot_prod_cycle": sum(
            task.get("dot_prod_cycle", 0)
            for task in compute_queue
            if "dot_product" in task["name"]
        ),
        "Total_latency_cycle": total_latency_cycle,
    }


def phase2_rest_round_sumcheck_latency(
    v_gu_data_dict,
    lv,
    DRAM_bandwidth_B_cycle,
    num_onchip_mle_sizes,
    sumcheck_pes,
    eval_engines,
    product_lanes,
):
    """
    Estimate phase-2 rest-round sumcheck latency from the dumped v-u/v-g ranges.

    The rest-round variable count is derived from the total v-range length. We
    validate that the v-u and v-g sides describe the same total v length.
    """
    binary_v_g = v_gu_data_dict["binary_v_g"][f"{lv}"]
    binary_v_u = v_gu_data_dict["binary_v_u"][f"{lv}"]

    v_u_easy_range_list = binary_v_u.get("v_easy", [])
    v_u_hard_range_list = binary_v_u.get("v_hard", [])
    v_g_easy_range_list = binary_v_g.get("v_easy", [])
    v_g_hard_range_list = binary_v_g.get("v_hard", [])

    total_v_u_easy_range_length = sum(end - start for start, end in v_u_easy_range_list)
    total_v_u_hard_range_length = sum(end - start for start, end in v_u_hard_range_list)
    total_v_g_easy_range_length = sum(end - start for start, end in v_g_easy_range_list)
    total_v_g_hard_range_length = sum(end - start for start, end in v_g_hard_range_list)

    total_v_u_range_length = total_v_u_easy_range_length + total_v_u_hard_range_length
    total_v_g_range_length = total_v_g_easy_range_length + total_v_g_hard_range_length

    if total_v_u_range_length < 0:
        raise ValueError("phase2 total v_u range length must be non-negative")
    if total_v_g_range_length < 0:
        raise ValueError("phase2 total v_g range length must be non-negative")
    if total_v_u_range_length != total_v_g_range_length:
        raise ValueError(
            "phase2 total v range length mismatch between binary_v_u and binary_v_g: "
            f"{total_v_u_range_length} vs {total_v_g_range_length}"
        )
    if DRAM_bandwidth_B_cycle <= 0:
        raise ValueError("DRAM_bandwidth_B_cycle must be positive")
    if num_onchip_mle_sizes <= 0:
        raise ValueError("num_onchip_mle_sizes must be positive")
    if total_v_u_range_length == 0:
        return {
            "inputs": {
                "lv": lv,
                "DRAM_bandwidth_B_cycle": DRAM_bandwidth_B_cycle,
                "num_onchip_mle_sizes": num_onchip_mle_sizes,
                "sumcheck_pes": sumcheck_pes,
                "eval_engines": eval_engines,
                "product_lanes": product_lanes,
            },
            "total_v_u_easy_range_length": total_v_u_easy_range_length,
            "total_v_u_hard_range_length": total_v_u_hard_range_length,
            "total_v_g_easy_range_length": total_v_g_easy_range_length,
            "total_v_g_hard_range_length": total_v_g_hard_range_length,
            "total_v_range_length": total_v_u_range_length,
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

    num_var = int(math.log2(total_v_u_range_length))

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
            "DRAM_bandwidth_B_cycle": DRAM_bandwidth_B_cycle,
            "num_onchip_mle_sizes": num_onchip_mle_sizes,
            "sumcheck_pes": sumcheck_pes,
            "eval_engines": eval_engines,
            "product_lanes": product_lanes,
        },
        "total_v_u_easy_range_length": total_v_u_easy_range_length,
        "total_v_u_hard_range_length": total_v_u_hard_range_length,
        "total_v_g_easy_range_length": total_v_g_easy_range_length,
        "total_v_g_hard_range_length": total_v_g_hard_range_length,
        "total_v_range_length": total_v_u_range_length,
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


def _phase2_sumcheck_all_rounds_latency(
    v_gu_data_dict,
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
):
    """
    Merge phase-2 round-1 and rest-round sumcheck models.
    """
    round_1_result = phase2_v_gu_round_1_sumcheck_latency(
        v_gu_data_dict=v_gu_data_dict,
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
    )
    rest_round_result = phase2_rest_round_sumcheck_latency(
        v_gu_data_dict=v_gu_data_dict,
        lv=lv,
        DRAM_bandwidth_B_cycle=DRAM_bandwidth_B_cycle,
        num_onchip_mle_sizes=num_onchip_mle_sizes,
        sumcheck_pes=sumcheck_pes,
        eval_engines=eval_engines,
        product_lanes=product_lanes,
    )
    round_adder_result = phase2_u_v_round_adder_latency(
        json_data_dict={"phase2": v_gu_data_dict},
        lv=lv,
        DRAM_bandwidth_B_cycle=DRAM_bandwidth_B_cycle,
        num_dp_mul=num_dp_mul,
        num_sc_Vg_sizes=num_sc_Vg_sizes,
        build_mle_throughput_per_cycle=build_mle_throughput_per_cycle,
    )

    return {
        "inputs": {
            "lv": lv,
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
            round_1_result["Total_latency_cycle"]
            + rest_round_result["rest_round_latency_cycle"]
            + round_adder_result["Total_latency_cycle"]
        ),
        "sumcheck_cost": rest_round_result["sumcheck_cost"],
        "round_1_result": round_1_result,
        "rest_round_result": rest_round_result,
        "round_adder_result": round_adder_result,
    }


def phase2_sumcheck_all_rounds_all_lv_latency(
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
    Sum phase-2 all-rounds latency across both `lv=0` and `lv=1`.
    """
    v_gu_data_dict = json_data_dict["phase2"]

    lv0_result = _phase2_sumcheck_all_rounds_latency(
        v_gu_data_dict=v_gu_data_dict,
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
    )
    lv1_result = _phase2_sumcheck_all_rounds_latency(
        v_gu_data_dict=v_gu_data_dict,
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
    )

    return {
        "inputs": {
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
        ("gpt2-small", "SqueezeMerge_1", 2),
    ]
    
    layer_json_path = f"./comp_data/{ModelSqueeze[0][0]}/{ModelSqueeze[0][1]}/{ModelSqueeze[0][0]}_layer_{ModelSqueeze[0][2]}_hardwins_{hard_window_sizes}_topn_{capture_y_sram_size_exp}.json"
    with open(layer_json_path, "r") as f:
        layer_data = json.load(f)
    
    # phase2_1 = phase2_v_gu_round_1_sumcheck_latency(
    #     v_gu_data_dict=layer_data["phase2"],
    #     lv=0,
    #     num_onchip_mle_sizes=1024,
    #     num_sc_Vg_sizes=128,
    #     build_mle_throughput_per_cycle=32,
    #     DRAM_bandwidth_B_cycle=512,
    #     num_dp_mul=64,
    #     num_elements_per_sram_feed_to_dp=64,
    #     sumcheck_pes=2,
    #     eval_engines=5,
    #     product_lanes=5,
    # )

    # phase2_adder = phase2_u_v_round_adder_latency(
    #     json_data_dict=layer_data,
    #     lv=0,
    #     num_sc_Vg_sizes=128,
    #     DRAM_bandwidth_B_cycle=512,
    #     num_dp_mul=64,
    #     build_mle_throughput_per_cycle=32,
    # )

    phase2_all_lv_0 = phase2_sumcheck_all_rounds_all_lv_latency(
        json_data_dict=layer_data,
        num_onchip_mle_sizes=1024,
        num_sc_Vg_sizes=128,
        compiler_option=0,
        build_mle_throughput_per_cycle=32,
        DRAM_bandwidth_B_cycle=1024,
        num_dp_mul=128,
        num_elements_per_sram_feed_to_dp=128,
        sumcheck_pes=16,
        eval_engines=5,
        product_lanes=5,
    )

    phase2_all_lv_1 = phase2_sumcheck_all_rounds_all_lv_latency(
        json_data_dict=layer_data,
        num_onchip_mle_sizes=1024,
        num_sc_Vg_sizes=128,
        compiler_option=1,
        build_mle_throughput_per_cycle=32,
        DRAM_bandwidth_B_cycle=1024,
        num_dp_mul=128,
        num_elements_per_sram_feed_to_dp=128,
        sumcheck_pes=16,
        eval_engines=5,
        product_lanes=5,
    )

    print("GKR_gates_model.py end.")
