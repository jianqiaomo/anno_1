import json
from pathlib import Path
from tqdm import tqdm
from hardware_experiments.sumcheck_NTT_sweep import sweep_sumcheck_configs_wo_fz
from GKR_gates_phase1_model import (
    build_mle_cost_input_not_2_pow,
    append_task,
    merge_continuous_easy_range_tasks,
)


def _load_layer_data(comp_data_dir, model_name, layer, hard_window_sizes, top_n_exp):
    layer_json_path = (
        Path(comp_data_dir)
        / f"{model_name}_layer_{layer}_hardwins_{hard_window_sizes}_topn_{top_n_exp}.json"
    )
    with layer_json_path.open("r", encoding="utf-8") as f:
        # print(f"Loaded layer data from: {layer_json_path}")
        return json.load(f), str(layer_json_path)


def _load_lasso_unique_count(comp_data_dir):
    unique_count_path = Path(comp_data_dir) / "lasso_unique_u_v_count.json"
    with unique_count_path.open("r", encoding="utf-8") as f:
        unique_count_data = json.load(f)
        # print(f"Loaded unique count data from: {unique_count_path}")
    return int(unique_count_data["total_unique_count"]), str(unique_count_path)


def phase_last_round_1_sumcheck_latency(
    comp_data_dir,
    layers,
    repeat_layers,
    repeat_times,
    hard_window_sizes,
    top_n_exp,
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
    Scheduling model for the last phase over all requested layers.
    `compiler_option=1` enables the compiler-aware compressed loading/build path.

    It loads each layer JSON from `comp_data_dir`, processes `u_hu` and `v_hv`
    on shared queues, repeats the specified layers `repeat_times - 1` extra
    times, and finally appends the last sumcheck/store-window stage using
    `lasso_unique_u_v_count.json`.
    """
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
    if num_sc_Vg_sizes < 0:
        raise ValueError("num_sc_Vg_sizes must be non-negative")
    if compiler_option not in (0, 1):
        raise ValueError("compiler_option must be 0 or 1")

    comp_data_dir = Path(comp_data_dir)
    model_name = comp_data_dir.parent.name
    squeeze_merge = comp_data_dir.name
    effective_num_dp_mul = min(num_elements_per_sram_feed_to_dp, num_dp_mul)

    compute_queue = []
    load_store_queue = []

    layer_schedule = list(layers) + list(repeat_layers) * max(repeat_times - 1, 0)
    processed_layers = []

    def process_relation(layer_data, relation_key, x_name, y_name, task_prefix):
        relation_dict = layer_data["lasso"][relation_key]
        x_easy_range_list = relation_dict.get(f"{x_name}_easy", [])
        x_hard_window_list = relation_dict.get(f"{x_name}_hard_window", [])
        x_hard_window_captured_y_count = relation_dict.get(
            f"{x_name}_hard_window_captured_{y_name}_count",
            [],
        )
        x_hard_window_uncaptured_y_count = relation_dict.get(
            f"{x_name}_hard_window_uncaptured_{y_name}_count",
            [],
        )
        x_hard_window_captured_unique_y_count = relation_dict.get(
            f"{x_name}_hard_window_captured_unique_{y_name}_count",
            [],
        )
        x_hard_window_final_size = relation_dict.get(f"{x_name}_hard_window_final_size")
        total_easy_y_count = relation_dict.get(f"total_{x_name}_easy_{y_name}_count", 0)
        total_y_count = relation_dict.get(f"total_{y_name}_count", 0)
        average_easy_y_count_per_x = (
            total_easy_y_count / sum(end - start for start, end in x_easy_range_list)
            if x_easy_range_list else 0
        )
        all_hard_uncapture_unq_y = relation_dict.get(
            f"total_{x_name}_hard_uncaptured_unique_{y_name}_count",
            0,
        )
        all_hard_uncapture_build_y = all_hard_uncapture_unq_y

        # Skip scheduling entirely when this relation has no y-values.
        if total_y_count == 0:
            return {
                "relation_key": relation_key,
                "x_name": x_name,
                "y_name": y_name,
                "total_y_count": total_y_count,
                "total_easy_y_count": total_easy_y_count,
                "range_tasks": [],
                "all_hard_uncapture_unq_y": all_hard_uncapture_unq_y,
                "all_hard_uncapture_build_y": all_hard_uncapture_build_y,
                "compute_queue_end_cycle": compute_queue[-1]["end_cycle"] if compute_queue else 0,
                "load_store_queue_end_cycle": load_store_queue[-1]["end_cycle"] if load_store_queue else 0,
                "Total_latency_cycle": 0,
            }

        if compiler_option == 1:
            total_min_y = relation_dict.get(f"total_min_{y_name}")
            total_max_y = relation_dict.get(f"total_max_{y_name}")
            if total_min_y is None or total_max_y is None:
                raise ValueError(f"Missing total_min_{y_name} or total_max_{y_name} for compiler_option=1")
            all_hard_uncapture_build_y = max(total_max_y - total_min_y, 0)

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

        build_all_hard_uncaptured_y_cycle = (
            build_mle_cost_input_not_2_pow(
                total_elements=all_hard_uncapture_build_y,
                build_mle_throughput_per_cycle=build_mle_throughput_per_cycle,
            )["total_cycles"]
            if all_hard_uncapture_build_y > 0 else 0
        )
        store_all_hard_uncaptured_y_cycle = int(
            (all_hard_uncapture_build_y * 32) / DRAM_bandwidth_B_cycle
        )
        append_task(
            compute_queue,
            f"{task_prefix}_{relation_key}_build_all_hard_uncaptured_y",
            compute_queue[-1]["end_cycle"] if compute_queue else 0,
            build_all_hard_uncaptured_y_cycle,
        )
        append_task(
            load_store_queue,
            f"{task_prefix}_{relation_key}_store_all_hard_uncaptured_y",
            load_store_queue[-1]["end_cycle"] if load_store_queue else 0,
            store_all_hard_uncaptured_y_cycle,
        )

        for task in range_tasks:
            if task["kind"] == "hard" and task["range_size"] > 1:
                if compiler_option == 1:
                    task["uncaptured_y_count"] += task["captured_y_count"]
                    task["captured_y_count"] = 0
                    task["captured_unique_y_count"] = 0
                captured_unique_y_count = task["captured_unique_y_count"]
                build_captured_bh_cycle = (
                    build_mle_cost_input_not_2_pow(
                        total_elements=captured_unique_y_count,
                        build_mle_throughput_per_cycle=build_mle_throughput_per_cycle,
                    )["total_cycles"]
                    if captured_unique_y_count > 0 else 0
                )
                build_task = append_task(
                    compute_queue,
                    f"{task_prefix}_{relation_key}_{task['start']}_{task['end']}_build_captured_bh_full_hard_window",
                    compute_queue[-1]["end_cycle"] if compute_queue else 0,
                    build_captured_bh_cycle,
                    metadata={"range": task["range"]},
                )

                avg_hard_window_total_y = int(
                    (task["captured_y_count"] + task["uncaptured_y_count"]) / task["range_size"]
                )
                avg_hard_window_uncaptured_y = int(task["uncaptured_y_count"] / task["range_size"])
                load_uncaptured_y_full_window_cycle = int(
                    avg_hard_window_uncaptured_y * 32 / DRAM_bandwidth_B_cycle * task["range_size"]
                )
                shared_start = max(
                    build_task["end_cycle"],
                    load_store_queue[-1]["end_cycle"] if load_store_queue else 0,
                )
                lookahead_cycle = num_sc_Vg_sizes * 32 / DRAM_bandwidth_B_cycle
                load_start = max(
                    shared_start - lookahead_cycle,
                    load_store_queue[-1]["end_cycle"] if load_store_queue else 0,
                )
                append_task(
                    load_store_queue,
                    f"{task_prefix}_{relation_key}_{task['start']}_{task['end']}_load_uncaptured_y_full_hard_window",
                    load_start,
                    load_uncaptured_y_full_window_cycle,
                    metadata={
                        "range": task["range"],
                        "avg_hard_window_total_y": avg_hard_window_total_y,
                        "avg_hard_window_uncaptured_y": avg_hard_window_uncaptured_y,
                        "num_sc_Vg_sizes": num_sc_Vg_sizes,
                        "lookahead_cycle": lookahead_cycle,
                    },
                )
                append_task(
                    compute_queue,
                    f"{task_prefix}_{relation_key}_{task['start']}_{task['end']}_sumup",
                    max(shared_start, compute_queue[-1]["end_cycle"] if compute_queue else 0),
                    int((avg_hard_window_total_y.bit_length() - 1) * task["range_size"]),
                    metadata={"range": task["range"]},
                )
            elif task["kind"] == "hard" and task["range_size"] == 1:
                total_y_count = task["captured_y_count"] + task["uncaptured_y_count"]
                if compiler_option == 1:
                    build_task_end = compute_queue[-1]["end_cycle"] if compute_queue else 0
                    load_bh_cycle = (total_y_count * 32) / DRAM_bandwidth_B_cycle
                    load_task = append_task(
                        load_store_queue,
                        f"{task_prefix}_{relation_key}_{task['start']}_{task['end']}_load_bh_one_point_hard_window",
                        load_store_queue[-1]["end_cycle"] if load_store_queue else 0,
                        load_bh_cycle,
                        metadata={"range": task["range"], "total_y_count": total_y_count},
                    )
                else:
                    build_task = append_task(
                        compute_queue,
                        f"{task_prefix}_{relation_key}_{task['start']}_{task['end']}_build_ez_bh_one_point_hard_window",
                        compute_queue[-1]["end_cycle"] if compute_queue else 0,
                        build_mle_cost_input_not_2_pow(
                            total_elements=total_y_count,
                            build_mle_throughput_per_cycle=build_mle_throughput_per_cycle,
                        )["total_cycles"] if total_y_count > 0 else 0,
                        metadata={"range": task["range"], "total_y_count": total_y_count},
                    )
                    build_task_end = build_task["end_cycle"]
                append_task(
                    compute_queue,
                    f"{task_prefix}_{relation_key}_{task['start']}_{task['end']}_sumup_one_point_hard_window",
                    max(
                        build_task_end,
                        load_task["end_cycle"] if compiler_option == 1 else 0,
                    ),
                    int((total_y_count.bit_length() - 1) * task["range_size"]),
                    metadata={"range": task["range"], "total_y_count": total_y_count},
                )
            elif task["kind"] == "easy":
                total_easy_y_count_in_range = int(average_easy_y_count_per_x * task["range_size"])
                if compiler_option == 1:
                    append_task(
                        load_store_queue,
                        f"{task_prefix}_{relation_key}_{task['start']}_{task['end']}_load_uncap_bh_easy_window",
                        load_store_queue[-1]["end_cycle"] if load_store_queue else 0,
                        total_easy_y_count_in_range * 32 / DRAM_bandwidth_B_cycle,
                        metadata={
                            "range": task["range"],
                            "total_easy_y_count_in_range": total_easy_y_count_in_range,
                        },
                    )
                else:
                    append_task(
                        compute_queue,
                        f"{task_prefix}_{relation_key}_{task['start']}_{task['end']}_build_ez_bh_easy_window",
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
            else:
                raise ValueError(f"Unknown task kind: {task['kind']}")

        return {
            "relation_key": relation_key,
            "x_name": x_name,
            "y_name": y_name,
            "range_tasks": range_tasks,
            "all_hard_uncapture_unq_y": all_hard_uncapture_unq_y,
        }

    for schedule_idx, layer in tqdm(enumerate(layer_schedule), desc="Scheduling layers"):
        layer_data, layer_json_path = _load_layer_data(
            comp_data_dir,
            model_name,
            layer,
            hard_window_sizes,
            top_n_exp,
        )
        u_hu_result = process_relation(layer_data, "u_hu", "u", "hu", f"layer{layer}_run{schedule_idx}")
        v_hv_result = process_relation(layer_data, "v_hv", "v", "hv", f"layer{layer}_run{schedule_idx}")
        processed_layers.append(
            {
                "layer": layer,
                "schedule_index": schedule_idx,
                "layer_json_path": layer_json_path,
                "u_hu_result": u_hu_result,
                "v_hv_result": v_hv_result,
            }
        )

    total_task_length, unique_count_path = _load_lasso_unique_count(comp_data_dir)

    store_start = max(
        compute_queue[-1]["end_cycle"] if compute_queue else 0,
        load_store_queue[-1]["end_cycle"] if load_store_queue else 0,
    )
    store_window_mle_cycle = total_task_length * 32 / DRAM_bandwidth_B_cycle
    append_task(
        load_store_queue,
        "final_store_window_mle",
        store_start,
        store_window_mle_cycle,
        metadata={
            "total_task_length": total_task_length,
            "store_window_mle_cycle": store_window_mle_cycle,
        },
    )

    if total_task_length > 0:
        sumcheck_sweep_df = sweep_sumcheck_configs_wo_fz(
            num_var_list=[total_task_length.bit_length() - 1],
            available_bw_list=[1e30],
            polynomial_list=[[["g1", "g2"]]],
            sweep_sumcheck_pes_range=[sumcheck_pes],
            sweep_eval_engines_range=[eval_engines],
            sweep_product_lanes_range=[product_lanes],
            sweep_onchip_mle_sizes_range=[num_onchip_mle_sizes],
            no_rd1_prefetch=True,
        )
        if sumcheck_sweep_df.empty:
            raise ValueError("sweep_sumcheck_configs_wo_fz returned an empty DataFrame")
        final_sumcheck_latency_cycle = sumcheck_sweep_df.iloc[0]["round_latencies"][0]
    else:
        final_sumcheck_latency_cycle = 0
    append_task(
        compute_queue,
        "final_sumcheck_r1",
        compute_queue[-1]["end_cycle"] if compute_queue else 0,
        final_sumcheck_latency_cycle,
        metadata={
            "total_task_length": total_task_length,
            "num_var": total_task_length.bit_length() - 1 if total_task_length > 0 else 0,
            "final_sumcheck_latency_cycle": final_sumcheck_latency_cycle,
        },
    )

    total_latency_cycle = max(
        compute_queue[-1]["end_cycle"] if compute_queue else 0,
        load_store_queue[-1]["end_cycle"] if load_store_queue else 0,
    )

    result_return = {
        "inputs": {
            "comp_data_dir": str(comp_data_dir),
            "model_name": model_name,
            "squeeze_merge": squeeze_merge,
            "layers": layers,
            "repeat_layers": repeat_layers,
            "repeat_times": repeat_times,
            "hard_window_sizes": hard_window_sizes,
            "top_n_exp": top_n_exp,
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
            "lasso_unique_count_path": unique_count_path,
        },
        "effective_num_dp_mul": effective_num_dp_mul,
        "processed_layers": processed_layers,
        "layer_schedule": layer_schedule,
        "total_task_length": total_task_length,
        "final_sumcheck_latency_cycle": final_sumcheck_latency_cycle,
        "store_window_mle_cycle": store_window_mle_cycle,
        "compute_queue": compute_queue,
        "load_store_queue": load_store_queue,
        "Total_latency_cycle": total_latency_cycle,
    }
    return result_return


def phase_last_rest_round_sumcheck_latency(
    comp_data_dir,
    DRAM_bandwidth_B_cycle,
    num_onchip_mle_sizes,
    sumcheck_pes,
    eval_engines,
    product_lanes,
):
    """
    Estimate last-phase rest-round sumcheck latency.

    The sumcheck `num_var` is derived from `lasso_unique_u_v_count.json`.
    """
    if DRAM_bandwidth_B_cycle <= 0:
        raise ValueError("DRAM_bandwidth_B_cycle must be positive")
    if num_onchip_mle_sizes <= 0:
        raise ValueError("num_onchip_mle_sizes must be positive")

    total_task_length, unique_count_path = _load_lasso_unique_count(comp_data_dir)
    if total_task_length <= 0:
        raise ValueError("last-phase total task length must be positive")

    num_var = total_task_length.bit_length() - 1
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

    result_return = {
        "inputs": {
            "comp_data_dir": str(comp_data_dir),
            "lasso_unique_count_path": unique_count_path,
            "DRAM_bandwidth_B_cycle": DRAM_bandwidth_B_cycle,
            "num_onchip_mle_sizes": num_onchip_mle_sizes,
            "sumcheck_pes": sumcheck_pes,
            "eval_engines": eval_engines,
            "product_lanes": product_lanes,
        },
        "total_task_length": total_task_length,
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
    return result_return


def phase_last_sumcheck_all_rounds_latency(
    comp_data_dir,
    layers,
    repeat_layers,
    repeat_times,
    hard_window_sizes,
    top_n_exp,
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
    Merge last-phase round-1 and rest-round sumcheck models.
    """
    round_1_result = phase_last_round_1_sumcheck_latency(
        comp_data_dir=comp_data_dir,
        layers=layers,
        repeat_layers=repeat_layers,
        repeat_times=repeat_times,
        hard_window_sizes=hard_window_sizes,
        top_n_exp=top_n_exp,
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
    rest_round_result = phase_last_rest_round_sumcheck_latency(
        comp_data_dir=comp_data_dir,
        DRAM_bandwidth_B_cycle=DRAM_bandwidth_B_cycle,
        num_onchip_mle_sizes=num_onchip_mle_sizes,
        sumcheck_pes=sumcheck_pes,
        eval_engines=eval_engines,
        product_lanes=product_lanes,
    )

    result_return = {
        "inputs": {
            "comp_data_dir": str(comp_data_dir),
            "layers": layers,
            "repeat_layers": repeat_layers,
            "repeat_times": repeat_times,
            "hard_window_sizes": hard_window_sizes,
            "top_n_exp": top_n_exp,
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
        "sumcheck_cost": rest_round_result["sumcheck_cost"],
        "round_1_result": round_1_result,
        "rest_round_result": rest_round_result,
    }
    return result_return


if __name__ == "__main__":
    hard_window_sizes = 512
    top_n_exp = 15

    # model, model, layer, repeat layer, repeat times.
    ModelSqueeze=[
        ("gpt2-small", "SqueezeMerge_1", [1, 2, 3, 4, 6, 8], [6, 8], 24),
        ("gpt2-medium", "SqueezeMerge_1", [1, 2, 3, 4, 6, 8], [6, 8], 48),
        ("opt-125m", "SqueezeMerge_0", [1,2,3,5,6,7,8,9,11,12,13,14,16,17,19,20,21,22,24,25,26,27,28,30,31,32,33,35,36,38], [20,21,22,24,25,26,27,28,30,31,32,33,35,36,38], 11),
    ]

    for model_name, squeeze_merge, layers, repeat_layers, repeat_times in ModelSqueeze:
        comp_data_dir = Path(f"./comp_data/{model_name}/{squeeze_merge}")
        example1_0 = phase_last_sumcheck_all_rounds_latency(
            comp_data_dir=comp_data_dir,
            layers=layers,
            repeat_layers=repeat_layers,
            repeat_times=repeat_times,
            hard_window_sizes=hard_window_sizes,
            top_n_exp=top_n_exp,
            num_onchip_mle_sizes=1024,
            num_sc_Vg_sizes=128,
            compiler_option=0,
            build_mle_throughput_per_cycle=64,
            DRAM_bandwidth_B_cycle=512,
            num_dp_mul=128,
            num_elements_per_sram_feed_to_dp=128,
            sumcheck_pes=2,
            eval_engines=5,
            product_lanes=5,
        )
        example1_1 = phase_last_sumcheck_all_rounds_latency(
            comp_data_dir=comp_data_dir,
            layers=layers,
            repeat_layers=repeat_layers,
            repeat_times=repeat_times,
            hard_window_sizes=hard_window_sizes,
            top_n_exp=top_n_exp,
            num_onchip_mle_sizes=1024,
            num_sc_Vg_sizes=128,
            compiler_option=1,
            build_mle_throughput_per_cycle=64,
            DRAM_bandwidth_B_cycle=512,
            num_dp_mul=128,
            num_elements_per_sram_feed_to_dp=128,
            sumcheck_pes=2,
            eval_engines=5,
            product_lanes=5,
        )

    print("GKR_last_model.py end.")
