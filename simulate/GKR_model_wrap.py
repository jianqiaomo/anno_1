import math
import json
from pathlib import Path
from tqdm import tqdm
from GKR_gates_phase1_model import phase1_sumcheck_all_rounds_all_lv_latency
from GKR_gates_phase2_model import phase2_sumcheck_all_rounds_all_lv_latency
from GKR_last_model import phase_last_sumcheck_all_rounds_latency
from matmul_model import mm_sumcheck_all_rounds_latency


def model_gkr_phase12_one_layer_latency(
    model_name,
    squeeze_merge,
    layer,
    hard_window_sizes,
    capture_y_sram_size_exp,
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
    data_root="./comp_data",
):
    layer_json_path = (
        Path(data_root)
        / model_name
        / squeeze_merge
        / f"{model_name}_layer_{layer}_hardwins_{hard_window_sizes}_topn_{capture_y_sram_size_exp}.json"
    )
    with layer_json_path.open("r", encoding="utf-8") as f:
        layer_data = json.load(f)

    phase1_result = phase1_sumcheck_all_rounds_all_lv_latency(
        json_data_dict=layer_data,
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
    phase2_result = phase2_sumcheck_all_rounds_all_lv_latency(
        json_data_dict=layer_data,
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

    total_latency_cycle = (
        phase1_result["Total_latency_cycle"] + phase2_result["Total_latency_cycle"]
    )
    return {
        "model_name": model_name,
        "squeeze_merge": squeeze_merge,
        "layer": layer,
        "layer_json_path": str(layer_json_path),
        "phase1_result": phase1_result,
        "phase2_result": phase2_result,
        "Total_latency_cycle": total_latency_cycle,
    }


def model_gkr_gates_phase12_all_layers_latency(
    model_name,
    squeeze_merge,
    layers,
    repeat_layers,
    repeat_times,
    hard_window_sizes,
    capture_y_sram_size_exp,
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
    data_root="./comp_data",
):
    """
    Estimate total phase-1 plus phase-2 GKR gate latency across all requested
    layers for one model configuration.

    The function runs the per-layer phase-1/phase-2 wrapper once for each layer
    in `layers`, stores every layer's latency, and then adds extra copies for
    `repeat_layers` so those layers execute exactly `repeat_times` in total.
    """
    layer_results = []
    layer_latency_map = {}

    for layer in tqdm(layers, desc=f"Processing layers for {model_name}"):
        result = model_gkr_phase12_one_layer_latency(
            model_name=model_name,
            squeeze_merge=squeeze_merge,
            layer=layer,
            hard_window_sizes=hard_window_sizes,
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
            data_root=data_root,
        )
        layer_results.append(result)
        layer_latency_map[layer] = result["Total_latency_cycle"]

    base_total_latency_cycle = sum(
        result["Total_latency_cycle"] for result in layer_results
    )
    repeat_layers_one_pass_latency_cycle = sum(
        layer_latency_map[layer] for layer in repeat_layers if layer in layer_latency_map
    )
    additional_repeat_latency_cycle = repeat_layers_one_pass_latency_cycle * max(repeat_times - 1, 0)

    result_return = {
        "model_name": model_name,
        "squeeze_merge": squeeze_merge,
        "layers": layers,
        "repeat_layers": repeat_layers,
        "repeat_times": repeat_times,
        "compiler_option": compiler_option,
        "layer_results": layer_results,
        "layer_latency_map": layer_latency_map,
        "base_total_latency_cycle": base_total_latency_cycle,
        "repeat_layers_one_pass_latency_cycle": repeat_layers_one_pass_latency_cycle,
        "additional_repeat_latency_cycle": additional_repeat_latency_cycle,
        "Total_latency_cycle": base_total_latency_cycle + additional_repeat_latency_cycle,
    }
    return result_return


def _load_matmul_layer_shapes(log_path):
    """
    Load `(n, m, k)` shapes from a `*_layers_fconn.log` table.
    """
    log_path = Path(log_path)
    shape_rows = []
    with log_path.open("r", encoding="utf-8") as f:
        header_seen = False
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if not header_seen:
                header_seen = True
                continue
            parts = stripped.split()
            if len(parts) < 8:
                continue
            shape_rows.append(
                {
                    "i": int(parts[0]),
                    "now_fc_id": int(parts[1]),
                    "n": int(parts[2]),
                    "fc_input_row": int(parts[3]),
                    "m": int(parts[4]),
                    "fc_row": int(parts[5]),
                    "k": int(parts[6]),
                    "fc_col": int(parts[7]),
                }
            )
    return shape_rows


def model_matmul_all_layers_latency(
    model_name,
    squeeze_merge,
    build_mle_throughput_per_cycle,
    DRAM_bandwidth_B_cycle,
    num_onchip_mle_sizes,
    num_dp_mul,
    num_elements_per_sram_feed_to_dp,
    sumcheck_pes,
    eval_engines,
    product_lanes,
    output_root="../output",
):
    """
    Estimate all-layer matmul latency by grouping repeated `(n, m, k)` shapes
    from the `*_layers_fconn.log` file and reusing one model evaluation per
    unique shape.
    """
    log_path = (
        Path(output_root)
        / model_name
        / squeeze_merge
        / f"{model_name}_layers_fconn.log"
    )
    shape_rows = _load_matmul_layer_shapes(log_path)
    if not shape_rows:
        raise ValueError(f"No matmul layer shapes found in {log_path}")

    shape_count_map = {}
    shape_rows_map = {}
    for row in shape_rows:
        shape = (row["n"], row["m"], row["k"])
        shape_count_map[shape] = shape_count_map.get(shape, 0) + 1
        shape_rows_map.setdefault(shape, []).append(row)

    unique_shape_results = {}
    for shape, repeat_count in tqdm(
        shape_count_map.items(),
        desc=f"Processing matmul shapes for {model_name}",
    ):
        n, m, k = shape
        model_result = mm_sumcheck_all_rounds_latency(
            n=n,
            m=m,
            k=k,
            build_mle_throughput_per_cycle=build_mle_throughput_per_cycle,
            DRAM_bandwidth_B_cycle=DRAM_bandwidth_B_cycle,
            num_onchip_mle_sizes=num_onchip_mle_sizes,
            num_dp_mul=num_dp_mul,
            num_elements_per_sram_feed_to_dp=num_elements_per_sram_feed_to_dp,
            sumcheck_pes=sumcheck_pes,
            eval_engines=eval_engines,
            product_lanes=product_lanes,
        )
        unique_shape_results[shape] = {
            "n": n,
            "m": m,
            "k": k,
            "repeat_count": repeat_count,
            "rows": shape_rows_map[shape],
            "model_result": model_result,
            "total_latency_cycle": model_result["Total_latency_cycle"] * repeat_count,
        }

    total_latency_cycle = sum(
        shape_info["total_latency_cycle"]
        for shape_info in unique_shape_results.values()
    )

    layer_results = []
    for row in shape_rows:
        shape = (row["n"], row["m"], row["k"])
        layer_results.append(
            {
                **row,
                "Total_latency_cycle": unique_shape_results[shape]["model_result"]["Total_latency_cycle"],
            }
        )

    result_return = {
        "model_name": model_name,
        "squeeze_merge": squeeze_merge,
        "log_path": str(log_path),
        "num_layers": len(shape_rows),
        "shape_count_map": {
            f"{shape[0]}_{shape[1]}_{shape[2]}": count
            for shape, count in shape_count_map.items()
        },
        "unique_shape_results": {
            f"{shape[0]}_{shape[1]}_{shape[2]}": result
            for shape, result in unique_shape_results.items()
        },
        "layer_results": layer_results,
        "Total_latency_cycle": total_latency_cycle,
    }
    return result_return


if __name__ == "__main__":
    hard_window_sizes = [256, 512, 1024, 2048]
    top_ns = [13, 14, 15, 16, 17]  # these are exponents, so actual top_n will be 2^13, 2^14, etc.
    hard_window_sizes = hard_window_sizes[0]
    capture_y_sram_size_exp = top_ns[0]

    # model, model, layer, repeat layer, repeat times.
    model_hyper_param = [
        ("gpt2-small", "SqueezeMerge_1", [1, 2, 3, 4, 6, 8], [6, 8], 24),
        ("gpt2-medium", "SqueezeMerge_1", [1, 2, 3, 4, 6, 8], [6, 8], 48),
        ("opt-125m", "SqueezeMerge_0", [1,2,3,5,6,7,8,9,11,12,13,14,16,17,19,20,21,22,24,25,26,27,28,30,31,32,33,35,36,38], [20,21,22,24,25,26,27,28,30,31,32,33,35,36,38], 11),
    ]

    example_model = model_hyper_param[0]
    example_result = model_gkr_gates_phase12_all_layers_latency(
        model_name=example_model[0],
        squeeze_merge=example_model[1],
        layers=example_model[2],
        repeat_layers=example_model[3],
        repeat_times=example_model[4],
        hard_window_sizes=hard_window_sizes,
        capture_y_sram_size_exp=capture_y_sram_size_exp,
        num_onchip_mle_sizes=1024,
        num_sc_Vg_sizes=128,
        compiler_option=0,
        build_mle_throughput_per_cycle=32,
        DRAM_bandwidth_B_cycle=1024,
        num_dp_mul=256,
        num_elements_per_sram_feed_to_dp=256,
        sumcheck_pes=8,
        eval_engines=5,
        product_lanes=5,
    )

    matmul_result = model_matmul_all_layers_latency(
        model_name=example_model[0],
        squeeze_merge=example_model[1],
        build_mle_throughput_per_cycle=32,
        DRAM_bandwidth_B_cycle=1024,
        num_onchip_mle_sizes=1024,
        num_dp_mul=256,
        num_elements_per_sram_feed_to_dp=256,
        sumcheck_pes=8,
        eval_engines=5,
        product_lanes=5,
        output_root="../output",
    )
    
    comp_data_dir = Path(f"./comp_data/{example_model[0]}/{example_model[1]}")
    example1_0 = phase_last_sumcheck_all_rounds_latency(
            comp_data_dir=comp_data_dir,
            layers=example_model[2],
            repeat_layers=example_model[3],
            repeat_times=example_model[4],
            hard_window_sizes=hard_window_sizes,
            top_n_exp=capture_y_sram_size_exp,
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

    print("GKR_model_wrap.py end.")
