import json
from pathlib import Path
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator
from software_baseline_latency import (
    load_software_latency_breakdown,
    categorize_jm_timer_latency,
)
from area_latency_breakdown import RUNTIME_COLORS


GLOBAL_PLOT_FONTSIZE = 12



RUNTIME_COMPONENTS = [
    ("logup_latency", "Range Proof"),
    ("phase12_all_layers_latency", "GKR Gate Layers"),
    ("matmul_all_layers_latency", "Matmul Layers"),
    ("last_latency", "GKR Combine"),
]
RUNTIME_TOTAL_KEY = "total_latency_with_commit_open_ns"
RUNTIME_TO_CPU_CATEGORY = {
    "logup_latency": "logup prove",
    "phase12_all_layers_latency": "GKR nonFC layer sumcheck",
    "matmul_all_layers_latency": "GKR matmul layer sumcheck",
    "last_latency": "GKR last sumcheck",
}
MODEL_NAME_TO_LABEL = {
    "gpt2-small": "GPT2-Small",
    "gpt2-medium": "GPT2-Medium",
    "opt-125m": "OPT-125M",
}


def _load_hardware_latency_breakdown(json_path):
    with Path(json_path).open("r", encoding="utf-8") as f:
        return json.load(f)["latency_breakdown_ns"]


def _build_model_design_locations(model_name, squeeze_merge, hardware_design_locations_gpt2_small):
    model_design_locations = {}
    for bw, path in hardware_design_locations_gpt2_small.items():
        path_str = str(path).replace("gpt2-small", model_name).replace("SqueezeMerge_1", squeeze_merge)
        model_design_locations[bw] = Path(path_str)
    return model_design_locations


def plot_component_speedup_grouped_by_model_and_bw(
    model_hyper_param,
    hardware_design_locations_gpt2_small,
    output_path="./plots/speedup_on_steps.png",
    figsize=(8, 3),
    group_spacing=0.55,
    bar_width=0.12,
    minor_y_tick_subdivisions=2,
    use_log_scale=False,
):
    cpu_speedups = {}
    for model_name, squeeze_merge in model_hyper_param:
        cpu_breakdown_ms = categorize_jm_timer_latency(
            load_software_latency_breakdown(model_name, squeeze_merge)
        )
        cpu_speedups[(model_name, squeeze_merge)] = cpu_breakdown_ms

    bandwidths = list(hardware_design_locations_gpt2_small.keys())
    component_keys = [component_key for component_key, _ in RUNTIME_COMPONENTS]
    component_labels = [component_label for _, component_label in RUNTIME_COMPONENTS]
    cluster_span = bar_width * len(component_keys)
    effective_group_spacing = max(group_spacing, cluster_span + bar_width * 0.8)

    group_centers = []
    component_speedups = {component_key: [] for component_key in component_keys}

    x_cursor = 0.0
    model_labels = []
    model_centers = []
    bandwidth_minor_positions = []
    bandwidth_minor_labels = []

    for model_name, squeeze_merge in model_hyper_param:
        model_design_locations = _build_model_design_locations(
            model_name=model_name,
            squeeze_merge=squeeze_merge,
            hardware_design_locations_gpt2_small=hardware_design_locations_gpt2_small,
        )
        cpu_breakdown_ms = cpu_speedups[(model_name, squeeze_merge)]
        model_start_idx = len(group_centers)

        for bw in bandwidths:
            hw_breakdown_ns = _load_hardware_latency_breakdown(model_design_locations[bw])
            group_centers.append(x_cursor)
            bandwidth_minor_positions.append(x_cursor)
            bandwidth_minor_labels.append(str(bw))

            for component_key in component_keys:
                cpu_key = RUNTIME_TO_CPU_CATEGORY[component_key]
                cpu_latency_ms = cpu_breakdown_ms[cpu_key]
                hw_latency_ms = hw_breakdown_ns[component_key] / 1_000_000
                component_speedups[component_key].append(cpu_latency_ms / hw_latency_ms if hw_latency_ms > 0 else 0)

            x_cursor += effective_group_spacing

        model_end_idx = len(group_centers) - 1
        model_centers.append((group_centers[model_start_idx] + group_centers[model_end_idx]) / 2)
        model_labels.append(MODEL_NAME_TO_LABEL.get(model_name, model_name))

    fig, ax = plt.subplots(figsize=figsize)
    colors = RUNTIME_COLORS[:len(component_keys)]
    offsets = np.linspace(
        -bar_width * (len(component_keys) - 1) / 2,
        bar_width * (len(component_keys) - 1) / 2,
        len(component_keys),
    )

    for idx, component_key in enumerate(component_keys):
        xs = [center + offsets[idx] for center in group_centers]
        ax.bar(
            xs,
            component_speedups[component_key],
            width=bar_width,
            color=colors[idx],
            label=component_labels[idx],
            edgecolor="black",
            linewidth=0.4,
        )

    ax.set_ylabel("Speedup", fontsize=GLOBAL_PLOT_FONTSIZE)
    ax.set_xticks(bandwidth_minor_positions)
    ax.set_xticklabels(bandwidth_minor_labels, fontsize=GLOBAL_PLOT_FONTSIZE + 1)
    ax.tick_params(axis="x", which="major", pad=2, length=0, labelsize=GLOBAL_PLOT_FONTSIZE + 1)
    ax.yaxis.set_minor_locator(AutoMinorLocator(minor_y_tick_subdivisions))
    ax.tick_params(axis="y", which="major", labelsize=GLOBAL_PLOT_FONTSIZE, labelrotation=0)
    ax.tick_params(axis="y", which="minor", labelleft=False)
    # ax.set_title("Component Speedup by Model and Bandwidth", fontsize=GLOBAL_PLOT_FONTSIZE)
    if use_log_scale:
        ax.set_yscale("log")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.grid(which="minor", axis="y", linestyle=":", alpha=0.15, linewidth=0.5)
    ax.legend(ncol=2, fontsize=GLOBAL_PLOT_FONTSIZE, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.05))
    for center, label in zip(model_centers, model_labels):
        ax.text(
            center,
            -0.1,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=GLOBAL_PLOT_FONTSIZE + 1,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return Path(output_path)



if __name__ == "__main__":

    model_hyper_param = [
        ("gpt2-small", "SqueezeMerge_1"),
        ("gpt2-medium", "SqueezeMerge_1"),
        ("opt-125m", "SqueezeMerge_0"),
    ]

    # hardware_design_locations_gpt2_small = {
    #     "2048": Path("./sim_data/latency_area_result/gpt2-small/SqueezeMerge_1/compile_0/bw_2048/build_thput_128_dp_mul_256/MSM_pe_32_hardwins_256_topn_17_onchip_4096_inv_4_sumcheck_pes_32_eval_2_product_4_sramfeed_512_scvg_128.json"),
    #     "1024": Path("./sim_data/latency_area_result/gpt2-small/SqueezeMerge_1/compile_0/bw_1024/build_thput_32_dp_mul_64/MSM_pe_32_hardwins_256_topn_15_onchip_4096_inv_4_sumcheck_pes_16_eval_2_product_3_sramfeed_512_scvg_128.json"),
    #     "512": Path("./sim_data/latency_area_result/gpt2-small/SqueezeMerge_1/compile_0/bw_512/build_thput_32_dp_mul_64/MSM_pe_8_hardwins_256_topn_15_onchip_4096_inv_4_sumcheck_pes_16_eval_2_product_3_sramfeed_512_scvg_128.json"),
    # }
    hardware_design_locations_gpt2_small = {
        "512": Path("./sim_data/latency_area_result/gpt2-small/SqueezeMerge_1/compile_0/bw_512/build_thput_32_dp_mul_64/MSM_pe_32_hardwins_256_topn_15_onchip_4096_inv_4_sumcheck_pes_16_eval_2_product_3_sramfeed_512_scvg_128.json"),
        "1024": Path("./sim_data/latency_area_result/gpt2-small/SqueezeMerge_1/compile_0/bw_1024/build_thput_32_dp_mul_64/MSM_pe_32_hardwins_256_topn_15_onchip_4096_inv_4_sumcheck_pes_16_eval_2_product_3_sramfeed_512_scvg_128.json"),
        "2048": Path("./sim_data/latency_area_result/gpt2-small/SqueezeMerge_1/compile_0/bw_2048/build_thput_32_dp_mul_64/MSM_pe_32_hardwins_256_topn_15_onchip_4096_inv_4_sumcheck_pes_16_eval_2_product_3_sramfeed_512_scvg_128.json"),
    }

    plot_path = plot_component_speedup_grouped_by_model_and_bw(
        model_hyper_param=model_hyper_param,
        hardware_design_locations_gpt2_small=hardware_design_locations_gpt2_small,
        output_path="./plots/speedup_on_steps.pdf",
        figsize=(6, 3),
        group_spacing=0.45,
        bar_width=0.1,
        use_log_scale=True,
    )
    print(f"Saved speedup plot to {plot_path}")

    
    print("speedup_on_steps.py end.")
