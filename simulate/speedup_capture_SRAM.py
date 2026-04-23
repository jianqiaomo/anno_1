import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator


GLOBAL_PLOT_FONTSIZE = 15


top_n_to_MB = {
    2**14: 0.5,
    2**15: 1.0,
    2**16: 2.0,
    2**17: 4.0,
}


SOFTWARE_PROOF_KEYS = (
    "logup prove",
    "GKR nonFC layer sumcheck",
    "GKR matmul layer sumcheck",
    "GKR last sumcheck",
)

TOP_N_RE = re.compile(r"topn_(\d+)")


def _model_label(model_name):
    if model_name == "gpt2-small":
        return "GPT2-Small"
    if model_name == "gpt2-medium":
        return "GPT2-Medium"
    if model_name == "opt-125m":
        return "OPT-125M"
    return model_name


def _load_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_software_proof_runtime_ms(model_name, squeeze_merge):
    path = (
        Path("./sim_data/software_baseline_latency")
        / model_name
        / squeeze_merge
        / "time_breakdown_categorized_multi_ms.json"
    )
    data = _load_json(path)
    categorized = data["time_breakdown_categorized_multi_ms"]
    return sum(float(categorized.get(key, 0.0)) for key in SOFTWARE_PROOF_KEYS)


def _replace_template_path(template_path, model_name, squeeze_merge, top_n_exp):
    path_str = str(template_path)
    path_str = path_str.replace("gpt2-small", model_name)
    path_str = path_str.replace("SqueezeMerge_1", squeeze_merge)
    path_str = TOP_N_RE.sub(f"topn_{top_n_exp}", path_str)
    return Path(path_str)


def load_hardware_proof_runtime_ms(hardware_json_path):
    data = _load_json(hardware_json_path)
    return float(data["latency_breakdown_ns"]["total_latency_proof_part"]) / 1_000_000.0


def collect_speedup_summary_for_top_n(
    model_squeeze_list,
    hardware_design_locations_gpt2_small,
    top_n_list,
):
    summary = {}
    for model_name, squeeze_merge in model_squeeze_list:
        software_runtime_ms = load_software_proof_runtime_ms(model_name, squeeze_merge)
        model_summary = {}
        for bandwidth_label, template_path in hardware_design_locations_gpt2_small.items():
            bandwidth_summary = {"software_runtime_ms": software_runtime_ms}
            for top_n_exp in top_n_list:
                top_n = 2**top_n_exp
                hardware_path = _replace_template_path(
                    template_path,
                    model_name,
                    squeeze_merge,
                    top_n_exp,
                )
                hardware_runtime_ms = load_hardware_proof_runtime_ms(hardware_path)
                speedup = (
                    software_runtime_ms / hardware_runtime_ms
                    if hardware_runtime_ms > 0
                    else 0.0
                )
                bandwidth_summary[top_n_exp] = {
                    "top_n": top_n,
                    "hardware_runtime_ms": hardware_runtime_ms,
                    "speedup": speedup,
                    "source_json": str(hardware_path),
                }
            model_summary[bandwidth_label] = bandwidth_summary
        summary.setdefault(model_name, {})[squeeze_merge] = model_summary
    return summary


def plot_speedup_by_top_n(
    speedup_summary,
    model_squeeze_list,
    bandwidth_order,
    top_n_list,
    output_path="./plots/proof_speedup_capture_sram.pdf",
    figsize=(8, 3),
    use_log_scale=False,
    minor_y_tick_subdivisions=2,
    legend_loc="upper left",
    legend_bbox_to_anchor=None,
):
    model_labels = [_model_label(model_name) for model_name, _ in model_squeeze_list]
    bar_groups = list(model_squeeze_list)

    fig, ax = plt.subplots(figsize=figsize)
    model_group_spacing = 1.18
    x_positions = [idx * model_group_spacing for idx in range(len(bar_groups))]

    bandwidth_labels = [label.replace("bw_", "") for label in bandwidth_order]
    top_n_colors = [
        mcolors.to_hex(plt.cm.viridis(value))
        for value in [0.15, 0.35, 0.6, 0.85][:len(top_n_list)][::-1]
    ]

    bandwidth_count = len(bandwidth_order)
    top_n_count = len(top_n_list)
    inter_top_n_gap = 0.008
    inter_bandwidth_gap = 0.035
    bar_width = 0.8 / (
        bandwidth_count * top_n_count
        + max(0, top_n_count - 1) * bandwidth_count * (inter_top_n_gap / 0.8)
        + max(0, bandwidth_count - 1) * (inter_bandwidth_gap / 0.8)
    )
    top_n_step = bar_width + inter_top_n_gap
    bandwidth_group_step = top_n_count * top_n_step + inter_bandwidth_gap
    total_span = bandwidth_count * top_n_count * bar_width
    total_span += bandwidth_count * max(0, top_n_count - 1) * inter_top_n_gap
    total_span += max(0, bandwidth_count - 1) * inter_bandwidth_gap

    minor_tick_positions = []
    minor_tick_labels = []

    for bw_idx, bandwidth_label in enumerate(bandwidth_order):
        subgroup_center_offset = (
            -total_span / 2
            + bw_idx * bandwidth_group_step
            + (top_n_count * top_n_step - inter_top_n_gap) / 2
        )
        minor_tick_positions.extend([x + subgroup_center_offset for x in x_positions])
        minor_tick_labels.extend([bandwidth_labels[bw_idx]] * len(x_positions))

        for top_n_idx, top_n_exp in enumerate(top_n_list):
            heights = []
            for model_name, squeeze_merge in bar_groups:
                top_n_summary = (
                    speedup_summary.get(model_name, {})
                    .get(squeeze_merge, {})
                    .get(bandwidth_label, {})
                    .get(top_n_exp)
                )
                if top_n_summary is None:
                    heights.append(0.0)
                else:
                    heights.append(top_n_summary["speedup"])

            offset = (
                -total_span / 2
                + bar_width / 2
                + bw_idx * bandwidth_group_step
                + top_n_idx * top_n_step
            )
            ax.bar(
                [x + offset for x in x_positions],
                heights,
                width=bar_width,
                color=top_n_colors[top_n_idx],
                edgecolor=top_n_colors[top_n_idx],
                linewidth=0.6,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(model_labels)
    ax.set_xticks(minor_tick_positions, minor=True)
    ax.set_xticklabels(minor_tick_labels, minor=True, fontsize=GLOBAL_PLOT_FONTSIZE)
    ax.tick_params(axis="x", which="major", pad=15, labelsize=GLOBAL_PLOT_FONTSIZE)
    ax.tick_params(axis="x", which="minor", pad=2, length=0, labelsize=GLOBAL_PLOT_FONTSIZE)
    ax.yaxis.set_minor_locator(AutoMinorLocator(minor_y_tick_subdivisions))
    ax.tick_params(axis="y", which="major", labelsize=GLOBAL_PLOT_FONTSIZE)
    ax.tick_params(axis="y", which="minor", labelleft=False)
    ax.set_ylabel("Speedup", fontsize=GLOBAL_PLOT_FONTSIZE + 1)
    if use_log_scale:
        ax.set_yscale("log")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.grid(which="minor", axis="y", linestyle=":", alpha=0.15, linewidth=0.5)

    top_n_handles = [
        Patch(
            facecolor=top_n_colors[idx],
            edgecolor=top_n_colors[idx],
            label=f"{top_n_to_MB[2**top_n_exp]:g}MB",
        )
        for idx, top_n_exp in enumerate(top_n_list)
    ]
    ax.legend(
        handles=top_n_handles,
        frameon=False,
        fontsize=GLOBAL_PLOT_FONTSIZE,
        title="Prefetch SRAM",
        title_fontsize=GLOBAL_PLOT_FONTSIZE + 1,
        loc=legend_loc,
        bbox_to_anchor=legend_bbox_to_anchor,
    )

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_path}")
    return output_path


if __name__ == "__main__":
    model_squeeze_list = [
        ("gpt2-small", "SqueezeMerge_1"),
        ("gpt2-medium", "SqueezeMerge_1"),
        ("opt-125m", "SqueezeMerge_0"),
    ]

    top_n_list = [14, 15, 16, 17]
    # hardware_design_locations_gpt2_small = {
    #     "bw_2048": Path("./sim_data/latency_area_result/gpt2-small/SqueezeMerge_1/compile_0/bw_2048/build_thput_128_dp_mul_256/MSM_pe_32_hardwins_256_topn_17_onchip_4096_inv_4_sumcheck_pes_32_eval_2_product_4_sramfeed_512_scvg_128.json"),
    #     "bw_1024": Path("./sim_data/latency_area_result/gpt2-small/SqueezeMerge_1/compile_0/bw_1024/build_thput_32_dp_mul_64/MSM_pe_32_hardwins_256_topn_15_onchip_4096_inv_4_sumcheck_pes_16_eval_2_product_3_sramfeed_512_scvg_128.json"),
    #     "bw_512": Path("./sim_data/latency_area_result/gpt2-small/SqueezeMerge_1/compile_0/bw_512/build_thput_32_dp_mul_64/MSM_pe_8_hardwins_256_topn_15_onchip_4096_inv_4_sumcheck_pes_16_eval_2_product_3_sramfeed_512_scvg_128.json"),
    # }
    hardware_design_locations_gpt2_small = {
        "bw_2048": Path("./sim_data/latency_area_result/gpt2-small/SqueezeMerge_1/compile_0/bw_2048/build_thput_1024_dp_mul_2048/MSM_pe_32_hardwins_256_topn_17_onchip_4096_inv_4_sumcheck_pes_32_eval_2_product_4_sramfeed_512_scvg_128.json"),
        "bw_1024": Path("./sim_data/latency_area_result/gpt2-small/SqueezeMerge_1/compile_0/bw_1024/build_thput_1024_dp_mul_2048/MSM_pe_32_hardwins_256_topn_15_onchip_4096_inv_4_sumcheck_pes_32_eval_2_product_4_sramfeed_512_scvg_128.json"),
        "bw_512": Path("./sim_data/latency_area_result/gpt2-small/SqueezeMerge_1/compile_0/bw_512/build_thput_1024_dp_mul_2048/MSM_pe_8_hardwins_256_topn_15_onchip_4096_inv_4_sumcheck_pes_32_eval_2_product_4_sramfeed_512_scvg_128.json"),
    }

    speedup_summary = collect_speedup_summary_for_top_n(
        model_squeeze_list,
        hardware_design_locations_gpt2_small,
        top_n_list,
    )
    plot_speedup_by_top_n(
        speedup_summary,
        model_squeeze_list,
        bandwidth_order=["bw_512", "bw_1024", "bw_2048"],
        top_n_list=top_n_list,
        output_path="./plots/proof_speedup_capture_sram.pdf",
        figsize=(8, 3),
        use_log_scale=True,
        minor_y_tick_subdivisions=2,
        legend_loc="upper left",
        legend_bbox_to_anchor=(0.0, 1.0),
    )

    print("speedup_capture_SRAM.py end.")
