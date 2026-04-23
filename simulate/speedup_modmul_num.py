import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator


GLOBAL_PLOT_FONTSIZE = 15


SOFTWARE_PROOF_KEYS = (
    "logup prove",
    "GKR nonFC layer sumcheck",
    "GKR matmul layer sumcheck",
    "GKR last sumcheck",
)

BUILD_THPUT_RE = re.compile(r"build_thput_(\d+)_dp_mul_(\d+)")


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


def _replace_template_path(template_path, model_name, squeeze_merge, throughput, dp_mul):
    path_str = str(template_path)
    path_str = path_str.replace("gpt2-small", model_name)
    path_str = path_str.replace("SqueezeMerge_1", squeeze_merge)
    path_str = BUILD_THPUT_RE.sub(
        f"build_thput_{throughput}_dp_mul_{dp_mul}",
        path_str,
    )
    return Path(path_str)


def load_hardware_proof_runtime_ms(hardware_json_path):
    data = _load_json(hardware_json_path)
    return float(data["latency_breakdown_ns"]["total_latency_proof_part"]) / 1_000_000.0


def collect_speedup_summary_for_dp_mul(
    model_squeeze_list,
    hardware_design_locations_gpt2_small,
    build_mle_throughput_per_cycle_dp_mul_list,
):
    summary = {}
    for model_name, squeeze_merge in model_squeeze_list:
        software_runtime_ms = load_software_proof_runtime_ms(model_name, squeeze_merge)
        model_summary = {}
        for bandwidth_label, template_path in hardware_design_locations_gpt2_small.items():
            bandwidth_summary = {"software_runtime_ms": software_runtime_ms}
            for throughput, dp_mul in build_mle_throughput_per_cycle_dp_mul_list:
                hardware_path = _replace_template_path(
                    template_path,
                    model_name,
                    squeeze_merge,
                    throughput,
                    dp_mul,
                )
                hardware_runtime_ms = load_hardware_proof_runtime_ms(hardware_path)
                speedup = (
                    software_runtime_ms / hardware_runtime_ms
                    if hardware_runtime_ms > 0
                    else 0.0
                )
                bandwidth_summary[dp_mul] = {
                    "throughput": throughput,
                    "dp_mul": dp_mul,
                    "hardware_runtime_ms": hardware_runtime_ms,
                    "speedup": speedup,
                    "source_json": str(hardware_path),
                }
            model_summary[bandwidth_label] = bandwidth_summary
        summary.setdefault(model_name, {})[squeeze_merge] = model_summary
    return summary


def plot_speedup_by_dp_mul(
    speedup_summary,
    model_squeeze_list,
    bandwidth_order,
    build_mle_throughput_per_cycle_dp_mul_list,
    output_path="./plots/proof_speedup_modmul_num.pdf",
    figsize=(8, 4),
    use_log_scale_x=False,
    use_log_scale=False,
    minor_y_tick_subdivisions=2,
    model_legend_loc="upper left",
    model_legend_bbox_to_anchor=None,
    bandwidth_legend_loc="upper left",
    bandwidth_legend_bbox_to_anchor=(0.0, 0.78),
    layout_rect=None,
):
    fig, ax = plt.subplots(figsize=figsize)

    model_colors = {
        "gpt2-small": mcolors.to_hex(plt.cm.tab10(0)),
        "gpt2-medium": mcolors.to_hex(plt.cm.tab10(1)),
        "opt-125m": mcolors.to_hex(plt.cm.tab10(2)),
    }
    bandwidth_linestyles = {
        "bw_512": ":",
        "bw_1024": "--",
        "bw_2048": "-",
    }

    x_values = [dp_mul for _, dp_mul in build_mle_throughput_per_cycle_dp_mul_list]

    for model_name, squeeze_merge in model_squeeze_list:
        for bandwidth_label in bandwidth_order:
            y_values = []
            for _, dp_mul in build_mle_throughput_per_cycle_dp_mul_list:
                entry = (
                    speedup_summary.get(model_name, {})
                    .get(squeeze_merge, {})
                    .get(bandwidth_label, {})
                    .get(dp_mul)
                )
                y_values.append(0.0 if entry is None else entry["speedup"])

            ax.plot(
                x_values,
                y_values,
                marker="o",
                markersize=5,
                linewidth=1.8,
                color=model_colors.get(model_name, "C0"),
                linestyle=bandwidth_linestyles.get(bandwidth_label, "-"),
            )

    ax.set_xlabel("Modmuls Amount", fontsize=GLOBAL_PLOT_FONTSIZE + 1)
    ax.set_ylabel("Speedup", fontsize=GLOBAL_PLOT_FONTSIZE + 1)
    ax.tick_params(axis="x", which="major", labelsize=GLOBAL_PLOT_FONTSIZE)
    ax.tick_params(axis="y", which="major", labelsize=GLOBAL_PLOT_FONTSIZE)
    ax.yaxis.set_minor_locator(AutoMinorLocator(minor_y_tick_subdivisions))
    ax.tick_params(axis="y", which="minor", labelleft=False)
    if use_log_scale_x:
        ax.set_xscale("log")
    if use_log_scale:
        ax.set_yscale("log")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.grid(which="minor", axis="y", linestyle=":", alpha=0.15, linewidth=0.5)
    ax.set_xticks(x_values)
    ax.set_xticklabels([str(x) for x in x_values])

    model_handles = [
        Line2D(
            [0],
            [0],
            color=model_colors.get(model_name, "C0"),
            linewidth=1.8,
            marker="o",
            markersize=5,
            linestyle="-",
            label=_model_label(model_name),
        )
        for model_name, _ in model_squeeze_list
    ]
    model_legend = ax.legend(
        handles=model_handles,
        frameon=False,
        fontsize=GLOBAL_PLOT_FONTSIZE,
        title="Model",
        title_fontsize=GLOBAL_PLOT_FONTSIZE + 1,
        loc=model_legend_loc,
        bbox_to_anchor=model_legend_bbox_to_anchor,
    )
    ax.add_artist(model_legend)

    bandwidth_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=1.8,
            linestyle=bandwidth_linestyles.get(bandwidth_label, "-"),
            label=bandwidth_label.replace("bw_", "BW "),
        )
        for bandwidth_label in bandwidth_order
    ]
    bandwidth_legend = ax.legend(
        handles=bandwidth_handles,
        frameon=False,
        fontsize=GLOBAL_PLOT_FONTSIZE,
        title="Bandwidth",
        title_fontsize=GLOBAL_PLOT_FONTSIZE + 1,
        loc=bandwidth_legend_loc,
        bbox_to_anchor=bandwidth_legend_bbox_to_anchor,
    )

    if layout_rect is not None:
        fig.tight_layout(rect=layout_rect)
    else:
        fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        bbox_extra_artists=[model_legend, bandwidth_legend],
    )
    plt.close(fig)
    print(f"Saved plot to {output_path}")
    return output_path


if __name__ == "__main__":
    model_squeeze_list = [
        ("gpt2-small", "SqueezeMerge_1"),
        ("gpt2-medium", "SqueezeMerge_1"),
        ("opt-125m", "SqueezeMerge_0"),
    ]
    build_mle_throughput_per_cycle_dp_mul_list = [
        (32, 64),
        (128, 256),
        (512, 1024),
        (1024, 2048),
    ]
    # hardware_design_locations_gpt2_small = {
    #     "bw_2048": Path("./sim_data/latency_area_result/gpt2-small/SqueezeMerge_1/compile_0/bw_2048/build_thput_128_dp_mul_256/MSM_pe_32_hardwins_256_topn_17_onchip_4096_inv_4_sumcheck_pes_32_eval_2_product_4_sramfeed_512_scvg_128.json"),
    #     "bw_1024": Path("./sim_data/latency_area_result/gpt2-small/SqueezeMerge_1/compile_0/bw_1024/build_thput_32_dp_mul_64/MSM_pe_32_hardwins_256_topn_15_onchip_4096_inv_4_sumcheck_pes_16_eval_2_product_3_sramfeed_512_scvg_128.json"),
    #     "bw_512": Path("./sim_data/latency_area_result/gpt2-small/SqueezeMerge_1/compile_0/bw_512/build_thput_32_dp_mul_64/MSM_pe_8_hardwins_256_topn_15_onchip_4096_inv_4_sumcheck_pes_16_eval_2_product_3_sramfeed_512_scvg_128.json"),
    # }
    hardware_design_locations_gpt2_small = {
        "bw_2048": Path("./sim_data/latency_area_result/gpt2-small/SqueezeMerge_1/compile_0/bw_2048/build_thput_32_dp_mul_64/MSM_pe_8_hardwins_256_topn_15_onchip_4096_inv_4_sumcheck_pes_16_eval_2_product_3_sramfeed_512_scvg_128.json"),
        "bw_1024": Path("./sim_data/latency_area_result/gpt2-small/SqueezeMerge_1/compile_0/bw_1024/build_thput_32_dp_mul_64/MSM_pe_8_hardwins_256_topn_15_onchip_4096_inv_4_sumcheck_pes_16_eval_2_product_3_sramfeed_512_scvg_128.json"),
        "bw_512": Path("./sim_data/latency_area_result/gpt2-small/SqueezeMerge_1/compile_0/bw_512/build_thput_32_dp_mul_64/MSM_pe_8_hardwins_256_topn_15_onchip_4096_inv_4_sumcheck_pes_16_eval_2_product_3_sramfeed_512_scvg_128.json"),
    }

    speedup_summary = collect_speedup_summary_for_dp_mul(
        model_squeeze_list,
        hardware_design_locations_gpt2_small,
        build_mle_throughput_per_cycle_dp_mul_list,
    )
    plot_speedup_by_dp_mul(
        speedup_summary,
        model_squeeze_list,
        bandwidth_order=["bw_512", "bw_1024", "bw_2048"],
        build_mle_throughput_per_cycle_dp_mul_list=build_mle_throughput_per_cycle_dp_mul_list,
        output_path="./plots/proof_speedup_modmul_num.pdf",
        figsize=(8, 4),
        use_log_scale=True,
        use_log_scale_x=True,
        minor_y_tick_subdivisions=2,
        model_legend_loc="upper left",
        model_legend_bbox_to_anchor=(1.02, 1.0),
        bandwidth_legend_loc="upper left",
        bandwidth_legend_bbox_to_anchor=(1.02, 0.5),
        layout_rect=(0.0, 0.0, 0.9, 1.0),
    )

    print("speedup_modmul_num.py end.")
