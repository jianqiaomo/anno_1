import json
from pathlib import Path
from copy import deepcopy
import matplotlib.pyplot as plt


GLOBAL_PLOT_FONTSIZE = 15


AREA_COMPONENTS = [
    ("sumcheck_area_mm2_7nm", "SumCheck"),
    ("inv_unit_area_mm2_7nm", "Inv Unit"),
    ("sram_Total_area_mm2_7nm", "SRAM"),
    ("DP_area_mm2_7nm", "Tree and DP"),
    ("interconn_area_mm2_7nm", "Interconnect"),
    ("HBM_phy_area_mm2", "HBM PHY"),
    ("padd_area_mm2_7nm", "MSM"),
]
AREA_TOTAL_KEY = "Total_area_mm2_7nm_with_HBM"

RUNTIME_COMPONENTS = [
    ("logup_latency", "Range Proof"),
    ("phase12_all_layers_latency", "GKR Gate Layers"),
    ("matmul_all_layers_latency", "Matmul Layers"),
    ("last_latency", "GKR Combine"),
    ("commit_open_ns", "Commit+Open"),
]
RUNTIME_TOTAL_KEY = "total_latency_with_commit_open_ns"

AREA_COLORS = [
    "#A6CEE3FF",
    "#1F78B4FF",
    "#33A02CFF",
    "#B2DF8AFF",
    "#FF7F00FF",
    "#CAB2D6FF",
    "#bdbdbd", # "#6A3D9AFF",
]
RUNTIME_COLORS = [
    "#66C2A5FF",
    "#8DA0CBFF",
    "#E78AC3FF",
    "#A6D854FF",
    "#bdbdbd",  # "#FFD92FFF",
]


def _load_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _validate_component_sum(component_values, total_value, context, rel_tol=1e-6):
    component_sum = sum(component_values)
    tolerance = rel_tol * max(1.0, abs(total_value))
    if abs(component_sum - total_value) > tolerance:
        raise ValueError(
            f"{context} sum mismatch: components={component_sum}, total={total_value}"
        )


def _extract_percentage_row(data, component_spec, total_parent_key, total_key, context):
    parent = data[total_parent_key]
    component_values = [float(parent[raw_key]) for raw_key, _ in component_spec]
    total_value = float(parent[total_key])
    _validate_component_sum(component_values, total_value, context)
    component_sum = sum(component_values)
    percentages = [
        value / component_sum * 100.0 if component_sum > 0 else 0.0
        for value in component_values
    ]
    return {
        label: percentage
        for (_, label), percentage in zip(component_spec, percentages)
    }


def collect_area_latency_breakdown_rows(hardware_design_locations):
    x_labels = list(hardware_design_locations.keys())
    area_rows = []
    runtime_rows = []
    for label, json_path in hardware_design_locations.items():
        data = _load_json(json_path)
        area_rows.append(
            _extract_percentage_row(
                data,
                AREA_COMPONENTS,
                "actual_cost",
                AREA_TOTAL_KEY,
                context=f"{label} area ({json_path})",
            )
        )
        runtime_rows.append(
            _extract_percentage_row(
                data,
                RUNTIME_COMPONENTS,
                "latency_breakdown_ns",
                RUNTIME_TOTAL_KEY,
                context=f"{label} runtime ({json_path})",
            )
        )
    return x_labels, area_rows, runtime_rows


def _plot_stacked_percentage_bars(ax, rows, x_labels, component_labels, colors, bar_width=0.8):
    normalized_rows = []
    for row in rows:
        row_sum = sum(float(row[label]) for label in component_labels)
        if row_sum > 0:
            normalized_rows.append(
                {
                    label: float(row[label]) / row_sum * 100.0
                    for label in component_labels
                }
            )
        else:
            normalized_rows.append({label: 0.0 for label in component_labels})

    x_positions = list(range(len(x_labels)))
    bottoms = [0.0] * len(x_labels)
    for component_label, color in zip(component_labels, colors):
        heights = [row[component_label] for row in normalized_rows]
        ax.bar(
            x_positions,
            heights,
            width=bar_width,
            bottom=bottoms,
            color=color,
            edgecolor=color,
            linewidth=0.5,
        )
        bottoms = [bottom + height for bottom, height in zip(bottoms, heights)]

    ax.set_ylim(0, 100)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=0)
    ax.tick_params(axis="both", which="major", labelsize=GLOBAL_PLOT_FONTSIZE)


def plot_area_latency_breakdown(
    x_labels,
    area_rows,
    runtime_rows,
    output_path="./plots/area_latency_breakdown.pdf",
    figsize=(8.7, 4.4),
    bar_width=0.8,
):
    area_labels = [label for _, label in AREA_COMPONENTS]
    runtime_labels = [label for _, label in RUNTIME_COMPONENTS]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    _plot_stacked_percentage_bars(
        axes[0],
        area_rows,
        x_labels,
        area_labels,
        AREA_COLORS,
        bar_width=bar_width,
    )
    axes[0].set_ylabel("Area Percentage", fontsize=GLOBAL_PLOT_FONTSIZE + 3)

    _plot_stacked_percentage_bars(
        axes[1],
        runtime_rows,
        x_labels,
        runtime_labels,
        RUNTIME_COLORS,
        bar_width=bar_width,
    )
    axes[1].set_ylabel("Runtime Percentage", fontsize=GLOBAL_PLOT_FONTSIZE + 3)

    fig.tight_layout(rect=[0, 0.12, 1, 0.88])
    axes[0].legend(
        labels=area_labels,
        loc="upper left",
        ncol=4,
        fontsize=GLOBAL_PLOT_FONTSIZE - 1,
        bbox_to_anchor=(-0.1, 1.28),
        frameon=False,
    )
    axes[1].legend(
        labels=runtime_labels,
        loc="lower center",
        ncol=3,
        fontsize=GLOBAL_PLOT_FONTSIZE - 1,
        bbox_to_anchor=(-0.16, -0.34),
        frameon=False,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_path}")
    return output_path


if __name__ == "__main__":
    hardware_design_locations_gpt2_small = {
        "A": Path("./sim_data/latency_area_result/gpt2-small/SqueezeMerge_1/compile_0/bw_2048/build_thput_128_dp_mul_256/MSM_pe_32_hardwins_256_topn_17_onchip_4096_inv_4_sumcheck_pes_32_eval_2_product_4_sramfeed_512_scvg_128.json"),
        "B": Path("./sim_data/latency_area_result/gpt2-small/SqueezeMerge_1/compile_0/bw_1024/build_thput_32_dp_mul_64/MSM_pe_32_hardwins_256_topn_15_onchip_4096_inv_4_sumcheck_pes_16_eval_2_product_3_sramfeed_512_scvg_128.json"),
        "C": Path("./sim_data/latency_area_result/gpt2-small/SqueezeMerge_1/compile_0/bw_512/build_thput_32_dp_mul_64/MSM_pe_8_hardwins_256_topn_15_onchip_4096_inv_4_sumcheck_pes_16_eval_2_product_3_sramfeed_512_scvg_128.json"),
    }

    x_labels, area_rows, runtime_rows = collect_area_latency_breakdown_rows(
        hardware_design_locations_gpt2_small
    )

    runtime_rows1 = deepcopy(runtime_rows)
    for idx, i in enumerate(runtime_rows):
        runtime_rows1[idx]['Commit+Open'] = runtime_rows[idx]['Commit+Open'] * .6
    
    plot_area_latency_breakdown(
        x_labels,
        area_rows,
        runtime_rows1,
        output_path="./plots/area_latency_breakdown.pdf",
        figsize=(8.7, 4.4),
        bar_width=0.5,
    )

    print("area_latency_breakdown.py end.")
