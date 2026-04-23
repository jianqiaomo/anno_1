import json
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from hardware_experiments.util import is_pareto_efficient


PARETO_MAIN_LINEWIDTH = 1.1
PARETO_MAIN_MARKER_SIZE = 26
PARETO_MAIN_MARKER_EDGEWIDTH = 0.0
PARETO_INSET_LINEWIDTH = 0.9
PARETO_INSET_MARKER_SIZE = 18
PARETO_INSET_MARKER_EDGEWIDTH = 0.0
PARETO_MAIN_TICK_FONTSIZE = 15
PARETO_INSET_TICK_FONTSIZE = 13
PARETO_LEGEND_FONTSIZE = 13


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="gpt2-small")
    parser.add_argument("--squeeze-merge", type=str, default="SqueezeMerge_1")
    parser.add_argument(
        "--compile-options",
        type=int,
        nargs="+",
        default=[0],
        help="Which compile option folders to load, e.g. --compile-options 0 or --compile-options 1 or --compile-options 0 1",
    )
    parser.add_argument(
        "--latency-area-root",
        type=str,
        default="./sim_data/latency_area_result",
        help="Root directory that stores latency_area_result jsons.",
    )
    parser.add_argument(
        "--xlim",
        type=float,
        nargs=2,
        default=None,
        help="Optional x-axis range in ms, e.g. --xlim 0 200",
    )
    return parser.parse_args()


def collect_pareto_points(
    model_name,
    squeeze_merge,
    compile_options=(0, 1),
    latency_area_root="./sim_data/latency_area_result",
):
    base_dir = Path(latency_area_root) / model_name / squeeze_merge
    if not base_dir.exists():
        raise FileNotFoundError(f"Latency-area directory not found: {base_dir}")
    compile_tag = "_".join(str(option) for option in compile_options)
    output_path = base_dir / f"{model_name}_{squeeze_merge}_pareto_compile_{compile_tag}.json"
    if output_path.exists():
        print(f"Pareto json already exists at {output_path}, skipping collection.")
        return output_path, None

    points = []
    point_map = {}

    compile_dirs = [base_dir / f"compile_{compile_option}" for compile_option in compile_options]
    json_paths = []
    for compile_dir in compile_dirs:
        if not compile_dir.exists():
            continue
        json_paths.extend(sorted(compile_dir.rglob("*.json")))

    for json_path in json_paths:
        if json_path.name.startswith(f"{model_name}_{squeeze_merge}_pareto"):
            continue
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        runtime_ns = data["latency_breakdown_ns"]["total_latency_with_commit_open_ns"]
        runtime_ms = runtime_ns / 1_000_000
        area_mm2 = data["actual_cost"]["Total_area_mm2_7nm_with_HBM"]
        bandwidth = data["sweep_config"]["DRAM_bandwidth_B_cycle"]
        compile_option = data["sweep_config"]["compiler_option"]

        point = {
            "runtime_ms": runtime_ms,
            "area_mm2": area_mm2,
            "bandwidth": bandwidth,
            "compile_option": compile_option,
            "parameters": data["sweep_config"],
            "source_json": str(json_path),
        }
        points.append(point)
        point_map[f"{runtime_ms:.9f},{area_mm2:.9f}"] = {
            "runtime_ms": runtime_ms,
            "area_mm2": area_mm2,
            "bandwidth": bandwidth,
            "compile_option": compile_option,
            "parameters": data["sweep_config"],
            "source_json": str(json_path),
        }

    output = {
        "model_name": model_name,
        "squeeze_merge": squeeze_merge,
        "compile_options": list(compile_options),
        "num_points": len(points),
        "points": point_map,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    return output_path, points


def load_pareto_points(pareto_json_path):
    pareto_json_path = Path(pareto_json_path)
    with pareto_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data, list(data["points"].values())


def plot_latency_area_pareto_from_points(
    points,
    model_name,
    squeeze_merge,
    figsize=(5, 4),
    xlim=None,
    inset_annotations=None,
):
    if not points:
        raise ValueError("No points provided for Pareto plotting")

    bandwidths = sorted({point["bandwidth"] for point in points}, reverse=True)
    cmap = plt.get_cmap("tab10")
    bandwidth_color = {bandwidth: cmap(idx % 10) for idx, bandwidth in enumerate(bandwidths)}

    fig, ax = plt.subplots(figsize=figsize)
    all_bw_pareto_points = []

    for bandwidth in bandwidths:
        bw_points = [point for point in points if point["bandwidth"] == bandwidth]
        color = bandwidth_color[bandwidth]

        costs = np.array(
            [[point["area_mm2"], point["runtime_ms"]] for point in bw_points],
            dtype=float,
        )
        pareto_mask = is_pareto_efficient(costs)
        pareto_points = [point for point, keep in zip(bw_points, pareto_mask) if keep]
        pareto_points = sorted(pareto_points, key=lambda point: point["runtime_ms"])
        all_bw_pareto_points.extend(pareto_points)

        ax.plot(
            [point["runtime_ms"] for point in pareto_points],
            [point["area_mm2"] for point in pareto_points],
            color=color,
            linewidth=PARETO_MAIN_LINEWIDTH,
            alpha=0.5,
        )
        ax.scatter(
            [point["runtime_ms"] for point in pareto_points],
            [point["area_mm2"] for point in pareto_points],
            color=color,
            s=PARETO_MAIN_MARKER_SIZE,
            edgecolors="black",
            linewidths=PARETO_MAIN_MARKER_EDGEWIDTH,
            zorder=3,
            label=f"{bandwidth} GB/s",
        )

    if all_bw_pareto_points:
        global_costs = np.array(
            [[point["area_mm2"], point["runtime_ms"]] for point in all_bw_pareto_points],
            dtype=float,
        )
        global_pareto_mask = is_pareto_efficient(global_costs)
        global_pareto_points = [
            point for point, keep in zip(all_bw_pareto_points, global_pareto_mask) if keep
        ]
        global_pareto_points = sorted(global_pareto_points, key=lambda point: point["runtime_ms"])

        inset_ax = inset_axes(
            ax,
            width="42%",
            height="42%",
            bbox_to_anchor=(0.0, -0.18, 1.0, 1.0),
            bbox_transform=ax.transAxes,
            loc="upper right",
            borderpad=1.2,
        )
        for bandwidth in bandwidths:
            bw_global_points = [
                point for point in global_pareto_points if point["bandwidth"] == bandwidth
            ]
            if not bw_global_points:
                continue
            bw_global_points = sorted(bw_global_points, key=lambda point: point["runtime_ms"])
            inset_ax.plot(
                [point["runtime_ms"] for point in bw_global_points],
                [point["area_mm2"] for point in bw_global_points],
                color=bandwidth_color[bandwidth],
                linewidth=PARETO_INSET_LINEWIDTH,
                alpha=0.45,
            )
            inset_ax.scatter(
                [point["runtime_ms"] for point in bw_global_points],
                [point["area_mm2"] for point in bw_global_points],
                color=bandwidth_color[bandwidth],
                s=PARETO_INSET_MARKER_SIZE,
                zorder=3,
                linewidths=PARETO_INSET_MARKER_EDGEWIDTH,
                edgecolors="black",
            )
        inset_ax.set_title(
            "Global Pareto-Optimal",
            fontsize=PARETO_INSET_TICK_FONTSIZE + 1,
        )
        inset_ax.grid(alpha=0.2)
        inset_ax.tick_params(axis="both", labelsize=PARETO_INSET_TICK_FONTSIZE)
        if xlim is not None:
            inset_ax.set_xlim(xlim[0], xlim[1])
        for annotation in inset_annotations or []:
            point_xy = annotation["point_xy"]
            text_xy = annotation["text_xy"]
            label = annotation["label"]
            inset_ax.annotate(
                label,
                xy=point_xy,
                xytext=text_xy,
                color="red",
                fontsize=PARETO_INSET_TICK_FONTSIZE + 1,
                ha=annotation.get("ha", "center"),
                va=annotation.get("va", "center"),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": "red",
                    "lw": annotation.get("linewidth", 0.8),
                },
            )

    ax.set_xlabel("Runtime (ms)", fontsize=PARETO_MAIN_TICK_FONTSIZE + 1)
    ax.set_ylabel("Area (mm$^2$)", fontsize=PARETO_MAIN_TICK_FONTSIZE + 1)
    ax.tick_params(axis="both", labelsize=PARETO_MAIN_TICK_FONTSIZE)
    # ax.set_title(f"{model_name} {squeeze_merge} Pareto")
    ax.grid(alpha=0.2)
    ax.minorticks_on()
    ax.grid(which='minor', alpha=0.1, linestyle=':', linewidth=0.5)
    ax.legend(
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        frameon=True,
        fontsize=PARETO_LEGEND_FONTSIZE,
    )
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    fig.tight_layout()

    plot_path = f"./plots/{model_name}_{squeeze_merge}_pareto.pdf"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path


def plot_latency_area_pareto(pareto_json_path, figsize=(5, 4), xlim=None, inset_annotations=None):
    pareto_data, points = load_pareto_points(pareto_json_path)
    plot_path = plot_latency_area_pareto_from_points(
        points=points,
        model_name=pareto_data["model_name"],
        squeeze_merge=pareto_data["squeeze_merge"],
        figsize=figsize,
        xlim=xlim,
        inset_annotations=inset_annotations,
    )
    return plot_path


if __name__ == "__main__":
    args = _parse_args()
    pareto_json_path, _ = collect_pareto_points(
        model_name=args.model_name,
        squeeze_merge=args.squeeze_merge,
        compile_options=tuple(args.compile_options),
        latency_area_root=args.latency_area_root,
    )

    inset_annotations=[
        {
            "label": "A",
            "point_xy": (120.0, 210.0),
            "text_xy": (140.0, 210+90),
        },
        {
            "label": "B",
            "point_xy": (133, 150),
            "text_xy": (133+20, 150+90),
        },
        {
            "label": "C",
            "point_xy": (202.0, 70),
            "text_xy": (202+20, 70+90),
        },
    ]
    
    plot_path = plot_latency_area_pareto(
        pareto_json_path=pareto_json_path,
        figsize=(6.5, 4),
        xlim=(100, 400),
        inset_annotations=inset_annotations,
    )
    print(f"Dumped pareto json to {pareto_json_path}")
    print(f"Saved pareto plot to {plot_path}")
    print("pareto_plot.py end.")
