import json
import itertools
import re
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Patch


TOTAL_Y_COUNT_RE = re.compile(r"^total_[^_]+_count$")
TOTAL_EASY_RE = re.compile(r"^total_[^_]+_easy_[^_]+_count$")
TOTAL_HARD_CAPTURED_RE = re.compile(r"^total_[^_]+_hard_captured_[^_]+_count$")
TOTAL_HARD_UNCAPTURED_RE = re.compile(r"^total_[^_]+_hard_uncaptured_[^_]+_count$")
HARD_WINDOW_RE = re.compile(r"^[^_]+_hard_window$")
HARD_WINDOW_CAPTURED_COUNT_RE = re.compile(r"^([^_]+)_hard_window_captured_([^_]+)_count$")
HARD_WINDOW_UNCAPTURED_COUNT_RE = re.compile(r"^([^_]+)_hard_window_uncaptured_([^_]+)_count$")


top_n_to_MB = {  # Actually need double as result
    2**14: 0.5,
    2**15: 1.0,
    2**16: 2.0,
    2**17: 4.0,
}

GLOBAL_PLOT_FONTSIZE = 11


def _format_compact_scientific(value):
    formatted = f"{value:.1e}"
    mantissa, exponent = formatted.split("e")
    exponent = exponent.lstrip("+")
    if exponent.startswith("-"):
        digits = exponent[1:].lstrip("0")
        exponent = f"-{digits or '0'}"
    else:
        exponent = exponent.lstrip("0") or "0"
    return f"{mantissa}e{exponent}"


def _collect_layer_totals(obj):
    total_y_count = 0
    total_x_easy_y_count = 0
    total_x_hard_captured_y_count = 0
    total_x_hard_uncaptured_y_count = 0
    total_x_hard_window_count = 0

    if isinstance(obj, dict):
        adjusted_values = {}
        for key, value in obj.items():
            if not isinstance(key, str):
                continue
            if not HARD_WINDOW_RE.fullmatch(key):
                continue
            if not isinstance(value, list):
                continue

            x_name = key[: -len("_hard_window")]
            singleton_indices = [
                idx for idx, window in enumerate(value)
                if isinstance(window, (list, tuple))
                and len(window) == 2
                and (window[1] - window[0] == 1)
            ]
            if not singleton_indices:
                continue

            for count_key, count_value in obj.items():
                if not isinstance(count_key, str) or not isinstance(count_value, list):
                    continue

                captured_match = HARD_WINDOW_CAPTURED_COUNT_RE.fullmatch(count_key)
                if captured_match and captured_match.group(1) == x_name:
                    y_name = captured_match.group(2)
                    uncaptured_key = f"{x_name}_hard_window_uncaptured_{y_name}_count"
                    uncaptured_value = obj.get(uncaptured_key)
                    if not isinstance(uncaptured_value, list):
                        continue
                    if len(count_value) != len(value) or len(uncaptured_value) != len(value):
                        continue

                    adjusted_captured = list(count_value)
                    adjusted_uncaptured = list(uncaptured_value)
                    moved_uncaptured_total = 0
                    for idx in singleton_indices:
                        moved_uncaptured_total += adjusted_uncaptured[idx]
                        adjusted_captured[idx] += adjusted_uncaptured[idx]
                        adjusted_uncaptured[idx] = 0

                    adjusted_values[count_key] = adjusted_captured
                    adjusted_values[uncaptured_key] = adjusted_uncaptured

                    total_captured_key = f"total_{x_name}_hard_captured_{y_name}_count"
                    total_uncaptured_key = f"total_{x_name}_hard_uncaptured_{y_name}_count"
                    if total_captured_key in obj:
                        adjusted_values[total_captured_key] = int(obj[total_captured_key]) + moved_uncaptured_total
                    if total_uncaptured_key in obj:
                        adjusted_values[total_uncaptured_key] = int(obj[total_uncaptured_key]) - moved_uncaptured_total

        for key, value in obj.items():
            value = adjusted_values.get(key, value)
            if isinstance(key, str):
                if TOTAL_Y_COUNT_RE.fullmatch(key):
                    total_y_count += int(value)
                    continue
                if TOTAL_EASY_RE.fullmatch(key):
                    total_x_easy_y_count += int(value)
                    continue
                if TOTAL_HARD_CAPTURED_RE.fullmatch(key):
                    total_x_hard_captured_y_count += int(value)
                    continue
                if TOTAL_HARD_UNCAPTURED_RE.fullmatch(key):
                    total_x_hard_uncaptured_y_count += int(value)
                    continue
                if HARD_WINDOW_RE.fullmatch(key):
                    total_x_hard_window_count += len(value)
                    continue

            (
                child_total_y_count,
                child_total_x_easy_y_count,
                child_total_x_hard_captured_y_count,
                child_total_x_hard_uncaptured_y_count,
                child_total_x_hard_window_count,
            ) = _collect_layer_totals(value)
            total_y_count += child_total_y_count
            total_x_easy_y_count += child_total_x_easy_y_count
            total_x_hard_captured_y_count += child_total_x_hard_captured_y_count
            total_x_hard_uncaptured_y_count += child_total_x_hard_uncaptured_y_count
            total_x_hard_window_count += child_total_x_hard_window_count
    elif isinstance(obj, list):
        for value in obj:
            (
                child_total_y_count,
                child_total_x_easy_y_count,
                child_total_x_hard_captured_y_count,
                child_total_x_hard_uncaptured_y_count,
                child_total_x_hard_window_count,
            ) = _collect_layer_totals(value)
            total_y_count += child_total_y_count
            total_x_easy_y_count += child_total_x_easy_y_count
            total_x_hard_captured_y_count += child_total_x_hard_captured_y_count
            total_x_hard_uncaptured_y_count += child_total_x_hard_uncaptured_y_count
            total_x_hard_window_count += child_total_x_hard_window_count

    return (
        total_y_count,
        total_x_easy_y_count,
        total_x_hard_captured_y_count,
        total_x_hard_uncaptured_y_count,
        total_x_hard_window_count,
    )


def collect_layer_statistics(
    model_name,
    squeeze_merge,
    top_n,
    hard_window_size,
    target_layers,
    duplicate_layers=None,
    repeat_count=1,
    comp_data_root="comp_data",
):
    """
    For a given model/config, walk the target-layer json files and collect:
    - total_y_count for each layer
    - total_x_easy_y_count for each layer
    - total_x_hard_captured_y_count for each layer
    - total_x_hard_uncaptured_y_count for each layer
    - sum of len(x_hard_window) for each layer

    The per-layer totals are computed by summing across all sections in that layer
    json (phase1, phase2, lasso).
    """
    top_n_exp = int(top_n).bit_length() - 1
    layer_stats = {}
    duplicate_layers = set(duplicate_layers or [])
    repeat_count = int(repeat_count)

    for layer in target_layers:
        filename = (
            Path(comp_data_root)
            / model_name
            / squeeze_merge
            / f"{model_name}_layer_{layer}_hardwins_{hard_window_size}_topn_{top_n_exp}.json"
        )
        if not filename.exists():
            print(f"File not found: {filename}")
            continue

        raw = filename.read_text(encoding="utf-8")
        data = json.loads(raw)
        (
            total_y_count,
            total_x_easy_y_count,
            total_x_hard_captured_y_count,
            total_x_hard_uncaptured_y_count,
            total_x_hard_window_count,
        ) = _collect_layer_totals(data)
        layer_stats[layer] = {
            "file": str(filename),
            "total_y_count": total_y_count,
            "total_x_easy_y_count": total_x_easy_y_count,
            "total_x_hard_captured_y_count": total_x_hard_captured_y_count,
            "total_x_hard_uncaptured_y_count": total_x_hard_uncaptured_y_count,
            "total_x_hard_window_count": total_x_hard_window_count,
        }
        if (
            total_y_count
            != total_x_easy_y_count
            + total_x_hard_captured_y_count
            + total_x_hard_uncaptured_y_count
        ):
            raise ValueError(
                f"Layer {layer} count mismatch in {filename}: "
                f"total_y_count={total_y_count}, "
                f"total_x_easy_y_count={total_x_easy_y_count}, "
                f"total_x_hard_captured_y_count={total_x_hard_captured_y_count}, "
                f"total_x_hard_uncaptured_y_count={total_x_hard_uncaptured_y_count}"
            )

        multiplier = repeat_count if layer in duplicate_layers else 1
        layer_stats[layer]["repeat_multiplier"] = multiplier
        layer_stats[layer]["effective_total_y_count"] = total_y_count * multiplier
        layer_stats[layer]["effective_total_x_easy_y_count"] = (
            total_x_easy_y_count * multiplier
        )
        layer_stats[layer]["effective_total_x_hard_captured_y_count"] = (
            total_x_hard_captured_y_count * multiplier
        )
        layer_stats[layer]["effective_total_x_hard_uncaptured_y_count"] = (
            total_x_hard_uncaptured_y_count * multiplier
        )
        layer_stats[layer]["effective_total_x_hard_window_count"] = (
            total_x_hard_window_count * multiplier
        )

    summary = {
        "model_name": model_name,
        "squeeze_merge": squeeze_merge,
        "top_n": top_n,
        "top_n_exp": top_n_exp,
        "hard_window_size": hard_window_size,
        "duplicate_layers": sorted(duplicate_layers),
        "repeat_count": repeat_count,
        "layers": layer_stats,
        "sum_total_y_count": sum(
            stats["effective_total_y_count"] for stats in layer_stats.values()
        ),
        "sum_total_x_easy_y_count": sum(
            stats["effective_total_x_easy_y_count"] for stats in layer_stats.values()
        ),
        "sum_total_x_hard_captured_y_count": sum(
            stats["effective_total_x_hard_captured_y_count"] for stats in layer_stats.values()
        ),
        "sum_total_x_hard_uncaptured_y_count": sum(
            stats["effective_total_x_hard_uncaptured_y_count"] for stats in layer_stats.values()
        ),
        "sum_total_len_x_hard_window_count": sum(
            stats["effective_total_x_hard_window_count"] for stats in layer_stats.values()
        ),
    }
    summary["uncaptured_y_rate"] = (
        summary["sum_total_x_hard_uncaptured_y_count"] / summary["sum_total_y_count"]
        if summary["sum_total_y_count"] > 0
        else 0.0
    )
    summary["uncaptured_y_rate_wo_opt"] = (
        (summary["sum_total_x_hard_uncaptured_y_count"] + summary["sum_total_x_hard_captured_y_count"]) / summary["sum_total_y_count"]
        if summary["sum_total_y_count"] > 0
        else 0.0
    )
    summary["y_traffic"] = top_n * summary["sum_total_len_x_hard_window_count"] + summary["sum_total_x_hard_uncaptured_y_count"]
    return summary


def _blend_with_white(color, blend):
    r, g, b = mcolors.to_rgb(color)
    return (
        r * (1 - blend) + blend,
        g * (1 - blend) + blend,
        b * (1 - blend) + blend,
    )


def plot_uncaptured_y_rate_grouped(
    model_result,
    model_squeeze_list,
    hard_window_sizes,
    top_ns,
    output_path="plot/uncaptured_y_rate.pdf",
    figsize=(12, 5),
):
    top_n_exps = [int(top_n).bit_length() - 1 for top_n in top_ns]
    hard_window_labels = [f"{hard_window_size}w" for hard_window_size in hard_window_sizes]
    model_labels = []
    bar_groups = []
    for model_name, squeeze_merge, _, _, _ in model_squeeze_list:
        if model_name == "gpt2-small":
            model_label = "GPT2-Small"
        elif model_name == "gpt2-medium":
            model_label = "GPT2-Medium"
        elif model_name == "opt-125m":
            model_label = "OPT-125M"
        else:
            model_label = model_name
        model_labels.append(model_label)
        bar_groups.append((model_name, squeeze_merge))

    fig, ax = plt.subplots(figsize=figsize)
    model_group_spacing = 1.18
    x_positions = [idx * model_group_spacing for idx in range(len(bar_groups))]
    base_colors = [
        mcolors.to_hex(plt.cm.viridis(value))
        for value in [0.15, 0.3, 0.45, 0.6, 0.75, 0.9][::-1]
    ]
    top_n_count = len(top_ns)
    hard_window_count = len(hard_window_sizes)
    inter_top_n_gap = 0.006
    inter_hard_window_gap = 0.03
    bar_width = 0.8 / (
        hard_window_count * top_n_count
        + max(0, top_n_count - 1) * hard_window_count * (inter_top_n_gap / 0.8)
        + max(0, hard_window_count - 1) * (inter_hard_window_gap / 0.8)
    )
    top_n_step = bar_width + inter_top_n_gap
    hard_window_group_step = top_n_count * top_n_step + inter_hard_window_gap
    total_span = hard_window_count * top_n_count * bar_width
    total_span += hard_window_count * max(0, top_n_count - 1) * inter_top_n_gap
    total_span += max(0, hard_window_count - 1) * inter_hard_window_gap
    minor_tick_positions = []
    minor_tick_labels = []

    for hw_idx, hard_window_size in enumerate(hard_window_sizes):
        subgroup_center_offset = (
            -total_span / 2
            + hw_idx * hard_window_group_step
            + (top_n_count * top_n_step - inter_top_n_gap) / 2
        )
        minor_tick_positions.extend([x + subgroup_center_offset for x in x_positions])
        minor_tick_labels.extend([hard_window_labels[hw_idx]] * len(x_positions))

        for top_idx, (top_n, top_n_exp) in enumerate(zip(top_ns, top_n_exps)):
            base_color = base_colors[top_idx % len(base_colors)]
            heights = []
            for model_name, squeeze_merge in bar_groups:
                summary = (
                    model_result.get(model_name, {})
                    .get(squeeze_merge, {})
                    .get(top_n_exp, {})
                    .get(hard_window_size)
                )
                if summary is None:
                    heights.append(0.0)
                else:
                    if summary["uncaptured_y_rate"] != 0:
                        heights.append(summary["uncaptured_y_rate"] * 100.0)
                    else:
                        heights.append(0.001 * 100.0)

            offset = (
                -total_span / 2
                + bar_width / 2
                + hw_idx * hard_window_group_step
                + top_idx * top_n_step
            )
            ax.bar(
                [x + offset for x in x_positions],
                heights,
                width=bar_width,
                color=base_color,
                edgecolor=base_color,
                linewidth=0.5,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(model_labels)
    ax.set_xticks(minor_tick_positions, minor=True)
    ax.set_xticklabels(minor_tick_labels, minor=True, fontsize=GLOBAL_PLOT_FONTSIZE)
    ax.tick_params(axis="x", which="major", pad=20, labelsize=GLOBAL_PLOT_FONTSIZE)
    ax.tick_params(axis="x", which="minor", pad=2, length=0, labelsize=GLOBAL_PLOT_FONTSIZE, rotation=20)
    ax.tick_params(axis="y", labelsize=GLOBAL_PLOT_FONTSIZE)
    ax.set_ylabel("Uncaptured Rate (%)", fontsize=GLOBAL_PLOT_FONTSIZE + 1)
    # ax.set_title("Uncaptured Rate by Model, Hard Window Size, and SRAM")
    legend_handles = [
        Patch(facecolor=base_colors[idx % len(base_colors)], edgecolor=base_colors[idx % len(base_colors)], label=f"{top_n_to_MB[top_n]:g}MB")
        for idx, top_n in enumerate(top_ns)
    ]
    ax.legend(
        handles=legend_handles,
        ncol=1,
        fontsize=GLOBAL_PLOT_FONTSIZE,
        frameon=False,
        loc="best",  # "upper left"
        # bbox_to_anchor=(1.01, 1),
        title="Prefetch SRAM",
        title_fontsize=GLOBAL_PLOT_FONTSIZE + 1,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_path}")
    return output_path


def plot_y_traffic_grouped(
    model_result,
    model_squeeze_list,
    hard_window_sizes,
    top_ns,
    output_path="plot/y_traffic.pdf",
    figsize=(12, 5),
):
    top_n_exps = [int(top_n).bit_length() - 1 for top_n in top_ns]
    top_n_labels = [f"{top_n_to_MB[top_n]:g}MB" for top_n in top_ns]
    model_labels = []
    bar_groups = []
    for model_name, squeeze_merge, _, _, _ in model_squeeze_list:
        if model_name == "gpt2-small":
            model_label = "GPT2-Small"
        elif model_name == "gpt2-medium":
            model_label = "GPT2-Medium"
        elif model_name == "opt-125m":
            model_label = "OPT-125M"
        else:
            model_label = model_name
        model_labels.append(model_label)
        bar_groups.append((model_name, squeeze_merge))

    fig, ax = plt.subplots(figsize=figsize)
    model_group_spacing = 1.18
    x_positions = [idx * model_group_spacing for idx in range(len(bar_groups))]
    # base_colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"]
    base_colors = [
        mcolors.to_hex(plt.cm.viridis(value))
        for value in [0.15, 0.3, 0.45, 0.6, 0.75, 0.9][::-1]
    ]
    top_n_count = len(top_ns)
    hard_window_count = len(hard_window_sizes)
    inter_top_n_gap = 0.006
    inter_hard_window_gap = 0.03
    bar_width = 0.8 / (
        hard_window_count * top_n_count
        + max(0, top_n_count - 1) * hard_window_count * (inter_top_n_gap / 0.8)
        + max(0, hard_window_count - 1) * (inter_hard_window_gap / 0.8)
    )
    hard_window_step = bar_width + inter_top_n_gap
    top_n_group_step = hard_window_count * hard_window_step + inter_hard_window_gap
    total_span = hard_window_count * top_n_count * bar_width
    total_span += top_n_count * max(0, hard_window_count - 1) * inter_top_n_gap
    total_span += max(0, top_n_count - 1) * inter_hard_window_gap
    minor_tick_positions = []
    minor_tick_labels = []

    for top_idx, (top_n, top_n_exp) in enumerate(zip(top_ns, top_n_exps)):
        subgroup_center_offset = (
            -total_span / 2
            + top_idx * top_n_group_step
            + (hard_window_count * hard_window_step - inter_top_n_gap) / 2
        )
        minor_tick_positions.extend([x + subgroup_center_offset for x in x_positions])
        minor_tick_labels.extend([top_n_labels[top_idx]] * len(x_positions))

        for hw_idx, hard_window_size in enumerate(hard_window_sizes):
            base_color = base_colors[hw_idx % len(base_colors)]
            heights = []
            for model_name, squeeze_merge in bar_groups:
                summary = (
                    model_result.get(model_name, {})
                    .get(squeeze_merge, {})
                    .get(top_n_exp, {})
                    .get(hard_window_size)
                )
                if summary is None:
                    heights.append(0.0)
                else:
                    heights.append(summary["y_traffic"])

            offset = (
                -total_span / 2
                + bar_width / 2
                + top_idx * top_n_group_step
                + hw_idx * hard_window_step
            )
            ax.bar(
                [x + offset for x in x_positions],
                heights,
                width=bar_width,
                color=base_color,
                edgecolor=base_color,
                linewidth=0.5,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(model_labels)
    ax.set_xticks(minor_tick_positions, minor=True)
    ax.set_xticklabels(minor_tick_labels, minor=True, fontsize=GLOBAL_PLOT_FONTSIZE)
    ax.tick_params(axis="x", which="major", pad=7, labelsize=GLOBAL_PLOT_FONTSIZE)
    ax.tick_params(axis="x", which="minor", pad=2, length=0, labelsize=GLOBAL_PLOT_FONTSIZE)
    ax.tick_params(axis="y", labelsize=GLOBAL_PLOT_FONTSIZE)
    ax.set_ylabel("DRAM Traffic in words", fontsize=GLOBAL_PLOT_FONTSIZE + 1)
    legend_handles = [
        Patch(facecolor=base_colors[idx % len(base_colors)], edgecolor=base_colors[idx % len(base_colors)], label=f"{hard_window_size}words")
        for idx, hard_window_size in enumerate(hard_window_sizes)
    ]
    ax.legend(
        handles=legend_handles,
        ncol=1,
        fontsize=GLOBAL_PLOT_FONTSIZE,
        frameon=False,
        loc="best",
        # bbox_to_anchor=(1.01, 1),
        title="Window Size",
        title_fontsize=GLOBAL_PLOT_FONTSIZE + 1,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_path}")
    return output_path


def plot_uncaptured_rate_compare_for_hard_window(
    model_result,
    model_squeeze_list,
    hard_window_size,
    top_ns,
    output_path="plot/uncaptured_rate_compare.pdf",
    figsize=(8, 3),
    baseline_legend_loc="upper right",
    baseline_legend_bbox_to_anchor=None,
    compiler_legend_loc="upper right",
    compiler_legend_bbox_to_anchor=(1.0, 0.8),
):
    top_n_exps = [int(top_n).bit_length() - 1 for top_n in top_ns]
    model_labels = []
    bar_groups = []
    for model_name, squeeze_merge, _, _, _ in model_squeeze_list:
        if model_name == "gpt2-small":
            model_label = "GPT2-Small"
        elif model_name == "gpt2-medium":
            model_label = "GPT2-Medium"
        elif model_name == "opt-125m":
            model_label = "OPT-125M"
        else:
            model_label = model_name
        model_labels.append(model_label)
        bar_groups.append((model_name, squeeze_merge))

    fig, ax = plt.subplots(figsize=figsize)
    model_group_spacing = 1.18
    x_positions = [idx * model_group_spacing for idx in range(len(bar_groups))]
    compare_labels = ["w/o compiler"] + [f"SRAM {top_n_to_MB[top_n]:g}MB" for top_n in top_ns]
    compare_colors = ["#BDBDBD"] + [
        mcolors.to_hex(plt.cm.viridis(value))
        for value in [0.15, 0.35, 0.6, 0.85][:len(top_ns)][::-1]
    ]
    compare_count = len(compare_labels)
    inter_bar_gap = 0.012
    bar_width = 0.8 / (
        compare_count + max(0, compare_count - 1) * (inter_bar_gap / 0.8)
    )
    compare_step = bar_width + inter_bar_gap
    total_span = compare_count * bar_width + max(0, compare_count - 1) * inter_bar_gap

    bar_heights_by_series = []
    bar_absolute_values_by_series = []
    wo_opt_heights = []
    wo_opt_absolute_values = []
    for model_name, squeeze_merge in bar_groups:
        base_summary = (
            model_result.get(model_name, {})
            .get(squeeze_merge, {})
            .get(top_n_exps[0], {})
            .get(hard_window_size)
        )
        if base_summary is None:
            wo_opt_heights.append(0.0)
            wo_opt_absolute_values.append(0.0)
        else:
            wo_opt_heights.append(base_summary["uncaptured_y_rate_wo_opt"] * 100.0)
            wo_opt_absolute_values.append(
                base_summary["sum_total_x_hard_captured_y_count"]
                + base_summary["sum_total_x_hard_uncaptured_y_count"]
            )
    bar_heights_by_series.append(wo_opt_heights)
    bar_absolute_values_by_series.append(wo_opt_absolute_values)

    for top_n_exp in top_n_exps:
        opt_heights = []
        opt_absolute_values = []
        for model_name, squeeze_merge in bar_groups:
            summary = (
                model_result.get(model_name, {})
                .get(squeeze_merge, {})
                .get(top_n_exp, {})
                .get(hard_window_size)
            )
            if summary is None:
                opt_heights.append(0.0)
                opt_absolute_values.append(0.0)
            else:
                if summary["uncaptured_y_rate"] != 0:
                    opt_heights.append(summary["uncaptured_y_rate"] * 100.0)
                else:
                    opt_heights.append(0.2)
                opt_absolute_values.append(summary["sum_total_x_hard_uncaptured_y_count"])
        bar_heights_by_series.append(opt_heights)
        bar_absolute_values_by_series.append(opt_absolute_values)

    for compare_idx, heights in enumerate(bar_heights_by_series):
        offset = -total_span / 2 + bar_width / 2 + compare_idx * compare_step
        bars = ax.bar(
            [x + offset for x in x_positions],
            heights,
            width=bar_width,
            color=compare_colors[compare_idx],
            edgecolor=compare_colors[compare_idx],
            linewidth=0.5,
            label=compare_labels[compare_idx],
        )
        absolute_values = bar_absolute_values_by_series[compare_idx]
        for bar, absolute_value in zip(bars, absolute_values):
            if absolute_value <= 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.6,
                _format_compact_scientific(absolute_value),
                ha="center",
                va="bottom",
                rotation=20,
                fontsize=max(6, GLOBAL_PLOT_FONTSIZE - 1),
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(model_labels)
    ax.tick_params(axis="both", labelsize=GLOBAL_PLOT_FONTSIZE)
    ax.set_ylabel("Uncaptured Rate (%)", fontsize=GLOBAL_PLOT_FONTSIZE + 1)
    baseline_handle = Patch(
        facecolor=compare_colors[0],
        edgecolor=compare_colors[0],
        label=compare_labels[0],
    )
    baseline_legend = ax.legend(
        handles=[baseline_handle],
        frameon=False,
        fontsize=GLOBAL_PLOT_FONTSIZE,
        # title=f"Window={hard_window_size}",
        # title_fontsize=GLOBAL_PLOT_FONTSIZE + 1,
        loc=baseline_legend_loc,
        bbox_to_anchor=baseline_legend_bbox_to_anchor,
    )
    ax.add_artist(baseline_legend)

    compiler_handles = [
        Patch(
            facecolor=compare_colors[idx + 1],
            edgecolor=compare_colors[idx + 1],
            label=compare_labels[idx + 1],
        )
        for idx in range(len(top_ns))
    ]
    ax.legend(
        handles=compiler_handles,
        frameon=False,
        fontsize=GLOBAL_PLOT_FONTSIZE,
        title="With compiler",
        title_fontsize=GLOBAL_PLOT_FONTSIZE,
        loc=compiler_legend_loc,
        bbox_to_anchor=compiler_legend_bbox_to_anchor,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_path}")
    return output_path


if __name__ == "__main__":
    hard_window_sizes = [256, 512, 1024, 2048]  # [128, 256, 512, 1024, 2048, 4096]
    top_ns = [2**14, 2**15, 2**16, 2**17]  #  

    # (model name, squeeze_merge, target layers to process, target layers to duplicate, repeat count)
    model_squeeze_list = [
        ("gpt2-small", "SqueezeMerge_1", [1, 2, 3, 4, 6, 8], [6, 8], 24),
        ("gpt2-medium", "SqueezeMerge_1", [1, 2, 3, 4, 6, 8], [6, 8], 48),
        ("opt-125m", "SqueezeMerge_0", [1,2,3,5,6,7,8,9,11,12,13,14,16,17,19,20,21,22,24,25,26,27,28,30,31,32,33,35,36,38], [20,21,22,24,25,26,27,28,30,31,32,33,35,36,38], 11),
    ]

    model_result = {}  # dict keyed by model_name -> squeeze_merge -> top_n -> hard_window_size
    for model_name, squeeze_merge, target_layers, duplicate_layers, repeat_count in model_squeeze_list:
        for hard_window_size, top_n in itertools.product(hard_window_sizes, top_ns):
            summary = collect_layer_statistics(
                model_name,
                squeeze_merge,
                top_n,
                hard_window_size,
                target_layers,
                duplicate_layers=duplicate_layers,
                repeat_count=repeat_count,
            )
            top_n_exp = summary["top_n_exp"]
            model_result.setdefault(model_name, {}).setdefault(squeeze_merge, {}).setdefault(top_n_exp, {})[
                hard_window_size
            ] = summary

    plot_uncaptured_y_rate_grouped(
        model_result,
        model_squeeze_list,
        hard_window_sizes,
        top_ns,
        output_path="./plots/uncaptured_y_rate.pdf",
        figsize=(6, 3),
    )

    # plot_y_traffic_grouped(
    #     model_result,
    #     model_squeeze_list,
    #     hard_window_sizes,
    #     top_ns,
    #     output_path="./plots/dram_traffic.pdf",
    #     figsize=(6, 2.5),
    # )

    hard_window_size = 1024
    plot_uncaptured_rate_compare_for_hard_window(
        model_result,
        model_squeeze_list,
        hard_window_size,
        top_ns,
        output_path=f"./plots/uncaptured_rate_compare_hw_{hard_window_size}.pdf",
        figsize=(6, 2.5),
        baseline_legend_loc="upper left",
        baseline_legend_bbox_to_anchor=(0.42, 0.88),
        compiler_legend_loc="upper left",
        compiler_legend_bbox_to_anchor=(0.10, 1),
    )


    print("compile_statistic.py end.")
