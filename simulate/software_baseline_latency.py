import re
import json
import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# Pattern to match lines like:
# (jmTimer) Description text time=123.45
JM_TIMER_PATTERN = re.compile(
    r"\(jmTimer\)\s*(.*?)\s*,?\s*time\s*=\s*([+-]?\d+(?:\.\d+)?)\s*([A-Za-z]+)?"
)


UNIT_TO_MS = {
    "s": 1000.0,
    "ms": 1.0,
    "us": 1e-3,
    "ns": 1e-6,
}


MS_TO_UNIT = {
    "s": 1e-3,
    "ms": 1.0,
    "us": 1e3,
    "ns": 1e6,
}


MODEL_NAME_TO_LABEL = {
    "gpt2-small": "GPT2-Small",
    "gpt2-medium": "GPT2-Medium",
    "opt-125m": "OPT-125M",
}


NN_CREATE_LAYER_PATTERN = re.compile(r"neuralNetwork::create,\s*layer\s+\d+/\d+")
GKR_NONFC_PATTERN = re.compile(r"verifier::verifyGKR,\s*GKRlayer\s+\d+,\s*nonFC")
GKR_FCONN_PATTERN = re.compile(r"verifier::verifyGKR,\s*GKRlayer\s+\d+,\s*FCONN")
REPO_ROOT = Path(__file__).resolve().parent.parent


legend_translate = {
    'logup commit': 'Range Pr Commit',
    'logup open': 'Range Pr Open',
    "GKR Open": "GKR Open",
    'logup prove': 'Range Pr',
    'GKR nonFC layer sumcheck': 'GKR Gates Pr',
    'GKR matmul layer sumcheck': 'GKR Matmuls Pr',
    "GKR last sumcheck": "GKR Comb Pr",
}


def _convert_to_ms(time_value, time_unit):
    normalized_unit = (time_unit or "ms").strip().lower()
    if normalized_unit not in UNIT_TO_MS:
        raise ValueError(f"Unsupported jmTimer unit: {time_unit}")
    return time_value * UNIT_TO_MS[normalized_unit]


def _convert_ms_to_unit(time_value_ms, unit="s"):
    normalized_unit = (unit or "s").strip().lower()
    if normalized_unit not in MS_TO_UNIT:
        raise ValueError(f"Unsupported plot unit: {unit}")
    return time_value_ms * MS_TO_UNIT[normalized_unit]


def load_jm_timer_breakdown_from_log(log_path):
    """
    Load a log file and capture all '(jmTimer)' lines.

    Return a dict:
        description -> [{"time": time_1_in_ms, "unit": "ms"}, ...]
    where description is the text after '(jmTimer)' and before 'time='.
    """
    log_path = Path(log_path)
    if not log_path.exists():
        raise FileNotFoundError(log_path)

    breakdown = {}
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        total = log_path.stat().st_size
        with tqdm(total=total, unit="B", unit_scale=True, desc=f"Reading {log_path.name}") as pbar:
            for line in f:
                pbar.update(len(line.encode("utf-8")))
                match = JM_TIMER_PATTERN.search(line)
                if not match:
                    continue

                description = match.group(1).strip()
                time_value = float(match.group(2))
                time_unit = match.group(3) or ""
                time_value_ms = _convert_to_ms(time_value, time_unit)
                breakdown.setdefault(description, []).append(
                    {"time": time_value_ms, "unit": "ms"}
                )

    return breakdown


def load_software_latency_breakdown(model_name, squeeze, log_filename="software_cout.log"):
    log_path = REPO_ROOT / "output" / model_name / squeeze / log_filename
    return load_jm_timer_breakdown_from_log(log_path)


def load_software_latency_breakdown_multi(model_squeeze_list, log_filename="software_cout.log"):
    """
    Load multiple software latency logs.

    Args:
        model_squeeze_list:
            iterable of (model_name, squeeze) tuples

    Returns:
        dict[str, dict] keyed by model label.
        If the same model name appears with different squeeze modes, the label is
        formatted as '{model_name} ({squeeze})'.
    """
    labels = []
    model_counts = {}
    for model_name, _ in model_squeeze_list:
        model_counts[model_name] = model_counts.get(model_name, 0) + 1

    for model_name, squeeze in model_squeeze_list:
        if model_counts[model_name] > 1:
            labels.append(f"{model_name} ({squeeze})")
        else:
            labels.append(model_name)

    result = {}
    for label, (model_name, squeeze) in zip(labels, model_squeeze_list):
        result[label] = load_software_latency_breakdown(model_name, squeeze, log_filename)
    return result


def get_latency_category_method_style_1():
    return {
        "logup commit": lambda desc: (
            "range_prover::logup" in desc and "commit" in desc
        ),
        "logup open": lambda desc: (
            "range_prover::logup" in desc and "open" in desc
        ),
        "logup prove": lambda desc: (
            "range_prover::logup" in desc
            and "commit" not in desc
            and "open" not in desc
        ),
        "nn create commit": lambda desc: (
            "neuralNetwork::create, commit input" in desc
        ),
        "nn create compute_exp_table": lambda desc: (
            "neuralNetwork::create, compute_e_table" in desc
        ),
        "nn create layers": lambda desc: (
            NN_CREATE_LAYER_PATTERN.search(desc) is not None
        ),
        "nn create merge_layer": lambda desc: (
            "merge_layer" in desc
        ),
        "nn create init subset": lambda desc: (
            "initSubset" in desc
        ),
        "GKR nonFC layer sumcheck": lambda desc: (
            GKR_NONFC_PATTERN.search(desc) is not None
        ),
        "GKR matmul layer sumcheck": lambda desc: (
            GKR_FCONN_PATTERN.search(desc) is not None
        ),
        "GKR last sumcheck": lambda desc: (
            "verifier::verifyLasso" in desc
        ),
        "GKR Open": lambda desc: (
            "verifier::openCommit" in desc
        ),
    }


LATENCY_CATEGORY_METHODS = {
    "style_1": get_latency_category_method_style_1,
}


def resolve_latency_category_method(category_style):
    if category_style is None:
        category_style = "style_1"

    if isinstance(category_style, dict):
        return category_style

    if isinstance(category_style, str):
        if category_style not in LATENCY_CATEGORY_METHODS:
            supported = ", ".join(sorted(LATENCY_CATEGORY_METHODS))
            raise ValueError(
                f"Unknown category_style '{category_style}'. Supported styles: {supported}"
            )
        return LATENCY_CATEGORY_METHODS[category_style]()

    raise TypeError(
        "category_style must be None, a style name string, or a dict[str, callable]"
    )


def categorize_jm_timer_latency(time_breakdown_all, category_style="style_1"):
    """
    Aggregate jmTimer results into user-defined categories.

    Args:
        time_breakdown_all:
            output of load_jm_timer_breakdown_from_log(...)
        category_style:
            - style name string, e.g. 'style_1'
            - dict[str, callable(description)->bool] for a custom scheme
            - None, which defaults to 'style_1'

    Returns:
        dict[str, float] where each value is total latency in ms.
    """
    category_method = resolve_latency_category_method(category_style)

    categorized = {category_name: 0.0 for category_name in category_method}

    for description, entries in tqdm(
        time_breakdown_all.items(),
        total=len(time_breakdown_all),
        desc="Categorizing jmTimer latency",
    ):
        total_time_ms = sum(entry["time"] for entry in entries)
        for category_name, matcher in category_method.items():
            if matcher(description):
                categorized[category_name] += total_time_ms

    return categorized


def categorize_jm_timer_latency_multi(time_breakdown_all_multi, category_style="style_1"):
    """
    Categorize multiple jmTimer breakdown dicts.

    Args:
        time_breakdown_all_multi:
            dict[str, dict] keyed by bar/model label

    Returns:
        dict[str, dict[str, float]] keyed by bar/model label
    """
    return {
        label: categorize_jm_timer_latency(breakdown, category_style=category_style)
        for label, breakdown in time_breakdown_all_multi.items()
    }


def select_latency_breakdown_for_chart(
    time_breakdown_categorized_ms,
    selected_keys,
    include_missing=False,
):
    """
    Filter a categorized latency dict down to the keys to be plotted.

    Args:
        time_breakdown_categorized_ms:
            dict[str, float], latency values in ms
        selected_keys:
            iterable of category names in the desired output order
        include_missing:
            if True, missing keys are inserted as 0.0; otherwise they are skipped

    Returns:
        dict[str, float] preserving the order of selected_keys
    """
    selected = {}
    for key in selected_keys:
        if key in time_breakdown_categorized_ms:
            selected[key] = time_breakdown_categorized_ms[key]
        elif include_missing:
            selected[key] = 0.0
    return selected


def select_latency_breakdown_for_chart_multi(
    time_breakdown_categorized_multi_ms,
    selected_keys,
    include_missing=False,
):
    """
    Filter multiple categorized latency dicts down to the keys to be plotted.

    Args:
        time_breakdown_categorized_multi_ms:
            dict[str, dict[str, float]]

    Returns:
        dict[str, dict[str, float]]
    """
    return {
        label: select_latency_breakdown_for_chart(
            breakdown,
            selected_keys,
            include_missing=include_missing,
        )
        for label, breakdown in time_breakdown_categorized_multi_ms.items()
    }


def dump_software_latency_breakdown_multi(
    model_squeeze_list,
    time_breakdown_all_multi_ms,
    time_breakdown_categorized_multi_ms,
    output_root="./sim_data/software_baseline_latency",
):
    output_root = REPO_ROOT / output_root
    labels = list(time_breakdown_all_multi_ms.keys())
    for label, (model_name, squeeze) in zip(labels, model_squeeze_list):
        output_dir = output_root / model_name / squeeze
        output_dir.mkdir(parents=True, exist_ok=True)

        all_path = output_dir / "time_breakdown_all_multi_ms.json"
        categorized_path = output_dir / "time_breakdown_categorized_multi_ms.json"

        all_payload = {
            "model_name": model_name,
            "squeeze": squeeze,
            "label": label,
            "time_breakdown_all_multi_ms": time_breakdown_all_multi_ms[label],
        }
        categorized_payload = {
            "model_name": model_name,
            "squeeze": squeeze,
            "label": label,
            "time_breakdown_categorized_multi_ms": time_breakdown_categorized_multi_ms[label],
        }

        all_path.write_text(json.dumps(all_payload, indent=2), encoding="utf-8")
        categorized_path.write_text(json.dumps(categorized_payload, indent=2), encoding="utf-8")
        print(f"Dumped software latency breakdown to {all_path}")
        print(f"Dumped categorized software latency breakdown to {categorized_path}")


def draw_stacked_bar_chart(
    data_ms,
    unit="s",
    title=None,
    ylabel=None,
    xlabel=None,
    save_path=None,
    show=True,
    figsize=(10, 6),
    bar_width=0.4,
    reverse_legend=True,
    show_total_labels=True,
    legend_ncol=None,
    legend_nrow=None,
    pie_chart_bar_label=None,
    pie_chart_title=None,
    selected_add_dot_in_pie=None,
    selected_dot_hatch=".",
    selected_dot_linewidth=0.6,
    selected_dot_color="lightgray",
    pie_wedge_linewidth=0.0,
    subplot_width_ratios=(2.2, 1.0),
    subplot_wspace=0.15,
    legend_loc="lower center",
    legend_bbox_to_anchor=(0.5, 1.02),
    global_fontsize=11,
):
    """
    Draw a stacked bar chart from latency data stored in ms.

    Supported input formats:
    1. Flat dict[str, float]:
       {"segment_a": ms_a, "segment_b": ms_b, ...}
       This draws one stacked bar.

    2. Nested dict[str, dict[str, float]]:
       {
           "bar_1": {"segment_a": ms_a, "segment_b": ms_b, ...},
           "bar_2": {"segment_a": ms_a, "segment_b": ms_b, ...},
       }
       This draws multiple stacked bars.
    """
    if not data_ms:
        raise ValueError("data_ms is empty")

    first_value = next(iter(data_ms.values()))
    if isinstance(first_value, dict):
        bar_data_ms = data_ms
    else:
        bar_data_ms = {"Latency": data_ms}

    bar_labels = list(bar_data_ms.keys())
    # Map known model names to prettier labels for the x-axis. This also
    # handles labels that include the squeeze suffix, e.g. "gpt2-small (SqueezeMerge_1)".
    def _map_model_label(label):
        for model_key, pretty in MODEL_NAME_TO_LABEL.items():
            if label == model_key:
                return pretty
            # handle cases like 'gpt2-small (SqueezeMerge_1)' or 'gpt2-small (squeeze)'
            if label.startswith(model_key + " ") or label.startswith(model_key + "("):
                return pretty
        return label

    display_labels = [_map_model_label(lbl) for lbl in bar_labels]
    segment_order = []
    seen = set()
    for segments in bar_data_ms.values():
        for segment_name in segments:
            if segment_name not in seen:
                seen.add(segment_name)
                segment_order.append(segment_name)

    x_positions = list(range(len(bar_labels)))
    bottoms = [0.0] * len(bar_labels)

    segment_colors = [plt.cm.tab10(idx % 10) for idx in range(len(segment_order))]

    if pie_chart_bar_label is not None:
        fig, axes = plt.subplots(
            1,
            2,
            figsize=figsize,
            gridspec_kw={"width_ratios": list(subplot_width_ratios)},
        )
        ax = axes[0]
        pie_ax = axes[1]
        fig.subplots_adjust(wspace=subplot_wspace)
    else:
        fig, ax = plt.subplots(figsize=figsize)
        pie_ax = None

    for segment_name, color in zip(segment_order, segment_colors):
        heights = [
            _convert_ms_to_unit(bar_data_ms[bar_label].get(segment_name, 0.0), unit)
            for bar_label in bar_labels
        ]
        ax.bar(
            x_positions,
            heights,
            bar_width,
            bottom=bottoms,
            label=segment_name,
            align="center",
            color=color,
        )
        bottoms = [bottom + height for bottom, height in zip(bottoms, heights)]

    # add horizontal padding so bars don't appear too wide against plot edges
    if x_positions:
        ax.set_xlim(x_positions[0] - bar_width, x_positions[-1] + bar_width)

    if show_total_labels:
        for x, total_height in zip(x_positions, bottoms):
            ax.text(
                x,
                total_height,
                f"{total_height:.1f}",
                ha="center",
                va="bottom",
                fontsize=global_fontsize,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(display_labels, rotation=0)
    ax.tick_params(axis="both", labelsize=global_fontsize)
    ax.set_ylabel(ylabel or f"Latency ({unit})", fontsize=global_fontsize + 1)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=global_fontsize + 1)
    if title is not None:
        ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    if reverse_legend:
        handles = handles[::-1]
        labels = labels[::-1]
    translated_labels = [legend_translate.get(label, label) for label in labels]
    if legend_ncol is not None and legend_nrow is not None:
        raise ValueError("Only one of legend_ncol or legend_nrow should be provided")
    if legend_ncol is None:
        if legend_nrow is not None:
            legend_ncol = max(1, math.ceil(len(translated_labels) / legend_nrow))
        else:
            legend_ncol = max(1, len(translated_labels))
    legend = fig.legend(
        handles,
        translated_labels,
        frameon=False,
        loc=legend_loc,
        bbox_to_anchor=legend_bbox_to_anchor,
        ncol=legend_ncol,
        fontsize=global_fontsize,
    )

    if pie_ax is not None:
        selected_add_dot_in_pie = set(selected_add_dot_in_pie or [])
        pie_bar_label = None
        for raw_label, display_label in zip(bar_labels, display_labels):
            if pie_chart_bar_label in {raw_label, display_label}:
                pie_bar_label = raw_label
                break
        if pie_bar_label is None:
            raise ValueError(f"pie_chart_bar_label '{pie_chart_bar_label}' not found in bars")

        pie_values = [
            _convert_ms_to_unit(bar_data_ms[pie_bar_label].get(segment_name, 0.0), unit)
            for segment_name in segment_order
        ]
        pie_labels = [legend_translate.get(segment_name, segment_name) for segment_name in segment_order]
        nonzero_items = [
            (segment_name, label, value, color)
            for segment_name, label, value, color in zip(segment_order, pie_labels, pie_values, segment_colors)
            if value > 0
        ]
        if nonzero_items:
            filtered_segment_names, _, filtered_values, filtered_colors = zip(*nonzero_items)
            wedges, _, autotexts = pie_ax.pie(
                filtered_values,
                colors=filtered_colors,
                autopct="%1.1f%%",
                startangle=90,
                counterclock=False,
                textprops={"fontsize": global_fontsize},
                wedgeprops={"linewidth": pie_wedge_linewidth, "edgecolor": "none"},
            )
            total_value = sum(filtered_values)
            for wedge, autotext, value in zip(wedges, autotexts, filtered_values):
                percentage = (value / total_value * 100.0) if total_value > 0 else 0.0
                if percentage < 5.0:
                    angle = 0.5 * (wedge.theta1 + wedge.theta2)
                    angle_rad = np.deg2rad(angle)
                    radius = 1.18
                    x = radius * np.cos(angle_rad)
                    y = radius * np.sin(angle_rad)
                    autotext.set_position((x, y))
                    autotext.set_ha("left" if x >= 0 else "right")
                    autotext.set_va("center")
            for wedge, segment_name in zip(wedges, filtered_segment_names):
                if segment_name in selected_add_dot_in_pie:
                    wedge.set_hatch(selected_dot_hatch)
                    wedge.set_edgecolor(selected_dot_color)
                    wedge.set_linewidth(selected_dot_linewidth)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_extra_artists=[legend], bbox_inches="tight")
        print(f"Saved chart to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


if __name__ == "__main__":
    model_squeeze_list = [
        ("gpt2-small", "SqueezeMerge_1"),
        ("opt-125m", "SqueezeMerge_0"),
        ("gpt2-medium", "SqueezeMerge_1"),
    ]

    time_breakdown_all_multi_ms = load_software_latency_breakdown_multi(model_squeeze_list)
    time_breakdown_categorized_multi_ms = categorize_jm_timer_latency_multi(
        time_breakdown_all_multi_ms,
        category_style="style_1",
    )
    dump_software_latency_breakdown_multi(
        model_squeeze_list,
        time_breakdown_all_multi_ms,
        time_breakdown_categorized_multi_ms,
    )
    selected_bars_for_chart = [
        'logup commit',
        'logup open',
        "GKR Open",
        'logup prove',
        'GKR nonFC layer sumcheck',
        'GKR matmul layer sumcheck',        
        "GKR last sumcheck",
    ]
    legend_groups = [
        ['logup commit',
        'logup open',
        "GKR Open",],
        ['logup prove',
        'GKR nonFC layer sumcheck',
        'GKR matmul layer sumcheck',        
        "GKR last sumcheck"],
    ]
    selected_add_dot_in_pie = [
        'logup prove',
        'GKR nonFC layer sumcheck',
        'GKR matmul layer sumcheck',        
        "GKR last sumcheck",
    ]
    selected_to_plot_bar_chart = select_latency_breakdown_for_chart_multi(
        time_breakdown_categorized_multi_ms,
        selected_bars_for_chart,
    )
    save_path = "./plots/software_baseline_latency_breakdown.pdf"
    draw_stacked_bar_chart(
        selected_to_plot_bar_chart,
        unit="s",
        title=None,
        save_path=save_path,
        figsize=(8, 4),
        bar_width=0.4,
        show=False,
        reverse_legend=True,
        show_total_labels=True,
        pie_chart_bar_label="gpt2-small",
        pie_chart_title="GPT2-Small",
        selected_add_dot_in_pie=selected_add_dot_in_pie,
        subplot_width_ratios=(1.2, 1.0),
        subplot_wspace=-0.5,
        legend_ncol=3,
        legend_loc="lower center",
        legend_bbox_to_anchor=(0.5, 0.85),
        global_fontsize=14,
        selected_dot_hatch=".",
        selected_dot_linewidth=0,
        pie_wedge_linewidth=0.0,
    )


    print("Capture software baseline latency end...")
