import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def _save_figure(fig, save_path, save_dpi, rasterize_points):
    fig.canvas.draw()
    if rasterize_points:
        fig.savefig(save_path, dpi=save_dpi)
    else:
        fig.savefig(save_path, dpi=save_dpi, bbox_inches="tight")
    print(f"Plot saved to {save_path}")


def _apply_scientific_ticks(ax, scientific_ticks):
    if not scientific_ticks:
        return
    formatter_x = ScalarFormatter(useMathText=True)
    formatter_x.set_scientific(True)
    formatter_x.set_powerlimits((0, 0))
    ax.xaxis.set_major_formatter(formatter_x)

    formatter_y = ScalarFormatter(useMathText=True)
    formatter_y.set_scientific(True)
    formatter_y.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(formatter_y)

    ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))


def _set_offset_text_fontsize(ax, tick_fontsize):
    ax.xaxis.get_offset_text().set_fontsize(tick_fontsize)
    ax.yaxis.get_offset_text().set_fontsize(tick_fontsize)


def _validate_range(range_value, name):
    if len(range_value) != 2:
        raise ValueError(f"{name} must be a (start, end) tuple")
    range_start, range_end = range_value
    if range_start >= range_end:
        raise ValueError(f"{name} must satisfy start < end")
    return range_start, range_end


def _load_p1_relation_records(filepath, lu_value):
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(filepath)

    with filepath.open("r", encoding="utf-8", errors="replace") as f:
        print(f"Reading data from {filepath}...")
        line_count = sum(1 for _ in f)
        f.seek(0)
        lines = f.readlines()

    header = None
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("sumcheck_id"):
            header = line
            start_idx = i + 1
            break
    if header is None:
        print("Header not found in file.")
        return

    header_cols = header.split()
    idx_lu = header_cols.index("lu/idx")
    idx_gateu = header_cols.index("gate.u")
    idx_gateg = header_cols.index("gate.g")

    records = []
    with tqdm(total=line_count - start_idx, desc="Reading and processing log file") as pbar:
        for line in lines[start_idx:]:
            pbar.update(1)
            if not line.strip() or line.strip().startswith("sumcheck_id"):
                continue
            cols = line.split()
            if len(cols) <= max(idx_lu, idx_gateu, idx_gateg):
                continue
            try:
                lu = int(cols[idx_lu])
                gateu = int(cols[idx_gateu])
                gateg = int(cols[idx_gateg])
            except ValueError:
                continue
            if lu != lu_value:
                continue
            records.append((gateu, gateg))

    return records


def _load_p2_relation_records(filepath, lv_value, relation_name):
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(filepath)

    with filepath.open("r", encoding="utf-8", errors="replace") as f:
        print(f"Reading data from {filepath}...")
        line_count = sum(1 for _ in f)
        f.seek(0)
        lines = f.readlines()

    header = None
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("sumcheck_id"):
            header = line
            start_idx = i + 1
            break
    if header is None:
        raise ValueError("Header not found in file.")

    header_cols = header.split()
    idx_lv = header_cols.index("lv/idx")
    idx_gateu = header_cols.index("gate.u")
    idx_gateg = header_cols.index("gate.g")
    idx_gatev = header_cols.index("gate.v")

    records = []
    with tqdm(total=line_count - start_idx, desc="Reading P2 relation log") as pbar:
        for line in lines[start_idx:]:
            pbar.update(1)
            if not line.strip() or line.strip().startswith("sumcheck_id"):
                continue
            cols = line.split()
            if len(cols) <= max(idx_lv, idx_gateu, idx_gateg, idx_gatev):
                continue
            try:
                lv = int(cols[idx_lv])
                gateu = int(cols[idx_gateu])
                gateg = int(cols[idx_gateg])
                gatev = int(cols[idx_gatev])
            except ValueError:
                continue
            if lv != lv_value:
                continue

            if relation_name == "unary_u_g":
                if gatev == -1:
                    records.append((gateu, gateg))
            elif relation_name == "binary_v_u":
                if gatev != -1:
                    records.append((gatev, gateu))
            elif relation_name == "binary_v_g":
                if gatev != -1:
                    records.append((gatev, gateg))
            else:
                raise ValueError(
                    "relation_name must be one of: unary_u_g, binary_v_u, binary_v_g"
                )

    return records


def _draw_relation_with_inset(
    ax,
    records,
    x_label,
    y_label,
    inset_x_range,
    subplot_title=None,
    inset_y_range=None,
    main_x_range=None,
    inset_position="upper right",
    inset_width="35%",
    inset_height="35%",
    inset_borderpad=1.0,
    inset_offset=(0.0, 0.0),
    dot_size=1,
    inset_dot_size=4,
    tick_fontsize=12,
    color="C0",
    box_color="red",
    rasterize_points=True,
    scientific_ticks=False,
):
    zoom_x_start, zoom_x_end = _validate_range(inset_x_range, "inset_x_range")
    if inset_y_range is not None:
        inset_y_start, inset_y_end = _validate_range(inset_y_range, "inset_y_range")
    if main_x_range is not None:
        main_x_start, main_x_end = _validate_range(main_x_range, "main_x_range")

    overlap_count = defaultdict(int)
    x_vals = []
    y_vals = []
    zoom_x_vals = []
    zoom_y_vals = []

    for x_value, y_value in tqdm(records, desc=f"Processing {x_label}->{y_label} records for plotting"):
        key = (x_value, y_value)
        count = overlap_count[key]
        overlap_count[key] += 1
        y_plot = y_value + count * 0.2

        x_vals.append(x_value)
        y_vals.append(y_plot)
        if zoom_x_start <= x_value < zoom_x_end:
            zoom_x_vals.append(x_value)
            zoom_y_vals.append(y_plot)

    if not x_vals:
        raise ValueError(f"No valid records found for {x_label}->{y_label}.")

    ax.scatter(
        x_vals,
        y_vals,
        s=dot_size,
        alpha=0.7,
        color=color,
        rasterized=rasterize_points,
    )
    ax.set_ylabel(y_label, fontsize=tick_fontsize + 2)
    ax.set_xlabel(x_label, fontsize=tick_fontsize + 2)
    if subplot_title is not None:
        ax.set_title(subplot_title, fontsize=tick_fontsize + 2)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.tick_params(labelsize=tick_fontsize)
    _apply_scientific_ticks(ax, scientific_ticks)
    _set_offset_text_fontsize(ax, tick_fontsize)
    if main_x_range is not None:
        ax.set_xlim(main_x_start, main_x_end)
    else:
        ax.set_xlim(0, max(x_vals))

    inset_ax = inset_axes(
        ax,
        width=inset_width,
        height=inset_height,
        loc=inset_position,
        bbox_to_anchor=(
            inset_offset[0],
            inset_offset[1],
            1.0,
            1.0,
        ),
        bbox_transform=ax.transAxes,
        borderpad=inset_borderpad,
    )
    inset_ax.scatter(
        zoom_x_vals,
        zoom_y_vals,
        s=inset_dot_size,
        alpha=0.8,
        color=color,
        rasterized=rasterize_points,
    )
    inset_ax.set_xlim(zoom_x_start, zoom_x_end)
    if inset_y_range is not None:
        inset_ax.set_ylim(inset_y_start, inset_y_end)
        box_y_start = inset_y_start
        box_y_end = inset_y_end
    elif zoom_y_vals:
        zoom_y_min = min(zoom_y_vals)
        zoom_y_max = max(zoom_y_vals)
        if zoom_y_min == zoom_y_max:
            y_pad = 1.0
        else:
            y_pad = max(1.0, (zoom_y_max - zoom_y_min) * 0.05)
        box_y_start = zoom_y_min - y_pad
        box_y_end = zoom_y_max + y_pad
        inset_ax.set_ylim(box_y_start, box_y_end)
    else:
        box_y_start = None
        box_y_end = None
    inset_ax.grid(True, linestyle="--", alpha=0.4)
    inset_ax.tick_params(labelsize=tick_fontsize)
    _apply_scientific_ticks(inset_ax, scientific_ticks)
    _set_offset_text_fontsize(inset_ax, tick_fontsize)

    if box_y_start is not None and box_y_end is not None:
        ax.add_patch(
            Rectangle(
                (zoom_x_start, box_y_start),
                zoom_x_end - zoom_x_start,
                box_y_end - box_y_start,
                fill=False,
                linestyle="--",
                linewidth=1.2,
                edgecolor=box_color,
            )
        )


def plot_gate_u_g_P1_relation_with_inset(
    filepath,
    lu_value,
    inset_x_range,
    subplot_title=None,
    inset_y_range=None,
    main_x_range=None,
    inset_position="upper right",
    inset_width="35%",
    inset_height="35%",
    inset_borderpad=1.0,
    inset_offset=(0.0, 0.0),
    dot_size=1,
    inset_dot_size=4,
    tick_fontsize=12,
    box_color="red",
    rasterize_points=True,
    scientific_ticks=False,
    save_dpi=200,
    save_path=None,
):
    records = _load_p1_relation_records(filepath, lu_value)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))
    _draw_relation_with_inset(
        ax=ax,
        records=records,
        x_label="u",
        y_label="g of eq(g)",
        subplot_title=subplot_title,
        inset_x_range=inset_x_range,
        inset_y_range=inset_y_range,
        main_x_range=main_x_range,
        inset_position=inset_position,
        inset_width=inset_width,
        inset_height=inset_height,
        inset_borderpad=inset_borderpad,
        inset_offset=inset_offset,
        dot_size=dot_size,
        inset_dot_size=inset_dot_size,
        tick_fontsize=tick_fontsize,
        color={0: "C0", 1: "C1"}.get(lu_value, "C0"),
        box_color=box_color,
        rasterize_points=rasterize_points,
        scientific_ticks=scientific_ticks,
    )
    fig.tight_layout()
    if save_path:
        _save_figure(fig, save_path, save_dpi, rasterize_points)
    plt.show()


def plot_gate_u_g_P1_P2_relations_with_insets(
    p1_filepath,
    p1_lu_value,
    p1_inset_x_range,
    p1_inset_y_range,
    p1_main_x_range,
    p2_filepath,
    p2_lv_value,
    p2_relation_name,
    p2_inset_x_range,
    p2_inset_y_range,
    p2_main_x_range,
    p1_subplot_title=None,
    p2_subplot_title=None,
    inset_position="upper right",
    p1_inset_position=None,
    p2_inset_position=None,
    inset_width="35%",
    inset_height="35%",
    inset_borderpad=1.0,
    p1_inset_offset=(0.0, 0.0),
    p2_inset_offset=(0.0, 0.0),
    dot_size=1,
    inset_dot_size=4,
    tick_fontsize=12,
    box_color="red",
    rasterize_points=True,
    scientific_ticks=False,
    save_dpi=200,
    save_path=None,
):
    p1_records = _load_p1_relation_records(p1_filepath, p1_lu_value)
    p2_records = _load_p2_relation_records(p2_filepath, p2_lv_value, p2_relation_name)
    if p1_inset_position is None:
        p1_inset_position = inset_position
    if p2_inset_position is None:
        p2_inset_position = inset_position

    p2_axis_labels = {
        "unary_u_g": ("u", "g of eq(g)"),
        "binary_v_u": ("v", "u"),
        "binary_v_g": ("v", "g of eq(g)"),
    }
    p2_x_label, p2_y_label = p2_axis_labels[p2_relation_name]

    fig, axes = plt.subplots(1, 2, figsize=(19, 8))
    _draw_relation_with_inset(
        ax=axes[0],
        records=p1_records,
        x_label="u",
        y_label="g of eq(g)",
        subplot_title=p1_subplot_title,
        inset_x_range=p1_inset_x_range,
        inset_y_range=p1_inset_y_range,
        main_x_range=p1_main_x_range,
        inset_position=p1_inset_position,
        inset_width=inset_width,
        inset_height=inset_height,
        inset_borderpad=inset_borderpad,
        inset_offset=p1_inset_offset,
        dot_size=dot_size,
        inset_dot_size=inset_dot_size,
        tick_fontsize=tick_fontsize,
        color={0: "C0", 1: "C1"}.get(p1_lu_value, "C0"),
        box_color=box_color,
        rasterize_points=rasterize_points,
        scientific_ticks=scientific_ticks,
    )
    _draw_relation_with_inset(
        ax=axes[1],
        records=p2_records,
        x_label="v",
        y_label="u of eq(r, u)",
        subplot_title=p2_subplot_title,
        inset_x_range=p2_inset_x_range,
        inset_y_range=p2_inset_y_range,
        main_x_range=p2_main_x_range,
        inset_position=p2_inset_position,
        inset_width=inset_width,
        inset_height=inset_height,
        inset_borderpad=inset_borderpad,
        inset_offset=p2_inset_offset,
        dot_size=dot_size,
        inset_dot_size=inset_dot_size,
        tick_fontsize=tick_fontsize,
        color={0: "C0", 1: "C1"}.get(p2_lv_value, "C0"),
        box_color=box_color,
        rasterize_points=rasterize_points,
        scientific_ticks=scientific_ticks,
    )

    fig.tight_layout()
    if save_path:
        _save_figure(fig, save_path, save_dpi, rasterize_points)
    plt.show()


if __name__ == "__main__":
    SqueezeMerge = "SqueezeMerge_1"
    ModelName = "gpt2-small"

    p1_data_file = (
        Path(f"../output/{ModelName}/{SqueezeMerge}/{ModelName}_initP1_mult_array")
        / f"{ModelName}_initP1_mult_array_layers_6.log"
    )
    p2_data_file = (
        Path(f"../output/{ModelName}/{SqueezeMerge}/{ModelName}_initP2_mult_array")
        / f"{ModelName}_initP2_mult_array_layers_4.log"
    )
    plot_gate_u_g_P1_P2_relations_with_insets(
        p1_filepath=p1_data_file,
        p1_lu_value=0,
        p1_subplot_title="A Rounding Layer",
        p1_inset_x_range=(-700, 20000),
        p1_inset_y_range=(0, 20000),
        p1_main_x_range=(-4000, 262144),
        p2_filepath=p2_data_file,
        p2_lv_value=0,
        p2_relation_name="binary_v_u",
        p2_subplot_title="A Masked QK Layer",
        p2_inset_x_range=(0, 1000),
        p2_inset_y_range=(295000, 295000 + 1000),
        p2_main_x_range=(-500, 25000),
        p1_inset_position="upper right",
        p2_inset_position="lower right",
        p1_inset_offset=(-0.02, -0.06),
        p2_inset_offset=(0.0, 0.15),
        inset_width="40%",
        inset_height="40%",
        dot_size=0.8,
        inset_dot_size=0.8,
        tick_fontsize=26,
        box_color="red",
        rasterize_points=True,
        scientific_ticks=True,
        save_path=f"./plots/{ModelName}_{SqueezeMerge}_gate_u_g_p1_p2_relations_with_inset.pdf",
    )

    print("gate_u_g_relation_plot.py end...")
