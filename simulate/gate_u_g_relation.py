import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm


def plot_gate_u_g_P1_relation(
    filepath=f"../output/gpt2-small_initP1_mult_array/gpt2-small_initP1_mult_array_layers_100.log",
    save_path: str = None,
):
    """
    Plots the relation between (lu/idx, gate.u) and gate.g from the log file.
    Overlapping points are shifted up slightly to avoid overlap.
    """

    overlap_count = defaultdict(int)

    filepath = Path(filepath)
    if not filepath.exists():
        print(f"File {filepath} does not exist.")
        return
    with open(filepath, 'r') as f:
        print(f"Reading data from {filepath}...")
        line_count = sum(1 for _ in f)
        f.seek(0)
        lines = f.readlines()

    # Find header and column indices
    header = None
    for i, line in enumerate(lines):
        if line.strip().startswith('sumcheck_id'):
            header = line
            start_idx = i + 1
            break
    if header is None:
        print("Header not found in file.")
        return

    header_cols = header.split()
    idx_lu = header_cols.index('lu/idx')
    idx_gateu = header_cols.index('gate.u')
    idx_gateg = header_cols.index('gate.g')

    # Single pass over file: collect records and track max gate.u
    records = []  # list of (lu, gateu, gateg)
    max_gateu = -1
    with tqdm(total=line_count - start_idx, desc="Reading and processing log file") as pbar:
        for _, line in enumerate(lines[start_idx:]):
            pbar.update(1)
            if not line.strip() or line.strip().startswith('sumcheck_id'):
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
            records.append((lu, gateu, gateg))
            if gateu > max_gateu:
                max_gateu = gateu

    if not records:
        print("No valid gate records found.")
        return

    # Build plot points grouped by lu using gate.u directly as x-axis.
    x_vals_by_lu = defaultdict(list)
    y_vals_by_lu = defaultdict(list)
    for lu, gateu, gateg in tqdm(records, desc="Processing records for plotting"):
        key = (lu, gateu, gateg)
        count = overlap_count[key]
        overlap_count[key] += 1
        y_plot = gateg + count * 0.2  # shift up by 0.2 per overlap
        x_vals_by_lu[lu].append(gateu)
        y_vals_by_lu[lu].append(y_plot)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    colors = {0: 'C0', 1: 'C1'}
    lu_order = [0, 1]

    for ax, lu_key in zip(axes, lu_order):
        ax.scatter(
            x_vals_by_lu.get(lu_key, []),
            y_vals_by_lu.get(lu_key, []),
            s=1,
            alpha=0.7,
            color=colors.get(lu_key, None),
        )
        ax.set_ylabel('gate.g')
        ax.set_title(f'lu={lu_key}')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlim(0, max_gateu)

    axes[-1].set_xlabel('gate.u')
    fig.suptitle('Scatter plot of gate.u vs gate.g, split by lu')
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    plt.show()


def plot_gate_u_g_P2_relation(
    filepath=f"../output/gpt2-small_initP2_mult_array/gpt2-small_initP2_mult_array_layers_100.log",
    save_path: str = None,
):
    """
    Parse P2 log and draw five subplots
    """

    filepath = Path(filepath)
    if not filepath.exists():
        print(f"File {filepath} does not exist.")
        return

    with open(filepath, 'r') as f:
        print(f"Reading data from {filepath}...")
        lines = f.readlines()

    # Find header and column indices
    header = None
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('sumcheck_id'):
            header = line
            start_idx = i + 1
            break
    if header is None:
        print("Header not found in file.")
        return

    header_cols = header.split()
    try:
        idx_lv = header_cols.index('lv/idx')
        idx_gateu = header_cols.index('gate.u')
        idx_gateg = header_cols.index('gate.g')
        idx_gatev = header_cols.index('gate.v')
    except ValueError as e:
        print("Expected header columns not found:", e)
        return

    records = []  # tuples (lv, gateu, gateg, gatev)
    for line in tqdm(lines[start_idx:], desc="Reading P2 log"):
        if not line.strip() or line.strip().startswith('sumcheck_id'):
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
        records.append((lv, gateu, gateg, gatev))

    if not records:
        print("No valid gate records found.")
        return

    # Split records based on gate.v and lv with progress bar
    vneg_lv0 = []
    vneg_lv1 = []
    lv0_vpos = []
    lv1_vpos = []
    for lv, gateu, gateg, gatev in tqdm(records, desc="Splitting records into arrays"):
        if gatev == -1:
            if lv == 0:
                vneg_lv0.append((gateu, gateg))
            elif lv == 1:
                vneg_lv1.append((gateu, gateg))
        else:
            if lv == 0:
                lv0_vpos.append((gatev, gateu, gateg))
            elif lv == 1:
                lv1_vpos.append((gatev, gateu, gateg))

    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])

    ax_top_lv0 = fig.add_subplot(gs[0, 0])
    ax_top_lv1 = fig.add_subplot(gs[0, 1])
    ax_lv0_u = fig.add_subplot(gs[0, 2])
    ax_lv0_g = fig.add_subplot(gs[1, 0])
    ax_lv1_u = fig.add_subplot(gs[1, 1])
    ax_lv1_g = fig.add_subplot(gs[1, 2])
    
    # gate.v == -1 for lv==0 -> gate.u vs gate.g
    if vneg_lv0:
        xu0, yg0 = zip(*vneg_lv0)
        ax_top_lv0.scatter(xu0, yg0, s=1, alpha=0.6)
    ax_top_lv0.set_title('lv=0, gate.v == -1 : gate.u vs gate.g')
    ax_top_lv0.set_xlabel('gate.u')
    ax_top_lv0.set_ylabel('gate.g')
    ax_top_lv0.grid(True, linestyle='--', alpha=0.5)

    # gate.v == -1 for lv==1 -> gate.u vs gate.g
    if vneg_lv1:
        xu1, yg1 = zip(*vneg_lv1)
        ax_top_lv1.scatter(xu1, yg1, s=1, alpha=0.6)
    ax_top_lv1.set_title('lv=1, gate.v == -1 : gate.u vs gate.g')
    ax_top_lv1.set_xlabel('gate.u')
    ax_top_lv1.set_ylabel('gate.g')
    ax_top_lv1.grid(True, linestyle='--', alpha=0.5)

    # lv==0 plots
    if lv0_vpos:
        x_v, x_u, x_g = zip(*lv0_vpos)
        ax_lv0_u.scatter(x_v, x_u, s=1, alpha=0.6)
        ax_lv0_g.scatter(x_v, x_g, s=1, alpha=0.6)
    ax_lv0_u.set_title('lv=0: gate.v vs gate.u')
    ax_lv0_g.set_title('lv=0: gate.v vs gate.g')
    ax_lv0_u.set_xlabel('gate.v')
    ax_lv0_u.set_ylabel('gate.u')
    ax_lv0_g.set_xlabel('gate.v')
    ax_lv0_g.set_ylabel('gate.g')
    ax_lv0_u.grid(True, linestyle='--', alpha=0.5)
    ax_lv0_g.grid(True, linestyle='--', alpha=0.5)

    # lv==1 plots
    if lv1_vpos:
        y_v, y_u, y_g = zip(*lv1_vpos)
        ax_lv1_u.scatter(y_v, y_u, s=1, alpha=0.6)
        ax_lv1_g.scatter(y_v, y_g, s=1, alpha=0.6)
    ax_lv1_u.set_title('lv=1: gate.v vs gate.u')
    ax_lv1_g.set_title('lv=1: gate.v vs gate.g')
    ax_lv1_u.set_xlabel('gate.v')
    ax_lv1_u.set_ylabel('gate.u')
    ax_lv1_g.set_xlabel('gate.v')
    ax_lv1_g.set_ylabel('gate.g')
    ax_lv1_u.grid(True, linestyle='--', alpha=0.5)
    ax_lv1_g.grid(True, linestyle='--', alpha=0.5)

    fig.suptitle('P2: gate relationships')
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    plt.show()


def plot_P2_zoom_in_v_vs_g_u(filepath, save_path: str = None, gatev_range=(0, int(0.25e6))):
    """
    Debug plot: load a P2 log, filter records where gate.v != -1 and
    gatev_range[0] <= gate.v < gatev_range[1], then make 2x2 subplots:
        [lv=0: gate.v vs gate.u] [lv=0: gate.v vs gate.g]
        [lv=1: gate.v vs gate.u] [lv=1: gate.v vs gate.g]
    `gatev_range` can be an int (interpreted as max with min=0) or a
    (min,max) tuple. Saves figure as provided save_path or '<file>_zoom_in_<min>_<max>.png'.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"File {filepath} does not exist.")
        return

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # find header
    header = None
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('sumcheck_id'):
            header = line
            start_idx = i + 1
            break
    if header is None:
        print("Header not found in file.")
        return

    header_cols = header.split()
    try:
        idx_lv = header_cols.index('lv/idx')
        idx_gateu = header_cols.index('gate.u')
        idx_gateg = header_cols.index('gate.g')
        idx_gatev = header_cols.index('gate.v')
    except ValueError as e:
        print("Expected header columns not found:", e)
        return

    # normalize gatev_range into min/max
    if isinstance(gatev_range, (list, tuple)) and len(gatev_range) == 2:
        gatev_min, gatev_max = int(gatev_range[0]), int(gatev_range[1])
    else:
        gatev_min, gatev_max = 0, int(gatev_range)

    # collect filtered records
    lv0 = []  # tuples (gatev, gateu, gateg)
    lv1 = []
    for line in lines[start_idx:]:
        if not line.strip() or line.strip().startswith('sumcheck_id'):
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
        if gatev == -1:
            continue
        if not (gatev_min <= gatev < gatev_max):
            continue
        if lv == 0:
            lv0.append((gatev, gateu, gateg))
        elif lv == 1:
            lv1.append((gatev, gateu, gateg))

    if not lv0 and not lv1:
        print("No records in the requested gate.v range.")
        return

    fig, axs = plt.subplots(2, 2, figsize=(14, 8))

    # lv=0: top row
    if lv0:
        v0, u0, g0 = zip(*lv0)
        axs[0, 0].scatter(v0, u0, s=1, alpha=0.6)
        axs[0, 1].scatter(v0, g0, s=1, alpha=0.6)
    axs[0, 0].set_title('lv=0: gate.v vs gate.u')
    axs[0, 1].set_title('lv=0: gate.v vs gate.g')
    axs[0, 0].set_xlim(gatev_min, gatev_max)
    axs[0, 1].set_xlim(gatev_min, gatev_max)
    axs[0, 0].set_xlabel('gate.v')
    axs[0, 1].set_xlabel('gate.v')
    axs[0, 0].set_ylabel('gate.u')
    axs[0, 1].set_ylabel('gate.g')

    # lv=1: bottom row
    if lv1:
        v1, u1, g1 = zip(*lv1)
        axs[1, 0].scatter(v1, u1, s=1, alpha=0.6)
        axs[1, 1].scatter(v1, g1, s=1, alpha=0.6)
    axs[1, 0].set_title('lv=1: gate.v vs gate.u')
    axs[1, 1].set_title('lv=1: gate.v vs gate.g')
    axs[1, 0].set_xlim(gatev_min, gatev_max)
    axs[1, 1].set_xlim(gatev_min, gatev_max)
    axs[1, 0].set_xlabel('gate.v')
    axs[1, 1].set_xlabel('gate.v')
    axs[1, 0].set_ylabel('gate.u')
    axs[1, 1].set_ylabel('gate.g')

    fig.suptitle(f'P2 zoom-in gate.v in [0, {gatev_max})')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        out_path = Path(save_path).with_suffix('')
        out_file = str(out_path) + f'_zoom_in_{gatev_min}_{gatev_max}.png'
    else:
        out_file = str(filepath.with_suffix('')) + f'_zoom_in_{gatev_min}_{gatev_max}.png'
    plt.savefig(out_file)
    print(f"Zoom-in plot saved to {out_file}")
    plt.show()


def plot_P1_zoom_in_u_vs_g(filepath, save_path: str = None, gateu_range=(0, int(0.25e6))):
    """
    Debug plot for P1: load a P1 log, filter records where gateu_range[0] <= gate.u < gateu_range[1],
    then make two stacked subplots for lu=0 and lu=1 showing gate.u vs gate.g.
    `gateu_range` can be an int (interpreted as max with min=0) or a (min,max) tuple.
    Saves figure as provided save_path or '<file>_zoom_in_<min>_<max>.png'.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"File {filepath} does not exist.")
        return

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # find header
    header = None
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('sumcheck_id'):
            header = line
            start_idx = i + 1
            break
    if header is None:
        print("Header not found in file.")
        return

    header_cols = header.split()
    try:
        idx_lu = header_cols.index('lu/idx')
        idx_gateu = header_cols.index('gate.u')
        idx_gateg = header_cols.index('gate.g')
    except ValueError as e:
        print("Expected header columns not found:", e)
        return

    lu0 = []  # list of (gateu, gateg)
    lu1 = []
    # normalize gateu_range into min/max
    if isinstance(gateu_range, (list, tuple)) and len(gateu_range) == 2:
        gateu_min, gateu_max = int(gateu_range[0]), int(gateu_range[1])
    else:
        gateu_min, gateu_max = 0, int(gateu_range)
    for line in lines[start_idx:]:
        if not line.strip() or line.strip().startswith('sumcheck_id'):
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
        if not (gateu_min <= gateu < gateu_max):
            continue
        if lu == 0:
            lu0.append((gateu, gateg))
        elif lu == 1:
            lu1.append((gateu, gateg))

    if not lu0 and not lu1:
        print("No records in the requested gate.u range.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    if lu0:
        xu0, yg0 = zip(*lu0)
        axes[0].scatter(xu0, yg0, s=1, alpha=0.6, color='C0')
    axes[0].set_title('lu=0: gate.u vs gate.g (zoom)')
    axes[0].set_xlim(gateu_min, gateu_max)
    axes[0].set_ylabel('gate.g')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    if lu1:
        xu1, yg1 = zip(*lu1)
        axes[1].scatter(xu1, yg1, s=1, alpha=0.6, color='C1')
    axes[1].set_title('lu=1: gate.u vs gate.g (zoom)')
    axes[1].set_xlim(gateu_min, gateu_max)
    axes[1].set_xlabel('gate.u')
    axes[1].set_ylabel('gate.g')
    axes[1].grid(True, linestyle='--', alpha=0.5)

    fig.suptitle(f'P1 zoom-in gate.u in [0, {gateu_max})')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        out_path = Path(save_path).with_suffix('')
        out_file = str(out_path) + f'_zoom_in_{gateu_min}_{gateu_max}.png'
    else:
        out_file = str(filepath.with_suffix('')) + f'_zoom_in_{gateu_min}_{gateu_max}.png'
    plt.savefig(out_file)
    print(f"P1 zoom-in plot saved to {out_file}")
    plt.show()


def plot_lasso_mult_array(filepath, sumcheck_id, save_path: str = None):
    """
    Plot lasso mult array relations for a given sumcheck_id.

    The log format is produced by `dumpLassoMultPattern` and contains
    repeated blocks with headers:
      sumcheck_id    u    hu
    and
      sumcheck_id    v    hv

    This function collects all (u, hu) and (v, hv) rows matching
    the requested `sumcheck_id` and produces a two-subplot scatter
    plot: left = u vs hu, right = v vs hv.

    If `save_path` is not provided the output is saved to
    `../plots/{ModelName}/{SqueezeMerge}/{sumcheck_id}_lasso_arr.png`
    when `filepath` follows the pattern
    `../output/{ModelName}/{SqueezeMerge}/{ModelName}_lasso_mult_array.log`.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"File {filepath} does not exist.")
        return

    with open(filepath, 'r') as f:
        lines = f.readlines()

    u_points = []  # list of (u, hu)
    v_points = []  # list of (v, hv)

    section = None
    for line in lines:
        s = line.strip()
        if not s:
            section = None
            continue
        parts = s.split()
        # detect headers
        if parts[0] == 'sumcheck_id' and len(parts) >= 3:
            # header like: sumcheck_id u hu  OR sumcheck_id v hv
            if parts[1] == 'u' and parts[2] == 'hu':
                section = 'u'
                continue
            if parts[1] == 'v' and parts[2] == 'hv':
                section = 'v'
                continue
        # parse data lines if in a section
        if section in ('u', 'v'):
            # Expect at least three columns: sumcheck_id, wire, local_wire
            if len(parts) < 3:
                continue
            try:
                sid = int(parts[0])
                wire = int(parts[1])
                local = int(parts[2])
            except ValueError:
                continue
            if sid != int(sumcheck_id):
                continue
            if section == 'u':
                u_points.append((wire, local))
            else:
                v_points.append((wire, local))

    if not u_points and not v_points:
        print(f"No records found for sumcheck_id={sumcheck_id} in {filepath}")
        return

    # Prepare output path
    if save_path:
        out_file = Path(save_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        # try to infer ModelName and SqueezeMerge from path
        model_name = filepath.name.replace('_lasso_mult_array.log', '')
        squeeze_merge = filepath.parent.name
        plots_dir = Path('..') / 'plots' / model_name / squeeze_merge
        plots_dir.mkdir(parents=True, exist_ok=True)
        out_file = plots_dir / f"{int(sumcheck_id)}_lasso_arr.png"

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    if u_points:
        xu, hu = zip(*u_points)
        axes[0].scatter(xu, hu, s=1, alpha=0.6, color='C0')
    axes[0].set_title(f'sumcheck_id={sumcheck_id}: u vs hu')
    axes[0].set_xlabel('u')
    axes[0].set_ylabel('hu')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    if v_points:
        xv, hv = zip(*v_points)
        axes[1].scatter(xv, hv, s=1, alpha=0.6, color='C1')
    axes[1].set_title(f'sumcheck_id={sumcheck_id}: v vs hv')
    axes[1].set_xlabel('v')
    axes[1].set_ylabel('hv')
    axes[1].grid(True, linestyle='--', alpha=0.5)

    fig.suptitle(f'lasso mult array for sumcheck_id={sumcheck_id}')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(str(out_file))
    print(f"Lasso mult array plot saved to {out_file}")
    plt.show()


if __name__ == "__main__":

    # ########## debug plots for specific layers ##########

    plot_P2_zoom_in_v_vs_g_u('../output/gpt2-small/SqueezeMerge_1/gpt2-small_initP2_mult_array/gpt2-small_initP2_mult_array_layers_3.log',
                             save_path="./plots/gpt2-small/SqueezeMerge_1/gpt2-small_initP2_mult_array_layers_3.png",
                             gatev_range=(int(0), int(100))
                             )

    # plot_P1_zoom_in_u_vs_g('../output/gpt2-small/SqueezeMerge_1/gpt2-small_initP1_mult_array/gpt2-small_initP1_mult_array_layers_4.log',
    #                          save_path="./plots/gpt2-small/SqueezeMerge_1/gpt2-small_initP1_mult_array_layers_4.png",
    #                          gateu_range=(int(296900), int(297100))
    #                          )
                             
    # exit(0)
    # ########## debug plots for specific layers ##########
    
    SqueezeMerge = "SqueezeMerge_1"
    ModelName = "opt-125m"
    # # existing P1 processing
    # for layer in tqdm([100, 98, 8, 6, 4, 3, 2, 1], desc="Processing P1 layers"):
    #     layers_file_name = f"{ModelName}_initP1_mult_array_layers_" + str(layer)
    #     u_g_data_file = Path(
    #         f"../output/{ModelName}/{SqueezeMerge}/{ModelName}_initP1_mult_array") / f"{layers_file_name}.log"
    #     plot_gate_u_g_P1_relation(
    #         u_g_data_file, f"./plots/{ModelName}/{SqueezeMerge}/{layers_file_name}_P1_u_g.png")

    # # add P2 processing
    # for layer in tqdm([100, 98, 8, 6, 4, 3, 2, 1], desc="Processing P2 layers"):
    #     layers_file_name = f"{ModelName}_initP2_mult_array_layers_" + str(layer)
    #     u_g_data_file = Path(
    #         f"../output/{ModelName}/{SqueezeMerge}/{ModelName}_initP2_mult_array") / f"{layers_file_name}.log"
    #     plot_gate_u_g_P2_relation(
    #         u_g_data_file, f"./plots/{ModelName}/{SqueezeMerge}/{layers_file_name}_P2_v_g.png")

    SqueezeMerge = "SqueezeMerge_0"
    # # existing P1 processing
    # for layer in tqdm(range(1, 19 + 1), desc="Processing P1 layers"):
    #     layers_file_name = f"{ModelName}_initP1_mult_array_layers_" + str(layer)
    #     u_g_data_file = Path(
    #         f"../output/{ModelName}/{SqueezeMerge}/{ModelName}_initP1_mult_array") / f"{layers_file_name}.log"
    #     plot_gate_u_g_P1_relation(
    #         u_g_data_file, f"./plots/{ModelName}/{SqueezeMerge}/{layers_file_name}_P1_u_g.png")

    # # add P2 processing
    # for layer in tqdm(range(1, 19 + 1), desc="Processing P2 layers"):
    #     layers_file_name = f"{ModelName}_initP2_mult_array_layers_" + str(layer)
    #     u_g_data_file = Path(
    #         f"../output/{ModelName}/{SqueezeMerge}/{ModelName}_initP2_mult_array") / f"{layers_file_name}.log"
    #     plot_gate_u_g_P2_relation(
    #         u_g_data_file, f"./plots/{ModelName}/{SqueezeMerge}/{layers_file_name}_P2_v_g.png")
    
    # lasso_file_name = f"../output/{ModelName}/{SqueezeMerge}/{ModelName}_lasso_mult_array.log"
    # for sumcheck_id in tqdm([100, 98, 8, 6, 4, 3, 2, 1], desc="Processing lasso mult array sumcheck_ids"):
    #     plot_lasso_mult_array(lasso_file_name, sumcheck_id, save_path=f"./plots/{ModelName}/{SqueezeMerge}/lasso_arr/sc_{sumcheck_id}.png")
    
    print("gate_u_g_relation end...")
