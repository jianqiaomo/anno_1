import math
from tqdm import tqdm
from hardware_experiments.sumcheck_NTT_sweep import sweep_sumcheck_configs_wo_fz
import json
from zksp2.frac_mle import modelModInv


def append_task(queue, name, start_cycle, duration_cycle):
    end_cycle = start_cycle + duration_cycle
    task = {
        "name": name,
        "start_cycle": start_cycle,
        "duration_cycle": duration_cycle,
        "end_cycle": end_cycle,
    }
    queue.append(task)
    return task


def round_1_sumcheck_latency(
    m,
    n,
    len_F_G_chunk,
    DRAM_bandwidth_B_cycle,
    inv_latency_cycle,
    sumcheck_r1_latency_cycle,
):
    """
    Scheduling: Estimate round-1 logup sumcheck latency with a simple two-queue pipeline model.

    Args:
        m: Length of each f vector. E.g., 2**21.
        n: Length of t. Kept in the result for bookkeeping, but not used by the
            current round-1 schedule. E.g., 2**7.
        len_F_G_chunk: Chunk size for F/G processing. E.g., 1024. So needs to process ceil(m / len_F_G_chunk) chunks sequentially.
        DRAM_bandwidth_B_cycle: Effective DRAM bandwidth in bytes per cycle. Eg. 1024 for 1024 GB/s with 1GHz DRAM clock.
        inv_latency_cycle: Latency of the inversion-and-F/G generation stage.
        sumcheck_r1_latency_cycle: Latency of the round-1 sumcheck stage per term.

    Scheduling model:
        Each term is split into seven steps:
        `load_f -> f+r -> inv_F_2_G -> store_back_F -> store_back_G -> store_back_eq -> sumcheck_r1`.
        Memory-facing steps are serialized in `load_store_queue`.
        Compute-facing steps are serialized in `compute_queue`, with dependency
        constraints between neighboring stages and between the same stage across
        adjacent terms.

    Returns:
        A dict containing:
        - Input parameters.
        - Derived per-step cycle counts such as `load_f_cycle` and `store_F_cycle`.
        - Suggested throughput/system-level values:
          `Num_mod_add`, `Min_Tree_eq_throughput`, `Suggest_Memory_block_num`.
        - `load_store_queue` and `compute_queue`, where each task entry stores
          `name`, `start_cycle`, `duration_cycle`, and `end_cycle`.
        - `Total_latency_cycle`, defined as the latest end time across the two queues.
    """
    if len_F_G_chunk <= 0:
        raise ValueError("len_F_G_chunk must be positive")
    if DRAM_bandwidth_B_cycle <= 0:
        raise ValueError("DRAM_bandwidth_B_cycle must be positive")
    if m < 0 or n < 0:
        raise ValueError("m and n must be non-negative")

    num_term = int(math.ceil(m / len_F_G_chunk))
    load_f_cycle = len_F_G_chunk * 1 / DRAM_bandwidth_B_cycle
    store_F_cycle = len_F_G_chunk * 32 / DRAM_bandwidth_B_cycle
    store_G_cycle = store_F_cycle
    store_eq_cycle = store_F_cycle
    f_add_r_cycle = load_f_cycle

    load_store_queue = []
    compute_queue = []
    previous_inv_end = None
    previous_sumcheck_end = None

    # print(f"Scheduling round-1 sumcheck for {num_term} terms with chunk size {len_F_G_chunk}...")
    for idx in range(num_term):
        load_f_name = f"{idx}_load_f"
        f_add_r_name = f"{idx}_f+r"
        inv_name = f"{idx}_inv_F_2_G"
        store_F_name = f"{idx}_store_back_F"
        store_G_name = f"{idx}_store_back_G"
        store_eq_name = f"{idx}_store_back_eq"
        sumcheck_name = f"{idx}_sumcheck_r1"

        if not load_store_queue:
            load_f_start = 0
        else:
            assert load_store_queue[-1]["name"].endswith("store_back_eq")
            load_f_start = load_store_queue[-1]["end_cycle"]
        load_f_task = append_task(
            load_store_queue,
            load_f_name,
            load_f_start,
            load_f_cycle,
        )

        if compute_queue:
            assert load_store_queue[-1]["name"] == load_f_name
        f_add_r_task = append_task(
            compute_queue,
            f_add_r_name,
            load_f_task["start_cycle"],
            f_add_r_cycle,
        )

        inv_start = f_add_r_task["end_cycle"]
        if previous_inv_end is not None:
            inv_start = max(inv_start, previous_inv_end)
        inv_task = append_task(
            compute_queue,
            inv_name,
            inv_start,
            inv_latency_cycle,
        )
        previous_inv_end = inv_task["end_cycle"]

        store_F_task = append_task(
            load_store_queue,
            store_F_name,
            load_f_task["end_cycle"],
            store_F_cycle,
        )

        store_G_serial_end = store_F_task["end_cycle"] + store_G_cycle
        if store_G_serial_end > inv_task["end_cycle"]:
            store_G_start = store_F_task["end_cycle"]
        else:
            store_G_start = inv_task["end_cycle"] - store_G_cycle
        store_G_task = append_task(
            load_store_queue,
            store_G_name,
            store_G_start,
            store_G_cycle,
        )

        append_task(
            load_store_queue,
            store_eq_name,
            store_G_task["end_cycle"],
            store_eq_cycle,
        )

        sumcheck_start = inv_task["end_cycle"]
        if previous_sumcheck_end is not None:
            sumcheck_start = max(sumcheck_start, previous_sumcheck_end)
        sumcheck_task = append_task(
            compute_queue,
            sumcheck_name,
            sumcheck_start,
            sumcheck_r1_latency_cycle,
        )
        previous_sumcheck_end = sumcheck_task["end_cycle"]

    total_latency_cycle = max(
        load_store_queue[-1]["end_cycle"] if load_store_queue else 0,
        compute_queue[-1]["end_cycle"] if compute_queue else 0,
    )

    return {
        "m": m,
        "n": n,
        "len_F_G_chunk": len_F_G_chunk,
        "DRAM_bandwidth_B_cycle": DRAM_bandwidth_B_cycle,
        "inv_latency_cycle": inv_latency_cycle,
        "sumcheck_r1_latency_cycle": sumcheck_r1_latency_cycle,
        "Num_term": num_term,
        "load_f_cycle": load_f_cycle,
        "store_F_cycle": store_F_cycle,
        "store_G_cycle": store_G_cycle,
        "store_eq_cycle": store_eq_cycle,
        "f_add_r_cycle": f_add_r_cycle,
        "Num_mod_add": DRAM_bandwidth_B_cycle,
        "Min_Tree_eq_throughput": math.ceil(
            len_F_G_chunk / (load_f_cycle + inv_latency_cycle)
        ),
        "Suggest_Memory_block_num": DRAM_bandwidth_B_cycle,
        "load_store_queue": load_store_queue.copy(),
        "compute_queue": compute_queue.copy(),
        "Total_latency_cycle": total_latency_cycle,
    }


def modelInv(
    N=pow(2, 20),
    num_units=1,
    CLK_FREQ=1e9,
    bitwidth=255,
    verbose=False
):
    """
    From modelFracMLE. Estimate latency, hardware resources, and bandwidth for batch inversion of N elements.

    Args:
        N: Number of elements to invert. E.g., 2**20.
        num_units: Number of parallel fracMLE units. Each unit processes a disjoint chunk of the N elements. E.g., 4.
        CLK_FREQ: Clock frequency in Hz. E.g., 1e9 for 1GHz.
        bitwidth: Bitwidth of each element. E.g., 255 for 255-bit field elements.
        verbose: If True, print the output dict in a pretty JSON format.
    """
    num_elements_per_unit = math.ceil(N / num_units)
    modInv_model = modelModInv(
        num_elements_per_unit,
        bitwidth=bitwidth,
        verbose=verbose,
    )
    # each fracMLE unit has multiple mod inv units
    num_modinv = modInv_model["num_units"] * num_units

    e2e_lat = modInv_model["e2e_lat"]
    last_out_lat = modInv_model["last_out_lat"]

    # hardware resources
    num_muls = (modInv_model["num_muls"]) * num_units

    # number of registers needed
    num_regs = (modInv_model["num_regs"]) * num_units

    # size of SRAM needed
    sram_size_KiB = (
        modInv_model["sram_size_KiB"]
    ) * num_units

    # off-chip memory bandwidth
    # there is no input bandwidth for fracMLE since
    # we can read N and D directly as they get produced
    # so the only bandwidth needed is to write one element of fracMLE per cycle
    # bandwidth_GiB_per_s = bitwidth / 8 * CLK_FREQ / pow(2, 30) * num_units

    out = {
        "u": N.bit_length()-1,
        "N": N,
        "modInv_num_units": num_modinv,
        "e2e_lat": e2e_lat,
        "last_out_lat": last_out_lat,
        # multipliers
        "num_muls": num_muls,
        # registers
        "num_regs": num_regs,
        # SRAM
        "sram_size_KiB": sram_size_KiB,
        # bandwidth
        # "bandwidth_GiB_per_s": bandwidth_GiB_per_s,
    }

    if verbose:
        print(json.dumps(out, indent=2))
    return out


def model_logup_proof_core_one_f_column_1_round(
    m,
    n,
    len_F_G_chunk,
    DRAM_bandwidth_B_cycle,
    num_inv_unit,
    num_sumcheck_pe,
    num_eval_engines,
    num_product_lanes,
):
    """
    Model the logup-proof core latency for one f column. Considering multiple chunks in sequence.

    This helper combines:
    - round-1 sumcheck latency from `sweep_sumcheck_configs_wo_fz`
    - modular inversion latency from `modelInv`
    - the chunked round-1 queue model in `round_1_sumcheck_latency`

    Returns:
        A dict containing the intermediate sweep/model outputs and the final
        round-1 latency breakdown.
    """
    num_onchip_mle_sizes = len_F_G_chunk
    sumcheck_sweep_df = sweep_sumcheck_configs_wo_fz(
        num_var_list=[len_F_G_chunk.bit_length() - 1],
        available_bw_list=[1e7],  # effectively ignore bandwidth constraint by setting it very high. Load store time in round_1_sumcheck_latency.
        polynomial_list=[[["g1", "g2", "fz"]]],
        sweep_sumcheck_pes_range=[num_sumcheck_pe],
        sweep_eval_engines_range=[num_eval_engines],
        sweep_product_lanes_range=[num_product_lanes],
        sweep_onchip_mle_sizes_range=[num_onchip_mle_sizes],
        no_rd1_prefetch=True,
    )
    if sumcheck_sweep_df.empty:
        raise ValueError("sweep_sumcheck_configs_wo_fz returned an empty DataFrame")

    sumcheck_r1_latency_cycle_per_FG_chunk = sumcheck_sweep_df.iloc[0]["round_latencies"][0]

    modinv_module = modelInv(
        N=len_F_G_chunk,
        num_units=num_inv_unit,
        CLK_FREQ=1e9,
        bitwidth=255,
        verbose=False,
    )
    inv_latency_cycle = modinv_module["last_out_lat"]

    # considering multiple chunks in sequence
    round_1_model = round_1_sumcheck_latency(
        m=m,
        n=n,
        len_F_G_chunk=len_F_G_chunk,
        DRAM_bandwidth_B_cycle=DRAM_bandwidth_B_cycle,
        inv_latency_cycle=inv_latency_cycle,
        sumcheck_r1_latency_cycle=sumcheck_r1_latency_cycle_per_FG_chunk,
    )

    return {
        "inputs": {
            "m": m,
            "n": n,
            "len_F_G_chunk": len_F_G_chunk,
            "DRAM_bandwidth_B_cycle": DRAM_bandwidth_B_cycle,
            "num_inv_unit": num_inv_unit,
            "num_sumcheck_pe": num_sumcheck_pe,
            "num_eval_engines": num_eval_engines,
            "num_product_lanes": num_product_lanes,
            "num_onchip_mle_sizes": num_onchip_mle_sizes,
        },
        "sumcheck_r1_latency_cycle_per_FG_chunk": sumcheck_r1_latency_cycle_per_FG_chunk,
        "modinv_module": modinv_module,
        "inv_latency_cycle": inv_latency_cycle,
        "round_1_model": round_1_model,
        "total_r1_latency_cycle": round_1_model["Total_latency_cycle"],
    }


def model_logup_proof_core_one_f_column_rest_rounds(
    m,
    n,
    len_F_G_chunk,
    DRAM_bandwidth_B_cycle,
    num_inv_unit,
    num_sumcheck_pe,
    num_eval_engines,
    num_product_lanes,
):
    """
    Model the remaining sumcheck rounds for one f column.

    This uses `sweep_sumcheck_configs_wo_fz(...)` and sums all round latencies
    except round 1, i.e. `sum(round_latencies[1:])`.

    Returns:
        A dict containing the remaining-round total latency and the other
        sumcheck-related cost fields from the selected sweep point.
    """
    # sumcheck_sweep_df = sweep_sumcheck_configs_wo_fz(
    #     num_var_list=[len_F_G_chunk.bit_length() - 1],
    #     available_bw_list=[DRAM_bandwidth_B_cycle],
    #     polynomial_list=[[["g1", "g2", "fz"]]],
    #     sweep_sumcheck_pes_range=[num_sumcheck_pe],
    #     sweep_eval_engines_range=[num_eval_engines],
    #     sweep_product_lanes_range=[num_product_lanes],
    #     no_rd1_prefetch=True,
    # )
    sumcheck_sweep_df = sweep_sumcheck_configs_wo_fz(
        num_var_list=[m.bit_length() - 1],
        available_bw_list=[DRAM_bandwidth_B_cycle],
        polynomial_list=[[["g1", "g2", "fz"]]],
        sweep_sumcheck_pes_range=[num_sumcheck_pe],
        sweep_eval_engines_range=[num_eval_engines],
        sweep_product_lanes_range=[num_product_lanes],
        sweep_onchip_mle_sizes_range=[len_F_G_chunk],
        no_rd1_prefetch=False,
    )
    if sumcheck_sweep_df.empty:
        raise ValueError("sweep_sumcheck_configs_wo_fz returned an empty DataFrame")

    sumcheck_row = sumcheck_sweep_df.iloc[0].to_dict()
    round_latencies = sumcheck_row["round_latencies"]
    if len(round_latencies) < 2:
        rest_rounds_total_latency_cycle = 0
    else:
        rest_rounds_total_latency_cycle = sum(round_latencies[1:])

    return {
        "inputs": {
            "m": m,
            "n": n,
            "len_F_G_chunk": len_F_G_chunk,
            "DRAM_bandwidth_B_cycle": DRAM_bandwidth_B_cycle,
            "num_inv_unit": num_inv_unit,
            "num_sumcheck_pe": num_sumcheck_pe,
            "num_eval_engines": num_eval_engines,
            "num_product_lanes": num_product_lanes,
        },
        "rest_rounds_total_latency_cycle": rest_rounds_total_latency_cycle,
        "sumcheck_cost": {
            "round_latencies": round_latencies,
            "total_latency": sumcheck_row.get("total_latency"),
            "area": sumcheck_row.get("area"),
            "area_with_hbm": sumcheck_row.get("area_with_hbm"),
            "modmul_count": sumcheck_row.get("modmul_count"),
            "design_modmul_area": sumcheck_row.get("design_modmul_area"),
            "total_onchip_memory_MB": sumcheck_row.get("total_onchip_memory_MB"),
            "utilization": sumcheck_row.get("utilization"),
            "per_round_utilization": sumcheck_row.get("per_round_utilization"),
            "hardware_config": sumcheck_row.get("hardware_config"),
        },
    }


def model_logup_proof_core_one_f_column_all_rounds(
    m,
    n,
    DRAM_bandwidth_B_cycle,
    num_inv_unit,
    num_sumcheck_pe,
    num_eval_engines,
    num_product_lanes,
    num_onchip_mle_sizes,
):
    """
    Wrap up the all-round cost of the logup-proof core for one f column.

    Args:
        m: Length of each f vector. E.g., 2**21.
        n: Length of t. Kept in the result for bookkeeping, but not used by the current schedule. E.g., 2**7.
        DRAM_bandwidth_B_cycle: Effective DRAM bandwidth in bytes per cycle. Eg. 1024 for 1024 GB/s with 1GHz DRAM clock.
        num_inv_unit: Number of parallel inversion units for the inversion-and-F/G generation stage. E.g., 4.
        num_sumcheck_pe: Number of PEs for the sumcheck stages. E.g., 4.
        num_eval_engines: Number of evaluation engines. E.g., 4.
        num_product_lanes: Number of product lanes. E.g., 4.
        num_onchip_mle_sizes: (len_F_G_chunk) Chunk size for F/G processing. E.g., 1024. So needs to process ceil(m / len_F_G_chunk) chunks sequentially.


    Returns:
        A dict with combined latency and resource summaries from:
        - round 1: inversion-backed chunk pipeline model
        - remaining rounds: sumcheck sweep model
    """
    len_F_G_chunk = num_onchip_mle_sizes
    round_1_result = model_logup_proof_core_one_f_column_1_round(
        m=m,
        n=n,
        len_F_G_chunk=len_F_G_chunk,
        DRAM_bandwidth_B_cycle=DRAM_bandwidth_B_cycle,
        num_inv_unit=num_inv_unit,
        num_sumcheck_pe=num_sumcheck_pe,
        num_eval_engines=num_eval_engines,
        num_product_lanes=num_product_lanes,
    )
    rest_rounds_result = model_logup_proof_core_one_f_column_rest_rounds(
        m=m,
        n=n,
        len_F_G_chunk=len_F_G_chunk,
        DRAM_bandwidth_B_cycle=DRAM_bandwidth_B_cycle,
        num_inv_unit=num_inv_unit,
        num_sumcheck_pe=num_sumcheck_pe,
        num_eval_engines=num_eval_engines,
        num_product_lanes=num_product_lanes,
    )

    modinv_module = round_1_result["modinv_module"]
    sumcheck_cost = rest_rounds_result["sumcheck_cost"]

    total_latency_cycle = (
        round_1_result["total_r1_latency_cycle"]
        + rest_rounds_result["rest_rounds_total_latency_cycle"]
    )

    modinv_num_units = modinv_module["modInv_num_units"]
    modmul_count_modinv = modinv_module["num_muls"]
    modmul_count_sumcheck_rest = sumcheck_cost.get("modmul_count", 0)
    total_modmul_count = modmul_count_modinv + modmul_count_sumcheck_rest

    num_regs_modinv = modinv_module["num_regs"]
    num_regs_sumcheck_rest = 0
    total_num_regs = num_regs_modinv + num_regs_sumcheck_rest

    sram_cost_modinv_KiB = modinv_module["sram_size_KiB"]
    sram_cost_modinv_MB = sram_cost_modinv_KiB / 1024
    sram_cost_sumcheck_rest_MB = sumcheck_cost.get("total_onchip_memory_MB", 0)
    total_sram_cost_MB = sram_cost_modinv_MB + sram_cost_sumcheck_rest_MB

    return {
        "inputs": {
            "m": m,
            "n": n,
            "len_F_G_chunk": len_F_G_chunk,
            "DRAM_bandwidth_B_cycle": DRAM_bandwidth_B_cycle,
            "num_inv_unit": num_inv_unit,
            "num_sumcheck_pe": num_sumcheck_pe,
            "num_eval_engines": num_eval_engines,
            "num_product_lanes": num_product_lanes,
            "num_onchip_mle_sizes": len_F_G_chunk,
        },
        "One_f_column_total_latency": total_latency_cycle,
        "Total_modInv_num_units": modinv_num_units,
        "Modmul_count_modinv": modmul_count_modinv,
        "Modmul_count_sumcheck_pes": modmul_count_sumcheck_rest,
        "Total_modmul_count": total_modmul_count,
        "Num_regs_modinv": num_regs_modinv,
        # "Num_regs_sumcheck_pes": num_regs_sumcheck_rest,
        "Total_num_regs": total_num_regs,
        "Sram_cost_modinv_KiB": sram_cost_modinv_KiB,
        "Sram_cost_modinv_MB": sram_cost_modinv_MB,
        "Sram_cost_sumcheck_pes_db_buf_MB": sram_cost_sumcheck_rest_MB,
        "Total_sram_cost_MB": total_sram_cost_MB,
        "round_1_result": round_1_result,
        "rest_rounds_result": rest_rounds_result,
    }


def model_logup_proof_core_one_c_column_all_rounds(
    m,
    n,
    DRAM_bandwidth_B_cycle,
    num_inv_unit,
    num_sumcheck_pe,
    num_eval_engines,
    num_product_lanes,
    num_onchip_mle_sizes,
):
    """
    Wrap up the all-round cost of the logup-proof core for one c and t column.

    Args:
        m: Length of each f vector. E.g., 2**21.
        n: Length of t and c. E.g., 2**7.
        DRAM_bandwidth_B_cycle: Effective DRAM bandwidth in bytes per cycle. Eg. 1024 for 1024 GB/s with 1GHz DRAM clock.
        num_inv_unit: Number of parallel inversion units for the inversion stage. E.g., 4.
        num_sumcheck_pe: Number of PEs for the sumcheck stages. E.g., 4.
        num_eval_engines: Number of evaluation engines. E.g., 4.
        num_product_lanes: Number of product lanes. E.g., 4.
        num_onchip_mle_sizes: Number of on-chip MLE sizes (#table entries) available for sumcheck. E.g., 1024. Final SRAM result is doubled buffer.

    This model:
    - runs sumcheck at `num_var_list=[n]`
    - runs modular inversion with `N=2**n`
    - sums sumcheck latency and modinv latency

    Raises:
        ValueError: if `2**n > num_onchip_mle_sizes`
    """
    mle_length = n
    if mle_length > num_onchip_mle_sizes:
        raise ValueError(
            f"for small c, t logup: 2**n={mle_length} exceeds num_onchip_mle_sizes={num_onchip_mle_sizes}"
        )

    sumcheck_sweep_df = sweep_sumcheck_configs_wo_fz(
        num_var_list=[n.bit_length() - 1],
        available_bw_list=[DRAM_bandwidth_B_cycle],
        polynomial_list=[[["g1", "g2", "fz"]]],
        sweep_sumcheck_pes_range=[num_sumcheck_pe],
        sweep_eval_engines_range=[num_eval_engines],
        sweep_product_lanes_range=[num_product_lanes],
        sweep_onchip_mle_sizes_range=[num_onchip_mle_sizes],
        no_rd1_prefetch=True,
    )
    if sumcheck_sweep_df.empty:
        raise ValueError("sweep_sumcheck_configs_wo_fz returned an empty DataFrame")

    sumcheck_cost = sumcheck_sweep_df.iloc[0].to_dict()
    sumcheck_latency_cycle = sumcheck_cost["total_latency"]

    modinv_module = modelInv(
        N=mle_length,
        num_units=num_inv_unit,
        CLK_FREQ=1e9,
        bitwidth=255,
        verbose=False,
    )
    modinv_latency_cycle = modinv_module["last_out_lat"]

    one_c_column_total_latency = sumcheck_latency_cycle + modinv_latency_cycle

    return {
        "inputs": {
            "m": m,
            "n": n,
            "DRAM_bandwidth_B_cycle": DRAM_bandwidth_B_cycle,
            "num_inv_unit": num_inv_unit,
            "num_sumcheck_pe": num_sumcheck_pe,
            "num_eval_engines": num_eval_engines,
            "num_product_lanes": num_product_lanes,
            "num_onchip_mle_sizes": num_onchip_mle_sizes,
        },
        "One_c_column_total_latency": one_c_column_total_latency,
        "sumcheck_latency_cycle": sumcheck_latency_cycle,
        "modinv_latency_cycle": modinv_latency_cycle,
        "Total_modInv_num_units": modinv_module["modInv_num_units"],
        "Modmul_count_modinv": modinv_module["num_muls"],
        "Modmul_count_sumcheck": sumcheck_cost.get("modmul_count", 0),
        "Total_modmul_count": modinv_module["num_muls"] + sumcheck_cost.get("modmul_count", 0),
        "Num_regs_modinv": modinv_module["num_regs"],
        "Num_regs_sumcheck": 0,
        "Total_num_regs": modinv_module["num_regs"],
        "Sram_cost_modinv_KiB": modinv_module["sram_size_KiB"],
        "Sram_cost_modinv_MB": modinv_module["sram_size_KiB"] / 1024,
        "Sram_cost_sumcheck_MB": sumcheck_cost.get("total_onchip_memory_MB", 0),
        "Total_sram_cost_MB": (modinv_module["sram_size_KiB"] / 1024)
        + sumcheck_cost.get("total_onchip_memory_MB", 0),
        "sumcheck_cost": {
            "round_latencies": sumcheck_cost.get("round_latencies"),
            "total_latency": sumcheck_cost.get("total_latency"),
            "area": sumcheck_cost.get("area"),
            "area_with_hbm": sumcheck_cost.get("area_with_hbm"),
            "modmul_count": sumcheck_cost.get("modmul_count"),
            "design_modmul_area": sumcheck_cost.get("design_modmul_area"),
            "total_onchip_memory_MB": sumcheck_cost.get("total_onchip_memory_MB"),
            "utilization": sumcheck_cost.get("utilization"),
            "per_round_utilization": sumcheck_cost.get("per_round_utilization"),
            "hardware_config": sumcheck_cost.get("hardware_config"),
        },
        "modinv_module": modinv_module,
    }


def model_logup_proof_core_all_columns(
    l,
    m,
    n,
    DRAM_bandwidth_B_cycle,
    num_inv_unit,
    num_sumcheck_pe,
    num_eval_engines,
    num_product_lanes,
    num_onchip_mle_sizes,
):
    """
    Wrap up total logup-proof core cost across all repeated f and c columns.

    Args:
        l: Number of repetitions of f and c columns. E.g., 7.
        m: Length of each f vector. E.g., 2**21.
        n: Length of t and c. E.g., 2**7.
        len_F_G_chunk: Chunk size for F/G processing in the f column. E.g., 1024. So needs to process ceil(m / len_F_G_chunk) chunks sequentiallyly.
        DRAM_bandwidth_B_cycle: Effective DRAM bandwidth in bytes per cycle. Eg. 1024 for 1024 GB/s with 1GHz DRAM clock.
        num_inv_unit: Number of parallel inversion units for the inversion stage. E.g., 4.
        num_sumcheck_pe: Number of PEs for the sumcheck stages. E.g., 4.
        num_eval_engines: Number of evaluation engines. E.g., 4.
        num_product_lanes: Number of product lanes. E.g., 4.
        num_onchip_mle_sizes: (len_F_G_chunk) Number of on-chip MLE sizes (#table entries) available for sumcheck. E.g., 1024. Final SRAM result is doubled buffer. 
            should equal len_F_G_chunk in f column so needs to process ceil(m / len_F_G_chunk) chunks sequentiallyly.

    Latency model:
        Repeat one-f-column latency `l` times and one-c-column latency `l` times.

    Cost model:
        For non-latency costs, take the elementwise max between the one-f-column
        and one-c-column results. These costs are not multiplied by `l`.
    """
    if l < 0:
        raise ValueError("l must be non-negative")

    f_column_result = model_logup_proof_core_one_f_column_all_rounds(
        m=m,
        n=n,
        DRAM_bandwidth_B_cycle=DRAM_bandwidth_B_cycle,
        num_inv_unit=num_inv_unit,
        num_sumcheck_pe=num_sumcheck_pe,
        num_eval_engines=num_eval_engines,
        num_product_lanes=num_product_lanes,
        num_onchip_mle_sizes=num_onchip_mle_sizes,
    )
    c_column_result = model_logup_proof_core_one_c_column_all_rounds(
        m=m,
        n=n,
        DRAM_bandwidth_B_cycle=DRAM_bandwidth_B_cycle,
        num_inv_unit=num_inv_unit,
        num_sumcheck_pe=num_sumcheck_pe,
        num_eval_engines=num_eval_engines,
        num_product_lanes=num_product_lanes,
        num_onchip_mle_sizes=num_onchip_mle_sizes,
    )

    total_latency = l * (
        f_column_result["One_f_column_total_latency"] + c_column_result["One_c_column_total_latency"]
    )

    return {
        "inputs": {
            "l": l,
            "m": m,
            "n": n,
            "DRAM_bandwidth_B_cycle": DRAM_bandwidth_B_cycle,
            "num_inv_unit": num_inv_unit,
            "num_sumcheck_pe": num_sumcheck_pe,
            "num_eval_engines": num_eval_engines,
            "num_product_lanes": num_product_lanes,
            "num_onchip_mle_sizes": num_onchip_mle_sizes,
        },
        "Total_latency_cycle": total_latency,
        "Total_latency_all_l": total_latency,
        "Total_modInv_num_units": max(
            f_column_result["Total_modInv_num_units"],
            c_column_result["Total_modInv_num_units"],
        ),
        "Modmul_count_modinv": max(
            f_column_result["Modmul_count_modinv"],
            c_column_result["Modmul_count_modinv"],
        ),
        "Modmul_count_sumcheck_pes": max(
            f_column_result["Modmul_count_sumcheck_pes"],
            c_column_result["Modmul_count_sumcheck"],
        ),
        "Total_modmul_count": max(
            f_column_result["Total_modmul_count"],
            c_column_result["Total_modmul_count"],
        ),
        "Total_num_regs": max(
            f_column_result["Total_num_regs"],
            c_column_result["Total_num_regs"],
        ),
        "Total_sram_cost_MB": max(
            f_column_result["Total_sram_cost_MB"],
            c_column_result["Total_sram_cost_MB"],
        ),
        "f_column_result": f_column_result,
        "c_column_result": c_column_result,
    }


if __name__ == "__main__":
    round_1_example = round_1_sumcheck_latency(
        m=2**12,
        n=2**7,
        len_F_G_chunk=1024,
        DRAM_bandwidth_B_cycle=512,
        inv_latency_cycle=500,
        sumcheck_r1_latency_cycle=1024,
    )

    all_rounds_example_one_f_col = model_logup_proof_core_one_f_column_all_rounds(
        m=2**21,
        n=2**7,
        DRAM_bandwidth_B_cycle=1024,
        num_inv_unit=32,
        num_sumcheck_pe=32,
        num_eval_engines=6,
        num_product_lanes=6,
        num_onchip_mle_sizes=2**10,
    )

    all_rounds_example_one_c_col = model_logup_proof_core_one_c_column_all_rounds(
        m=2**21,
        n=2**7,
        DRAM_bandwidth_B_cycle=1024,
        num_inv_unit=32,
        num_sumcheck_pe=32,
        num_eval_engines=6,
        num_product_lanes=6,
        num_onchip_mle_sizes=2**10,
    )

    all_rounds_example_all_cols = model_logup_proof_core_all_columns(
        l=7,
        m=2**21,
        n=2**7,
        DRAM_bandwidth_B_cycle=1024,
        num_inv_unit=32,
        num_sumcheck_pe=32,
        num_eval_engines=6,
        num_product_lanes=6,
        num_onchip_mle_sizes=2**10,
    )


    print("logup_model.py end.")
