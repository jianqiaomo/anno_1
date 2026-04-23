import math
import json
from hardware_experiments.params import modmul_latency, extensions_latency
from hardware_experiments.sumcheck_NTT_sweep import sweep_sumcheck_configs_wo_fz
from zksp2.build_mle import BuildMleCostReport


def build_mle_cost(
    num_vars=12, 
    build_mle_throughput_per_cycle=32,
):
    """
    wrapper to BuildMleCostReport
    """
    build_mle_cost = BuildMleCostReport(
        num_vars=num_vars,
        mod_mul_latency=modmul_latency,
        mod_add_latency=1,
        num_zerocheck_pes=int(build_mle_throughput_per_cycle/2),
        num_mod_mul=-1,
        num_mod_add=-1,
        available_bandwidth=-1
    )
    result_dict = build_mle_cost.cost()
    return result_dict


def round_1_mm_sumcheck_latency(
    n,
    m,
    k,
    build_mle_throughput_per_cycle,
    DRAM_bandwidth_B_cycle,
    num_onchip_mle_sizes,
    num_dp_mul,
    num_elements_per_sram_feed_to_dp,
    sumcheck_pes,
    eval_engines,
    product_lanes,
):
    """
    Estimate round-1 Matmul sumcheck latency for matmul.

    Args:
        n: A's row length.
        m: A's column length.
        k: B's column length.
        build_mle_throughput_per_cycle: (assume always two parallel build units A, B) Build-MLE throughput in elements/cycle.
        DRAM_bandwidth_B_cycle: Effective DRAM bandwidth in bytes/cycle.
        num_onchip_mle_sizes: On-chip MLE chunk size per A~, B~.
        num_dp_mul: Number of multipliers for dot product. One unit to support A, B in series. 1 mul processes (1 eq * 1 A[]). Assume each mul has an accumulator.
        num_elements_per_sram_feed_to_dp: Number of elements per cycle SRAM can feed into the DP array. (#macro, bank per macro, port per bank, 1r/2r per port).
        sumcheck_pes: Number of sumcheck processing elements.
        eval_engines: Number of evaluation engines for sumcheck.
        product_lanes: Number of parallel product lanes for sumcheck.

    Returns:
        A dict with the derived cycle terms and final total latency.
    """
    if build_mle_throughput_per_cycle <= 0:
        raise ValueError("build_mle_throughput_per_cycle must be positive")
    if DRAM_bandwidth_B_cycle <= 0:
        raise ValueError("DRAM_bandwidth_B_cycle must be positive")
    if num_onchip_mle_sizes <= 0:
        raise ValueError("num_onchip_mle_sizes must be positive")
    if num_dp_mul <= 0:
        raise ValueError("num_dp_mul must be positive")
    if num_elements_per_sram_feed_to_dp <= 0:
        raise ValueError("num_elements_per_sram_feed_to_dp must be positive")
    if m < 0 or n < 0 or k < 0:
        raise ValueError("m, n, and k must be non-negative")

    sumcheck_sweep_df = sweep_sumcheck_configs_wo_fz(
        num_var_list=[m.bit_length() - 1],
        available_bw_list=[1e7],  # effectively ignore bandwidth constraint by setting it very high. Load store time in round_1_sumcheck_latency.
        polynomial_list=[[["g1", "g2"]]],  # A~, B~
        sweep_sumcheck_pes_range=[sumcheck_pes],
        sweep_eval_engines_range=[eval_engines],
        sweep_product_lanes_range=[product_lanes],
        sweep_onchip_mle_sizes_range=[num_onchip_mle_sizes],
        no_rd1_prefetch=True,
    )
    if sumcheck_sweep_df.empty:
        raise ValueError("sweep_sumcheck_configs_wo_fz returned an empty DataFrame")
    sumcheck_r1_latency_cycle = sumcheck_sweep_df.iloc[0]["round_latencies"][0]

    num_chunk = int(math.ceil(m / num_onchip_mle_sizes))
    load_A_B_cycle = 2 * (n + k) / DRAM_bandwidth_B_cycle
    store_A_B_cycle = 2 * 32 / DRAM_bandwidth_B_cycle
    build_mle_cycle = int(math.ceil(m / build_mle_throughput_per_cycle))
    build_mle_cost_result = build_mle_cost(num_vars=12, build_mle_throughput_per_cycle=build_mle_throughput_per_cycle)
    effective_num_dp_mul = min(num_elements_per_sram_feed_to_dp, num_dp_mul)
    dot_prod_cycle = int((n + k) / effective_num_dp_mul) + modmul_latency + math.log2(effective_num_dp_mul)

    latency = max(build_mle_cycle, load_A_B_cycle)
    dp_store_cycle_all_mle = max(dot_prod_cycle, store_A_B_cycle) * num_onchip_mle_sizes

    per_chunk_latency = []
    for idx in range(num_chunk):
        if sumcheck_r1_latency_cycle > dp_store_cycle_all_mle:
            chunk_latency = sumcheck_r1_latency_cycle
        else:
            chunk_latency = dp_store_cycle_all_mle + extensions_latency + modmul_latency
        latency += chunk_latency
        per_chunk_latency.append(
            {
                "chunk_idx": idx,
                "dp_store_cycle_all_mle": dp_store_cycle_all_mle,
                "sumcheck_r1_latency_cycle": sumcheck_r1_latency_cycle,
                "added_latency": chunk_latency,
            }
        )

    return {
        "m": m,
        "n": n,
        "k": k,
        "build_mle_throughput_per_cycle": build_mle_throughput_per_cycle,
        "DRAM_bandwidth_B_cycle": DRAM_bandwidth_B_cycle,
        "num_onchip_mle_sizes": num_onchip_mle_sizes,
        "num_dp_mul": num_dp_mul,
        "num_elements_per_sram_feed_to_dp": num_elements_per_sram_feed_to_dp,
        "effective_num_dp_mul": effective_num_dp_mul,
        "sumcheck_r1_latency_cycle": sumcheck_r1_latency_cycle,
        "Num_chunk": num_chunk,
        "load_A_B_cycle": load_A_B_cycle,
        "store_A_B_cycle": store_A_B_cycle,
        "build_mle_cycle": build_mle_cycle,
        "dot_prod_cycle": dot_prod_cycle,
        "dp_store_cycle_all_mle": dp_store_cycle_all_mle,
        "per_chunk_latency": per_chunk_latency,
        "Total_latency_cycle": latency,
        "build_mle_cost_result": build_mle_cost_result,
    }


def rest_mm_sumcheck_latency(
    n,
    m,
    k,
    DRAM_bandwidth_B_cycle,
    num_onchip_mle_sizes,
    num_elements_per_sram_feed_to_dp,
    sumcheck_pes,
    eval_engines,
    product_lanes,
):
    """
    Estimate the remaining Matmul sumcheck rounds by sweeping the sumcheck model and
    summing `round_latencies[1:]`.

    Args:
        n: A's row length.
        m: A's column length.
        k: B's column length.
        DRAM_bandwidth_B_cycle: Effective DRAM bandwidth in bytes/cycle.
        num_onchip_mle_sizes: On-chip MLE chunk size.
        num_elements_per_sram_feed_to_dp: Number of elements per cycle SRAM can feed into the DP array. (#macro, bank per macro, port per bank, 1r/2r per port).
        sumcheck_pes: Number of sumcheck processing elements.
        eval_engines: Number of evaluation engines for sumcheck.
        product_lanes: Number of parallel product lanes for sumcheck.

    Returns:
        A dict containing the remaining-round latency and selected sumcheck cost
        fields from the sweep result.
    """
    if DRAM_bandwidth_B_cycle <= 0:
        raise ValueError("DRAM_bandwidth_B_cycle must be positive")
    if num_onchip_mle_sizes <= 0:
        raise ValueError("num_onchip_mle_sizes must be positive")
    if num_elements_per_sram_feed_to_dp <= 0:
        raise ValueError("num_elements_per_sram_feed_to_dp must be positive")
    if n < 0 or m < 0 or k < 0:
        raise ValueError("n, m, and k must be non-negative")

    sumcheck_sweep_df = sweep_sumcheck_configs_wo_fz(
        num_var_list=[m.bit_length() - 1],
        available_bw_list=[DRAM_bandwidth_B_cycle],
        polynomial_list=[[["g1", "g2"]]],  # A~, B~
        sweep_sumcheck_pes_range=[sumcheck_pes],
        sweep_eval_engines_range=[eval_engines],
        sweep_product_lanes_range=[product_lanes],
        sweep_onchip_mle_sizes_range=[num_onchip_mle_sizes],
        no_rd1_prefetch=True,
    )
    if sumcheck_sweep_df.empty:
        raise ValueError("sweep_sumcheck_configs_wo_fz returned an empty DataFrame")
    sumcheck_sweep_df_no_dram_latency = sweep_sumcheck_configs_wo_fz(
        num_var_list=[m.bit_length() - 1],
        available_bw_list=[1e7],  # effectively ignore bandwidth constraint by setting it very high. Load store time in round_1_sumcheck_latency.
        polynomial_list=[[["g1", "g2"]]],  # A~, B~
        sweep_sumcheck_pes_range=[sumcheck_pes],
        sweep_eval_engines_range=[eval_engines],
        sweep_product_lanes_range=[product_lanes],
        sweep_onchip_mle_sizes_range=[num_onchip_mle_sizes],
        no_rd1_prefetch=True,
    )
    if sumcheck_sweep_df.empty:
        raise ValueError("sweep_sumcheck_configs_wo_fz returned an empty DataFrame")
    if sumcheck_sweep_df_no_dram_latency.empty:
        raise ValueError("sweep_sumcheck_configs_wo_fz returned an empty DataFrame")

    sumcheck_row = sumcheck_sweep_df.iloc[0].to_dict()
    sumcheck_row_no_dram_latency = sumcheck_sweep_df_no_dram_latency.iloc[0].to_dict()
    round_latencies = sumcheck_row["round_latencies"]
    round_latencies_no_dram_latency = sumcheck_row_no_dram_latency["round_latencies"]
    if num_onchip_mle_sizes >= m:
        # If the on-chip MLE can cover the entire dimension, does not need to load in round 2.
        rest_round_latency_cycle = (round_latencies_no_dram_latency[1] if len(round_latencies_no_dram_latency) > 1 else 0) + (sum(round_latencies[2:]) if len(round_latencies) >= 3 else 0)
    else:
        rest_round_latency_cycle = sum(round_latencies[1:]) if len(round_latencies) >= 2 else 0

    return {
        "inputs": {
            "n": n,
            "m": m,
            "k": k,
            "DRAM_bandwidth_B_cycle": DRAM_bandwidth_B_cycle,
            "num_onchip_mle_sizes": num_onchip_mle_sizes,
            "num_elements_per_sram_feed_to_dp": num_elements_per_sram_feed_to_dp,
            "sumcheck_pes": sumcheck_pes,
            "eval_engines": eval_engines,
            "product_lanes": product_lanes,
        },
        "rest_round_latency_cycle": rest_round_latency_cycle,
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


def mm_sumcheck_all_rounds_latency(
    n,
    m,
    k,
    build_mle_throughput_per_cycle,
    DRAM_bandwidth_B_cycle,
    num_onchip_mle_sizes,
    num_dp_mul,
    num_elements_per_sram_feed_to_dp,
    sumcheck_pes,
    eval_engines,
    product_lanes,
):
    """
    Matmul sumcheck (Thaler13) cost across all rounds.

    This combines:
    - round 1 latency and build-MLE cost from `round_1_mm_sumcheck_latency`
    - remaining-round latency and sumcheck cost from `rest_mm_sumcheck_latency`

    Args:
        n: A's row length.
        m: A's column length.
        k: B's column length.
        build_mle_throughput_per_cycle: (assume always two parallel build units A, B) Build-MLE throughput in elements/cycle.
        DRAM_bandwidth_B_cycle: Effective DRAM bandwidth in bytes/cycle.
        num_onchip_mle_sizes: On-chip MLE chunk size per A~, B~.
        num_dp_mul: Number of multipliers for dot product. One unit to support A, B in series. 1 mul processes (1 eq * 1 A[]). Assume each mul has an accumulator.
        num_elements_per_sram_feed_to_dp: Number of elements per cycle SRAM can feed into the DP array. (#macro, bank per macro, port per bank, 1r/2r per port).
        sumcheck_pes: Number of sumcheck processing elements.
        eval_engines: Number of evaluation engines for sumcheck.
        product_lanes: Number of parallel product lanes for sumcheck.

    Totals:
    - latency = round 1 + rest rounds
    - sumcheck cost comes from rest rounds
    - build-MLE cost comes from round 1
    - total modmuls/modadds/memory_MB are summed across build-MLE and sumcheck
    """
    round_1_result = round_1_mm_sumcheck_latency(
        n=n,
        m=m,
        k=k,
        build_mle_throughput_per_cycle=build_mle_throughput_per_cycle,
        DRAM_bandwidth_B_cycle=DRAM_bandwidth_B_cycle,
        num_onchip_mle_sizes=num_onchip_mle_sizes,
        num_dp_mul=num_dp_mul,
        num_elements_per_sram_feed_to_dp=num_elements_per_sram_feed_to_dp,
        sumcheck_pes=sumcheck_pes,
        eval_engines=eval_engines,
        product_lanes=product_lanes,
    )
    rest_round_result = rest_mm_sumcheck_latency(
        n=n,
        m=m,
        k=k,
        DRAM_bandwidth_B_cycle=DRAM_bandwidth_B_cycle,
        num_onchip_mle_sizes=num_onchip_mle_sizes,
        num_elements_per_sram_feed_to_dp=num_elements_per_sram_feed_to_dp,
        sumcheck_pes=sumcheck_pes,
        eval_engines=eval_engines,
        product_lanes=product_lanes,
    )

    build_mle_cost_result = round_1_result["build_mle_cost_result"]
    sumcheck_cost = rest_round_result["sumcheck_cost"]

    # Two build-MLEs per element (eqA, eqB)
    build_mle_modmuls = build_mle_cost_result.get("req_mod_mul_num", 0) * 2
    build_mle_modadds = build_mle_cost_result.get("req_mod_add_num", 0) * 2
    build_mle_memory_MB = build_mle_cost_result.get("required_sram_MB", 0) * 2

    sumcheck_modmuls = sumcheck_cost.get("modmul_count", 0)
    sumcheck_modadds = 0
    sumcheck_memory_MB = sumcheck_cost.get("total_onchip_memory_MB", 0)

    eq_1_2_memory_MB = 2 * (2 ** m) * 32 / (1024 * 1024)

    result_dict = {
        "inputs": {
            "n": n,
            "m": m,
            "k": k,
            "build_mle_throughput_per_cycle": build_mle_throughput_per_cycle,
            "DRAM_bandwidth_B_cycle": DRAM_bandwidth_B_cycle,
            "num_onchip_mle_sizes": num_onchip_mle_sizes,
            "num_dp_mul": num_dp_mul,
            "num_elements_per_sram_feed_to_dp": num_elements_per_sram_feed_to_dp,
            "sumcheck_pes": sumcheck_pes,
            "eval_engines": eval_engines,
            "product_lanes": product_lanes,
        },
        "Total_latency_cycle": (
            round_1_result["Total_latency_cycle"] + rest_round_result["rest_round_latency_cycle"]
        ),
        "build_mle_cost": build_mle_cost_result,
        "sumcheck_cost": sumcheck_cost,
        "Build_MLE_modmuls": build_mle_modmuls,
        "Build_MLE_modadds": build_mle_modadds,
        "Build_MLE_memory_MB": build_mle_memory_MB,
        "Eq_1_2_memory_MB": eq_1_2_memory_MB,
        "Dot_product_total_modmuls": num_dp_mul,
        "Dot_product_total_modadds": num_dp_mul.bit_length(),  # assuming a tree reduction for the dot product
        "Sumcheck_modmuls": sumcheck_modmuls,
        "Sumcheck_modadds": sumcheck_modadds,
        "Sumcheck_memory_MB": sumcheck_memory_MB,  # A~, B~, double buffer.
        "round_1_result": round_1_result,
        "rest_round_result": rest_round_result,
    }

    result_dict["Total_memory_MB"] = result_dict["Build_MLE_memory_MB"] + result_dict["Sumcheck_memory_MB"] + result_dict["Eq_1_2_memory_MB"]
    result_dict["Total_modadds"] = result_dict["Build_MLE_modadds"] + result_dict["Sumcheck_modadds"] + result_dict["Dot_product_total_modadds"]
    result_dict["Total_modmuls"] = max(result_dict["Build_MLE_modmuls"], result_dict["Sumcheck_modmuls"], result_dict["Dot_product_total_modmuls"])

    return result_dict


if __name__ == "__main__":
    example1 = round_1_mm_sumcheck_latency(
        n=2**5,
        m=2**12,
        k=2**12,
        build_mle_throughput_per_cycle=16,
        DRAM_bandwidth_B_cycle=1024,  # 64 bytes/cycle
        num_onchip_mle_sizes=2**12,
        num_dp_mul=256,
        num_elements_per_sram_feed_to_dp=128,
        sumcheck_pes=8,
        eval_engines=5,
        product_lanes=5,
    )

    example2 = rest_mm_sumcheck_latency(
        n=2**5,
        m=2**12,
        k=2**12,
        DRAM_bandwidth_B_cycle=1024,  # 64 bytes/cycle
        num_onchip_mle_sizes=2**12,
        num_elements_per_sram_feed_to_dp=128,
        sumcheck_pes=8,
        eval_engines=5,
        product_lanes=5,
    )

    # SRAM bank: single port, double rate.
    num_bank_per_macro = 64


    example3 = mm_sumcheck_all_rounds_latency(
        n=2**5,
        m=2**12,
        k=2**10,
        build_mle_throughput_per_cycle=32,
        DRAM_bandwidth_B_cycle=1024,  # 64 bytes/cycle
        num_onchip_mle_sizes=2**12,
        num_dp_mul=256,
        num_elements_per_sram_feed_to_dp=num_bank_per_macro * 2,
        sumcheck_pes=8,
        eval_engines=5,
        product_lanes=5,
    )

    print("matmul_model.py end.")
