import json
import math
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from hardware_experiments import params
from hardware_experiments import helper_funcs
from tqdm import tqdm


SUMCHECK_COST_KEYS = ("area", "modmul_count", "total_onchip_memory_MB")
INTEC_FECTOR = 5.85


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--multiprocess",
        action="store_true",
        default=False,
        help="Use multiprocessing for collecting merged latency/area results.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=70,
        help="Maximum number of workers.",
    )

    return parser.parse_args()


def _pick_one_matmul_shape_result(data):
    unique_shape_results = data["matmul"]["unique_shape_results"]
    if not unique_shape_results:
        raise ValueError("matmul.unique_shape_results is empty")
    first_key = sorted(unique_shape_results.keys())[0]
    return first_key, unique_shape_results[first_key]


def _max_sumcheck_cost(*cost_dicts):
    return {
        key: max(cost.get(key, 0) for cost in cost_dicts)
        for key in SUMCHECK_COST_KEYS
    }


def load_commit_open_result(
    model_name,
    squeeze_merge,
    num_MSM_PEs,
    dp_mul,
    bw,
    sim_data_root="./sim_data",
):
    commit_open_path = (
        Path(sim_data_root)
        / model_name
        / squeeze_merge
        / "commit_open"
        / f"MSM_pe_{num_MSM_PEs}_modmul_{dp_mul}_bw_{bw}.json"
    )
    with commit_open_path.open("r", encoding="utf-8") as f:
        return json.load(f), commit_open_path

def extract_actual_cost_dict(
    proof_data,
    num_MSM_PEs,
    dp_mul,
    sim_data_root="./sim_data",
):
    bw = proof_data["sweep_config"]["DRAM_bandwidth_B_cycle"]
    
    # MSM
    commit_open_data, commit_open_path = load_commit_open_result(
        model_name=proof_data["model_name"],
        squeeze_merge=proof_data["squeeze_merge"],
        num_MSM_PEs=num_MSM_PEs,
        dp_mul=dp_mul,
        bw=bw,
        sim_data_root=sim_data_root,
    )
    commit_open_breakdown = commit_open_data["commit_open_latency_breakdown"]
    commit_open_ns = commit_open_breakdown["total_ms"] * 1_000_000
    padd_area_mm2_7nm = commit_open_breakdown["padd_area_mm2"] / params.scale_factor_12_to_7nm

    hbm_phy_area_mm2 = helper_funcs.get_phy_cost(proof_data["sweep_config"]["DRAM_bandwidth_B_cycle"])

    matmul_shape_key, matmul_shape_result = _pick_one_matmul_shape_result(proof_data)
    matmul_model_result = matmul_shape_result["model_result"]
    build_mle_req_mem_bit = matmul_model_result["build_mle_cost"]["req_mem_bit"]

    # Inv
    total_inv_unit = proof_data["logup"]["Total_modInv_num_units"]
    modinv_area_mm2_7nm = total_inv_unit * params.modinv_area_mm2_7nm

    # SumCheck
    logup_sumcheck_cost = proof_data["logup"]["f_column_result"]["rest_rounds_result"]["sumcheck_cost"]
    matmul_sumcheck_cost = matmul_model_result["sumcheck_cost"]
    gkr_sumcheck_cost = proof_data["gkr_gates"]["layer_results"][0]["phase1_result"]["lv0_result"]["sumcheck_cost"]
    max_sumcheck_cost = _max_sumcheck_cost(
        logup_sumcheck_cost,
        matmul_sumcheck_cost,
        gkr_sumcheck_cost,
    )

    # SRAM
    msm_aggregate_mem_area_mm2_7nm = commit_open_breakdown["aggregate_mem_area_mm2"] / params.scale_factor_12_to_7nm
    capture_y_sram_size_exp = proof_data["sweep_config"]["capture_y_sram_size_exp"]
    mod_inv_sram_mb = proof_data["logup"]["f_column_result"]["Sram_cost_modinv_MB"]
    matmul_eq_sram_mb = 2 * (2 ** 14) * 32 / (1024 * 1024)
    matmul_buffer_sram_mb = 256 / 1024  # 256 kB
    gkr_capture_sram_mb = 2 * (2 ** capture_y_sram_size_exp) * 32 / 1024 / 1024
    gkr_buffer_sram_mb = (
        proof_data["gkr_gates"]["layer_results"][0]["phase1_result"]["inputs"]["num_sc_Vg_sizes"]
        * (4 + 32 + 32)
        / 1024
        / 1024
    )
    sumcheck_sram_mb = max_sumcheck_cost["total_onchip_memory_MB"]
    total_sram_mb = (
        max(mod_inv_sram_mb, matmul_eq_sram_mb + matmul_buffer_sram_mb)
        + gkr_capture_sram_mb
        + gkr_buffer_sram_mb
        + sumcheck_sram_mb
    )
    # remove SRAM from SumCheck
    sumcheck_sram_area_mm2_7nm = sumcheck_sram_mb * params.MB_CONVERSION_FACTOR_mm2_7nm
    max_sumcheck_cost['area'] -= sumcheck_sram_area_mm2_7nm
    total_sram_area_mm2_7nm = total_sram_mb * params.MB_CONVERSION_FACTOR_mm2_7nm + msm_aggregate_mem_area_mm2_7nm

    sumcheck_area_mm2_7nm = max_sumcheck_cost["area"]
    sumcheck_modmul_count = max_sumcheck_cost["modmul_count"]

    # DP Tree
    dp_modmul = max(
        proof_data["logup"]["Modmul_count_modinv"],
        matmul_model_result["Build_MLE_modmuls"],
        matmul_model_result["Dot_product_total_modmuls"],
        # max_sumcheck_cost["modmul_count"],
    )
    dp_modadd = max(
        matmul_model_result["Build_MLE_modadds"],
        matmul_model_result["Dot_product_total_modadds"],
    )
    dp_register_mb = build_mle_req_mem_bit / 1024 / 1024
    dp_register_count_255b = math.ceil(build_mle_req_mem_bit / params.bits_per_scalar)
    dp_modmul_area_mm2_7nm = dp_modmul * params.modmul_area_mm2_7nm
    dp_modadd_area_mm2_7nm = dp_modadd * params.modadd_area_mm2_7nm
    dp_register_area_mm2_7nm = dp_register_count_255b * params.reg_area_mm2_7nm
    dp_area_mm2_7nm = dp_modmul_area_mm2_7nm + dp_modadd_area_mm2_7nm + dp_register_area_mm2_7nm

    interconn_area_mm2_7nm = INTEC_FECTOR

    total_area_mm2_7nm = sum([
        padd_area_mm2_7nm,
        modinv_area_mm2_7nm,
        sumcheck_area_mm2_7nm,
        total_sram_area_mm2_7nm,
        dp_area_mm2_7nm,
        interconn_area_mm2_7nm,
    ])
    total_area_mm2_7nm_with_HBM = total_area_mm2_7nm + hbm_phy_area_mm2

    result_return = {
        "model_name": proof_data["model_name"],
        "squeeze_merge": proof_data["squeeze_merge"],
        "sweep_config": proof_data["sweep_config"],
        "latency_breakdown_ns": proof_data["latency_breakdown"],
        "actual_cost": {
            "commit_open_path": str(commit_open_path),
            "matmul_shape_key_used": matmul_shape_key,
            "padd_area_mm2_7nm": padd_area_mm2_7nm,
            "inv_unit_count": total_inv_unit,
            "inv_unit_area_mm2_7nm": modinv_area_mm2_7nm,
            "sumcheck_max_cost": max_sumcheck_cost,
            "sumcheck_area_mm2_7nm": sumcheck_area_mm2_7nm,
            "sumcheck_modmul_count": sumcheck_modmul_count,
            "sram_msm_aggregate_mem_area_mm2_7nm": msm_aggregate_mem_area_mm2_7nm,
            "sram_matmul_mod_inv_MB": max(mod_inv_sram_mb, matmul_eq_sram_mb + matmul_buffer_sram_mb),
            "sram_GKR_capture_MB": gkr_capture_sram_mb,
            "sram_GKR_buffer_MB": gkr_buffer_sram_mb,
            "sram_Sumcheck_MB": sumcheck_sram_mb,
            "sram_Total_MB": total_sram_mb,
            "sram_Total_area_mm2_7nm": total_sram_area_mm2_7nm,
            "DP_modmul": dp_modmul,
            "DP_modadd": dp_modadd,
            "DP_register_MB": dp_register_mb,
            "DP_area_mm2_7nm": dp_area_mm2_7nm,
            "interconn_area_mm2_7nm": interconn_area_mm2_7nm,
            "HBM_phy_area_mm2": hbm_phy_area_mm2,
            "Total_area_mm2_7nm": total_area_mm2_7nm,
            "Total_area_mm2_7nm_with_HBM": total_area_mm2_7nm_with_HBM,
        },
    }
    result_return["latency_breakdown_ns"]["commit_open_ns"] = commit_open_ns
    result_return["latency_breakdown_ns"]["total_latency_with_commit_open_ns"] = int(
            result_return["latency_breakdown_ns"]["total_latency_proof_part"]
            + result_return["latency_breakdown_ns"]["commit_open_ns"]
    )

    return result_return


def _collect_one_result(task):
    proof_json_path = Path(task["proof_json_path"])
    output_json_path = Path(task["output_json_path"])
    num_MSM_PEs = task["num_MSM_PEs"]
    dp_mul = task["dp_mul"]
    sim_data_root = task["sim_data_root"]

    try:
        with proof_json_path.open("r", encoding="utf-8") as f:
            proof_data = json.load(f)
    except json.JSONDecodeError as e:
        return {
            "status": "invalid_proof_json",
            "proof_json_path": str(proof_json_path),
            "output_json_path": str(output_json_path),
            "error": str(e),
        }

    try:
        merged_result = extract_actual_cost_dict(
            proof_data=proof_data,
            num_MSM_PEs=num_MSM_PEs,
            dp_mul=dp_mul,
            sim_data_root=sim_data_root,
        )
    except (json.JSONDecodeError, FileNotFoundError, KeyError, ValueError) as e:
        return {
            "status": "skipped",
            "proof_json_path": str(proof_json_path),
            "output_json_path": str(output_json_path),
            "error": f"{type(e).__name__}: {e}",
        }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(merged_result, f, indent=2)

    return {
        "status": "dumped",
        "proof_json_path": str(proof_json_path),
        "output_json_path": str(output_json_path),
    }


if __name__ == "__main__":
    args = _parse_args()
    script_dir = Path(__file__).resolve().parent
    sim_data_root = script_dir / "sim_data"

    model_name = "gpt2-small"
    squeeze_merge = "SqueezeMerge_1"
    num_MSM_PEs_list = [1, 2, 4, 8, 16, 32, 64]

    proof_root = sim_data_root / model_name / squeeze_merge
    output_root = sim_data_root / "latency_area_result" / model_name / squeeze_merge

    tasks = []
    for compile_dir in sorted(proof_root.glob("compile_*")):
        if not compile_dir.is_dir():
            continue
        for bw_dir in sorted(compile_dir.glob("bw_*")):
            if not bw_dir.is_dir():
                continue
            for build_dir in sorted(bw_dir.glob("build_thput_*_dp_mul_*")):
                if not build_dir.is_dir():
                    continue
                try:
                    dp_mul = int(build_dir.name.split("_dp_mul_")[-1])
                except ValueError:
                    continue

                for proof_json_path in sorted(build_dir.glob("*.json")):
                    for num_MSM_PEs in num_MSM_PEs_list:
                        output_json_path = (
                            output_root
                            / compile_dir.name
                            / bw_dir.name
                            / build_dir.name
                            / f"MSM_pe_{num_MSM_PEs}_{proof_json_path.name}"
                        )
                        tasks.append(
                            {
                                "proof_json_path": str(proof_json_path),
                                "output_json_path": str(output_json_path),
                                "num_MSM_PEs": num_MSM_PEs,
                                "dp_mul": dp_mul,
                                "sim_data_root": str(sim_data_root),
                            }
                        )

    print(f"Prepared {len(tasks)} collect-result tasks for {model_name}/{squeeze_merge}.")

    if args.multiprocess:
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(_collect_one_result, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Collecting result jsons"):
                result = future.result()
                if result["status"] == "dumped":
                    print(f"Dumped {result['output_json_path']}")
                else:
                    print(
                        f"Skipped {result['proof_json_path']} for {result['output_json_path']}: "
                        f"{result['error']}"
                    )
    else:
        for task in tqdm(tasks, desc="Collecting result jsons"):
            result = _collect_one_result(task)
            if result["status"] == "dumped":
                print(f"Dumped {result['output_json_path']}")
            else:
                print(
                    f"Skipped {result['proof_json_path']} for {result['output_json_path']}: "
                    f"{result['error']}"
                )

    print("collect_result.py end.")
