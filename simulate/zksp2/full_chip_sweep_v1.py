from .databus_cost import databus_cost_v1
from .params import *
import numpy as np
from .util import is_pareto_efficient, input_mle_size_vanilla
import math
from .sumcheck_models import *

# # this should take the in the permcheck_mle_msm datapoint, extract the latency and bw stats, and based off that, see how long it takes zerocheck to run
def get_masked_zerocheck_latency(num_vars, permcheck_data, bw_limit, sumcheck_hw_config, masked_sumcheck_polynomial):
    
    permcheck_mle_cycles, permcheck_mle_total_cycles, avg_msm_bw = permcheck_data
    assert permcheck_mle_cycles < permcheck_mle_total_cycles

    num_sumcheck_pes, num_eval_engines, num_product_lanes, onchip_mle_size = sumcheck_hw_config

    sumcheck_hardware_params = num_sumcheck_pes, num_eval_engines, num_product_lanes, mle_update_latency, extensions_latency, modmul_latency, onchip_mle_size
    
    # we'll asssume that the zerocheck can run right after permcheck mle cycles. assuming average msm bandwidth, we have
    # whatever is leftover to run the zerocheck

    remaining_bw = bw_limit - avg_msm_bw
    supplemental_data = bits_per_scalar, remaining_bw, freq
    # print(masked_sumcheck_polynomial)
    print(remaining_bw, avg_msm_bw)
    masked_latency, *_ = performance_model(num_vars, masked_sumcheck_polynomial, "zerocheck", sumcheck_hardware_params, [avg_bits_per_witness_word, jf_99percent_skip_fraction_dict], supplemental_data)
    # print(masked_latency)
    new_permcheck_total_latency = max(masked_latency + permcheck_mle_cycles, permcheck_mle_total_cycles)
    return new_permcheck_total_latency, masked_latency

def get_pareto_sumcheck_points(num_vars, sumcheck_stats_dict, batch_eval_data, mle_combine_model_data):
    
    sumcheck_design_space = dict()
    for sumcheck_hardware_config, data in sumcheck_stats_dict.items():
        num_pes, num_eval_engines, num_product_lanes, onchip_mle_size = sumcheck_hardware_config
        # print(f"num_pes: {num_pes}, num_eval_engines: {num_eval_engines}, num_product_lanes: {num_product_lanes}, onchip_mle_size: {onchip_mle_size}")
        num_sumcheck_pes, num_eval_engines, num_product_lanes, onchip_mle_size = sumcheck_hardware_config

        total_mle_combine_latency = mle_combine_model_data[onchip_mle_size]['total_latency']                  # <---------------------- use this

        zerocheck_latency = sumcheck_stats_dict[sumcheck_hardware_config]['zerocheck']
        permcheck_latency = sumcheck_stats_dict[sumcheck_hardware_config]['permcheck']
        opencheck_latency = sumcheck_stats_dict[sumcheck_hardware_config]['opencheck']

        sumcheck_latency_breakdown = (zerocheck_latency, permcheck_latency, opencheck_latency)
        total_sumcheck_latency = sum(sumcheck_latency_breakdown)                                              # <---------------------- use this

        # sumcheck related area
        rr_ctrl_area_mm2 = data['rr_ctrl_area_mm2']
        eval_engine_area_mm2 = data['eval_engine_area_mm2']
        
        # sumcheck registers
        num_delay_buffers = data['num_delay_buffers']
        delay_buffer_reg_area = num_delay_buffers*reg_area

        # Sumcheck SRAM buffer area
        sumcheck_buffer_area_mb = num_sumcheck_sram_buffers*onchip_mle_size*bits_per_scalar/BITS_PER_MB
        sumcheck_buffer_area_mm2 = sumcheck_buffer_area_mb*MB_CONVERSION_FACTOR

        total_sumcheck_area = rr_ctrl_area_mm2 + eval_engine_area_mm2 + delay_buffer_reg_area + sumcheck_buffer_area_mm2

        # mle combine area
        mle_combine_buffer_area_mb = num_mle_combine_sram_buffers*onchip_mle_size*bits_per_scalar/BITS_PER_MB
        mle_combine_buffer_area_mm2 = mle_combine_buffer_area_mb*MB_CONVERSION_FACTOR
        total_mle_combine_area = mle_combine_modmul_area + mle_combine_buffer_area_mm2 + barycentric_reg_area

        # controller for build MLE
        sleep_ctrl_area_mm2 = data['sleep_ctrl_area_mm2']
        hash_challenge_area = opencheck_build_mle_reg_area

        multifunction_tree_buffer_area_mb = num_multifunction_tree_sram_buffers*onchip_mle_size*bits_per_scalar/BITS_PER_MB
        multifunction_tree_buffer_area_mm2 = multifunction_tree_buffer_area_mb*MB_CONVERSION_FACTOR

        fixed_multifunction_tree_area = multifunction_tree_buffer_area_mm2 + hash_challenge_area + sleep_ctrl_area_mm2

        # multifunction tree stats
        forest_stats = data['forest_stats']
        for pl_depth in range(1, max_pl_offset):
            multifunction_tree_hw_cost = forest_stats[pl_depth]
            tree_mul_object = forest_stats['module']
            multifunction_tree_regs = multifunction_tree_hw_cost['req_mem_reg_num']
            multifunction_tree_modmuls = multifunction_tree_hw_cost['req_mod_mul_num']
            multifunction_tree_modadds = multifunction_tree_hw_cost['req_mod_add_num']

            multifunction_tree_regs_mm2 = multifunction_tree_regs*reg_area
            multifunction_tree_modmuls_mm2 = multifunction_tree_modmuls*modmul_area
            multifunction_tree_modadds_mm2 = multifunction_tree_modadds*modadd_area

            total_multifunction_tree_area = multifunction_tree_regs_mm2 + multifunction_tree_modmuls_mm2 + multifunction_tree_modadds_mm2 + fixed_multifunction_tree_area

            total_batch_eval_latency = batch_eval_data[num_vars][num_sumcheck_pes][num_product_lanes][num_eval_engines][pl_depth]['batch_eval_latency_max'] 

            final_eval_latency = max(tree_mul_object.get_mle_batch_eval_cost(num_vars_list=[num_vars]))   # <---------------------- use this

            total_latency = total_sumcheck_latency + total_mle_combine_latency + total_batch_eval_latency + final_eval_latency
            total_area = total_sumcheck_area + total_mle_combine_area + total_multifunction_tree_area
            sumcheck_design_space[num_sumcheck_pes, num_eval_engines, num_product_lanes, onchip_mle_size, pl_depth] = (int(math.ceil(total_latency)), round(total_area, 3), total_sumcheck_latency, total_mle_combine_latency, total_batch_eval_latency, final_eval_latency)
    
    # for k, v in sumcheck_design_space.items():
    #     print(k, v)
    
    print("Design space size:", len(sumcheck_design_space))
    print()
    # print(len(sumcheck_design_space))
    costs = np.array([[latency, area] for latency, area, *_ in sumcheck_design_space.values()])
    pareto_mask = is_pareto_efficient(costs)
    pareto_design_points = {label: design for i, (label, design) in enumerate(sumcheck_design_space.items()) if pareto_mask[i]}
    pareto_design_points_sorted = dict(
        sorted(pareto_design_points.items(), key=lambda x: (x[1][0], x[1][1]))
    )
    print("\nPareto-optimal design points:")
    for idx, (dp, val) in enumerate(pareto_design_points_sorted.items()):
        print("Design", idx+1, dp, val)

    # tolerance = 0.00

    # # Create a new list for designs within 10% of Pareto optimality
    # near_pareto_designs = []
    # for dp, val in sumcheck_design_space.items():
    #     for pareto_val in pareto_design_points.values():
    #         if val[0] <= (1 + tolerance) * pareto_val[0] and val[1] <= (1 + tolerance) * pareto_val[1]:
    #             near_pareto_designs.append((dp, val))
    #             break

    # print(f"\nDesigns within {tolerance*100}% of Pareto optimality:")
    # for idx, (dp, val) in enumerate(near_pareto_designs):
    #     print("Design", idx+1, dp, val)

    return [dp for dp, _ in pareto_design_points_sorted.items()]


def get_full_chip_sweep_sc_pareto_v1(target_num_vars_range, comprehensive_msm_data_dict, sumcheck_core_stats, sumcheck_pareto_indices, batch_eval_data, mle_combine_model_data, off_chip_bandwidth, hbm_area_estimate, onchip_sram_penalty=False, mask_sc_opt=False, masked_sumcheck_polynomial=None):


    full_chip_designs_dict = dict()

    # transcript latencies
    num_witness_transcript_appends = 3
    num_zerocheck_transcript_appends = 1
    num_permcheck_transcript_generations = 2
    num_permcheck_msm_transcript_appends = 2
    num_permcheck_msm_transcript_generations = 1
    num_zerocheck_in_permcheck_transcript_appends = 1
    r_pi_transcript_generations = 1
    mle_eval_point_generations = 5

    for num_vars in target_num_vars_range:
        num_zerocheck_transcript_generations = num_vars
        num_zerocheck_in_permcheck_transcript_generations = num_vars

        total_transcript_latency = transcript_latency * (num_witness_transcript_appends + num_zerocheck_transcript_appends + num_permcheck_transcript_generations +
            num_permcheck_msm_transcript_appends + num_permcheck_msm_transcript_generations + num_zerocheck_in_permcheck_transcript_appends +
            r_pi_transcript_generations + mle_eval_point_generations + num_zerocheck_transcript_generations + num_zerocheck_in_permcheck_transcript_generations)    # <---------------------- use this
        
        for core_key in ["single_core", "dual_core"]:
            num_msm_cores = 1 if core_key == "single_core" else 2
    
            msm_data_dict = comprehensive_msm_data_dict[num_vars][core_key]

            for (msm_design, num_frac_mle_units), data_dict in msm_data_dict.items():

                num_msm_pes = msm_design[2]

                total_msm_related_latency = data_dict['total_msm_latency']   # <---------------------- use this
                
                msm_area_stats = data_dict['msm_area_stats']
                permcheck_mle_area_stats = data_dict['permcheck_mle_area_stats']

                # fixed area stats, no resource sharing
                msm_mem_size_mb = msm_area_stats['total_memory_mb']
                msm_mem_area = msm_area_stats['total_memory_area']
                msm_logic_area = msm_area_stats['padd_area']

                total_msm_area = msm_logic_area + msm_mem_area

                sparse_msm_total_latency = data_dict['sparse_msm_total_latency']
                permcheck_mle_msm_total_latency = data_dict['permcheck_mle_msm_total_latency']
                permcheck_mle_total_latency = data_dict['permcheck_mle_total_latency']
                permcheck_msm_avg_bw = data_dict['permcheck_mle_msm_bw_stats'][1][1]

                masked_zerocheck_permcheck_data = permcheck_mle_total_latency, permcheck_mle_msm_total_latency, permcheck_msm_avg_bw

                permcheck_msm_only_latency = data_dict['permcheck_msm_only_latency']
                permcheck_nd_latency       = data_dict['permcheck_nd_latency']
                permcheck_frac_mle_latency = data_dict['permcheck_frac_mle_latency']
                permcheck_prod_mle_latency = data_dict['permcheck_prod_mle_latency']

                # this is used for checking an assumption downstream
                polyopen_msm_total_latency = data_dict['polyopen_msm_total_latency']
                polyopen_msm_only_latency = data_dict['polyopen_msm_only_latency']
                polyopen_g_prime_latency = data_dict['polyopen_g_prime_latency']
                polyopen_point_merge_latency = data_dict['polyopen_point_merge_latency']

                msm_latency_breakdown = (sparse_msm_total_latency, permcheck_mle_msm_total_latency, polyopen_msm_total_latency)
                

                # frac mle memory data
                frac_mle_sram_mb = permcheck_mle_area_stats["frac_mle_sram_mb"]
                frac_mle_sram_area = permcheck_mle_area_stats["frac_mle_sram_area"]

                # permcheck mle logic area
                permcheck_mle_logic_area_breakdown = permcheck_mle_area_stats['permcheck_mle_logic_area_breakdown']

                # rewrite this to have registers in the area breakdown
                total_nd_area = permcheck_mle_logic_area_breakdown["nd_area"]
                nd_modmul_area = permcheck_mle_logic_area_breakdown["nd_modmul_area"]
                nd_modadd_area = permcheck_mle_logic_area_breakdown["nd_modadd_area"]
                nd_reg_area = permcheck_mle_logic_area_breakdown["nd_reg_area"]
                modinv_area = permcheck_mle_logic_area_breakdown["modinv_area"]
                frac_mle_modmul_area = permcheck_mle_logic_area_breakdown["frac_mle_modmul_area"]
                frac_mle_reg_area = permcheck_mle_logic_area_breakdown["frac_mle_reg_area"]
                frac_mle_nonmem_area = permcheck_mle_logic_area_breakdown["frac_mle_area"]

                total_frac_mle_area = frac_mle_nonmem_area + frac_mle_sram_area

                # overall memory cost
                msm_and_frac_mle_mem_mb = msm_mem_size_mb + frac_mle_sram_mb
                msm_and_frac_mle_mem_area = msm_mem_area + frac_mle_sram_area 

                sumcheck_stats_dict = sumcheck_core_stats[num_vars]
                
                for sumcheck_design in sumcheck_pareto_indices:
                    sumcheck_hardware_config = sumcheck_design[0:4]
                    num_sumcheck_pes, num_eval_engines, num_product_lanes, onchip_mle_size, pl_depth = sumcheck_design
                    
                    data = sumcheck_stats_dict[sumcheck_hardware_config]
                    
                    # print(core_key, msm_design, num_frac_mle_units, sumcheck_hardware_config)

                    # here, we should add in logic to get the zerocheck latency given the avg msm bw
                    if mask_sc_opt:
                        new_permcheck_mle_msm_latency, masked_zerocheck_latency = get_masked_zerocheck_latency(num_vars, masked_zerocheck_permcheck_data, off_chip_bandwidth, sumcheck_hardware_config, masked_sumcheck_polynomial)
                    

                    total_mle_combine_latency = mle_combine_model_data[onchip_mle_size]['total_latency']                  # <---------------------- use this

                    zerocheck_latency = sumcheck_stats_dict[sumcheck_hardware_config]['zerocheck']
                    permcheck_latency = sumcheck_stats_dict[sumcheck_hardware_config]['permcheck']
                    opencheck_latency = sumcheck_stats_dict[sumcheck_hardware_config]['opencheck']

                    sumcheck_latency_breakdown = (zerocheck_latency, permcheck_latency, opencheck_latency)
                    if mask_sc_opt:
                        sumcheck_latency_breakdown = (masked_zerocheck_latency, permcheck_latency, opencheck_latency)
                        total_sumcheck_latency = permcheck_latency + opencheck_latency
                    else:
                        total_sumcheck_latency = sum(sumcheck_latency_breakdown)                                              # <---------------------- use this


                    # sumcheck related area
                    rr_ctrl_area_mm2 = data['rr_ctrl_area_mm2']
                    eval_engine_area_mm2 = data['eval_engine_area_mm2']
                    eval_engine_modmuls = data['eval_engine_modmuls']
                    
                    # sumcheck registers
                    num_delay_buffers = data['num_delay_buffers']
                    delay_buffer_reg_area = num_delay_buffers*reg_area

                    # Sumcheck SRAM buffer area
                    sumcheck_buffer_area_mb = num_sumcheck_sram_buffers*onchip_mle_size*bits_per_scalar/BITS_PER_MB
                    sumcheck_buffer_area_mm2 = sumcheck_buffer_area_mb*MB_CONVERSION_FACTOR

                    total_sumcheck_area = rr_ctrl_area_mm2 + eval_engine_area_mm2 + delay_buffer_reg_area + sumcheck_buffer_area_mm2

                    # mle combine area
                    mle_combine_buffer_area_mb = num_mle_combine_sram_buffers*onchip_mle_size*bits_per_scalar/BITS_PER_MB
                    mle_combine_buffer_area_mm2 = mle_combine_buffer_area_mb*MB_CONVERSION_FACTOR
                    total_mle_combine_area = mle_combine_modmul_area + mle_combine_buffer_area_mm2 + barycentric_reg_area

                    # controller for build MLE
                    sleep_ctrl_area_mm2 = data['sleep_ctrl_area_mm2']
                    hash_challenge_area = opencheck_build_mle_reg_area

                    multifunction_tree_buffer_area_mb = num_multifunction_tree_sram_buffers*onchip_mle_size*bits_per_scalar/BITS_PER_MB
                    multifunction_tree_buffer_area_mm2 = multifunction_tree_buffer_area_mb*MB_CONVERSION_FACTOR

                    fixed_multifunction_tree_area = multifunction_tree_buffer_area_mm2 + hash_challenge_area + sleep_ctrl_area_mm2

                    # multifunction tree stats
                    forest_stats = data['forest_stats']
                    multifunction_tree_hw_cost = forest_stats[pl_depth]
                    tree_mul_object = forest_stats['module']
                    multifunction_tree_regs = multifunction_tree_hw_cost['req_mem_reg_num']
                    multifunction_tree_modmuls = multifunction_tree_hw_cost['req_mod_mul_num']
                    multifunction_tree_modadds = multifunction_tree_hw_cost['req_mod_add_num']

                    multifunction_tree_regs_mm2 = multifunction_tree_regs*reg_area
                    multifunction_tree_modmuls_mm2 = multifunction_tree_modmuls*modmul_area
                    multifunction_tree_modadds_mm2 = multifunction_tree_modadds*modadd_area

                    total_multifunction_tree_area = multifunction_tree_regs_mm2 + multifunction_tree_modmuls_mm2 + multifunction_tree_modadds_mm2 + fixed_multifunction_tree_area

                    total_batch_eval_latency = batch_eval_data[num_vars][num_sumcheck_pes][num_product_lanes][num_eval_engines][pl_depth]['batch_eval_latency_max'] 

                    final_eval_latency = max(tree_mul_object.get_mle_batch_eval_cost(num_vars_list=[num_vars]))   # <---------------------- use this

                    if mask_sc_opt:
                        total_msm_related_latency = sparse_msm_total_latency + new_permcheck_mle_msm_latency + polyopen_msm_total_latency

                    total_latency = total_msm_related_latency + total_sumcheck_latency + total_mle_combine_latency + total_batch_eval_latency + final_eval_latency + total_transcript_latency

                    overall_runtime =  round(total_latency/freq*1000, 3)

                    total_area_22nm = total_msm_area + total_sumcheck_area + total_nd_area + total_frac_mle_area + total_mle_combine_area + total_multifunction_tree_area + sha_area

                    max_bus_bit_width, (MSMs_crossbar_area_14nm, MSMs_crossbar_TDP_W_14nm), _ = databus_cost_v1(num_sumcheck_pes, (num_eval_engines + pl_depth), num_product_lanes, num_frac_mle_units, num_msm_cores, num_msm_pes)
                    # BTS: 1 bit/cycle = (3.06÷24576) mm2 and (45.93÷24576) TDP(W) in 7nm. F1: 1 bit/cycle = (7.26÷36352) W in 14nm.
                    BTS_noc_area_mm2_per_bit = 3.06 / 24576
                    BTS_noc_TDP_W_per_bit = 45.93 / 24576
                    F1_crossbar_TDP_W_per_bit = 7.26 / 36352
                    noc_area_estimate = max_bus_bit_width*BTS_noc_area_mm2_per_bit + 8*off_chip_bandwidth*BTS_noc_area_mm2_per_bit
                    
                    # interconnect_power_estimate = max_bus_bit_width*F1_crossbar_TDP_W_per_bit + 8*off_chip_bandwidth*F1_crossbar_TDP_W_per_bit

                    total_mem_area_22nm = msm_and_frac_mle_mem_area + sumcheck_buffer_area_mm2 + mle_combine_buffer_area_mm2 + multifunction_tree_buffer_area_mm2
                    total_mem_area_7nm = total_mem_area_22nm / scale_factor_22_to_7nm
                    total_mem_mb = msm_and_frac_mle_mem_mb + sumcheck_buffer_area_mb + mle_combine_buffer_area_mb + multifunction_tree_buffer_area_mb

                    # remove the existing onchip SRAM and replace with the zkspeed maximum
                    if onchip_sram_penalty:
                        total_area_22nm -= (multifunction_tree_buffer_area_mm2 + sumcheck_buffer_area_mm2 + mle_combine_buffer_area_mm2)
                        total_area_22nm += input_mle_size_vanilla(bits_per_scalar, 23)*MB_CONVERSION_FACTOR

                    total_area_7nm = total_area_22nm / scale_factor_22_to_7nm
                    total_area_7nm += noc_area_estimate + hbm_area_estimate
                    total_logic_area = msm_logic_area + nd_modmul_area + nd_modadd_area + frac_mle_modmul_area + multifunction_tree_modmuls_mm2 + multifunction_tree_modadds_mm2 + sha_area + rr_ctrl_area_mm2 + eval_engine_area_mm2
                    detailed_area_breakdown = {
                        "overall_area": round(total_area_7nm, 3),
                        "overall_area_without_hbm_and_interconnect": round(total_area_22nm / scale_factor_22_to_7nm, 3),
                        "module_area": {
                            "total_logic_area": round(total_logic_area / scale_factor_22_to_7nm, 3),
                            "msm_logic_area": round(msm_logic_area / scale_factor_22_to_7nm, 3),
                            "nd_area": round((nd_modmul_area + nd_modadd_area) / scale_factor_22_to_7nm, 3),
                            "frac_mle_area": round(frac_mle_modmul_area / scale_factor_22_to_7nm, 3),
                            "mle_combine_area": round(mle_combine_modmul_area / scale_factor_22_to_7nm, 3),
                            "sumcheck_core_area": round((rr_ctrl_area_mm2 + eval_engine_area_mm2) / scale_factor_22_to_7nm, 3),
                            "multifunction_tree_area": round((multifunction_tree_modmuls_mm2 + multifunction_tree_modadds_mm2) / scale_factor_22_to_7nm, 3),
                            "sha_area": round(sha_area / scale_factor_22_to_7nm, 3)
                        },
                        # update with memory specific breakdowns
                        "memory_area_mm2": {
                            "on_chip_memory_area": round(total_mem_area_7nm, 3),
                            "msm_memory_area": round(msm_mem_area / scale_factor_22_to_7nm, 3),
                            "frac_mle_memory_area": round(frac_mle_sram_area / scale_factor_22_to_7nm, 3),
                            "mle_combine_memory_area": round(mle_combine_buffer_area_mm2 / scale_factor_22_to_7nm, 3),
                            "multifunction_tree_memory_area": round((multifunction_tree_buffer_area_mm2) / scale_factor_22_to_7nm, 3),
                            "sumcheck_memory_area": round(sumcheck_buffer_area_mm2 / scale_factor_22_to_7nm, 3)
                        },
                        "memory_size_mb": {
                            "total_memory_size_mb": round(total_mem_mb, 3),
                            "msm_memory_size_mb": round(msm_mem_size_mb, 3),
                            "frac_mle_memory_size_mb": round(frac_mle_sram_mb, 3),
                            "mle_combine_memory_size_mb": round(mle_combine_buffer_area_mb, 3),
                            "multifunction_tree_memory_size_mb": round(multifunction_tree_buffer_area_mb, 3),
                            "sumcheck_memory_size_mb": round(sumcheck_buffer_area_mb, 3)
                        },
                        "modmul_and_modadd_counts": {
                            "nd_modmul": nd_modmul_area / modmul_area,
                            "nd_modadd": nd_modadd_area / modadd_area,
                            "frac_mle_modmul": frac_mle_modmul_area / modmul_area,
                            "multifunction_tree_modmul": multifunction_tree_modmuls,
                            "multifunction_tree_modadd": multifunction_tree_modadds,
                            "eval_engine_modmuls": eval_engine_modmuls
                        },
                        "registers_area_mm2": {
                            "total_reg_area": round((nd_reg_area + frac_mle_reg_area + multifunction_tree_regs_mm2 + hash_challenge_area + barycentric_reg_area + delay_buffer_reg_area) / scale_factor_22_to_7nm, 3),
                            "nd_reg_area": round(nd_reg_area / scale_factor_22_to_7nm, 3),
                            "frac_mle_reg_area": round(frac_mle_reg_area / scale_factor_22_to_7nm, 3),
                            "multifunction_tree_regs_area": round((multifunction_tree_regs_mm2 + hash_challenge_area) / scale_factor_22_to_7nm, 3),
                            "barycentric_reg_area (mle_combine)": round(barycentric_reg_area / scale_factor_22_to_7nm, 3),
                            "delay_buffer_reg_area (sumcheck)": round(delay_buffer_reg_area / scale_factor_22_to_7nm, 3)
                        },
                        "interconnect_area": round(noc_area_estimate, 3),
                        "hbm_area": round(hbm_area_estimate, 3)
                    }

                    
                    ################################### Latency Calculations ##########################################
                    # fine-grained breakdown

                    ws_t_latency = num_witness_transcript_appends*transcript_latency 
                    
                    gi_t_latency = (num_zerocheck_transcript_appends + num_zerocheck_transcript_generations)*transcript_latency

                    wi_t_latency = (num_permcheck_transcript_generations + num_permcheck_msm_transcript_appends + num_permcheck_msm_transcript_generations + num_zerocheck_in_permcheck_transcript_appends + num_zerocheck_in_permcheck_transcript_generations) * transcript_latency
                    
                    po_t_latency = (r_pi_transcript_generations + mle_eval_point_generations) * transcript_latency

                    full_chip_design_point_label = (num_vars, msm_design, core_key, num_frac_mle_units, sumcheck_hardware_config, pl_depth)

                    ######################################### This is the data we actually care about ##############################################
                    
                    runtime_breakdown = {
                        "witness_commit": sparse_msm_total_latency,
                        "gate_identity": zerocheck_latency,
                        "wire_identity_commit": permcheck_mle_msm_total_latency,
                        "wire_identity_sumcheck": permcheck_latency,
                        "batch_eval": total_batch_eval_latency,
                        "opencheck_latency": opencheck_latency,
                        "polyopen": total_mle_combine_latency + polyopen_msm_total_latency + final_eval_latency,
                    }
                

                    # witness_step = {
                    #         "total": sparse_msm_total_latency + ws_t_latency, 
                    #         "MSM": sparse_msm_total_latency,
                    #         "Sumcheck": 0,
                    #         "MLE Update": 0,
                    #         "Multifunction": 0,
                    #         "ND": 0,
                    #         "FracMLE": 0,
                    #         "MLE Combine": 0,
                    #         "SHA3": ws_t_latency
                    #     }
                        
                    # gate_identity_step = {
                    #     "total" : zerocheck_latency + gi_t_latency,
                    #     "MSM": 0,
                    #     "Sumcheck": zerocheck_latency,
                    #     "Multifunction": None, # update this with build MLE latency
                    #     "ND": 0,
                    #     "FracMLE": 0,
                    #     "MLE Combine": 0,
                    #     "SHA3": gi_t_latency
                    # }

                    # wire_identity_step = {
                    #     "total": permcheck_mle_msm_total_latency + permcheck_latency + wi_t_latency,
                    #     "MSM": permcheck_msm_only_latency,
                    #     "Sumcheck": permcheck_latency,
                    #     "Multifunction": None, # update this with prod MLE latency and build MLE latency
                    #     "ND": permcheck_nd_latency,
                    #     "FracMLE": permcheck_frac_mle_latency,
                    #     "MLE Combine": 0,
                    #     "SHA3": wi_t_latency
                    # }


                    # batch_eval_step = {
                    #     "total": total_batch_eval_latency, 
                    #     "MSM": 0,
                    #     "Sumcheck": 0,
                    #     "MLE Update": 0,
                    #     "Multifunction": total_batch_eval_latency,
                    #     "ND": 0,
                    #     "FracMLE": 0,
                    #     "MLE Combine": 0,
                    #     "SHA3": 0
                    # }
                    
                    # polyopen_step = {
                    #     "total": polyopen_msm_total_latency + opencheck_latency + po_t_latency,
                    #     "MSM": polyopen_msm_only_latency,
                    #     "Sumcheck": opencheck_latency,
                    #     "Multifunction": None, # update this with build MLE latency
                    #     "ND": 0,
                    #     "FracMLE": 0,
                    #     "MLE Combine": total_mle_combine_latency + polyopen_point_merge_latency + polyopen_g_prime_latency, 
                    #     "SHA3": po_t_latency
                    # }
                    # final_batch_eval_step = {
                    #     "total": final_eval_latency, 
                    #     "MSM": 0,
                    #     "Sumcheck": 0,
                    #     "MLE Update": 0,
                    #     "Multifunction": final_eval_latency,
                    #     "ND": 0,
                    #     "FracMLE": 0,
                    #     "MLE Combine": 0,
                    #     "SHA3": 0
                    # }

                    full_chip_designs_dict[full_chip_design_point_label] = \
                        {
                            "overall_runtime" : overall_runtime,
                            "overall_area"    : round(total_area_7nm, 3),
                            "total_cycles"    : total_latency,
                            "total_msm_related_latency": total_msm_related_latency,
                            "total_sumcheck_related_latency" : total_sumcheck_latency,
                            "msm_latency_breakdown" : msm_latency_breakdown,
                            "sumcheck_latency_breakdown" : sumcheck_latency_breakdown,
                            "total_mle_combine_latency": total_mle_combine_latency,
                            "total_batch_eval_latency": total_batch_eval_latency,
                            "final_eval_latency": final_eval_latency,
                            "total_transcript_latency": total_transcript_latency,
                            "detailed_area_breakdown" : detailed_area_breakdown,
                            "runtime_breakdown": runtime_breakdown
                        }

    return full_chip_designs_dict

