from .databus_cost import databus_cost
from .params import *

def get_full_chip_sweep(target_num_vars_range, comprehensive_msm_data_dict, sumcheck_sweep_params, sumcheck_core_stats, batch_eval_data, onchip_mle_data, sha_area, hbm_area, scale_factor_22_to_7nm):
    
    sumcheck_pe_unroll_factors, max_mles, mle_update_unroll_factors = sumcheck_sweep_params
    onchip_mle_size, onchip_mle_area = onchip_mle_data

    full_chip_designs_dict = dict()
    full_chip_designs_dict_with_sc_throttling = dict()

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
            r_pi_transcript_generations + mle_eval_point_generations + num_zerocheck_transcript_generations + num_zerocheck_in_permcheck_transcript_generations)
        
        onchip_mle_area_mm2 = onchip_mle_area[num_vars]
        onchip_mle_size_mb = onchip_mle_size[num_vars]
        be_nv_data = batch_eval_data[num_vars]
        for core_key in ["single_core", "dual_core"]:
            num_cores = 1 if core_key == "single_core" else 2
    
            msm_data_dict = comprehensive_msm_data_dict[num_vars][core_key]

            for (msm_design, load_rate), data_dict in msm_data_dict.items():

                total_msm_related_latency = data_dict['total_msm_latency']
                
                msm_area_stats = data_dict['msm_area_stats']
                permcheck_mle_area_stats = data_dict['permcheck_mle_area_stats']
                polyopen_mle_area_stats = data_dict['polyopen_mle_area_stats']

                # fixed area stats, no resource sharing
                msm_mem_size_mb = msm_area_stats['total_memory_mb']
                msm_mem_area = msm_area_stats['total_memory_area']
                msm_logic_area = msm_area_stats['padd_area']
               

                sparse_msm_total_latency = data_dict['sparse_msm_total_latency']
                permcheck_mle_msm_total_latency = data_dict['permcheck_mle_msm_total_latency']

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

                # fixed area stats, no resource sharing
                nd_area = permcheck_mle_logic_area_breakdown["nd_area"]
                frac_mle_area = permcheck_mle_logic_area_breakdown["frac_mle_area"]
                
                # area stats for resource sharing modules
                frac_mle_mul_tree_modmuls = permcheck_mle_logic_area_breakdown["frac_mle_mul_tree_modmuls"]
                frac_mle_mul_tree_numregs = permcheck_mle_logic_area_breakdown["frac_mle_mul_tree_numregs"]
                prod_mle_modmuls = permcheck_mle_logic_area_breakdown["prod_mle_modmuls"]
                prod_mle_numregs = permcheck_mle_logic_area_breakdown["prod_mle_numregs"]
                polyopen_mle_combine_area = polyopen_mle_area_stats['min_po_mle_area']

                # overall memory cost
                overall_mem_mb = msm_mem_size_mb + frac_mle_sram_mb + onchip_mle_size_mb
                overall_mem_area = msm_mem_area + frac_mle_sram_area + onchip_mle_area_mm2

                for num_sumcheck_core_pes in sumcheck_pe_unroll_factors:
                    # this is used to check an assumption downstream
                    batch_eval_latency = be_nv_data[num_sumcheck_core_pes][0]['total_cycles']

                    # area stats for resource sharing modules
                    batch_eval_modmuls = be_nv_data[num_sumcheck_core_pes][0]['req_mod_mul_num']
                    batch_eval_modadds = be_nv_data[num_sumcheck_core_pes][0]['req_mod_add_num']
                    batch_eval_regs = be_nv_data[num_sumcheck_core_pes][0]['req_mem_reg_num']

                    final_batch_eval_modmuls = be_nv_data[num_sumcheck_core_pes][1]['req_mod_mul_num']
                    final_batch_eval_modadds = be_nv_data[num_sumcheck_core_pes][1]['req_mod_add_num']
                    final_batch_eval_regs = be_nv_data[num_sumcheck_core_pes][1]['req_mem_reg_num']

                    final_eval_latency = be_nv_data[num_sumcheck_core_pes][1]['total_cycles']
                    
                    for num_mles_in_parallel in range(max_mles, 0, -1):
                        for pes_per_mle_update in mle_update_unroll_factors:
                            sumcheck_design = (num_vars, num_sumcheck_core_pes, num_mles_in_parallel, pes_per_mle_update)

                            #################################### Area Calculations ############################################

                            sumcheck_related_mle_combine_area = sumcheck_core_stats[sumcheck_design]['mle_combine_area']
                            
                            # resource sharing MLE combine
                            mle_combine_area = max(sumcheck_related_mle_combine_area, polyopen_mle_combine_area)

                            mle_combine_max_area = sumcheck_related_mle_combine_area + polyopen_mle_combine_area

                            # fixed sumcheck area
                            sumcheck_core_area = sumcheck_core_stats[sumcheck_design]['sumcheck_core_area']
                            mle_update_core_area = sumcheck_core_stats[sumcheck_design]['mle_update_core_area']

                            # resource sharing multifunction tree
                            build_mle_stats = sumcheck_core_stats[sumcheck_design]['build_mle_stats']
                            build_mle_modmuls = build_mle_stats["build_mle_modmuls"]
                            build_mle_modadds = build_mle_stats["build_mle_modadds"]
                            build_mle_regs = build_mle_stats["build_mle_regs"]

                            single_build_mle_latency = build_mle_stats["single_build_mle_latency"]

                            permcheck_multifunction_tree_modmuls = frac_mle_mul_tree_modmuls + prod_mle_modmuls
                            permcheck_multifunction_tree_numregs = frac_mle_mul_tree_numregs + prod_mle_numregs

                            polyopen_multifunction_tree_modmuls = batch_eval_modmuls + build_mle_modmuls
                            polyopen_multifunction_tree_numregs = batch_eval_regs + build_mle_regs

                            multifunction_tree_modmuls = max(permcheck_multifunction_tree_modmuls, polyopen_multifunction_tree_modmuls)
                            multifunction_tree_modmul_area = multifunction_tree_modmuls*modmul_area

                            multifunction_tree_numregs = max(permcheck_multifunction_tree_numregs, polyopen_multifunction_tree_numregs)
                            multifunction_tree_reg_area = multifunction_tree_numregs*reg_area

                            multifunction_tree_modadd_area = (batch_eval_modadds + build_mle_modadds)*modadd_area
                            
                            multifunction_tree_area = multifunction_tree_modadd_area + multifunction_tree_modmul_area + multifunction_tree_reg_area

                            # no sharing
                            multifunction_tree_modmuls_noshare = batch_eval_modmuls + build_mle_modmuls*(6/5) + permcheck_multifunction_tree_modmuls + final_batch_eval_modmuls
                            multifunction_tree_modadd_noshare = batch_eval_modadds + build_mle_modadds*(6/5) + final_batch_eval_modadds
                            multifunction_tree_reg_noshare = batch_eval_regs + build_mle_regs*(6/5) + permcheck_multifunction_tree_numregs + final_batch_eval_regs
                            multifunction_tree_area_noshare = multifunction_tree_modadd_noshare*modadd_area + multifunction_tree_modmuls_noshare*modmul_area + multifunction_tree_reg_noshare*reg_area
                            
                            # total logic area = msm + nd + frac + mle update + sumcheck + sha + mle combine + tree
                            total_logic_area = msm_logic_area + nd_area + frac_mle_area + mle_combine_area + sha_area + sumcheck_core_area + mle_update_core_area + multifunction_tree_area

                            max_bit_width, _ = databus_cost(num_sumcheck_core_pes, load_rate, num_sumcheck_core_pes, num_sumcheck_core_pes, 1)
                            interconnect_estimate = max_bit_width*reg_area

                            overall_area = total_logic_area + overall_mem_area + interconnect_estimate

                            overall_area /= scale_factor_22_to_7nm
                            overall_area_without_hbm = round(overall_area, 3)
                            overall_area += hbm_area
                            overall_area = round(overall_area, 3)

                            detailed_area_breakdown = {
                                "overall_area_without_hbm": overall_area_without_hbm,
                                "overall_area" : overall_area,
                                "total_logic_area" : round(total_logic_area/scale_factor_22_to_7nm, 3),
                                "interconnected_area" : round(interconnect_estimate/scale_factor_22_to_7nm, 3),
                                "on_chip_memory" : round(overall_mem_area/scale_factor_22_to_7nm, 3),
                                "msm_logic_area" : round(msm_logic_area/scale_factor_22_to_7nm, 3),
                                "nd_area" : round(nd_area/scale_factor_22_to_7nm, 3),
                                "frac_mle_area" : round(frac_mle_area/scale_factor_22_to_7nm, 3),
                                "mle_combine_area" : round(mle_combine_area/scale_factor_22_to_7nm, 3),
                                "sha_area" : round(sha_area/scale_factor_22_to_7nm, 3),
                                "sumcheck_core_area" : round(sumcheck_core_area/scale_factor_22_to_7nm, 3),
                                "mle_update_core_area" : round(mle_update_core_area/scale_factor_22_to_7nm, 3),
                                "multifunction_tree_area" : round(multifunction_tree_area/scale_factor_22_to_7nm, 3),
                                "multifunction_tree_area_noshare": round(multifunction_tree_area_noshare/scale_factor_22_to_7nm, 3),
                                "hbm_area": hbm_area
                            }

                            
                            ################################### Latency Calculations ##########################################

                            total_sumcheck_related_latency = sumcheck_core_stats[sumcheck_design]['total_sumcheck_latency']
                            unthrottled_sumcheck_latency_breakdown = sumcheck_core_stats[sumcheck_design]['latency_breakdowns']

                            # check our batch evaluation assumptions
                            opencheck_latency = sumcheck_core_stats[sumcheck_design]['opencheck_latency']
                            assert opencheck_latency + polyopen_msm_total_latency > batch_eval_latency

                            # add in the final eval latency here
                            overall_latency = total_msm_related_latency + total_sumcheck_related_latency + final_eval_latency + total_transcript_latency

                            sumcheck_related_bw_data = sumcheck_core_stats[sumcheck_design]['bandwidth_stats']

                            # runtime in ms
                            overall_runtime = round(overall_latency/freq*1000, 3)
                            msm_runtime_portion = round(total_msm_related_latency / overall_latency, 3)
                            sumcheck_runtime_portion = round(total_sumcheck_related_latency / overall_latency, 3)
                            final_eval_runtime_portion = round(final_eval_latency / overall_latency, 3)

                            # fine-grained breakdown

                            ws_t_latency = num_witness_transcript_appends*transcript_latency 
                            witness_step = {
                                "total": sparse_msm_total_latency + ws_t_latency, 
                                "MSM": sparse_msm_total_latency,
                                "Sumcheck": 0,
                                "MLE Update": 0,
                                "Multifunction": 0,
                                "ND": 0,
                                "FracMLE": 0,
                                "MLE Combine": 0,
                                "SHA3": ws_t_latency
                            }
                            
                            gi_t_latency = (num_zerocheck_transcript_appends + num_zerocheck_transcript_generations)*transcript_latency
                            gate_identity_step = {
                                "total" : sumcheck_core_stats[sumcheck_design]['zerocheck_latency'] + gi_t_latency,
                                "MSM": 0,
                                "Sumcheck": unthrottled_sumcheck_latency_breakdown[0],
                                "MLE Update": unthrottled_sumcheck_latency_breakdown[1],
                                "Multifunction": single_build_mle_latency,
                                "ND": 0,
                                "FracMLE": 0,
                                "MLE Combine": 0,
                                "SHA3": gi_t_latency
                            }

                            wi_t_latency = (num_permcheck_transcript_generations + num_permcheck_msm_transcript_appends + num_permcheck_msm_transcript_generations + num_zerocheck_in_permcheck_transcript_appends + num_zerocheck_in_permcheck_transcript_generations) * transcript_latency
                            wire_identity_step = {
                                "total": permcheck_mle_msm_total_latency + sumcheck_core_stats[sumcheck_design]['permcheck_latency'] + wi_t_latency,
                                "MSM": permcheck_msm_only_latency,
                                "Sumcheck": unthrottled_sumcheck_latency_breakdown[2],
                                "MLE Update": unthrottled_sumcheck_latency_breakdown[3],
                                "Multifunction": single_build_mle_latency + permcheck_prod_mle_latency,
                                "ND": permcheck_nd_latency,
                                "FracMLE": permcheck_frac_mle_latency,
                                "MLE Combine": 0,
                                "SHA3": wi_t_latency
                            }

                            batch_eval_step = {
                                "total": batch_eval_latency, 
                                "MSM": 0,
                                "Sumcheck": 0,
                                "MLE Update": 0,
                                "Multifunction": batch_eval_latency,
                                "ND": 0,
                                "FracMLE": 0,
                                "MLE Combine": 0,
                                "SHA3": 0
                            }
                            po_t_latency = (r_pi_transcript_generations + mle_eval_point_generations) * transcript_latency
                            polyopen_step = {
                                "total": polyopen_msm_total_latency + sumcheck_core_stats[sumcheck_design]['opencheck_latency'] + po_t_latency,
                                "MSM": polyopen_msm_only_latency,
                                "Sumcheck": unthrottled_sumcheck_latency_breakdown[4],
                                "MLE Update": unthrottled_sumcheck_latency_breakdown[5],
                                "Multifunction": single_build_mle_latency,
                                "ND": 0,
                                "FracMLE": 0,
                                "MLE Combine": single_build_mle_latency + polyopen_point_merge_latency + polyopen_g_prime_latency, 
                                "SHA3": po_t_latency
                            }

                            final_batch_eval_step = {
                                "total": final_eval_latency, 
                                "MSM": 0,
                                "Sumcheck": 0,
                                "MLE Update": 0,
                                "Multifunction": final_eval_latency,
                                "ND": 0,
                                "FracMLE": 0,
                                "MLE Combine": 0,
                                "SHA3": 0
                            }

                            full_chip_design_point_label = (num_vars, msm_design, core_key, load_rate, (num_sumcheck_core_pes, num_mles_in_parallel, pes_per_mle_update))
                            full_chip_designs_dict[full_chip_design_point_label] = \
                                {
                                    "overall_runtime" : overall_runtime,
                                    "overall_area"    : overall_area,
                                    "overall_area_without_hbm": overall_area_without_hbm,
                                    "overall_cycles"  : overall_latency,
                                    "overall_mem_mb" : overall_mem_mb,
                                    "total_msm_related_latency": total_msm_related_latency,
                                    "total_sumcheck_related_latency" : total_sumcheck_related_latency,
                                    "final_eval_latency" : final_eval_latency,
                                    "msm_latency_breakdown" : msm_latency_breakdown,
                                    # "unthrottled_sumcheck_latency_breakdown" : unthrottled_sumcheck_latency_breakdown,
                                    "msm_runtime_portion" : msm_runtime_portion,
                                    "sumcheck_runtime_portion" : sumcheck_runtime_portion,
                                    "final_eval_runtime_portion" : final_eval_runtime_portion,
                                    "detailed_area_breakdown" : detailed_area_breakdown,
                                    "witness_step": witness_step,
                                    "gate_identity_step": gate_identity_step,
                                    "wire_identity_step": wire_identity_step,
                                    "batch_eval_step": batch_eval_step,
                                    "polyopen_step": polyopen_step,
                                    "final_batch_eval_step": final_batch_eval_step
                                }


                            ######################################### This is the data we actually care about ##############################################

                            total_sumcheck_related_latency_throttled = sumcheck_core_stats[sumcheck_design]['throttled_latencies']['total_sumcheck_latency']
                            throttled_sumcheck_latency_breakdown = sumcheck_core_stats[sumcheck_design]['throttled_latencies']['throttled_latency_breakdown']

                            # add in the final eval latency here
                            overall_latency_throttled = total_msm_related_latency + total_sumcheck_related_latency_throttled + final_eval_latency + total_transcript_latency

                            sumcheck_related_bw_data_throttled = sumcheck_core_stats[sumcheck_design]['throttled_bandwidths']

                            # runtime in ms
                            overall_runtime_throttled = round(overall_latency_throttled/freq*1000, 3)
                            msm_runtime_portion = round(total_msm_related_latency / overall_latency_throttled, 3)
                            sumcheck_runtime_portion = round(total_sumcheck_related_latency_throttled / overall_latency_throttled, 3)
                            final_eval_runtime_portion = round(final_eval_latency / overall_latency_throttled, 3)

                            zerocheck_sc_latency = sumcheck_core_stats[sumcheck_design]['throttled_latencies']['zerocheck_sc_latency']
                            zerocheck_mu_latency = sumcheck_core_stats[sumcheck_design]['throttled_latencies']['zerocheck_mu_latency']
                            permcheck_sc_latency = sumcheck_core_stats[sumcheck_design]['throttled_latencies']['permcheck_sc_latency']
                            permcheck_mu_latency = sumcheck_core_stats[sumcheck_design]['throttled_latencies']['permcheck_mu_latency']
                            opencheck_sc_latency = sumcheck_core_stats[sumcheck_design]['throttled_latencies']['opencheck_sc_latency']
                            opencheck_mu_latency = sumcheck_core_stats[sumcheck_design]['throttled_latencies']['opencheck_mu_latency']

                            gate_identity_step = {
                                "total" : zerocheck_sc_latency + zerocheck_mu_latency + gi_t_latency,
                                "MSM": 0,
                                "Sumcheck": unthrottled_sumcheck_latency_breakdown[0],
                                "MLE Update": unthrottled_sumcheck_latency_breakdown[1],
                                "Multifunction": single_build_mle_latency,
                                "ND": 0,
                                "FracMLE": 0,
                                "MLE Combine": 0,
                                "SHA3": gi_t_latency
                            }

                            wire_identity_step = {
                                "total": permcheck_mle_msm_total_latency + permcheck_sc_latency + permcheck_mu_latency + wi_t_latency,
                                "MSM": permcheck_msm_only_latency,
                                "Sumcheck": unthrottled_sumcheck_latency_breakdown[2],
                                "MLE Update": unthrottled_sumcheck_latency_breakdown[3],
                                "Multifunction": single_build_mle_latency + permcheck_prod_mle_latency,
                                "ND": permcheck_nd_latency,
                                "FracMLE": permcheck_frac_mle_latency,
                                "MLE Combine": 0,
                                "SHA3": wi_t_latency
                            }

                            polyopen_step = {
                                "total": polyopen_msm_total_latency + opencheck_sc_latency + opencheck_mu_latency + po_t_latency,
                                "MSM": polyopen_msm_only_latency,
                                "Sumcheck": unthrottled_sumcheck_latency_breakdown[4],
                                "MLE Update": unthrottled_sumcheck_latency_breakdown[5],
                                "Multifunction": single_build_mle_latency,
                                "ND": 0,
                                "FracMLE": 0,
                                "MLE Combine": single_build_mle_latency + polyopen_point_merge_latency + polyopen_g_prime_latency, 
                                "SHA3": po_t_latency
                            }

                            full_chip_designs_dict_with_sc_throttling[full_chip_design_point_label] = \
                                {
                                    "overall_runtime" : overall_runtime_throttled,
                                    "overall_area"    : overall_area,
                                    "overall_area_without_hbm": overall_area_without_hbm,
                                    "overall_cycles"  : overall_latency_throttled,
                                    "overall_mem_mb" : overall_mem_mb,
                                    "total_msm_related_latency": total_msm_related_latency,
                                    "total_sumcheck_related_latency" : total_sumcheck_related_latency_throttled,
                                    "final_eval_latency" : final_eval_latency,
                                    "msm_latency_breakdown" : msm_latency_breakdown,
                                    # "throttled_sumcheck_latency_breakdown" : throttled_sumcheck_latency_breakdown,
                                    "msm_runtime_portion" : msm_runtime_portion,
                                    "sumcheck_runtime_portion" : sumcheck_runtime_portion,
                                    "final_eval_runtime_portion" : final_eval_runtime_portion,
                                    "detailed_area_breakdown" : detailed_area_breakdown,
                                    "witness_step": witness_step,
                                    "gate_identity_step": gate_identity_step,
                                    "wire_identity_step": wire_identity_step,
                                    "batch_eval_step": batch_eval_step,
                                    "polyopen_step": polyopen_step,
                                    "final_batch_eval_step": final_batch_eval_step
                                }

    return full_chip_designs_dict, full_chip_designs_dict_with_sc_throttling
