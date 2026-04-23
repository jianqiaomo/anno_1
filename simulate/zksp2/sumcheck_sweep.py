from .sumcheck_models_v0 import *
from .build_mle import *
from .params import *

def get_fetch_latency_v0(num_vars, num_mles, bitwidth, bandwidth, freq, mle_update=False):
    if mle_update:
        lower_bound = 1
        available_bw = 2/3*bandwidth
    else:
        lower_bound = 0
        available_bw = bandwidth

    running_sum = 0
    for i in range(num_vars, lower_bound, -1):
        num_words = (1 << i)*num_mles
        running_sum += math.ceil(num_words*bitwidth/BITS_PER_GB/available_bw*freq)
    return running_sum

def throttle_sumcheck_v0(latency_breakdowns, bw_list, mle_stats, other_stats):
    
    num_vars, bits_per_scalar, available_bw, freq = other_stats
    zerocheck_mles, permcheck_mles, opencheck_mles, num_mles_in_parallel_zc, num_mles_in_parallel_pc, num_mles_in_parallel_oc = mle_stats

    zc_sc_latency, zc_mu_latency, pc_sc_latency, pc_mu_latency, oc_sc_latency, oc_mu_latency, total_transcript_latency = latency_breakdowns
    zerocheck_peak_bw, zerocheck_mu_peak_bw, permcheck_peak_bw, permcheck_mu_peak_bw, opencheck_peak_bw, opencheck_mu_peak_bw = bw_list

    if zerocheck_peak_bw > available_bw:
        zc_sc_latency = get_fetch_latency_v0(num_vars, zerocheck_mles, bits_per_scalar, available_bw, freq)
        zerocheck_peak_bw = available_bw

    if permcheck_peak_bw > available_bw:
        pc_sc_latency = get_fetch_latency_v0(num_vars, permcheck_mles, bits_per_scalar, available_bw, freq)
        permcheck_peak_bw = available_bw
    
    if opencheck_peak_bw > available_bw:
        oc_sc_latency = get_fetch_latency_v0(num_vars, opencheck_mles, bits_per_scalar, available_bw, freq)
        opencheck_peak_bw = available_bw

    # mle update
    if zerocheck_mu_peak_bw > available_bw:
        zc_mu_latency = get_fetch_latency_v0(num_vars, num_mles_in_parallel_zc, bits_per_scalar, available_bw, freq, mle_update=True)
        scale_factor = math.ceil(zerocheck_mles/num_mles_in_parallel_zc)
        zc_mu_latency *= scale_factor
        zerocheck_mu_peak_bw = available_bw

    if permcheck_mu_peak_bw > available_bw:
        pc_mu_latency = get_fetch_latency_v0(num_vars, num_mles_in_parallel_pc, bits_per_scalar, available_bw, freq, mle_update=True)
        scale_factor = math.ceil(permcheck_mles/num_mles_in_parallel_pc)
        pc_mu_latency *= scale_factor
        permcheck_mu_peak_bw = available_bw
    
    if opencheck_mu_peak_bw > available_bw:
        oc_mu_latency = get_fetch_latency_v0(num_vars, num_mles_in_parallel_oc, bits_per_scalar, available_bw, freq, mle_update=True)
        scale_factor = math.ceil(opencheck_mles/num_mles_in_parallel_oc)
        oc_mu_latency *= scale_factor
        opencheck_mu_peak_bw = available_bw
    
    latencies = zc_sc_latency, zc_mu_latency, pc_sc_latency, pc_mu_latency, oc_sc_latency, oc_mu_latency
    bandwidths = zerocheck_peak_bw, zerocheck_mu_peak_bw, permcheck_peak_bw, permcheck_mu_peak_bw, opencheck_peak_bw, opencheck_mu_peak_bw
    return latencies, bandwidths

def get_sumcheck_sweep_data_v0(sweep_params, modmul_counts, sumcheck_core_latencies, required_mles, supplemental_latencies, supplemental_data, primitive_stats):

    num_vars_range, sumcheck_pe_unroll_factors, max_mles, mle_update_unroll_factors = sweep_params
    modmuls_per_zerocheck_core, modmuls_per_permcheck_core, modmuls_per_opencheck_core, modmuls_for_unified_sumcheck_core = modmul_counts
    zerocheck_pe_latency, permcheck_pe_latency, opencheck_pe_latency, zerocheck_reduction_latency, permcheck_reduction_latency, opencheck_reduction_latency = sumcheck_core_latencies
    total_required_mles_zerocheck, total_required_mles_permcheck, total_required_mles_opencheck, max_build_mles_in_parallel = required_mles
    initial_build_mle_latency, mle_update_pe_latency, mle_combine_pe_latency, transcript_latency = supplemental_latencies
    bits_per_scalar, available_bw, freq = supplemental_data

    modmul_area, modadd_area, _, reg_area = primitive_stats

    sumcheck_core_stats = dict()

    # TODO: add the number of modadds in the sumcheck module
    for num_vars in num_vars_range:
        for num_sumcheck_core_pes in sumcheck_pe_unroll_factors:
            for num_mles_in_parallel in range(max_mles, 0, -1):
                for pes_per_mle_update in mle_update_unroll_factors:
                    
                    sumcheck_design = (num_vars, num_sumcheck_core_pes, num_mles_in_parallel, pes_per_mle_update)
                    sumcheck_core_stats[sumcheck_design] = dict()

                    num_mles_in_parallel_zc = num_mles_in_parallel
                    if num_mles_in_parallel > total_required_mles_zerocheck:
                        num_mles_in_parallel_zc = total_required_mles_zerocheck

                    num_zerocheck_core_pes = num_sumcheck_core_pes
                    zerocheck_latency, zerocheck_runtime, zc_peak_bw_util, zc_mu_peak_bw_util, zerocheck_modmuls, zc_accumulated_sc_latency, zc_accumulated_mu_latency, zc_accumulated_tc_latency = sumcheck_latency(
                        num_vars,
                        zerocheck_pe_latency,
                        zerocheck_reduction_latency,
                        num_zerocheck_core_pes,
                        modmuls_per_zerocheck_core,
                        mle_update_pe_latency,
                        pes_per_mle_update,
                        modmuls_per_mle_update_pe,
                        transcript_latency,
                        freq,
                        bits_per_scalar
                        , total_required_mles = total_required_mles_zerocheck 
                        , num_mles_in_parallel = num_mles_in_parallel_zc
                    )
                    
                    assert zc_accumulated_sc_latency + zc_accumulated_mu_latency + zc_accumulated_tc_latency == zerocheck_latency

                    num_mles_in_parallel_pc = num_mles_in_parallel
                    if num_mles_in_parallel > total_required_mles_permcheck:
                        num_mles_in_parallel_pc = total_required_mles_permcheck

                    num_permcheck_core_pes = num_sumcheck_core_pes
                    permcheck_latency, permcheck_runtime, pc_peak_bw_util, pc_mu_peak_bw_util, permcheck_modmuls, pc_accumulated_sc_latency, pc_accumulated_mu_latency, pc_accumulated_tc_latency = sumcheck_latency(
                        num_vars,
                        permcheck_pe_latency,
                        permcheck_reduction_latency,
                        num_permcheck_core_pes,
                        modmuls_per_permcheck_core,
                        mle_update_pe_latency,
                        pes_per_mle_update,
                        modmuls_per_mle_update_pe,
                        transcript_latency,
                        freq,
                        bits_per_scalar
                        , total_required_mles = total_required_mles_permcheck 
                        , num_mles_in_parallel = num_mles_in_parallel_pc
                    )
                    assert pc_accumulated_sc_latency + pc_accumulated_mu_latency + pc_accumulated_tc_latency == permcheck_latency

                    num_mles_in_parallel_oc = num_mles_in_parallel
                    if num_mles_in_parallel > total_required_mles_opencheck:
                        num_mles_in_parallel_oc = total_required_mles_opencheck
                    
                    num_opencheck_core_pes = num_sumcheck_core_pes
                    opencheck_latency, opencheck_runtime, oc_peak_bw_util, oc_mu_peak_bw_util, opencheck_modmuls, oc_accumulated_sc_latency, oc_accumulated_mu_latency, oc_accumulated_tc_latency = sumcheck_latency(
                        num_vars,
                        opencheck_pe_latency,
                        opencheck_reduction_latency,
                        num_opencheck_core_pes,
                        modmuls_per_opencheck_core,
                        mle_update_pe_latency,
                        pes_per_mle_update,
                        modmuls_per_mle_update_pe,
                        transcript_latency,
                        freq,
                        bits_per_scalar
                        , total_required_mles = total_required_mles_opencheck 
                        , num_mles_in_parallel = num_mles_in_parallel_oc # one of the MLEs only has 1 element in it. this term is exclusively for mle update
                    )
                    assert oc_accumulated_sc_latency + oc_accumulated_mu_latency + oc_accumulated_tc_latency == opencheck_latency

                    # this is the cost of a single build MLE unit
                    build_mle_cost = BuildMleCostReport(num_vars=num_vars, mod_mul_latency=modmul_latency, mod_add_latency=modadd_latency, num_zerocheck_pes=num_sumcheck_core_pes,
                                            num_mod_mul=-1, num_mod_add=-1, available_bandwidth=-1)
                    build_mle_result = build_mle_cost.cost()
                    
                    # latency calculations
                    warmup_latency_build_mle = build_mle_result['compulsory_cycles']
                    # this will be used for multifunction tree latency
                    single_build_mle_latency = build_mle_result['total_cycles']

                    total_zerocheck_latency = int(warmup_latency_build_mle + zerocheck_latency)
                    total_permcheck_latency = int(warmup_latency_build_mle + permcheck_latency)
                    
                    num_mle_combine_pes = 2*num_opencheck_core_pes
                    warmup_latency_mle_combine = mle_combine_latency(num_vars, mle_combine_pe_latency, num_mle_combine_pes) - (1 << num_vars)/num_opencheck_core_pes
                    opencheck_warmup_latency = max(warmup_latency_build_mle, warmup_latency_mle_combine + initial_build_mle_latency)
                    total_opencheck_latency = int(opencheck_warmup_latency + opencheck_latency)

                    total_transcript_latency = zc_accumulated_tc_latency + pc_accumulated_tc_latency + oc_accumulated_tc_latency 

                    latency_breakdowns = [zc_accumulated_sc_latency, zc_accumulated_mu_latency, pc_accumulated_sc_latency, pc_accumulated_mu_latency, oc_accumulated_sc_latency, oc_accumulated_mu_latency, total_transcript_latency]
                    sumcheck_core_stats[sumcheck_design]['zerocheck_latency'] = total_zerocheck_latency
                    sumcheck_core_stats[sumcheck_design]['permcheck_latency'] = total_permcheck_latency
                    sumcheck_core_stats[sumcheck_design]['opencheck_latency'] = total_opencheck_latency
                    sumcheck_core_stats[sumcheck_design]['latency_breakdowns'] = latency_breakdowns
                    sumcheck_core_stats[sumcheck_design]['total_sumcheck_latency'] = total_zerocheck_latency + total_permcheck_latency + total_opencheck_latency

                    # area calculations
                    build_mle_modmuls = build_mle_result['req_mod_mul_num']*max_build_mles_in_parallel
                    build_mle_modadds = build_mle_result['req_mod_add_num']*max_build_mles_in_parallel
                    build_mle_regs    = build_mle_result['req_mem_reg_num']*max_build_mles_in_parallel
                    # build_mle_mem     = build_mle_result['req_mem_bit']*max_build_mles_in_parallel

                    build_mle_stats = {
                        "build_mle_modmuls" : build_mle_modmuls,
                        "build_mle_modadds" : build_mle_modadds,
                        "build_mle_regs" : build_mle_regs,
                        "single_build_mle_latency": single_build_mle_latency
                    }

                    mle_combine_modmuls = modmuls_per_mle_combine_pe*num_mle_combine_pes
                    
                    # fixed_modmul_cost = build_mle_modmuls + mle_combine_modmuls
                    fixed_modmul_cost = build_mle_modmuls

                    unified_core_modmul_cost = num_sumcheck_core_pes*modmuls_for_unified_sumcheck_core
                    unified_mle_update_modmul_cost = num_mles_in_parallel*pes_per_mle_update*modmuls_per_mle_update_pe

                    min_num_modmuls = unified_core_modmul_cost + fixed_modmul_cost
                    max_num_modmuls = zerocheck_modmuls + permcheck_modmuls + opencheck_modmuls + fixed_modmul_cost

                    min_area_estimate = min_num_modmuls*modmul_area + build_mle_modadds*modadd_area + build_mle_regs*reg_area
                    max_area_estimate = max_num_modmuls*modmul_area + build_mle_modadds*modadd_area + build_mle_regs*reg_area

                    sumcheck_core_stats[sumcheck_design]['min_num_modmuls']   = min_num_modmuls
                    sumcheck_core_stats[sumcheck_design]['max_num_modmuls']   = max_num_modmuls
                    sumcheck_core_stats[sumcheck_design]['min_area_estimate'] = min_area_estimate
                    sumcheck_core_stats[sumcheck_design]['max_area_estimate'] = max_area_estimate
                    # sumcheck_core_stats[sumcheck_design]['register_mem'] = build_mle_mem/BITS_PER_MB
                    
                    # use this updated data
                    sumcheck_core_stats[sumcheck_design]['mle_combine_area'] = mle_combine_modmuls*modmul_area
                    sumcheck_core_stats[sumcheck_design]['build_mle_stats'] = build_mle_stats
                    sumcheck_core_stats[sumcheck_design]['sumcheck_core_area'] = unified_core_modmul_cost*modmul_area
                    sumcheck_core_stats[sumcheck_design]['mle_update_core_area'] = unified_mle_update_modmul_cost*modmul_area
                    
                    # bandwidth data
                    sc_bw_stats = (zc_peak_bw_util, zc_mu_peak_bw_util, pc_peak_bw_util, pc_mu_peak_bw_util, oc_peak_bw_util, oc_mu_peak_bw_util)
                    sc_bw_stats = [round(i, 3) for i in sc_bw_stats]
                    # sumcheck_core_stats[sumcheck_design]['bandwidth_stats'] = sc_bw_stats
                    sumcheck_core_stats[sumcheck_design]['bandwidth_stats'] = \
                        {
                            "zerocheck_peak_bw"            : sc_bw_stats[0],
                            "zerocheck_mle_update_peak_bw" : sc_bw_stats[1],
                            "permcheck_peak_bw"            : sc_bw_stats[2],
                            "permcheck_mle_update_peak_bw" : sc_bw_stats[3],
                            "opencheck_peak_bw"            : sc_bw_stats[4],
                            "opencheck_mle_update_peak_bw" : sc_bw_stats[5]
                        }

                    # with throttling, these are our results. assume build MLE internally stalls itself but pays the same hardware cost
                    other_stats = num_vars, bits_per_scalar, available_bw, freq
                    mle_stats = total_required_mles_zerocheck, total_required_mles_permcheck, total_required_mles_opencheck, num_mles_in_parallel_zc, num_mles_in_parallel_pc, num_mles_in_parallel_oc
                    throttled_latencies, throttled_bandwidths = throttle_sumcheck_v0(latency_breakdowns, sc_bw_stats, mle_stats, other_stats)

                    total_sumcheck_latency_with_throttle = sum(throttled_latencies) + total_transcript_latency + opencheck_warmup_latency + warmup_latency_build_mle*2
                    throttled_latency_breakdown = list(throttled_latencies) + [total_transcript_latency]
                    sumcheck_core_stats[sumcheck_design]['throttled_latencies'] = \
                        {
                            "total_sumcheck_latency": total_sumcheck_latency_with_throttle,
                            "zerocheck_sc_latency": throttled_latencies[0],
                            "zerocheck_mu_latency": throttled_latencies[1],
                            "permcheck_sc_latency": throttled_latencies[2],
                            "permcheck_mu_latency": throttled_latencies[3],
                            "opencheck_sc_latency": throttled_latencies[4],
                            "opencheck_mu_latency": throttled_latencies[5],
                            "throttled_latency_breakdown" : throttled_latency_breakdown
                        }
                    sumcheck_core_stats[sumcheck_design]['throttled_bandwidths'] = \
                        {
                            "zerocheck_peak_bw"            : throttled_bandwidths[0],
                            "zerocheck_mle_update_peak_bw" : throttled_bandwidths[1],
                            "permcheck_peak_bw"            : throttled_bandwidths[2],
                            "permcheck_mle_update_peak_bw" : throttled_bandwidths[3],
                            "opencheck_peak_bw"            : throttled_bandwidths[4],
                            "opencheck_mle_update_peak_bw" : throttled_bandwidths[5]
                        }

    return sumcheck_core_stats


def get_sumcheck_sweep_data_v1(sweep_params, sumcheck_polynomials, sparsity_data):

    num_vars_range, sumcheck_pes_range, eval_engines_range, product_lanes_range, onchip_mle_sizes_range = sweep_params
    zerocheck_polynomial, permcheck_polynomial, opencheck_polynomial = sumcheck_polynomials
    sumcheck_types = ["zerocheck", "permcheck", "opencheck"]

    sumcheck_core_stats = dict()
    for num_vars in num_vars_range:
        sumcheck_core_stats[num_vars] = dict()
        for num_pes in sumcheck_pes_range:
            for num_eval_engines in eval_engines_range:

                assert num_eval_engines > 1

                for num_product_lanes in product_lanes_range:
                    assert num_product_lanes > 2

                    for onchip_mle_size in onchip_mle_sizes_range:
                        sumcheck_hardware_params = num_pes, num_eval_engines, num_product_lanes, mle_update_latency, extensions_latency, modmul_latency, onchip_mle_size

                        sumcheck_hardware_config = num_pes, num_eval_engines, num_product_lanes, onchip_mle_size
                        sumcheck_core_stats[num_vars][sumcheck_hardware_config] = dict()
                        
                        mul_tree_area_stats_dict = dict()
                        for pl_depth in range(1, max_pl_offset):
                            
                            trees_modules = shared_tree_cost(setup_config_dict = {
                                "basic": {
                                    "mod_mul_latency": modmul_latency,
                                    "mod_add_latency": modadd_latency,
                                },
                                "multiply_lane_tree": {
                                    "num_input_entries_per_cycle": num_eval_engines + pl_depth,  # should be at least num_eval_engines
                                    "num_lanes_per_sc_pe": num_product_lanes,
                                    "total_sc_pe": num_pes
                                },
                            })
                            hardware_size = trees_modules.get_hardware_cost_cost()
                            mul_tree_area_stats_dict[pl_depth] = hardware_size

                        sumcheck_core_stats[num_vars][sumcheck_hardware_config]['forest_area'] = mul_tree_area_stats_dict
                        sumcheck_core_stats[num_vars][sumcheck_hardware_config]['rr_ctrl_area'] = rr_ctrl_area*num_pes
                        sumcheck_core_stats[num_vars][sumcheck_hardware_config]['sleep_ctrl_area'] = sleep_ctrl_area*num_pes*2 # double factor for all control logic
                        sumcheck_core_stats[num_vars][sumcheck_hardware_config]['eval_engine_area'] = 2*modmul_area*num_eval_engines*num_pes
                        sumcheck_core_stats[num_vars][sumcheck_hardware_config]['delay_buffer_area'] = per_pe_delay_buffer_count*num_pes

                        for sumcheck_type, sumcheck_polynomial in zip(sumcheck_types, [zerocheck_polynomial, permcheck_polynomial, opencheck_polynomial]):
                            total_latency, round_latencies, _ = performance_model(num_vars, sumcheck_polynomial, sumcheck_type, sumcheck_hardware_params, sparsity_data, supplemental_data)
                            sumcheck_core_stats[num_vars][sumcheck_hardware_config][sumcheck_type] = total_latency

    return sumcheck_core_stats


if __name__ == "__main__":

    from params import *
    from sumcheck_models_v0 import *
    from build_mle import *

    available_bw = 2048 
    target_num_vars_range = [20]
    
    sumcheck_pe_unroll_factors = [2**i for i in range(1,2)]  # 2^1 to 2^3
    mle_update_unroll_factors = [2**i for i in range(2,3)]  # 2^1 to 2^3

    print(sumcheck_pe_unroll_factors)
    print(mle_update_unroll_factors)

    modmul_counts = modmuls_per_zerocheck_core, modmuls_per_permcheck_core, modmuls_per_opencheck_core, modmuls_for_unified_sumcheck_core
    sumcheck_core_latencies = zerocheck_pe_latency, permcheck_pe_latency, opencheck_pe_latency, zerocheck_reduction_latency, permcheck_reduction_latency, opencheck_reduction_latency
    required_mles = total_required_mles_zerocheck, total_required_mles_permcheck, total_required_mles_opencheck, max_build_mles_in_parallel
    supplemental_latencies = initial_build_mle_latency, mle_update_pe_latency, mle_combine_pe_latency, transcript_latency
    supplemental_data = bits_per_scalar, available_bw, freq

    sumcheck_sweep_params = target_num_vars_range, sumcheck_pe_unroll_factors, max_mles, mle_update_unroll_factors

    sumcheck_core_stats = get_sumcheck_sweep_data_v0(sumcheck_sweep_params, modmul_counts, sumcheck_core_latencies, required_mles, supplemental_latencies, supplemental_data, primitive_stats)

    for design, data in sumcheck_core_stats.items():
        print(design)
        for k, v in data.items():
            print(f"{k}: {v}")
        exit()
