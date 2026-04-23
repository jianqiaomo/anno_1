from .build_mle import *
from .params import *
from .reverse_binary_tree import shared_tree_cost
from .sumcheck_models import *

def get_sumcheck_sweep_data_v1(sweep_params, sumcheck_polynomials, sparsity_data, supplemental_data):

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
                            mul_tree_area_stats_dict['module'] = trees_modules

                        eval_engine_modmuls = 2*num_eval_engines

                        sumcheck_core_stats[num_vars][sumcheck_hardware_config]['forest_stats'] = mul_tree_area_stats_dict
                        sumcheck_core_stats[num_vars][sumcheck_hardware_config]['rr_ctrl_area_mm2'] = rr_ctrl_area*num_pes
                        sumcheck_core_stats[num_vars][sumcheck_hardware_config]['sleep_ctrl_area_mm2'] = sleep_ctrl_area*num_pes*2 # double factor for all control logic
                        sumcheck_core_stats[num_vars][sumcheck_hardware_config]['eval_engine_modmuls'] = eval_engine_modmuls*num_pes
                        sumcheck_core_stats[num_vars][sumcheck_hardware_config]['eval_engine_area_mm2'] = eval_engine_modmuls*modmul_area*num_pes
                        sumcheck_core_stats[num_vars][sumcheck_hardware_config]['num_delay_buffers'] = per_pe_delay_buffer_count*num_pes

                        # print(f"num_vars: {num_vars}, num_pes: {num_pes}, num_eval_engines: {num_eval_engines}, num_product_lanes: {num_product_lanes}, onchip_mle_size: {onchip_mle_size}")
                        for sumcheck_type, sumcheck_polynomial in zip(sumcheck_types, [zerocheck_polynomial, permcheck_polynomial, opencheck_polynomial]):
                            # print(f"  sumcheck_type: {sumcheck_type}, sumcheck_polynomial: {sumcheck_polynomial}")
                            # print(f"  sparsity_data: {sparsity_data}")
                            total_latency, round_latencies, _ = performance_model(num_vars, sumcheck_polynomial, sumcheck_type, sumcheck_hardware_params, sparsity_data, supplemental_data)
                            sumcheck_core_stats[num_vars][sumcheck_hardware_config][sumcheck_type] = total_latency

    return sumcheck_core_stats

