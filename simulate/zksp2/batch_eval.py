from .step_batch_evaluation import *
from .reverse_binary_tree import *

def get_batch_eval_data(sweep_params, relevant_latencies):

    num_vars_range, sumcheck_pe_unroll_factors = sweep_params
    modadd_latency, mle_update_pe_latency = relevant_latencies

    batch_eval_data = dict()
    cycle_poly_open = 1000000000000000 # assume batch eval is not on critical path
    num_mod_add = -1
    for num_vars in num_vars_range:
        batch_eval_data[num_vars] = dict()
        for num_sumcheck_core_pes in sumcheck_pe_unroll_factors:
            num_input_rows_per_cycle = 2*num_sumcheck_core_pes  # 2 elements of 𝜙 and 𝜋 per cycle

            batch_evaluation_cost = step_batch_evaluation_model(num_vars=num_vars,
                                                                mod_add_latency=modadd_latency,
                                                                mle_update_latency=mle_update_pe_latency,
                                                                num_input_rows_per_cycle=num_input_rows_per_cycle,
                                                                cycle_poly_open=cycle_poly_open,
                                                                optmize_p2p3p4=True,
                                                                num_mle_update=-1,
                                                                num_mod_add=num_mod_add
                                                                )
            
            a = MleEvalCostReport(num_vars=num_vars,
                                     mle_update_latency=mle_update_pe_latency,
                                     mod_add_latency=modadd_latency,
                                     num_previous_result=num_input_rows_per_cycle,
                                     num_mle_update=-1,
                                     num_mod_add=-1,
                                     available_bandwidth=-1
                                     )
            single_eval_cost = a.cost()
            batch_eval_data[num_vars][num_sumcheck_core_pes] = (batch_evaluation_cost, single_eval_cost)

    return batch_eval_data


def get_batch_eval_data_v1(sweep_params, relevant_latencies):
    num_vars_range, num_pes_range, product_lanes_range, eval_engines_range, pl_offset_range, [gate_type] = sweep_params
    modmul_latency, modadd_latency = relevant_latencies

    batch_eval_data = dict()
    # cycle_poly_open = 1000000000000000  # assume batch eval is not on critical path
    # num_mod_add = -1
    for num_vars in num_vars_range:
        batch_eval_data[num_vars] = dict()
        for num_pes in num_pes_range:
            batch_eval_data[num_vars][num_pes] = dict()
            for num_product_lanes in product_lanes_range:
                batch_eval_data[num_vars][num_pes][num_product_lanes] = dict()
                for num_eval_engines in eval_engines_range:
                    batch_eval_data[num_vars][num_pes][num_product_lanes][num_eval_engines] = dict()
                    for pl_depth in pl_offset_range:

                        trees_modules = shared_tree_cost(setup_config_dict={
                            "basic": {
                                "mod_mul_latency": modmul_latency,
                                "mod_add_latency": modadd_latency,
                            },
                            "multiply_lane_tree": {
                                "num_input_entries_per_cycle": num_eval_engines + pl_depth,
                                # should be at least num_eval_engines
                                "num_lanes_per_sc_pe": num_product_lanes,
                                "total_sc_pe": num_pes
                            },
                        })

                        original_num_vars_list, num_vars_offset_by_p_sparse = [], []
                        if gate_type == "vanilla":
                            original_num_vars_list = [num_vars] * (22 - 1)  # batch open for 21
                            # p1 * 8, p2 * 2, p3 * 2, skip p4 * 1, p5 * 8, p6 * 1,
                            num_vars_offset_by_p_sparse = [0] * 8 + [-1] * 2 + [-1] * 2 + [0] * 8 + [0] * 1
                        elif gate_type == "jellyfish":
                            original_num_vars_list = [num_vars] * (36 - 1)  # batch open for 35
                            # p1 * 12, p2 * 2, p3 * 2, skip p4 * 1, p5 * 18, p6 * 1,
                            num_vars_offset_by_p_sparse = [0] * 12 + [-1] * 2 + [-1] * 2 + [0] * 18 + [0] * 1
                        elif gate_type == "custom_vanilla":
                            original_num_vars_list = [num_vars] * (18 - 1)  # batch open for 17 (skip p4)
                            # p1 * 6, p2 * 2, p3 * 2, skip p4 * 1, p5 * 6, p6 * 1,
                            num_vars_offset_by_p_sparse = [0] * 6 + [-1] * 2 + [-1] * 2 + [0] * 6 + [0] * 1
                        else:
                            raise ValueError(f"batch_eval.py: Unknown gate type: {gate_type}")
                        assert len(num_vars_offset_by_p_sparse) == len(original_num_vars_list)
                        num_vars_offset_list = [num_vars + offset for num_vars, offset in
                                                zip(original_num_vars_list, num_vars_offset_by_p_sparse)]

                        batch_eval_latency = trees_modules.get_mle_batch_eval_cost(
                            num_vars_list=num_vars_offset_list)

                        batch_eval_data[num_vars][num_pes][num_product_lanes][num_eval_engines][
                            pl_depth] = {
                            "tree_module": trees_modules,
                            "batch_eval_latency_each": batch_eval_latency,
                            "batch_eval_latency_max": max(batch_eval_latency),
                        }

    return batch_eval_data


if __name__ == "__main__":
    from params import *
    target_num_vars_range = [20]
    sumcheck_pe_unroll_factors = [1]
    batch_eval_sweep_params = target_num_vars_range, sumcheck_pe_unroll_factors
    batch_eval_relevant_latencies = modadd_latency, mle_update_pe_latency
    batch_eval_data = get_batch_eval_data(batch_eval_sweep_params, batch_eval_relevant_latencies)
    print(batch_eval_data)

    batch_eval_sweep_params = [20], [2], range(2, 5), range(2, 6), range(1, 9), ["Vanilla", "Jellyfish"]
    batch_eval_relevant_latencies = modmul_latency, modadd_latency
    batcheval_data_GateType_NumVars_NumPes_NumProductLanes_NumEvalEngines_PlDepth = get_batch_eval_data_v1(batch_eval_sweep_params, batch_eval_relevant_latencies)
    print(batcheval_data_GateType_NumVars_NumPes_NumProductLanes_NumEvalEngines_PlDepth)