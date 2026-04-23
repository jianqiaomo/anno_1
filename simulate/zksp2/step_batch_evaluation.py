from .reverse_binary_tree import MleEvalCostReport
import heapq


# Intelligent Scheduling: Assign longest task first
def schedule_tasks(task_cycles, num_processors):
    processors = [(0, i) for i in range(num_processors)]  # (next available time, processor id)
    heapq.heapify(processors)

    # Sort tasks by descending order of processing time (longest task first)
    # task_cycles_sorted = sorted(task_cycles, reverse=True)

    # for task_cycle in task_cycles_sorted:
    for task_cycle in task_cycles:
        available_time, processor_id = heapq.heappop(processors)
        heapq.heappush(processors, (available_time + task_cycle, processor_id))

    return max([time for time, _ in processors])


def step_batch_evaluation_model(num_vars,
                                mle_update_latency,
                                mod_add_latency,
                                num_input_rows_per_cycle,
                                cycle_poly_open,
                                optmize_p2p3p4=False,
                                num_mle_update=-1,
                                num_mod_add=-1):
    """
    grand model for batch evaluation step

    :param num_vars: (μ) e.g., 20 for 2**20
    :param mle_update_latency: (cycle) latency of mle update (2 mod mult parallel, and 1 mod add)
    :param mod_add_latency: (cycle) total latency of pipelined mod add (for warm up cycles
    :param num_input_rows_per_cycle: num of inputs fetching per cycle, e.g., 2 for 2 𝜙 and 𝜋 per cycle
    :param cycle_poly_open: run time of polynomial opening
    :param optmize_p2p3p4: consider 0s (optimize) of p2, p4, etc.
    :param num_mle_update: (-1 for unlimited mle update to match fetching) num of mle update PEs
    :param num_mod_add: (-1 for unlimited mod add to match fetching) num of mod add PEs
    :return: {}
    :rtype: dict
    """
    if optmize_p2p3p4:
        p2_num_vars = num_vars - 1
        p3_num_vars = num_vars - 1
        p4_num_vars = 0  # input[2^(μ-1)]
        p6_num_vars = num_vars
    else:
        p2_num_vars = num_vars
        p3_num_vars = num_vars
        p4_num_vars = num_vars
        p6_num_vars = num_vars


    # 𝜙, 𝜋, 𝜎123, w123 : p1
    p1_mle = {
        f"{i}_p1": MleEvalCostReport(num_vars=num_vars,
                                     mle_update_latency=mle_update_latency,
                                     mod_add_latency=mod_add_latency,
                                     num_previous_result=num_input_rows_per_cycle,
                                     num_mle_update=num_mle_update,
                                     num_mod_add=num_mod_add,
                                     available_bandwidth=-1
                                     ) for i in
        ['phi', 'pi'] + [f'sigma{j}' for j in range(1, 4)] + [f'w{j}' for j in range(1, 4)]
    }
    # 𝜙, 𝜋 : p2
    p2_mle = {
        f"{i}_p2": MleEvalCostReport(num_vars=p2_num_vars,
                                     mle_update_latency=mle_update_latency,
                                     mod_add_latency=mod_add_latency,
                                     num_previous_result=num_input_rows_per_cycle,
                                     num_mle_update=num_mle_update,
                                     num_mod_add=num_mod_add,
                                     available_bandwidth=-1
                                     ) for i in ['phi', 'pi']
    }
    # 𝜙, 𝜋 : p3
    p3_mle = {
        f"{i}_p3": MleEvalCostReport(num_vars=p3_num_vars,
                                     mle_update_latency=mle_update_latency,
                                     mod_add_latency=mod_add_latency,
                                     num_previous_result=num_input_rows_per_cycle,
                                     num_mle_update=num_mle_update,
                                     num_mod_add=num_mod_add,
                                     available_bandwidth=-1
                                     ) for i in ['phi', 'pi']
    }
    # 𝜋 : p4
    p4_mle = {
        f"{i}_p4": MleEvalCostReport(num_vars=p4_num_vars,
                                     mle_update_latency=mle_update_latency,
                                     mod_add_latency=mod_add_latency,
                                     num_previous_result=num_input_rows_per_cycle,
                                     num_mle_update=num_mle_update,
                                     num_mod_add=num_mod_add,
                                     available_bandwidth=-1
                                     ) for i in ['pi']
    }
    # w123, qi : p5
    p5_mle = {
        f"{i}_p5": MleEvalCostReport(num_vars=num_vars,
                                     mle_update_latency=mle_update_latency,
                                     mod_add_latency=mod_add_latency,
                                     num_previous_result=num_input_rows_per_cycle,
                                     num_mle_update=num_mle_update,
                                     num_mod_add=num_mod_add,
                                     available_bandwidth=-1
                                     ) for i in
        [f'w{j}' for j in range(1, 4)] + [f'q{j}' for j in ['L', 'R', 'M', 'O', 'C']]
    }
    # w1 : p6
    p6_mle = {
        f"{i}_p6": MleEvalCostReport(num_vars=p6_num_vars,
                                     mle_update_latency=mle_update_latency,
                                     mod_add_latency=mod_add_latency,
                                     num_previous_result=num_input_rows_per_cycle,
                                     num_mle_update=num_mle_update,
                                     num_mod_add=num_mod_add,
                                     available_bandwidth=-1
                                     ) for i in ['w1']
    }

    phi_pi_mle = [p1_mle['phi_p1'], p2_mle['phi_p2'], p3_mle['phi_p3'],
                  p1_mle['pi_p1'], p2_mle['pi_p2'], p3_mle['pi_p3'], p4_mle['pi_p4']]
    other_mle = [p1_mle['sigma1_p1'], p1_mle['sigma2_p1'], p1_mle['sigma3_p1'],
                    p1_mle['w1_p1'], p1_mle['w2_p1'], p1_mle['w3_p1'],
                    p5_mle['w1_p5'], p5_mle['w2_p5'], p5_mle['w3_p5'],
                    p5_mle['qL_p5'], p5_mle['qR_p5'], p5_mle['qM_p5'], p5_mle['qO_p5'], p5_mle['qC_p5'],
                    p6_mle['w1_p6']]

    task_cycles = []
    for pi_mles in phi_pi_mle + other_mle:
        task_cycles.append(pi_mles.cost()['total_cycles'])

    # at least: 4 rounds with 7 PEs
    min_runtime = -1
    processors_needed = -1
    for processors_needed in range(6, 23):
        min_runtime = schedule_tasks(task_cycles, processors_needed)
        if min_runtime <= cycle_poly_open:
            break

    if min_runtime == -1 or processors_needed == -1:
        raise Exception("No solution found to schedule batch evaluation step.")

    cost = {
        "total_mle_eval_pes": processors_needed,
        "total_cycles": min_runtime,
        # "compulsory_cycles": min_runtime,
        "req_mod_mul_num": p1_mle['phi_p1'].cost()['req_mod_mul_num'] * processors_needed,
        "req_mod_add_num": p1_mle['phi_p1'].cost()['req_mod_add_num'] * processors_needed,
        "req_mem_bit": p1_mle['phi_p1'].cost()['req_mem_bit'] * processors_needed,
        "req_mem_reg_num": p1_mle['phi_p1'].cost()['req_mem_reg_num'] * processors_needed,
        "bandwidth_bperc": sum([pe_i.cost()['bandwidth_bperc'] for pe_i in [p1_mle['phi_p1'], p1_mle['pi_p1']]]),
    }

    return cost


if __name__ == "__main__":

    num_vars = 20
    mod_add_latency = 1
    mle_update_latency = 10 + mod_add_latency  # 2 mod mult parallel, and 1 mod add
    num_input_rows_per_cycle = 4  # 2 elements of 𝜙 and 𝜋 per cycle
    cycle_poly_open = 46140000  # target: to match the latency of polynomial opening
    num_mod_add = 1

    for optmize_p2p3p4 in [False, True]:
        print(f"Optmize p2, p3, p4: {optmize_p2p3p4}")
        batch_evaluation_cost = step_batch_evaluation_model(num_vars=num_vars,
                                                            mod_add_latency=mod_add_latency,
                                                            mle_update_latency=mle_update_latency,
                                                            num_input_rows_per_cycle=num_input_rows_per_cycle,
                                                            cycle_poly_open=cycle_poly_open,
                                                            optmize_p2p3p4=optmize_p2p3p4,
                                                            num_mle_update=-1,
                                                            num_mod_add=num_mod_add
                                                            )
        print(batch_evaluation_cost)

    a = MleEvalCostReport(num_vars=num_vars,
                                     mle_update_latency=mle_update_latency,
                                     mod_add_latency=mod_add_latency,
                                     num_previous_result=num_input_rows_per_cycle,
                                     num_mle_update=-1,
                                     num_mod_add=-1,
                                     available_bandwidth=-1
                                     )

    print(a.cost())
