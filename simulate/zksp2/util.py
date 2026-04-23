import math
import numpy as np
from .params import *

# Faster than is_pareto_efficient_simple, but less readable.
def is_pareto_efficient(costs, return_mask = True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    # print(n_points)
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient

# fixed for vanilla
def input_mle_size_vanilla(bitwidth, num_vars):

    scale_factor = bitwidth/BITS_PER_MB

    packed_enables = 4*(1 << num_vars)/BITS_PER_MB
    address_translation_unit = (1 << num_vars)*(1 + num_vars)/BITS_PER_MB
    ones_zeros_table = round(0.9*(1 << num_vars))/BITS_PER_MB
    full_width_table = round(0.1*(1 << num_vars)*bitwidth)/BITS_PER_MB
    permutation_table = ((1 << num_vars) * (num_vars + 2))/BITS_PER_MB

    total_input_mle_storage = packed_enables + 4*(address_translation_unit + ones_zeros_table + full_width_table) + 3*permutation_table

    return total_input_mle_storage

def bitsPerCycle_to_GiBPerS(bitsPerCycle_rate, CLK_FREQ):
    return bitsPerCycle_rate * CLK_FREQ / 8 / pow(2, 30)
    
def calc_bw(bits_per_element, elements_per_cycle, freq):
    bits_per_cycle = bits_per_element*elements_per_cycle
    GB_per_cycle = bits_per_cycle/BITS_PER_GB
    GB_per_second = GB_per_cycle*freq
    # return GB_per_second
    return np.round(GB_per_second, 3)

# bandwidth is in GB/s
def calc_rate(bits_per_element, bandwidth, freq):
    elements_per_cycle = bandwidth*BITS_PER_GB/(freq*bits_per_element)
    return elements_per_cycle

def get_extrap_traces(target_num_vars, baseline_num_vars, config, params):
    
    assert target_num_vars > baseline_num_vars

    current_ws, current_ppw, current_ocw, current_qd, current_ii = config
    
    base_dir, ADD_POLICY, padd_latency = params

    file_dir = base_dir + f"SYNTHETIC_{baseline_num_vars}/{ADD_POLICY}/{padd_latency}_cycles/"
    file_path = os.path.join(file_dir, f"ws{current_ws}_ppw{current_ppw}_ocw{current_ocw}_qd{current_qd}_ii{current_ii}.pkl")

    # Check if the file exists
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            block_latency_array = np.atleast_1d(np.squeeze(data['block_latency_array'])).tolist()
            elements_in_block_array = np.atleast_1d(np.squeeze(data['elements_in_block_array'])).tolist()
            window_reduce_cycles = np.squeeze(data['window_reduction_cycles_matrix'])
    else:
        print(file_path)
        exit(f"{file_path} not found even tho it should be there")

    block_latency_array = block_latency_array*(1 << (target_num_vars - baseline_num_vars))
    elements_in_block_array = elements_in_block_array*(1 << (target_num_vars - baseline_num_vars))
    sort_cycles = sum(block_latency_array)

    return block_latency_array, elements_in_block_array, sort_cycles, window_reduce_cycles

def get_extrap_traces_sparse(target_len, baseline_len, config, params):
    
    assert target_len > baseline_len

    current_ws, current_ppw, current_ocw, current_qd, current_ii = config
    
    base_dir, ADD_POLICY, padd_latency = params

    file_dir = base_dir + f"SYNTHETIC_{baseline_len}/{ADD_POLICY}/{padd_latency}_cycles/"
    file_path = os.path.join(file_dir, f"ws{current_ws}_ppw{current_ppw}_ocw{current_ocw}_qd{current_qd}_ii{current_ii}.pkl")

    # Check if the file exists
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            block_latency_array = np.atleast_1d(np.squeeze(data['block_latency_array'])).tolist()
            elements_in_block_array = np.atleast_1d(np.squeeze(data['elements_in_block_array'])).tolist()
            window_reduce_cycles = np.squeeze(data['window_reduction_cycles_matrix'])
    else:
        print(file_path)
        exit(f"{file_path} not found even tho it should be there")

    block_latency_array = block_latency_array*math.ceil(target_len/baseline_len)
    elements_in_block_array = elements_in_block_array*math.ceil(target_len/baseline_len)
    sort_cycles = sum(block_latency_array)

    return block_latency_array, elements_in_block_array, sort_cycles, window_reduce_cycles

def get_msm_load_overhead(num_pes, num_words, fill_rate, rate_match=True):
    
    if rate_match:
        # how fast are we reading values from on-chip memory
        drain_rate = 2*num_pes

        lower_rate = min(drain_rate, fill_rate)
        backpressure_input = fill_rate > drain_rate

        if backpressure_input:
            latency_overhead = 0
        else:
            latency_overhead = int(math.ceil(num_words * (drain_rate - fill_rate) / (drain_rate*fill_rate)))
    
    # just get the number of fill_cycles
    else:
        max_fill_rate = 2*num_pes
        real_fill_rate = min(max_fill_rate, fill_rate)
        latency_overhead = int(math.ceil(num_words / real_fill_rate))
        backpressure_input = False
        lower_rate = real_fill_rate

    # assert latency_overhead <= num_words

    return latency_overhead, backpressure_input, lower_rate

def construct_list(a, b, c, d, group_size):
    # Create the list of a's and b's
    list_b = [b] * d
    list_a = [a] * c
    
    # Combine b's and a's, with b's at the beginning
    combined_list = list_b + list_a
    
    # Initialize the grouped list
    grouped_list = []
    
    # Start grouping from the tail
    while combined_list:
        # Take up to 'group_size' elements from the tail
        group = combined_list[-group_size:]
        # Prepend the group to the grouped list
        grouped_list.insert(0, group)
        # Remove these elements from the combined list
        combined_list = combined_list[:-group_size]
    
    return grouped_list

def find_values(total_bits, primary_window_size, secondary_window_size):
    valid_combinations = []
    
    if total_bits % primary_window_size == 0:
        return (total_bits // primary_window_size, 0)

    for a in range(1, total_bits):  # (chat-gpt did this) Upper bound 256 as 5*256 = 1280 which is already much greater than 256.
        b = (total_bits - primary_window_size * a) / secondary_window_size
        if b.is_integer() and a > b and b > 0:
            valid_combinations.append((int(a), int(b)))
            
    return valid_combinations[-1]

def calculate_total_bucket_reduce_latency_opt(num_scalar_bits, ws, ocw, optimal_latency_dict):
    a = ws
    b = ws - 1
    c, d = find_values(num_scalar_bits, a, b)
    window_groups = construct_list(a, b, c, d, ocw)
    total_bucket_reduce_latency = 0
    for window_group in window_groups:
        window_sizes = set(window_group)
        group_bucket_latencies = []
        for window_size in window_sizes:
            group_bucket_latencies.append(optimal_latency_dict[window_size][0])
        total_bucket_reduce_latency += max(group_bucket_latencies)
    return total_bucket_reduce_latency

# this should yield roughly (num_words/num_padds)
def ones_reduction_latency(num_words, padd_latency, num_padds, queue_depth=2):

    compute_latency = padd_latency + 2*queue_depth
    # immediate_start_rounds = math.floor(math.log2(num_words/compute_latency))
    immediate_start_rounds = math.floor(math.log2(num_words/(compute_latency*num_padds)))

    # latency for stages where the effective number of inputs (half the MLE size since we read 2 elements at 
    # at time) is greater than compute depth. can start reading in next round of MLEs once first set of inputs
    total_cycles = 0
    for i in range(1, immediate_start_rounds + 1):
        total_cycles += math.ceil(num_words / ((1 << i) * num_padds))

    effective_num_rounds = math.ceil(math.log2(num_words / num_padds))

    # technically there's a few more cycles, but it is negligible (~log2(num_words))
    total_cycles += (effective_num_rounds - immediate_start_rounds + math.ceil(math.log2(num_padds)))*compute_latency
    return total_cycles

def get_full_stats(trace_data, supplemental_data, debug=False):
    
    elements_in_block_array, block_latency_array, adjusted_latency_array = trace_data
    ocw, bits_per_element, freq, available_bw = supplemental_data

    first_block_size = elements_in_block_array[0]

    available_rate = calc_rate(bits_per_element, available_bw, freq)   
    points_loading_latency, _, lower_rate = get_msm_load_overhead(ocw, first_block_size, available_rate, rate_match=False)

    block_latency_array = [points_loading_latency] + block_latency_array

    total_latency = int(math.ceil(sum(block_latency_array)))
    loading_bw = calc_bw(bits_per_element, lower_rate, freq)
    main_compute_latency = sum(adjusted_latency_array)
    transfer_less_compute_latency = block_latency_array[-1]

    latency_stats = [points_loading_latency, main_compute_latency, transfer_less_compute_latency]

    return total_latency, loading_bw, latency_stats, first_block_size, lower_rate

# average bandwidth might not be the best metric to use here?
def get_main_compute_stats(trace_data, supplemental_data, debug=False):
    
    elements_in_block_array, block_latency_array = trace_data
    bits_per_element, freq, available_bw = supplemental_data
    
    assert len(elements_in_block_array) == len(block_latency_array)
    
    # in this situation, we have already loaded all data, so no additional loading is required
    # by design, we are setting bandwidth during compute phase to 0.
    # the last block duration is excluded from "main_compute" because that also includes
    # latency of reductions
    if len(elements_in_block_array) == 1:
        return [0], 0, 0

    adjusted_elements_array = np.array(elements_in_block_array[1:])
    adjusted_latency_array = np.array(block_latency_array[:-1])
    elements_per_cycle = adjusted_elements_array/adjusted_latency_array

    # bits per element here should be 2D points
    GB_per_second = calc_bw(bits_per_element, elements_per_cycle, freq)
    
    if debug:
        print(f"adjusted_elements_array: {adjusted_elements_array.tolist()}")
        print(f"adjusted_latency_array: {adjusted_latency_array.tolist()}")
        print(f"elements_per_cycle: {[round(i, 3) for i in elements_per_cycle.tolist()]}")
        print(f"GB_per_second: {GB_per_second.tolist()}")
        print()

    # increase the latency if bw exceeded
    if np.any(GB_per_second > available_bw):
        mask = GB_per_second > available_bw
        
        adjusted_latency_array[mask] = np.ceil(adjusted_elements_array[mask]*bits_per_element/BITS_PER_GB/available_bw*freq)
        rate_limited_entries = adjusted_elements_array[mask]/adjusted_latency_array[mask]
        rate_limited_bw = calc_bw(bits_per_element, rate_limited_entries, freq)
        within_1_percent = np.all(np.abs((rate_limited_bw - available_bw) / available_bw) < 0.01)
        if within_1_percent:
            GB_per_second[mask] = available_bw
        else:
            exit("something fishy here")
        block_latency_array[:-1] = adjusted_latency_array

    # we have adjusted bandwidth and adjusted latency. just extract peak and average bandwidth from this

    avg_bw = np.sum(GB_per_second*adjusted_latency_array) / np.sum(adjusted_latency_array)
    peak_bw = np.max(GB_per_second)

    return adjusted_latency_array, avg_bw, peak_bw

def construct_ones_trace(ppw, ocw, num_ones, padd_latency): 
    points_on_chip = ppw*ocw
    num_groups = math.ceil(num_ones/points_on_chip)

    final_reduce = num_groups > 1
    if num_ones % points_on_chip == 0:
        elements_in_block_array = [points_on_chip]*num_groups
        num_full_groups = num_groups
        num_rem_groups = 0
    else:
        rem = num_ones % points_on_chip
        elements_in_block_array = [points_on_chip]*(num_groups-1) + [rem]

        # if last block has enough space for the results of all full groups
        # just add them to the workload. they arent going to be fetched from
        # off chip tho so no need to change elements_in_block_array
        if rem + num_groups-1 <= points_on_chip:
            rem += num_groups-1
            final_reduce = False

        num_full_groups = num_groups - 1
        num_rem_groups = 1

    full_group_latency = math.ceil(ones_reduction_latency(points_on_chip, padd_latency, ocw, queue_depth=0))

    block_latency_array = [full_group_latency]*num_full_groups
    if num_rem_groups > 0:
        rem_group_latency = math.ceil(ones_reduction_latency(rem, padd_latency, ocw, queue_depth=0))
        block_latency_array += [rem_group_latency]

    # combine results of all groups. we can reasonably assume no need to do off-chip transfers
    # because the size of the on-chip scratchpad memory area (num_buckets*num_windows) exceeds
    # the number of groups (.45*num_points)/ppw (setting ocw = 1, num_windows = scalar_size/window_size)
    if final_reduce:
        final_reduce_latency = ones_reduction_latency(num_groups, padd_latency, ocw, queue_depth=0)
        block_latency_array[-1] += final_reduce_latency

    return elements_in_block_array, block_latency_array
