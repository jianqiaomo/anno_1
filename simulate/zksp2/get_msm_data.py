
from .util import *
import itertools
import numpy as np

from .params import *

def get_msm_area_estimate(ws, ppw, ocw, qd, bits_per_scalar, bits_per_point, padd_area, num_cores=1):

    num_buckets = (2**ws - 1)

    # each PE's SNP memory needs to be 2 banks with separate RW ports
    # double buffer to reduce on bandwidth. Also, we use bitwidth of points
    # since in sparse mode we use X, Y, Z banks to store partial results and in
    # dense mode we use X, Y to store points and Z stores the scalars
    snp_memory_per_pe_mb = (bits_per_point*ppw/BITS_PER_MB)*2
    snp_memory_mb = snp_memory_per_pe_mb*ocw

    # these can just be registers. double buffer so that writes to scratchpad are masked
    bucket_sum_regs_per_pe_mb = (num_buckets*bits_per_point/BITS_PER_MB)*2
    bucket_sum_regs_mb = bucket_sum_regs_per_pe_mb*ocw
    
    # these need to be true dual-ported memories
    bucket_addr_queues_per_pe_mb = num_buckets*qd*np.log2(ppw)/BITS_PER_MB
    bucket_addr_queues_mb = bucket_addr_queues_per_pe_mb*ocw

    # can just be normal single-ported memory 
    bucket_scratchpad_mb = num_buckets*np.ceil(bits_per_scalar/ws)*bits_per_point/BITS_PER_MB

    padd_mm2 = padd_area*ocw*num_cores

    mem_area_snp_mm2 = snp_memory_mb*MB_CONVERSION_FACTOR
    mem_area_bregs_mm2 = bucket_sum_regs_mb*MB_CONVERSION_FACTOR
    mem_area_bqs_mm2 = bucket_addr_queues_mb*MB_CONVERSION_FACTOR
    mem_area_spad_mm2 = bucket_scratchpad_mb*MB_CONVERSION_FACTOR

    total_mem_area_mm2 = mem_area_snp_mm2 + mem_area_bqs_mm2 + mem_area_spad_mm2 + mem_area_bregs_mm2
    total_mem_size_mb = snp_memory_mb + bucket_sum_regs_mb + bucket_addr_queues_mb + bucket_scratchpad_mb

    memory_area_stats = [total_mem_area_mm2, mem_area_snp_mm2, mem_area_bregs_mm2, mem_area_bqs_mm2, mem_area_spad_mm2]
    memory_area_stats = [round(i*num_cores, 3) for i in memory_area_stats]

    memory_size_stats = [total_mem_size_mb, snp_memory_mb, bucket_sum_regs_mb, bucket_addr_queues_mb, bucket_scratchpad_mb]
    memory_size_stats = [round(i*num_cores, 3) for i in memory_size_stats]

    return memory_size_stats, memory_area_stats, padd_mm2


def stitch_msms(msm_trace_1, msm_trace_2, debug=False):

    latency_1, latency_stats_1, bw_stats_1, first_block_size_1 = msm_trace_1
    latency_2, latency_stats_2, bw_stats_2, first_block_size_2 = msm_trace_2

    assert type(latency_1) == int and len(latency_stats_1) == 3 and len(bw_stats_1) == 3
    assert type(latency_2) == int and len(latency_stats_2) == 3 and len(bw_stats_2) == 3

    loading_latency_1, compute_latency_1, last_latency_1 = latency_stats_1
    loading_latency_2, compute_latency_2, last_latency_2 = latency_stats_2

    loading_bw_1, avg_bw_1, peak_bw_1 = bw_stats_1
    loading_bw_2, avg_bw_2, peak_bw_2 = bw_stats_2

    transition_latency = max(last_latency_1, loading_latency_2)

    old_bw = loading_bw_2
    new_bw = old_bw*loading_latency_2/transition_latency

    transition_bw = new_bw

    if debug:
        print(f"transition_latency: {transition_latency}")
        print(f"old_bw: {old_bw}")
        print(f"transition_bw: {transition_bw}")
        print()

    new_loading_latency = loading_latency_1
    new_compute_latency = compute_latency_1 + transition_latency + compute_latency_2
    new_last_latency = last_latency_2

    new_peak_bw = max([peak_bw_1, transition_bw, peak_bw_2])
    new_avg_bw = (avg_bw_1*compute_latency_1 + transition_bw*transition_latency + avg_bw_2*compute_latency_2) / new_compute_latency

    new_latency_stats = [new_loading_latency, new_compute_latency, new_last_latency]
    new_bw_stats = [loading_bw_1, new_avg_bw, new_peak_bw]
    new_total_latency = int(math.ceil(new_loading_latency + new_compute_latency + new_last_latency))

    return new_total_latency, new_latency_stats, new_bw_stats, first_block_size_1

def scale_bw_opencheck(msm_trace, scale_factor, available_bw, debug=False):
    latency, latency_stats, bw_stats, first_block_size = msm_trace

    assert type(latency) == int and len(latency_stats) == 3 and len(bw_stats) == 3

    loading_latency, compute_latency, last_latency = latency_stats
    loading_bw, avg_bw, peak_bw = bw_stats

    new_loading_bw = loading_bw*scale_factor
    new_avg_bw = avg_bw*scale_factor
    new_peak_bw = peak_bw*scale_factor

    if debug:
        print()
        print(f"original_loading_latency: {loading_latency}")

    # Compute the amount of data transferred with loading_bw over loading_latency
    data_transferred_loading = new_loading_bw * loading_latency

    # Compute the actual latency using available_bw
    min_loading_latency = math.ceil(data_transferred_loading / available_bw)
    hit_max_load_rate = False


    # If actual_loading_latency > loading_latency, add the difference to loading_latency
    if min_loading_latency > loading_latency:
        
        # want the fill rate to be either 0 < f <= 1, or an integer f > 1
        effective_fill_rate = math.floor(first_block_size / min_loading_latency)

        if effective_fill_rate == 0:
            # fill rate is < 1
            actual_loading_latency = min_loading_latency 
        else:
            # fill rate is an integer
            actual_loading_latency = math.ceil(first_block_size / effective_fill_rate)

        loading_latency = actual_loading_latency
        new_loading_bw = available_bw
        hit_max_load_rate = True

    if debug:
        print(f"new_loading_bw: {new_loading_bw}")
        print(f"hit_max_load_rate: {hit_max_load_rate}")
        print(f"loading_latency: {loading_latency}")
        print()


    # Compute the amount of data transferred with avg_bw over compute_latency
    data_transferred_compute = new_avg_bw * compute_latency

    # Compute the actual latency using available_bw
    actual_compute_latency = math.ceil(data_transferred_compute / available_bw)

    # If actual_compute_latency > compute_latency, add the difference to compute_latency
    if actual_compute_latency > compute_latency:
        compute_latency = actual_compute_latency
        new_avg_bw = available_bw
        new_peak_bw = available_bw

    new_latency_stats = [loading_latency, compute_latency, last_latency]
    new_total_latency = int(math.ceil(loading_latency + compute_latency + last_latency))

    new_bw_stats = [new_loading_bw, new_avg_bw, new_peak_bw]

    return (new_total_latency, new_latency_stats, new_bw_stats, first_block_size), hit_max_load_rate

def scale_bw_opencheck_v1(msm_trace, scale_factor, available_bw, debug=False):
    latency, latency_stats, bw_stats, first_block_size = msm_trace

    assert type(latency) == int and len(latency_stats) == 3 and len(bw_stats) == 3

    loading_latency, compute_latency, last_latency = latency_stats
    loading_bw, avg_bw, peak_bw = bw_stats

    new_loading_bw = loading_bw*scale_factor
    new_avg_bw = avg_bw*scale_factor
    new_peak_bw = peak_bw*scale_factor

    if debug:
        print()
        print(f"original_loading_latency: {loading_latency}")

    # Compute the amount of data transferred with loading_bw over loading_latency
    data_transferred_loading = new_loading_bw * loading_latency

    # Compute the actual latency using available_bw
    min_loading_latency = math.ceil(data_transferred_loading / available_bw)
    hit_max_load_rate = False


    # If actual_loading_latency > loading_latency, add the difference to loading_latency
    if min_loading_latency > loading_latency:
        loading_latency = min_loading_latency
        new_loading_bw = available_bw
        hit_max_load_rate = True

    if debug:
        print(f"new_loading_bw: {new_loading_bw}")
        print(f"hit_max_load_rate: {hit_max_load_rate}")
        print(f"loading_latency: {loading_latency}")
        print()

    # Compute the amount of data transferred with avg_bw over compute_latency
    data_transferred_compute = new_avg_bw * compute_latency

    # Compute the actual latency using available_bw
    actual_compute_latency = math.ceil(data_transferred_compute / available_bw)

    # If actual_compute_latency > compute_latency, add the difference to compute_latency
    if actual_compute_latency > compute_latency:
        compute_latency = actual_compute_latency
        new_avg_bw = available_bw
        new_peak_bw = available_bw

    new_latency_stats = [loading_latency, compute_latency, last_latency]
    new_total_latency = int(math.ceil(loading_latency + compute_latency + last_latency))

    new_bw_stats = [new_loading_bw, new_avg_bw, new_peak_bw]

    return (new_total_latency, new_latency_stats, new_bw_stats, first_block_size), hit_max_load_rate

# old dense latency bw model
def _old2(elements_in_block_array, block_latency_array, ocw, bits_per_element, padd_latency, available_bw, freq, debug=False):

    adjusted_elements_array = np.array(elements_in_block_array[1:])
    adjusted_latency_array = np.array(block_latency_array[:-1])
    elements_per_cycle = adjusted_elements_array/adjusted_latency_array

    # bits per element here should be 2D points
    GB_per_second = calc_bw(bits_per_element, elements_per_cycle, freq)
    
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

    available_rate = calc_rate(bits_per_element, available_bw, freq)   
    points_loading_latency, _, lower_rate = get_msm_load_overhead(ocw, elements_in_block_array[0], available_rate, rate_match=False)

    block_latency_array = [points_loading_latency] + block_latency_array

    total_latency = sum(block_latency_array)
    loading_bw = calc_bw(bits_per_element, lower_rate, freq)
    main_compute_latency = sum(adjusted_latency_array)
    transfer_less_compute_latency = block_latency_array[-1]

    latency_stats = [points_loading_latency, main_compute_latency, transfer_less_compute_latency]
    bw_stats = [loading_bw, avg_bw, peak_bw]
    return total_latency, latency_stats, bw_stats

def dense_latency_bw_model(elements_in_block_array, block_latency_array, ocw, bits_per_element, padd_latency, available_bw, freq, debug=False):

    if debug:
        print("dense latency bw model")
        print(block_latency_array)
        print(elements_in_block_array)
        print()

    trace_data = elements_in_block_array, block_latency_array
    supplemental_data = bits_per_element, freq, available_bw

    # avg_bw and peak_bw are for the main compute duration in a given trace, initial loading is not included
    adjusted_latency_array, avg_bw, peak_bw = get_main_compute_stats(trace_data, supplemental_data, debug=debug)
    
    trace_data_2 = elements_in_block_array, block_latency_array, adjusted_latency_array
    supplemental_data_2 = ocw, bits_per_element, freq, available_bw
    total_latency, loading_bw, latency_stats, first_block_size, fill_rate = get_full_stats(trace_data_2, supplemental_data_2, debug=debug)

    bw_stats = [loading_bw, avg_bw, peak_bw]
    return total_latency, latency_stats, bw_stats, first_block_size, fill_rate

# old ones latency bw model
def _old(num_ones, ppw, ocw, bits_per_element, padd_latency, available_bw, freq, debug=False):

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

    adjusted_elements_array = np.array(elements_in_block_array[1:])
    adjusted_latency_array = np.array(block_latency_array[:-1])
    elements_per_cycle = adjusted_elements_array/adjusted_latency_array

    # bits per element here should be 2D points
    GB_per_second = calc_bw(bits_per_element, elements_per_cycle, freq)
    
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

    # combine results of all groups. we can reasonably assume no need to do off-chip transfers
    # because the size of the on-chip scratchpad memory area (num_buckets*num_windows) exceeds
    # the number of groups (.45*num_points)/ppw (setting ocw = 1, num_windows = scalar_size/window_size)
    if final_reduce:
        final_reduce_latency = ones_reduction_latency(num_groups, padd_latency, ocw, queue_depth=0)
        block_latency_array[-1] += final_reduce_latency

    available_rate = calc_rate(bits_per_element, available_bw, freq)   
    points_loading_latency, exceeded_drain_rate, lower_rate = get_msm_load_overhead(ocw, elements_in_block_array[0], available_rate, rate_match=False)

    block_latency_array = [points_loading_latency] + block_latency_array

    total_latency = sum(block_latency_array)
    loading_bw = calc_bw(bits_per_element, lower_rate, freq)
    main_compute_latency = sum(adjusted_latency_array)
    transfer_less_compute_latency = block_latency_array[-1]

    latency_stats = [points_loading_latency, main_compute_latency, transfer_less_compute_latency]
    bw_stats = [loading_bw, avg_bw, peak_bw]
    
    return total_latency, latency_stats, bw_stats

def ones_latency_bw_model(num_ones, ppw, ocw, bits_per_element, padd_latency, available_bw, freq, debug=False):

    elements_in_block_array, block_latency_array = construct_ones_trace(ppw, ocw, num_ones, padd_latency)
    
    if debug:
        print("ones latency bw model")
        print(f"block_latency_array: {block_latency_array}")
        print(f"elements_in_block_array: {elements_in_block_array}")
        print()

    trace_data = elements_in_block_array, block_latency_array
    supplemental_data = bits_per_element, freq, available_bw
    
    adjusted_latency_array, avg_bw, peak_bw = get_main_compute_stats(trace_data, supplemental_data, debug=debug)

    trace_data_2 = elements_in_block_array, block_latency_array, adjusted_latency_array
    supplemental_data_2 = ocw, bits_per_element, freq, available_bw

    total_latency, loading_bw, latency_stats, first_block_size, _ = get_full_stats(trace_data_2, supplemental_data_2, debug=debug)
    
    bw_stats = [loading_bw, avg_bw, peak_bw]
    return total_latency, latency_stats, bw_stats, first_block_size


# get the list of all designs we are investigating
def get_designs(num_vars_list, distribution, ws_list, ppw_list, ocw_list, qd_list, ii_list, padd_latency):

    all_num_fraction_combos = []
    for num_vars, (fraction_ones, fraction_dense) in itertools.product(num_vars_list, distribution):
        if fraction_ones == 0:
            number_suffix = num_vars
            half_number_suffix = num_vars - 1
            if number_suffix == 0:
                half_number_suffix = 0


        # if there are 1s, we need to use the dense proportion
        else:
            number_suffix = round((1 << num_vars) * fraction_dense)
            half_number_suffix = round((1 << (num_vars - 1)) * fraction_dense)

        all_num_fraction_combos.append((num_vars, fraction_ones, number_suffix, half_number_suffix, fraction_dense))

    all_designs = list(itertools.product(
        all_num_fraction_combos,   # (num_vars, fraction_ones, number_suffix, fraction_dense)
        ws_list,                   # window_sizes
        ppw_list,                  # ppws
        ocw_list,                  # ocws
        qd_list,                   # queue_depths
        ii_list,                   # init_intervals
        [padd_latency]             # padd_latency
    ))

    return all_designs

def process_file(file_path, target_ws, target_ocw, optimal_br_latency_dict, return_first_compute_block_latency=False):

    # Check if the file exists
    if os.path.exists(file_path):            
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

            # checking that simulation stats are consistent
            total_cycles = np.squeeze(data['total_cycles_matrix'])
            sort_cycles = np.squeeze(data['sort_cycles_matrix'])
            actual_reduction_cycles = np.squeeze(data['bucket_reduction_cycles_matrix'] + data['window_reduction_cycles_matrix']) 
            assert total_cycles == sort_cycles + actual_reduction_cycles
            
            # hacky fix for now
            bucket_reduction_cycles = calculate_total_bucket_reduce_latency_opt(bits_per_scalar, target_ws, target_ocw, optimal_br_latency_dict)
            actual_reduction_cycles = bucket_reduction_cycles + np.squeeze(data['window_reduction_cycles_matrix'])

            total_cycles = sort_cycles + actual_reduction_cycles

            block_latency_array = np.atleast_1d(np.squeeze(data['block_latency_array'])).tolist()
            elements_in_block_array = np.atleast_1d(np.squeeze(data['elements_in_block_array'])).tolist()
    else:
        print(file_path)
        exit(f"{file_path} not found even tho it should be there")

    block_latency_array[-1] += actual_reduction_cycles

    if return_first_compute_block_latency:
        return block_latency_array, elements_in_block_array, block_latency_array[0]

    return block_latency_array, elements_in_block_array

# while witness MSMs are running, we want to fetch other MLEs into on-chip storage
# def get_mle_loading_penalty():

def get_sparse_msm_stats(designs, bits_per_type, padd_area, available_bw, freq, base_dir, optimal_bucket_reduction_latency_dict, gate_type="vanilla", debug=False):

    data_dict = dict()

    bits_per_scalar, bits_per_point, bits_per_point_reduced = bits_per_type
    
    # max_data_size = 52429
    max_data_size = 1677722

    for design in designs:
        
        data_dict[design] = dict()
        (num_vars, fraction_ones, number_suffix, half_number_suffix, fraction_dense), target_ws, target_ppw, target_ocw, target_qd, target_ii, padd_latency = design
        
        if debug:
            print(f"num_vars: {num_vars}, fraction_ones: {fraction_ones}, number_suffix: {number_suffix}, half_number_suffix: {half_number_suffix}, fraction_dense: {fraction_dense}")
            print()
        
        for idx, num_dense_words in enumerate([number_suffix, half_number_suffix]):
            
            sub_debug = False
            # if idx > 0:
            #     continue
            
            file_dir = base_dir + f"SYNTHETIC_{num_dense_words}/{ADD_POLICY}/{padd_latency}_cycles/"
            file_path = os.path.join(file_dir, f"ws{target_ws}_ppw{target_ppw}_ocw{target_ocw}_qd{target_qd}_ii{target_ii}.pkl")

            # block_latency_array, elements_in_block_array = process_file(file_path, target_ws, target_ocw, optimal_bucket_reduction_latency_dict)
            
            if num_dense_words <= max_data_size:
                block_latency_array, elements_in_block_array = process_file(file_path, target_ws, target_ocw, optimal_bucket_reduction_latency_dict)
            else:
                config = target_ws, target_ppw, target_ocw, target_qd, target_ii
                params = base_dir, ADD_POLICY, padd_latency
                block_latency_array, elements_in_block_array, sort_cycles, window_reduce_cycles = get_extrap_traces_sparse(num_dense_words, max_data_size, config, params)
                first_compute_block_latency = block_latency_array[0]

                bucket_reduction_cycles = calculate_total_bucket_reduce_latency_opt(bits_per_scalar, target_ws, target_ocw, optimal_bucket_reduction_latency_dict)
                actual_reduction_cycles = bucket_reduction_cycles + window_reduce_cycles
                total_cycles = sort_cycles + actual_reduction_cycles

                block_latency_array[-1] += actual_reduction_cycles
            

            if debug:
                print(f"num_dense_words: {num_dense_words}")
                print(f"block_latency_array: {block_latency_array}")
                print(f"elements_in_block_array: {elements_in_block_array}")
                print()
            
            input_bw = available_bw >> idx

            # for sparse MSMs, we just need to fetch points initially #
            bits_per_element = bits_per_point_reduced

            num_sparse_words = round(fraction_ones*(1 << (num_vars - idx)))
            if debug:
                print(f"num_sparse_words: {num_sparse_words}")
    
            total_ones_latency, ones_latency_stats, ones_bw_stats, first_ones_block_size = ones_latency_bw_model(num_sparse_words, target_ppw, target_ocw, bits_per_element, padd_latency, input_bw, freq, debug=sub_debug)

            if debug:
                print(f"total_ones_latency: {total_ones_latency}")
                print(f"ones_latency_stats: {ones_latency_stats}")
                print(f"ones_bw_stats: {ones_bw_stats}")
                print(f"first_ones_block_size: {first_ones_block_size}")
                print()
            
            msm_trace_1 = total_ones_latency, ones_latency_stats, ones_bw_stats, first_ones_block_size

            bits_per_element = bits_per_scalar + bits_per_point_reduced
            total_dense_latency, dense_latency_stats, dense_bw_stats, first_dense_block_size, _ = dense_latency_bw_model(elements_in_block_array, block_latency_array, target_ocw, bits_per_element, padd_latency, input_bw, freq, debug=sub_debug)

            if debug:
                print(f"total_dense_latency: {total_dense_latency}")
                print(f"dense_latency_stats: {dense_latency_stats}")
                print(f"dense_bw_stats: {dense_bw_stats}")
                print(f"first_dense_block_size: {first_dense_block_size}")
                print()

            msm_trace_2 = total_dense_latency, dense_latency_stats, dense_bw_stats, first_dense_block_size

            # construct a trace depending upon the gate we are using
            sparse_msm_trace = stitch_msms(msm_trace_1, msm_trace_2, debug=sub_debug)
            
            if debug:
                print(f"total_sparse_latency: {sparse_msm_trace[0]}")
                print(f"sparse_latency_stats: {sparse_msm_trace[1]}")
                print(f"sparse_bw_stats: {sparse_msm_trace[2]}")
                print(f"sparse_first_block_size: {sparse_msm_trace[3]}")
                print()

            if gate_type == "vanilla":
                sparse_msm_trace_1    = stitch_msms(sparse_msm_trace, sparse_msm_trace)
                full_sparse_msm_trace = stitch_msms(sparse_msm_trace, sparse_msm_trace_1)
            elif gate_type == "jellyfish":
                sparse_msm_trace_1    = stitch_msms(sparse_msm_trace, sparse_msm_trace)
                sparse_msm_trace_2    = stitch_msms(sparse_msm_trace, sparse_msm_trace_1)
                sparse_msm_trace_3    = stitch_msms(sparse_msm_trace, sparse_msm_trace_2)
                full_sparse_msm_trace = stitch_msms(sparse_msm_trace, sparse_msm_trace_3)
            else:
                exit("gate type not supported")

            if debug:
                if gate_type == "vanilla":
                    print(sparse_msm_trace_1)
                    print(full_sparse_msm_trace)
                elif gate_type == "jellyfish":
                    print(sparse_msm_trace_1)
                    print(sparse_msm_trace_2)
                    print(sparse_msm_trace_3)
                    print(full_sparse_msm_trace)
                print()
            total_sparse_latency, sparse_latency_stats, sparse_bw_stats, sparse_first_block_size = full_sparse_msm_trace

            num_cores = 1 if idx == 0 else 2
            sparse_area_stats = get_msm_area_estimate(target_ws, target_ppw, target_ocw, target_qd, \
            bits_per_scalar, bits_per_point, padd_area, num_cores)

            core_key = 'single_core' if idx == 0 else 'dual_core'
            data_dict[design][core_key] = {
                'total_latency': total_sparse_latency,
                'latency_stats': sparse_latency_stats,
                'bw_stats': sparse_bw_stats,
                'first_block_size': sparse_first_block_size,
                'area_stats': sparse_area_stats
            }

    return data_dict

def get_dense_msm_stats(designs, bits_per_type, padd_area, available_bw, freq, base_dir, optimal_bucket_reduction_latency_dict):

    data_dict = dict()

    bits_per_scalar, bits_per_point, bits_per_point_reduced = bits_per_type
    max_var_data_size = 20
    # max_var_data_size = 16
    for design in designs:
        
        data_dict[design] = dict()

        (num_vars, _, number_suffix, half_number_suffix, _), target_ws, target_ppw, target_ocw, target_qd, target_ii, padd_latency = design
        
        debug = False
        sub_debug = False

        # for last condition, should we stack msm traces on top of each other?
        for idx, (num_vars_in, input_bw) in enumerate([
            (number_suffix, available_bw),          # Full bandwidth for number_suffix --> 1 core model
            (number_suffix, available_bw >> 1),     # Half bandwidth for number_suffix --> 2 core model in permcheck
            (half_number_suffix, available_bw >> 1) # Half bandwidth for half_number_suffix --> 2 core model in opencheck
        ]):
            file_dir = base_dir + f"SYNTHETIC_{num_vars_in}/{ADD_POLICY}/{padd_latency}_cycles/"
            file_path = os.path.join(file_dir, f"ws{target_ws}_ppw{target_ppw}_ocw{target_ocw}_qd{target_qd}_ii{target_ii}.pkl")

            if num_vars_in <= max_var_data_size:
                block_latency_array, elements_in_block_array, first_compute_block_latency = process_file(file_path, target_ws, target_ocw, optimal_bucket_reduction_latency_dict, True)
            else:
                config = target_ws, target_ppw, target_ocw, target_qd, target_ii
                params = base_dir, ADD_POLICY, padd_latency
                block_latency_array, elements_in_block_array, sort_cycles, window_reduce_cycles = get_extrap_traces(num_vars_in, max_var_data_size, config, params)
                first_compute_block_latency = block_latency_array[0]

                bucket_reduction_cycles = calculate_total_bucket_reduce_latency_opt(bits_per_scalar, target_ws, target_ocw, optimal_bucket_reduction_latency_dict)
                actual_reduction_cycles = bucket_reduction_cycles + window_reduce_cycles
                total_cycles = sort_cycles + actual_reduction_cycles

                block_latency_array[-1] += actual_reduction_cycles
            
            bits_per_element = bits_per_scalar + bits_per_point_reduced
            
            total_dense_latency, dense_latency_stats, dense_bw_stats, first_dense_block_size, dense_fill_rate = \
                dense_latency_bw_model(elements_in_block_array, block_latency_array, target_ocw, bits_per_element, padd_latency, input_bw, freq, debug=sub_debug)

            if debug:
                print(f"num_vars: {num_vars_in}, input_bw: {input_bw}")
                print(f"total_dense_latency: {total_dense_latency}")
                print(f"dense_latency_stats: {dense_latency_stats}")
                print(f"dense_bw_stats: {dense_bw_stats}")
                print(f"first_dense_block_size: {first_dense_block_size}")
                print(f"assumed fill rate: {dense_fill_rate}")
                print()

            # Determine core key based on the setting
            if idx == 0:
                core_key = 'single_core'
            elif idx == 1:
                core_key = 'dual_core_permcheck'
            else:
                core_key = 'dual_core_opencheck'

            num_cores = 1 if idx == 0 else 2
            dense_area_stats = get_msm_area_estimate(target_ws, target_ppw, target_ocw, target_qd, \
            bits_per_scalar, bits_per_point, padd_area, num_cores)

            data_dict[design][core_key] = {
                'total_latency': total_dense_latency,
                'latency_stats': dense_latency_stats,
                'bw_stats': dense_bw_stats,
                'first_block_size': first_dense_block_size,
                "fill_rate": dense_fill_rate,
                'first_compute_block_latency': first_compute_block_latency,
                'area_stats': dense_area_stats
            }

    return data_dict

def calculate_transfer_time(num_elements, r1, t1, r2):
    """
    Calculate the total time to transfer num_points of data elements with two transfer rates.

    Parameters:
        num_points (int): Number of points to transfer.
        bits_per_element (float): Size of each element in bits.
        r1 (elements/cycle): Rate of transfer for the first time period.
        t1 (cycles): Time during which data is transferred at rate r1.
        r2 (elements/cycle): Rate of transfer for the remaining time period.

    Returns:
        float: Total time required to transfer all elements (seconds).
    """
    # Total data to transfer in bits
    total_data = num_elements

    # Data transferred at rate r1 during time t1
    data_transferred_r1 = r1 * t1

    # If all data is transferred within t1
    if data_transferred_r1 >= total_data:
        return math.ceil(total_data / r1)  # All data transferred at rate r1

    # Remaining data to transfer
    remaining_data = total_data - data_transferred_r1

    # Time required to transfer remaining data at rate r2
    t2 = math.ceil(remaining_data / r2)

    # Total time
    total_time = t1 + t2
    return total_time


# the point of this function is to more accurately model the initial loading of the MSM to account for the bandwidth
# used up by the MLE generation in permcheck. Key assumption: the time to load and compute on the first block of MSM
# inputs is greater than the time to generate all scalars.
# TODO: if this assumption is not true, account for it
def adjust_loading_bw_old(bw_stats, size_data, rate_data, supplemental_data):
    num_points, num_elements_total = size_data
    first_compute_block_latency, t_load_scalars_block, t_generate_scalars, available_rate, max_rate = rate_data
    bits_per_point_reduced, scalar_bits_transferred_per_cycle, max_bw, freq = supplemental_data
    
    t_load_points_block_1 = math.ceil(num_elements/available_rate)
    t_load_msm_block_1 = max(t_load_scalars_block, t_load_points_block_1)

    # check that our existing model is not underestimating the time to load the second block of points
    if t_generate_scalars > t_load_msm_block_1:
        # remaining time to write MLE tables in permcheck
        t_rest = t_generate_scalars - t_load_msm_block_1
        assert t_rest < first_compute_block_latency

        t_load_points_block_2 = calculate_transfer_time(num_points, available_rate, t_rest, max_rate)
    
        assert t_load_points_block_2 > 0
        
        # assuming that even with bandwidth limits, we can still load points before the first block finishs computing 
        assert first_compute_block_latency > t_load_points_block_2

        # with said assumption, we can calculate the lower bandwidth bound for the first block
        first_block_bw = calc_bw(bits_per_point_reduced, num_points/first_compute_block_latency, freq)

        # we dont need to use this information for now, because we are assuming a higher fetch bandwidth during the
        # first compute block, since it assumes fetching of scalars and points. we're not fetching scalars during the
        # first compute block, but no need to update the bandwidth stats here

    total_data_transferred = (2*num_points)*bits_per_point_reduced + num_elements_total*scalar_bits_transferred_per_cycle
    min_transfer_cycles = total_data_transferred/BITS_PER_GB/max_bw*freq

    assert min_transfer_cycles <= t_load_msm_block_1 + first_compute_block_latency

    new_loading_bw = calc_bw(bits_per_point_reduced, num_points/t_load_msm_block_1, freq)
    new_loading_latency = t_load_msm_block_1

    return new_loading_bw, new_loading_latency


# use this function
def adjust_loading_bw_permcheck(msm_trace, fill_rate, bits_per_element, available_bw, freq, debug=False):

    total_latency, latency_stats, bw_stats, first_block_size = msm_trace
    
    loading_latency, compute_latency, last_latency = latency_stats
    loading_bw, avg_bw, peak_bw = bw_stats

    num_elements = first_block_size

    scalar_load_cycles = math.ceil(num_elements/fill_rate)
    available_points_load_rate = calc_rate(bits_per_element, available_bw, freq)
    point_load_cycles = math.ceil(num_elements/available_points_load_rate)

    if debug:
        print(f"available bw: {available_bw}")
        print(f"scalar_load_cycles: {scalar_load_cycles}")
        print(f"point_load_cycles: {point_load_cycles}")
        print()

    min_load_cycles = max(scalar_load_cycles, point_load_cycles)
    new_loading_bw = calc_bw(bits_per_element, num_elements/min_load_cycles, freq)

    new_loading_latency = min_load_cycles
    new_total_latency = int(math.ceil(new_loading_latency + compute_latency + last_latency))

    new_msm_trace = new_total_latency, [new_loading_latency, compute_latency, last_latency], [new_loading_bw, avg_bw, peak_bw], first_block_size
    return new_msm_trace

# use this function
def adjust_loading_bw_polyopen(msm_trace, fill_rate, metadata, debug=False):

    total_latency, latency_stats, bw_stats, first_block_size = msm_trace
    bits_per_scalar_adjusted, bits_per_point_reduced, available_bw, freq = metadata

    loading_latency, compute_latency, last_latency = latency_stats
    loading_bw, avg_bw, peak_bw = bw_stats

    num_elements = first_block_size

    desired_scalar_bw = calc_bw(bits_per_scalar_adjusted, fill_rate, freq)
    scalar_load_cycles = math.ceil(num_elements/fill_rate)

    rest_bw = available_bw - desired_scalar_bw
    assert rest_bw > 0

    if debug:
        print(f"desired_scalar_bw: {desired_scalar_bw}")
        print(f"scalar_load_cycles: {scalar_load_cycles}")
        print(f"rest_bw: {rest_bw}")

    available_points_load_rate = calc_rate(bits_per_point_reduced, rest_bw, freq)
    points_loaded = available_points_load_rate * scalar_load_cycles
    
    if points_loaded < num_elements:
        remaining_points = num_elements - points_loaded
    else:
        remaining_points = 0
    
    if debug:
        print(f"available_points_load_rate: {available_points_load_rate}")
        print(f"points_loaded: {points_loaded}")
        print(f"remaining_points: {remaining_points}") 


    available_points_load_rate = calc_rate(bits_per_point_reduced, available_bw, freq)
    additional_cycles = math.ceil(remaining_points/available_points_load_rate)

    new_loading_latency = scalar_load_cycles + additional_cycles
    net_effective_rate = num_elements / new_loading_latency
    new_loading_bw = calc_bw(bits_per_scalar_adjusted + bits_per_point_reduced, net_effective_rate, freq)
    
    if debug:
        print(f"new_loading_latency: {new_loading_latency}")
        print(f"net_effective_rate: {net_effective_rate}")
        print(f"new_loading_bw: {new_loading_bw}")

    new_total_latency = int(math.ceil(new_loading_latency + compute_latency + last_latency))

    new_msm_trace = new_total_latency, [new_loading_latency, compute_latency, last_latency], [new_loading_bw, avg_bw, peak_bw], first_block_size
    return new_msm_trace


if __name__ == "__main__":
    x = get_msm_area_estimate(9, 4096, 32, 16, 255, 2*381, 23.77, num_cores=1)
    print(x)
