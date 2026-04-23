
import math
import numpy as np
from .util import *
import re

# # key assumptions
# # 1. streaming architecture - only nominal amount of SRAM provisioned for queuing data
# # 2. modeling up to 2^30 variables
# # 3. all MLE inputs and intermediates stored off chip
# def mle_update_bw_utilization(pes_per_mle_update, bitwidth, freq, num_mles_in_parallel):

#     num_words_read_per_cycle_per_pe = 2*num_mles_in_parallel
#     num_words_written_per_cycle_per_pe = num_mles_in_parallel

#     total_num_words_xferred_per_cycle = (num_words_read_per_cycle_per_pe + num_words_written_per_cycle_per_pe)*pes_per_mle_update
#     peak_bw_util = total_num_words_xferred_per_cycle*freq*bitwidth/BITS_PER_GB
#     return peak_bw_util

# def get_mle_update_latency(entries_per_mle, pe_latency, num_pes, queue_depth=0):
    
#     if entries_per_mle/(2*num_pes) < 1:
#         total_cycles = pe_latency + 2*queue_depth
#     else:
#         total_cycles = entries_per_mle/(2*num_pes) + pe_latency + 2*queue_depth - 1

#     return total_cycles

# def sumcheck_core_bw_utilization(num_pes, num_mles, bitwidth, freq):
#     num_words_read_per_cycle_per_pe = 2*num_mles
#     total_num_words_read_per_cycle = num_words_read_per_cycle_per_pe*num_pes
#     peak_bw_util = total_num_words_read_per_cycle*freq*bitwidth/BITS_PER_GB
#     return peak_bw_util

# # this currently assumes infinite bandwidth
# # all MLEs must be accessed in parallel in each PE
# def sumcheck_core_latency(entries_per_mle, pe_latency, reduction_latency, num_pes, queue_depth=0):

#     if entries_per_mle/(2*num_pes) < 1:
#         compute_cycles = pe_latency + queue_depth
#     else:
#         compute_cycles = entries_per_mle/(2*num_pes) + pe_latency + queue_depth - 1

#     # this is an upper bound
#     pe_reduction_latency = int(math.ceil(math.log2(num_pes)))

#     total_cycles = compute_cycles + reduction_latency + pe_reduction_latency

#     return total_cycles

# def sumcheck_latency(num_vars, sumcheck_pe_latency, sumcheck_reduction_latency, num_sumcheck_core_pes, modmuls_per_sumcheck_core, mle_update_pe_latency, \
#     pes_per_mle_update, modmuls_per_mle_update_pe, transcript_latency, freq, bitwidth, total_required_mles=9, num_mles_in_parallel=9):

#     entries_per_mle = 1 << num_vars
#     total_latency = 0

#     scale_factor = math.ceil(total_required_mles/num_mles_in_parallel)
    
#     accumulated_sc_latency = 0
#     accumulated_mu_latency = 0
#     accumulated_tc_latency = 0
#     # pes_per_mle_update means for a single MLE update, how many parallel accesses are there
#     total_modmuls = num_mles_in_parallel*pes_per_mle_update*modmuls_per_mle_update_pe + num_sumcheck_core_pes*modmuls_per_sumcheck_core

#     for i in range(num_vars - 1):
#         sc_latency = sumcheck_core_latency(entries_per_mle, sumcheck_pe_latency, sumcheck_reduction_latency, num_sumcheck_core_pes)
#         mu_latency = get_mle_update_latency(entries_per_mle, mle_update_pe_latency, pes_per_mle_update)

#         accumulated_sc_latency += sc_latency
#         accumulated_mu_latency += (mu_latency*scale_factor)
#         accumulated_tc_latency += transcript_latency
        
#         total_latency += sc_latency + transcript_latency + (mu_latency*scale_factor)
#         entries_per_mle >>= 1
    
#     sc_latency = sumcheck_core_latency(entries_per_mle, sumcheck_pe_latency, sumcheck_reduction_latency, num_sumcheck_core_pes)
#     total_latency += sc_latency + transcript_latency
#     accumulated_sc_latency += sc_latency
#     accumulated_tc_latency += transcript_latency
    
#     # runtime in ms
#     total_runtime = total_latency/freq*1000
    
#     sc_peak_bw_util = sumcheck_core_bw_utilization(num_sumcheck_core_pes, total_required_mles, bitwidth, freq)
#     mu_peak_bw_util = mle_update_bw_utilization(pes_per_mle_update, bitwidth, freq, num_mles_in_parallel)

#     return total_latency, total_runtime, sc_peak_bw_util, mu_peak_bw_util, total_modmuls, accumulated_sc_latency, accumulated_mu_latency, accumulated_tc_latency

# for a given number of polynomials and physical columns, return how many rounds are needed to accumulate the product of all polynomials
# and if the final round has any leftover polynomials, the number of leftover polynomials
# this function effectively returns num_full rounds and 1 round with leftover polynomials, if any
def get_poly_rounds(num_polynomials, num_eval_engines):
    
    rounds = 0
    if num_polynomials > num_eval_engines:
        # Full groups
        full = num_polynomials // num_eval_engines
        # leftover group if remainder > 0
        rem = num_polynomials % num_eval_engines
        num_polynomials = full + rem
        counts, remainder_polynomials = get_poly_rounds(num_polynomials, num_eval_engines)
        rounds += full + counts
    else:
        return (1, num_polynomials % num_eval_engines)
    # update num_polynomials
    return rounds, remainder_polynomials

# this function returns the latency for one stripe of data for one term with multiple polynomials
def stripe_latency(num_inputs, num_rows, num_eval_engines, num_extensions, num_polynomials, pipeline_depth_data, actual_ii=None):
    
    # for product_depth_dict, we're assuming the product lanes allow for early exit
    extensions_depth, product_depth_dict = pipeline_depth_data

    # actual_ii is passed in if we are under a bandwidth constraint
    if actual_ii == None:
        rowwise_ii = num_extensions/num_rows
    else:
        rowwise_ii = actual_ii

    total_rounds, remainder_polynomials = get_poly_rounds(num_polynomials, num_eval_engines)
    if remainder_polynomials > 0:
        full_rounds = total_rounds - 1
        remainder_rounds = 1
    else:
        full_rounds = total_rounds
        remainder_rounds = 0
    
    full_rounds_depth = extensions_depth + product_depth_dict[num_eval_engines]
    remainder_rounds_depth = extensions_depth + product_depth_dict[remainder_polynomials]

    # full rounds
    full_rounds_cycles = full_rounds * (full_rounds_depth + (num_inputs - 1)*rowwise_ii)

    # remainder rounds
    remainder_rounds_cycles = remainder_rounds * (remainder_rounds_depth + (num_inputs - 1)*rowwise_ii)

    total_cycles = full_rounds_cycles + remainder_rounds_cycles
    
    return rowwise_ii, total_cycles


##########################################
#### we are using the below functions ####
##########################################

def group_strings_recursive(strings, n, round_idx=1, node_counter=1, results=None):
    """
    Groups 'strings' into batches of size 'n'.
    - Each full batch (size == n) is replaced by a single 'int_x' label in the next iteration.
    - Any leftover batch (size < n) is carried over to the next iteration unchanged.
    - Continues until len(strings) <= n.

    We track:
      1) round_idx  : which "round" of grouping (1-based)
      2) node_counter: a global "node" index that increments each time a new group is formed
      3) group_label : the name used for that group in the next iteration
      4) group_items : the actual items that formed this group

    Returns:
      (results, next_node_counter)

      where 'results' is a list of tuples (round_idx, node_id, group_label, group_items).
    """
    if results is None:
        results = []
    
    k = len(strings)
    
    # If k <= n, then this is the final group
    if k <= n:
        # Make a label for the final group
        group_label = f"int_final" if k == n else "leftover_final"
        this_node_id = node_counter
        node_counter += 1
        
        # Record this final group
        results.append((round_idx, this_node_id, group_label, strings))
        return results, node_counter

    # Otherwise, break strings into full groups of size n, plus leftover
    full = k // n           # number of full groups
    leftover_size = k % n   # leftover count
    new_strings_for_next_round = []

    # Process each full group
    for i in range(full):
        batch = strings[i*n : i*n + n]    # slice of size n
        group_label = f"int_{round_idx}_{i+1}"

        # Record that we formed a new node
        this_node_id = node_counter
        node_counter += 1
        
        results.append((round_idx, this_node_id, group_label, batch))

        # In the next iteration, this full group becomes a single item (the group_label)
        new_strings_for_next_round.append(group_label)
    
    # Handle leftover (fewer than n elements)
    # We do not create a new node here; we just carry them forward unchanged.
    if leftover_size > 0:
        leftover_batch = strings[full*n : ]
        new_strings_for_next_round.extend(leftover_batch)
    
    # Recurse: move to round_idx+1
    return group_strings_recursive(new_strings_for_next_round, n, round_idx+1, node_counter, results)

def process_list_of_lists(list_of_lists, n):
    """
    For each sublist in 'list_of_lists':
      1) Run group_strings_recursive(sublist, n).
      2) Collect the unique "original" strings encountered 
         (i.e., anything not starting with 'int_' or ending with '_final').
      3) Print or return them.
    """
    all_results = []  # We'll store the final info for each sublist
    
    # Collect unique original strings from the 'results'
    unique_elements = set()
    
    for list_idx, sublist in enumerate(list_of_lists, start=1):
        results, _ = group_strings_recursive(sublist, n)
        
        for idx, (rnd, node_id, label, items) in enumerate(results):
            unique_elements_in_sublist = []
            for it in items:
                # Filter out generated labels like "int_1_1" or "int_final"
                # We'll treat anything that doesn't match those patterns as original
                if not it.startswith("int_") and not it.endswith("_final"):
                    if it not in unique_elements:
                        unique_elements.add(it)
                        unique_elements_in_sublist.append(it)
            results[idx] = (rnd, node_id, label, items, unique_elements_in_sublist)
        
        # Save or print info
        sublist_info = {
            "index": list_idx,
            "sublist": sublist,
            "grouping_results": results,
            "unique_elements": sorted(unique_elements),
        }
        all_results.append(sublist_info)
    
    return all_results

def annotate_prefetches(traversal_list):
    computation_round = 1
    schedule = [[0, None, None, None, None, None]]
    round_0_prefetches = None
    for step in traversal_list:
        num_polynomials_in_term = len(step["sublist"])
        first_mle = step["sublist"][0]
        control_mle = first_mle if (first_mle.startswith("q")) else "None"

        num_extensions_needed = num_polynomials_in_term + 1
        for (level, node_id, _, curr_polynomials, new_polynomials) in step["grouping_results"]:
            # print(f"  Level {level}, Node {node_id}: group={curr_polynomials}, new_items={new_polynomials}")
            schedule.append([computation_round, num_extensions_needed, curr_polynomials, new_polynomials, control_mle, None])
            # print(schedule)
            if len(new_polynomials) > 0:
                schedule[computation_round - 1][-1] = new_polynomials # which polynomials are prefetched

            if computation_round - 1 == 0:
                round_0_prefetches = new_polynomials 

            computation_round += 1
    schedule[computation_round - 1][-1] = round_0_prefetches
    schedule.append(schedule[-1][:])
    schedule[-1][-1] = None
    return schedule

# TODO: refactor this function to better balance the prefetches without wraparound
def distribute_excess(excess_list, current_round_idx, rounds_data, max_prefetch):
    """
    Attempt to distribute polynomials in 'excess_list' to other rounds
    that have fewer than max_prefetch items in their 'prefetch'.
    
    We do a cyclical search order as described:
      For round i, check (i-1, i-2, ..., 0, num_rounds-1, ..., i+1)
      or some variant. We'll show one approach.

    Returns any polynomials that could not be placed.
    """
    num_rounds = len(rounds_data)
    if num_rounds <= 1:
        return excess_list  # nowhere else to put them

    # Build the search order
    # e.g. if current_round_idx=7, we want [6,5,4,3,2,1,0, 8,9,...] but that might exceed
    # actual # of rounds. We'll do something simpler:
    order = []
    # go backwards from i-1 down to 0
    for x in range(current_round_idx-1, -1, -1):
        order.append(x)
    # then from the end down to current_round_idx+1 (wrapping around):
    for x in range(num_rounds-1, current_round_idx, -1):
        order.append(x)

    # Now distribute
    while excess_list and order:
        # We'll try to place the "last" polynomial in excess_list into some slot
        poly = excess_list.pop()
        placed = False

        for o in order:
            pfetch_count = len(rounds_data[o]["prefetch"])
            if pfetch_count < max_prefetch:
                rounds_data[o]["prefetch"].append(poly)
                placed = True
                break

        if not placed:
            # We couldn't place this polynomial in any candidate round,
            # so we must keep it. Put it back into the leftover
            excess_list.append(poly)
            break  # no point continuing

    return excess_list

# TODO: refactor this function to better balance the prefetches without wraparound
def balance_prefetches(schedule):
    """
    Given a 'schedule' of the form:
      schedule = [
        (round_idx, current_polys, prefetch_polys),  # for each round
        ...
      ]
    we rebalance the prefetch_polys so that no round has more than
    'max_prefetch' polynomials, where:
      max_prefetch = ceil(total_prefetch / num_rounds).

    Returns an updated 'schedule' with the same structure but balanced prefetches.
    """

    num_rounds = len(schedule)

    # 1) Gather total number of prefetched polynomials across all rounds
    total_prefetch = 0
    for _, _, pfetch in schedule:
        if pfetch is not None:
            total_prefetch += len(pfetch)

    if num_rounds == 0 or total_prefetch == 0:
        # Nothing to do
        return schedule

    max_prefetch = math.ceil(total_prefetch / num_rounds)

    # Convert the schedule into a mutable structure so we can reassign
    # Something like:
    # rounds_data = [
    #   {
    #     "round_idx": ...,
    #     "current": [...],
    #     "prefetch": [...]
    #   },
    #   ...
    # ]
    rounds_data = []
    for (rnd_idx, curr, pfetch) in schedule:
        rounds_data.append({
            "round_idx": rnd_idx,
            "current": curr,
            "prefetch": (pfetch if pfetch is not None else [])
        })

    # 2) Walk from the last round down to the first
    for i in range(num_rounds - 1, -1, -1):
        prefetch_list = rounds_data[i]["prefetch"]
        count_here = len(prefetch_list)

        # Check if we exceed the max_prefetch
        excess = count_here - max_prefetch
        if excess <= 0:
            # No problem in this round
            continue

        # We have 'excess' polynomials that we want to move into "earlier" rounds
        # in a cyclical search (per your specification).
        # We'll do a function that tries to place 'excess' polynomials 
        # in other rounds that have fewer than max_prefetch.
        # We'll prefer to place them in rounds that have 0 prefetch 
        # (if that is the desired behavior), else partial.

        # We'll treat the polynomials in LIFO or FIFO order as we distribute them out.
        # For simplicity, let's say we pop from the end.
        to_reassign = []
        while len(prefetch_list) > max_prefetch:
            to_reassign.append(prefetch_list.pop())  # remove from the end

        # Now 'to_reassign' is the chunk of polynomials we need to place
        # to other rounds, from last to first in the 'to_reassign' list.

        # We'll cycle from (i-1, i-2, ..., 0, num_rounds-1, ..., i+1)
        # but in practice, your specification said for round i
        # we check i-1, i-2,... 1, then wrap to num_rounds-1, etc.
        # We'll define a helper method:
        to_reassign = distribute_excess(to_reassign, i, rounds_data, max_prefetch)

        # If 'to_reassign' is not empty, it means we couldn't place everything.
        # We'll just put them back in the current round for lack of a better place.
        # (Or you could do something else if you want to forcibly fail, etc.)
        if to_reassign:
            prefetch_list.extend(to_reassign)

    # 3) Reconstruct the schedule from rounds_data
    new_schedule = []
    for rd in rounds_data:
        pfetch = rd["prefetch"] if rd["prefetch"] else None
        new_schedule.append((rd["round_idx"], rd["current"], pfetch))

    return new_schedule

def annotate_ii_and_latency(schedule, num_product_lanes, extensions_latency, modmul_latency):
    updated_schedule = []
    for round_info in schedule:
        computation_round, num_extensions_needed, curr_polynomials, new_polynomials, control_mle, prefetched_polynomials = round_info
        if curr_polynomials != None:
            num_polynomials = len(curr_polynomials)
            best_ii = num_extensions_needed/num_product_lanes
            if best_ii < 1:
                best_ii = 1
            if best_ii == int(best_ii):
                best_ii = int(best_ii)

            product_lane_latency = math.ceil(math.log2(num_polynomials))*modmul_latency
            extensions_and_product_latency = extensions_latency + product_lane_latency
        else:
            best_ii = None
            extensions_and_product_latency = None
        updated_schedule.append([computation_round, num_extensions_needed, curr_polynomials, new_polynomials, control_mle, prefetched_polynomials, best_ii, extensions_and_product_latency])
    return updated_schedule

# this function is specifically for rounds 3-mu for zerocheck and all rounds for permcheck and opencheck
def add_stepwise_latency(schedule, num_pes, words_per_mle, supplemental_data, no_offchip_read=False, no_offchip_write=False, round_1=False):
    
    bits_per_element, available_bw, freq = supplemental_data
    total_step_latency = 0
    initial_prefetch_latency = 0
    last_step_latency = 0
    total_words_written = 0
    updated_schedule = []
 
    num_words_stored = 0
    num_steps = len(schedule)
 
    pattern = re.compile(r'^fz\d*$')
 
    for step_id, step in enumerate(schedule):
        computation_round, num_extensions_needed, curr_polynomials, new_polynomials, control_mle, prefetched_polynomials, best_ii, extensions_and_product_latency = step
            
        num_ports_per_mle = 4*num_pes
        max_read_rate = calc_rate(bits_per_element, available_bw, freq) # "reading into" RF (actually a write operation)

        if prefetched_polynomials != None:
            fz_present_prefetched_poly = any(pattern.match(elem) for elem in prefetched_polynomials)
        else:
            fz_present_prefetched_poly = False
        if curr_polynomials == None:
            
            if prefetched_polynomials == None:
                exit("this can't be correct")
            

            if fz_present_prefetched_poly and round_1:
                num_prefetched_polynomials = len(prefetched_polynomials) - 1
            else:
                num_prefetched_polynomials = len(prefetched_polynomials)
            
            desired_read_rate = num_ports_per_mle*num_prefetched_polynomials # can read up to these many words per cycle
            num_words_read = num_prefetched_polynomials*words_per_mle
            actual_read_rate = min(desired_read_rate, max_read_rate)
            needed_read_bw = calc_bw(bits_per_element, actual_read_rate, freq)

            # in rounds when everything is on chip
            if no_offchip_read:
                num_words_read = 0
                needed_read_bw = 0

            step_latency = math.ceil(num_words_read/actual_read_rate)
            performance_numbers = step_latency, needed_read_bw, num_words_read, 0, 0, 0, 0
            initial_prefetch_latency = step_latency

        else:
            # round 1, read 2 entries per MLE
            if round_1:
                effective_num_inputs = math.ceil(words_per_mle/(2*num_pes))
            # for round 2, we are reading 4 entries per MLE
            else:
                effective_num_inputs = math.ceil(words_per_mle/(4*num_pes))           
            
            pipeline_latency = (effective_num_inputs - 1)*best_ii + extensions_and_product_latency
            pipeline_latency = math.ceil(pipeline_latency)
            if prefetched_polynomials == None or no_offchip_read:
                num_words_read = 0
                needed_read_bw = 0
                step_latency = pipeline_latency
            else:

                if fz_present_prefetched_poly and round_1:
                    num_prefetched_polynomials = len(prefetched_polynomials) - 1
                else:
                    num_prefetched_polynomials = len(prefetched_polynomials)

                num_words_read = num_prefetched_polynomials*words_per_mle
    
                desired_read_rate = num_words_read/pipeline_latency
                actual_read_rate = min(desired_read_rate, max_read_rate)
                needed_read_bw = calc_bw(bits_per_element, actual_read_rate, freq)

                # bandwidth throttled read latency (if applicable)
                if num_words_read == 0:
                    read_latency = 0
                else:
                    read_latency = math.ceil(num_words_read/actual_read_rate)

                # latency of this step is maximum of the read cycles and computation cycles
                step_latency = max(read_latency, pipeline_latency)

            fz_present = any(pattern.match(elem) for elem in new_polynomials)

            # if there's an fz in round 1, it means we are using build MLE in round 1, and we need to perform writebacks to HBM
            if round_1 and fz_present:
                num_words_generated = words_per_mle
                available_write_bw = available_bw - needed_read_bw

                # TODO: this should be based off of read ports (i.e. rate should not exceed number of read ports)
                writeable_rate = calc_rate(bits_per_element, available_write_bw, freq)

                num_words_written = int(min(writeable_rate*step_latency, num_words_generated))
                num_words_stored += num_words_generated - num_words_written

            else:
                if no_offchip_write:
                    num_words_generated = 0
                    available_write_bw = 0
                else:
                    num_inputs = math.ceil(words_per_mle/4)
                    num_words_generated = num_inputs*2*len(new_polynomials)

                    available_write_bw = available_bw - needed_read_bw
                
                # TODO: this should be based off of read ports (i.e. rate should not exceed number of read ports)
                writeable_rate = calc_rate(bits_per_element, available_write_bw, freq)

                num_words_written = int(min(writeable_rate*step_latency, num_words_generated))
                num_words_stored += num_words_generated - num_words_written

            performance_numbers = step_latency, needed_read_bw, num_words_read, available_write_bw, num_words_generated, num_words_written, num_words_stored

            if step_id == num_steps - 1:
                last_step_latency = step_latency
            else:
                total_step_latency += step_latency
            total_words_written += num_words_written

        hardware_params = num_extensions_needed, curr_polynomials, new_polynomials, control_mle, prefetched_polynomials, best_ii, extensions_and_product_latency

        updated_schedule.append([computation_round, hardware_params, performance_numbers])

    # TODO: make this more realistic based on the number of read ports in the MLE Update FIFOs. basically should be effective number of 
    # MLEs stored (a function of the number of words divided by buffer size)*2 ports

    # if we still have words pending writes to HBM, assume max read rate (here "read" is a literal read from the FIFOs implemented as RF) 
    if no_offchip_write:
        additional_write_cycles = 0
    else:
        additional_write_cycles = math.ceil(num_words_stored/max_read_rate)
    
    if additional_write_cycles == 0:
        additional_write_data = additional_write_cycles, 0, 0, 0, total_words_written
    else:
        additional_write_data = additional_write_cycles, available_bw, max_read_rate, num_words_stored, total_words_written

    total_step_latency += additional_write_cycles
    latency_data = initial_prefetch_latency, total_step_latency, last_step_latency
    return updated_schedule, latency_data, additional_write_data


def sparse_read_stats(num_ports_per_mle, prefetched_polynomials, words_per_mle, bits_per_element, avg_bits_per_witness_word, round_num):
    
    desired_read_rate = num_ports_per_mle*len(prefetched_polynomials) # can read up to these many words per cycle
    
    q_count = sum(1 for s in prefetched_polynomials if s.startswith('q') and s != "qc")
    
    # not supporting multiple qi polynomials handled at once for now
    assert q_count == 1 or q_count == 0
    
    # for round 1, fetch 1 bit per MLE entry and log2(K/2) bits for the K/2 differences (stored). for round 2, just fetch the raw bits
    overhead = 1 if round_num == 1 else 0
    overhead_bits = overhead * math.log2(words_per_mle/2)*(words_per_mle/2)
    count = q_count*(1*words_per_mle + overhead_bits)/bits_per_element

    # in round 1, we are building fz in zerocheck, not fetching from off-chip
    if round_num == 2 and "fz" in prefetched_polynomials:
        count += words_per_mle

    w_count = sum(1 for s in prefetched_polynomials if s.startswith('w') or s == "qc")
    
    # fetch avg # witness bits per MLE entry and log2(K) bits for the K differences (not stored but count towards bandwidth)
    count += w_count*(avg_bits_per_witness_word + math.log2(words_per_mle))*words_per_mle/bits_per_element
    # print(q_count, w_count, count)
    return count, desired_read_rate

# """
# this function is specifically for round 1 for zerocheck
def add_stepwise_latency_rounds12_zerocheck(schedule, num_pes, words_per_mle, supplemental_data, avg_bits_per_witness_word, round_num, skip_fraction_dict=None):
    
    bits_per_element, available_bw, freq = supplemental_data
    total_step_latency = 0
    initial_prefetch_latency = 0
    last_step_latency = 0
    total_words_written = 0
    updated_schedule = []
 
    num_words_stored = 0
 
    assert round_num == 1 or round_num == 2
    if round_num == 1:
        assert skip_fraction_dict != None

    pattern = re.compile(r'^fz\d*$')

    num_steps = len(schedule)

    # print_schedule(schedule)
 
    for step_id, step in enumerate(schedule):
        computation_round, num_extensions_needed, curr_polynomials, new_polynomials, control_mle, prefetched_polynomials, best_ii, extensions_and_product_latency = step
            
        num_ports_per_mle = 4*num_pes
        max_read_rate = calc_rate(bits_per_element, available_bw, freq) # "reading into" RF (actually a write operation)
               
        if curr_polynomials == None:
            
            if prefetched_polynomials == None:
                exit("this can't be correct")

            num_words_read, desired_read_rate = sparse_read_stats(num_ports_per_mle, prefetched_polynomials, words_per_mle, bits_per_element, avg_bits_per_witness_word, round_num)
            
            actual_read_rate = min(desired_read_rate, max_read_rate)
            needed_read_bw = calc_bw(bits_per_element, actual_read_rate, freq)

            step_latency = math.ceil(num_words_read/actual_read_rate)
            performance_numbers = step_latency, needed_read_bw, num_words_read, 0, 0, 0, 0
            initial_prefetch_latency = step_latency

        else:
            # for round 1, reading 2 entries per MLE, and some of them are skipped because consecutive 0s in the q polynomial mean multiplication with 0
            if round_num == 1:
                skip_fraction = skip_fraction_dict[control_mle] # sparsity based early exit
                effective_num_inputs = math.ceil((words_per_mle/2)*(1 - skip_fraction)/(num_pes))
            # for round 2, we are reading 4 entries per MLE
            else:
                effective_num_inputs = math.ceil(words_per_mle/(4*num_pes))

            pipeline_latency = (effective_num_inputs - 1)*best_ii + extensions_and_product_latency
            pipeline_latency = math.ceil(pipeline_latency)
            
            if prefetched_polynomials == None:
                num_words_read = 0
                needed_read_bw = 0
                step_latency = pipeline_latency
            else:
                num_words_read, _ = sparse_read_stats(num_ports_per_mle, prefetched_polynomials, words_per_mle, bits_per_element, avg_bits_per_witness_word, round_num)
                desired_read_rate = num_words_read/pipeline_latency
                actual_read_rate = min(desired_read_rate, max_read_rate)
                needed_read_bw = calc_bw(bits_per_element, actual_read_rate, freq)

                # print(step)

                # bandwidth throttled read latency (if applicable)
                if num_words_read == 0:
                    read_latency = 0
                else:
                    read_latency = math.ceil(num_words_read/actual_read_rate)
                
                # latency of this step is maximum of the read cycles and computation cycles
                step_latency = max(read_latency, pipeline_latency)

            fz_present = any(pattern.match(elem) for elem in new_polynomials)

            # no MLE update in round 1
            if round_num == 1:
                
                if not fz_present:
                    num_words_generated = 0
                    num_words_written = 0
                    num_words_stored = 0
                    available_write_bw = 0
                
                # if there's an fz, it means we are using build MLE in round 1, and we need to perform writebacks to HBM
                else:
                    num_words_generated = words_per_mle
                    available_write_bw = available_bw - needed_read_bw

                    # TODO: this should be based off of read ports (i.e. rate should not exceed number of read ports)
                    writeable_rate = calc_rate(bits_per_element, available_write_bw, freq)

                    num_words_written = int(min(writeable_rate*step_latency, num_words_generated))
                    num_words_stored += num_words_generated - num_words_written

            # MLE update in round 2
            else:
                # MLE update if we are not in round 1 and there are new polynomials encountered
                num_inputs = math.ceil(words_per_mle/4)
                num_words_generated = num_inputs*2*len(new_polynomials)

                available_write_bw = available_bw - needed_read_bw
                
                # TODO: this should be based off of read ports (i.e. rate should not exceed number of read ports)
                writeable_rate = calc_rate(bits_per_element, available_write_bw, freq)

                num_words_written = int(min(writeable_rate*step_latency, num_words_generated))
                num_words_stored += num_words_generated - num_words_written

            performance_numbers = step_latency, needed_read_bw, num_words_read, available_write_bw, num_words_generated, num_words_written, num_words_stored
            
            if step_id == num_steps - 1:
                last_step_latency = step_latency
            else:
                total_step_latency += step_latency
            
            total_words_written += num_words_written

        hardware_params = num_extensions_needed, curr_polynomials, new_polynomials, control_mle, prefetched_polynomials, best_ii, extensions_and_product_latency

        updated_schedule.append([computation_round, hardware_params, performance_numbers])

    # TODO: make this more realistic based on the number of read ports in the MLE Update FIFOs. basically should be effective number of 
    # MLEs stored (a function of the number of words divided by buffer size)*2 ports

    # if we still have words pending writes to HBM, assume max read rate (here "read" is a literal read from the FIFOs implemented as RF) 
    additional_write_cycles = math.ceil(num_words_stored/max_read_rate)
    if additional_write_cycles == 0:
        additional_write_data = additional_write_cycles, 0, 0, 0, total_words_written
    else:
        additional_write_data = additional_write_cycles, available_bw, max_read_rate, num_words_stored, total_words_written

    total_step_latency += additional_write_cycles
    latency_data = initial_prefetch_latency, total_step_latency, last_step_latency
    return updated_schedule, latency_data, additional_write_data

def create_schedule(sumcheck_polynomial, num_eval_engines, num_product_lanes, extensions_latency, modmul_latency):
    
    # Process each sublist
    schedule_tree = process_list_of_lists(sumcheck_polynomial, num_eval_engines)

    # Annotate prefetches and latency
    schedule = annotate_prefetches(schedule_tree)
    annotated_schedule = annotate_ii_and_latency(schedule, num_product_lanes, extensions_latency, modmul_latency)

    return annotated_schedule

def print_schedule(schedule):
    print()
    for step in schedule:
        print(f"comp phase: {step[0]}, ext needed: {step[1]}, curr polys: {step[2]}, new_polys: {step[3]}, control_mle: {step[4]}, prefetch polys: {step[5]}, best ii: {step[6]}, ext+prod latency: {step[7]}")
        print()

def print_schedule_with_perf(schedule, total_step_latency, write_data):
    print()
    for step in schedule:
        computation_round, hardware_params, performance_numbers = step
        print(f"comp phase: {computation_round}")
        ext_needed, curr_polys, new_polys, control_mle, prefetch_polys, best_ii, ext_prod_latency = hardware_params
        print(f"ext needed: {ext_needed}, curr polys: {curr_polys}, new_polys: {new_polys}, control_mle: {control_mle}, prefetch polys: {prefetch_polys}, best ii: {best_ii}, ext+prod latency: {ext_prod_latency}")
        latency, needed_read_bw, num_words_read, available_write_bw, num_words_generated, num_words_written, num_words_stored = performance_numbers
        print(f"latency: {latency}, read bw: {needed_read_bw}, write bw: {available_write_bw}, num words read: {num_words_read}, num words generated: {num_words_generated}, num words written: {num_words_written}, num words stored: {num_words_stored}")
        print()

    additional_write_cycles, write_bw, words_written_per_cycle, spillover_words, directly_written_words = write_data
    print(f"additional write cycles: {additional_write_cycles}, write bw: {write_bw}, words written per cycle: {words_written_per_cycle}, spillover words: {spillover_words}, directly written words: {directly_written_words}")
    print()
    print(f"total step latency: {total_step_latency}")


def create_zerocheck_schedule(num_vars, sumcheck_polynomial, sumcheck_hardware_params, sparsity_data, supplemental_data, debug=False):
    num_pes, num_eval_engines, num_product_lanes, mle_update_latency, extensions_latency, modmul_latency, mle_buffer_size = sumcheck_hardware_params
    avg_bits_per_witness_word, skip_fraction_dict = sparsity_data
    bits_per_element, available_bw, freq = supplemental_data

    schedule_no_mle_update = create_schedule(sumcheck_polynomial, num_eval_engines, num_product_lanes, extensions_latency, modmul_latency)
    # print_schedule(schedule_no_mle_update)
    round_latencies = [None] * num_vars
    schedules = [None] * num_vars

    # assume we always have enough modmuls in MLE combine/dot product unit to do all barycentric dot products in paralle
    # 10 is based off of jellyfish (highest degree of 7)
    barycentric_dp_latency = num_vars*(modmul_latency + 10)

    input_size = 1<<num_vars
    words_per_mle = mle_buffer_size

    if input_size < mle_buffer_size:
        words_per_mle = input_size

    schedule_round_1, latency_data, additional_write_data = add_stepwise_latency_rounds12_zerocheck(schedule_no_mle_update, num_pes, words_per_mle, supplemental_data, avg_bits_per_witness_word, round_num=1, skip_fraction_dict=skip_fraction_dict)
    
    initial_prefetch_latency_round_1, total_step_latency_round_1, last_step_latency_round_1 = latency_data 

    if debug:
        print("#############################################################")
        print(f"round 1: Input Size: 2^{int(math.log2(input_size))}")

        print_schedule_with_perf(schedule_round_1, total_step_latency_round_1, additional_write_data)
    
    # adding build mle warmup latency as bias to first round
    num_build_mle = 1
    round_latencies[0] = (total_step_latency_round_1*math.ceil(input_size/mle_buffer_size)) + initial_prefetch_latency_round_1 + last_step_latency_round_1 + num_vars*modmul_latency*num_build_mle
    schedules[0] = schedule_round_1
    
    if debug:
        print(f"round 1: Output Size: 2^{int(math.log2(input_size))}, latency: {round_latencies[0]}")
        print("#############################################################")

    schedule_mle_update = create_schedule(sumcheck_polynomial, num_eval_engines, num_product_lanes, extensions_latency + mle_update_latency, modmul_latency)
    # print_schedule(schedule_mle_update)

    # we don't condense input size yet
    words_per_mle = mle_buffer_size

    if input_size < mle_buffer_size:
        words_per_mle = input_size

    if debug:
        print(f"round 2: Input Size: 2^{int(math.log2(input_size))}")

    schedule_round_2, latency_data, additional_write_data = add_stepwise_latency_rounds12_zerocheck(schedule_mle_update, num_pes, mle_buffer_size, supplemental_data, avg_bits_per_witness_word, round_num=2)
    initial_prefetch_latency_round_2, total_step_latency_round_2, last_step_latency_round_2 = latency_data 
    if debug:
        print_schedule_with_perf(schedule_round_2, total_step_latency_round_2, additional_write_data)
    
    round_latencies[1] = (total_step_latency_round_2*math.ceil(input_size/mle_buffer_size)) + initial_prefetch_latency_round_2 + last_step_latency_round_2
    schedules[1] = schedule_round_2

    input_size >>= 1
    
    if debug:
        print(f"round 2: Output Size: 2^{int(math.log2(input_size))}, latency: {round_latencies[1]}")
        print("#############################################################")

    for round_num in range(2, num_vars):

        if debug:
            print(f"round {round_num + 1}: Input Size: 2^{int(math.log2(input_size))}")

        words_per_mle = mle_buffer_size
        if input_size < mle_buffer_size:
            words_per_mle = input_size

        if input_size > 2*mle_buffer_size:
            no_offchip_read = False
            no_offchip_write = False
        
        # in this case, we dont need to write to off-chip. we have enough MLE Update RF size so that writes are decoupled and buffered, they can be written back into MLE buffers later
        elif input_size == 2*mle_buffer_size:
            no_offchip_read = False
            no_offchip_write = True
        elif input_size <= mle_buffer_size:
            no_offchip_read = True
            no_offchip_write = True

        schedule_round, latency_data, additional_write_data = add_stepwise_latency(schedule_mle_update, num_pes, words_per_mle, supplemental_data, no_offchip_read=no_offchip_read, no_offchip_write=no_offchip_write)
        initial_prefetch_latency_round_x, total_step_latency_round_x, last_step_latency_round_x = latency_data
        if debug:
            print_schedule_with_perf(schedule_round, total_step_latency_round_x, additional_write_data)
        round_latencies[round_num] = (total_step_latency_round_x*math.ceil(input_size/mle_buffer_size))  + initial_prefetch_latency_round_x + last_step_latency_round_x
        schedules[round_num] = schedule_round

        # dont add the last latency once we're all on-chip
        if input_size <= mle_buffer_size:
            round_latencies[round_num] -= last_step_latency_round_x
                
        input_size >>= 1
        
        if debug:
            print(f"round {round_num + 1}: Output Size: 2^{int(math.log2(input_size))}, latency: {round_latencies[round_num]}")
            print("#############################################################")

        # exit()
    # print(round_latencies)
    # print(sum(round_latencies))

    round_latencies.append(barycentric_dp_latency)

    return round_latencies, schedules

def create_sumcheck_schedule(num_vars, sumcheck_polynomial, sumcheck_hardware_params, sparsity_data, num_build_mle, supplemental_data, debug=False):
    num_pes, num_eval_engines, num_product_lanes, mle_update_latency, extensions_latency, modmul_latency, mle_buffer_size = sumcheck_hardware_params
    avg_bits_per_witness_word, skip_fraction_dict = sparsity_data
    bits_per_element, available_bw, freq = supplemental_data

    schedule_no_mle_update = create_schedule(sumcheck_polynomial, num_eval_engines, num_product_lanes, extensions_latency, modmul_latency)
    # print_schedule(schedule_no_mle_update)
    round_latencies = [None] * num_vars
    schedules = [None] * num_vars

    # assume we always have enough modmuls in MLE combine/dot product unit to do all barycentric dot products in paralle
    # 10 is based off of jellyfish (highest degree of 7)
    barycentric_dp_latency = num_vars*(modmul_latency + 10)

    input_size = 1<<num_vars
    words_per_mle = mle_buffer_size

    if input_size < mle_buffer_size:
        words_per_mle = input_size

    schedule_round_1, latency_data, additional_write_data = add_stepwise_latency(schedule_no_mle_update, num_pes, words_per_mle, supplemental_data, no_offchip_read=False, no_offchip_write=True, round_1=True)

    initial_prefetch_latency_round_1, total_step_latency_round_1, last_step_latency_round_1 = latency_data 

    if debug:
        print("#############################################################")
        print(f"round 1: Input Size: 2^{int(math.log2(input_size))}")

        print_schedule_with_perf(schedule_round_1, total_step_latency_round_1, additional_write_data)
    
    # adding build mle warmup latency as bias to first round
    round_latencies[0] = (total_step_latency_round_1*math.ceil(input_size/mle_buffer_size)) + initial_prefetch_latency_round_1 + last_step_latency_round_1  + num_vars*modmul_latency*num_build_mle
    schedules[0] = schedule_round_1
    
    if debug:
        print(f"round 1: Output Size: 2^{int(math.log2(input_size))}, latency: {round_latencies[0]}")
        print("#############################################################")

    schedule_mle_update = create_schedule(sumcheck_polynomial, num_eval_engines, num_product_lanes, extensions_latency + mle_update_latency, modmul_latency)
    # print_schedule(schedule_mle_update)

    # we don't condense input size yet
    words_per_mle = mle_buffer_size

    if input_size < mle_buffer_size:
        words_per_mle = input_size

    if debug:
        print(f"round 2: Input Size: 2^{int(math.log2(input_size))}")

    schedule_round_2, latency_data, additional_write_data = add_stepwise_latency(schedule_mle_update, num_pes, words_per_mle, supplemental_data, no_offchip_read=False, no_offchip_write=False)
    initial_prefetch_latency_round_2, total_step_latency_round_2, last_step_latency_round_2 = latency_data 
    if debug:
        print_schedule_with_perf(schedule_round_2, total_step_latency_round_2, additional_write_data)
    
    round_latencies[1] = (total_step_latency_round_2*math.ceil(input_size/mle_buffer_size)) + initial_prefetch_latency_round_2 + last_step_latency_round_2
    schedules[1] = schedule_round_2

    input_size >>= 1

    if debug:
        print(f"round 2: Output Size: 2^{int(math.log2(input_size))}, latency: {round_latencies[1]}")
        print("#############################################################")

    for round_num in range(2, num_vars):

        if debug:
            print(f"round {round_num + 1}: Input Size: 2^{int(math.log2(input_size))}")

        words_per_mle = mle_buffer_size
        if input_size < mle_buffer_size:
            words_per_mle = input_size

        if input_size > 2*mle_buffer_size:
            no_offchip_read = False
            no_offchip_write = False
        
        # in this case, we dont need to write to off-chip. we have enough MLE Update RF size so that writes are decoupled and buffered, they can be written back into MLE buffers later
        elif input_size == 2*mle_buffer_size:
            no_offchip_read = False
            no_offchip_write = True
        elif input_size <= mle_buffer_size:
            no_offchip_read = True
            no_offchip_write = True

        schedule_round, latency_data, additional_write_data = add_stepwise_latency(schedule_mle_update, num_pes, words_per_mle, supplemental_data, no_offchip_read=no_offchip_read, no_offchip_write=no_offchip_write)
        initial_prefetch_latency_round_x, total_step_latency_round_x, last_step_latency_round_x = latency_data
        if debug:
            print_schedule_with_perf(schedule_round, total_step_latency_round_x, additional_write_data)
        round_latencies[round_num] = (total_step_latency_round_x*math.ceil(input_size/mle_buffer_size))  + initial_prefetch_latency_round_x + last_step_latency_round_x
        
        # dont add the last latency once we're all on-chip
        if input_size <= mle_buffer_size:
            round_latencies[round_num] -= last_step_latency_round_x
        
        schedules[round_num] = schedule_round
        
        input_size >>= 1
    
        if debug:
            print(f"round {round_num + 1}: Output Size: 2^{int(math.log2(input_size))}, latency: {round_latencies[round_num]}")
            print("#############################################################")

        # exit()
    # print(round_latencies)
    # print(sum(round_latencies))
    round_latencies.append(barycentric_dp_latency)
    return round_latencies, schedules
    

def performance_model(num_vars, sumcheck_polynomial, sumcheck_type, sumcheck_hardware_params, sparsity_data, supplemental_data, debug=False):
    
    # Create the schedule
    if sumcheck_type == "zerocheck":
        round_latencies, schedules = create_zerocheck_schedule(num_vars, sumcheck_polynomial, sumcheck_hardware_params, sparsity_data, supplemental_data, debug)
    elif sumcheck_type in ["permcheck", "opencheck"]:
        num_build_mle = 5 if sumcheck_type == "opencheck" else 1
        round_latencies, schedules = create_sumcheck_schedule(num_vars, sumcheck_polynomial, sumcheck_hardware_params, sparsity_data, num_build_mle, supplemental_data, debug)
    total_latency = sum(round_latencies)
    return total_latency, round_latencies, schedules


if __name__ == "__main__":

    # for num_eval_engines in range(3, 8):
    #     for num_polynomials in range(1,8):
    #         print(f"({num_eval_engines}, {num_polynomials}): {get_poly_rounds(num_polynomials, num_eval_engines)}")

    # for num_eval_engines in range(3, 8):
    #     for num_polynomials in range(1, 8):
    #         # Create a list of num_polynomials strings
    #         # e.g. if num_polynomials=3, strings=["a","b","c"]
    #         strings = [chr(ord('a') + i) for i in range(num_polynomials)]

    #         # Call the grouping function
    #         results, _ = group_strings_recursive(strings, num_eval_engines)

    #         print(f"\n=== num_eval_engines={num_eval_engines}, num_polynomials={num_polynomials}, strings={strings} ===")
    #         for (rnd, node_id, label, items) in results:
    #             print(f"  Level {rnd}, Node {node_id}: label='{label}', group={items}")
    
    # print("\n\n")
    
    sumcheck_polynomial = [
        ["q1", "w1", "fz"],
        ["q2", "w2", "fz"],
        ["q3", "w3", "fz"],
        ["qM", "w1", "w2", "fz"],
        ["qc", "fz"],
    ]
    # sumcheck_polynomial = [
    #     ["q1", "w1", "fz"],
    #     ["q2", "w2", "fz"],
    #     ["q3", "w3", "fz"],
    #     ["q4", "w4", "fz"],
    #     ["q5", "w5", "fz"],
    #     ["qM1", "w1", "w2", "fz"],
    #     ["qM2", "w3", "w4", "fz"],
    #     ["qH1", "w1", "w1", "w1", "w1", "w1", "fz"],
    #     ["qH2", "w2", "w2", "w2", "w2", "w2", "fz"],
    #     ["qH3", "w3", "w3", "w3", "w3", "w3", "fz"],
    #     ["qH4", "w4", "w4", "w4", "w4", "w4", "fz"],
    #     ["qECC", "w1", "w2", "w3", "w4", "w5", "fz"],
    #     ["qc", "fz"],
    # ]
    
    # sumcheck_polynomial = [
    #     ["a", "n", "s"],
    #     ["b", "o", "s"],
    #     ["c", "p", "s"],
    #     ["d", "q", "s"],
    #     ["e", "r", "s"],
    #     ["f", "n", "o", "s"],
    #     ["g", "p", "q", "s"],
    #     ["h", "n", "n", "n", "n", "n", "s"],
    #     ["i", "o", "o", "o", "o", "o", "s"],
    #     ["j", "p", "p", "p", "p", "p", "s"],
    #     ["k", "q", "q", "q", "q", "q", "s"],
    #     ["l", "n", "o", "p", "q", "r", "s"],
    #     ["m", "s"],
    # ]

    # sumcheck_polynomial = [
    #     ["pi", "fz"],
    #     ["p1", "p2", "fz"],
    #     ["phi", "d1", "d2", "d3", "d4", "d5", "fz"],
    #     ["n1", "n2", "n3", "n4", "n5", "fz"],
    # ]
    
    # sumcheck_polynomial = [
    #     ["pi", "fz"],
    #     ["p1", "p2", "fz"],
    #     ["phi", "d1", "d2", "d3", "fz"],
    #     ["n1", "n2", "n3", "fz"],
    # ]

    # sumcheck_polynomial = [
    #     ["y1", "t1"],
    #     ["y2", "t2"],
    #     ["y3", "t3"],
    #     ["y4", "t4"],
    #     ["y5", "t5"],
    #     ["y6", "t6"],
    # ]


    num_eval_engines = 3

    # Process each sublist
    results = process_list_of_lists(sumcheck_polynomial, num_eval_engines)

    # Print out the results
    for info in results:
        print(f"\n=== Sublist #{info['index']}: {info['sublist']} ===")
        
        # Show the entire grouping trace
        for (rnd, node_id, label, items, new_items) in info["grouping_results"]:
            print(f"  Tree Level {rnd}, Node {node_id}: label='{label}', group={items}, new_items={new_items}")
    
    print()
    print("Unique elements encountered:", info["unique_elements"], "num unique elements:", len(info["unique_elements"]))
    print()
    schedule = annotate_prefetches(results)
    for step in schedule:
        print(f"comp phase: {step[0]}, ext needed: {step[1]}, curr polys: {step[2]}, new polys: {step[3]}, control mle: {step[4]}, prefetch polys: {step[5]}")


    # TODO: need to better balance the MLEs being prefetched

    # print("\n=== Original Schedule ===")
    # for (r,c,p) in schedule:
    #     print(f"Round {r}, current={c}, prefetch={p}")

    # balanced = balance_prefetches(schedule[1:])

    # print("\n=== Balanced Schedule ===")
    # for (r,c,p) in balanced:
    #     print(f"Round {r}, current={c}, prefetch={p}")

    num_product_lanes = 4
    extensions_latency = 5
    modmul_latency = 10
    updated_schedule = annotate_ii_and_latency(schedule, num_product_lanes, extensions_latency, modmul_latency)
    print()

    for step in updated_schedule:
        print(f"comp phase: {step[0]}, ext needed: {step[1]}, curr polys: {step[2]}, new_polys: {step[3]}, control_mle: {step[4]}, prefetch polys: {step[5]}, best ii: {step[6]}, ext+prod latency: {step[7]}")
    
    num_pes = 4
    mle_buffer_size = 256

    bits_per_element = 256 # for simplicity

    freq = 1e9
    bw_limit = 128 # GB/s
    supplemental_data = bits_per_element, bw_limit, freq

    no_offchip_read = True
    no_offchip_write = True

    updated_schedule, total_step_latency, additional_write_data = add_stepwise_latency(updated_schedule, num_pes, mle_buffer_size, supplemental_data, no_offchip_read, no_offchip_write)
    print()
    print()
    print()
    print()
    
    for step in updated_schedule:
        computation_round, hardware_params, performance_numbers = step
        print(f"comp phase: {computation_round}")
        ext_needed, curr_polys, new_polys, control_mle, prefetch_polys, best_ii, ext_prod_latency = hardware_params
        print(f"ext needed: {ext_needed}, curr polys: {curr_polys}, new_polys: {new_polys}, control_mle: {control_mle}, prefetch polys: {prefetch_polys}, best ii: {best_ii}, ext+prod latency: {ext_prod_latency}")
        latency, needed_read_bw, num_words_read, available_write_bw, num_words_generated, num_words_written, num_words_stored = performance_numbers
        print(f"latency: {latency}, read bw: {needed_read_bw}, write bw: {available_write_bw}, num words read: {num_words_read}, num words generated: {num_words_generated}, num words written: {num_words_written}, num words stored: {num_words_stored}")
        print()
    
    additional_write_cycles, write_bw, words_written_per_cycle, spillover_words, directly_written_words = additional_write_data
    print(f"additional write cycles: {additional_write_cycles}, write bw: {write_bw}, words written per cycle: {words_written_per_cycle}, spillover words: {spillover_words}, directly written words: {directly_written_words}")
    print()
    print(f"total step latency: {total_step_latency}")
    
    print()
    print()
    print()
    print()


    # Create the schedule 
    
    exit()

    
    
    ### old testing code ###
    
    
    # Parameter sweep ranges
    num_vars_range = range(20, 21)
    zerocheck_pe_unroll_factors = [2**i for i in range(5)]  # 2^1 to 2^4
    mle_update_unroll_factors = [2**i for i in range(5)]  # 2^1 to 2^4

    # Example fixed values
    zerocheck_pe_latency = 61

    mle_update_pe_latency = 10
    transcript_latency = 36
    freq = 1e9
    bitwidth = 255

    modmuls_per_mle_update_pe = 1
    modmuls_per_zerocheck_core = 117

    # placeholder values
    permcheck_pe_latency = 20
    opencheck_pe_latency = 20
    modmuls_per_permcheck_core = 40
    modmuls_per_opencheck_core = 40

    # mm^2
    modmul_area = 0.478

    # Prepare a list to store the results
    results = []

    # Perform the parameter sweep for zerocheck
    for num_vars in num_vars_range:
        for num_zerocheck_core_pes in zerocheck_pe_unroll_factors:
            for pes_per_mle_update in mle_update_unroll_factors:
                latency, runtime, zc_peak_bw_util, mu_peak_bw_util, num_modmuls = get_sumcheck_latency(
                    num_vars,
                    zerocheck_pe_latency,
                    num_zerocheck_core_pes,
                    modmuls_per_zerocheck_core,
                    mle_update_pe_latency,
                    pes_per_mle_update,
                    modmuls_per_mle_update_pe,
                    transcript_latency,
                    freq,
                    bitwidth
                )
                # Store the result in the format (num_vars, pe_unroll_factor, mem_update_unroll_factor, latency)
                results.append((num_vars, num_zerocheck_core_pes, pes_per_mle_update, latency, runtime, zc_peak_bw_util, mu_peak_bw_util, num_modmuls))



    # Print the results for Zerocheck sweep
    print("ZC BW = zerocheck bandwidth utilization")
    print("MU BW = mle update bandwidth utilization")
    for result in results:
        print(f"num_vars: {int(result[0]):2},  zerocheck PEs: {int(result[1]):2},  "
            f"PEs per mle update: {int(result[2]):2},  latency (cycles): {int(result[3]):7},  "
            f"runtime (ms): {result[4]:.3f},  ZC BW (GB/s): {result[5]:8.2f},  "
            f"MU BW (GB/s): {result[6]:9.2f},  num_modmuls = {int(result[7]):5},  "
            f"modmul_area = {result[7]*modmul_area:8.2f}")
    print()


    num_vars = 24
    mem_amt = get_max_mem_amt(bitwidth, "input_mles", num_vars)
    print(round(mem_amt, 2), "MB")


