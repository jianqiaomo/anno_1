
import math
import numpy as np

BITS_PER_GB = 1<<33
BITS_PER_MB = 1<<23


# key assumptions
# 1. streaming architecture - only nominal amount of SRAM provisioned for queuing data
# 2. modeling up to 2^20 variables
# 3. input MLEs can be stored on chip
# 4. generated full MLEs and intermediate MLEs will be stored in off-chip memory
# 5. latency model and peak bandwidth utilization is unaffected by whether or not
#    MLEs are stored on chip or off-chip because kernels are fully pipelined and
#    we are streaming data 


# maximum memory required for each kernel

def get_max_mem_amt(bitwidth, kernel, num_vars, num_mles=None):

    scale_factor = bitwidth/BITS_PER_MB

    packed_enables = 4*(1 << num_vars)/BITS_PER_MB
    address_translation_unit = (1 << num_vars)*(1 + num_vars)/BITS_PER_MB
    ones_zeros_table = round(0.9*(1 << num_vars))/BITS_PER_MB
    full_width_table = round(0.1*(1 << num_vars)*bitwidth)/BITS_PER_MB
    permutation_table = ((1 << num_vars) * (num_vars + 2))/BITS_PER_MB

    total_input_mle_storage = packed_enables + 4*(address_translation_unit + ones_zeros_table + full_width_table) + 3*permutation_table

    if kernel in ("zerocheck_core", "permcheck_core", "opencheck_core"):
        return (1 << (num_vars - 1))*scale_factor*num_mles
    
    elif kernel == "mle_update":
        return (1 << num_vars)*scale_factor*num_mles
    
    # extra memory storage beyond the input
    elif kernel == "mle_combine":
        num_evals = 1 << num_vars
        phi_and_pi = num_evals*scale_factor*2
        return phi_and_pi

    elif kernel == "input_mles":
        return total_input_mle_storage

# this is peak bandwidth
def sumcheck_core_bw_utilization(num_pes, num_mles, bitwidth, freq):
    num_words_read_per_cycle_per_pe = 2*num_mles
    total_num_words_read_per_cycle = num_words_read_per_cycle_per_pe*num_pes
    peak_bw_util = total_num_words_read_per_cycle*freq*bitwidth/BITS_PER_GB
    return peak_bw_util

# this currently assumes infinite bandwidth
# all MLEs must be accessed in parallel in each PE
def sumcheck_core_latency(entries_per_mle, pe_latency, reduction_latency, num_pes, queue_depth=0):

    if entries_per_mle/(2*num_pes) < 1:
        compute_cycles = pe_latency + queue_depth
    else:
        compute_cycles = entries_per_mle/(2*num_pes) + pe_latency + queue_depth - 1

    # this is an upper bound
    pe_reduction_latency = int(math.ceil(math.log2(num_pes)))

    total_cycles = compute_cycles + reduction_latency + pe_reduction_latency

    return total_cycles

def mle_update_bw_utilization(pes_per_mle_update, bitwidth, freq, num_mles_in_parallel):

    num_words_read_per_cycle_per_pe = 2*num_mles_in_parallel
    num_words_written_per_cycle_per_pe = num_mles_in_parallel

    total_num_words_xferred_per_cycle = (num_words_read_per_cycle_per_pe + num_words_written_per_cycle_per_pe)*pes_per_mle_update
    peak_bw_util = total_num_words_xferred_per_cycle*freq*bitwidth/BITS_PER_GB
    return peak_bw_util

# this is the latency approximation for 1 MLE. given K MLEs we need to update, the total latency
# is K/num_parallel_mles. This is handled by the invoking function
# num_pes is effectively how many modmuls are allocated for this singular MLE update
def mle_update_latency(entries_per_mle, pe_latency, num_pes, queue_depth=0):
    
    if entries_per_mle/(2*num_pes) < 1:
        total_cycles = pe_latency + 2*queue_depth
    else:
        total_cycles = entries_per_mle/(2*num_pes) + pe_latency + 2*queue_depth - 1

    return total_cycles

def sumcheck_latency(num_vars, sumcheck_pe_latency, sumcheck_reduction_latency, num_sumcheck_core_pes, modmuls_per_sumcheck_core, mle_update_pe_latency, \
    pes_per_mle_update, modmuls_per_mle_update, transcript_latency, freq, bitwidth, total_required_mles=9, num_mles_in_parallel=9):

    entries_per_mle = 1 << num_vars
    total_latency = 0

    scale_factor = math.ceil(total_required_mles/num_mles_in_parallel)
    
    accumulated_sc_latency = 0
    accumulated_mu_latency = 0
    accumulated_tc_latency = 0
    # pes_per_mle_update means for a single MLE update, how many parallel accesses are there
    total_modmuls = num_mles_in_parallel*pes_per_mle_update*modmuls_per_mle_update + num_sumcheck_core_pes*modmuls_per_sumcheck_core

    for i in range(num_vars - 1):
        sc_latency = sumcheck_core_latency(entries_per_mle, sumcheck_pe_latency, sumcheck_reduction_latency, num_sumcheck_core_pes)
        mu_latency = mle_update_latency(entries_per_mle, mle_update_pe_latency, pes_per_mle_update)

        accumulated_sc_latency += sc_latency
        accumulated_mu_latency += (mu_latency*scale_factor)
        accumulated_tc_latency += transcript_latency
        
        total_latency += sc_latency + transcript_latency + (mu_latency*scale_factor)
        entries_per_mle >>= 1
    
    sc_latency = sumcheck_core_latency(entries_per_mle, sumcheck_pe_latency, sumcheck_reduction_latency, num_sumcheck_core_pes)
    total_latency += sc_latency + transcript_latency
    accumulated_sc_latency += sc_latency
    accumulated_tc_latency += transcript_latency
    
    # runtime in ms
    total_runtime = total_latency/freq*1000
    
    sc_peak_bw_util = sumcheck_core_bw_utilization(num_sumcheck_core_pes, total_required_mles, bitwidth, freq)
    mu_peak_bw_util = mle_update_bw_utilization(pes_per_mle_update, bitwidth, freq, num_mles_in_parallel)

    return total_latency, total_runtime, sc_peak_bw_util, mu_peak_bw_util, total_modmuls, accumulated_sc_latency, accumulated_mu_latency, accumulated_tc_latency


# 13 to 6
# each pe reads in from all MLEs, 2 of which are stored in HBM, rest on chip
def mle_combine_bw_utilization(num_pes, bitwidth, freq):
    num_words_read_per_cycle_per_pe = 2

    # this might be just 5, need to verify
    num_words_written_per_cycle_per_pe = 5

    total_num_words_xferred_per_cycle = (num_words_read_per_cycle_per_pe + num_words_written_per_cycle_per_pe)*num_pes
    peak_bw_util = total_num_words_xferred_per_cycle*freq*bitwidth/BITS_PER_GB
    return peak_bw_util

def mle_combine_latency(num_vars, pe_latency, num_pes, queue_depth=0):
    entries_per_mle = 1<<num_vars
    return entries_per_mle/num_pes + pe_latency + queue_depth - 1

# each pe reads in from all 6 MLEs, 5 (?) of which are stored in HBM
def build_g_prime_bw_utilization(num_pes, bitwidth, freq, exclude_write_bw=True):
    
    # this might be just 5, need to verify
    num_words_read_per_cycle_per_pe = 5
    if exclude_write_bw:
        num_words_written_per_cycle_per_pe = 0
    else:
        num_words_written_per_cycle_per_pe = 1
    
    total_num_words_xferred_per_cycle = (num_words_read_per_cycle_per_pe + num_words_written_per_cycle_per_pe)*num_pes
    peak_bw_util = total_num_words_xferred_per_cycle*freq*bitwidth/BITS_PER_GB
    return peak_bw_util

# 6 MLEs are summed into 1. assume we compute on all at the same time
def build_g_prime_latency(num_vars, pe_latency, num_pes, queue_depth=0):
    entries_per_mle = 1<<num_vars
    return entries_per_mle/num_pes + pe_latency + queue_depth - 1

if __name__ == "__main__":

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

    modmuls_per_mle_update = 1
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
                latency, runtime, zc_peak_bw_util, mu_peak_bw_util, num_modmuls = sumcheck_latency(
                    num_vars,
                    zerocheck_pe_latency,
                    num_zerocheck_core_pes,
                    modmuls_per_zerocheck_core,
                    mle_update_pe_latency,
                    pes_per_mle_update,
                    modmuls_per_mle_update,
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
