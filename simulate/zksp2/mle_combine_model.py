from .util import *

def mle_combine_model(num_vars, num_witnesses, num_selectors, words_per_mle_range, supplemental_data):

    bitwidths, available_bw, freq = supplemental_data

    bits_per_dense_element, avg_bits_per_witness_word, bits_per_permutation_element = bitwidths

    num_witnesses += 1  # include qc
    num_selectors -= 1  # exclude qc

    mle_combine_model_data = dict()

    for words_per_mle in words_per_mle_range:
        
        num_rounds = (1<<num_vars) / words_per_mle

        # for MLE combine, we're just assuming 2 elements per MLE
        num_ports_per_mle = 2
        max_read_rate = calc_rate(bits_per_dense_element, available_bw, freq) # "reading into" RF (actually a write operation)
        
        # phase 1 - fetch phi and pi, compute y1, y2, y3, y4
        num_mles_fetched = 2
        desired_read_rate = num_ports_per_mle*num_mles_fetched # can read up to these many words per cycle
        num_words_read = num_mles_fetched*words_per_mle
        actual_read_rate = min(desired_read_rate, max_read_rate)
        needed_read_bw = calc_bw(bits_per_dense_element, actual_read_rate, freq)
        phase_1_latency = math.ceil(num_words_read/actual_read_rate)

        # phase 2 - fetch sigmas and selectors
        num_mles_fetched = num_selectors + num_witnesses
        desired_read_rate = num_ports_per_mle*num_mles_fetched
        
        scale_factor = (1 + bits_per_permutation_element)/bits_per_dense_element
        num_words_read = num_mles_fetched*words_per_mle*scale_factor
        actual_read_rate = min(desired_read_rate, max_read_rate)
        needed_read_bw = calc_bw(bits_per_dense_element, actual_read_rate, freq)
        phase_2_latency = math.ceil(num_words_read/actual_read_rate)

        # phase 3 - fetch witnesses
        num_mles_fetched = num_witnesses
        desired_read_rate = num_ports_per_mle*num_mles_fetched

        scale_factor = (avg_bits_per_witness_word + math.log2(words_per_mle))/bits_per_dense_element
        num_words_read = num_mles_fetched*words_per_mle*scale_factor
        actual_read_rate = min(desired_read_rate, max_read_rate)
        needed_read_bw = calc_bw(bits_per_dense_element, actual_read_rate, freq)
        phase_3_latency = math.ceil(num_words_read/actual_read_rate)

        # ideally, by this point we have computed all 6 MLEs for from 1 stripe of data from all input MLEs

        round_latency = phase_1_latency + phase_2_latency + phase_3_latency
        total_latency = num_rounds*round_latency

        total_data_pi_phi_gb = 2*(1<<num_vars)*bits_per_dense_element/BITS_PER_GB
        total_data_selectors_gb = num_selectors*(1<<num_vars)*1/BITS_PER_GB
        total_data_witnesses_gb = num_witnesses*(1<<num_vars)*avg_bits_per_witness_word/BITS_PER_GB
        total_data_sigmas_gb = num_witnesses*(1<<num_vars)*bits_per_permutation_element/BITS_PER_GB
        total_data_read_gb = total_data_pi_phi_gb + total_data_selectors_gb + total_data_witnesses_gb + total_data_sigmas_gb

        avg_read_bw = total_data_read_gb/total_latency*freq

        available_write_bw = available_bw - avg_read_bw
        if available_write_bw < 0:
            available_write_bw = 0

        total_data_to_write_gb = 6*(1<<num_vars)*bits_per_dense_element/BITS_PER_GB

        writeable_data_gb = available_write_bw*total_latency/freq

        remaining_data_gb = total_data_to_write_gb - writeable_data_gb
        if remaining_data_gb > 0:
            additional_latency = math.ceil(remaining_data_gb/available_bw*freq)
        else:
            additional_latency = 0
        
        total_latency += additional_latency

        total_mle_combine_data_gb = total_data_read_gb + total_data_to_write_gb
        average_mle_combine_bw = total_mle_combine_data_gb/total_latency*freq

        mle_combine_model_data[words_per_mle] = {
            "total_latency": total_latency,
            "average_mle_combine_bw": average_mle_combine_bw,
            "phase_1_latency": phase_1_latency,
            "phase_2_latency": phase_2_latency,
            "phase_3_latency": phase_3_latency,
        }
    
    return mle_combine_model_data




