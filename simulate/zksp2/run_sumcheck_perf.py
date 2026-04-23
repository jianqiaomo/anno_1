from .sumcheck_models import *

sumcheck_polynomial = [
        ["q1", "w1", "fz"],
        ["q2", "w2", "fz"],
        ["q3", "w3", "fz"],
        ["qM", "w1", "w2", "fz"],
        ["qc", "fz"],
    ]
sumcheck_type = "zerocheck"

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
#     ["qECC", "w1", "w2", "w3", "w4", "fz"],
#     ["qc", "fz"],
# ]
# sumcheck_type = "zerocheck"

# sumcheck_polynomial = [
#     ["pi", "fz"],
#     ["p1", "p2", "fz"],
#     ["phi", "d1", "d2", "d3", "d4", "d5", "fz"],
#     ["n1", "n2", "n3", "n4", "n5", "fz"],
# ]
# sumcheck_type = "permcheck"

# sumcheck_polynomial = [
#     ["pi", "fz"],
#     ["p1", "p2", "fz"],
#     ["phi", "d1", "d2", "d3", "d4", "d5", "fz"],
#     ["n1", "n2", "n3", "n4", "n5", "fz"],
# ]
    
# sumcheck_type = "permcheck"

# sumcheck_polynomial = [
#     ["y1", "fz1"],
#     ["y2", "fz2"],
#     ["y3", "fz3"],
#     ["y4", "fz4"],
#     ["y5", "fz5"],
#     ["y6", "fz6"],
# ]
# sumcheck_type = "opencheck"


bits_per_element = 255
num_vars = 24
num_pes = 16
num_eval_engines = 7
num_product_lanes = 5
mle_update_latency = 20
extensions_latency = 5
modmul_latency = 20
available_bw = 2048 # GB/s
freq = 1e9
onchip_mle_size = 16384
sumcheck_hardware_params = num_pes, num_eval_engines, num_product_lanes, mle_update_latency, extensions_latency, modmul_latency, onchip_mle_size
percent_sparse = 0.9
avg_bits_per_witness_word = bits_per_element*(1 - percent_sparse) + 1*(percent_sparse)

skip_fraction_dict_1 = {
    "q1": 0.56,
    "q2": 0.72,
    "qM": 0.04,
    "q3": 0.01,
    "qc": 0.90
}
skip_fraction_dict_2 = {
    "q1": 0.42,
    "q2": 0.64,
    "qM": 0.06,
    "q3": 0.02,
    "qc": 0.90
}

skip_fraction_dict_3 = {
    "q1": 0.23,
    "q2": 0.28,
    "qM": 0.28,
    "q3": 0.00,
    "qc": 0.90
}

skip_fraction_dict_4 = {
    "q1": 0.02,
    "q2": 0.9,
    "qM": 0.01,
    "q3": 0.64,
    "qc": 0.90
}

skip_fraction_dict = {
    "q1": 0.99,
    "q2": 0.99,
    "q3": 0.99,
    "q4": 0.99,
    "q5": 0.99,
    "qM1": 0.99,
    "qM2": 0.99,
    "qH1": 0.99,
    "qH2": 0.99,
    "qH3": 0.99,
    "qH4": 0.99,
    "qECC": 0.99,
    "qc": 0.00,
}

zero_skip_fraction_dict = {
    "q1":   0,
    "q2":   0,
    "q3":   0,
    "qM":   0,
    "qc":   0,
    "q4":   0,
    "q5":   0,
    "qM1":  0,
    "qM2":  0,
    "qH1":  0,
    "qH2":  0,
    "qH3":  0,
    "qH4":  0,
    "qECC": 0,
}

supplemental_data = bits_per_element, available_bw, freq
debug = False
skip_fraction_dict = zero_skip_fraction_dict
sparsity_data = avg_bits_per_witness_word, skip_fraction_dict
results = performance_model(num_vars, sumcheck_polynomial, sumcheck_type, sumcheck_hardware_params, sparsity_data, supplemental_data, debug=debug)
print(results[0])


skip_fraction_dict = skip_fraction_dict_1
sparsity_data = avg_bits_per_witness_word, skip_fraction_dict
results = performance_model(num_vars, sumcheck_polynomial, sumcheck_type, sumcheck_hardware_params, sparsity_data, supplemental_data, debug=debug)
print(results[0])

skip_fraction_dict = skip_fraction_dict_2
sparsity_data = avg_bits_per_witness_word, skip_fraction_dict
results = performance_model(num_vars, sumcheck_polynomial, sumcheck_type, sumcheck_hardware_params, sparsity_data, supplemental_data, debug=debug)
print(results[0])

skip_fraction_dict = skip_fraction_dict_3
sparsity_data = avg_bits_per_witness_word, skip_fraction_dict
results = performance_model(num_vars, sumcheck_polynomial, sumcheck_type, sumcheck_hardware_params, sparsity_data, supplemental_data, debug=debug)
print(results[0])

skip_fraction_dict = skip_fraction_dict_4
sparsity_data = avg_bits_per_witness_word, skip_fraction_dict
results = performance_model(num_vars, sumcheck_polynomial, sumcheck_type, sumcheck_hardware_params, sparsity_data, supplemental_data, debug=debug)
print(results[0])

# old sumcheck model
# zerocheck_pe_latency = 35
# zerocheck_reduction_latency = 0
# num_zerocheck_core_pes = 2
# modmuls_per_zerocheck_core = 67
# mle_update_pe_latency = 10
# pes_per_mle_update = 4
# modmuls_per_mle_update_pe = 1
# transcript_latency = 0
# total_required_mles_zerocheck = 9
# num_mles_in_parallel_zc = 9
# zerocheck_latency, zerocheck_runtime, zc_peak_bw_util, zc_mu_peak_bw_util, zerocheck_modmuls, zc_accumulated_sc_latency, zc_accumulated_mu_latency, zc_accumulated_tc_latency = sumcheck_latency(
#     num_vars,
#     zerocheck_pe_latency,
#     zerocheck_reduction_latency,
#     num_zerocheck_core_pes,
#     modmuls_per_zerocheck_core,
#     mle_update_pe_latency,
#     pes_per_mle_update,
#     modmuls_per_mle_update_pe,
#     transcript_latency,
#     freq,
#     bits_per_element
#     , total_required_mles = total_required_mles_zerocheck 
#     , num_mles_in_parallel = num_mles_in_parallel_zc
# )

# print()
# print(f"Zerocheck Latency: {zerocheck_latency}, Peak Bandwidth Utilization: {zc_peak_bw_util}, MU Peak Bandwidth Utilization: {zc_mu_peak_bw_util}")
# print(f"Zerocheck Modmuls: {zerocheck_modmuls}")
# print(f"Accumulated SC Latency: {zc_accumulated_sc_latency}")
# print(f"Accumulated MU Latency: {zc_accumulated_mu_latency}")
# print(f"Accumulated TC Latency: {zc_accumulated_tc_latency}")
