from .get_msm_data import adjust_loading_bw_polyopen, scale_bw_opencheck, stitch_msms
from .params import *
import math

def polyopen_sweep(num_vars_range, dense_msm_data_dict, point_merge_stats, metadata):

    core_keys = ["single_core", "dual_core_opencheck"]
    point_merge_cycles, point_merge_area_stats = point_merge_stats

    bits_per_scalar_adjusted, bits_per_scalar, bits_per_point_reduced, available_bw, freq = metadata
    polyopen_data = dict()
    for num_vars in num_vars_range:
        point_merge_latency = point_merge_cycles[num_vars]*math.ceil(num_vars/modmuls_for_mle_combine) # scale up because we need num_vars modmuls but have only 12
        polyopen_data[num_vars] = dict()
        for core_key in core_keys:
            polyopen_data[num_vars][core_key] = dict()
            
            num_cores = 1 if core_key == "single_core" else 2
            
            for design, data in dense_msm_data_dict.items():
                
                (num_vars_in_design, fraction_ones, number_suffix, half_number_suffix, fraction_dense), target_ws, target_ppw, target_ocw, target_qd, target_ii, padd_latency = design
                
                # we have to start with half the MSM length
                if num_vars_in_design != num_vars - 1:
                    continue

                dense_design = (target_ws, target_ppw, target_ocw, target_qd, target_ii)

                msm_dict = data[core_key]
                total_dense_latency = msm_dict["total_latency"]
                dense_latency_stats = msm_dict["latency_stats"]
                dense_bw_stats = msm_dict["bw_stats"]
                first_block_size = msm_dict["first_block_size"]
                fill_rate_per_msm = msm_dict["fill_rate"]
                msm_area_data = msm_dict['area_stats']

                debug = False
                sub_debug = False

                if core_key == "dual_core_opencheck":
                    dense_bw_stats = [i*2 for i in dense_bw_stats]
                    first_block_size *= 2

                if debug:
                    print(f"core_key: {core_key}")
                    print(f"num_vars: {num_vars}")
                    print(f"total_dense_latency: {total_dense_latency}")
                    print(f"dense_latency_stats: {dense_latency_stats}")
                    print(f"dense_bw_stats: {dense_bw_stats}")
                    print(f"first_block_size: {first_block_size}")
                    print(f"fill_rate_per_msm: {fill_rate_per_msm}")

                # scale bandwidth for round 1 assuming a fill rate of 1
                round_1_scale_factor = (bits_per_scalar_adjusted + bits_per_point_reduced) / (bits_per_scalar + bits_per_point_reduced)
                msm_trace = [total_dense_latency, dense_latency_stats, dense_bw_stats, first_block_size]

                # this gives us either the latency at the target bandwidth or approx. the best we can do
                # first, scale the bandwidth to what we would actually need, and then increase the latency as needed
                round_1_msm_trace, _ = scale_bw_opencheck(msm_trace, round_1_scale_factor, available_bw, debug=sub_debug)

                _, (new_loading_latency, *_), *_ = round_1_msm_trace

                # based on the new latency that we need, we want to have those many g' PEs available to generate
                # that much data
                effective_fill_rate = math.ceil(first_block_size / new_loading_latency)

                total_g_prime_cycles = (1 << num_vars)/effective_fill_rate

                # area stats
                g_prime_pes = 2*effective_fill_rate
                mle_update_pes = effective_fill_rate

                if debug:
                    print(f"effective_fill_rate: {effective_fill_rate}")

                round_2_scale_factor = (3*bits_per_scalar + bits_per_point_reduced) / (bits_per_scalar + bits_per_point_reduced)

                if debug:
                    print(f"round_1_scale_factor: {round_1_scale_factor}")
                    print(f"msm_trace: {msm_trace}")
                    print(f"round_1_msm_trace: {round_1_msm_trace}")
                    print(f"round_2_scale_factor: {round_2_scale_factor}")
                    print()

                polyopen_design = dense_design
                
                # compute the rest of the msm stitching
                full_msm_trace = round_1_msm_trace

                round_i_number_suffix = number_suffix
                round_i_half_number_suffix = half_number_suffix
                round_i_num_vars_in_design = num_vars_in_design

                for round_num in range(2, num_vars + 1):

                    # get ith round trace
                    round_i_number_suffix -= 1
                    round_i_half_number_suffix -= 1
                    round_i_num_vars_in_design -= 1

                    if round_i_number_suffix == 0:
                        round_i_half_number_suffix = 0

                    round_i_design = (round_i_num_vars_in_design, fraction_ones, round_i_number_suffix, round_i_half_number_suffix, fraction_dense), target_ws, target_ppw, target_ocw, target_qd, target_ii, padd_latency
                    round_i_dict = dense_msm_data_dict[round_i_design][core_key]

                    round_i_total_dense_latency = round_i_dict["total_latency"]
                    round_i_dense_latency_stats = round_i_dict["latency_stats"]
                    round_i_dense_bw_stats = round_i_dict["bw_stats"]
                    round_i_first_block_size = round_i_dict["first_block_size"]

                    if core_key == "dual_core_opencheck" and round_i_number_suffix > 0:
                        round_i_dense_bw_stats = [i*2 for i in round_i_dense_bw_stats]
                        round_i_first_block_size *= 2
                    
                    if debug:
                        print(f"round_{round_num}_total_dense_latency: {round_i_total_dense_latency}")
                        print(f"round_{round_num}_dense_latency_stats: {round_i_dense_latency_stats}")
                        print(f"round_{round_num}_dense_bw_stats: {round_i_dense_bw_stats}")
                        print(f"round_{round_num}_first_block_size: {round_i_first_block_size}")

                    round_i_msm_trace = [round_i_total_dense_latency, round_i_dense_latency_stats, round_i_dense_bw_stats, round_i_first_block_size]
                    
                    if debug:
                        print(f"round_{round_num}_msm_trace: {round_i_msm_trace}")
                    
                    round_i_msm_trace, _ = scale_bw_opencheck(round_i_msm_trace, round_2_scale_factor, available_bw)
                    
                    if debug:
                        print(f"round_{round_num}_msm_trace_adjusted: {round_i_msm_trace}")
                    
                    full_msm_trace = stitch_msms(full_msm_trace, round_i_msm_trace)

                    if debug:
                        print(f"full_msm_trace after round {round_num}: {full_msm_trace}")
                        print()

                    full_msm_total_latency, full_latency_stats, full_bw_stats, full_first_block_size = full_msm_trace
                
                num_g_prime_modmuls = g_prime_pes*modmuls_per_g_prime_pe
                num_mle_update_modmuls = mle_update_pes*modmuls_per_mle_update_pe
                point_merge_modmuls = point_merge_area_stats[num_vars]["num_modmuls"]
                
                g_prime_area = round(num_g_prime_modmuls*modmul_area, 3)
                mle_update_area = round(num_mle_update_modmuls*modmul_area, 3)
                point_merge_area = round(point_merge_modmuls*modmul_area, 3)
                
                min_num_modmuls = max([num_mle_update_modmuls, point_merge_modmuls, num_g_prime_modmuls])
                min_po_mle_area = min_num_modmuls*modmul_area

                polyopen_mle_area_data = {
                    "g_prime_pes"            : g_prime_pes,
                    "mle_update_pes"         : mle_update_pes,
                    "g_prime_area"           : g_prime_area,
                    "mle_update_area"        : mle_update_area,
                    "point_merge_area"       : point_merge_area,
                    "num_g_prime_modmuls"    : num_g_prime_modmuls,
                    "num_mle_update_modmuls" : num_mle_update_modmuls,
                    "point_merge_modmuls"    : point_merge_modmuls,
                    "min_num_modmuls"        : min_num_modmuls,
                    "min_po_mle_area"        : min_po_mle_area
                }

                if debug:
                    for k, v in polyopen_mle_area_data.items():
                        print(f"{k}: {v}")
                    print()
                
                polyopen_data[num_vars][core_key][polyopen_design] = {
                    "total_latency": full_msm_total_latency + point_merge_latency,
                    "dense_msm_total_latency": full_msm_total_latency,
                    "g_prime_latency": total_g_prime_cycles,
                    "point_merge_latency": point_merge_latency,
                    "dense_msm_latency_stats": full_latency_stats,
                    "dense_msm_bw_stats": full_bw_stats,
                    "first_block_size": full_first_block_size,
                    "msm_area_data": msm_area_data,
                    "mle_area_data": polyopen_mle_area_data
                }
    
    return polyopen_data

