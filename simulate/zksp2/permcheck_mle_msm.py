from .get_msm_data import adjust_loading_bw_permcheck, stitch_msms
from .params import *
from .step_permcheck import *
from .util import calc_bw
import math

def print_permcheck_model(result, prod_mle_data, print_area=False):
    print("Permcheck Model Results:")
    print(f"Start time ND out: {result['gen_nd_first_out_lat']}")
    print(f"End time ND out: {result['gen_nd_last_out_lat']}")
    print(f"Start time frac MLE out: {result['frac_mle_first_out_lat']}")
    print(f"End time frac MLE out: {result['frac_mle_last_out_lat']}")
    print(f"Start time prod MLE out: {result['product_mle_first_out_lat']}")
    print(f"End time prod MLE out: {result['product_mle_last_out_lat']}")
    print(f"Frac MLE end-to-end latency: {result['frac_mle_e2e_lat']}")
    print(f"ND bandwidth (GiB/s): {result['nd_bandwidth_GiB_per_s']}")
    print(f"Frac MLE bandwidth (GiB/s): {result['fracMLE_bandwidth_GiB_per_s']}")
    # print(f"Prod MLE bandwidth (GiB/s): {result['prodMLE_bandwidth_GiB_per_s']}")
    bits_per_element, permcheck_mle_units, freq = prod_mle_data
    print(f"Prod MLE bandwidth (GiB/s): {calc_bw(bits_per_element, permcheck_mle_units, freq)}")
    
    if not print_area:
        print()
        return

    print(f"SRAM size (KiB): {result['sram_size_KiB']}")
    print(f"Number of modular multiplications (ND): {result['gen_nd_num_mod_muls']}")
    print(f"Number of modular additions (ND): {result['gen_nd_num_mod_adds']}")
    print(f"Number of registers (ND): {result['gen_nd_num_regs']}")
    print(f"Number of modular inversions: {result['num_mod_invs']}")
    print(f"Number of modular multiplications (Frac MLE excl. MT): {result['frac_mle_excl_mt_num_mod_muls']}")
    print(f"Number of registers (Frac MLE excl. MT): {result['frac_mle_excl_mt_num_regs']}")
    print(f"Number of modular multiplications (Frac MLE MT): {result['frac_mle_mt_num_mod_muls']}")
    print(f"Number of registers (Frac MLE MT): {result['frac_mle_mt_num_regs']}")
    print(f"Number of modular multiplications (Prod MLE): {result['product_mle_num_mod_muls']}")
    print(f"Number of registers (Prod MLE): {result['product_mle_num_regs']}")
    print()

def print_sweep_data(sweep_data):
    for k, v in sweep_data.items():
        print(f"{k}: {v}")
    print()

# get the area and bandwidth data for the permcheck MLE units
# TODO: add in the input reading functionality here
def get_permcheck_mle_data(num_vars_range, permcheck_mle_units_range, bitwidth_data, freq, available_bw, gate_type="vanilla"):
    bits_per_element, witness_bitwidth = bitwidth_data
    num_witness_pairs = 3 if gate_type == "vanilla" else 5
    sweep_data = dict()
    for num_vars in num_vars_range:
        sweep_data[num_vars] = dict()
        for permcheck_mle_units in permcheck_mle_units_range:
        
            # Run the model with current parameters
            result = step_permcheck_model(permcheck_mle_units, permcheck_mle_units, 
                num_witness_pairs=num_witness_pairs,
                num_vars=num_vars,
                assume_onchip_storage=False,
                witness_bitwidth=witness_bitwidth,
                onchip_mle_size=onchip_mle_size,
                bitwidth=bits_per_element
            )

            debug = False
            prod_mle_data = (bits_per_element, permcheck_mle_units, freq)
            if debug:
                print_permcheck_model(result, prod_mle_data)

            start_time_nd_out = result['gen_nd_first_out_lat']
            end_time_nd_out = result['gen_nd_last_out_lat']
            start_time_frac_mle_out = result['frac_mle_first_out_lat']
            end_time_frac_mle_out = result['frac_mle_last_out_lat']
            start_time_prod_mle_out = result['product_mle_first_out_lat']
            end_time_prod_mle_out = result['product_mle_last_out_lat']
            frac_mle_e2e_lat = result['frac_mle_e2e_lat']

            nd_cycles = int(end_time_nd_out - start_time_nd_out)
            frac_mle_cycles = int(end_time_frac_mle_out - start_time_frac_mle_out)
            prod_mle_cycles = int(end_time_prod_mle_out - start_time_prod_mle_out)
            full_permcheck_mle_cycles = int(end_time_prod_mle_out)

            nd_bw = result['nd_bandwidth_GiB_per_s']
            frac_bw = result['fracMLE_bandwidth_GiB_per_s']
            # prod_bw = result['prodMLE_bandwidth_GiB_per_s'] <-- this is peak bw
            prod_bw = calc_bw(bits_per_element, permcheck_mle_units, freq)

            # define the latencies
            block1_lat = end_time_nd_out - start_time_prod_mle_out
            block2_lat = frac_mle_e2e_lat
            block3_lat = end_time_prod_mle_out - end_time_frac_mle_out
            
            # convert to seconds
            block1_lat_s = block1_lat / freq
            block2_lat_s = block2_lat / freq
            block3_lat_s = block3_lat / freq

            # convert to GB
            block1_data = block1_lat_s * (nd_bw + frac_bw + prod_bw)
            block2_data = block2_lat_s * (frac_bw + prod_bw)
            block3_data = block3_lat_s * prod_bw

            overlapped_data_transferred = block1_data + block2_data + block3_data
            overlapped_transfer_duration = block1_lat + block2_lat + block3_lat

            peak_mle_bw = nd_bw + frac_bw + prod_bw

            nd_lat_s = nd_cycles / freq
            frac_mle_lat_s = frac_mle_cycles / freq
            prod_mle_lat_s = prod_mle_cycles / freq

            nd_data = nd_lat_s * nd_bw
            frac_mle_data = frac_mle_lat_s * frac_bw
            prod_mle_data = prod_mle_lat_s * prod_bw

            total_data_transferred = nd_data + frac_mle_data + prod_mle_data

            # TODO: make this estimate more accurate
            if peak_mle_bw > available_bw:
                print(f"Permcheck with {permcheck_mle_units} frac MLE units exceeds the bandwidth")
                # continue

                # if the bandwidth is exceeded, then we will simply extend the overlap transfer duration
                # to be the time to transfer all the data. this will be a time penalty for the MLE ops
                # the penalty for MSM overlapped with MLE ops is accounted for downstream

                min_transfer_cycles = total_data_transferred / available_bw * freq
                time_penalty = min_transfer_cycles - full_permcheck_mle_cycles
                assert time_penalty > 0

                full_permcheck_mle_cycles = min_transfer_cycles
            else:
                time_penalty = 0
                peak_mle_bw = available_bw


            sweep_data[num_vars][permcheck_mle_units] = dict()

            peak_mle_bw = round(peak_mle_bw, 3)

            # area data
            frac_mle_sram_mb = round(result['sram_size_KiB']/1024, 3)
            frac_mle_sram_area = round(frac_mle_sram_mb*MB_CONVERSION_FACTOR, 3)
            
            nd_modmuls = result["gen_nd_num_mod_muls"]
            nd_modadds = result["gen_nd_num_mod_adds"]
            nd_numregs = result["gen_nd_num_regs"]
            num_modinvs = result['num_mod_invs']

            mle_modinv_area = round(modinv_area*num_modinvs, 3)
            
            frac_mle_modmuls = result["frac_mle_excl_mt_num_mod_muls"]
            frac_mle_numregs = result["frac_mle_excl_mt_num_regs"]
            
            frac_mle_mul_tree_modmuls = result["frac_mle_mt_num_mod_muls"]
            frac_mle_mul_tree_numregs = result["frac_mle_mt_num_regs"]
            
            prod_mle_modmuls = result["product_mle_num_mod_muls"]
            prod_mle_numregs = result["product_mle_num_regs"]

            nd_area = nd_modmuls*modmul_area + nd_modadds*modadd_area + nd_numregs*reg_area
            nd_area = round(nd_area, 3)
            frac_mle_area = frac_mle_modmuls*modmul_area + frac_mle_numregs*reg_area + mle_modinv_area
            frac_mle_area = round(frac_mle_area, 3)

            permcheck_mle_logic_area_breakdown = {
                "nd_area": nd_area,
                "frac_mle_area": frac_mle_area,
                "frac_mle_mul_tree_modmuls": frac_mle_mul_tree_modmuls,
                "frac_mle_mul_tree_numregs": frac_mle_mul_tree_numregs,
                "prod_mle_modmuls": prod_mle_modmuls,
                "prod_mle_numregs": prod_mle_numregs
            }
            
            updated_mle_area_data = {
                "permcheck_mle_logic_area_breakdown" : permcheck_mle_logic_area_breakdown,
                "frac_mle_sram_area" : frac_mle_sram_area,
                "frac_mle_sram_mb" : frac_mle_sram_mb
            }

            sweep_data[num_vars][permcheck_mle_units] = {
                "peak_mle_bw": peak_mle_bw,
                "total_mle_cycles": full_permcheck_mle_cycles,
                "time_penalty": time_penalty,
                "overlapped_data_transferred": overlapped_data_transferred,
                "total_data_transferred": total_data_transferred,
                "initial_cycles": start_time_prod_mle_out,
                "msm_overlap_cycles": overlapped_transfer_duration,
                "nd_cycles" : nd_cycles,
                "frac_mle_cycles" : frac_mle_cycles,
                "prod_mle_cycles" : prod_mle_cycles,
                "updated_mle_area_data": updated_mle_area_data
            }

            if debug:
                print_sweep_data(sweep_data[num_vars][permcheck_mle_units])

    return sweep_data

def print_permcheck_mle_msm_data(data):
    for k, v in data.items():
        print(f"{k}: {v}")
    print("-------------------------------------------------------------")
    print()

def permcheck_mle_msm_sweep(num_vars_range, load_rate_range, permcheck_mle_sweep_data, dense_msm_data_dict, metadata):

    bits_per_point_reduced, available_bw, freq, no_duplicated_points = metadata

    core_keys = ["single_core", "dual_core_permcheck"]

    permcheck_mle_msm_data = dict()
    for num_vars in num_vars_range:
        permcheck_mle_msm_data[num_vars] = dict()
        for core_key in core_keys:
            permcheck_mle_msm_data[num_vars][core_key] = dict()
            for load_rate in load_rate_range:
                for design, data in dense_msm_data_dict.items():
            
                    debug = False
                    sub_debug = False

                    (num_vars_in_design, fraction_ones, number_suffix, half_number_suffix, fraction_dense), target_ws, target_ppw, target_ocw, target_qd, target_ii, _ = design

                    if num_vars_in_design != num_vars:
                        continue
                    
                    dense_design = (target_ws, target_ppw, target_ocw, target_qd, target_ii)
                    permcheck_mle_msm_design = (load_rate, dense_design)

                    if load_rate not in permcheck_mle_sweep_data[num_vars]:
                        continue

                    mle_dict = permcheck_mle_sweep_data[num_vars][load_rate]

                    msm_overlap_cycles = mle_dict["msm_overlap_cycles"]
                    initial_mle_cycles = mle_dict["initial_cycles"]
                    overlapped_permcheck_data_written = mle_dict["overlapped_data_transferred"]
                    total_permcheck_data_written = mle_dict["total_data_transferred"]
                    mle_total_cycles = mle_dict["total_mle_cycles"]
                    permcheck_mle_time_penalty = mle_dict["time_penalty"]
                    peak_mle_bw = mle_dict["peak_mle_bw"]
                    mle_area_data = mle_dict["updated_mle_area_data"]

                    nd_cycles = mle_dict["nd_cycles"]
                    frac_mle_cycles = mle_dict["frac_mle_cycles"]
                    prod_mle_cycles = mle_dict["prod_mle_cycles"]


                    msm_dict = data[core_key]
                    total_dense_latency = msm_dict["total_latency"]
                    dense_latency_stats = msm_dict["latency_stats"]
                    dense_bw_stats = msm_dict["bw_stats"]
                    first_block_size = msm_dict["first_block_size"]
                   
                    # make sure the two core model is correct here
                    loading_latency, compute_latency, last_latency = dense_latency_stats
                    loading_bw, avg_bw, peak_bw = dense_bw_stats

                    # if this is true, then after mle is done, average msm bandwidth is the dominant bandwidth utilization
                    assert compute_latency + last_latency > msm_overlap_cycles

                    if debug:
                        print(f"core_type: {core_key}")
                        print(f"total_dense_latency: {total_dense_latency}")
                        print(f"dense latency stats: {dense_latency_stats}")
                        print(f"dense bw stats: {dense_bw_stats}")
                        print(f"first block size: {first_block_size}")
                        print()

                    msm_trace = total_dense_latency, dense_latency_stats, dense_bw_stats, first_block_size

                    # construct the msm trace based on the fact that we are loading only points, not scalars, from off-chip memory
                    # TODO: make this estimate more accurate
                    # TODO: What if the permcheck MLE rate > rate acceptable by MSM unit?
                    if core_key == "single_core":
                        adjusted_msm_trace = adjust_loading_bw_permcheck(msm_trace, load_rate, bits_per_point_reduced, available_bw, freq, debug=sub_debug)

                        # stitch the two traces together
                        full_msm_trace = stitch_msms(adjusted_msm_trace, msm_trace)
                        
                        new_msm_total_latency, new_latency_stats, new_bw_stats, first_block_size = full_msm_trace

                    elif core_key == "dual_core_permcheck":

                        adjusted_msm_trace = adjust_loading_bw_permcheck(msm_trace, load_rate, bits_per_point_reduced, available_bw / 2, freq, debug=sub_debug)
                        new_msm_total_latency, new_latency_stats, new_bw_stats, first_block_size = adjusted_msm_trace
                        
                        # account for pts bw
                        if no_duplicated_points:
                            new_bw_stats = [new_bw_stats[0], new_bw_stats[1]*dual_core_permcheck_scale_factor, new_bw_stats[-1]*dual_core_permcheck_scale_factor] 
                        else:
                            new_bw_stats = [i*2 for i in new_bw_stats]

                        first_block_size *= 2

                        full_msm_trace = new_msm_total_latency, new_latency_stats, new_bw_stats, first_block_size

                    new_loading_latency = new_latency_stats[0]
                    new_loading_bw = new_bw_stats[0]

                    # up to this point the MSM trace is still fairly ideal
                    
                    # total latency will be the sum of the initial MLE latency, the dense MSM latency, and the time penalty from overlapping MLE and MSM
                    if permcheck_mle_time_penalty == 0:

                        msm_data_read_during_overlap = new_loading_bw*new_loading_latency/freq + avg_bw*(msm_overlap_cycles-new_loading_latency)/freq

                        max_data_transferrable = available_bw*msm_overlap_cycles/freq
                        excess_data = overlapped_permcheck_data_written + msm_data_read_during_overlap - max_data_transferrable

                        # this means that in the overlap duration, we saturate the bandwidth
                        # after overlap duration, we are still utilizing bandwidth for the msm
                        # the remaining bandwidth available is what we use for fetching the remaining data
                        if excess_data > 0:
                            remaining_bw = available_bw - avg_bw
                            time_penalty = math.ceil((excess_data / remaining_bw) * freq)
                        else:
                            time_penalty = 0
                        
                        total_latency = initial_mle_cycles + new_msm_total_latency + time_penalty
                    
                    # we can only run the MSM after the permcheck MLEs are fully constructed. in this time, we've loaded the first MSM block
                    # of scalars
                    else:
                        total_latency = mle_total_cycles + new_msm_total_latency
                        time_penalty = permcheck_mle_time_penalty

                    if debug:
                        print(f"new msm total latency: {new_msm_total_latency}")
                        print(f"new latency stats: {new_latency_stats}")
                        print(f"new bw stats: {new_bw_stats}")
                        print(f"first block size for {core_key}: {first_block_size}")
                        print(f"overlapped permcheck data written: {overlapped_permcheck_data_written}")
                        print(f"msm data read during overlap: {msm_data_read_during_overlap}")
                        print(f"max data transferrable: {max_data_transferrable}")
                        print(f"excess data: {excess_data}")
                        print()


                    msm_area_data = msm_dict['area_stats']
           
                    permcheck_mle_msm_data[num_vars][core_key][permcheck_mle_msm_design] = {
                        "total_latency": total_latency,
                        "initial_mle_cycles": initial_mle_cycles,
                        "dense_msm_total_latency": new_msm_total_latency,
                        "time_penalty": time_penalty,
                        "mle_msm_overlap_cycles": msm_overlap_cycles, 
                        "mle_total_latency": mle_total_cycles,
                        "nd_cycles" : nd_cycles,
                        "frac_mle_cycles" : frac_mle_cycles,
                        "prod_mle_cycles" : prod_mle_cycles,
                        "dense_msm_latency_stats": new_latency_stats,
                        "dense_msm_bw_stats": new_bw_stats,
                        "peak_mle_bw": peak_mle_bw,
                        "mle_area_data": mle_area_data,
                        "msm_area_data": msm_area_data
                    }

                    if debug:
                        print_permcheck_mle_msm_data(permcheck_mle_msm_data[num_vars][core_key][permcheck_mle_msm_design])
                        
    
    return permcheck_mle_msm_data
