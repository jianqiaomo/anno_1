import argparse
from .params import *
from .get_msm_data import *
from .permcheck_mle_msm_v1 import *
from .sumcheck_sweep_v1 import *
from .poly_opening_v1 import *
from .batch_eval import get_batch_eval_data_v1
from .get_point_merge_data import *
from .combine_msms_v1 import *
from .full_chip_sweep_v1 import *
from .mle_combine_model import *
import json
import time

if __name__ == "__main__":
  
    parser = argparse.ArgumentParser(description='Script Description')
    parser.add_argument('nv', type=int)
    parser.add_argument('available_bw', type=int)
    parser.add_argument('gate_type', type=str)
    parser.add_argument('--pickle', action='store_true')
    parser.add_argument('--design_point', type=str)
    parser.add_argument('--special_experiment', type=str)
    parser.add_argument('--skip_pareto', action='store_true')
    parser.add_argument('--skip_percent', type=int)
    parser.add_argument('--zc_mask', action='store_true')
    parser.add_argument('--point_duplication', action='store_true')

    args = parser.parse_args()
    print(args)
    
    zc_opt = args.zc_mask
    print(zc_opt)

    point_duplication = args.point_duplication
    if point_duplication:
        no_point_duplication = False
    else:
        no_point_duplication = True

    if args.skip_pareto != None:
        get_pareto_points = not args.skip_pareto
    else:
        get_pareto_points = True

    dump_data = args.pickle
    available_bw = args.available_bw
    if available_bw <= 512:
        # bad estimate for now
        hbm_area = (available_bw/512)*14.9
    
    else:
        hbm_area *= available_bw/1024
    
    design_point = args.design_point
    if args.design_point != None:
        print(f"Design point: {design_point}")
        # Extract values from the design point
        design_point_values = eval(design_point)  # Convert string to tuple
        print(design_point_values)
        window_size_in, ppw_in, ocw_in, core_type_in, fracmle_units_in, sumcheck_pes_in, eval_engines_in, product_lanes_in, onchip_mle_buf_size_in, pl_depth_val_in = (
            design_point_values[0][0],
            design_point_values[0][1],
            design_point_values[0][2],
            design_point_values[1],
            design_point_values[2],
            design_point_values[3][0],
            design_point_values[3][1],
            design_point_values[3][2],
            design_point_values[3][3],
            design_point_values[4]
        )

        print(f"Window size: {window_size_in}")
        print(f"PPW: {ppw_in}")
        print(f"OCW: {ocw_in}")
        print(f"Core type: {core_type_in}")
        print(f"FracMLE units: {fracmle_units_in}")
        print(f"Sumcheck PEs: {sumcheck_pes_in}")
        print(f"Product lanes: {product_lanes_in}")
        print(f"Eval engines: {eval_engines_in}")
        print(f"Onchip MLE buffer size: {onchip_mle_buf_size_in}")
        print(f"PL depth: {pl_depth_val_in}")

        max_frac_mle_units = fracmle_units_in
        ws_list  = [window_size_in]
        ppw_list = [ppw_in]
        ocw_list = [ocw_in]
        qd_list  = [16]
        ii_list  = [1]

        permcheck_mle_units_range = [fracmle_units_in]

        sumcheck_pes_range = [sumcheck_pes_in]
        eval_engines_range = [eval_engines_in]
        product_lanes_range = [product_lanes_in]
        onchip_mle_sizes_range = [onchip_mle_buf_size_in]

        design_point_str = f"{window_size_in}_{ppw_in}_{ocw_in}_{core_type_in}_{fracmle_units_in}_{sumcheck_pes_in}_{eval_engines_in}_{product_lanes_in}_{onchip_mle_buf_size_in}"

    gate_type = args.gate_type
    assert gate_type in ["vanilla", "jellyfish"]

    # general sweep params
    max_num_vars = args.nv
    min_num_vars = max_num_vars # can change this later

    skip_percent = args.skip_percent
    enable_sparsity = False

    special_experiment = args.special_experiment

    if special_experiment != None:
        if special_experiment == "sparsity_experiment":

            if gate_type == "vanilla":
                exit("not yet implemented")
            else:
                assert skip_percent != None
                enable_sparsity = True
                if skip_percent == 10:
                    skip_fraction_dict = jf_10percent_skip_fraction_dict
                elif skip_percent == 25:
                    skip_fraction_dict = jf_25percent_skip_fraction_dict
                elif skip_percent == 50:
                    skip_fraction_dict = jf_50percent_skip_fraction_dict
                elif skip_percent == 75:
                    skip_fraction_dict = jf_75percent_skip_fraction_dict
                elif skip_percent == 99:
                    skip_fraction_dict = jf_99percent_skip_fraction_dict
                else:
                    exit(
                        "not implemented for this skip fraction. Please use 10, 25, 50, 75, or 99"
                    )
        elif special_experiment == "arbitrary_prime":
            
            assert modmul_area == .478
            assert padd_area == 23.77

        elif special_experiment == "onchip_sram_penalty":
            assert onchip_sram_penalty == True
        
        elif special_experiment == "onchip_sram_and_arbitrary_prime":
            assert modmul_area == .478
            assert padd_area == 23.77
            assert onchip_sram_penalty == True

    if special_experiment != "arbitrary_prime" and special_experiment != "onchip_sram_and_arbitrary_prime":
        assert modmul_area == 0.264 and padd_area == modmuls_in_padd * modmul_area_381b / modmul_frac_in_hls
    
    if special_experiment != "onchip_sram_penalty" and special_experiment != "onchip_sram_and_arbitrary_prime":
        assert onchip_sram_penalty == False



    if gate_type == "vanilla":
        sumcheck_polynomials = [vanilla_zerocheck_polynomial, vanilla_permcheck_polynomial, opencheck_polynomial]
        num_selectors = 5
        num_witnesses = 3
        if not enable_sparsity:
            skip_fraction_dict = vanilla_skip_none_fraction_dict
    else:
        sumcheck_polynomials = [jellyfish_zerocheck_polynomial, jellyfish_permcheck_polynomial, opencheck_polynomial]
        num_selectors = 13
        num_witnesses = 5
        if not enable_sparsity:
            skip_fraction_dict = jellyfish_skip_none_fraction_dict

    bits_per_permutation_element = max_num_vars + 2
    mle_combine_bitwidths = bits_per_scalar, avg_bits_per_witness_word, bits_per_permutation_element

    num_vars_range = range(max_num_vars + 1)
    target_num_vars_range = range(min_num_vars, max_num_vars + 1)

    print(f"enable_sparsity: {enable_sparsity}")
    print(f"no_point_duplication: {no_point_duplication}")
    

    ################################################################################################

    # witness commit step
    sparse_distribution = [(sparse_fraction_ones, sparse_fraction_dense)]

    debug_sparse = False

    sparse_designs = get_designs(target_num_vars_range, sparse_distribution, ws_list, ppw_list, ocw_list, qd_list, ii_list, padd_latency)
    sparse_msm_data_dict = get_sparse_msm_stats(sparse_designs, bits_per_type, padd_area, available_bw, freq, msm_base_dir, optimal_bucket_reduction_latency_dict, gate_type=gate_type, debug=debug_sparse)

    ################################################################################################

    # we'll need to stitch together MSMs separately for permcheck and poly opening steps
    dense_distribution = [(0, 1)]
    dense_designs = get_designs(num_vars_range, dense_distribution, ws_list, ppw_list, ocw_list, qd_list, ii_list, padd_latency)
    dense_msm_data_dict = get_dense_msm_stats(dense_designs, bits_per_type, padd_area, available_bw, freq, msm_base_dir, optimal_bucket_reduction_latency_dict)

    permcheck_bitwidth_data = bits_per_scalar, avg_bits_per_witness_word
    
    permcheck_mle_diff_buffer_size = 256 # assume this is the difference buffer size used for witnesses in permcheck
    
    permcheck_mle_sweep_data = get_permcheck_mle_data(target_num_vars_range, permcheck_mle_units_range, permcheck_bitwidth_data, freq, available_bw, permcheck_mle_diff_buffer_size, gate_type=gate_type, assume_onchip_storage=onchip_sram_penalty)
    
    # no_point_duplication = True

    # compute the outer product of permcheck MLEs with dense MSM
    load_rate_range = permcheck_mle_units_range
    metadata = bits_per_point_reduced, available_bw, freq, no_point_duplication
    permcheck_mle_msm_data_dict = permcheck_mle_msm_sweep(target_num_vars_range, load_rate_range, permcheck_mle_sweep_data, dense_msm_data_dict, metadata)
  
    mle_combine_supplemental_data = mle_combine_bitwidths, available_bw, freq
    mle_combine_model_data = mle_combine_model(max_num_vars, num_witnesses, num_selectors, onchip_mle_sizes_range, mle_combine_supplemental_data)

    metadata = 13*bits_per_scalar, bits_per_scalar, bits_per_point_reduced, available_bw, freq
    
    point_merge_stats = get_point_merge_data(target_num_vars_range, modmul_latency, modmul_area)
    
    # point_merge_latency_scale_factor = modmuls_for_mle_combine/num_vars
    polyopen_data_dict = polyopen_sweep_v1(target_num_vars_range, dense_msm_data_dict, point_merge_stats, metadata)

    comprehensive_msm_data_dict = combine_msm_data_v1(target_num_vars_range, sparse_msm_data_dict, permcheck_mle_msm_data_dict, polyopen_data_dict, permcheck_mle_units_range)

    # V1 sumcheck model
    sumcheck_sweep_params = target_num_vars_range, sumcheck_pes_range, eval_engines_range, product_lanes_range, onchip_mle_sizes_range

    sparsity_data = avg_bits_per_witness_word, skip_fraction_dict
    sumcheck_supplemental_data = bits_per_scalar, available_bw, freq
    
    start_time = time.time()
    sumcheck_core_stats = get_sumcheck_sweep_data_v1(sumcheck_sweep_params, sumcheck_polynomials, sparsity_data, sumcheck_supplemental_data)
    end_time = time.time()
    print(f"Time taken for get_sumcheck_sweep_data_v1: {end_time - start_time:.4f} seconds")

    batch_eval_sweep_params = target_num_vars_range, sumcheck_pes_range, product_lanes_range, eval_engines_range, range(1, max_pl_offset), [gate_type]
    batch_eval_relevant_latencies = modmul_latency, modadd_latency
    
    # dict, order by: [GateType][NumVars][NumPes][NumProductLanes][NumEvalEngines][PlDepth]
    batch_eval_data = get_batch_eval_data_v1(batch_eval_sweep_params, batch_eval_relevant_latencies)
    

    if get_pareto_points:
        # Measure time for sumcheck_pareto_indices
        start_time = time.time()
        sumcheck_pareto_indices = get_pareto_sumcheck_points(max_num_vars, sumcheck_core_stats[max_num_vars], batch_eval_data, mle_combine_model_data)
        end_time = time.time()
        print(f"Time taken for get_pareto_sumcheck_points: {end_time - start_time:.4f} seconds")

        # Measure time for full_chip_designs_dict with get_full_chip_sweep_sc_pareto_v1
        start_time = time.time()
        full_chip_designs_dict = get_full_chip_sweep_sc_pareto_v1(target_num_vars_range, comprehensive_msm_data_dict, sumcheck_core_stats, sumcheck_pareto_indices, batch_eval_data, mle_combine_model_data, available_bw, hbm_area, onchip_sram_penalty=onchip_sram_penalty, mask_sc_opt=zc_opt, masked_sumcheck_polynomial=sumcheck_polynomials[0]) 
        end_time = time.time()
        print(f"Time taken for get_full_chip_sweep_sc_pareto_v1: {end_time - start_time:.4f} seconds")

        # exit()

        costs = np.array([(design["overall_runtime"], design["overall_area"]) for design in full_chip_designs_dict.values()])
        pareto_mask = is_pareto_efficient(costs)
        pareto_design_points = {label: design for i, (label, design) in enumerate(full_chip_designs_dict.items()) if pareto_mask[i]}
        pareto_design_points_sorted = dict(
            sorted(pareto_design_points.items(), key=lambda x: (x[1]["overall_runtime"], x[1]["overall_area"]))
        )

        print("\nPareto-optimal design points:")
        for dp, val in pareto_design_points_sorted.items():
            print(dp, val)
            print()

        if dump_data:
            pareto_file_dir = "pareto_data"
            if special_experiment != None:
                special_experiment_dir = os.path.join(pareto_file_dir, special_experiment)
                os.makedirs(special_experiment_dir, exist_ok=True)
                file_path = os.path.join(special_experiment_dir, f"{gate_type}_{max_num_vars}vars_{available_bw}gbs.pkl") 
            
            elif design_point != None:
                design_point_dir = os.path.join(pareto_file_dir, "individual_design_points")
                os.makedirs(design_point_dir, exist_ok=True)
                file_path = os.path.join(design_point_dir, f"{gate_type}_{max_num_vars}vars_{available_bw}gbs_{design_point_str}.pkl")
                print(file_path)
            else:
                file_path = os.path.join(pareto_file_dir, f"{gate_type}_{max_num_vars}vars_{available_bw}gbs.pkl")
        
            with open(file_path, "wb") as f:

                data_to_dump = {
                    "all_msm_stats": comprehensive_msm_data_dict, 
                    "sumcheck_core_stats": sumcheck_core_stats,
                    "batch_eval_data": batch_eval_data,
                    "all_designs": full_chip_designs_dict,
                    "pareto_design_points": pareto_design_points_sorted, 
                }
                pickle.dump(data_to_dump, f)
    else:
        data_dir = "raw_design_data"
        os.makedirs(data_dir, exist_ok=True)
        file_path = os.path.join(data_dir, f"{gate_type}_{max_num_vars}vars_{available_bw}gbs.pkl")
        with open(file_path, "wb") as f:
            results_dict = {
                "sumcheck_core_stats": sumcheck_core_stats,
                "batch_eval_data": batch_eval_data,
                "mle_combine_model_data": mle_combine_model_data,
                "comprehensive_msm_data_dict": comprehensive_msm_data_dict,
            }
            pickle.dump(results_dict, f)
