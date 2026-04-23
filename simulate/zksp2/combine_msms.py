

def combine_msm_data(target_num_vars_range, sparse_msm_data_dict, permcheck_mle_msm_data_dict, polyopen_data_dict, permcheck_mle_units_range):

    comprehensive_msm_data = dict()
    for num_vars in target_num_vars_range:
        comprehensive_msm_data[num_vars] = dict()
        for core_key in ["single_core", "dual_core"]:
            num_cores = 1 if core_key == "single_core" else 2
            comprehensive_msm_data[num_vars][core_key] = dict()
            if num_cores == 1:
                spmsm_core_key = "single_core"
                pmchkmsm_core_key = "single_core"
                pomsm_core_key = "single_core"
            else:
                spmsm_core_key = "dual_core"
                pmchkmsm_core_key = "dual_core_permcheck"
                pomsm_core_key = "dual_core_opencheck"

            for sparse_msm_design, data_dict in sparse_msm_data_dict.items():
                (num_vars_in_design, *_), target_ws, target_ppw, target_ocw, target_qd, target_ii, padd_latency = sparse_msm_design

                if num_vars_in_design != num_vars:
                    continue
                
                dense_msm_design = (target_ws, target_ppw, target_ocw, target_qd, target_ii)

                sparse_msm_data = data_dict[spmsm_core_key]
                polyopen_data = polyopen_data_dict[num_vars][pomsm_core_key][dense_msm_design]
                
                for permcheck_mle_units in permcheck_mle_units_range:


                    # get permcheck MLE data
                    permcheck_mle_msm_design = (permcheck_mle_units, dense_msm_design)
                    if permcheck_mle_msm_design not in permcheck_mle_msm_data_dict[num_vars][pmchkmsm_core_key]:
                        continue
                    permcheck_mle_msm_data = permcheck_mle_msm_data_dict[num_vars][pmchkmsm_core_key][permcheck_mle_msm_design]

                    # let msm_design be additionally parameterized by the number of N/D, phi, pi PEs
                    msm_design = (dense_msm_design, permcheck_mle_units)
                    comprehensive_msm_data[num_vars][core_key][msm_design] = dict()
                    design_dict = comprehensive_msm_data[num_vars][core_key][msm_design]
                    

                    # confirm the area data is correct first of all
                    sparse_msm_area_stats = sparse_msm_data['area_stats']
                    permcheck_msm_area_stats = permcheck_mle_msm_data['msm_area_data']
                    polyopen_msm_area_stats = polyopen_data['msm_area_data']
                    assert sparse_msm_area_stats == permcheck_msm_area_stats
                    assert sparse_msm_area_stats == polyopen_msm_area_stats

                    # get total MSM-related latencies
                    sparse_msm_total_latency = sparse_msm_data['total_latency']
                    permcheck_mle_msm_total_latency = permcheck_mle_msm_data['total_latency']
                    polyopen_msm_total_latency = polyopen_data['total_latency']

                    permcheck_msm_only_latency = permcheck_mle_msm_data['dense_msm_total_latency']
                    permcheck_nd_latency       = permcheck_mle_msm_data['nd_cycles']
                    permcheck_frac_mle_latency = permcheck_mle_msm_data['frac_mle_cycles']
                    permcheck_prod_mle_latency = permcheck_mle_msm_data['prod_mle_cycles']

                    polyopen_msm_only_latency = polyopen_data['dense_msm_total_latency']
                    polyopen_g_prime_latency = polyopen_data['g_prime_latency']
                    polyopen_point_merge_latency = polyopen_data['point_merge_latency']



                    total_msm_latency = sparse_msm_total_latency + permcheck_mle_msm_total_latency + polyopen_msm_total_latency

                    memory_size_stats, memory_area_stats, padd_mm2 = sparse_msm_area_stats
                    
                    msm_area_dict = \
                        {
                            "total_area"        : memory_area_stats[0] + padd_mm2,
                            "total_memory_area" : memory_area_stats[0],
                            "snp_area"          : memory_area_stats[1],
                            "breg_area"         : memory_area_stats[2],
                            "bqs_area"          : memory_area_stats[3],
                            "spad_area"         : memory_area_stats[4],
                            "total_memory_mb"   : memory_size_stats[0],
                            "snp_mb"            : memory_size_stats[1],
                            "breg_mb"           : memory_size_stats[2],
                            "bqs_mb"            : memory_size_stats[3],
                            "spad_mb"           : memory_size_stats[4],
                            "padd_area"         : padd_mm2
                        }

                    design_dict['msm_area_stats'] = msm_area_dict
                    design_dict['permcheck_mle_area_stats'] = permcheck_mle_msm_data['mle_area_data']
                    design_dict['polyopen_mle_area_stats'] = polyopen_data['mle_area_data']

                    design_dict['total_msm_latency'] = total_msm_latency
                    design_dict['sparse_msm_total_latency'] = sparse_msm_total_latency
                    design_dict['permcheck_mle_msm_total_latency'] = permcheck_mle_msm_total_latency
                    design_dict['polyopen_msm_total_latency'] = polyopen_msm_total_latency

                    design_dict['permcheck_msm_only_latency'] = permcheck_msm_only_latency
                    design_dict['permcheck_nd_latency']       = permcheck_nd_latency
                    design_dict['permcheck_frac_mle_latency'] = permcheck_frac_mle_latency
                    design_dict['permcheck_prod_mle_latency'] = permcheck_prod_mle_latency

                    design_dict['polyopen_msm_only_latency'] = polyopen_msm_only_latency
                    design_dict['polyopen_g_prime_latency'] = polyopen_g_prime_latency
                    design_dict['polyopen_point_merge_latency'] = polyopen_point_merge_latency

                    design_dict['sparse_msm_bw_stats'] = sparse_msm_data['bw_stats']
                    design_dict['permcheck_mle_msm_bw_stats'] = (permcheck_mle_msm_data['peak_mle_bw'], permcheck_mle_msm_data['dense_msm_bw_stats'])
                    design_dict['polyopen_msm_bw_stats'] = polyopen_data['dense_msm_bw_stats']

    return comprehensive_msm_data
