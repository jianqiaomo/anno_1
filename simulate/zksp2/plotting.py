
from .params import *
import argparse
from .plot_funcs import *
import os

def make_path(plot_file_path):
    os.makedirs(plot_file_path, exist_ok=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('func', type=str)
    parser.add_argument('--gate_type', type=str)
    parser.add_argument('--num_vars_list', type=str, default=None)
    parser.add_argument('--bw_vec', type=str, default=None)
    parser.add_argument('--num_threads', type=int, default=None)
    args = parser.parse_args()

    # Directory and parameters
    pareto_file_dir = "pareto_data"

    func = args.func
    num_vars_list = args.num_vars_list
    gate_type = args.gate_type

    if num_vars_list != None:
        num_vars_list = [int(x) for x in num_vars_list.split(',')]

    bits_per_scalar = 255
    bw_vec = args.bw_vec
    if bw_vec != None:
        available_bw_vec = [x for x in bw_vec.split(',')]

    num_threads = args.num_threads

    mm2_per_cpu_core = 9.25

    colors = ["brown", "black", '#9467bd', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    if func == "pareto_data_with_global_inset":

        plot_file_path = "plots/bw_pareto_plots" 
        make_path(plot_file_path)
        overwrite = False
        for num_vars in num_vars_list:

            overall_time_limit = None
            inset_time_limit = None
            overall_area_limit = None
            inset_area_limit = None

            if num_vars == 24:
                if gate_type == "vanilla":
                    overall_time_limit = 1500
                    inset_time_limit = 250
                if gate_type == "jellyfish":
                    overall_time_limit = 2000
                    inset_time_limit = 750
                    overall_area_limit = 600
                    inset_area_limit = 600
            if num_vars == 22:
                if gate_type == "jellyfish":
                    overall_time_limit = 1500
                    inset_time_limit = 250                    
             
            plot_pareto_data_with_global_inset(num_vars, gate_type, available_bw_vec, pareto_file_dir, plot_file_path, overwrite, overall_time_limit, inset_time_limit, overall_area_limit, inset_area_limit)
    
    elif func == "util_traces":
        plot_file_path = "plots/util_traces"
        make_path(plot_file_path)
        for num_vars in num_vars_list:
            target_design = (num_vars, (9, 2048, 16, 16, 1), 'single_core', 1, (2, 11, 4))
            plot_util_trace(target_design, available_bw_vec, pareto_file_dir, plot_file_path)

    elif func == "relative_util_traces":
        plot_file_path = "plots/relative_util_traces"
        make_path(plot_file_path)
        for num_vars in num_vars_list:    
            target_design = (num_vars, (9, 2048, 16, 16, 1), 'single_core', 1, (2, 11, 4))
            plot_relative_util_trace(target_design, available_bw_vec, pareto_file_dir, plot_file_path)

    elif func == "cpu_runtime_breakdown":
        plot_file_path = "plots/cpu_runtime_breakdown"
        make_path(plot_file_path)
        data_path_prefix = "./"
        for num_vars in num_vars_list:
            plot_cpu_runtime_breakdown(num_vars, num_threads, data_path_prefix, plot_file_path)

    elif func == "relative_util_traces_stacked":
        plot_file_path = "plots/relative_util_traces_stacked"
        make_path(plot_file_path)
        for num_vars in num_vars_list:    
            target_design = (num_vars, (9, 2048, 16, 16, 1), 'single_core', 1, (2, 11, 4))
            plot_relative_util_trace_stacked(target_design, available_bw_vec, pareto_file_dir, plot_file_path)

    elif func == "zkspeed_runtime_breakdown":
        plot_file_path = "plots/zkspeed_runtime_breakdown"
        make_path(plot_file_path)
        data_path_prefix = "./"
        for num_vars in num_vars_list:
            target_design = (num_vars, (9, 2048, 16, 16, 1), 'single_core', 1, (2, 11, 4))
            plot_zkspeed_runtime_breakdown(target_design, available_bw_vec, pareto_file_dir, plot_file_path)
    
    elif func == "cpu_zkspeed_runtime_breakdown":

        plot_file_path = "plots/cpu_zkspeed_runtime_breakdown"
        make_path(plot_file_path)
        data_path_prefix = "./"
        design_bw = available_bw_vec[0]
        for num_vars in num_vars_list:
            target_design = (num_vars, (9, 2048, 16, 16, 1), 'single_core', 1, (2, 11, 4))
            plot_cpu_zkspeed_runtime_breakdown(target_design, design_bw, pareto_file_dir, num_vars, num_threads, data_path_prefix, plot_file_path)

    elif func == "get_global_pareto_points":
        for num_vars in num_vars_list:
            global_pareto_points, global_pareto_points_array, runtimes_vec, areas_vec, all_colors, pareto_mask = \
                get_global_pareto_points(num_vars, gate_type, available_bw_vec, pareto_file_dir, colors=colors, overwrite=True)

            sorted_pareto_points = sorted(global_pareto_points.items(), key=lambda x: x[1]['area'], reverse=True)
            
            for design, val in sorted_pareto_points:
                print(design, f"area: {val['area']}, runtime: {val['runtime']}, bw: {val['bandwidth']}")
            print()
            print()
            print()
            
    elif func == "get_iso_cpu_points":
        for num_vars in num_vars_list:
            global_pareto_points, global_pareto_points_array, runtimes_vec, areas_vec, all_colors, pareto_mask = \
                get_global_pareto_points(num_vars, gate_type, available_bw_vec, pareto_file_dir, colors=colors, overwrite=False)

            sorted_pareto_points = sorted(global_pareto_points.items(), key=lambda x: x[1]['area'], reverse=True)
            
            closest_point = min(global_pareto_points.items(), key=lambda x: abs(x[1]['area'] - 296))
            print(f"Closest point to 300 mm2 in area: {closest_point[0]}, area: {closest_point[1]['area']}, runtime: {closest_point[1]['runtime']}, bandwidth: {closest_point[1]['bandwidth']}")

    elif func == "get_iso_zkspeed_points":
        for num_vars in num_vars_list:
            global_pareto_points, global_pareto_points_array, runtimes_vec, areas_vec, all_colors, pareto_mask = \
                get_global_pareto_points(num_vars, gate_type, available_bw_vec, pareto_file_dir, colors=colors, overwrite=False)

            sorted_pareto_points = sorted(global_pareto_points.items(), key=lambda x: x[1]['area'], reverse=True)
            
            closest_point = min(global_pareto_points.items(), key=lambda x: abs(x[1]['area'] - 363))
            print(f"Closest point to 365 mm2 in area: {closest_point[0]}, area: {closest_point[1]['area']}, runtime: {closest_point[1]['runtime']}, bandwidth: {closest_point[1]['bandwidth']}")

    elif func == "plot_ablation_study":
        plot_file_path = "plots/ablation_study"
        make_path(plot_file_path)
        # latencies = np.array([
        #     11.405,  # zkspeed (2^20)
        #     12.083,  # zkphire w/ SRAM, arbitrary prime (2^20)
        #     7.791,   # zkphire w/ SRAM, fixed prime (2^20)
        #     7.887,   # zkphire w/o SRAM, arbitrary prime (2^20)
        #     5.793,   # zkphire w/o SRAM, fixed prime (2^20)
        #     171.61,  # zkspeed (2^24)
        #     182.326, # zkphire w/ SRAM, arbitrary prime (2^24)
        #     115.437, # zkphire w/ SRAM, fixed prime (2^24)
        #     116.433, # zkphire w/ arbitrary prime modmuls (2^24)
        #     82.474   # zkphire w/ fixed prime modmuls (2^24)
        # ])

        latencies = np.array([
            171.61,   # zkspeed (2^24)
            174.772,  # zkphire w/ SRAM, arbitrary prime (2^24)
            108.426,  # zkphire w/ SRAM, fixed prime (2^24)
            109.116,  # zkphire w/ arbitrary prime modmuls (2^24)
            78.975,   # zkphire w/ fixed prime modmuls (2^24)
            4.117
        ])
        plot_ablation_study(latencies, plot_file_path)

    elif func == "plot_sparsity_study":
        plot_file_path = "plots/sparsity_study"
        make_path(plot_file_path)

        latencies_2tbs = np.array([
            125.566,  # 0% sparsity
            # 124.78,   # 10% sparsity
            123.83,   # 25% sparsity
            121.53,   # 50% sparsity
            119.242,  # 75% sparsity
            117.056   # 99% sparsity
        ]) 
        latencies_512gbs = np.array([
            387.491,  # 0% sparsity
            # 382.764,  # 10% sparsity
            375.309,  # 25% sparsity
            363.07,   # 50% sparsity
            350.791,  # 75% sparsity
            339.182   # 99% sparsity
        ])
        plot_sparsity_study(latencies_2tbs, plot_file_path, "2TB")
        plot_sparsity_study(latencies_512gbs, plot_file_path, "512GB")

        plot_speedup_comparison(latencies_2tbs, latencies_512gbs, plot_file_path)


    elif func == "plot_sparsity_study_vanilla":
        plot_file_path = "plots/sparsity_study_vanilla"
        make_path(plot_file_path)

        percent_improvement = np.array([
            12.58,
            15.57,
            17.29,
        ]) 
        plot_sparsity_study_vanilla(percent_improvement, plot_file_path)

        # plot_speedup_comparison(latencies_2tbs, latencies_512gbs, plot_file_path)

    elif func == "plot_workload_ablation":

        plot_file_path = "plots/workload_ablation"
        make_path(plot_file_path)

        plot_workload_ablation("zcash", 17,	15, plot_file_path)
        plot_workload_ablation("rescue hash", 21, 20, plot_file_path)
        plot_workload_ablation("zexe", 22, 17, plot_file_path)
        plot_workload_ablation("rollup", 30, 25, plot_file_path)
        plot_workload_ablation("zkevm", 30,	27, plot_file_path)





