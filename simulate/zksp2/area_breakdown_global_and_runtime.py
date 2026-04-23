import pickle
import os
import pandas as pd
from copy import deepcopy
import openpyxl
import tqdm
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

if __name__ == "__main__":
    with open('jellyfish_24vars_2048gbs_9_2048_16_dual_core_1_16_7_5_16384.pkl', 'rb') as f:
        data = pickle.load(f)
        # (9, 2048, 16, 16, 1), 'dual_core', 1, (16, 7, 5, 16384), 1)  <-- this is the architecture label
        # full_chip_design_point_label = (num_vars, msm_design, core_key, num_frac_mle_units, sumcheck_hardware_config (num_sumcheck_pes, num_eval_engines, num_product_lanes, onchip_mle_size), pl_depth)
        arch_label = data['pareto_design_points'][(24, (9, 2048, 16, 16, 1), 'dual_core', 1, (16, 7, 5, 16384), 1)]
        pass

    # colors_style = ['#1f77b4', '#f5d44b','#2ca02c', '#f5d44b', '#b73508', '#006400', "#6e77a2", "#87c9c3", "#08326E", 'r']
    # colors_style_area = ['#f5d44b', '#ff7f0e', '#2ca02c', '#006400', "#87c9c3", '#1f77b4', "#feaac2", "#6e77a2", "#08326E",
    #                 '#b73508', 'r', "#b9181a"]
    colors_style_area = ["#A6CEE3FF", "#1F78B4FF", "#33A02CFF", "#B2DF8AFF",
                         # "#FB9A99FF",
                         # "#E31A1CFF",
                         # "#FDBF6FFF",
                         "#FF7F00FF", "#CAB2D6FF", "#6A3D9AFF",
                         # "#FFFF99FF",
                         "#B15928FF"]
    # colors_style_runtime = ['#f5d44b', '#ff7f0e', '#b73508', "#87c9c3", '#1f77b4', "#6e77a2", "#08326E", '#2ca02c', '#006400',
    #                 "#feaac2", 'r', "#b9181a"]
    # colors_style_runtime = ['#b2c4d7', '#8f979a', '#b73508', 'r', '#c59d17', '#08326E', '#a1656d', "#75aa7a", '#1f77b4', "#6e77a2", "#08326E", '#2ca02c',
    #                         '#006400', "#feaac2", "#b9181a"]
    colors_style_runtime = ["#66C2A5FF",
                            # "#FC8D62FF",
                            "#8DA0CBFF", "#E78AC3FF", "#A6D854FF", "#FFD92FFF", "#E5C494FF", "#B3B3B3FF"]
    pd.options.mode.chained_assignment = None  # default='warn'

    pareto_design_points_global_different = [
    ]
    pareto_design_points_global_each_bw_pick_highest = [
        # {'Design': (24, (10, 2048, 4, 16, 1), 'dual_core', 1, (4, 7, 6, 4096), 1), 'Area': 99.706,
        #  'Runtime': 371.195, 'Bandwidth': 1024},
        # {'Design': (24, (10, 1024, 32, 16, 1), 'single_core', 1, (8, 7, 3, 8192), 1), 'Area': 192.831,
        #  'Runtime': 195.241, 'Bandwidth': 1024},
        # {'Design': (24, (9, 4096, 32, 16, 1), 'single_core', 3, (16, 7, 5, 16384), 1), 'Area': 310.297,
        #  'Runtime': 123.579, 'Bandwidth': 2048},
        # {'Design': (24, (10, 2048, 32, 16, 1), 'dual_core', 1, (16, 7, 3, 16384), 1), 'Area': 451.459,
        #  'Runtime': 90.593, 'Bandwidth': 4096},
        {'Design': (24, (9, 2048, 16, 16, 1), 'single_core', 1, (4, 7, 7, 4096), 1),
         'Runtime': 326.689, 'Area': 114.139, 'Bandwidth': 512},
        {'Design': (24, (10, 2048, 32, 16, 1), 'single_core', 1, (8, 7, 6, 16384), 1),
         'Runtime': 169.359, "Area": 223.184, "Bandwidth": 1024},
        {'Design': (24, (10, 2048, 32, 16, 1), 'dual_core', 1, (16, 7, 6, 32768), 1),
        'Runtime': 91.941, 'Area': 442.133, 'Bandwidth': 2048},
        {'Design': (24, (10, 2048, 32, 16, 1), 'dual_core', 3, (32, 7, 5, 32768), 1),
         'Runtime': 69.856, 'Area': 598.545, 'Bandwidth': 4096},
    ]
    pareto_design_points_local_each_bw_pick_highest = [
    ]

    pareto_file_dir = "."
    pareto_design_points_rows_area = []
    pareto_design_points_rows_latency = []
    for pareto_design_point_name in tqdm.tqdm(pareto_design_points_global_different +
                                              pareto_design_points_global_each_bw_pick_highest +
                                              pareto_design_points_local_each_bw_pick_highest):
        pareto_file_dir = "."
        num_vars = pareto_design_point_name["Design"][0]
        one_design = pareto_design_point_name["Design"][1]
        two_design = pareto_design_point_name["Design"][4]
        available_bw = pareto_design_point_name["Bandwidth"]
        modInv_batch_size = 64
        file_path = os.path.join(pareto_file_dir,
                                 f"jellyfish_{num_vars}vars_{available_bw}gbs_{one_design[0]}_{one_design[1]}_"
                                 f"{one_design[2]}_{pareto_design_point_name['Design'][2]}_{pareto_design_point_name['Design'][3]}_"
                                 f"{two_design[0]}_{two_design[1]}_{two_design[2]}_{two_design[3]}.pkl")

        # Load data
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                input_data = pickle.load(f)
                pareto_design_points = input_data['pareto_design_points'][pareto_design_point_name["Design"]]  # 'pareto_design_points_no_hbm'
                assert pareto_design_points['overall_area'] == pareto_design_point_name['Area']
                detailed_area_breakdown = deepcopy(pareto_design_points['detailed_area_breakdown'])
                detailed_latency_breakdown = deepcopy(pareto_design_points['runtime_breakdown'])
                # pareto_design_points_rows_area.append(pareto_design_points)
                pareto_design_points_rows_area.append({
                    'HBM PHY': detailed_area_breakdown['hbm_area'],
                    'Interconnect': detailed_area_breakdown['interconnect_area'],
                    'Onchip Mem': detailed_area_breakdown['memory_area_mm2']['on_chip_memory_area']
                                  + detailed_area_breakdown['registers_area_mm2']['total_reg_area'],
                    # 'ProductCheck': detailed_area_breakdown['module_area']['frac_mle_area']
                    #                 + detailed_area_breakdown['module_area']['nd_area'],
                    'MSM': detailed_area_breakdown['module_area']['msm_logic_area'],
                    'MultiFunc Forest': detailed_area_breakdown['module_area']['multifunction_tree_area'],
                    'SumCheck': detailed_area_breakdown['module_area']['sumcheck_core_area'],
                    'Misc': detailed_area_breakdown['module_area']['sha_area']
                            + detailed_area_breakdown['module_area']['mle_combine_area']
                            + detailed_area_breakdown['module_area']['frac_mle_area']
                            + detailed_area_breakdown['module_area']['nd_area'],
                })
                # runtime_breakdown = {
                #     "witness_commit": sparse_msm_total_latency,
                #     "gate_identity": zerocheck_latency,
                #     "wire_identity_commit": permcheck_mle_msm_total_latency,
                #     "wire_identity_sumcheck": permcheck_latency,
                #     "batch_eval": total_batch_eval_latency,
                #     "opencheck_latency": opencheck_latency,
                #     "polyopen": total_mle_combine_latency + polyopen_msm_total_latency + final_eval_latency,
                # }
                pareto_design_points_rows_latency.append({
                    'Witness MSM': detailed_latency_breakdown['witness_commit'],
                    'ZeroCheck': detailed_latency_breakdown['gate_identity'],
                    'PermCheck': detailed_latency_breakdown['wire_identity_sumcheck'],
                    'Wiring MSM': detailed_latency_breakdown['wire_identity_commit'],
                    'OpenCheck': detailed_latency_breakdown['opencheck_latency'],
                    'PolyOpen MSM': detailed_latency_breakdown['polyopen'],
                    'Other': detailed_latency_breakdown['batch_eval'],
                })

    # Collect results
    pareto_design_points_area_df = pd.DataFrame(pareto_design_points_rows_area)
    pareto_design_points_latency_df = pd.DataFrame(pareto_design_points_rows_latency)

    # Load data for runtime
    # file_path = os.path.join(pareto_file_dir, f"20vars_pareto_latency_breakdown.pkl")
    # if os.path.exists(file_path):
    #     with open(file_path, 'rb') as f:
    #         runtime_data = pickle.load(f)

    # Collect results runtime
    runtime_bar_chart_legend = ['Witness MSM', 'Wiring MSM', 'PolyOpen MSM', 'ZeroCheck', 'PermCheck', 'OpenCheck', 'Other']
    # runtime_bar_chart_columns = ['sparse_msm_latency', 'permcheck_mle_msm_latency', 'polyopen_msm_latency',
    #                              'zerocheck_latency', 'permcheck_latency', 'opencheck_latency', 'final_eval_latency',
    #                              'other']
    runtime_bar_chart_columns = runtime_bar_chart_legend
    runtime_data_percent_df = pareto_design_points_latency_df
    for col in runtime_bar_chart_columns:
        runtime_data_percent_df[f"{col}_percentage"] = runtime_data_percent_df[col] / sum([
            runtime_data_percent_df[col] for col in runtime_bar_chart_columns
        ]) * 100
    runtime_data_percent_df_chunks = [runtime_data_percent_df.iloc[i:i + 4] for i in
                                      range(0, len(runtime_data_percent_df), 4)]
    s_percent = sum([runtime_data_percent_df.iloc[0][f"{col}_percentage"] for col in runtime_bar_chart_columns])

    ##############################################################################
    # plot: area breakdown
    pareto_design_points_area_bar_chart_columns = ['SumCheck', 'MultiFunc Forest', 'MSM',
                                                   # 'ProductCheck',
                                                   'Onchip Mem', 'HBM PHY', 'Interconnect', "Misc"]
    pareto_design_points_area_bar_chart_legend = pareto_design_points_area_bar_chart_columns
    # sum of columns in pareto_design_points_area_bar_chart_misc_columns
    # pareto_design_points_area_df['Misc'] = sum([pareto_design_points_area_df[col]
    #                                             for col in pareto_design_points_area_bar_chart_misc_columns])

    # get each of percentage
    for col in pareto_design_points_area_bar_chart_columns:
        pareto_design_points_area_df[f"{col}_percentage"] = \
            pareto_design_points_area_df[col] / sum([
                pareto_design_points_area_df[col] for col in pareto_design_points_area_bar_chart_columns
            ]) * 100

    pareto_design_points_area_bar_chart_columns_chunks = [pareto_design_points_area_df.iloc[i:i + 4] for i in
                                                          range(0, len(pareto_design_points_area_df), 4)]

    x_ticks_labels = ["A", "B", "C", "D"]
    fig, axes = plt.subplots(1, 2, figsize=(8.7, 4.4))  # 1 row, 3 columns
    for ax_idx, ax in enumerate(axes):
        ax.tick_params(axis='both', which='major', labelsize=15)
        if ax_idx == 0:
            ax.set_ylabel("Area Percentage", fontsize=18)
        elif ax_idx == 1:
            ax.set_ylabel("Runtime Percentage", fontsize=18)

    max_height = 100
    # for idx, (ax, chunk) in enumerate(zip(axes, pareto_design_points_area_bar_chart_columns_chunks)):
    #     chunk.plot(
    #         y=[f"{col}_percentage" for col in pareto_design_points_area_bar_chart_columns],
    #         kind='bar', stacked=True,  # width=barwidth, figsize=(3, 4),
    #         color=colors_style_area, ax=ax, legend=False)
    #     ax.set_ylim(0, max_height)
    #     ax.set_xticklabels(x_ticks_labels[:4] if idx == 0 else x_ticks_labels[4:], rotation=0)
    pareto_design_points_area_bar_chart_columns_chunks[0].plot(
        y=[f"{col}_percentage" for col in pareto_design_points_area_bar_chart_columns],
        kind='bar', stacked=True,  # width=barwidth, figsize=(3, 4),
        color=colors_style_area, ax=axes[0], legend=False)
    axes[0].set_ylim(0, max_height)
    axes[0].set_xticklabels(x_ticks_labels[:4], rotation=0)


    ##############################################################################
    # plot: runtime breakdown

    # for idx, (ax, chunk) in enumerate(zip(axes, runtime_data_percent_df_chunks)):
    #     chunk.plot(
    #         y=[f"{col}_percentage" for col in runtime_bar_chart_columns],
    #         kind='bar', stacked=True,  # figsize=(6, 4), width=bar_width,
    #         color=colors_style_runtime, ax=ax, legend=False)
    #     ax.set_ylim(0, max_height)
    #     ax.set_xticklabels(x_ticks_labels[:4] if idx == 0 else x_ticks_labels[4:], rotation=0)
    runtime_data_percent_df_chunks[0].plot(
        y=[f"{col}_percentage" for col in runtime_bar_chart_columns],
        kind='bar', stacked=True,  # figsize=(6, 4), width=bar_width,
        color=colors_style_runtime, ax=axes[1], legend=False)
    axes[1].set_ylim(0, max_height)
    axes[1].set_xticklabels(x_ticks_labels[:4], rotation=0)

    fig.tight_layout(rect=[0, 0.12, 1, 0.88])
    axes[0].legend(labels=pareto_design_points_area_bar_chart_legend, loc='upper left',
                  ncol=4, fontsize=14, bbox_to_anchor=(-0.25, 1.28), frameon=False)
    axes[1].legend(labels=runtime_bar_chart_legend, loc='lower center', ncol=4, fontsize=14,
               bbox_to_anchor=(-0.26, -0.37), frameon=False)



    # draw text at top
    # for idx in range(num_bars_in_chart):
    #     plt.text(idx * barwidth * 2, 101,
    #              f"{pareto_design_points_sorted_df_overall_area_sorted_bar_chart_filtered.loc[idx, 'overall_area']:0.0f}",
    #              horizontalalignment='center', verticalalignment='center', fontsize=8, color='crimson'
    #              )

    # plt.savefig("area_breakdown_pareto_points.pdf", dpi=100, bbox_inches="tight")
    plt.show()

    print("end")
