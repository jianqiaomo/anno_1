import pickle
import os
import pandas as pd
from copy import deepcopy
import openpyxl
import tqdm
import matplotlib.pyplot as plt
import matplotlib
import yaml
from . import reverse_binary_tree

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

if __name__ == "__main__":
    # colors_style = ['#1f77b4', '#f5d44b','#2ca02c', '#f5d44b', '#b73508', '#006400', "#6e77a2", "#87c9c3", "#08326E", 'r']
    colors_style = ['#f5d44b', '#ff7f0e', '#2ca02c', '#006400', "#87c9c3", '#1f77b4', "#6e77a2", "#08326E", "#feaac2",
                    '#b73508', 'r', "#b9181a"]
    pd.options.mode.chained_assignment = None  # default='warn'

    pareto_design_points_global_different = [
    ]
    pareto_design_points_global_each_bw_pick_highest = [
        {'Design': (24, (10, 2048, 32, 16, 1), 'dual_core', 1, (16, 7, 3, 16384), 1), 'Area': 451.459,
         'Runtime': 90.593, 'Bandwidth': 4096},
        {'Design': (24, (9, 4096, 32, 16, 1), 'single_core', 3, (16, 7, 5, 16384), 1), 'Area': 310.297,
         'Runtime': 123.579, 'Bandwidth': 2048},
        {'Design': (24, (10, 1024, 32, 16, 1), 'single_core', 1, (8, 7, 3, 8192), 1), 'Area': 192.831,
         'Runtime': 195.241, 'Bandwidth': 1024},
        {'Design': (24, (10, 2048, 4, 16, 1), 'dual_core', 1, (4, 7, 6, 4096), 1), 'Area': 99.706,
         'Runtime': 371.195, 'Bandwidth': 1024},
    ]
    pareto_design_points_local_each_bw_pick_highest = [
    ]

    pareto_design_points_bw_throttled_rows = []
    for pareto_design_point in tqdm.tqdm(pareto_design_points_global_different +
                                         pareto_design_points_global_each_bw_pick_highest +
                                         pareto_design_points_local_each_bw_pick_highest):
        pareto_file_dir = "."
        num_vars = pareto_design_point["Design"][0]
        one_design = pareto_design_point["Design"][1]
        two_design = pareto_design_point["Design"][4]
        available_bw = pareto_design_point["Bandwidth"]
        modInv_batch_size = 64
        file_path = os.path.join(pareto_file_dir,
                                 f"jellyfish_{num_vars}vars_{available_bw}gbs_{one_design[0]}_{one_design[1]}_"
                                 f"{one_design[2]}_{pareto_design_point['Design'][2]}_{pareto_design_point['Design'][3]}_"
                                 f"{two_design[0]}_{two_design[1]}_{two_design[2]}_{two_design[3]}.pkl")

        # Load data
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                input_data = pickle.load(f)
                pareto_design_points_bw_throttled = input_data['pareto_design_points'][pareto_design_point["Design"]]  # 'pareto_design_points_bw_throttled_no_hbm'
                assert pareto_design_points_bw_throttled['overall_area'] == pareto_design_point['Area']
                pareto_design_points_bw_throttled_rows.append(pareto_design_points_bw_throttled)

    # Collect results
    pareto_design_points_bw_throttled_df = pd.DataFrame(
        [i['detailed_area_breakdown'] for i in pareto_design_points_bw_throttled_rows])

    ##############################################################################
    # plot: area breakdown
    pareto_design_points_area_bar_chart_misc_columns = []
    pareto_design_points_area_bar_chart_columns = ['sumcheck_core_area','msm_logic_area','mle_update_core_area',
                                                   'mle_combine_area','multifunction_tree_area','on_chip_memory',
                                                   'hbm_area', 'frac_mle_area', 'interconnected_area', 'nd_area', 'sha_area',]
    pareto_design_points_area_bar_chart_legend = ['Sumcheck', 'MSM Unit', 'MLE Update',
                                                   'MLE Combine', 'Multifunc Tree', 'Onchip Mem',
                                                   'HBM PHY', 'Frac MLE', 'Interconnect', "Constr ND", "SHA3 Unit"]
    # sum of columns in pareto_design_points_area_bar_chart_misc_columns
    pareto_design_points_bw_throttled_df['Misc'] = sum([pareto_design_points_bw_throttled_df[col]
                                                        for col in pareto_design_points_area_bar_chart_misc_columns])

    # get each of percentage
    for col in pareto_design_points_area_bar_chart_columns:
        pareto_design_points_bw_throttled_df[f"{col}_percentage"] = \
            pareto_design_points_bw_throttled_df[col] / \
            pareto_design_points_bw_throttled_df['overall_area'] * 100

    pareto_design_points_area_bar_chart_columns_chunks = [pareto_design_points_bw_throttled_df.iloc[i:i + 4] for i in
                                                          range(0, len(pareto_design_points_bw_throttled_df), 4)]

    x_ticks_labels = ["A", "B", "C", "D", "E", "F", "G", "H"]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.4))  # 1 row, 3 columns
    for ax_idx, ax in enumerate(axes):
        ax.tick_params(axis='both', which='major', labelsize=15)
        if ax_idx == 0:
            ax.set_ylabel("Area Percentage", fontsize=18)


    max_height = 100
    for idx, (ax, chunk) in enumerate(zip(axes, pareto_design_points_area_bar_chart_columns_chunks)):
        chunk.plot(
            y=[f"{col}_percentage" for col in pareto_design_points_area_bar_chart_columns],
            kind='bar', stacked=True, # width=barwidth, figsize=(3, 4),
            color=colors_style, ax=ax, legend=False)
        ax.set_ylim(0, max_height)
        ax.set_xticklabels(x_ticks_labels[:4] if idx == 0 else x_ticks_labels[4:], rotation=0)

    # fig.text(0.5, 0.04, 'Design Points', ha='center', fontsize=14.5)

    # draw text at top
    # for idx in range(num_bars_in_chart):
    #     plt.text(idx * barwidth * 2, 101,
    #              f"{pareto_design_points_sorted_df_overall_area_sorted_bar_chart_filtered.loc[idx, 'overall_area']:0.0f}",
    #              horizontalalignment='center', verticalalignment='center', fontsize=8, color='crimson'
    #              )

    fig.tight_layout(rect=[0, 0, 1, 0.83])

    fig.legend(labels=pareto_design_points_area_bar_chart_legend,
               loc='upper center', ncol=4, fontsize=12.5, bbox_to_anchor=(0.51, 1.0), frameon=False)

    plt.savefig("area_breakdown_pareto_points.pdf", dpi=100, bbox_inches="tight")
    plt.show()

    print("end")
