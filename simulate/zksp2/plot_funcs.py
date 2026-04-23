import numpy as np
from .util import is_pareto_efficient
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
import pickle
import matplotlib.colors as mcolors
import time
import numpy as np
import re

def find_closest_area_design(pareto_points, target_area, target_bandwidth):
    """
    Finds the design with an area closest to the target area for a given bandwidth.
    
    Args:
        pareto_points (dict): Dictionary where keys are design labels and values are dictionaries 
                              with "area" and "bandwidth" keys.
        target_area (float): The target area to match.
        target_bandwidth (float): The bandwidth to filter designs.
                              
    Returns:
        tuple: The design label and its area closest to the target area for the given bandwidth.
    """
    # Filter designs by the target bandwidth
    filtered_designs = [
        (design, details["area"]) for design, details in pareto_points.items()
        if details["bandwidth"] == target_bandwidth
    ]
    
    if not filtered_designs:
        raise ValueError(f"No designs found for bandwidth {target_bandwidth}")
    
    # Find the design with the area closest to the target area
    closest_design = min(filtered_designs, key=lambda x: abs(x[1] - target_area))
    
    return closest_design

def find_highest_area_per_bandwidth(pareto_points):
    """
    Finds the design with the highest area for each bandwidth.
    
    Args:
        pareto_points (dict): Dictionary where keys are design labels and values are dictionaries 
                              with "area" and "bandwidth" keys.
                              
    Returns:
        dict: A dictionary with bandwidths as keys and the design with the highest area as values.
    """

    # Group designs by bandwidth
    grouped_by_bandwidth = defaultdict(list)
    for design, details in pareto_points.items():
        bandwidth = details["bandwidth"]
        grouped_by_bandwidth[bandwidth].append((design, details["area"]))

    # Find the highest area design for each bandwidth
    highest_area_per_bandwidth = {}
    for bandwidth, designs in grouped_by_bandwidth.items():
        highest_design = max(designs, key=lambda x: x[1])  # Find the design with the max area
        highest_area_per_bandwidth[bandwidth] = highest_design

    return highest_area_per_bandwidth


def get_target_design(pareto_points, target_area, pareto_bandwidths, ignore_bw=False):
    areas = np.array([v["overall_area"] for v in pareto_points.values()])
    keys = list(pareto_points.keys())

    # Find the index of the closest area to the target
    closest_idx = np.argmin(np.abs(areas - target_area))

    # Retrieve the corresponding key and actual area
    target_design = keys[closest_idx]
    actual_area = areas[closest_idx]
    if not ignore_bw:
        bandwidth = pareto_bandwidths[closest_idx]
        print(f"Selected Design: {target_design}, Actual Area: {actual_area}, Bandwidth: {bandwidth}")
    else:
        bandwidth = None
    # Print and return the result
    return target_design, actual_area, bandwidth


def get_global_pareto_points(num_vars, gate_type, available_bw_vec, pareto_file_dir, colors=None, overwrite=False):
    global_pareto_file_path = os.path.join(pareto_file_dir, f"{gate_type}_{num_vars}vars_all_bws.pkl")
    if not overwrite and os.path.exists(global_pareto_file_path):
        with open(global_pareto_file_path, 'rb') as f:
            input_data = pickle.load(f)
            global_pareto_points = input_data['global_pareto_points']
            global_pareto_points_array = input_data["global_pareto_points_array"]
            runtimes_vec = input_data['runtimes_vec']
            areas_vec = input_data['areas_vec']
            all_colors = input_data['all_colors']
            pareto_mask = input_data['pareto_mask']
    else:
        # Initialize lists to store runtimes and areas for each bandwidth
        runtimes_vec = []
        areas_vec = []
        if colors == None:
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', "black", "brown"]
            colors.reverse()

        all_points = []  # To store all points for Pareto-optimal calculation
        all_colors = []  # To store colors for each point
        all_bandwidths = []
        all_design_labels = []

        # Loop through each bandwidth value, load data, and store runtimes and areas
        for i, available_bw in enumerate(available_bw_vec):
            file_path = os.path.join(pareto_file_dir, f"{gate_type}_{num_vars}vars_{available_bw}gbs.pkl")
            
            # Check if the file exists and load the Pareto points
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    pareto_points = pickle.load(f)['pareto_design_points']

                # Extract runtimes and areas from the Pareto points dictionary format
                pareto_runtimes = np.array([design["overall_runtime"] for design in pareto_points.values()])
                pareto_areas = np.array([design["overall_area"] for design in pareto_points.values()])
                pareto_labels = list(pareto_points.keys())
                
                # Append to the lists for plotting
                runtimes_vec.append(pareto_runtimes)
                areas_vec.append(pareto_areas)

                # Collect points for Pareto-optimal calculation and their corresponding colors
                points = np.column_stack((pareto_runtimes, pareto_areas))
                all_points.append(points)
                all_colors.extend([colors[i]] * len(points))  # Assign the same color to all points in this curve
                all_bandwidths.extend([available_bw] * len(points))
                all_design_labels.extend(pareto_labels)
            else:
                print(f"File not found: {file_path}")
                exit("File missing")

        # Combine all points for Pareto-optimal calculation
        all_points = np.vstack(all_points)  # Combine all points into a single array
        all_colors = np.array(all_colors)   # Convert colors list to a numpy array
        all_bandwidths = np.array(all_bandwidths)

        pareto_mask = is_pareto_efficient(all_points, return_mask=True)
        global_pareto_points_array = all_points[pareto_mask]
        global_pareto_bandwidths = all_bandwidths[pareto_mask]
        
        global_pareto_labels = [all_design_labels[i] for i in range(len(all_design_labels)) if pareto_mask[i]]
        # global_pareto_labels = all_design_labels[pareto_mask]  # Get the labels of Pareto-optimal points

        global_pareto_points = {
            label: {"runtime": runtime, "area": area, "bandwidth": bandwidth}
            for label, (runtime, area), bandwidth in zip(
                global_pareto_labels, global_pareto_points_array, global_pareto_bandwidths
            )
        }

        global_pareto_file_path = os.path.join(pareto_file_dir, f"{gate_type}_{num_vars}vars_all_bws.pkl")
        with open(global_pareto_file_path, "wb") as f:
            data_to_dump = {
                "global_pareto_points": global_pareto_points,
                "global_pareto_points_array" : global_pareto_points_array,
                "runtimes_vec": runtimes_vec,
                "areas_vec": areas_vec,
                "all_colors": all_colors,
                "pareto_mask": pareto_mask,
            }
            pickle.dump(data_to_dump, f)
    return global_pareto_points, global_pareto_points_array, runtimes_vec, areas_vec, all_colors, pareto_mask


def plot_pareto_data_with_global_inset(num_vars, gate_type, available_bw_vec, pareto_file_dir, plot_file_path, overwrite=False, overall_time_limit=None, inset_time_limit=None, overall_area_limit=None, inset_area_limit=None):

    colors = ["brown", "black", '#9467bd', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    color_bandwidths = {
        'brown':   64,
        'black':   128,
        '#9467bd': 256,
        '#2ca02c': 2048,
        '#ff7f0e': 1024,
        '#1f77b4': 512,
        '#d62728': 4096
    }
    start_time = time.time()
    print("getting global pareto points")

    global_pareto_points, global_pareto_points_array, runtimes_vec, areas_vec, all_colors, pareto_mask = get_global_pareto_points(num_vars, gate_type, available_bw_vec, pareto_file_dir, colors=colors, overwrite=False)
  
    end_time = time.time()
    print(f"Time taken for getting global pareto points: {end_time - start_time:.4f} seconds")
  
    divide_factor = 1/4 if gate_type == "jellyfish" else 1

    # Create the main figure
    fig, ax = plt.subplots(figsize=(7, 4.2))

    start_time = time.time()
    print("plotting")
    # Main plot: Overall Pareto curves
    for i, (runtimes, areas) in enumerate(zip(runtimes_vec, areas_vec)):

        r_vec = np.array(runtimes)
        a_vec = np.array(areas)
        if overall_time_limit == None:
            overall_time_limit = 200*(2**(num_vars - 20))/divide_factor
        
        if overall_area_limit == None:
            overall_area_limit = 1e99
        mask = (r_vec < overall_time_limit) & (a_vec < overall_area_limit)
        r_vec = r_vec[mask]
        a_vec = a_vec[mask]

        ax.scatter(r_vec, a_vec, s=15, marker='o', label=f"{available_bw_vec[i]} GB/s", color=colors[i])

    ax.set_xlabel("Runtime (ms)", fontsize=16)
    ax.set_ylabel("Area (mm$^2$)", fontsize=16)
    ax.set_xlim(0, overall_time_limit)
    if overall_area_limit == 1e99:
        ax.set_ylim(0, 800)
    else:
        ax.set_ylim(0, overall_area_limit+50)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], ncol=4, loc='upper center', bbox_to_anchor=(0.531, .99))
    ax.grid(True)

    # Save and show the plot
    save_path = os.path.join(plot_file_path, f"pareto_no_global_inset_{num_vars}vars_{gate_type}.png")
    plt.tight_layout()
    plt.savefig(save_path)
    save_path = os.path.join(plot_file_path, f"pareto_no_global_inset_{num_vars}vars_{gate_type}.pdf")
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    
    end_time = time.time()
    print(f"Time taken for plotting points: {end_time - start_time:.4f} seconds")
    

    # return
    # Add an inset plot for global Pareto-optimal points
    ax_inset = inset_axes(ax, width="55%", height="47%", loc='center right', bbox_to_anchor=(0.21, 0.13, .8, .8), bbox_transform=ax.transAxes, borderpad=2)
    ax_inset.tick_params(axis='both', which='major', labelsize=13)
    group2_point_area = [100, 200, 300, 450]
    group2_point = []  # (runtime, area)
    for color in np.unique(all_colors[pareto_mask]):
        color_mask = (all_colors[pareto_mask] == color)
        points_in_color = global_pareto_points_array[color_mask]

        r_vec = points_in_color[:, 0]
        a_vec = points_in_color[:, 1]
        if inset_time_limit == None:
            inset_time_limit = 50 * (2 ** (num_vars - 20)) / divide_factor

        if inset_area_limit == None:
            inset_area_limit = 1e99

        mask = (r_vec < inset_time_limit) & (a_vec < inset_area_limit)
        r_vec = r_vec[mask]
        a_vec = a_vec[mask]

        ax_inset.scatter(r_vec, a_vec, s=15, marker='o', color=color)

        if color in ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']:
            group2_point.append((r_vec[0], a_vec[0], color_bandwidths[color]))

    # print(group2_point)
    ax_inset.set_xlim(0, inset_time_limit)
    if inset_area_limit == 1e99:
        ax_inset.set_ylim(0, 800)
    else:
        ax_inset.set_ylim(0, inset_area_limit+50)

    # Find the design points with areas closest to group2_point_area
    closest_designs = []
    for target_area in group2_point_area:
        closest_design = min(
            global_pareto_points.items(),
            key=lambda x: abs(x[1]["area"] - target_area)
        )
        closest_designs.append(closest_design)
    

    # Print the closest designs for debugging
    for idx, (design_label, design_data) in enumerate(closest_designs):
        print(f"Group {chr(65 + idx)}: Design {design_label}, Area: {design_data['area']}, Runtime: {design_data['runtime']}, Bandwidth: {design_data['bandwidth']}")
    for idx, (design_label, design_data) in enumerate(closest_designs):
        aa, bb, cc, dd, ee, ff = design_label
        print(f"{aa}, {design_data['bandwidth']}, {bb}, \'{cc}\', {dd}, {ee}, {ff}")
    
    for idx, (runtime, area, bw) in enumerate(group2_point):
        print(f"Group {chr(65 + idx)}: Runtime: {runtime}, Area: {area}, Bandwidth: {bw}")

    for point in group2_point:
        runtime, area, bw = point
        for idx, (design_label, design_data) in enumerate(global_pareto_points.items()):
            if (design_data["runtime"] == runtime and 
                design_data["area"] == area and 
                design_data["bandwidth"] == bw):
                print(f"Matching design found at index {idx}: {design_label}")
                break

    letters = ["A", "C", "D", "B"]
    # for idx, (design_label, design_data) in enumerate(closest_designs):
    for idx, design in enumerate(group2_point):
        # x, y = design_data["runtime"], design_data["area"]
        x, y, _ = design
        if idx == 3:
            ax_inset.annotate(
                f"{letters[idx]}",  # Annotation text
                xy=(x, y),  # Point to highlight
                xytext=(x + 100, y+30),  # Text position (adjust as needed)
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5),  # Arrow properties
                fontsize=16,
                color="black"
            )
        else:
            ax_inset.annotate(
                f"{letters[idx]}",  # Annotation text
                xy=(x, y),  # Point to highlight
                xytext=(x + 80, y-10),  # Text position (adjust as needed)
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5),  # Arrow properties
                fontsize=16,
                color="black"
            )
        if idx == 0:
            ax_inset.scatter(x, y, s=15, marker='o', edgecolors='black', facecolors='#1F77B4', linewidth=1.5)
        if idx == 1:
            ax_inset.scatter(x, y, s=15, marker='o', edgecolors='black', facecolors=colors[-2], linewidth=1.5)
            
        else:
            ax_inset.scatter(x, y, s=15, marker='o', edgecolors='black', facecolors='none', linewidth=1.5)

    
    # ax_inset.set_xlabel("Runtime (ms)", fontsize=12)
    # ax_inset.set_ylabel("Area (mm$^2$)", fontsize=12)
    ax_inset.set_title("Global Pareto-Optimal Points", fontsize=12)
    ax_inset.grid(True)

    # Save and show the plot
    save_path = os.path.join(plot_file_path, f"pareto_with_global_inset_{num_vars}vars_{gate_type}.png")
    plt.tight_layout()
    plt.savefig(save_path)
    save_path = os.path.join(plot_file_path, f"pareto_with_global_inset_{num_vars}vars_{gate_type}.pdf")
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def plot_util_trace(target_design, available_bw_vec, pareto_file_dir, plot_file_path):
    
    num_vars = target_design[0]
    
    for bw in available_bw_vec:
        pareto_file_path = os.path.join(pareto_file_dir, f"{num_vars}vars_{bw}gbs.pkl")
        
        if os.path.exists(pareto_file_path):
            with open(pareto_file_path, 'rb') as f:
                target_design_data = pickle.load(f)['all_designs'][target_design]
        else:
            exit("Error")
        print(target_design_data.keys())
        witness_step = target_design_data['witness_step']
        gate_identity_step = target_design_data['gate_identity_step']
        wire_identity_step = target_design_data['wire_identity_step']
        batch_eval_step = target_design_data['batch_eval_step']
        polyopen_step = target_design_data['polyopen_step']
        final_batch_eval_step = target_design_data['final_batch_eval_step']


        # Step labels
        steps = [
            "Witness Step",
            "Gate Identity",
            "Wire Identity",
            "Batch Eval",
            "PolyOpen",
            "Final Batch Eval"
        ]

        # Categories for bars
        categories = ["MSM", "Sumcheck", "MLE Update", "Multifunction", "ND", "FracMLE", "MLE Combine", "SHA3"]

        # Combine data into a structured dictionary
        data = {
            "Witness Step": witness_step,
            "Gate Identity": gate_identity_step,
            "Wire Identity": wire_identity_step,
            "Batch Eval": batch_eval_step,
            "PolyOpen": polyopen_step,
            "Final Batch Eval": final_batch_eval_step
        }

        # Bar width and positioning
        num_labels = len(categories)
        num_steps = len(steps)
        x_positions = np.arange(num_steps)  # Base x positions for each step
        bar_width = 0.1  # Width of each individual bar

        # Plot each category as a separate bar within each step
        fig, ax = plt.subplots(figsize=(10, 6))

        for i, category in enumerate(categories):
            values = [data[step].get(category, 0) for step in steps]  # Extract values for each step
            ax.bar(x_positions + i * bar_width, values, width=bar_width, label=category)

        # Adjust x-axis
        ax.set_xticks(x_positions + (num_labels / 2) * bar_width)
        ax.set_xticklabels(steps, rotation=30, ha='right')

        # Labels and legend
        ax.set_ylabel("Latency")
        ax.set_title("Latency Breakdown by Step (Side-by-Side Bars)")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Save and show the plot
        save_path = os.path.join(plot_file_path, f"util_trace_{num_vars}vars_{bw}gbps.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close()


def get_relative_util_data(protocol_step, total_protocol_latency, keys):
    """
    Compute the relative utilization of each micro step within a macro step.
    If a key is missing, assume its value is 0.
    """
    for k, v in protocol_step.items():
        print(k, v)
    print()
    return np.array([protocol_step.get(key, 0) / total_protocol_latency for key in keys])

def plot_relative_util_trace(target_design, available_bw_vec, pareto_file_dir, plot_file_path):
    """
    Generate a side-by-side grouped bar plot showing the proportion of total latency spent on each step.
    """
    num_vars = target_design[0]
    
    for bw in available_bw_vec:
        pareto_file_path = os.path.join(pareto_file_dir, f"{num_vars}vars_{bw}gbs.pkl")
        
        if os.path.exists(pareto_file_path):
            with open(pareto_file_path, 'rb') as f:
                target_design_data = pickle.load(f)['all_designs'][target_design]
        else:
            exit("Error: Pareto file not found")

        macro_steps = [
            'witness_step', 'gate_identity_step', 'wire_identity_step',
            'batch_eval_step', 'polyopen_step', 'final_batch_eval_step'
        ]

        # Define micro step categories (excluding 'total')
        categories = ["MSM", "Sumcheck", "MLE Update", "Multifunction", "ND", "FracMLE", "MLE Combine", "SHA3"]
        # categories = ["MSM", "Sumcheck", "MLE Update", "Multifunction", "ND", "FracMLE", "MLE Combine"]
        
        # Compute total latency
        total_protocol_latency = sum(
            target_design_data[step]['total'] for step in macro_steps if step != 'batch_eval_step'
        )
        print(total_protocol_latency)
        # Combine batch_eval_step and polyopen_step into a single entry
        combined_macro_steps = [
            'witness_step', 'gate_identity_step', 'wire_identity_step',
            'batch_eval_polyopen_step'
        ]


        # Compute relative utilization data, merging batch_eval_step and polyopen_step
        relative_util_data_matrix = np.array([
            get_relative_util_data(target_design_data[step], total_protocol_latency, categories)
            if step != 'batch_eval_polyopen_step' else
            get_relative_util_data(target_design_data['batch_eval_step'], total_protocol_latency, categories) +
            get_relative_util_data(target_design_data['polyopen_step'], total_protocol_latency, categories) + 
            get_relative_util_data(target_design_data['final_batch_eval_step'], total_protocol_latency, categories)
            for step in combined_macro_steps
        ])
        print()
        print(np.round(relative_util_data_matrix, 3))
        print()
        print(np.round(np.sum(relative_util_data_matrix, axis=0), 3))

        # Add the first row as the sum of all rows
        relative_util_data_matrix = np.vstack([np.sum(relative_util_data_matrix, axis=0), relative_util_data_matrix])


        # rename the categories when labeling
        labels = ["MSM", "Sumcheck", "MLE Update", "Multifunction", "Construct N&D", "FracMLE", "MLE Combine", "SHA3"]

        combined_macro_steps = [
            'Total', 'Witness MSMs', 'Gate Identity', 'Wire Identity', 'Batch Eval\n& Poly Open'
        ]

        # exit()

        # Plot setup
        num_labels = len(labels)
        num_macro_steps = len(combined_macro_steps)
        x_positions = np.arange(num_macro_steps)  # X positions for each macro step
        bar_width = 0.1  # Width of each label bar within a macro step

        # Create the grouped bar chart
        fig, ax = plt.subplots(figsize=(7, 5))

        for i, label in enumerate(labels):
            micro_heights = relative_util_data_matrix[:, i]

            # Offset each label's bars within each macro step
            ax.bar(x_positions + i * bar_width, micro_heights, width=bar_width, label=label)

        # Labels and formatting
        ax.set_xticks(x_positions + (num_labels / 2) * bar_width)
        ax.set_xticklabels(combined_macro_steps, rotation=30, ha='right')

        ax.set_ylabel("Utilization")
        ax.set_title(f"Utilization breakdown for {num_vars} Vars, {bw} GB/s")

        # Legend outside the plot
        ax.legend()

        # Save and show the plot
        save_path = os.path.join(plot_file_path, f"relative_util_trace_{num_vars}vars_{bw}gbps.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close()

        ################################################

        step_labels = [step for step in macro_steps if step not in ['batch_eval_step', 'final_batch_eval_step']]
        
        step_values = [
            target_design_data[step]['total'] if step != 'polyopen_step' 
            else target_design_data['polyopen_step']['total'] + target_design_data['final_batch_eval_step']['total']
            for step in step_labels
        ]

        step_labels = [
            'Witness\nMSMs', 'Gate\nIdentity', 'Wire Identity', 'Batch Evals &\n Poly Open'
        ]
        macro_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']

        # Create a simple pie chart
        plt.figure(figsize=(5, 4))
        plt.pie(
            step_values, 
            labels=step_labels, 
            autopct='%.1f%%', 
            startangle=128, 
            # colors=plt.cm.Set3.colors,
            colors=macro_colors,
            wedgeprops={'edgecolor': 'black'}
        )

        plt.title(f"zkSpeed Runtime Breakdown for $\mu$ = {num_vars} at {bw} GB/s")
        plt.tight_layout()
        plt.show()
        save_path = os.path.join(plot_file_path, f"stepwise_breakdown_{num_vars}vars_{bw}gbps.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close()

def plot_zkspeed_runtime_breakdown(target_design, available_bw_vec, pareto_file_dir, plot_file_path):
    """
    Generate a side-by-side grouped bar plot showing the proportion of total latency spent on each step.
    """
    num_vars = target_design[0]
    
    for bw in available_bw_vec:
        pareto_file_path = os.path.join(pareto_file_dir, f"{num_vars}vars_{bw}gbs.pkl")
        
        if os.path.exists(pareto_file_path):
            with open(pareto_file_path, 'rb') as f:
                target_design_data = pickle.load(f)['all_designs'][target_design]
        else:
            exit("Error: Pareto file not found")

        macro_steps = [
            'witness_step', 'gate_identity_step', 'wire_identity_step',
            'batch_eval_step', 'polyopen_step', 'final_batch_eval_step'
        ]

        step_labels = [step for step in macro_steps if step not in ['batch_eval_step', 'final_batch_eval_step']]
        
        step_values = [
            target_design_data[step]['total'] if step != 'polyopen_step' 
            else target_design_data['polyopen_step']['total'] + target_design_data['final_batch_eval_step']['total']
            for step in step_labels
        ]

        step_labels = [
            'Witness\nMSMs', 'Gate\nIdentity', 'Wire Identity', 'Batch Evals &\n Poly Open'
        ]
        macro_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']

        # Create a simple pie chart
        plt.figure(figsize=(5, 4))
        plt.pie(
            step_values, 
            labels=step_labels, 
            autopct='%.1f%%', 
            startangle=128, 
            # colors=plt.cm.Set3.colors,
            colors=macro_colors,
            wedgeprops={'edgecolor': 'black'}
        )

        plt.title(f"zkSpeed Runtime Breakdown for $\mu$ = {num_vars} at {bw} GB/s")
        plt.tight_layout()
        plt.show()
        save_path = os.path.join(plot_file_path, f"stepwise_breakdown_{num_vars}vars_{bw}gbps.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close()

def darken_color(color, factor=0.7):
    rgb = mcolors.to_rgb(color)  # Convert to RGB
    darker_rgb = tuple(max(min(c * factor, 1.0), 0.0) for c in rgb)  # Darken each channel
    return mcolors.to_hex(darker_rgb)

def plot_relative_util_trace_stacked(target_design, available_bw_vec, pareto_file_dir, plot_file_path):
    """
    Generate a stacked bar plot showing the proportion of total latency spent on each step.
    """
    num_vars = target_design[0]

    for bw in available_bw_vec:
        pareto_file_path = os.path.join(pareto_file_dir, f"{num_vars}vars_{bw}gbs.pkl")

        if os.path.exists(pareto_file_path):
            with open(pareto_file_path, 'rb') as f:
                target_design_data = pickle.load(f)['all_designs'][target_design]
        else:
            exit("Error: Pareto file not found")

        macro_steps = [
            'witness_step', 'gate_identity_step', 'wire_identity_step',
            'batch_eval_step', 'polyopen_step', 'final_batch_eval_step'
        ]

        # Define micro step categories (excluding 'total')
        categories = ["MSM", "Sumcheck", "MLE Update", "Multifunction", "ND", "FracMLE", "MLE Combine", "SHA3"]


        # Compute total latency
        total_protocol_latency = sum(
            target_design_data[step]['total'] for step in macro_steps if step != 'batch_eval_step'
        )
        print(total_protocol_latency)
        # Combine batch_eval_step and polyopen_step into a single entry
        combined_macro_steps = [
            'witness_step', 'gate_identity_step', 'wire_identity_step',
            'batch_eval_polyopen_step'
        ]

        # Compute relative utilization data, merging batch_eval_step and polyopen_step
        relative_util_data_matrix = np.array([
            get_relative_util_data(target_design_data[step], total_protocol_latency, categories)
            if step != 'batch_eval_polyopen_step' else
            get_relative_util_data(target_design_data['batch_eval_step'], total_protocol_latency, categories) +
            get_relative_util_data(target_design_data['polyopen_step'], total_protocol_latency, categories) +
            get_relative_util_data(target_design_data['final_batch_eval_step'], total_protocol_latency, categories)
            for step in combined_macro_steps
        ]) * 100
        print()
        print(np.round(relative_util_data_matrix, 3))
        print()
        print(np.round(np.sum(relative_util_data_matrix, axis=0), 3))

        # rename the categories when labeling
        labels = ["MSM", "Sumcheck", "MLE Update", "Multifunction", "Construct N&D", "FracMLE", "MLE Combine", "SHA3"]

        combined_macro_steps = [
            'Witness MSMs', 'Gate Identity', 'Wire Identity', 'Batch Eval\n& Poly Open'
        ]

        # area calcs
        area_breakdown = target_design_data['detailed_area_breakdown']
        total_logic_area = area_breakdown['total_logic_area']

        msm_logic_area_portion = round(area_breakdown['msm_logic_area'] / total_logic_area, 4)
        nd_area_portion = round(area_breakdown['nd_area'] / total_logic_area, 4)
        frac_mle_area_portion = round(area_breakdown['frac_mle_area'] / total_logic_area, 4)
        mle_combine_area_portion = round(area_breakdown['mle_combine_area'] / total_logic_area, 4)
        sha_area_portion = round(area_breakdown['sha_area'] / total_logic_area, 4)
        sumcheck_core_area_portion = round(area_breakdown['sumcheck_core_area'] / total_logic_area, 4)
        mle_update_core_area_portion = round(area_breakdown['mle_update_core_area'] / total_logic_area, 4)
        multifunction_tree_area_portion = round(area_breakdown['multifunction_tree_area'] / total_logic_area, 4)

        area_portions = [msm_logic_area_portion, sumcheck_core_area_portion, mle_update_core_area_portion, multifunction_tree_area_portion, nd_area_portion, frac_mle_area_portion, mle_combine_area_portion, sha_area_portion]

        # Plot setup
        bar_width = 0.8  # Width of each label bar within a macro step
        fig, ax = plt.subplots(figsize=(6, 2.7))
        x = np.arange(len(labels))  # X positions for bars
        # colors = ["#F8CECC", "#B9E0A5", "#DAE8FC", "#E1D5E7", "#FFE6CC"]
        # colors = [darken_color(c, factor=0.7) for c in colors]
        # colors = ["red", "green", "blue", "purple", "orange"]

        # colors = ["#EE8183", "#6CCBAB", "#6399E8", "#B19EE1", "#F4B07F"]
        colors = ["#EE8183", "#B9E0A5", "#6399E8", "#B19EE1", "#F4B07F"]

        # Plot stacked bars
        bottom = np.zeros(len(labels))
        for i in range(len(combined_macro_steps)):
            ax.bar(x, relative_util_data_matrix[i], label=combined_macro_steps[i], bottom=bottom, width=bar_width, color=colors[i], alpha=1)
            bottom += relative_util_data_matrix[i]  # Update bottom for next stack

        # temp harcoded values
        area_vals = [105.64, 24.96, 5.84, 12.28, 1.35, 1.92, 9.56, 0.00]
        area_portions = [v / 163.53 for v in area_vals]
        for i, area_portion in enumerate(area_portions):
            ax.text(
                x[i] + .05 if i == 0 else x[i],  # X-coordinate: same as bar center
                bottom[i] - .3 if i == 0 else bottom[i],  # Y-coordinate: top of the bar
                f"{area_portion*100:.2f}% AU",  # Annotation text (formatted to 2 decimals)
                ha='center',  # Horizontal alignment: center
                va='bottom',  # Vertical alignment: bottom of text at top of bar
                fontsize=7,
                fontweight='bold',
                color='black'
            )

        # Formatting
        ax.grid(axis='y', linestyle='dotted')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20)
        ax.set_yticks(np.arange(0, 80, 10))
        ax.set_ylabel("Utilization %")
        # ax.set_title(f"Utilization breakdown for {num_vars} Vars, {bw} GB/s")
        ax.legend()
        plt.tight_layout()
        plt.show()
        save_path = os.path.join(plot_file_path, f"relative_util_trace_{num_vars}vars_{bw}gbps.png")
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        save_path = os.path.join(plot_file_path, f"relative_util_trace_{num_vars}vars_{bw}gbps.pdf")
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close()


# Add subcomponent divisions as annotations (within the Wire Identity and Polynomial Opening slices)
# Function to annotate subcomponents inside a slice
def annotate_subcomponents(center_angle, radius, sub_percentages, sub_labels, colors):
    angle = center_angle - sum(sub_percentages) / 2
    for i, (sub_pct, sub_label) in enumerate(zip(sub_percentages, sub_labels)):
        angle += sub_pct / 2
        x = np.cos(np.deg2rad(angle)) * radius
        y = np.sin(np.deg2rad(angle)) * radius
        plt.text(x, y, f"{sub_label}\n{sub_pct:.1f}%", ha='center', fontsize=7, bbox=dict(facecolor='white', alpha=0.5))
        angle += sub_pct / 2

def plot_cpu_runtime_breakdown(num_vars, num_th, data_path_prefix, plot_file_path):

    file_path = data_path_prefix + f'cpu_runtime_data/cpu_runtime_{num_vars}_vars_{num_th}_th.pkl'
    print(file_path)
    # Check if the file exists and load the Pareto points
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            cpu_nums = pickle.load(f)

    cpu_latency_steps_wise = np.array([
    cpu_nums["witness_commit"],
    cpu_nums['zerocheck'],
    cpu_nums["permcheck_total"],
    cpu_nums["batch_eval"],
    cpu_nums["poly_open_total"]])
    labels = ['Witness\nMSMs', 'Gate\nIdentity', 'Wiring\nIdentity', 'Batch\nEvaluations', 'Polynomial\nOpening']

    overall_runtime = cpu_nums["total"]
    relative_portion_steps_wise = cpu_latency_steps_wise / overall_runtime
    print(sum(relative_portion_steps_wise))
    print(cpu_latency_steps_wise)

    
    plt.figure(figsize=(5, 4))
    plt.pie(relative_portion_steps_wise, labels=labels, autopct='%1.1f%%', startangle=140, wedgeprops={"edgecolor":"k", 'linewidth': 0.5})
    plt.title(f"CPU Runtime Breakdown for $\mu$ = {num_vars} at {num_th} threads")
    plt.tight_layout()
    save_path = os.path.join(plot_file_path, f"{num_vars}vars_steps_wise_{num_th}th.png")
    plt.savefig(save_path)
    plt.close()

    # cpu_nums["witness_commit"],
    # cpu_nums['zerocheck'],
    # cpu_nums["nd"], cpu_nums["frac_mle"], cpu_nums["prod_mle"], cpu_nums['wiring_msms'],
    # cpu_nums["batch_eval"],
    # cpu_nums["permcheck"],
    # cpu_nums["mle_combine"], cpu_nums["build_mle_opencheck"], cpu_nums["point_merge_and_g_prime"], cpu_nums["opencheck"], cpu_nums["poly_open_msms"]


    # Stack Group 1
    stack1 = [
        cpu_nums["nd"],
        cpu_nums["frac_mle"],
        cpu_nums["prod_mle"],
        cpu_nums["wiring_msms"],
        cpu_nums["permcheck"]
    ]
    labels_group1 = ["Construct N/D", "FracMLE", "ProdMLE", "Dense MSMs", "PermCheck"]
    # Stack Group 2
    stack2 = [
        cpu_nums["mle_combine"],
        cpu_nums["build_mle_opencheck"],
        cpu_nums["point_merge_and_g_prime"],
        cpu_nums["opencheck"],
        cpu_nums["poly_open_msms"]
    ]
    labels_group2 = ["MLE Combine", "Build 6 MLEs", "Point Combine and Build g'", "OpenCheck", "Dense MSMs"]

    witness_commit = cpu_nums["witness_commit"]
    zerocheck = cpu_nums["zerocheck"]
    batch_eval = cpu_nums["batch_eval"]

    # X positions for bars
    x_positions = np.arange(5)  # 5 bars total

    # Create the plot
    plt.figure(figsize=(7, 5))

    # Witness Commit (Single Bar)
    plt.bar(x_positions[0], witness_commit, label="Sparse MSM")

    # Zerocheck (Single Bar)
    plt.bar(x_positions[1], zerocheck, label="ZeroCheck")

    # Stack Group 1
    bottom_val = 0
    for i, val in enumerate(stack1):
        plt.bar(x_positions[2], val, bottom=bottom_val, label=labels_group1[i])
        bottom_val += val

    # Batch Eval (Single Bar)
    plt.bar(x_positions[3], batch_eval, label="Batch Eval")

    # Stack Group 2
    bottom_val = 0
    for i, val in enumerate(stack2):
        plt.bar(x_positions[4], val, bottom=bottom_val, label=labels_group2[i])
        bottom_val += val

    # Add labels
    plt.xticks(x_positions, ["Witness MSMs", "Gate Identity", "Wire Identity", "Batch Evaluations", "Polynomial Opening"], rotation=20)
    plt.ylabel("CPU Time (ms)")
    plt.title(f"CPU Runtime Breakdown for $\mu$ = {num_vars} at {num_th} threads")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    save_path = os.path.join(plot_file_path, f"{num_vars}vars_stacked_bar_{num_th}th.png")
    plt.savefig(save_path)

    subcomponent_labels = [
        "Sparse MSMs", "Gate Identity", 
        "Construct\n MLEs", "Dense MSMs", "PermCheck",
        "Batch Evals",
        "Construct MLEs", "OpenCheck", "Dense MSMs"
    ]

    subcomponent_sizes = np.array([
        cpu_nums["witness_commit"],
        cpu_nums['zerocheck'],
        cpu_nums["nd"] + cpu_nums["frac_mle"] + cpu_nums["prod_mle"], cpu_nums['wiring_msms'], cpu_nums["permcheck"],
        cpu_nums["batch_eval"],
        cpu_nums["mle_combine"] + cpu_nums["build_mle_opencheck"] + cpu_nums["point_merge_and_g_prime"], cpu_nums["opencheck"], cpu_nums["poly_open_msms"]
    ]) / cpu_nums["total"]


    macro_colors = {
        "Witness": '#1f77b4',       # Blue
        "Gate": '#ff7f0e',          # Orange
        "Wiring": '#2ca02c',        # Green
        "Batch": '#d62728',         # Red
        "Polynomial": '#9467bd'     # Purple
    }
    # Custom colors
    subcomponent_colors = [
    macro_colors["Witness"],  # Witness
    macro_colors["Gate"],     # Gate
    *[macro_colors["Wiring"]] * 3,  # Wiring subcomponents
    macro_colors["Batch"],    # Batch
    *[macro_colors["Polynomial"]] * 3  # Polynomial subcomponents
    ]


    # Create the pie chart
    plt.figure(figsize=(5, 4))
    wedges, texts, autotexts = plt.pie(
        subcomponent_sizes, labels=subcomponent_labels, autopct='%.1f%%', startangle=128,
        colors=subcomponent_colors, wedgeprops={'edgecolor': 'black'}, pctdistance=.75,
        textprops={'fontsize': 10}
    )

    if num_vars == 20 and num_th == 1:
        # Shift percentage labels slightly downward
        for idx, autotext in enumerate(autotexts):
            x, y = autotext.get_position()
            if idx == 2:
                # autotext.set_position((x - 0.15, y - 0.01))  # Lower by adjusting the y-position
                # autotext.set_position((x, y - 0.09))  # Lower by adjusting the y-position
                plt.annotate(
                    autotext.get_text(), 
                    xy=(x - 0.15, y + 0.01), 
                    xytext=(x - 0.1, y - 0.2),  # Position label offset
                    arrowprops=dict(arrowstyle="-", linewidth=1, color='black')
                )
                autotext.set_visible(False)  # Hide default text, since annotated

            elif idx == 5:
                autotext.set_position((x + 0.05, y - 0.01))  # Lower by adjusting the y-position
            elif idx == 6:
                autotext.set_position((x + 0.05, y - 0.01))  # Lower by adjusting the y-position
            elif idx == 7:
                autotext.set_position((x + 0.05, y))  # Lower by adjusting the y-position
            

    # Custom Legend: Only show legend entries for selected macro steps
    custom_legend_labels = ["Witness MSMs", "Gate Identity", "Wiring Identity", "Batch Evals", "Poly Opening"]
    custom_legend_colors = [subcomponent_colors[i] for i in [0, 1, 2, 5, 8]]

    plt.legend(
        handles=[plt.Line2D([0], [0], color=c, lw=6) for c in custom_legend_colors],
        labels=custom_legend_labels,
        # , loc = "lower right"
        # , bbox_to_anchor=(.95, 1), loc='lower right', fontsize=8
        bbox_to_anchor=(1.24, 0),  # (x, y): x=1.02 means just outside the plot, y=0 means bottom
        loc='lower right',
        fontsize=10
    )

    plt.title(f"CPU Runtime Breakdown for $\mu$ = {num_vars} at {num_th} threads")
    plt.tight_layout()
    save_path = os.path.join(plot_file_path, f"{num_vars}vars_pie_{num_th}th.png")
    plt.savefig(save_path)

def plot_cpu_runtime_breakdown_other(num_vars, num_th, data_path_prefix, plot_file_path):
    
    cpu_latency_list = np.array([
    cpu_nums["witness_commit"],
    cpu_nums['zerocheck'],
    cpu_nums["nd"] + cpu_nums["frac_mle"] + cpu_nums["prod_mle"] + cpu_nums['wiring_msms'],
    cpu_nums["permcheck"],
    cpu_nums["opencheck"],
    cpu_nums["poly_open_msms"]])

    cpu_latency_list_compact = np.array([
        cpu_nums["witness_commit"] + cpu_nums['wiring_msms'] + cpu_nums["poly_open_msms"],
        cpu_nums['zerocheck'] + cpu_nums["permcheck"] + cpu_nums["opencheck"],
        cpu_nums["nd"] + cpu_nums["frac_mle"] + cpu_nums["prod_mle"]
    ])

    labels = ['Witness MSMs', 'Gate Identity\nZeroCheck', 'Wiring Identity\nMLEs and MSMs', 'Wiring Identity\nPermcheck', 'Polynomial Opening\nSumCheck', "Polynomial Opening\nMSMs"]
    labels_compact = ['MSMs', 'SumCheck', 'Rest']
    
    overall_runtime = cpu_nums["total"]
    relative_portion = cpu_latency_list/overall_runtime
    relative_portion_compact = cpu_latency_list_compact/overall_runtime
    
    colors = np.array(["#6399E8", "#EE8183", "#6CCBAB", "#F4B07F"])

    # Plotting the pie chart
    plt.figure(figsize=(8, 6))
    plt.pie(relative_portion, labels=labels, autopct='%1.1f%%', startangle=140, wedgeprops={"edgecolor":"k", 'linewidth': 0.5})
    # plt.title(f"Relative Runtimes on CPU for $2^{{{num_vars}}}$ Variables")
    plt.tight_layout()
    save_path = os.path.join(plot_file_path, f"{num_vars}vars_{num_th}th.png")
    plt.savefig(save_path)
    plt.close()

    plt.figure(figsize=(3, 3))
    # plt.pie(relative_portion_compact, labels=labels_compact, autopct='%1.1f%%', colors=colors[:len(relative_portion_compact)], startangle=140, wedgeprops={"edgecolor":"k", 'linewidth': 0.5})
    wedges, texts, autotexts = plt.pie(
    relative_portion_compact,
    labels=labels_compact,
    autopct='%1.1f%%',
    colors=colors[:len(relative_portion_compact)],
    startangle=140,
    wedgeprops={"edgecolor": "k", 'linewidth': 0.5},
    # explode=explode,  # Apply the explosion
    pctdistance=0.85,  # Adjust the position of percentage labels
    labeldistance=1.1  # Adjust the position of the labels
    )
    pctdists = [1.34 if label == "Rest" else 0.85 for label in labels_compact]

    # Adjust positions of percentage texts based on custom pctdists
    for autotext, pctdist in zip(autotexts, pctdists):
        # Get current position of the autotext
        x, y = autotext.get_position()
        # Calculate the distance from the origin
        r = np.sqrt(x**2 + y**2)
        # Calculate the angle (phi) in polar coordinates
        phi = np.arctan2(y, x)
        # Update position based on desired distance
        new_x = pctdist * r * np.cos(phi)
        new_y = pctdist * r * np.sin(phi)
        autotext.set_position((new_x, new_y))


    # Adjust positions of labels based on custom label distances
    for text in texts:
        if text.get_text() == "Rest":
            x, y = text.get_position()
            new_x = x + .1
            new_y = y + .23
            text.set_position((new_x, new_y))
        elif text.get_text() == "SumCheck":
            x, y = text.get_position()
            new_x = x + .45
            new_y = y + .05
            text.set_position((new_x, new_y))

    # # plt.title(f"CPU Runtime breakdown for $2^{{{num_vars}}}$ Gates")
    # plt.title(f"CPU Runtime Breakdown")
    plt.tight_layout()
    save_path = os.path.join(plot_file_path, f"{num_vars}vars_compact_{num_th}th.png")
    plt.savefig(save_path)
    plt.close()



def plot_cpu_zkspeed_runtime_breakdown(
    target_design, bw, pareto_file_dir, 
    num_vars, num_th, data_path_prefix, plot_file_path
):
    """
    Generate a combined plot with two side-by-side pie charts:
    1. zkSpeed Runtime Breakdown
    2. CPU Runtime Breakdown
    """
    fig, axs = plt.subplots(1, 2, figsize=(7, 4))


    # ----------------------- CPU RUNTIME BREAKDOWN (LEFT PIE) -----------------------
    cpu_file_path = data_path_prefix + f'cpu_runtime_data/cpu_runtime_{num_vars}_vars_{num_th}_th.pkl'
    
    if os.path.exists(cpu_file_path):
        with open(cpu_file_path, 'rb') as f:
            cpu_nums = pickle.load(f)
    else:
        exit("Error: CPU runtime data file not found")

    labels = [
        "Sparse\nMSMs", "Gate\nIdentity", 
        "Create\nPermCheck\nMLEs", "PermCheck Dense MSMs", "PermCheck",
        "Batch Evals",
        "MLE Combine", "OpenCheck", "Poly Open Dense MSMs"
    ]

    relative_portion_steps_wise = np.array([
        cpu_nums["witness_commit"],
        cpu_nums['zerocheck'],
        cpu_nums["nd"] + cpu_nums["frac_mle"] + cpu_nums["prod_mle"], cpu_nums['wiring_msms'], cpu_nums["permcheck"],
        cpu_nums["batch_eval"],
        cpu_nums["mle_combine"] + cpu_nums["build_mle_opencheck"] + cpu_nums["point_merge_and_g_prime"], cpu_nums["opencheck"], cpu_nums["poly_open_msms"]
    ]) / cpu_nums["total"]


    # macro_colors = {
    #     "Witness": '#1f77b4',       # Blue
    #     "Gate": '#ff7f0e',          # Orange
    #     "Wiring": '#2ca02c',        # Green
    #     "Batch": '#d62728',         # Red
    #     "Polynomial": '#9467bd'     # Purple
    # }

    macro_colors = {
        "Witness": '#EE8183',       # Red
        "Gate": '#B9E0A5',          # Green
        "Wiring": '#6399E8',        # Blue
        "Batch": '#F4B07F',         # Purple
        "Polynomial": '#B19EE1'     # Orange
    }


    # Custom colors
    subcomponent_colors = [
    macro_colors["Witness"],  # Witness
    macro_colors["Gate"],     # Gate
    *[macro_colors["Wiring"]] * 3,  # Wiring subcomponents
    macro_colors["Batch"],    # Batch
    *[macro_colors["Polynomial"]] * 3  # Polynomial subcomponents
    ]

    wedges, texts, autotexts = axs[0].pie(
        relative_portion_steps_wise, 
        labels=labels, 
        autopct='%1.1f%%', 
        startangle=128, 
        wedgeprops={"edgecolor":"k", 'linewidth': 0.5},
        colors = subcomponent_colors,
        pctdistance=.8
    )

    if num_vars == 20 and num_th == 1:
        for idx, (autotext, text) in enumerate(zip(autotexts, texts)):
            x, y = autotext.get_position()
            label_x, label_y = text.get_position()
            x_text, y_text = text.get_position()

            if idx == 2:
                # Adjust percentage label (autotext)
                autotext.set_position((x, y - 0.15))
                
                # Adjust pie label (text)
                text.set_position((label_x + 0.16, label_y - 0.48))

                # Add annotation line
                axs[0].annotate(
                    autotext.get_text(), 
                    xy=(x - 0.15, y + 0.01), 
                    xytext=(x - 0.1, y - 0.2),
                    arrowprops=dict(arrowstyle="-", linewidth=1, color='black')
                )
                autotext.set_visible(False)  # Hide default percentage text
                
                axs[0].annotate(
                    text.get_text(), 
                    xy=(x-.2, y), # what we are pointing to
                    xytext=(x_text - .4, y_text - 1), # location of label text
                    arrowprops=dict(arrowstyle="-", linewidth=1, color='black')
                )
                text.set_visible(False)  # Hide default percentage text
                
            elif idx == 3:

                # Adjust pie label (text)
                text.set_position((label_x + 1, label_y))

            elif idx == 5:
                autotext.set_position((x + 0.09, y ))  # Lower by adjusting the y-position
            elif idx == 6:
                autotext.set_position((x + 0.05, y - 0.01))  # Lower by adjusting the y-position
            elif idx == 7:
                autotext.set_position((x + 0.05, y-.01))  # Lower by adjusting the y-position
        
            elif idx == len(labels) - 1:
                text.set_position((label_x - 0.9, label_y))
                

    # --------------------- ZKSPEED RUNTIME BREAKDOWN (RIGHT PIE) ---------------------
    pareto_file_path = os.path.join(pareto_file_dir, f"{num_vars}vars_{bw}gbs.pkl")
    
    if os.path.exists(pareto_file_path):
        with open(pareto_file_path, 'rb') as f:
            target_design_data = pickle.load(f)['all_designs'][target_design]
    else:
        exit("Error: zkSpeed Pareto file not found")

    macro_steps = [
        'witness_step', 'gate_identity_step', 'wire_identity_step',
        'batch_eval_step', 'polyopen_step', 'final_batch_eval_step'
    ]

    step_labels = [step for step in macro_steps if step not in ['batch_eval_step', 'final_batch_eval_step']]
    
    step_values = [
        target_design_data[step]['total'] if step != 'polyopen_step' 
        else target_design_data['polyopen_step']['total'] + target_design_data['final_batch_eval_step']['total']
        for step in step_labels
    ]

    step_labels = [
        'Witness\nMSMs', 'Gate\nIdentity', 'Wire Identity', 'Batch Evals & Poly Open'
    ]
    # macro_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']
    macro_colors = ["#EE8183", "#B9E0A5", "#6399E8", "#B19EE1", "#F4B07F"]

    wedges, texts, autotexts = axs[1].pie(
        step_values, 
        labels=step_labels, 
        autopct='%.1f%%', 
        startangle=128, 
        colors=macro_colors,
        wedgeprops={'edgecolor': 'black', 'linewidth': 0.5},
        pctdistance=.78
    )
    bw = float(bw)
    if bw >= 1024:
        if bw.is_integer():
            bw = int(bw)
            bw_str = f"{bw//1024} TB/s"
        else:
            bw_str = f"{bw/1024} TB/s"
            
    else:
        bw_str = f"{bw} GB/s"

    if num_vars == 20 and bw == 2048:
        for idx, (autotext, text) in enumerate(zip(autotexts, texts)):
            x, y = autotext.get_position()
            label_x, label_y = text.get_position()
            x_text, y_text = text.get_position()
        
            if idx == 2:
                text.set_position((label_x - 0.4, label_y))
            elif idx == len(step_labels) - 1:
                text.set_position((label_x - 1.33, label_y + .1))

    # ----------------------- FINAL PLOT SETTINGS -----------------------
    
    axs[0].set_xlabel(f"a) CPU", fontsize=14)
    axs[1].set_xlabel(f"b) zkSpeed with {bw_str}", fontsize=14)

    plt.tight_layout()

    save_path = os.path.join(plot_file_path, f"combined_runtime_breakdown_{num_vars}vars_{num_th}th_{bw}gbps.png")
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    save_path = os.path.join(plot_file_path, f"combined_runtime_breakdown_{num_vars}vars_{num_th}th_{bw}gbps.pdf")
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()

def plot_ablation_study_old(latencies, plot_file_path):

    labels = [
        "zkspeed",
        "Global SRAM & Arb prime",
        "Global SRAM & Fixed prime",
        "No Global SRAM & Arb prime",
        "No Global SRAM & Fixed prime",
        "zkspeed",
        "Global SRAM & Arb prime",
        "Global SRAM & Fixed prime",
        "No Global SRAM & Arb prime",
        "No Global SRAM & Fixed prime",
    ]

    # Normalize latencies to respective zkSpeed numbers
    normalized_latencies = np.array([
        latencies[i] / latencies[0] if i < 5 else latencies[i] / latencies[5]
        for i in range(len(latencies))
    ])

    # Separate latencies for 2^20 and 2^24
    latencies_20 = latencies[:5]
    latencies_24 = latencies[5:]

    # Normalize latencies to zkSpeed for 2^20 and 2^24
    speedups_20 = latencies_20[0] / latencies_20
    speedups_24 = latencies_24[0] / latencies_24 
    colors = ["#778898", "#6399E8", "#EE8183", "#6CCBAB", "#B19EE1", "#F4B07F"]
    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(7, 4))

    # Plot normalized latencies for 2^20
    x_20 = np.arange(len(latencies_20))
    bars1 = axs[0].bar(
        x_20, 
        speedups_20, 
        color=[colors[1] if i == 0 else colors[2] for i in range(len(latencies_20))]
    )
    axs[0].set_xticks(x_20)
    axs[0].set_xticklabels(labels[:5], rotation=45, ha="right")
    axs[0].set_ylabel("Speedup")
    axs[0].set_title("Speedup (2$^{20}$ Gates)")
    axs[0].set_ylim(0, 2.3)
    for bar in bars1:
        yval = bar.get_height()
        axs[0].text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.1f}x", ha='center', va='bottom', fontsize=8)

    # Plot normalized latencies for 2^24
    x_24 = np.arange(len(latencies_24))
    bars2 = axs[1].bar(
        x_24, 
        speedups_24, 
        color=[colors[3] if i == 0 else colors[4] for i in range(len(latencies_24))]
    )
    axs[1].set_xticks(x_24)
    axs[1].set_xticklabels(labels[5:], rotation=45, ha="right")
    axs[1].set_ylabel("Normalized Latency")
    axs[1].set_title("Normalized Latency (2$^{24}$ Gates)")
    for bar in bars2:
        yval = bar.get_height()
        axs[1].text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.1f}x", ha='center', va='bottom', fontsize=8)
    axs[1].set_ylim(0, 2.3)

    plt.tight_layout()
    save_path = os.path.join(plot_file_path, f"ablation_study_normalized.png")
    plt.savefig(save_path, dpi=100, bbox_inches="tight")


def plot_ablation_study(latencies, plot_file_path):

    labels = [
        "zkspeed",
        "Global SRAM\n& Arb prime",
        "Global SRAM\n& Fixed prime",
        "No Global SRAM\n& Arb prime",
        "No Global SRAM\n& Fixed prime",
        # "No Global SRAM\n& Fixed prime\n& Jellyfish"
    ]
    
    # 2^24 
    xticks = np.arange(len(latencies))
    speedups = latencies[0] / latencies 
    colors = ["#778898", "#6399E8", "#EE8183", "#6CCBAB", "#B19EE1", "#F4B07F"]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(
        xticks, 
        speedups, 
        color=[colors[1] if i == 0 else colors[2] for i in range(len(latencies))]
    )
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Speedup")
    ax.set_title("Speedup for 2$^{24}$ Gates")
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.2f}x", ha='center', va='bottom', fontsize=8)
    ax.set_ylim(0, 2.3)
    # ax.set_ylim(0, 45)

    plt.tight_layout()
    save_path = os.path.join(plot_file_path, f"ablation_study_normalized_2_24.png")
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    save_path = os.path.join(plot_file_path, f"ablation_study_normalized_2_24.pdf")
    plt.savefig(save_path, dpi=100, bbox_inches="tight")

def plot_ablation_study(latencies, plot_file_path):
    labels = [
        "zkspeed",
        "Global SRAM\n& Arb prime",
        "Global SRAM\n& Fixed prime",
        "No Global SRAM\n& Arb prime",
        "No Global SRAM\n& Fixed prime",
        "+ Jellyfish"
    ]

    speedups = latencies[0] / latencies
    xticks = np.arange(len(latencies))
    colors = ["#778898", "#6399E8", "#EE8183", "#6CCBAB", "#B19EE1", "#F4B07F"]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(5.25, 4), gridspec_kw={'height_ratios': [1, 3]})
    fig.subplots_adjust(hspace=0.05)  # adjust space between Axes

    # Define axis limits
    ax1.set_ylim(3, 52)    # Top part (for extreme value)
    ax2.set_ylim(0, 2.3)   # Bottom part

    # Plot bars on both axes
    bars1 = ax1.bar(xticks, speedups, color=colors)
    bars2 = ax2.bar(xticks, speedups, color=colors)

    # Zigzag effect
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.tick_params(bottom=False)  # Hide x-ticks on upper plot

    # Break mark settings
    kwargs = dict(marker=[(-1, -1), (1, 1)], markersize=10,
                linestyle="none", color='k', clip_on=False)

    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    # Labels and formatting
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(labels, rotation=45, ha="right")
    ax2.set_ylabel("Speedup")
    ax1.set_title("Speedup for Problem with 2$^{24}$ Vanilla Gates")

    # Annotate bars
    for ax in [ax1, ax2]:
        for bar in ax.patches:
            yval = bar.get_height()
            if (ax == ax1 and yval > 3) or (ax == ax2 and yval <= 2.3):
                ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.2f}x", 
                        ha='center', va='bottom', fontsize=8)

    # plt.tight_layout()
    os.makedirs(plot_file_path, exist_ok=True)
    for ext in ["png", "pdf"]:
        save_path = os.path.join(plot_file_path, f"ablation_study_broken_yaxis_2_24.{ext}")
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.show()

def plot_sparsity_study(latency, plot_file_path, bw):
    labels = [
        "0% ",
        "25%",
        "50%",
        "75%",
        "99%",
    ]

    speedups = latency[0] / latency
    xticks = np.arange(len(latency))
    colors = ["#6399E8" if i == 0 else "#EE8183" for i in range(len(latency))]

    fig, ax = plt.subplots(figsize=(5.5, 3))
    bars = ax.bar(xticks, speedups, color=colors)

    ax.set_xticks(xticks)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Speedup")
    ax.set_title(f"Speedup for {bw}/s Design")
    ax.set_ylim(.5, 1.2)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.02, f"{yval:.2f}x", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    os.makedirs(plot_file_path, exist_ok=True)
    save_path = os.path.join(plot_file_path, f"sparsity_study_speedup_{bw}.png")
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.show()

def plot_speedup_comparison(latency1, latency2, plot_file_path):
    """
    Plots the speedup comparison between two latency vectors as a line plot.

    Args:
        latency1 (list): The first latency vector.
        latency2 (list): The second latency vector.
        labels (list): Labels for the x-axis.
        plot_file_path (str): Path to save the plot.
        title (str): Title of the plot.
        bw (str): Bandwidth information for the title.
    """
    labels = [
        "0% ",
        "25%",
        "50%",
        "75%",
        "99%",
    ]
    speedups1 = latency1[0] / latency1
    speedups2 = latency2[0] / latency2
    xticks = np.arange(len(latency1))

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(xticks, speedups1, marker='o', label='Design 1', color="#6399E8")
    ax.plot(xticks, speedups2, marker='s', label='Design 2', color="#EE8183")

    ax.set_xticks(xticks)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Speedup")
    # ax.set_title(f"{title} for {bw}/s Design")
    ax.set_ylim(.99, 1.2)
    ax.legend()

    for i, (s1, s2) in enumerate(zip(speedups1, speedups2)):
        ax.text(i, s1 + 0.02, f"{s1:.2f}x", ha='center', va='bottom', fontsize=8, color="#6399E8")
        ax.text(i, s2 + 0.02, f"{s2:.2f}x", ha='center', va='bottom', fontsize=8, color="#EE8183")

    plt.tight_layout()
    os.makedirs(plot_file_path, exist_ok=True)
    save_path = os.path.join(plot_file_path, f"speedup_comparison.png")
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.show()

def plot_sparsity_study_vanilla(speedups, plot_file_path):
    labels = [
        "Dist 1",
        "Dist 2",
        "Dist 3",
    ]
    colors = ["#778898", "#6399E8", "#EE8183", "#6CCBAB", "#B19EE1", "#F4B07F"]

    xticks = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(3, 2))
    bars = ax.bar(xticks, speedups, color=colors)

    ax.set_xticks(xticks)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("% Improvement")
    # ax.set_title(f"Sparsity Impact on Performance")
    ax.set_ylim(10, 19)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.02, f"{yval:.2f}%", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    os.makedirs(plot_file_path, exist_ok=True)
    save_path = os.path.join(plot_file_path, f"vanilla_sparsity.png")
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    save_path = os.path.join(plot_file_path, f"vanilla_sparsity.pdf")
    plt.savefig(save_path, dpi=100, bbox_inches="tight")

def plot_workload_ablation(application_name, vanilla_gates_exp, jf_gates_exp, plot_file_path):

    design_str = '(9, 2048, 16, 16, 1), \'dual_core\', 1, (16, 7, 5, 16384), 1)'
    file_dir = "./debug_full_chip/ablation_study/"
    
    vanilla_file = file_dir + f"vanilla_{vanilla_gates_exp}_2048.txt"
    jellyfish_file = file_dir + f"jellyfish_{jf_gates_exp}_2048.txt"

    jellyfish_sparsity_file = file_dir + f"jellyfish_sparsity_{jf_gates_exp}_2048.txt"
    jellyfish_sparsity_zc_mask_file = file_dir + f"jellyfish_sparsity_zc_mask_{jf_gates_exp}_2048.txt"

    files = [
        ("Vanilla", vanilla_file),
        ("Jellyfish", jellyfish_file),
        ("Jellyfish Sparsity", jellyfish_sparsity_file),
        ("Jellyfish Sparsity + ZC Mask", jellyfish_sparsity_zc_mask_file)
    ]

    runtimes = []

    for label, file_path in files:
        try:
            with open(file_path, "r") as f:
                for line in f:
                    if design_str in line:
                        match = re.search(r"'overall_runtime': ([\d.]+)", line)
                        if match:
                            runtimes.append((label, float(match.group(1))))
                            print(file_path)
                            break
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            runtimes.append((label, None))

    print("Runtimes:", runtimes)
    # Filter out None values
    runtimes = [(label, runtime) for label, runtime in runtimes if runtime is not None]
    exit()
    # Plot the bar chart
    labels, values = zip(*runtimes)
    colors = ["#6399E8", "#EE8183", "#6CCBAB", "#B19EE1"]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values, color=colors)

    plt.ylabel("Overall Runtime (ms)")
    plt.title("Runtime Comparison for Different Workloads")
    plt.ylim(0, max(values) * 1.2)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 1, f"{yval:.2f}", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(plot_file_path, f"{application_name}.png")
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    save_path = os.path.join(plot_file_path, f"{application_name}.pdf")
    plt.savefig(save_path, dpi=100, bbox_inches="tight")


