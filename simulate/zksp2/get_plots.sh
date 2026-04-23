
#!/bin/bash


# Should always rerun this whenever we have new full chip data

# gate_types=("vanilla" "jellyfish")
# gate_types=("vanilla")
# # gate_types=("jellyfish")
# num_vars_array=(15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30)
# bw_vec="64,128,256,512,1024,2048,4096"

# for gate_type in "${gate_types[@]}"; do
#     for num_vars in "${num_vars_array[@]}"
#     do
#         python plotting.py get_global_pareto_points --num_vars_list $num_vars --bw_vec $bw_vec --gate_type $gate_type > global_pareto_points_data/global_pareto_points_${num_vars}_$gate_type.txt &
#     done
# done
# exit

###########################################################
###########################################################
###########################################################



# num_vars_list="20 21 22 23 24"
# num_vars_list="24"
# bw_vec="64,128,256,512,1024,2048,4096"

# for num_vars in $num_vars_list; do
#     python plotting.py pareto_data_with_global_inset --num_vars_list $num_vars --bw_vec $bw_vec --gate_type jellyfish &
# done
# exit

# num_vars_list="20"
# python plotting.py cpu_runtime_breakdown --num_vars_list $num_vars_list --num_threads 32
# python plotting.py cpu_runtime_breakdown --num_vars_list $num_vars_list --num_threads 1




# exit

# num_vars_list="15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30"
# bw_vec="64,128,256,512,1024,2048,4096"
# gate_type="jellyfish"
# comp_type="zkspeed"
# first_iteration=true
# for num_vars in $num_vars_list; do
#     if [ "$first_iteration" = true ]; then
#         python plotting.py get_iso_${comp_type}_points --num_vars_list $num_vars --bw_vec $bw_vec --gate_type $gate_type > iso_${comp_type}_$gate_type.txt
#         first_iteration=false
#     else
#         python plotting.py get_iso_${comp_type}_points --num_vars_list $num_vars --bw_vec $bw_vec --gate_type $gate_type >> iso_${comp_type}_$gate_type.txt
#     fi
# done



# num_vars_array=(17 18 19 20 21 22 23 24)

# num_vars_array=(24)
# bw_array=(2048)


# Define an array of designs
# designs=(
# "24, 1024, (10, 2048, 4, 16, 1), 'dual_core', 1, (4, 7, 6, 4096), 1"
# "24, 1024, (10, 1024, 32, 16, 1), 'single_core', 1, (8, 7, 3, 8192), 1"
# "24, 2048, (9, 4096, 32, 16, 1), 'single_core', 3, (16, 7, 5, 16384), 1"
# "24, 4096, (10, 2048, 32, 16, 1), 'dual_core', 1, (16, 7, 3, 16384), 1"
# )
# designs=(
# "24, 512,  (9, 2048, 16, 16, 1), 'single_core', 1, (4, 7, 7, 4096), 1"
# "24, 1024, (10, 2048, 32, 16, 1), 'single_core', 1, (8, 7, 6, 16384), 1"
# "24, 2048, (10, 2048, 32, 16, 1), 'dual_core', 1, (16, 7, 6, 32768), 1"
# "24, 4096, (10, 2048, 32, 16, 1), 'dual_core', 3, (32, 7, 5, 32768), 1"
# )
# for DESIGN in "${designs[@]}"
# do
#     # Extract the first number (NUM_VARS) and the second number (BW)
#     NUM_VARS=$(echo "$DESIGN" | awk -F'[(), ]+' '{print $1}')
#     BW=$(echo "$DESIGN" | awk -F'[(), ]+' '{print $2}')
#     # Extract the rest as the design point
#     DESIGN_POINT=$(echo "$DESIGN" | cut -d',' -f3-)
#     echo "NUM_VARS: $NUM_VARS"
#     echo "BW: $BW"
#     echo "Design Point: $DESIGN_POINT"
#     # Run the script with the extracted values
#     python full_chip.py $NUM_VARS $BW jellyfish --design_point "$DESIGN_POINT" --pickle
# done

# python plotting.py plot_ablation_study

# python plotting.py plot_sparsity_study
# python plotting.py plot_sparsity_study_vanilla



python plotting.py plot_workload_ablation