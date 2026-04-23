#!/bin/bash

# should always rerun this whenever we have new full chip data

# num_vars_array=(15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30)
# bw_array=(64 128 256 512 1024 2048 4096)

# # Loop over the provided NUM_VARS array
# for NUM_VARS in "${num_vars_array[@]}"
# do
#     # Loop over the predefined BW array
#     for BW in "${bw_array[@]}"
#     do
#         python full_chip.py $NUM_VARS $BW vanilla   --pickle > debug_full_chip/vanilla_${NUM_VARS}_${BW}.txt &
#         # python full_chip.py $NUM_VARS $BW jellyfish --pickle > debug_full_chip/jellyfish_${NUM_VARS}_${BW}.txt &
#     done
# done
# exit
# num_vars_array=(24)
# bw_array=(2048)

# # # Loop over the provided NUM_VARS array
# for NUM_VARS in "${num_vars_array[@]}"
# do
#     # Loop over the predefined BW array
#     for BW in "${bw_array[@]}"
#     do
#         python full_chip.py $NUM_VARS $BW vanilla > test_64.txt 
#     done
# done



###########################################################
###########################################################
###########################################################

# num_vars_array=(20 24)
# # special_experiment="arbitrary_prime"
# # special_experiment="onchip_sram_penalty"
# # special_experiment="onchip_sram_and_arbitrary_prime"
# bw_array=(64 128 256 512 1024 2048 4096)

# exp_dir=debug_full_chip/$special_experiment
# mkdir -p $exp_dir

# # Loop over the provided NUM_VARS array
# for NUM_VARS in "${num_vars_array[@]}"
# do
#     # Loop over the predefined BW array
#     for BW in "${bw_array[@]}"
#     do
#         python full_chip.py $NUM_VARS $BW vanilla   --special_experiment $special_experiment > $exp_dir/vanilla_${NUM_VARS}_${BW}.txt &
#         python full_chip.py $NUM_VARS $BW jellyfish --special_experiment $special_experiment > $exp_dir/jellyfish_${NUM_VARS}_${BW}.txt &
#     done
# done


# num_vars_array=(15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30)

# # Loop over the provided NUM_VARS array
# for NUM_VARS in "${num_vars_array[@]}"
# do
#     # Loop over the predefined BW array
#     for BW in "${bw_array[@]}"
#     do
#         python full_chip.py $NUM_VARS $BW vanilla   --skip_pareto &
#         python full_chip.py $NUM_VARS $BW jellyfish --skip_pareto &
#     done
# done


# num_vars_array=(24)
# special_experiment="sparsity_experiment"
# bw_array=(64 128 256 512 1024 2048 4096)

# exp_dir=debug_full_chip/$special_experiment
# mkdir -p $exp_dir

# # Loop over the provided NUM_VARS array
# for NUM_VARS in "${num_vars_array[@]}"
# do
#     # Loop over the predefined BW array
#     for BW in "${bw_array[@]}"
#     do
#         # python full_chip.py $NUM_VARS $BW vanilla   --special_experiment $special_experiment > $exp_dir/vanilla_${NUM_VARS}_${BW}.txt &
        
#         python full_chip.py $NUM_VARS $BW jellyfish --special_experiment $special_experiment --skip_percent 10 > $exp_dir/jellyfish_${NUM_VARS}_${BW}_sp10.txt &
#         python full_chip.py $NUM_VARS $BW jellyfish --special_experiment $special_experiment --skip_percent 25 > $exp_dir/jellyfish_${NUM_VARS}_${BW}_sp25.txt &
#         python full_chip.py $NUM_VARS $BW jellyfish --special_experiment $special_experiment --skip_percent 50 > $exp_dir/jellyfish_${NUM_VARS}_${BW}_sp50.txt &
#         python full_chip.py $NUM_VARS $BW jellyfish --special_experiment $special_experiment --skip_percent 75 > $exp_dir/jellyfish_${NUM_VARS}_${BW}_sp75.txt &
#         python full_chip.py $NUM_VARS $BW jellyfish --special_experiment $special_experiment --skip_percent 99 > $exp_dir/jellyfish_${NUM_VARS}_${BW}_sp99.txt &
#     done
# done


# num_vars_array=(19)
# special_experiment="zc_mask"
# bw_array=(64 128 256 512 1024 2048 4096)

# exp_dir=debug_full_chip/$special_experiment
# mkdir -p $exp_dir
# design='((9, 2048, 16, 16, 1), "dual_core", 1, (16, 7, 5, 16384), 1)'

# # Loop over the provided NUM_VARS array
# for NUM_VARS in "${num_vars_array[@]}"
# do
#     # Loop over the predefined BW array
#     for BW in "${bw_array[@]}"
#     do
#         # python full_chip.py $NUM_VARS $BW vanilla   --special_experiment $special_experiment > $exp_dir/vanilla_${NUM_VARS}_${BW}.txt &
        
#         python full_chip.py $NUM_VARS $BW jellyfish --zc_mask --special_experiment $special_experiment --design_point "$design" > $exp_dir/jellyfish_${NUM_VARS}_${BW}.txt &
#         python full_chip.py $NUM_VARS $BW jellyfish --zc_mask --special_experiment $special_experiment --design_point "$design" > $exp_dir/jellyfish_${NUM_VARS}_${BW}.txt &
#         python full_chip.py $NUM_VARS $BW jellyfish --zc_mask --special_experiment $special_experiment --design_point "$design" > $exp_dir/jellyfish_${NUM_VARS}_${BW}.txt &
#         python full_chip.py $NUM_VARS $BW jellyfish --zc_mask --special_experiment $special_experiment --design_point "$design" > $exp_dir/jellyfish_${NUM_VARS}_${BW}.txt &
#         python full_chip.py $NUM_VARS $BW jellyfish --zc_mask --special_experiment $special_experiment --design_point "$design" > $exp_dir/jellyfish_${NUM_VARS}_${BW}.txt &
#     done
# done


num_vars_array=(15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30)

design='((9, 2048, 16, 16, 1), "dual_core", 1, (16, 7, 5, 16384), 1)'
BW=2048
for NUM_VARS in "${num_vars_array[@]}"
do

    # Extract the rest as the design point
    echo "NUM_VARS: $NUM_VARS"
    echo "BW: $BW"
    echo "Design Point: $DESIGN_POINT"
    # Run the script with the extracted values
    python full_chip.py $NUM_VARS $BW "vanilla" --point_duplication --design_point "$design" > debug_full_chip/ablation_study/vanilla_${NUM_VARS}_${BW}.txt &
    python full_chip.py $NUM_VARS $BW "vanilla" --design_point "$design" > debug_full_chip/ablation_study/vanilla_point_sharing_${NUM_VARS}_${BW}.txt &
    python full_chip.py $NUM_VARS $BW "jellyfish" --point_duplication --design_point "$design" > debug_full_chip/ablation_study/jellyfish_${NUM_VARS}_${BW}.txt &
    python full_chip.py $NUM_VARS $BW "jellyfish" --point_duplication --special_experiment "sparsity_experiment" --skip_percent 99  --design_point "$design" > debug_full_chip/ablation_study/jellyfish_sparsity_${NUM_VARS}_${BW}.txt &
    python full_chip.py $NUM_VARS $BW "jellyfish" --point_duplication --special_experiment "sparsity_experiment" --skip_percent 99 --zc_mask --design_point "$design" > debug_full_chip/ablation_study/jellyfish_sparsity_zc_mask_${NUM_VARS}_${BW}.txt &
    python full_chip.py $NUM_VARS $BW "jellyfish" --special_experiment "sparsity_experiment" --skip_percent 99 --zc_mask --design_point "$design" > debug_full_chip/ablation_study/jellyfish_sparsity_zc_mask_point_sharing${NUM_VARS}_${BW}.txt &

done