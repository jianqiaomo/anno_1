import math

# this is the latency for 1 point-merging. each point consists of num_vars field
# elements. point merging involves element-wise multiply of field elements in both
# points, followed by a bitshift and subtraction operation. Then, the resulting 
# vector is reduced by multiplying all elements together, hence a multiply tree with
# log2(num_vars) stages. Assume each point-merge has enough multipliers to do all element-
# wise mults in parallel

# there are 6 point merges in total. we can do all point merges in series, 2 at time,
# 3 at a time, or all 6 at a time
def point_merge(num_vars, modmul_latency, num_points_in_parallel):
    assert num_points_in_parallel in (6, 3, 2, 1)
    intial_latency = modmul_latency + 2
    total_latency = math.ceil(math.log2(num_vars))*modmul_latency + intial_latency
    total_latency *= 6/num_points_in_parallel

    # assume all 6 g_prime computation is done
    g_prime_pipeline_latency = modmul_latency

    # return total_latency
    return total_latency + g_prime_pipeline_latency
    
def get_point_merge_data(num_vars_range, modmul_latency, modmul_area):
    point_merge_cycles = dict()
    point_merge_area_stats = dict()
    for num_vars in num_vars_range:
        modmuls_in_point_merging = num_vars
        num_points_in_parallel = 1
        point_merge_cycles[num_vars] = point_merge(num_vars, modmul_latency, num_points_in_parallel)
        point_merge_area_stats[num_vars] = \
        {
            "total_area"  : num_points_in_parallel*modmuls_in_point_merging*modmul_area,
            "num_modmuls" : num_points_in_parallel*modmuls_in_point_merging
        }

    point_merge_stats = point_merge_cycles, point_merge_area_stats
    return point_merge_stats