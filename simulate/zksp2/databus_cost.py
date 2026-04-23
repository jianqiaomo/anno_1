import math


def databus_cost(num_pes_zerocheck, num_pes_fracMLE, num_pes_sumcheck_in_permcheck, num_pes_sumcheck_in_polyopen,
                 num_pes_build_g_MLE):
    """
    Calculate the cost of the max width of communicate databus

    :param num_pes_zerocheck: num of zerocheck PEs
    :param num_pes_fracMLE: num of frac MLE PEs
    :param num_pes_sumcheck_in_permcheck: num of sumcheck PEs in perm check
    :param num_pes_sumcheck_in_polyopen: num of sumcheck PEs in poly open
    :param num_pes_build_g_MLE: num of build_g_MLE PEs
    :return: max required bit width of communicate databus
    :rtype: int
    """
    continuous_communication_bit = {
        "zero check: build MLE to sum check": [2, num_pes_zerocheck, 255],
        "perm check: ND to frac MLE, frac MLE to prod MLE + MSM, prod MLE to MSM":
            [2 + 1 + 1, num_pes_fracMLE, 255],  # N and D,
        "perm check: build MLE to sum check": [2, num_pes_sumcheck_in_permcheck, 255],
        "poly open: input to sum check": [(6 + 6) * 2, num_pes_sumcheck_in_polyopen, 255],
        "poly open: build g' MLE to MSM": [1, num_pes_build_g_MLE, 255],
    }
    continuous_communication_bit_mul = {}
    for key, value in continuous_communication_bit.items():
        # multiply the list of each key
        continuous_communication_bit_mul[key] = math.prod(value)
    max_bit_width = max(continuous_communication_bit_mul.values())

    dbg_msg = f""
    for key, value in continuous_communication_bit.items():
        dbg_msg += f"{key}: "
        dbg_msg += " * ".join([str(i) for i in value])
        dbg_msg += f" = {continuous_communication_bit_mul[key]}\n"
    dbg_msg += f"max_bit_width = {max_bit_width} bits\n"
    return max_bit_width, dbg_msg


def databus_cost_v1(num_sumcheck_pes, num_lane_fan_in, num_lanes, num_pes_fracMLE, num_MSM_core, num_MSM_pes):
    """
    Calculate the cost of the max width of communicate databus

    :param num_sumcheck_pes: num of sumcheck PEs
    :param num_lane_fan_in: >= num eval engines
    :return: max required bit width of communicate databus; PADD crossbar area in mm^2 (14 nm)
    :rtype: int
    """
    same_time_bus_communication_bit = [
        # group 1
        {  # they are in the same time
            "perm check: frac MLE to prod MLE + MSM": [1 + 1, num_pes_fracMLE, 255],
            "perm check: prod MLE to MSM": [1 if num_MSM_core > 1 else 0, num_pes_fracMLE, 255],
        },
        # group 2
        {  # they are in the same time
            "sumcheck: build MLE to sumcheck buffer": [num_lane_fan_in, num_sumcheck_pes, 255],
            "sumcheck: RR selector to lanes": [num_lane_fan_in, num_sumcheck_pes, num_lanes, 255],
        },
    ]

    same_time_bus_communication_bit_sum = [sum(math.prod(v) for v in group.values()) for group in same_time_bus_communication_bit]
    max_bit_width = max(same_time_bus_communication_bit_sum)

    dbg_msg = f""
    for idx, group in enumerate(same_time_bus_communication_bit):
        dbg_msg += f"Group {idx + 1}:\n"
        for key, value in group.items():
            dbg_msg += f"{key}: "
            dbg_msg += " * ".join([str(i) for i in value])
            dbg_msg += f" = {group[key]}\n"
    dbg_msg += f"max_bit_width = {max_bit_width} bits\n"

    # PADD crossbar: (num_MSM_pes * num_MSM_pes * 381b) * 3(for XYZ)
    F1_crossbar_mm2_14nm_32x32x142Byte = 3.7053125
    F1_crossbar_TDP_W_14nm_32x32x142Byte = 7.26640625
    PADD_crossbar_mm2_14nm_142Byte = F1_crossbar_mm2_14nm_32x32x142Byte * (num_MSM_pes / 32) * (num_MSM_pes / 32) * num_MSM_core
    PADD_crossbar_TDP_W_14nm_142Byte = F1_crossbar_TDP_W_14nm_32x32x142Byte * (num_MSM_pes / 32) * (num_MSM_pes / 32) * num_MSM_core

    return max_bit_width, (PADD_crossbar_mm2_14nm_142Byte, PADD_crossbar_TDP_W_14nm_142Byte), dbg_msg


if __name__ == '__main__':
    num_pes_zerocheck = 1
    num_pes_fracMLE = 1
    num_pes_sumcheck_in_permcheck = 1
    num_pes_sumcheck_in_polyopen = 1
    num_pes_build_g_MLE = 1

    max_bit_width, (area, tdp),  msg = databus_cost_v1(num_pes_zerocheck, num_pes_fracMLE, num_pes_sumcheck_in_permcheck,
                                 num_pes_sumcheck_in_polyopen, num_pes_build_g_MLE, 1)
    print(msg)
