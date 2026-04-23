import math
from copy import deepcopy
import numpy as np
import pickle
import os


class padd_unit:
    def __init__(self, padd_stages):
        self.padd_stages = padd_stages
        self.padd_pipeline = [None] * padd_stages

    def output(self):
        if len(self.padd_pipeline) > 0:
            return self.padd_pipeline.pop()
        else:
            return None

    def send_operation(self, operation_id):
        if len(self.padd_pipeline) < self.padd_stages:
            self.padd_pipeline.insert(0, operation_id)
            return True
        else:
            return False


class PaddBucketsWithoutDouble:
    """
    Evaluate PADD buckets without Pdouble operations. Simply Padd one element to the bucket per time.
    """

    def __init__(self, num_bucket, padd_stages, bucket_size_bit=3 * 381, num_padd_unit=1):
        """
        set up buckets of PADD. Assume buckets (ts, rs) are already loaded. 

        :param num_bucket: number of buckets
        :param padd_stages: number of PADD stages (cycles)
        :param bucket_size_bit: size of each bucket (bit)
        :param num_padd_unit: number of PADD units
        """
        self.num_bucket = num_bucket
        self.padd_stages = padd_stages
        self.num_padd_unit = num_padd_unit
        if self.num_bucket < 1:
            raise ValueError(f"num_bucket should be at least 1, but got {self.num_bucket}")
        if self.padd_stages < 1:
            raise ValueError(f"padd_stages should be at least 1, but got {self.padd_stages}")
        if self.num_padd_unit < 1:
            raise ValueError(f"num_padd_unit should be at least 1, but got {self.num_padd_unit}")
        self.bucket_size_bit = bucket_size_bit

        self.total_cycles = -1
        self.cost_report = {}

    def cost(self):
        """
        simulate cost

        - total_cycles: total latency to the last output generated
        - req_mem_reg_num: num of required register (or bucket)
        - req_bandwidth_bperc: (bit per cycle) HBM.
        :return: self.cost_report
        :rtype: dict
        """
        if len(self.cost_report) != 0:
            return self.cost_report

        adders = [padd_unit(self.padd_stages) for _ in range(self.num_padd_unit)]

        self.all_result_cycle = [[]]

        add_op_id = []  # (round_id, bucket_id)
        i, j = 0, 0
        for i in range(self.num_bucket - 1):
            for j in range(self.num_bucket, i + 1, -1):
                add_op_id.append((i + 1, j))
            # add op: near two buckets sum
            if i + 1 > 2:
                add_op_id.append((i + 1, j - 1 + 0.1))
        add_op_id.append((i + 2, j + 0.1))
        last_add_op_id = add_op_id[-1]

        round_id = 1
        result_cycle = []
        while last_add_op_id not in self.all_result_cycle[-1]:
            for adder_id, adder in enumerate(adders):
                # pop result from the adder
                if adder_id == 0:
                    self.all_result_cycle.append(deepcopy(result_cycle))
                    result_cycle = [adder.output()]
                else:
                    result_cycle.append(adder.output())

                if len(add_op_id) > 0:
                    add_op = add_op_id[0]
                else:
                    add_op = None

                if add_op is not None and add_op[0] != round_id:
                    # new round, check if the first result of last round is ready
                    if result_cycle[-1] is not None and (
                            result_cycle[-1][0] == round_id and result_cycle[-1][1] <= add_op[1]):
                        # safe to send new operation
                        if adder.send_operation(add_op):
                            add_op_id.pop(0)
                        else:
                            raise ValueError(f"adder {adder_id} unable to send new operation")
                        round_id += 1
                    else:
                        # unable to send new operation
                        adder.send_operation(None)
                else:
                    # same round
                    if adder.send_operation(add_op):
                        if len(add_op_id) > 0:
                            add_op_id.pop(0)
                    else:
                        raise ValueError(f"adder {adder_id} unable to send new operation")
        for _ in range(3):
            self.all_result_cycle.pop(0)

        self.cost_report = {
            'total_cycles': len(self.all_result_cycle),
            'req_mem_reg_num': self.num_bucket * 2,
            'req_mem_bit': (self.num_bucket * 2) * self.bucket_size_bit,
            'req_PADD': self.num_padd_unit,
        }

        return self.cost_report

    def dbg_prt(self):
        """
        print the result of the cost function

        :return: debug message per cycle
        :rtype: str
        """
        if len(self.cost_report) == 0:
            self.cost()
        # print the list in self.all_result_cycle by line
        r = ""
        for i, results in enumerate(self.all_result_cycle):
            presult = ""
            for j, result in enumerate(results):
                if result is None:
                    presult += "None, "
                else:
                    presult += f"{result[0] + 1}·b{result[1]}, "
            r += f"{i + 1:<4d} {presult}\n"
        return r


def bucket_binary_tree_add(total_nodes):
    """Generates a binary tree (add two) walk based on the given total node count.

    Args:
        total_nodes (int): The number of nodes (e.g., 6 or 7).

    Returns:
        List of tuples with layer and merge operation (layer, "nodeX to nodeY").
    """

    def merge_nodes(node_in_temp_list, node2):
        """Merge the node (from the temp list) with the current node."""
        node_in_temp_list = node_in_temp_list[1]  # Get the node from the tuple
        node2 = node2[1]  # Get the node from the tuple
        if "to" in node_in_temp_list:
            node_in_temp_list_higher = int(node_in_temp_list.split("to")[0])
            node2_lower = int(node2.split("to")[-1])
            return f"{node_in_temp_list_higher} to {node2_lower}"
        else:
            if "to" in node2:
                node2_lower = int(node2.split("to")[-1])
                return f"{node_in_temp_list} to {node2_lower}"
            else:
                return f"{node_in_temp_list} to {node2}"

    # Initialize variables
    results = []
    current_layer = [(-1, f"{i + 1}") for i in range(total_nodes)]  # Nodes from 1 to total_nodes
    temp_list_odd = []

    # Calculate tree depth (bit length of total_nodes - 1)
    depth = (total_nodes - 1).bit_length()

    # Process each layer up to the calculated depth
    for layer in range(depth):
        next_layer = []

        # Pair up nodes in the current layer
        for i in range(0, len(current_layer) - 1, 2):
            next_two_add = merge_nodes(current_layer[i + 1], current_layer[i])
            next_layer.append((layer, next_two_add))
            if current_layer[i][0] >= 0:
                next_two_add_high, next_two_add_low = map(int, next_two_add.split("to"))
                results[current_layer[i + 1][0]][i + 1] = (
                    results[layer - 1][i + 1][0], results[layer - 1][i + 1][1] + f" at {next_two_add_high}")
                results[current_layer[i][0]][i] = (
                    results[layer - 1][i][0], results[layer - 1][i][1] + f" at {next_two_add_low}")

        if len(current_layer) % 2 == 1:  # Handle the odd element case
            if temp_list_odd:
                # Merge with the temp list node
                temp_list_odd_item = temp_list_odd.pop()
                next_two_add = merge_nodes(temp_list_odd_item, current_layer[-1])
                next_layer.append((layer, next_two_add))
                next_two_add_high, next_two_add_low = map(int, next_two_add.split("to"))
                if temp_list_odd_item[0] >= 0:
                    results[temp_list_odd_item[0]][-1] = (results[temp_list_odd_item[0]][-1][0],
                                                          results[temp_list_odd_item[0]][-1][
                                                              1] + f" at {next_two_add_high}")
                if current_layer[-1][0] >= 0:
                    results[current_layer[-1][0]][-1] = (results[current_layer[-1][0]][-1][0],
                                                         results[current_layer[-1][0]][-1][
                                                             1] + f" at {next_two_add_low}")
            else:
                # Move the odd element to the temp list
                temp_list_odd.append(current_layer[-1])

        # Move to the next layer
        current_layer = next_layer[:]
        results.append(next_layer[:])

    results[-1][0] = (results[-1][0][0], results[-1][0][1] + " at 1")  # Add the final result
    results = [i[::-1] for i in results]
    return results


def merge_num_list(a, b):
    """Merge two numbers or lists into a single list.

    Args:
        a (int, float, list): The first number or list.
        b (int, float, list): The second number or list.

    Returns:
        List: The merged list.
    """
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return sorted([a, b])
    # If one input is already a list, merge the other item into it
    elif isinstance(a, list) and not isinstance(b, list):
        return sorted(a + [b])
    elif isinstance(b, list) and not isinstance(a, list):
        return sorted([a] + b)
    # If both are lists, concatenate them
    elif isinstance(a, list) and isinstance(b, list):
        return sorted(a + b)
    else:
        raise TypeError("Both inputs must be numbers or lists")


def double_round_needed(bucket_id):
    """
    check how many rounds needed to double the bucket.
    E.g., 7 (0b111) needs 2 rounds of double; 8 (0b1000) needs 3 rounds of double.

    :param bucket_id: bucket id
    :return:
    :rtype: int
    """
    return bucket_id.bit_length() - 1


def add_equal_needed(bucket_id, round_id):
    """
    if current round_id is 1, and the rest are not all 0s, then need to add equal.
    E.g., 7 (0b111) at double round 2, need to add equal; 8 (0b1000) at double round 3, no need to add equal.
    9 (0b1001) at double round 2, no need to add equal, but at double round 3, need to add equal.

    :param bucket_id: bucket id
    :param round_id: current round id
    :return:
    :rtype: bool
    """
    # check if the bit 0 to bit (round_id-1) are not all 0s
    if round_id <= 1:
        return False
    rest_check_bool = (bucket_id & ((1 << (round_id - 1)) - 1)) != 0
    current_round_check_bool = bucket_id & (1 << (round_id - 1)) != 0
    return rest_check_bool & current_round_check_bool


class PaddBucketsDouble(PaddBucketsWithoutDouble):
    def cost(self):
        if len(self.cost_report) != 0:
            return self.cost_report

        if self.num_padd_unit != 1:
            raise NotImplementedError(f"num_padd_unit={self.num_padd_unit} is not supported now.")
        # adders = [padd_unit(self.padd_stages) for _ in range(self.num_padd_unit)]
        adder = padd_unit(self.padd_stages)

        self.all_result_cycle = []

        # prepare the instruction list
        inst_id = []  # (round_id, bucket_id, instruction)
        round_idx, bucket_idx_1 = 0, 0
        for round_idx in range(1, double_round_needed(self.num_bucket) + 2):
            for bucket_idx_1 in range(self.num_bucket, 2 - 1, -1):
                if add_equal_needed(bucket_idx_1, round_idx):
                    inst_id.append((round_idx, bucket_idx_1, 'add_equal'))
                # if double needed
                if bucket_idx_1.bit_length() > round_idx:
                    inst_type = 'double_eq' if int(bin(bucket_idx_1)[2:][-round_idx:]) == 0 \
                                               and int(bin(bucket_idx_1)[2:][-round_idx - 1]) == 1 else 'double'
                    inst_id.append((round_idx, bucket_idx_1, inst_type))

        tree_add_ops = bucket_binary_tree_add(self.num_bucket)
        if self.num_bucket & 1 != 0:  # odd
            for tree_layer_idx, tree_add_op in enumerate(tree_add_ops):
                inst_id += [(round_idx + tree_layer_idx, int(two_add[1].split("to")[0]), two_add[1]) for two_add in
                            tree_add_op]
        else:
            for tree_layer_idx, tree_add_op in enumerate(tree_add_ops):
                inst_id += [(round_idx + tree_layer_idx + 1, int(two_add[1].split("to")[0]), two_add[1]) for two_add in
                            tree_add_op]
        last_add_op_id = inst_id[-1]

        # simulate
        buckets = [{"doubler": 1, "sums": i % 2} for i in range(1, self.num_bucket + 1)]
        self.buckets_all = []
        output = None
        while last_add_op_id != output:
            # Read the padd unit, process output
            output = adder.output()
            if output:
                bucket_idx_1 = output[1] - 1
                if output[2] == "double":
                    if 0 <= bucket_idx_1 < self.num_bucket:
                        buckets[bucket_idx_1]["doubler"] = 2 ** output[0]
                elif output[2] == "double_eq":
                    if 0 <= bucket_idx_1 < self.num_bucket:
                        buckets[bucket_idx_1]["doubler"] = 2 ** output[0]
                        buckets[bucket_idx_1]["sums"] = 2 ** output[0]
                elif output[2] == "add_equal":
                    if 0 <= bucket_idx_1 < self.num_bucket:
                        buckets[bucket_idx_1]["sums"] += buckets[bucket_idx_1]["doubler"]
                elif "to" in output[2]:
                    from_high, from_low, dest_loc = map(int, output[2].replace("to", ",").replace("at", ",").split(','))
                    if 0 < from_high <= self.num_bucket and 0 < from_low <= self.num_bucket:
                        buckets[dest_loc - 1]["sums"] = [bkt for bkt in range(from_low, from_high + 1)]
                else:
                    raise ValueError(f"Unknown output type {output}")
            self.all_result_cycle.append(output)

            self.buckets_all.append(deepcopy(buckets))

            # Attempt to launch a new instruction
            if inst_id:
                inst = inst_id[0]  # Fetch the next instruction
                inst_level, bucket_id, inst_type = inst
                bucket_idx = bucket_id - 1

                if "double" in inst_type:
                    if buckets[bucket_idx]["doubler"] == 2 ** (inst_level - 1):
                        # Launch the instruction to the padd unit
                        if adder.send_operation(inst):
                            inst_id.pop(0)
                        else:
                            raise ValueError(f"adder unable to send new operation {inst}")
                    else:
                        adder.send_operation(None)
                elif inst_type == "add_equal":
                    if buckets[bucket_idx]["doubler"] == 2 ** (inst_level - 1):
                        # Launch the instruction to the padd unit
                        if adder.send_operation(inst):
                            inst_id.pop(0)
                        else:
                            raise ValueError(f"adder unable to send new operation {inst}")
                    else:
                        adder.send_operation(None)
                elif "to" in inst_type:
                    b_start, b_end, _ = map(int, inst_type.replace("to", ",").replace("at", ",").split(','))
                    b_start_idx, b_end_idx = b_start - 1, b_end - 1
                    if merge_num_list(buckets[b_start_idx]["sums"], buckets[b_end_idx]["sums"]) == [
                        bkt for bkt in range(b_end, b_start + 1)]:
                        # Launch the instruction to the padd unit
                        if adder.send_operation(inst):
                            inst_id.pop(0)
                        else:
                            raise ValueError(f"adder unable to send new operation {inst}")
                    else:
                        adder.send_operation(None)
                else:
                    raise ValueError(f"Unknown instruction type {inst_type}")

        self.cost_report = {
            'total_cycles': len(self.all_result_cycle),
            'req_mem_reg_num': self.num_bucket * 3,
            'req_mem_bit': (self.num_bucket * 3) * self.bucket_size_bit,
            'req_PADD': self.num_padd_unit,
        }

        return self.cost_report

    def dbg_prt(self):
        """
        print the result of the cost function

        :return: debug message per cycle
        :rtype: str
        """
        if len(self.cost_report) == 0:
            self.cost()

        bkt_rpt_init = ""
        for i, bucket_i in enumerate(self.buckets_all[0]):
            bkt_rpt_init += f"b{i + 1:<4d} doubler {bucket_i['doubler']}·b{i + 1:<4d}, sums {bucket_i['sums']}·b{i + 1:<4d}\n"

        r = f"buckets init:\n{bkt_rpt_init}\n"
        r += f"total cycles: {self.cost_report['total_cycles']}\n"

        for i, results in enumerate(self.all_result_cycle):
            presult = ""
            if results is None:
                presult += "None, "
            elif results[2] == "double":
                presult += f"b{results[1]}, {results[2]} gets {2 ** results[0]}·b{results[1]} in doubler "
            elif results[2] == "double_eq":
                presult += f"b{results[1]}, {results[2]} gets {2 ** results[0]}·b{results[1]} in doubler and sums "
            elif results[2] == "add_equal":
                presult += f"b{results[1]}, {self.buckets_all[i - 1][results[1] - 1]['sums']}·b{results[1]} " \
                           f"{results[2]} {self.buckets_all[i - 1][results[1] - 1]['doubler']}·b{results[1]} to get " \
                           f"{self.buckets_all[i][results[1] - 1]['sums']}·b{results[1]} sums "
            elif "to" in results[2]:
                b_start, b_end, dest_b = map(int, results[2].replace("to", ",").replace("at", ",").split(','))
                presult += f"b{dest_b}, sum of {b_start}·b{b_start} to {b_end}b·{b_end} "
            r += f"{i + 1:<4d} {presult}\n"
        return r


def dict_add(d1, d2):
    """
    Add two dictionaries

    :param d1: dictionary 1
    :param d2: dictionary 2
    :return: sum of two dictionaries
    """
    d = {}
    all_keys = set(d1.keys()).union(set(d2.keys()))
    for k in all_keys:
        d[k] = d1.get(k, 0) + d2.get(k, 0)
    return d


class PaddBucketsGroup:
    """
    Evaluate PADD buckets with group parallelism. (PriorMSM)
    """

    def __init__(self, num_bucket, group_size, padd_stages, bucket_size_bit=3 * 381, num_padd_unit=1):
        """
        set up buckets of PADD. Assume buckets (rs, T, R) are already loaded.

        :param num_bucket: number of buckets
        :param group_size: number of buckets in a group
        :param padd_stages: number of PADD stages (cycles)
        :param bucket_size_bit: size of each bucket (bit)
        :param num_padd_unit: number of PADD units
        """
        self.num_bucket = num_bucket
        self.group_size = group_size
        self.padd_stages = padd_stages
        self.num_padd_unit = num_padd_unit
        if self.num_bucket < 1:
            raise ValueError(f"num_bucket should be at least 1, but got {self.num_bucket}")
        if not 1 < self.group_size < self.num_bucket:
            raise ValueError(f"group_size should be 1 < group_size < {self.num_bucket}, but got {self.group_size}")
        # group_size should be 2's power
        if not (self.group_size & (self.group_size - 1) == 0):
            raise ValueError(f"group_size should be 2's power, but got {self.group_size}")
        if self.padd_stages < 1:
            raise ValueError(f"padd_stages should be at least 1, but got {self.padd_stages}")
        if self.num_padd_unit < 1:
            raise ValueError(f"num_padd_unit should be at least 1, but got {self.num_padd_unit}")
        self.bucket_size_bit = bucket_size_bit

        self.total_cycles = -1
        self.cost_report = {}

    def cost(self):
        if len(self.cost_report) != 0:
            return self.cost_report

        if self.num_padd_unit != 1:
            raise NotImplementedError(f"num_padd_unit={self.num_padd_unit} is not supported now.")
        # adders = [padd_unit(self.padd_stages) for _ in range(self.num_padd_unit)]
        adder = padd_unit(self.padd_stages)

        self.all_result_cycle = []

        # prepare the instruction list
        # step 1
        inst_all = []  # (round_id, dest (group id), instruction, src (bucket id))
        bucket_left = self.num_bucket % self.group_size
        self.num_group = self.num_bucket // self.group_size + (1 if bucket_left > 0 else 0)
        # T
        group_size_each = [self.group_size] * (self.num_group - 1) + ([bucket_left] if bucket_left > 0 else [
            self.group_size])
        for rd in range(self.group_size):
            inst_rd = []
            for g_idx in range(len(group_size_each)):
                if group_size_each[g_idx] > 1:
                    need_add = group_size_each[g_idx] - 1  # ops of "add to T" needed
                    inst_rd.append((need_add, g_idx, 'AddEq_T', f"b_{need_add + g_idx * self.group_size}"))
                    group_size_each[g_idx] -= 1
                else:
                    continue
            inst_all.extend(inst_rd[::-1])
        # R
        inst_Ts = deepcopy(inst_all)
        T_iter_idx = []
        group_size_each = np.array([self.group_size] * (self.num_group - 1) + ([bucket_left] if bucket_left > 0 else [
            self.group_size]))
        for i in range(self.group_size - 1):
            T_iter_idx.append(np.sum(group_size_each > 1))
            group_size_each -= 1
        assert sum(T_iter_idx) == len(inst_Ts)
        T_iter_idx = [0] + [sum(T_iter_idx[:i + 1]) for i in range(len(T_iter_idx))]
        for idx in reversed(range(len(T_iter_idx) - 1)):
            Ts_chunk = inst_Ts[T_iter_idx[idx]:T_iter_idx[idx + 1]]
            inst_Rs = []
            for inst_T in Ts_chunk:
                inst_Rs.append((inst_T[0], inst_T[1], 'AddEq_R', f"T_{inst_T[1]}"))
            inst_all[T_iter_idx[idx + 1]:T_iter_idx[idx + 1]] = inst_Rs

        # step 2
        inst_all.append((0, 1, 'Copy_Y', f'T_{self.num_group - 1}'))
        for rd in reversed(range(self.num_group - 2)):
            inst_all.append((rd, 1, 'AddEq_Y1T', f'T_{rd + 1}'))
            inst_all.append((rd, 1, 'AddEq_Y1R', f'Y1T'))
        # if self.group_size & 1:  # odd
        #     inst_all[-1] = (inst_all[-1][0], 1, 'AddEq_Y1RY2', f'Y1R')
        double_idx = len(inst_all)
        double_need = double_round_needed(self.group_size)
        inst_all.append((double_need, 2, 'Double_Y2', f'Y1R'))
        for rd in range(double_need - 1, 0, -1):
            inst_all.append((rd, 2, 'Double_Y2', f'Y2'))

        # step 3
        binary_tree_adds_list = bucket_binary_tree_add(self.num_group)
        for layer_idx, layer in enumerate(binary_tree_adds_list):
            for idx, add in enumerate(layer):
                R_src1, R_src2, R_dest = map(int, add[1].replace('to', ',').replace('at', ',').split(','))
                binary_tree_adds_list[layer_idx][idx] = (
                layer_idx, R_dest - 1, f'TreeAdd_R{R_dest - 1}', f'R{R_src1 - 1}_R{R_src2 - 1}')
        # separate the binary_tree_adds_list[0] into (len(inst_all) - double_idx + 2) parts
        chunk_size = math.ceil(len(binary_tree_adds_list[0]) / (len(inst_all) - double_idx + 1))
        # distribute the binary_tree_adds_list[0] to inst_all idx
        for idx in reversed(range(double_idx, len(inst_all) + 1)):
            inserted = binary_tree_adds_list[0][chunk_size * (idx - double_idx):chunk_size * (idx - double_idx + 1)]
            if len(inserted) != 0:
                inst_all[idx:idx] = inserted
        # append the rest of binary_tree_adds_list one by one
        for layer in binary_tree_adds_list[1:]:
            inst_all.extend(layer)
        R_result_idx = 0
        for i in reversed(inst_all):
            if 'TreeAdd' in i[2]:
                R_result_idx = i[1]
                break
        inst_all.append((0, 0, 'AddEq_Target', f'R{R_result_idx}'))

        # simulate
        self.buckets_all = {
            "T": [],
            "R": [],
            "Y1T": {},
            "Y1R": {},
            "Y2": {},
            "Target": {}
        }
        all_buckets = [f"b_{i + 1}" for i in range(self.num_bucket)]
        # each self.group_size buckets are grouped into a group
        for bk_group_idx in range(self.num_group):
            bk_group = all_buckets[bk_group_idx * self.group_size:bk_group_idx * self.group_size + self.group_size]
            bk_group_dict = {bk: 0 for bk in bk_group}
            bk_group_dict[bk_group[-1]] = 1
            self.buckets_all["T"].append(deepcopy(bk_group_dict))
            self.buckets_all["R"].append(deepcopy(bk_group_dict))
        group_size_each = [self.group_size] * (self.num_group - 1) + ([bucket_left] if bucket_left > 0 else [
            self.group_size])

        while len(self.buckets_all["Target"]) == 0:
            # Read the padd unit, process output
            output = adder.output()
            if output:
                if output[2] == 'AddEq_T':
                    self.buckets_all["T"][output[1]][output[3]] += 1
                elif output[2] == "AddEq_R":
                    self.buckets_all["R"][output[1]] = deepcopy(dict_add(self.buckets_all["R"][output[1]],
                                                                         self.buckets_all["T"][
                                                                             int(output[3].split('_')[1])]))
                elif output[2] == "AddEq_Y1T":
                    self.buckets_all["Y1T"] = deepcopy(dict_add(self.buckets_all["Y1T"],
                                                                self.buckets_all["T"][int(output[3].split('_')[1])]))
                elif output[2] == "AddEq_Y1R":
                    self.buckets_all["Y1R"] = deepcopy(dict_add(self.buckets_all["Y1R"], self.buckets_all["Y1T"]))
                elif output[2] == "Double_Y2":
                    self.buckets_all["Y2"] = deepcopy(
                        dict_add(self.buckets_all[output[3]], self.buckets_all[output[3]]))
                elif output[2] == "AddEq_Target":
                    self.buckets_all["Target"] = deepcopy(dict_add(self.buckets_all["Y2"],
                                                                   self.buckets_all[output[3][0]][int(output[3][1:])]))
                elif "TreeAdd" in output[2]:
                    src1, src2 = map(int, output[3].replace('R', '').split('_'))
                    self.buckets_all["R"][output[1]] = deepcopy(dict_add(self.buckets_all["R"][src1],
                                                                         self.buckets_all["R"][src2]))
                else:
                    raise ValueError(f"Unknown output type {output}")
            self.all_result_cycle.append(output)

            # Attempt to launch a new instruction
            if inst_all:
                inst = inst_all[0]  # Fetch the next instruction
                inst_level, dest_bucket_idx, inst_type, src_bucket = inst

                if inst_type == 'AddEq_T':
                    # count the number of 0 in bucket T[dest_bucket_idx] ?= inst_level
                    if list(self.buckets_all["T"][dest_bucket_idx].values()).count(0) == inst_level:
                        # Launch the instruction to the padd unit
                        if adder.send_operation(inst):
                            inst_all.pop(0)
                        else:
                            raise ValueError(f"adder unable to send new operation {inst}")
                    else:
                        adder.send_operation(None)
                elif inst_type == "AddEq_R":
                    T_ready = list(self.buckets_all["T"][int(src_bucket.split('_')[1])].values()).count(
                        0) == inst_level - 1
                    R_ready = list(self.buckets_all["R"][dest_bucket_idx].values()).count(0) == inst_level
                    if T_ready and R_ready:
                        # Launch the instruction to the padd unit
                        if adder.send_operation(inst):
                            inst_all.pop(0)
                        else:
                            raise ValueError(f"adder unable to send new operation {inst}")
                    else:
                        adder.send_operation(None)
                elif inst_type == "Copy_Y":
                    # copy src_bucket to Y1T, Y1R
                    self.buckets_all["Y1T"] = deepcopy(self.buckets_all[src_bucket[0]][int(src_bucket.split('_')[1])])
                    self.buckets_all["Y1R"] = deepcopy(self.buckets_all["Y1T"])
                    inst_all.pop(0)
                elif inst_type == 'AddEq_Y1T':
                    T_ready = list(self.buckets_all["T"][int(src_bucket.split('_')[1])].values()).count(0) == 0
                    Y1T_ready = (len(self.buckets_all["Y1T"]) + (inst_level + 2) * self.group_size == self.num_bucket)
                    if T_ready and Y1T_ready:
                        # Launch the instruction to the padd unit
                        if adder.send_operation(inst):
                            inst_all.pop(0)
                        else:
                            raise ValueError(f"adder unable to send new operation {inst}")
                    else:
                        adder.send_operation(None)
                elif inst_type == 'AddEq_Y1R':
                    Y1T_ready = (len(self.buckets_all["Y1T"]) + (inst_level + 1) * self.group_size == self.num_bucket)
                    Y1R_ready = (len(self.buckets_all["Y1R"]) + (inst_level + 2) * self.group_size == self.num_bucket)
                    if Y1R_ready and Y1T_ready:
                        # Launch the instruction to the padd unit
                        if adder.send_operation(inst):
                            inst_all.pop(0)
                        else:
                            raise ValueError(f"adder unable to send new operation {inst}")
                    else:
                        adder.send_operation(None)
                elif inst_type == 'Double_Y2':
                    if f"b_{self.group_size + 1}" in self.buckets_all[src_bucket]:
                        src_ready = self.buckets_all[src_bucket][f"b_{self.group_size + 1}"] == \
                                    2 ** (double_need - inst_level)
                    else:
                        src_ready = False
                    if src_ready:
                        # Launch the instruction to the padd unit
                        if adder.send_operation(inst):
                            inst_all.pop(0)
                        else:
                            raise ValueError(f"adder unable to send new operation {inst}")
                    else:
                        adder.send_operation(None)
                elif inst_type == 'AddEq_Target':
                    Y2_ready = self.buckets_all["Y2"][f"b_{self.group_size + 1}"] == 2 ** double_need
                    src_ready = len(self.buckets_all["R"][int(src_bucket[1:])]) == self.num_bucket
                    if src_ready and Y2_ready:
                        # Launch the instruction to the padd unit
                        if adder.send_operation(inst):
                            inst_all.pop(0)
                        else:
                            raise ValueError(f"adder unable to send new operation {inst}")
                    else:
                        adder.send_operation(None)
                elif "TreeAdd" in inst_type:
                    R_src2_idx, R_src1_idx = map(int, src_bucket.replace('R', '').split('_'))
                    src_ready = len(self.buckets_all["R"][R_src1_idx]) + len(self.buckets_all["R"][R_src2_idx]) \
                                == sum(group_size_each[R_src1_idx:R_src2_idx + 1])
                    if src_ready:
                        # Launch the instruction to the padd unit
                        if adder.send_operation(inst):
                            inst_all.pop(0)
                        else:
                            raise ValueError(f"adder unable to send new operation {inst}")
                    else:
                        adder.send_operation(None)
                else:
                    raise ValueError(f"Unknown instruction type {inst_type}")

        # check if all element of "Target" has value == int(key.split('_')[1])
        if not all([v == int(k.split('_')[1]) for k, v in self.buckets_all["Target"].items()]):
            raise ValueError("Target not correct")

        self.cost_report = {
            'total_cycles': len(self.all_result_cycle),
            'req_mem_reg_num': self.num_bucket + self.num_group * 2 + 2 + 1,
            'req_mem_bit': (self.num_bucket + self.num_group * 2 + 2 + 1) * self.bucket_size_bit,
            'req_PADD': self.num_padd_unit,
        }

        return self.cost_report

    def dbg_prt(self):
        if len(self.cost_report) == 0:
            self.cost()
        r = "total cycles: " + str(self.cost_report['total_cycles']) + "\n"
        for idx, i in enumerate(self.all_result_cycle):
            r += f"cycle {idx}: {i}\n"
        return r


def sweep_archs(window_sizes, max_group_exponent, padd_stages, file_path, dump_data):
    num_padd_unit = 1
    optimal_latency = dict()

    group_labels = ["Double"]
    for group_size in range(2, max_group_exponent):
        group_labels.append(f"Group Size {2 ** group_size}")

    for ws in window_sizes:
        results = []
        br_latencies = []
        num_bucket = (1 << ws) - 1
        if ws <= 0:
            padd_buckets2 = PaddBucketsDouble(num_bucket=num_bucket, padd_stages=padd_stages,
                                              num_padd_unit=num_padd_unit)
            result2 = padd_buckets2.cost()
            print("\nPadd Buckets Using Double")
            print(result2)
            br_latencies.append(result2['total_cycles'])
            results.append(result2)
        else:
            br_latencies.append(np.inf)
            results.append(None)

        for group_size in range(2, max_group_exponent):
            # for ws = w (e.g. 7), we have 2^w - 1 (e.g. 127) buckets. then a group size of 2^w (e.g.) 128 doesnt work
            if (1 << group_size) >= (1 << ws):
                continue
            padd_buckets3 = PaddBucketsGroup(num_bucket=num_bucket, group_size=2 ** group_size, padd_stages=padd_stages,
                                             num_padd_unit=num_padd_unit)
            result3 = padd_buckets3.cost()
            print(f"\nPadd Buckets Using Group Size {2 ** group_size}")
            print(result3)
            br_latencies.append(result3['total_cycles'])
            results.append(result3)

        mask = np.argmin(br_latencies)
        optimal_latency[ws] = (min(br_latencies), group_labels[mask], results[mask])
        print()

    for k, v in optimal_latency.items():
        print(k, v)

    if dump_data:
        with open(file_path, "wb") as f:
            pickle.dump(optimal_latency, f)


if __name__ == '__main__':

    # window_size = 9
    # num_bucket = (1 << window_size) - 1
    padd_stages = 123
    num_padd_unit = 1

    # padd_buckets = PaddBucketsWithoutDouble(num_bucket=num_bucket, padd_stages=padd_stages, num_padd_unit=num_padd_unit)
    # result = padd_buckets.cost()
    # print("\nPadd Buckets Without Double")
    # print(result)
    # # print(padd_buckets.dbg_prt())

    # padd_buckets2 = PaddBucketsDouble(num_bucket=num_bucket, padd_stages=padd_stages, num_padd_unit=num_padd_unit)
    # result2 = padd_buckets2.cost()
    # print("\nPadd Buckets Using Double")
    # print(result2)
    # # print(padd_buckets2.dbg_prt())

    # for group_size in range(2, 8):
    #     padd_buckets3 = PaddBucketsGroup(num_bucket=num_bucket, group_size=2 ** group_size, padd_stages=padd_stages,
    #                                      num_padd_unit=num_padd_unit)
    #     result3 = padd_buckets3.cost()
    #     print(f"\nPadd Buckets Using Group Size {2 ** group_size}")
    #     print(result3)
    #     # print(padd_buckets3.dbg_prt())

    # exit()
    # print()

    file_path = f"bucket_reduction_latencies_{padd_stages}_stages.pkl"
    window_sizes = [6, 7, 8, 9, 10, 11, 12]
    max_group_exponent = 8
    sweep_archs(window_sizes, max_group_exponent, padd_stages, file_path, True)
    print()
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            optimal_latency = pickle.load(f)

        for k, v in optimal_latency.items():
            print(k, v)