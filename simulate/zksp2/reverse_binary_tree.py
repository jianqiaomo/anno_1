import math
from copy import deepcopy
from .build_mle import BuildMleCostReport
import yaml
import sys
import warnings


class ReverseBinaryTreeCostReport:
    """
    Evaluate reverse binary tree

    1 2 3 4 5 6 7 8
    | / | / | / | /
    9   10  11  12
    |   /   |   /
    13      14
    |       /
    15
    """

    def __init__(self, num_vars, mod_mul_latency, num_previous_result, num_mod_mul=-1, available_bandwidth=-1):
        """
        set up reverse binary tree

        :param num_vars: (μ) e.g., 20 for 2**20
        :param mod_mul_latency: (cycle) total latency of pipelined mod mult (r/w register, compute, etc.)
        :param num_mod_mul: (-1 for unlimited mod mults to match receiving fraction mle) num of mod mult PEs
        :param num_previous_result: num of previous (fraction mle) result per cycle
        :param available_bandwidth: (bit/cycle) available bandwidth to HBM for output
        """
        self.num_vars = num_vars
        self.mod_mul_latency = mod_mul_latency
        self.num_mod_mul = num_mod_mul
        if num_previous_result < 1:
            raise ValueError("num_previous_result should be at least 1")
        self.num_previous_result = num_previous_result
        self.available_bandwidth_bperc = available_bandwidth

        self.req_mod_mul_num_per_layer = []
        self.req_reg_num_per_layer = []
        self.req_mem_bit = -1
        self.req_bandwidth_bperc = -1
        self.compulsory_cycles = -1
        self.total_cycles = -1

        self.cost_report = {}

    def cost_bandwidth_bperc(self):
        """
        compute bandwidth: assume store every reverse binary tree result to HBM

        - req_bandwidth_bperc: (bit per cycle) store every reverse binary tree result to HBM.
        :return: req_bandwidth_bperc
        """
        if len(self.req_mod_mul_num_per_layer) != 0:
            bandwidth_bperc = math.ceil(sum(self.req_mod_mul_num_per_layer)) * 255
        else:
            raise ValueError("req_mod_mul_num_per_layer is empty")

        return bandwidth_bperc

    def cost(self):
        """
        compute cost

        - total_cycles: total latency to the last output generated
        - compulsory_cycles: the cycle needed before the first output (layer 1) generated
        - req_mod_mul_num: num of required reverse binary tree PEs (mod mult)
        - req_mem_bit: num of required register size (bit)
        - req_mem_reg_num: num of required register
        - req_bandwidth_bperc: (bit per cycle) store every reverse binary tree result to HBM.
        :return: self.cost_report
        :rtype: dict
        """
        if len(self.cost_report) != 0:
            return self.cost_report
        if self.num_mod_mul == -1:
            # unlimited mod muls to match receiving fraction mle
            K = self.num_previous_result  # fraction mle sending K results per cycle

            self.req_reg_num_per_layer.append(1 if int(K) & 1 else 0)
            k0 = K
            for i in range(self.num_vars - 1):
                if k0 > 2:
                    if int(k0) & 1:  # odd
                        k0 = (k0 + 1) / 2
                        self.req_mod_mul_num_per_layer.append(k0)
                        self.req_reg_num_per_layer.append(1)
                    else:  # even
                        k0 = k0 / 2
                        self.req_mod_mul_num_per_layer.append(k0)
                        self.req_reg_num_per_layer.append(0)
                else:
                    k0 = k0 / 2
                    self.req_mod_mul_num_per_layer.append(k0)
                    self.req_reg_num_per_layer.append(1 if k0 >= 0.5 else 2)

            self.req_mod_mul_num = math.ceil(sum(self.req_mod_mul_num_per_layer))
            self.req_mem_bit = math.ceil(sum(self.req_reg_num_per_layer)) * 255

            self.req_bandwidth_bperc = self.cost_bandwidth_bperc()

            self.compulsory_cycles = 1 * self.mod_mul_latency  # included pe latency of the last layer
            self.total_cycles = (2 ** self.num_vars) // K + self.num_vars * self.mod_mul_latency

            if self.available_bandwidth_bperc != -1 and self.req_bandwidth_bperc > self.available_bandwidth_bperc:
                extra_unit_cycles = self.req_bandwidth_bperc // self.available_bandwidth_bperc - 1
                self.total_cycles = (2 ** self.num_vars) // K + self.num_vars * (
                        self.mod_mul_latency + extra_unit_cycles)
                self.req_bandwidth_bperc = self.available_bandwidth_bperc

        else:
            print(self.num_mod_mul)
            self.req_mod_mul_num = self.num_mod_mul
            raise NotImplementedError

        self.cost_report = {
            "total_cycles": self.total_cycles,
            "compulsory_cycles": self.compulsory_cycles,
            "req_mod_mul_num": self.req_mod_mul_num,
            "req_mem_bit": self.req_mem_bit,
            "req_mem_reg_num": math.ceil(sum(self.req_reg_num_per_layer)),
            'req_mod_add_num': self.req_mod_mul_num * 2,
            "bandwidth_bperc": self.req_bandwidth_bperc
        }

        return self.cost_report

    def dbg_prt(self):
        result_dict = self.cost() if len(self.cost_report) == 0 else self.cost_report
        result = f"Setting: num_vars(μ)={self.num_vars}, mod_mul_latency={self.mod_mul_latency} cycles, " \
                 f"num_mod_mul={self.num_mod_mul}, num_previous_result={self.num_previous_result}, \n" \
                 f"mod_mul per layer {self.req_mod_mul_num_per_layer}, \n" \
                 f"reg# per layer [{self.req_reg_num_per_layer[0]}] {self.req_reg_num_per_layer[1:]}, \n" \
                 f"With {result_dict['req_mod_mul_num']} mod_mul, to consume {self.num_previous_result} inputs per " \
                 f"cycle, \ncompulsory warm up in {result_dict['compulsory_cycles']} cycles, " \
                 f"and total {result_dict['total_cycles']} cycles, \n" \
                 f"it requires {result_dict['req_mem_bit']} bit register (num={result_dict['req_mem_reg_num']}), " \
                 f"and {result_dict['bandwidth_bperc']} bit/cycle bandwidth to HBM.\n"
        return result


class ProductMLECostReport(ReverseBinaryTreeCostReport):
    """
    Evaluate Product MLE cost report
    """

    def __init__(self, num_vars, mod_mul_latency, num_previous_result, num_mod_mul=-1, available_bandwidth=-1):
        """
        set up Product MLE (reverse binary tree)

        :param num_vars: (μ) e.g., 20 for 2**20
        :param mod_mul_latency: (cycle) total latency of pipelined mod mult (r/w register, compute, etc.)
        :param num_mod_mul: (-1 for unlimited mod mults to match receiving fraction mle) num of mod mult PEs
        :param num_previous_result: num of previous (fraction mle) result which is the previous step
        :param available_bandwidth: (bit/cycle) available bandwidth to HBM for MLE update
        """
        super(ProductMLECostReport, self).__init__(num_vars=num_vars, mod_mul_latency=mod_mul_latency,
                                                   num_previous_result=num_previous_result, num_mod_mul=num_mod_mul,
                                                   available_bandwidth=available_bandwidth)


class MulTreeCostReport(ReverseBinaryTreeCostReport):
    """
    Evaluate Mul Tree cost report
    """

    def __init__(self, num_vars, mod_mul_latency, num_previous_result, num_mod_mul=-1, available_bandwidth=-1):
        """
        set up MulTree (reverse binary tree)

        :param num_vars: total inputs of level 0, e.g., 20 for 2**20
        :param mod_mul_latency: (cycle) total latency of pipelined mod mult (r/w register, compute, etc.)
        :param num_mod_mul: (-1 for unlimited mod muls to match fetching) num of mod mult PEs
        :param num_previous_result: num of inputs of fetching per cycle
        :param available_bandwidth: (bit/cycle) available bandwidth to HBM for output
        """
        super(MulTreeCostReport, self).__init__(num_vars=num_vars, mod_mul_latency=mod_mul_latency,
                                                num_previous_result=num_previous_result, num_mod_mul=num_mod_mul,
                                                available_bandwidth=available_bandwidth)

    def cost_bandwidth_bperc(self):
        """
        compute bandwidth: assume no result to HBM

        - req_bandwidth_bperc: (bit per cycle) store every reverse binary tree result to HBM.
        :return: req_bandwidth_bperc
        """

        return 0


class MleEvalCostReport(MulTreeCostReport):
    """
    Cost report for one MLE evaluation within batch evaluation
    (μ round MLE updates, reverse binary tree)
    """

    def __init__(self, num_vars, mle_update_latency, mod_add_latency, num_previous_result, num_mle_update=-1,
                 num_mod_add=-1, available_bandwidth=-1):
        """
        set up one MLE evaluation

        :param num_vars: (μ) e.g., 20 for 2**20
        :param mle_update_latency: (cycle) latency of mle update (2 mod mult parallel, and 1 mod add)
        :param mod_add_latency: (cycle) total latency of pipelined mod add (for warm up cycles)
        :param num_mle_update: (-1 for unlimited mle update to match fetching) num of mle update PEs
        :param num_previous_result: num of inputs fetching per cycle
        :param num_mod_add: (-1 for unlimited mod add to match fetching) num of mod add PEs
        :param available_bandwidth: (bit/cycle) available bandwidth to HBM for output
        """
        super(MleEvalCostReport, self).__init__(num_vars=num_vars, mod_mul_latency=mle_update_latency,
                                                num_previous_result=num_previous_result, num_mod_mul=num_mle_update,
                                                available_bandwidth=available_bandwidth)
        self.mod_add_latency = mod_add_latency
        self.num_mod_add = num_vars if num_mod_add == -1 else num_mod_add

    def cost(self):
        """
        compute cost of one MLE evaluation

        - total_cycles: total latency to the last output generated
        - compulsory_cycles: mod add + the cycle needed before the first output (layer 1) generated
        - req_mod_mul_num: num of required mle update PE
        - req_mem_bit: num of required register size (bit)
        - req_mem_reg_num: num of required register
        - req_bandwidth_bperc: (bit per cycle) store every reverse binary tree result to HBM.
        :return: self.cost_report
        :rtype: dict
        """
        if len(self.cost_report) != 0:
            return self.cost_report
        if self.num_vars == 0:
            self.compulsory_cycles = 1
            self.total_cycles = 1
            self.num_mod_add = 0
            self.num_mod_mul = 0
            self.req_mem_bit = 0
            self.req_bandwidth_bperc = 0
            self.cost_report = {
                'total_cycles': self.total_cycles,
                'compulsory_cycles': self.compulsory_cycles,
                'req_mod_mul_num': self.num_mod_mul,
                'req_mem_bit': self.req_mem_bit,
                'req_mem_reg_num': 0,
                'req_bandwidth_bperc': self.req_bandwidth_bperc,
                'req_mod_add_num': self.num_mod_add
            }
            return self.cost_report

        result_dict = deepcopy(super(MleEvalCostReport, self).cost())

        extra_mod_add_cycles = self.mod_add_latency + (self.num_vars - 1) // self.num_mod_add  # pipelined mod add
        self.compulsory_cycles += extra_mod_add_cycles
        self.total_cycles += extra_mod_add_cycles

        self.req_reg_num_per_layer.append(2 * self.num_vars)  # can optimize to SRAM

        result_dict['compulsory_cycles'] = self.compulsory_cycles
        result_dict['total_cycles'] = self.total_cycles
        result_dict['req_mod_add_num'] = self.num_mod_add + result_dict['req_mod_mul_num']
        # result_dict['req_mod_mul_num'] *= 2  # 2 parallel mod mults per mle update
        result_dict['req_mod_mul_num'] *= 1  # 1 mod mults per mle update
        result_dict['req_mem_bit'] = math.ceil(sum(self.req_reg_num_per_layer)) * 255
        result_dict['req_mem_reg_num'] = math.ceil(sum(self.req_reg_num_per_layer))
        # self.req_bandwidth_bperc = 0
        # result_dict['bandwidth_bperc'] = self.req_bandwidth_bperc

        self.cost_report = result_dict
        return result_dict

    def dbg_prt(self):
        result_dict = self.cost() if len(self.cost_report) == 0 else self.cost_report
        result = f"Setting: num_vars(μ)={self.num_vars}, mle_update_latency={self.mod_mul_latency} cycles, " \
                 f"mod_add_latency={self.mod_add_latency} cycles, num_mod_add={self.num_mod_add}, " \
                 f"num_mle_update={self.num_mod_mul}, num_previous_result={self.num_previous_result}, " \
                 f"available_bandwidth={self.available_bandwidth_bperc} bit/cycle, \n" \
                 f"num inputs per cycle={self.num_previous_result} result/cycle \n" \
                 f"mle_update_PE per layer {self.req_mod_mul_num_per_layer}, \n" \
                 f"reg# per layer [{self.req_reg_num_per_layer[0]}] {self.req_reg_num_per_layer[1:-1]} " \
                 f"[{self.req_reg_num_per_layer[-1]}], \n" \
                 f"With {result_dict['req_mod_mul_num']} mle_update_PE, {result_dict['req_mod_add_num']} mod_add, " \
                 f"to consume {self.num_previous_result} inputs per cycle, \n" \
                 f"mod add + compulsory warm up in {result_dict['compulsory_cycles']} cycles, " \
                 f"and total {result_dict['total_cycles']} cycles, \n" \
                 f"it requires {result_dict['req_mem_bit']} bit register (num={result_dict['req_mem_reg_num']}), " \
                 f"and {result_dict['bandwidth_bperc']} bit/cycle bandwidth to HBM.\n"
        return result


class MultifuncTreeCostReport:
    """
    General cost report for the multifunctional tree unit.
    (μ round MLE updates, reverse binary tree)
    """

    def __init__(self, mod_mul_latency, mod_add_latency, fan_in_or_out, available_bandwidth=-1):
        """
        set up one MLE evaluation

        :param mod_mul_latency: (cycle) latency of mod_mul
        :param mod_add_latency: (cycle) total latency of pipelined mod add (for warm up cycles)
        :param num_PE: num of PEs, can be 1+1, 2+1+1, 4+2+1+1, 8+4+2+1+1, 16+8+4+2+1+1, etc.
        :param fan_in_or_out: fan_in_or_out per cycle
        :param available_bandwidth: (bit/cycle) available bandwidth to HBM for output
        """
        self.mod_mul_latency = mod_mul_latency
        self.mod_add_latency = mod_add_latency
        # if num_PE & (num_PE - 1) != 0 or num_PE <= 1:
        #     raise ValueError(f"num_PE should be 2's power, but got {num_PE}")
        # self.num_PE = num_PE
        if fan_in_or_out % 2 != 0 or fan_in_or_out <= 1:
            # print(f"fan_in_or_out should be even, but got {fan_in_or_out}, set to even")
            fan_in_or_out = fan_in_or_out + 1
        self.available_bandwidth_bperc = available_bandwidth
        self.cost_report = {}

        self.fan_in_or_out = fan_in_or_out

    def cost(self, num_vars, num_in_or_out=-1):
        """
        compute cost of one MLE evaluation.

        :param num_vars: (μ) e.g., 20 for 2**20
        :param num_in_or_out: num of inputs (Product MLE) or outputs (Build MLE) per cycle

        - total_cycles: total latency to the last output generated
        - compulsory_cycles: mod add + the cycle needed before the first output (layer 1) generated
        - req_mod_mul_num: num of required mle update PE
        - req_mem_bit: num of required register size (bit)
        - req_mem_reg_num: num of required register
        - req_bandwidth_bperc: (bit per cycle) store every reverse binary tree result to HBM.
        :return: self.cost_report
        :rtype: dict
        """
        self.num_vars = num_vars
        build_mle_cost = BuildMleCostReport(num_vars=self.num_vars, mod_mul_latency=self.mod_mul_latency,
                                            mod_add_latency=self.mod_add_latency,
                                            num_zerocheck_pes=self.fan_in_or_out / 2,
                                            num_mod_mul=-1, num_mod_add=-1, available_bandwidth=-1)
        self.build_mle_result = build_mle_cost.cost()

        prod_mle_cost = ProductMLECostReport(num_vars=self.num_vars, mod_mul_latency=self.mod_mul_latency,
                                             num_previous_result=self.fan_in_or_out, num_mod_mul=-1,
                                             available_bandwidth=-1)
        self.product_mle_result = prod_mle_cost.cost()

        mul_tree_cost = MulTreeCostReport(num_vars=self.num_vars, mod_mul_latency=self.mod_mul_latency,
                                          num_previous_result=self.fan_in_or_out, num_mod_mul=-1,
                                          available_bandwidth=-1)
        self.mul_tree_result = mul_tree_cost.cost()

        mle_eval_cost = MleEvalCostReport(num_vars=self.num_vars, mle_update_latency=self.mod_mul_latency,
                                          mod_add_latency=self.mod_add_latency, num_previous_result=self.fan_in_or_out,
                                          num_mod_add=-1, num_mle_update=-1, available_bandwidth=-1)
        self.mle_eval_result = mle_eval_cost.cost()

        # num_in_or_out should be 2's power
        if num_in_or_out != -1 and num_in_or_out & (num_in_or_out - 1) != 0:
            raise ValueError(f"num_in_or_out should be 2's power, but got {num_in_or_out}")
        if num_in_or_out == -1:
            all_modules = [self.build_mle_result, self.product_mle_result, self.mul_tree_result, self.mle_eval_result]
            hardware_requirement = {
                "req_mem_bit": max([module["req_mem_bit"] for module in all_modules]),
                "req_mem_reg_num": max([module["req_mem_reg_num"] for module in all_modules]),
                'req_mod_add_num': max([module["req_mod_add_num"] for module in all_modules]),
                "req_mod_mul_num": max([module["req_mod_mul_num"] for module in all_modules]),
            }
            return {
                "build_mle_cost": self.build_mle_result,
                "product_mle_cost": self.product_mle_result,
                "mul_tree_cost": self.mul_tree_result,
                "mle_eval_cost": self.mle_eval_result,
                "max_hardware_requirement": hardware_requirement
            }
        else:
            build_mle_result_new = self.build_mle_result.copy()
            product_mle_result_new = self.product_mle_result.copy()
            mul_tree_result_new = self.mul_tree_result.copy()
            mle_eval_result_new = self.mle_eval_result.copy()

            if num_in_or_out >= self.fan_in_or_out:
                # provide more than it can handle, need buffers and II
                if num_in_or_out > self.fan_in_or_out:
                    rate = f"Provide more than Tree can handle, need buffers and II stall. " \
                           f"It should input/output {self.fan_in_or_out} times every {num_in_or_out} cycles."
                else:
                    rate = f"Provide exactly what Tree can handle."

                build_mle_result_new["req_mem_reg_num"] += num_in_or_out - self.fan_in_or_out
                build_mle_result_new["req_mem_bit"] += (num_in_or_out - self.fan_in_or_out) * 255
                product_mle_result_new["req_mem_reg_num"] += num_in_or_out - self.fan_in_or_out
                product_mle_result_new["req_mem_bit"] += (num_in_or_out - self.fan_in_or_out) * 255
                mul_tree_result_new["req_mem_reg_num"] += num_in_or_out - self.fan_in_or_out
                mul_tree_result_new["req_mem_bit"] += (num_in_or_out - self.fan_in_or_out) * 255
                mle_eval_result_new["req_mem_reg_num"] += num_in_or_out - self.fan_in_or_out
                mle_eval_result_new["req_mem_bit"] += (num_in_or_out - self.fan_in_or_out) * 255
                all_modules = [build_mle_result_new, product_mle_result_new, mul_tree_result_new, mle_eval_result_new]
                hardware_requirement = {
                    "req_mem_bit": max([module["req_mem_bit"] for module in all_modules]),
                    "req_mem_reg_num": max([module["req_mem_reg_num"] for module in all_modules]),
                    'req_mod_add_num': max([module["req_mod_add_num"] for module in all_modules]),
                    "req_mod_mul_num": max([module["req_mod_mul_num"] for module in all_modules]),
                }

                return {
                    "build_mle_cost": build_mle_result_new,
                    "product_mle_cost": product_mle_result_new,
                    "mul_tree_cost": mul_tree_result_new,
                    "mle_eval_cost": mle_eval_result_new,
                    "rate": rate,
                    "max_hardware_requirement": hardware_requirement
                }

            else:  # provide less than it will handle, need longer latency
                rate = f"Provide less than Tree will handle, slower, longer latency."

                build_mle_cost = BuildMleCostReport(num_vars=self.num_vars, mod_mul_latency=self.mod_mul_latency,
                                                    mod_add_latency=self.mod_add_latency,
                                                    num_zerocheck_pes=self.fan_in_or_out / 2,
                                                    num_mod_mul=-1, num_mod_add=-1, available_bandwidth=-1)
                build_mle_result_smaller = build_mle_cost.cost()
                prod_mle_cost = ProductMLECostReport(num_vars=self.num_vars, mod_mul_latency=self.mod_mul_latency,
                                                     num_previous_result=self.fan_in_or_out, num_mod_mul=-1,
                                                     available_bandwidth=-1)
                product_mle_result_smaller = prod_mle_cost.cost()
                mul_tree_cost = MulTreeCostReport(num_vars=self.num_vars, mod_mul_latency=self.mod_mul_latency,
                                                  num_previous_result=self.fan_in_or_out, num_mod_mul=-1,
                                                  available_bandwidth=-1)
                mul_tree_result_smaller = mul_tree_cost.cost()
                mle_eval_cost = MleEvalCostReport(num_vars=self.num_vars, mle_update_latency=self.mod_mul_latency,
                                                  mod_add_latency=self.mod_add_latency,
                                                  num_previous_result=self.fan_in_or_out,
                                                  num_mod_add=-1, num_mle_update=-1, available_bandwidth=-1)
                mle_eval_result_smaller = mle_eval_cost.cost()

                build_mle_result_new["compulsory_cycles"] = build_mle_result_smaller["compulsory_cycles"]
                build_mle_result_new["total_cycles"] = build_mle_result_smaller["total_cycles"]
                product_mle_result_new["compulsory_cycles"] = product_mle_result_smaller["compulsory_cycles"]
                product_mle_result_new["total_cycles"] = product_mle_result_smaller["total_cycles"]
                mul_tree_result_new["compulsory_cycles"] = mul_tree_result_smaller["compulsory_cycles"]
                mul_tree_result_new["total_cycles"] = mul_tree_result_smaller["total_cycles"]
                mle_eval_result_new["compulsory_cycles"] = mle_eval_result_smaller["compulsory_cycles"]
                mle_eval_result_new["total_cycles"] = mle_eval_result_smaller["total_cycles"]
                all_modules = [build_mle_result_new, product_mle_result_new, mul_tree_result_new, mle_eval_result_new]
                hardware_requirement = {
                    "req_mem_bit": max([module["req_mem_bit"] for module in all_modules]),
                    "req_mem_reg_num": max([module["req_mem_reg_num"] for module in all_modules]),
                    'req_mod_add_num': max([module["req_mod_add_num"] for module in all_modules]),
                    "req_mod_mul_num": max([module["req_mod_mul_num"] for module in all_modules]),
                }

                return {
                    "build_mle_cost": build_mle_result_new,
                    "product_mle_cost": product_mle_result_new,
                    "mul_tree_cost": mul_tree_result_new,
                    "mle_eval_cost": mle_eval_result_new,
                    "rate": rate,
                    "max_hardware_requirement": hardware_requirement
                }


setup_config_dict = {
    "basic": {
        "mod_mul_latency": 10,
        "mod_add_latency": 1,
    },
    "multiply_lane_tree": {  # for sumcheck:
        "num_input_entries_per_cycle": 4,  # num_eval_engines
        # "num_eval_engines": 4,  # num_input_entries_per_cycle
        "num_lanes_per_sc_pe": 5,
        "total_sc_pe": 1
    },
    # "mle_batch_eval": {  # for batch eval:
    #     "num_input_entries_per_cycle": 4,  # we'd better match it with multiply_lane_tree, to have uniform trees
    #     "num_parallel_eval": 7,  # we'd better match it = num_lanes_per_sc_pe * total_sc_pe + 1
    # },
    # "build_mle": {
    #     "num_gen_entries_per_cycle": 4,  # should be even
    #     "buffer_size": 64,  # number of entries in the buffer
    #     "total_sc_pe": 2,
    # },
    # "compute_product_mle": {
    #     "num_input_entries_per_cycle": 4,  # should be even
    # },
}


class shared_tree_cost():
    """
    Given the setup_config_dict, compute the amount of tree needed.
    setup_config_dict = {
    "basic": {
        "mod_mul_latency": 10,
        "mod_add_latency": 1,
    },
    "multiply_lane_tree": {  # for sumcheck:
        "num_input_entries_per_cycle": 4,  # num_eval_engines
        "num_lanes_per_sc_pe": 3,
        "total_sc_pe": 2
    },
    }
    """

    def __init__(self, setup_config_dict):
        self.mod_mul_latency = setup_config_dict["basic"]["mod_mul_latency"]
        self.mod_add_latency = setup_config_dict["basic"]["mod_add_latency"]

        self.multiply_lane_tree = setup_config_dict["multiply_lane_tree"]
        # self.mle_batch_eval = setup_config_dict["mle_batch_eval"]

        # compute the number of tree modules needed
        num_trees_need_by_all_sc = self.multiply_lane_tree["num_lanes_per_sc_pe"] * self.multiply_lane_tree[
            "total_sc_pe"]
        # num_trees_need_by_batch_eval = self.mle_batch_eval["num_parallel_eval"]
        num_trees_need_by_batch_eval = num_trees_need_by_all_sc + 1 * self.multiply_lane_tree["total_sc_pe"]
        # use one more tree for build MLE if needed
        # num_trees_need = (num_trees_need_by_all_sc + 1) if \
        #     num_trees_need_by_all_sc >= num_trees_need_by_batch_eval else num_trees_need_by_batch_eval
        # if num_trees_need_by_all_sc >= num_trees_need_by_batch_eval:
        #     # raise warning: the number of trees in batch eval should be larger than that in sumcheck
        #     warnings.warn(f"The number of trees in batch eval should be greater than that in sumcheck, "
        #                   f"current batch eval trees: {num_trees_need_by_batch_eval}, "
        #                   f"sumcheck lanes: {num_trees_need_by_all_sc}")
        num_trees_need = (num_trees_need_by_all_sc + 1 * self.multiply_lane_tree["total_sc_pe"])  # a PE has own build MLE

        # compute the size of tree modules. Use a uniform size for all trees
        # num_inout_per_sc_lane = self.multiply_lane_tree["num_input_entries_per_cycle"] if \
        #     self.multiply_lane_tree["num_input_entries_per_cycle"] % 2 == 0 else (
        #             self.multiply_lane_tree["num_input_entries_per_cycle"] + 1)
        num_inout_per_sc_lane = self.multiply_lane_tree["num_input_entries_per_cycle"]
        # num_inout_per_batch_eval = self.mle_batch_eval["num_input_entries_per_cycle"] if \
        #     self.mle_batch_eval["num_input_entries_per_cycle"] % 2 == 0 else (
        #             self.mle_batch_eval["num_input_entries_per_cycle"] + 1)
        num_inout_per_batch_eval = num_inout_per_sc_lane

        tree_info = {
            "num_inout": 0,  # size
            "function": [],
        }
        self.tree_size_info = []
        # if the more # of tree needed & # of input/output per tree is higher, then all trees should be at higher size
        # else, for the lesser one, we can use the smaller size; and the rest of the trees should be at higher size
        if num_trees_need_by_batch_eval > num_trees_need_by_all_sc:
            if num_inout_per_batch_eval >= num_inout_per_sc_lane:
                tree_info0 = deepcopy(tree_info)
                tree_info0["num_inout"] = num_inout_per_batch_eval  # size of the larger
                tree_info0["function"] = ["mle_batch_eval"]
                for _ in range(num_trees_need_by_batch_eval):
                    self.tree_size_info.append(deepcopy(tree_info0))
                for i in range(num_trees_need_by_all_sc):
                    self.tree_size_info[i]["function"].append("multiply_lane_tree")
                for i in range(num_trees_need_by_all_sc, num_trees_need_by_batch_eval):
                    self.tree_size_info[i]["function"].append("build_mle")
                self.tree_size_info[num_trees_need_by_all_sc]["function"] += ["compute_product_mle"]
            else:
                tree_info0 = deepcopy(tree_info)
                tree_info0["num_inout"] = num_inout_per_sc_lane  # size of the larger
                tree_info0["function"] = ["multiply_lane_tree", "mle_batch_eval"]
                for _ in range(num_trees_need_by_all_sc):
                    self.tree_size_info.append(deepcopy(tree_info0))
                tree_info1 = deepcopy(tree_info)
                tree_info1["num_inout"] = num_inout_per_batch_eval  # size of the smaller
                tree_info1["function"] = ["mle_batch_eval", "build_mle", "compute_product_mle"]
                self.tree_size_info.append(deepcopy(tree_info1))
                tree_info1["function"] = ["mle_batch_eval"]
                for i in range(num_trees_need_by_batch_eval - 1 - num_trees_need_by_all_sc):
                    self.tree_size_info.append(deepcopy(tree_info1))
        else:  # num_trees_need_by_all_sc >= num_trees_need_by_batch_eval
            if num_inout_per_sc_lane >= num_inout_per_batch_eval:
                tree_info0 = deepcopy(tree_info)
                tree_info0["num_inout"] = num_inout_per_sc_lane  # size of the larger
                tree_info0["function"] = ["multiply_lane_tree"]
                for _ in range(num_trees_need_by_all_sc):
                    self.tree_size_info.append(deepcopy(tree_info0))
                tree_info1 = deepcopy(tree_info)
                tree_info1["num_inout"] = num_inout_per_batch_eval  # size of the smaller
                tree_info1["function"] = ["build_mle"]
                self.tree_size_info.append(deepcopy(tree_info1))
                for i in range(num_trees_need_by_batch_eval):
                    self.tree_size_info[i]["function"].append("mle_batch_eval")
                self.tree_size_info[0]["function"].append("compute_product_mle")
            else:
                tree_info0 = deepcopy(tree_info)
                tree_info0["num_inout"] = num_inout_per_batch_eval  # size of the larger
                tree_info0["function"] = ["multiply_lane_tree", "mle_batch_eval"]
                for _ in range(num_trees_need_by_batch_eval):
                    self.tree_size_info.append(deepcopy(tree_info0))
                self.tree_size_info[0]["function"].append("compute_product_mle")
                tree_info1 = deepcopy(tree_info)
                tree_info1["num_inout"] = num_inout_per_sc_lane  # size of the smaller
                tree_info1["function"] = ["multiply_lane_tree"]
                for _ in range(num_trees_need_by_all_sc - num_trees_need_by_batch_eval):
                    self.tree_size_info.append(deepcopy(tree_info1))
                tree_info1["function"] = ["build_mle"]
                self.tree_size_info.append(deepcopy(tree_info1))
        if len(self.tree_size_info) != num_trees_need:
            raise ValueError(
                f"Error in tree size info generation: {len(self.tree_size_info)} != {num_trees_need}"
            )

    def get_hardware_cost_cost(self):
        tree = MultifuncTreeCostReport(mod_mul_latency=self.mod_mul_latency, mod_add_latency=self.mod_add_latency,
                                       fan_in_or_out=self.multiply_lane_tree['num_input_entries_per_cycle'])
        tree_cost = tree.cost(20)
        hardware_cost = {
            'req_mem_bit': max([tree['req_mem_bit'] for tree in tree_cost.values()]) * len(self.tree_size_info),
            'req_mem_reg_num': max([tree['req_mem_reg_num'] for tree in tree_cost.values()]) * len(self.tree_size_info),
            'req_mod_add_num': max([tree['req_mod_add_num'] for tree in tree_cost.values()]) * len(self.tree_size_info),
            'req_mod_mul_num': max([tree['req_mod_mul_num'] for tree in tree_cost.values()]) * len(self.tree_size_info),
            'num_lanes': len(self.tree_size_info),
        }
        return hardware_cost

    def get_build_x_mle_cost(self, x: int, buffer_entries: int):
        """
        Returns the cost report for build 6 MLE into buffers.
        Use Result['total_cycles'] (or Result['compulsory_cycles']).
        """
        if x == 0:
            raise ValueError("x cannot be 0")
        elif x == 1:
            return self.get_build_mle_cost(buffer_entries)
        elif x <= len(self.tree_size_info):
            return self.get_build_mle_cost(buffer_entries)
        else:
            cost = self.get_build_mle_cost(buffer_entries)
            serial_times = math.ceil(x / len(self.tree_size_info))
            cost["total_cycles"] = cost["total_cycles"] * serial_times
            cost["compulsory_cycles"] = cost["compulsory_cycles"] * serial_times
            return cost

    def get_build_mle_cost(self, buffer_entries: int):
        """
        Returns the cost report for build MLE.
        Use Result['total_cycles'] (or Result['compulsory_cycles']).
        """
        build_mle = MultifuncTreeCostReport(mod_mul_latency=self.mod_mul_latency, mod_add_latency=self.mod_add_latency,
                                            fan_in_or_out=self.multiply_lane_tree['num_input_entries_per_cycle'])
        cost = build_mle.cost(int(math.log2(buffer_entries)))["build_mle_cost"]
        return cost

    def get_build_fz_mle_before_sc_rd1_cost(self, buffer_entries: int):
        """
        Build fz MLE before SC round 1 (build the very beginning mle).
        We use all lanes to build the MLE at the very beginning.
        Use Result['total_cycles'] (or Result['compulsory_cycles']).
        """
        build_mle = MultifuncTreeCostReport(mod_mul_latency=self.mod_mul_latency, mod_add_latency=self.mod_add_latency,
                                            fan_in_or_out=self.multiply_lane_tree['num_input_entries_per_cycle'])
        cost = build_mle.cost(int(math.log2(buffer_entries)))["build_mle_cost"]
        cost["total_cycles"] = cost["total_cycles"] // len(self.tree_size_info)
        return cost

    def get_compute_product_mle_cost(self, num_vars: int):
        """
        Returns the cost report for compute product MLE.
        Use Result['total_cycles'] (or Result['compulsory_cycles']).
        """
        compute_product_mle = MultifuncTreeCostReport(mod_mul_latency=self.mod_mul_latency,
                                                      mod_add_latency=self.mod_add_latency,
                                                      fan_in_or_out=self.multiply_lane_tree[
                                                          'num_input_entries_per_cycle'])
        cost = compute_product_mle.cost(num_vars)['product_mle_cost']
        return cost

    def get_mle_batch_eval_cost(self, num_vars_list: list):
        """
        Returns the cycles for MLE evaluation in each tree.
        Use: max(result) is the cost of the entire MLE batch evaluation.
        """
        num_vars_list_sorted = sorted(num_vars_list)
        batch_eval = MultifuncTreeCostReport(mod_mul_latency=self.mod_mul_latency, mod_add_latency=self.mod_add_latency,
                                             fan_in_or_out=self.multiply_lane_tree['num_input_entries_per_cycle'])
        total_cost_all_trees = [deepcopy([]) for _ in range(len(self.tree_size_info))]
        idx = 0
        for num_vars in num_vars_list_sorted:
            cost = batch_eval.cost(num_vars)["mle_eval_cost"]
            total_cost_all_trees[idx].append(cost)
            idx = (idx + 1) % len(self.tree_size_info)
        compulsory_cycles_all_trees = []
        for cost_per_tree in total_cost_all_trees:
            compulsory_cycles_all_trees.append(sum([
                cost["compulsory_cycles"] for cost in cost_per_tree
            ]))
        total_cycles_all_trees = []
        for cost_per_tree in total_cost_all_trees:
            total_cycles_all_trees.append(sum([
                cost["total_cycles"] for cost in cost_per_tree
            ]))
        return total_cycles_all_trees

    def get_mul_tree_cost(self, fan_in: int, num_total_entries: int, parallel=1):
        """
        Returns the cycles for multiplication tree.
        Use Result['total_cycles'] (or Result['compulsory_cycles']).
        """
        num_vars = int(math.log2(num_total_entries))
        if parallel > len(self.tree_size_info):
            raise NotImplementedError
        else:
            if fan_in > self.multiply_lane_tree['num_input_entries_per_cycle']:
                print(f"Warning: fan_in {fan_in} is larger than the number of input entries allowed "
                      f"per cycle {self.multiply_lane_tree['num_input_entries_per_cycle']}.")
            mul_tree = MultifuncTreeCostReport(mod_mul_latency=self.mod_mul_latency,
                                               mod_add_latency=self.mod_add_latency,
                                               fan_in_or_out=min(fan_in, self.multiply_lane_tree['num_input_entries_per_cycle']))
            cost = mul_tree.cost(num_vars)["mul_tree_cost"]
            return cost


if __name__ == '__main__':
    # computation density
    for num_vars in range(17, 24):
        print(f"Build MLE  : μ={num_vars} requires {sum(2 ** i for i in range(1, num_vars))} modmul ops.")
        print(f"Mult Tree  : μ={num_vars} requires {sum(2 ** i for i in range(num_vars))} modmul ops.")
        # print(f"Batch Eval : μ={num_vars} requires {2 * sum(2 ** i for i in range(num_vars))} modmul ops.")
        print(f"Batch Eval : μ={num_vars} requires {sum(2 ** i for i in range(num_vars))} modmul ops.")
        print(f"Product MLE: μ={num_vars} requires {sum(2 ** i for i in range(num_vars))} modmul ops.")

    print("product mle result")
    prod_mle_cost = ProductMLECostReport(num_vars=5, mod_mul_latency=1, num_previous_result=4, num_mod_mul=-1,
                                         available_bandwidth=-1)
    product_mle_result = prod_mle_cost.cost()
    print(product_mle_result)
    print()

    # debug
    print(prod_mle_cost.dbg_prt())

    print("mul tree for mod inv precompute")
    mul_tree_cost = MulTreeCostReport(num_vars=10, mod_mul_latency=1, num_previous_result=8, num_mod_mul=-1,
                                      available_bandwidth=-1)
    mul_tree_result = mul_tree_cost.cost()
    print(mul_tree_result)
    print()

    # debug
    print(mul_tree_cost.dbg_prt())

    print("MLE evaluation in batch evaluation result")
    mle_eval_cost = MleEvalCostReport(num_vars=10, mle_update_latency=1, mod_add_latency=1, num_previous_result=4,
                                      num_mod_add=-1, num_mle_update=-1, available_bandwidth=-1)
    mle_eval_result = mle_eval_cost.cost()
    print(mle_eval_result)
    print()

    # debug
    print(mle_eval_cost.dbg_prt())

    multifunc_tree = MultifuncTreeCostReport(mod_mul_latency=10, mod_add_latency=1, fan_in_or_out=8)
    multifunc_tree_result = multifunc_tree.cost(num_vars=6, num_in_or_out=64)
    yaml.dump(multifunc_tree_result, sys.stdout)
    multifunc_tree_result = multifunc_tree.cost(num_vars=20, num_in_or_out=8)
    yaml.dump(multifunc_tree_result, sys.stdout)

    a = shared_tree_cost(setup_config_dict)

    print(a.get_hardware_cost_cost())
    print(a.get_build_x_mle_cost(6, 1024))
    print(a.get_build_mle_cost(buffer_entries=1024))
    print(a.get_build_fz_mle_before_sc_rd1_cost(buffer_entries=1024))
    print(a.get_compute_product_mle_cost(num_vars=6))
    print(a.get_mle_batch_eval_cost(num_vars_list=[5, 5, 5, 5, 6, 6, 6, 20]))
    print(a.get_mul_tree_cost(fan_in=4, num_total_entries=64))
