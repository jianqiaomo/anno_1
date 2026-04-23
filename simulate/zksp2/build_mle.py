import math


class BuildMleNonContinuousCostReport:
    """
    evaluate the cost of building MLE, non-continuous output index (1 N/2, 2 N/2+1, 3 N/2+2, ...)
    """

    def __init__(self, num_vars, mod_mul_latency, mod_add_latency, num_zerocheck_pes, num_mod_mul=-1, num_mod_add=-1,
                 available_bandwidth=-1):
        """
        set up building MLE

        :param num_vars: (μ) e.g., 20 for 2**20
        :param mod_mul_latency: (cycle) total latency of pipelined mod mult (r/w register, compute, etc.)
        :param mod_add_latency: (cycle) total latency of pipelined mod add (for warm up cycles)
        :param num_mod_mul: (-1 for unlimited mod mults to match following zerocheck PEs) num of mod mult PEs
        :param num_mod_add: num of mod add (for warm up cycles)
        :param num_zerocheck_pes: num of the zerocheck PEs which is the next step consuming build MLE, assume 2 non-consecutive reads per cycle
        :param available_bandwidth: (bit/cycle) available bandwidth to HBM for MLE update
        """
        self.num_vars = num_vars
        self.mod_mul_latency = mod_mul_latency
        self.mod_add_latency = mod_add_latency
        self.num_mod_mul = num_mod_mul
        self.num_mod_add = num_vars if num_mod_add == -1 else num_mod_add
        if num_zerocheck_pes < 1:
            raise ValueError("num_zerocheck_pes should be at least 1")
        self.num_zerocheck_pes = num_zerocheck_pes
        self.available_bandwidth_bperc = available_bandwidth

        self.cost_report = {}

    def reg_cost(self, req_pe_num_per_layer):
        """
        compute the cost of num registers

        :param req_pe_num_per_layer: num of required PEs per layer
        :type req_pe_num_per_layer: list
        :return: req reg num per layer
        :rtype: list
        """
        req_reg_num_per_layer = []
        for pe_num in req_pe_num_per_layer[:-1]:
            if pe_num < 1:
                req_reg_num_per_layer.append(2)
            elif pe_num == 1:
                req_reg_num_per_layer.append(4)
            else:
                req_reg_num_per_layer.append(pe_num)
        req_reg_num_per_layer.append(2 * self.num_vars)
        return req_reg_num_per_layer

    def mod_mul_cost(self, num_non_continuous_output) -> list:
        """
        compute the cost of num mod mults

        :param num_non_continuous_output: num of non-continuous output
        :return: req mod_mul num per layer
        """
        # req_mod_mul_num_per_layer = []
        # # consider strategy: Least Common Multiple (LCM)
        # k0 = num_non_continuous_output
        # for i in range(self.num_vars - 1):
        #     if k0 > 1:
        #         if int(k0) & 1:  # odd
        #             k0 = k0 + 1
        #             req_mod_mul_num_per_layer.insert(0, k0)
        #             k0 = k0 / 2
        #         else:  # even
        #             req_mod_mul_num_per_layer.insert(0, k0)
        #             k0 = k0 / 2
        #     else:
        #         req_mod_mul_num_per_layer.insert(0, k0)
        #         k0 = k0 / 2
        # return req_mod_mul_num_per_layer
        raise NotImplementedError

    def cost(self):
        """
        compute cost

        - total_cycles: total latency to build MLE
        - compulsory_cycles: compute (1-r) and the cycle needed before the first output generated (cannot overlap with zerocheck)
        - req_mod_mul_num: num of required build MLE PEs (mod mult)
        - req_mod_add_num: num of required mod add
        - req_mem_bit: num of required register size (bit)
        - req_mem_reg_num: num of required register
        - bandwidth_bperc: (bit per cycle) store build MLE to HBM for MLE update.
        :return: self.cost_report
        :rtype: dict
        """
        if len(self.cost_report) != 0:
            return self.cost_report
        self.req_mod_mul_num_per_layer = []
        self.req_reg_num_per_layer = []
        self.req_mem_bit = -1
        self.req_bandwidth_bperc = -1
        self.compulsory_cycles = -1
        self.total_cycles = -1

        if self.num_mod_mul == -1:
            # unlimited mod mults to match following consumption in zerocheck PEs
            K = self.num_zerocheck_pes * 2  # zerocheck cores consume K non-consecutive buildMLE per cycle

            self.req_mod_mul_num_per_layer = self.mod_mul_cost(K)
            self.req_mod_mul_num = math.ceil(sum(self.req_mod_mul_num_per_layer))

            self.req_reg_num_per_layer = self.reg_cost(self.req_mod_mul_num_per_layer)
            self.req_mem_bit = math.ceil(sum(self.req_reg_num_per_layer)) * 255
            self.req_bandwidth_bperc = K * 255

            warmup_layer_num = self.num_vars - 1
            # included pe latency of the last layer
            self.compulsory_cycles = warmup_layer_num * self.mod_mul_latency + self.mod_add_latency + (
                    self.num_vars - 1) // self.num_mod_add  # pipelined mod add
            self.total_cycles = self.compulsory_cycles + (2 ** self.num_vars - K) // K

            if self.available_bandwidth_bperc != -1 and self.req_bandwidth_bperc > self.available_bandwidth_bperc:
                extra_unit_cycles = self.req_bandwidth_bperc // self.available_bandwidth_bperc - 1
                self.compulsory_cycles += extra_unit_cycles * (2 ** self.num_vars) // K
                self.total_cycles += extra_unit_cycles * (2 ** self.num_vars) // K
                self.req_bandwidth_bperc = self.available_bandwidth_bperc

        else:
            self.req_mod_mul_num = self.num_mod_mul
            raise NotImplementedError

        self.cost_report = {
            "total_cycles": self.total_cycles,
            "compulsory_cycles": self.compulsory_cycles,
            "req_mod_mul_num": self.req_mod_mul_num,
            "req_mod_add_num": self.num_mod_add,
            "req_mem_bit": self.req_mem_bit,
            "req_mem_reg_num": math.ceil(sum(self.req_reg_num_per_layer)),
            "bandwidth_bperc": self.req_bandwidth_bperc
        }
        return self.cost_report

    def dbg_prt(self):
        result_dict = self.cost() if len(self.cost_report) == 0 else self.cost_report
        result = f"Setting: num_vars(μ)={self.num_vars}, mod_mul_latency={self.mod_mul_latency} cycles, " \
                 f"mod_add_latency={self.mod_add_latency} cycles, num_mod_add={self.num_mod_add}, " \
                 f"num_mod_mul={self.num_mod_mul}, num_zerocheck_pes={self.num_zerocheck_pes}, " \
                 f"available_bandwidth={self.available_bandwidth_bperc} bit/cycle, \n" \
                 f"zerocheck_consuming={2 * self.num_zerocheck_pes} result/cycle \n" \
                 f"mod_mul per layer {self.req_mod_mul_num_per_layer}, \n" \
                 f"reg# per layer {self.req_reg_num_per_layer[:-1]} [{self.req_reg_num_per_layer[-1]}], \n" \
                 f"With {result_dict['req_mod_mul_num']} mod_mul, {result_dict['req_mod_add_num']} mod_add, " \
                 f"to feed {self.num_zerocheck_pes} zerocheck core, \n" \
                 f"mod add + compulsory warm up in {result_dict['compulsory_cycles']} cycles, " \
                 f"and total {result_dict['total_cycles']} cycles, \n" \
                 f"it requires {result_dict['req_mem_bit']} bit register (num={result_dict['req_mem_reg_num']}), " \
                 f"and {result_dict['bandwidth_bperc']} bit/cycle bandwidth to HBM.\n"
        return result


class BuildMleCostReport(BuildMleNonContinuousCostReport):
    """
    Compute the cost of build MLE with continuous output index (1-2, 3-4, 5-6, ...)
    """
    def reg_cost(self, req_pe_num_per_layer):
        """
        compute the cost of num registers (continuous output index)

        :param req_pe_num_per_layer: num of required PEs per layer
        :type req_pe_num_per_layer: list
        :return: req reg num per layer
        :rtype: list
        """
        req_reg_num_per_layer = []
        for pe_num in req_pe_num_per_layer[:-1]:
            if pe_num <= 1:
                req_reg_num_per_layer.append(1)
            else:
                req_reg_num_per_layer.append(pe_num)
        req_reg_num_per_layer.append(2 * self.num_vars)
        return req_reg_num_per_layer

    def mod_mul_cost(self, num_continuous_output) -> list:
        """
        compute the cost of num mod mults (PEs)

        :param num_continuous_output: num of continuous output
        :return: req mod_mul num per layer
        """
        req_mod_mul_num_per_layer = []
        # consider strategy: Least Common Multiple (LCM)
        k0 = num_continuous_output
        for i in range(self.num_vars - 1):
            if k0 > 1:
                if int(k0) & 1:  # odd
                    k0 = k0 + 1
                    req_mod_mul_num_per_layer.insert(0, int(k0 / 2))
                    k0 = k0 / 2
                else:  # even
                    req_mod_mul_num_per_layer.insert(0, int(k0 / 2))
                    k0 = k0 / 2
            else:
                req_mod_mul_num_per_layer.insert(0, k0 / 2)
                k0 = k0 / 2
        return req_mod_mul_num_per_layer


if __name__ == '__main__':
    build_mle_cost = BuildMleCostReport(num_vars=20, mod_mul_latency=10, mod_add_latency=5, num_zerocheck_pes=1,
                                        num_mod_mul=-1, num_mod_add=-1, available_bandwidth=-1)
    build_mle_result = build_mle_cost.cost()
    print(build_mle_result)
    print()

    # debug report
    print(build_mle_cost.dbg_prt())
