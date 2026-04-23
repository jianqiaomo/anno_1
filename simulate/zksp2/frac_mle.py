import json
import math
from .util import bitsPerCycle_to_GiBPerS


def modelAdd(N: int, depth=3):
    """models fully pipelined modular adder"""
    return (depth, depth + N - 1)


def modelMul(N: int, depth=20):
    """models fully pipelined modular multiplier"""
    return (depth, depth + N - 1)


def modelSeqMul(N=128):
    if N == 1:
        return 1, 1
    e2e_mul_lat = modelMul(1)[0]
    if N == 2:
        return e2e_mul_lat, e2e_mul_lat

    # if N % 2 == 0: 2 outputs ready, if N % 2 == 1: 1 output ready
    first_out_lat = N / 2
    last_out_lat = (N - 2) * e2e_mul_lat

    return first_out_lat, last_out_lat


# N and D gen model
def modelGenND(N=pow(2, 20), num_witness_pairs=3, num_units=1, CLK_FREQ=1e9, bitwidth=255, witness_bitwidth=26.4, onchip_mle_size=512, assume_onchip_storage=True, verbose=False):
    """
    for now assumes unlimited memory bandwidth
    and only model the latency and area cost

    Returns:
        total latency needed to process N elements
    """
    # assert we only have vanilla or jellyfish gates
    assert ((num_witness_pairs == 3) or (num_witness_pairs == 5) or (num_witness_pairs == 2))

    # each "unit" processes input witnesses in parallel
    num_elements_per_unit = math.ceil(N / num_units)

    # each PE processes one witness pair
    # and generates one intermediate output pair
    num_pes_per_unit = num_witness_pairs

    # get base latency numbers for add and mul
    modAdd_e2e_lat, modAdd_last_out_lat = modelAdd(num_elements_per_unit)
    modMul_e2e_lat, modMul_last_out_lat = modelMul(num_elements_per_unit)

    # 1: generate intermediate Ns and Ds
    # 3 for vanilla gate, 5 for jellyfish
    # vanilla - D_0, D_1, D_2 / N_0, N_1, N_2
    # jellyfish - D_0, D_1, D_2, D_3, D_4 / N_0, N_1, N_2, N_3, N_4
    # operations: 1 Mul, 2 Adds
    # we can add w and gamma while we mul beta and sigma

    # the first output latency is the e2e latency of the
    # parallel mul and add + the e2e latency of the second add
    pe_e2e_lat = max(modAdd_e2e_lat, modMul_e2e_lat) + modAdd_e2e_lat
    # the last output latency is the total latency of the
    # parallel mul and add + the e2e latency of the second add
    pe_last_out_lat = max(modAdd_last_out_lat, modMul_last_out_lat) + modAdd_e2e_lat

    # 2: combine intermediate D / N
    # the end to end latency is the number of "stages" * modMul latency
    comb_e2e_lat = math.ceil(math.log2(num_pes_per_unit)) * modMul_e2e_lat

    # 3: combine with addition since its sequential
    e2e_lat = pe_e2e_lat + comb_e2e_lat
    last_out_lat = pe_last_out_lat + comb_e2e_lat

    # througput calc
    # each unit reads 6 elements from off-chip every clock cycle
    # input_bandwidth_GiB_per_s = 6 * bitwidth / 8 * num_units * CLK_FREQ / pow(2, 30)
    # inputs are read from on-chip memory
    if assume_onchip_storage:
        input_bandwidth_GiB_per_s = 0
    else:
        num_vars = math.log2(N)
        permutation_bits_per_witness_entry = (num_vars + 2)
        avg_bits_per_witness_entry = witness_bitwidth
        overhead_bits_per_witness_entry = math.ceil(math.log2(onchip_mle_size))
        bits_per_cycle_per_witness = permutation_bits_per_witness_entry + avg_bits_per_witness_entry + overhead_bits_per_witness_entry
        # print(bits_per_cycle_per_witness, "bits per cycle per witness")
        input_bandwidth_GiB_per_s = bitsPerCycle_to_GiBPerS(num_pes_per_unit * (bits_per_cycle_per_witness) * num_units, CLK_FREQ=CLK_FREQ)
        
        # input_bandwidth_GiB_per_s = bitsPerCycle_to_GiBPerS(num_pes_per_unit * (bitwidth + 2 + math.log2(N)) * num_units, CLK_FREQ=CLK_FREQ)

    # each unit outputs num_pes_per_unit * 2 elements (each pe outputs 1 N and 1 D)
    output_bandwidth_GiB_per_s = bitsPerCycle_to_GiBPerS(
        num_pes_per_unit * 2 * bitwidth * num_units, CLK_FREQ=CLK_FREQ
    )
    total_bandwidth_GiB_per_s = (input_bandwidth_GiB_per_s + output_bandwidth_GiB_per_s)
    # number of registers
    # we need registers to balance the parallel modMul and modAdd pipelines in each PE
    num_regs = abs(modMul_e2e_lat - modAdd_e2e_lat) * 5 * num_units

    # a PE processes one witness pair
    # each PE needs 3 adders
    num_adds = 3 * 5 * num_units

    # each PE needs 2 multiplier
    # each unit also needs additional modMuls to combine PE results
    # vanilla - D_0, D_1, D_2 and N_0, N_1, N_2
    # jellyfish - D_0, D_1, D_2, D_3, D_4 / N_0, N_1, N_2, N_3, N_4
    # combine intermediate results using an unbalanced "tree" with
    num_muls = ((2 * 5) + (5 - 1)) * num_units

    if verbose:
        print("N:", N)
        print("pe latency:", pe_e2e_lat, pe_last_out_lat)
        print("comb latency:", comb_e2e_lat)
        print("total latency:", e2e_lat, last_out_lat, "\n")

    return {
        "N": N,
        "num_units": num_units,
        "e2e_lat": e2e_lat,
        "last_out_lat": last_out_lat,
        "num_muls": num_muls,
        "num_adds": num_adds,
        "num_regs": num_regs,
        "bandwidth_GiB_per_s": total_bandwidth_GiB_per_s
    }


def modelModInvPipeline(
    N=pow(2, 20),
    bitwidth=255
):
    """ effectively extremely long 509 cycle pipeline """
    # the number of "stages" we have is the latency of one modular inverse
    mi_e2e_lat = 2 * bitwidth - 1
    # the total latency is the latency to push all elements through the "pipeline"
    mi_total_lat = mi_e2e_lat + N - 1
    # we need enough units to completely mask the latency of inversion
    num_units = mi_e2e_lat
    out = {
        "N": N,
        "num_units": num_units,
        "e2e_lat": mi_e2e_lat,
        "last_out_lat": mi_total_lat
    }
    return out


def simModInvPipeline(
    N=pow(2, 20),
    bitwidth=255
):
    """ effectively extremely long 510 cycle pipeline """
    # the number of "stages" we have is the latency of one modular inverse
    mi_e2e_lat = 2 * bitwidth - 1
    units = {f"unit{i}": 0 for i in range(mi_e2e_lat)}
    cycles = 0
    n = N
    dispatched = False

    while True:
        dispatched = False
        cycles += 1
        # first decrement
        for unit, cycles_left in units.items():
            if cycles_left > 0:
                units[unit] -= 1
        # start new operation if needed
        if n > 0:
            for unit, cycles_left in units.items():
                if cycles_left == 0 and not dispatched:
                    n -= 1
                    units[unit] = mi_e2e_lat
                    dispatched = True
                if dispatched:
                    break
            # error checking
            if not dispatched:
                raise Exception()
        else:
            cnt = sum(units.values())
            if cnt == 0:
                break

    print(cycles)


def modelModInv(
    N=pow(2, 20),
    bitwidth=255,
    verbose=False,
):
    """model for Modular Inverse"""

    # we assume 1 element per cycle
    fill_lat = 2
    # constant time modinv latency
    mi_e2e_lat = 2 * bitwidth - 1
    mul_e2e_lat, _ = modelMul(N)
    total_e2e_lat = fill_lat + mul_e2e_lat + mi_e2e_lat + mul_e2e_lat
    total_lat = N + mul_e2e_lat + mi_e2e_lat + mul_e2e_lat

    # need enough units to mask the latency of batching + inverting
    num_units = math.ceil((mul_e2e_lat + mi_e2e_lat) / fill_lat) + 1

    # need 1 mul to batch, 1 mul to isolate
    num_muls = 2

    # size of SRAM storage needed
    num_sram_KiB = num_units * 2 * bitwidth / 8 / 1024

    out = {
        "N": N,
        "e2e_lat": total_e2e_lat,
        "last_out_lat": total_lat,
        # number of mod inv units
        "num_units": num_units,
        # mod muls
        "num_muls": num_muls,
        # regs
        "num_regs": 0,
        # SRAM
        "sram_size_KiB": num_sram_KiB
    }

    if verbose:
        print(json.dumps(out, indent=4), "\n")

    return out


def modelFracMLE(
    N=pow(2, 20),
    num_units=1,
    CLK_FREQ=1e9,
    bitwidth=255,
    verbose=False
):
    num_elements_per_unit = math.ceil(N / num_units)
    modInv_model = modelModInv(
        num_elements_per_unit,
        bitwidth=bitwidth,
        verbose=verbose,
    )
    # each fracMLE unit has multiple mod inv units
    num_modinv = modInv_model["num_units"] * num_units
    modMul_e2e_lat, _ = modelMul(N)

    # multiply D^-1 with N to get FracMLE
    fracMLE_e2e_lat = modInv_model["e2e_lat"] + modMul_e2e_lat
    last_out_lat = modInv_model["last_out_lat"] + modMul_e2e_lat

    # hardware resources
    num_muls = (modInv_model["num_muls"] + 1) * num_units

    # number of registers needed
    num_regs = (modInv_model["num_regs"]) * num_units

    # size of SRAM needed
    # sram needed for modInv and to buffer elements of N while we are inverting D
    sram_size_KiB = (
        modInv_model["sram_size_KiB"] + (modInv_model["e2e_lat"] * bitwidth / 8 / 1024)
    ) * num_units

    # off-chip memory bandwidth
    # there is no input bandwidth for fracMLE since
    # we can read N and D directly as they get produced
    # so the only bandwidth needed is to write one element of fracMLE per cycle
    bandwidth_GiB_per_s = bitwidth / 8 * CLK_FREQ / pow(2, 30) * num_units

    out = {
        "u": 20,
        "N": pow(2, 20),
        "modInv_num_units": num_modinv,
        "e2e_lat": fracMLE_e2e_lat,
        "last_out_lat": last_out_lat,
        # multipliers
        "num_muls": num_muls,
        # registers
        "num_regs": num_regs,
        # SRAM
        "sram_size_KiB": sram_size_KiB,
        # bandwidth
        "bandwidth_GiB_per_s": bandwidth_GiB_per_s,
    }

    if verbose:
        print(json.dumps(out, indent=2))
    return out


def main():
    # for i in range(1, 6):
    # print(modelGenND(num_witness_pairs=5))
    # simModInvPipeline()
    # print(modelModInvPipeline())
    # modelModInv()
    out = modelFracMLE(verbose=True)

    # num_muls = 2
    # num_modinv_units = math.ceil((2+10+509)/2)
    # sram_size_KiB = 2 * num_modinv_units * 255 / 8 / 1024
    # print(num_muls, num_modinv_units, sram_size_KiB)
    mod_inv_area = out['modInv_num_units'] * 0.027
    mul_area = out['num_muls'] * 0.478
    sram_area = out['sram_size_KiB'] / 1024 * 1.76  # 1.76mm^2 per MB
    total_area = (mod_inv_area + mul_area + sram_area) / 3.6
    print(total_area, "mm^2")
    print(f"{10.178576 / total_area} x reduction")
    print(0.478 / 0.027)
    # reg_area = num_regs * 0.00058     # 1DFF is 580 um^2
    # stats = []
    # for bs in [pow(2, i) for i in range(15)]:
    #     out = modelFracMLE(
    #         modInv_batch_size=bs, modInv_mulTree_num_inputs=-1, verbose=True
    #     )
    #     if out:
    #         stats.append(out)

    # df = pd.DataFrame(stats)
    # ax = sns.scatterplot(df, x="batch_size", y="avg_lat", palette=sns.color_palette('husl', 12))
    # ax.set_xscale("log", base=2)
    # # ax.set_yscale("log")
    # plt.show()

    # ax = sns.scatterplot(df, x="batch_size", y="modInv_stall_cycles", palette=sns.color_palette('husl', 12))
    # ax.set_xscale("log", base=2)
    # plt.grid(True)
    # plt.show()
    # modelFracMLE(1, -1)


if __name__ == "__main__":
    main()