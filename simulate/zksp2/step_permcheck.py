import json
from .frac_mle import modelGenND, modelFracMLE
from .reverse_binary_tree import ProductMLECostReport
from .util import bitsPerCycle_to_GiBPerS
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt


def step_permcheck_model(
    # gen N/D config
    gen_nd_num_units,
    # FracMLE config
    fracMLE_num_units,
    # ProductMLE config
    productMLE_num_mod_muls=-1,
    productMLE_bandwidth_bpc=-1,
    mod_mul_latency=20,
    # general config
    num_witness_pairs=3,
    num_vars=20,  # 2^20
    CLK_FREQ=1e9,
    bitwidth=255,
    assume_onchip_storage=True,
    witness_bitwidth=26.4,
    onchip_mle_size=512 
):
    """model for overall permCheck step

    Args:
        gen_nd_num_units (int): number of Gen N/D units to use
        fracMLE_num_units (int): number of FracMLE units to use
        fracMLE_num_modInv_units (int): number of modInv units to use per FracMLE unit
        fracMLE_modInv_batch_size (int): batch size to use for modInv (should be power of 2)
        fracMLE_modInv_mulTree_num_inputs (int): number of inputs for mulTree to compute modInv input
        productMLE_num_mod_muls (int, optional): number of multipliers to use for productMLE. Defaults to -1.
        productMLE_bandwidth_bpc (int, optional): bandwidth constraint for productMLE. Defaults to -1.
        mod_mul_latency (int, optional): latency for modMul. Defaults to 10.
        num_vars (int, optional): number of variables. Defaults to 20.
        bitwidth (int, optional): data bitwidth. Defaults to 255.

    Returns:
        Dict:
    {
        "gen_nd_e2e_lat": int,
        "gen_nd_first_out_lat": int,
        "gen_nd_last_out_lat": int,
        "frac_mle_e2e_lat": int,
        "frac_mle_first_out_lat": int,
        "frac_mle_last_out_lat": int,
        "product_mle_e2e_lat": int,
        "product_mle_first_out_lat": int,
        "product_mle_last_out_lat": int,
        "num_mod_muls": int,
        "num_mod_adds": int,
        "num_mod_invs": int,
        "num_regs": int,
        "sram_size_KiB": int,
        "bandwidth_GiB_per_s": int
    }
    """

    # permCheck steps:
    # 1. generate N and D
    # 2. fracMLE
    # 3. productMLE
    # 4. MSM (?)
    # 5. SHA3 challenge
    # 6. zeroCheck

    assert gen_nd_num_units == fracMLE_num_units
    gen_nd_stats = modelGenND(
        N=pow(2, num_vars),
        num_witness_pairs=num_witness_pairs,  # 3 for vanilla, 5 for jelly
        num_units=gen_nd_num_units,
        CLK_FREQ=CLK_FREQ,
        bitwidth=bitwidth,
        witness_bitwidth=witness_bitwidth, 
        onchip_mle_size=onchip_mle_size, 
        assume_onchip_storage=assume_onchip_storage
    )

    fracMLE_stats = modelFracMLE(
        N=pow(2, num_vars),
        num_units=fracMLE_num_units,
        CLK_FREQ=CLK_FREQ,
        bitwidth=bitwidth
    )

    # each fracMLE unit produces 1 output per cycle
    productMLE_num_inputs_per_cycle = fracMLE_num_units
    productMLE_stats = ProductMLECostReport(
        num_vars=num_vars,
        mod_mul_latency=mod_mul_latency,
        num_previous_result=productMLE_num_inputs_per_cycle,
        num_mod_mul=productMLE_num_mod_muls,
        available_bandwidth=productMLE_bandwidth_bpc,
    ).cost()

    # end-to-end latency of gen N/D stage
    gen_nd_e2e_lat = gen_nd_stats["e2e_lat"]
    gen_nd_first_out_lat = gen_nd_e2e_lat
    # latency to process all elements
    gen_nd_last_out_lat = gen_nd_stats["last_out_lat"]

    # end-to-end latency to get FracMLE
    frac_mle_e2e_lat = fracMLE_stats["e2e_lat"]
    frac_mle_first_out_lat = gen_nd_e2e_lat + frac_mle_e2e_lat
    # latency to produce all elements of fracMLE (phi)
    frac_mle_last_out_lat = gen_nd_last_out_lat + fracMLE_stats["e2e_lat"]
    # df = pd.DataFrame({"time": nd_fracMLE_bw_trace.keys(), "BW": nd_fracMLE_bw_trace.values()})
    # sns.lineplot(df, x='time', y='BW')
    # plt.show()

    # end-to-end latency to get productMLE
    # we get the first output of productMLE after gen N/D, fracMLE, and productMLE
    product_mle_e2e_lat = productMLE_stats["compulsory_cycles"]
    product_mle_first_out_lat = frac_mle_first_out_lat + product_mle_e2e_lat
    product_mle_last_out_lat = product_mle_first_out_lat + productMLE_stats['total_cycles'] - productMLE_stats["compulsory_cycles"]

    gen_nd_mod_muls = gen_nd_stats["num_muls"]
    frac_mle_mod_muls = fracMLE_stats["num_muls"]
    product_mle_mod_muls = productMLE_stats["req_mod_mul_num"]
    num_mod_muls = gen_nd_mod_muls + frac_mle_mod_muls + product_mle_mod_muls

    gen_nd_mod_adds = gen_nd_stats["num_adds"]
    frac_mle_mod_adds = fracMLE_stats.get("num_adds", 0)    # zero
    product_mle_mod_adds = productMLE_stats.get("num_adds", 0)  # zero
    num_mod_adds = gen_nd_mod_adds + frac_mle_mod_adds + product_mle_mod_adds

    gen_nd_regs = gen_nd_stats["num_regs"]
    frac_mle_regs = fracMLE_stats["num_regs"]
    product_mle_regs = productMLE_stats["req_mem_reg_num"]
    num_regs = gen_nd_regs + frac_mle_regs + product_mle_regs

    sram_size_KiB = fracMLE_stats["sram_size_KiB"]
    bandwidth_GiB_per_s = (
        gen_nd_stats["bandwidth_GiB_per_s"]
        + fracMLE_stats["bandwidth_GiB_per_s"]
        + bitsPerCycle_to_GiBPerS(productMLE_stats["bandwidth_bperc"], CLK_FREQ)
    )
    out = {
        # latency
        "gen_nd_e2e_lat": gen_nd_e2e_lat,
        "gen_nd_first_out_lat": gen_nd_first_out_lat,
        "gen_nd_last_out_lat": gen_nd_last_out_lat,
        "frac_mle_e2e_lat": frac_mle_e2e_lat,
        "frac_mle_first_out_lat": frac_mle_first_out_lat,
        "frac_mle_last_out_lat": frac_mle_last_out_lat,
        "product_mle_e2e_lat": product_mle_e2e_lat,
        "product_mle_first_out_lat": product_mle_first_out_lat,
        "product_mle_last_out_lat": product_mle_last_out_lat,
        # total resources
        "num_mod_muls": num_mod_muls,
        "num_mod_adds": num_mod_adds,
        "num_mod_invs": fracMLE_stats["modInv_num_units"],
        "num_regs": num_regs,
        "sram_size_KiB": sram_size_KiB,
        # gen ND resources
        "gen_nd_num_mod_muls": gen_nd_mod_muls,
        "gen_nd_num_mod_adds": gen_nd_mod_adds,
        "gen_nd_num_regs": gen_nd_regs,
        # fracMLE resources
        "frac_mle_num_mod_muls": frac_mle_mod_muls,
        "frac_mle_num_regs": frac_mle_regs,
        # prodMLE resources
        "product_mle_num_mod_muls": product_mle_mod_muls,
        "product_mle_num_regs": product_mle_regs,
        # bandwidth
        "bandwidth_GiB_per_s": bandwidth_GiB_per_s,
        "nd_bandwidth_GiB_per_s": gen_nd_stats["bandwidth_GiB_per_s"],
        "fracMLE_bandwidth_GiB_per_s": fracMLE_stats["bandwidth_GiB_per_s"],
        "prodMLE_bandwidth_GiB_per_s": bitsPerCycle_to_GiBPerS(productMLE_stats["bandwidth_bperc"], CLK_FREQ)
    }

    return out


if __name__ == "__main__":
    for num_units in [1, 2, 3, 4]:
        for num_vars in [17]: #, 20, 21, 22, 23]:


            # num_units = 4
            # num_vars = 20
            num_witness_pairs = 3  # 3 for vanilla, 5 for jelly
            assume_onchip_storage = False

            out = dict()
            # add sweep parameters
            out['u'] = num_vars
            out['N'] = pow(2, num_vars)
            out["num_gen_nd_fracMLE_units"] = num_units

            bitwidth = 255
            witness_bitwidth = 0.9 + 0.1 * bitwidth
            onchip_mle_size = 256
            out = out | step_permcheck_model(
                num_units, num_units,  # number of genND and fracMLE units
                num_witness_pairs=num_witness_pairs,
                num_vars=num_vars,
                assume_onchip_storage=assume_onchip_storage,
                witness_bitwidth=witness_bitwidth,
                onchip_mle_size=onchip_mle_size,
                bitwidth=bitwidth
            )


            print(json.dumps(out, indent=2))
