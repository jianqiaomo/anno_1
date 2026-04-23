# zkInf Repository

Warning: before start, we suggest leaving at least `300 GB` memory and `150 GB` disk space available.

## Workflow

1. Run the compiling step for ZKP LLM from zkGPT.

This step mocks the forward pass of the LLM and creates the GKR gates, auxiliary data for range proof, and commitments. It then runs the range proof, GKR gate proof, Thaler13 matmul proof, and verifier.

```bash
cd ./LLM

./llm.sh release O3 gpt2-small --squeeze-merge --debug-print | tee ../output/gpt2-small/SqueezeMerge_1/software_cout.log
```

Model data and its ZKP-related outputs are written under `../output`.

2. Run the zkInf compiler optimization to filter captured data.

```bash
cd ../simulate

python3 compile_gate_range.py --multiprocess --multithread --max-workers [N, depends on your machine]
```

The compiled information will be stored under `./comp_data`.

3. Run the architecture model to estimate the performance of zkInf.

```bash
python3 model_sweep.py --multiprocess --max-workers [N, depends on your machine]
```
The simulation results will be stored under `./sim_data`.
