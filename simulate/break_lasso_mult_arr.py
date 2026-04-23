from pathlib import Path
from tqdm import tqdm
import argparse

def split_lasso_mult_array_tables(
    filename="output/opt-125m/SqueezeMerge_0/opt-125m_lasso_mult_array.log",
):
    """
    Split each whitespace-delimited table in the source log into per-sumcheck files.

    Supported table headers:
    - sumcheck_id u hu
    - sumcheck_id v hv

    Output files are written to:
    output/opt-125m/SqueezeMerge_0/opt-125m_lasso_mult_array/
    with names like:
    - {sumcheck_id}_u_hu.log
    - {sumcheck_id}_v_hv.log
    """
    source_path = Path(filename)
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    output_dir = source_path.with_suffix("")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output files will be written to {output_dir}/...")

    header_map = {
        ("sumcheck_id", "u", "hu"): ("u", "hu"),
        ("sumcheck_id", "v", "hv"): ("v", "hv"),
    }

    current_table = None
    open_files = {}
    created_files = set()

    try:
        with source_path.open("r", encoding="utf-8", errors="replace") as infile:
            for line in infile:
                parts = line.split()
                if not parts:
                    continue

                header_key = tuple(parts[:3])
                if header_key in header_map:
                    current_table = header_map[header_key]
                    continue

                if current_table is None or len(parts) < 3:
                    continue

                try:
                    sumcheck_id = int(parts[0])
                except ValueError:
                    continue

                left_name, right_name = current_table
                output_name = f"{sumcheck_id}_{left_name}_{right_name}.log"
                output_path = output_dir / output_name

                if output_path not in open_files:
                    outfile = output_path.open("w", encoding="utf-8")
                    print(f"Created file {output_path} for sumcheck_id {sumcheck_id} with columns {left_name}, {right_name}")
                    outfile.write(f"sumcheck_id\t{left_name}\t{right_name}\n")
                    open_files[output_path] = outfile
                    created_files.add(str(output_path))

                open_files[output_path].write(
                    f"{parts[0]}\t{parts[1]}\t{parts[2]}\n"
                )
    finally:
        for outfile in open_files.values():
            outfile.close()

    return sorted(created_files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split lasso_mult_array log into per-sumcheck files."
    )
    parser.add_argument(
        "model_dir",
        nargs="?",
        default="../output/opt-125m/SqueezeMerge_0/opt-125m_lasso_mult_array.log",
        help="Path to the lasso_mult_array log file to parse",
    )
    args = parser.parse_args()

    created_files = split_lasso_mult_array_tables(args.model_dir)
    print(f"Created {len(created_files)} files.")
    print("Done.")
