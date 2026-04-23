import json
import re
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


NOISY_KEYS = {"compute_queue", "load_store_queue", "range_tasks"}
NOISY_KEY_PATTERNS = [
    re.compile(r"^[A-Za-z0-9]+_easy_range_list$"),
    re.compile(r"^[A-Za-z0-9]+_easy_avg_unique_[A-Za-z0-9]+_per_[A-Za-z0-9]+$"),
    re.compile(r"^[A-Za-z0-9]+_hard_window_list$"),
    re.compile(r"^[A-Za-z0-9]+_hard_window_captured_[A-Za-z0-9]+_count$"),
    re.compile(r"^[A-Za-z0-9]+_hard_window_uncaptured_[A-Za-z0-9]+_count$"),
    re.compile(r"^[A-Za-z0-9]+_hard_window_captured_unique_[A-Za-z0-9]+_count$"),
    re.compile(r"^[A-Za-z0-9_]+_info_range_sort$"),
]


def should_drop_key(key):
    if key in NOISY_KEYS:
        return True
    return any(pattern.match(key) for pattern in NOISY_KEY_PATTERNS)


def strip_large_schedule_fields(obj):
    if isinstance(obj, dict):
        cleaned = {}
        for key, value in obj.items():
            if should_drop_key(key):
                continue
            cleaned[key] = strip_large_schedule_fields(value)
        return cleaned
    if isinstance(obj, list):
        return [strip_large_schedule_fields(value) for value in obj]
    return obj


def is_effectively_empty(obj):
    if obj is None:
        return True
    if isinstance(obj, dict):
        return len(obj) == 0 or all(is_effectively_empty(value) for value in obj.values())
    if isinstance(obj, list):
        return len(obj) == 0 or all(is_effectively_empty(value) for value in obj)
    return False


def _process_one_json(json_path_str):
    json_path = Path(json_path_str)
    if not json_path.exists():
        return {"status": "missing", "path": str(json_path)}
    if json_path.stat().st_size == 0:
        json_path.unlink()
        return {"status": "removed_empty", "path": str(json_path)}

    try:
        with json_path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
    except json.JSONDecodeError:
        return {"status": "invalid_json", "path": str(json_path)}

    cleaned_obj = strip_large_schedule_fields(obj)
    if is_effectively_empty(cleaned_obj):
        json_path.unlink()
        return {"status": "removed_empty", "path": str(json_path)}

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(cleaned_obj, f, indent=2)
    return {"status": "rewritten", "path": str(json_path)}


def cleanup_sim_data_jsons(sim_data_root, max_workers=32):
    """
    Cleanup JSON files in the simulation data directory.
    """
    sim_data_root = Path(sim_data_root)
    json_paths = sorted(sim_data_root.rglob("*.json"))

    total_files = len(json_paths)
    removed_empty_files = 0
    rewritten_files = 0
    skipped_invalid_json = 0
    missing_files = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_process_one_json, str(json_path))
            for json_path in json_paths
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Cleaning sim_data jsons"):
            result = future.result()
            if result["status"] == "rewritten":
                rewritten_files += 1
            elif result["status"] == "removed_empty":
                removed_empty_files += 1
            elif result["status"] == "invalid_json":
                skipped_invalid_json += 1
            elif result["status"] == "missing":
                missing_files += 1

    return {
        "sim_data_root": str(sim_data_root),
        "total_files": total_files,
        "rewritten_files": rewritten_files,
        "removed_empty_files": removed_empty_files,
        "skipped_invalid_json": skipped_invalid_json,
        "missing_files": missing_files,
        "max_workers": max_workers,
    }


def _check_one_json(json_path_str):
    """Helper function to check a single JSON file."""
    json_path = Path(json_path_str)
    try:
        with open(json_path, 'r') as f:
            json.load(f)
        return {"status": "valid", "path": str(json_path)}
    except json.JSONDecodeError as e:
        return {"status": "invalid", "path": str(json_path), "error": str(e)}
    except Exception as e:
        return {"status": "error", "path": str(json_path), "error": f"Error reading file: {str(e)}"}


def check_json_files(directories, delete_invalid=False, max_workers=64):
    """Check if all JSON files in given directories are valid.
    
    Args:
        directories: List of directory paths to check
        delete_invalid: If True, delete invalid JSON files (default: False)
        max_workers: Number of worker processes (default: 64)
    """
    invalid_files = []
    deleted_files = 0
    print(f"Checking JSON files in directories: {directories}")
    
    # Collect all JSON files first
    all_json_files = []
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Directory not found: {directory}")
            continue
        all_json_files.extend(sorted(Path(directory).rglob("*.json")))
    
    print(f"Found {len(all_json_files)} JSON files to check")
    
    # Check files with multiprocessing
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_check_one_json, str(json_path))
            for json_path in all_json_files
        ]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Checking JSON files"):
            result = future.result()
            if result["status"] in ("invalid", "error"):
                invalid_files.append({
                    'file': result["path"],
                    'error': result.get("error", "Unknown error")
                })
                
                if delete_invalid:
                    try:
                        Path(result["path"]).unlink()
                        deleted_files += 1
                    except Exception as del_err:
                        print(f"Failed to delete {result['path']}: {del_err}")
    
    # Print results
    if invalid_files:
        print(f"\nFound {len(invalid_files)} invalid JSON file(s):\n")
        for item in invalid_files:
            print(f"❌ {item['file']}")
            print(f"   Error: {item['error']}\n")
        if delete_invalid:
            print(f"🗑️  Deleted {deleted_files} invalid file(s).")
    else:
        print("\n✓ All JSON files are valid!")
    
    return invalid_files


if __name__ == "__main__":
    sim_data_root = Path(__file__).resolve().parent / "sim_data"

    # result = cleanup_sim_data_jsons(sim_data_root)
    # print(json.dumps(result, indent=2))

    dirs = [sim_data_root]
    check_json_files(dirs, delete_invalid=True)

    print("temp.py end.")