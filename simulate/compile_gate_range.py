import argparse
import json
from bisect import bisect_left
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm
import itertools


def count_unique_u_v_in_lasso_logs(model, squeeze, output_root="output", dump_json_dir=None):
    """
    Read all log files under:
        output/{model}/{squeeze}/{model}_lasso_mult_array/

    Each log is expected to be tab-separated with a header like:
        sumcheck_id    u    hu
    or:
        sumcheck_id    v    hv

    Returns the total unique count observed across all `u`/`v` logs and dumps
    the summary JSON into the same log directory.
    """
    log_dir = Path(output_root) / model / squeeze / f"{model}_lasso_mult_array"
    if not log_dir.exists():
        raise FileNotFoundError(f"Lasso log directory not found: {log_dir}")

    unique_x = set()
    log_files = sorted(log_dir.glob("*.log"))

    for log_file in log_files:
        print(f"Processing log file: {log_file}")
        with log_file.open("r", encoding="utf-8") as f:
            header = f.readline().strip().split("\t")
            if len(header) < 2:
                continue
            x_name = header[1]
            if x_name not in {"u", "v"}:
                continue

            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                try:
                    x_value = int(parts[1])
                except ValueError:
                    continue
                unique_x.add(x_value)

    result = {
        "log_dir": str(log_dir),
        "num_log_files": len(log_files),
        "total_unique_count": len(unique_x),
        # "unique_values": sorted(unique_x),
    }
    if dump_json_dir:
        dump_path = Path(dump_json_dir) / f"lasso_unique_u_v_count.json"
    else:
        dump_path = log_dir / model / squeeze / f"lasso_unique_u_v_count.json"
    with dump_path.open("w", encoding="utf-8") as f:
        print(f"Dumping unique u/v count result to: {dump_path}")
        json.dump(result, f, indent=2)
    result["dump_path"] = str(dump_path)
    return result


def _compress_ranges(values):
    if not values:
        return []
    values = sorted(values)
    ranges = []
    start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append((start, prev + 1))
        start = prev = value
    ranges.append((start, prev + 1))
    return ranges


def _merge_half_open_ranges(ranges):
    if not ranges:
        return []
    ranges = sorted(ranges)
    merged = [list(ranges[0])]
    for start, end in ranges[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1][1] = max(last_end, end)
        else:
            merged.append([start, end])
    return [tuple(x) for x in merged]


def _absorb_small_easy_ranges(u_easy, u_hard, min_size=20, desc="Absorbing small easy ranges"):
    tagged = [(start, end, "easy") for start, end in u_easy]
    tagged.extend((start, end, "hard") for start, end in u_hard)
    tagged.sort()

    def append_tagged(output, item):
        start, end, kind = item
        if not output:
            output.append([start, end, kind])
            return

        last_start, last_end, last_kind = output[-1]
        if kind == last_kind and start <= last_end:
            output[-1][1] = max(last_end, end)
        else:
            output.append([start, end, kind])

    merged_tagged = []
    skip_until = -1
    for i in tqdm(range(len(tagged)), desc=desc):
        if i <= skip_until:
            continue

        start, end, kind = tagged[i]
        next_item = tagged[i + 1] if i + 1 < len(tagged) else None

        if kind == "easy" and end - start < min_size:
            prev_ok = (
                bool(merged_tagged)
                and merged_tagged[-1][2] == "hard"
                and merged_tagged[-1][1] == start
            )
            next_ok = (
                next_item is not None
                and next_item[2] == "hard"
                and next_item[0] == end
            )

            if prev_ok and next_ok:
                merged_tagged[-1][1] = next_item[1]
                skip_until = i + 1
                continue
            if prev_ok:
                merged_tagged[-1][1] = end
                continue
            if next_ok:
                append_tagged(merged_tagged, (start, next_item[1], "hard"))
                skip_until = i + 1
                continue

        append_tagged(merged_tagged, (start, end, kind))

    easy = [(start, end) for start, end, kind in merged_tagged if kind == "easy"]
    hard = [(start, end) for start, end, kind in merged_tagged if kind == "hard"]
    return _merge_half_open_ranges(easy), _merge_half_open_ranges(hard)


def _find_easy_ranges(count_one_pairs):
    if not count_one_pairs:
        return []
    count_one_pairs = sorted(count_one_pairs)
    ranges = []
    start_u, start_g = count_one_pairs[0]
    prev_u, prev_g = count_one_pairs[0]
    delta = start_g - start_u

    for u, g in count_one_pairs[1:]:
        if u == prev_u + 1 and g == prev_g + 1 and (g - u) == delta:
            prev_u, prev_g = u, g
            continue
        if prev_u > start_u:
            ranges.append((start_u, prev_u + 1))
        start_u, start_g = u, g
        prev_u, prev_g = u, g
        delta = start_g - start_u

    if prev_u > start_u:
        ranges.append((start_u, prev_u + 1))
    return ranges


def _find_easy_ranges_count_two(count_two_pairs):
    if not count_two_pairs:
        return []
    normalized = sorted((u, tuple(sorted(gs))) for u, gs in count_two_pairs)
    ranges = []
    start_u, start_gs = normalized[0]
    prev_u, prev_gs = normalized[0]
    deltas = (start_gs[0] - start_u, start_gs[1] - start_u)

    for u, gs in normalized[1:]:
        next_deltas = (gs[0] - u, gs[1] - u)
        if (
            u == prev_u + 1
            and gs[0] == prev_gs[0] + 1
            and gs[1] == prev_gs[1] + 1
            and next_deltas == deltas
        ):
            prev_u, prev_gs = u, gs
            continue
        if prev_u > start_u:
            ranges.append((start_u, prev_u + 1))
        start_u, start_gs = u, gs
        prev_u, prev_gs = u, gs
        deltas = (start_gs[0] - start_u, start_gs[1] - start_u)

    if prev_u > start_u:
        ranges.append((start_u, prev_u + 1))
    return ranges


def _empty_relation_result(x_name, y_name):
    return {
        f"{x_name}_to_{y_name}": [],
        f"{x_name}_count": [],
        f"{x_name}_skip": [],
        f"{x_name}_easy": [],
        f"{x_name}_hard": [],
        f"{x_name}_hard_window": [],
        f"{x_name}_hard_window_captured_{y_name}_count": [],
        f"{x_name}_hard_window_uncaptured_{y_name}_count": [],
        f"{x_name}_hard_window_min_{y_name}": [],
        f"{x_name}_hard_window_max_{y_name}": [],
        f"{x_name}_hard_window_captured_unique_{y_name}_count": [],
        f"{x_name}_hard_window_uncaptured_unique_{y_name}_count": [],
        f"{x_name}_hard_window_uncaptured_unique_{y_name}_set": [],
        f"total_{x_name}_easy_{y_name}_count": 0,
        f"total_{x_name}_hard_captured_{y_name}_count": 0,
        f"total_{x_name}_hard_uncaptured_{y_name}_count": 0,
        f"total_{x_name}_hard_uncaptured_unique_{y_name}_count": 0,
        f"total_{y_name}_count": 0,
        f"total_min_{y_name}": None,
        f"total_max_{y_name}": None,
    }


def _split_ranges_to_windows(ranges, window_size):
    if window_size is None or window_size <= 0:
        return list(ranges)
    windows = []
    for start, end in ranges:
        cur = start
        while cur < end:
            nxt = min(cur + window_size, end)
            windows.append((cur, nxt))
            cur = nxt
    return windows


def _suggest_hard_window_size_for_range(x_to_y, x_range, top_n):
    start, end = x_range
    if start >= end:
        return None

    y_min = None
    y_max = None
    for x in range(start, end):
        ys = x_to_y[x]
        if not ys:
            continue
        cur_min = ys[0]
        cur_max = ys[-1]
        y_min = cur_min if y_min is None else min(y_min, cur_min)
        y_max = cur_max if y_max is None else max(y_max, cur_max)

    if y_min is None or y_max is None:
        return None

    if top_n is None:
        top_y = y_max
    else:
        top_y = min(y_min + top_n, y_max)

    highest_valid_x = None
    for x in range(start, end):
        ys = x_to_y[x]
        if not ys:
            highest_valid_x = x
            continue

        if ys[0] < y_min or ys[-1] > top_y:
            break
        highest_valid_x = x

    if highest_valid_x is None:
        return None

    # Window sizes are used as half-open lengths, so include the last valid x.
    return highest_valid_x - start + 1


def _suggest_hard_window_size(x_to_y, x_hard, top_n, max_ranges=3):
    suggestions = []
    for x_range in x_hard[:max_ranges]:
        suggestion = _suggest_hard_window_size_for_range(x_to_y, x_range, top_n)
        if suggestion is not None:
            suggestions.append(suggestion)

    if not suggestions:
        return None
    return max(suggestions)


def _resolve_hard_window_size(user_hard_window_size, suggested_hard_window_size):
    candidates = [
        size for size in (user_hard_window_size, suggested_hard_window_size)
        if size is not None and size > 0
    ]
    if not candidates:
        return None
    return max(candidates)


def _top_n_freq_for_windows(x_to_y, windows, top_n, desc):
    """
    For each half-open range (start, end) in windows, compute the frequency of y values for x in [start, end),
    and return the top_n most common y values sorted by y. If top_n is 0, return an empty list.

    Return a list of length len(windows), where each element is a sorted list of y values.
    """
    if top_n is None or top_n <= 0:
        top_n = 0

    top_n = int(top_n)
    freq_rows = []
    for start, end in tqdm(windows, desc=desc):
        counter = Counter()
        for x in range(start, end):
            counter.update(x_to_y[x])
        most_common = counter.most_common(top_n) if top_n else []
        freq_rows.append(sorted(y for y, _ in most_common))
    return freq_rows


def _contains_sorted_value(sorted_values, target):
    idx = bisect_left(sorted_values, target)
    return idx < len(sorted_values) and sorted_values[idx] == target


def _count_uncaptured_y_for_window(x_to_y, window, captured_y_values):
    start, end = window
    captured_count = 0
    uncaptured_count = 0
    min_y = None
    max_y = None
    captured_unique_y = set()
    uncaptured_unique_y = set()
    for x in range(start, end):
        for y in x_to_y[x]:
            if min_y is None or y < min_y:
                min_y = y

            if max_y is None or y > max_y:
                max_y = y

            if _contains_sorted_value(captured_y_values, y):
                captured_count += 1
                captured_unique_y.add(y)
            else:
                uncaptured_count += 1
                uncaptured_unique_y.add(y)
    return (
        captured_count,
        uncaptured_count,
        min_y,
        max_y,
        len(captured_unique_y),
        len(uncaptured_unique_y),
        uncaptured_unique_y,
    )


def _count_uncaptured_y_for_windows(
    x_to_y,
    windows,
    captured_y_rows,
    desc,
    use_multithread=True,
    max_workers=32,
):
    if len(windows) != len(captured_y_rows):
        raise ValueError("windows and captured_y_rows must have the same length")

    if not windows:
        return [], [], [], [], [], [], []

    captured_counts = [0] * len(windows)
    uncaptured_counts = [0] * len(windows)
    min_y_values = [None] * len(windows)
    max_y_values = [None] * len(windows)
    captured_unique_counts = [0] * len(windows)
    uncaptured_unique_counts = [0] * len(windows)
    uncaptured_unique_sets = [set() for _ in range(len(windows))]
    if not use_multithread or len(windows) <= 1:
        for idx, (window, captured_y_values) in enumerate(
            tqdm(zip(windows, captured_y_rows), total=len(windows), desc=desc)
        ):
            (
                captured_counts[idx],
                uncaptured_counts[idx],
                min_y_values[idx],
                max_y_values[idx],
                captured_unique_counts[idx],
                uncaptured_unique_counts[idx],
                uncaptured_unique_sets[idx],
            ) = _count_uncaptured_y_for_window(
                x_to_y,
                window,
                captured_y_values,
            )
        return (
            captured_counts,
            uncaptured_counts,
            min_y_values,
            max_y_values,
            captured_unique_counts,
            uncaptured_unique_counts,
            uncaptured_unique_sets,
        )

    worker_count = max_workers if max_workers is not None else len(windows)
    worker_count = max(1, min(worker_count, len(windows)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_idx = {
            executor.submit(
                _count_uncaptured_y_for_window,
                x_to_y,
                window,
                captured_y_values,
            ): idx
            for idx, (window, captured_y_values) in enumerate(
                zip(windows, captured_y_rows)
            )
        }
        with tqdm(total=len(windows), desc=desc) as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                (
                    captured_counts[idx],
                    uncaptured_counts[idx],
                    min_y_values[idx],
                    max_y_values[idx],
                    captured_unique_counts[idx],
                    uncaptured_unique_counts[idx],
                    uncaptured_unique_sets[idx],
                ) = future.result()
                pbar.update(1)
    return (
        captured_counts,
        uncaptured_counts,
        min_y_values,
        max_y_values,
        captured_unique_counts,
        uncaptured_unique_counts,
        uncaptured_unique_sets,
    )


def _count_y_points_in_ranges(x_point_count, ranges):
    total_count = 0
    for start, end in tqdm(ranges, desc="Counting y points in ranges"):
        for x in range(start, end):
            total_count += x_point_count[x]
    return total_count


def _count_y_points_and_unique_y_in_ranges(x_to_y, x_point_count, ranges, desc):
    total_count = 0
    range_total_counts = []
    range_unique_counts = []
    range_avg_unique_y_per_x = []

    for start, end in tqdm(ranges, desc=desc):
        range_total_count = 0
        range_unique_y = set()
        range_unique_dots = set()
        for x in range(start, end):
            range_total_count += x_point_count[x]
            range_unique_y.update(x_to_y[x])
            range_unique_dots.update((x, y) for y in x_to_y[x])
        total_count += range_total_count
        range_total_counts.append(range_total_count)
        range_unique_counts.append(len(range_unique_y))
        range_len = end - start
        range_avg_unique_y_per_x.append(
            (len(range_unique_dots) / range_len) if range_len > 0 else 0.0
        )

    return total_count, range_total_counts, range_unique_counts, range_avg_unique_y_per_x


def _shift_ranges(ranges, offset):
    if offset == 0:
        return list(ranges)
    return [(start + offset, end + offset) for start, end in ranges]


def _summarize_index_to_targets(
    rows,
    max_x,
    x_name,
    y_name,
    desc_prefix,
    hard_window_size=None,
    top_n=10,
):
    if max_x < 0:
        return _empty_relation_result(x_name, y_name)

    x_to_y = [[] for _ in range(max_x + 1)]
    x_point_count = [0] * (max_x + 1)
    x_count = [0] * (max_x + 1)
    total_min_y = None
    total_max_y = None

    for x, ys in tqdm(
        rows.items(),
        total=len(rows),
        desc=f"Building {desc_prefix} {x_name}->{y_name}",
    ):
        ys_sorted = sorted(ys)
        x_to_y[x] = ys_sorted
        point_count = len(ys_sorted)
        x_point_count[x] = point_count
        x_count[x] = point_count
        if ys_sorted:
            total_min_y = ys_sorted[0] if total_min_y is None else min(total_min_y, ys_sorted[0])
            total_max_y = ys_sorted[-1] if total_max_y is None else max(total_max_y, ys_sorted[-1])

    skip_values = []
    count_one_pairs = []
    count_two_pairs = []
    hard_values = []
    for x, count in tqdm(
        enumerate(x_count),
        total=len(x_count),
        desc=f"Scanning {desc_prefix} {x_name} counts",
    ):
        if count == 0:
            skip_values.append(x)
        if count == 1:
            count_one_pairs.append((x, x_to_y[x][0]))
        if count == 2:
            count_two_pairs.append((x, tuple(x_to_y[x])))
        if count >= 3:
            hard_values.append(x)

    x_skip = _compress_ranges(skip_values)
    x_easy = _find_easy_ranges(count_one_pairs)
    x_easy.extend(_find_easy_ranges_count_two(count_two_pairs))
    x_easy = sorted(x_easy)

    easy_x_values = set()
    for start, end in tqdm(x_easy, desc=f"Expanding {desc_prefix} {x_name} easy"):
        easy_x_values.update(range(start, end))

    for x, _ in tqdm(
        count_one_pairs,
        desc=f"Classifying {desc_prefix} {x_name} count=1 hard",
    ):
        if x not in easy_x_values:
            hard_values.append(x)
    for x, _ in tqdm(
        count_two_pairs,
        desc=f"Classifying {desc_prefix} {x_name} count=2 hard",
    ):
        if x not in easy_x_values:
            hard_values.append(x)

    x_skip = _merge_half_open_ranges(x_skip)
    x_easy = _merge_half_open_ranges(x_easy)
    x_hard = _merge_half_open_ranges(_compress_ranges(sorted(hard_values)))
    x_easy, x_hard = _absorb_small_easy_ranges(
        x_easy,
        x_hard,
        min_size=200,
        desc=f"Absorbing {desc_prefix} small easy",
    )
    suggested_hard_window_size = _suggest_hard_window_size(x_to_y, x_hard, top_n)
    real_hard_window_size = _resolve_hard_window_size(
        hard_window_size,
        suggested_hard_window_size,
    )
    x_hard_window = _split_ranges_to_windows(x_hard, real_hard_window_size)
    x_hard_window_freq_y = _top_n_freq_for_windows(
        x_to_y,
        x_hard_window,
        top_n,
        desc=f"Counting {desc_prefix} hard-window freq {y_name}",
    )
    (
        x_hard_window_captured_y_count,
        x_hard_window_uncaptured_y_count,
        x_hard_window_min_y,
        x_hard_window_max_y,
        x_hard_window_captured_unique_y_count,
        x_hard_window_uncaptured_unique_y_count,
        x_hard_window_uncaptured_unique_y_set,
    ) = _count_uncaptured_y_for_windows(
        x_to_y,
        x_hard_window,
        x_hard_window_freq_y,
        desc=f"Counting {desc_prefix} hard-window uncaptured {y_name}",
    )
    total_hard_uncaptured_unique_y_count = len(
        set().union(*x_hard_window_uncaptured_unique_y_set)
    ) if x_hard_window_uncaptured_unique_y_set else 0
    total_hard_captured_y_count = sum(x_hard_window_captured_y_count)
    total_hard_uncaptured_y_count = sum(x_hard_window_uncaptured_y_count)
    (
        total_easy_y_count,
        x_easy_total_y_count,
        x_easy_unique_y_count,
        x_easy_avg_unique_y_per_x,
    ) = _count_y_points_and_unique_y_in_ranges(
        x_to_y,
        x_point_count,
        x_easy,
        desc=f"Counting {desc_prefix} easy {y_name}",
    )
    total_y_count = sum(x_point_count)
    total_count_check_passed = (
        total_hard_captured_y_count
        + total_hard_uncaptured_y_count
        + total_easy_y_count
        == total_y_count
    )

    if not total_count_check_passed:
        raise ValueError(
            f"{desc_prefix} total count mismatch: "
            f"hard captured={total_hard_captured_y_count}, "
            f"hard uncaptured={total_hard_uncaptured_y_count}, "
            f"easy={total_easy_y_count}, total={total_y_count}"
        )

    covered = [False] * len(x_count)
    for category_name, category_ranges in (
        (f"{x_name}_skip", x_skip),
        (f"{x_name}_easy", x_easy),
        (f"{x_name}_hard", x_hard),
    ):
        for start, end in category_ranges:
            if start < 0 or end > len(x_count) or start >= end:
                raise ValueError(
                    f"Invalid range in {category_name} for {desc_prefix}: ({start}, {end})"
                )
            for x in range(start, end):
                covered[x] = True

    no_category_values = [x for x, is_covered in enumerate(covered) if not is_covered]
    if no_category_values:
        no_category_ranges = _compress_ranges(no_category_values)
        raise ValueError(
            f"{desc_prefix} has no-category ranges: {no_category_ranges}"
        )

    return {
        f"{x_name}_to_{y_name}": x_to_y,
        f"{x_name}_count": x_count,
        f"{x_name}_skip": x_skip,
        f"{x_name}_easy": x_easy,
        f"{x_name}_easy_total_{y_name}_count": x_easy_total_y_count,
        f"{x_name}_easy_unique_{y_name}_count": x_easy_unique_y_count,
        f"{x_name}_easy_avg_unique_{y_name}_per_{x_name}": x_easy_avg_unique_y_per_x,
        f"{x_name}_hard": x_hard,
        f"{x_name}_hard_window": x_hard_window,
        f"{x_name}_hard_window_captured_{y_name}_count": x_hard_window_captured_y_count,
        f"{x_name}_hard_window_uncaptured_{y_name}_count": x_hard_window_uncaptured_y_count,
        f"{x_name}_hard_window_min_{y_name}": x_hard_window_min_y,
        f"{x_name}_hard_window_max_{y_name}": x_hard_window_max_y,
        f"{x_name}_hard_window_captured_unique_{y_name}_count": x_hard_window_captured_unique_y_count,
        f"{x_name}_hard_window_uncaptured_unique_{y_name}_count": x_hard_window_uncaptured_unique_y_count,
        f"{x_name}_hard_window_uncaptured_unique_{y_name}_set": x_hard_window_uncaptured_unique_y_set,
        f"total_{x_name}_easy_{y_name}_count": total_easy_y_count,
        f"total_{x_name}_hard_captured_{y_name}_count": total_hard_captured_y_count,
        f"total_{x_name}_hard_uncaptured_{y_name}_count": total_hard_uncaptured_y_count,
        f"total_{x_name}_hard_uncaptured_unique_{y_name}_count": total_hard_uncaptured_unique_y_count,
        f"total_{y_name}_count": total_y_count,
        f"total_min_{y_name}": total_min_y,
        f"total_max_{y_name}": total_max_y,
        f"{x_name}_hard_window_suggested_size": suggested_hard_window_size,
        f"{x_name}_hard_window_final_size": real_hard_window_size,
    }


def _summarize_lasso_index_to_targets(
    rows,
    min_x,
    max_x,
    x_name,
    y_name,
    desc_prefix,
    hard_window_size=None,
    top_n=10,
):
    if max_x < 0 or not rows:
        return _empty_relation_result(x_name, y_name)

    if min_x < 0 or min_x > max_x:
        raise ValueError(
            f"Invalid {x_name} offset for {desc_prefix}: min_x={min_x}, max_x={max_x}"
        )

    offset = min_x
    compact_len = max_x - offset + 1
    x_to_y = [[] for _ in range(compact_len)]
    x_point_count = [0] * compact_len
    x_count = [0] * compact_len
    total_min_y = None
    total_max_y = None

    for x, ys in tqdm(
        rows.items(),
        total=len(rows),
        desc=f"Building {desc_prefix} {x_name}->{y_name}",
    ):
        rel_x = x - offset
        if rel_x < 0 or rel_x >= compact_len:
            raise ValueError(
                f"{desc_prefix} has out-of-range {x_name}={x} for offset={offset}, max_x={max_x}"
            )
        ys_sorted = sorted(ys)
        x_to_y[rel_x] = ys_sorted
        point_count = len(ys_sorted)
        x_point_count[rel_x] = point_count
        x_count[rel_x] = point_count
        if ys_sorted:
            total_min_y = ys_sorted[0] if total_min_y is None else min(total_min_y, ys_sorted[0])
            total_max_y = ys_sorted[-1] if total_max_y is None else max(total_max_y, ys_sorted[-1])

    skip_values = []
    count_one_pairs = []
    count_two_pairs = []
    hard_values = []
    for rel_x, count in tqdm(
        enumerate(x_count),
        total=len(x_count),
        desc=f"Scanning {desc_prefix} {x_name} counts",
    ):
        if count == 0:
            skip_values.append(rel_x)
        if count == 1:
            count_one_pairs.append((rel_x, x_to_y[rel_x][0]))
        if count == 2:
            count_two_pairs.append((rel_x, tuple(x_to_y[rel_x])))
        if count >= 3:
            hard_values.append(rel_x)

    x_skip = _compress_ranges(skip_values)
    x_easy = _find_easy_ranges(count_one_pairs)
    x_easy.extend(_find_easy_ranges_count_two(count_two_pairs))
    x_easy = sorted(x_easy)

    easy_x_values = set()
    for start, end in tqdm(x_easy, desc=f"Expanding {desc_prefix} {x_name} easy"):
        easy_x_values.update(range(start, end))

    for x, _ in tqdm(
        count_one_pairs,
        desc=f"Classifying {desc_prefix} {x_name} count=1 hard",
    ):
        if x not in easy_x_values:
            hard_values.append(x)
    for x, _ in tqdm(
        count_two_pairs,
        desc=f"Classifying {desc_prefix} {x_name} count=2 hard",
    ):
        if x not in easy_x_values:
            hard_values.append(x)

    x_skip = _merge_half_open_ranges(x_skip)
    x_easy = _merge_half_open_ranges(x_easy)
    x_hard = _merge_half_open_ranges(_compress_ranges(sorted(hard_values)))
    x_easy, x_hard = _absorb_small_easy_ranges(
        x_easy,
        x_hard,
        min_size=200,
        desc=f"Absorbing {desc_prefix} small easy",
    )
    suggested_hard_window_size = _suggest_hard_window_size(x_to_y, x_hard, top_n)
    real_hard_window_size = _resolve_hard_window_size(
        hard_window_size,
        suggested_hard_window_size,
    )
    x_hard_window = _split_ranges_to_windows(x_hard, real_hard_window_size)
    x_hard_window_freq_y = _top_n_freq_for_windows(
        x_to_y,
        x_hard_window,
        top_n,
        desc=f"Counting {desc_prefix} hard-window freq {y_name}",
    )
    (
        x_hard_window_captured_y_count,
        x_hard_window_uncaptured_y_count,
        x_hard_window_min_y,
        x_hard_window_max_y,
        x_hard_window_captured_unique_y_count,
        x_hard_window_uncaptured_unique_y_count,
        x_hard_window_uncaptured_unique_y_set,
    ) = _count_uncaptured_y_for_windows(
        x_to_y,
        x_hard_window,
        x_hard_window_freq_y,
        desc=f"Counting {desc_prefix} hard-window uncaptured {y_name}",
    )
    total_hard_uncaptured_unique_y_count = len(
        set().union(*x_hard_window_uncaptured_unique_y_set)
    ) if x_hard_window_uncaptured_unique_y_set else 0
    total_hard_captured_y_count = sum(x_hard_window_captured_y_count)
    total_hard_uncaptured_y_count = sum(x_hard_window_uncaptured_y_count)
    (
        total_easy_y_count,
        x_easy_total_y_count,
        x_easy_unique_y_count,
        x_easy_avg_unique_y_per_x,
    ) = _count_y_points_and_unique_y_in_ranges(
        x_to_y,
        x_point_count,
        x_easy,
        desc=f"Counting {desc_prefix} easy {y_name}",
    )
    total_y_count = sum(x_point_count)
    total_count_check_passed = (
        total_hard_captured_y_count
        + total_hard_uncaptured_y_count
        + total_easy_y_count
        == total_y_count
    )

    if not total_count_check_passed:
        raise ValueError(
            f"{desc_prefix} total count mismatch: "
            f"hard captured={total_hard_captured_y_count}, "
            f"hard uncaptured={total_hard_uncaptured_y_count}, "
            f"easy={total_easy_y_count}, total={total_y_count}"
        )

    covered = [False] * len(x_count)
    for category_name, category_ranges in (
        (f"{x_name}_skip", x_skip),
        (f"{x_name}_easy", x_easy),
        (f"{x_name}_hard", x_hard),
    ):
        for start, end in category_ranges:
            if start < 0 or end > len(x_count) or start >= end:
                raise ValueError(
                    f"Invalid range in {category_name} for {desc_prefix}: ({start}, {end})"
                )
            for x in range(start, end):
                covered[x] = True

    no_category_values = [x for x, is_covered in enumerate(covered) if not is_covered]
    if no_category_values:
        no_category_ranges = _compress_ranges(no_category_values)
        raise ValueError(
            f"{desc_prefix} has no-category ranges: {no_category_ranges}"
        )

    absolute_skip = _shift_ranges(x_skip, offset)
    if offset > 0:
        absolute_skip = _merge_half_open_ranges([(0, offset)] + absolute_skip)
    absolute_easy = _shift_ranges(x_easy, offset)
    absolute_hard = _shift_ranges(x_hard, offset)
    absolute_hard_window = _shift_ranges(x_hard_window, offset)

    return {
        f"{x_name}_offset": offset,
        f"{x_name}_to_{y_name}": x_to_y,
        f"{x_name}_count": x_count,
        f"{x_name}_skip": absolute_skip,
        f"{x_name}_easy": absolute_easy,
        f"{x_name}_easy_total_{y_name}_count": x_easy_total_y_count,
        f"{x_name}_easy_unique_{y_name}_count": x_easy_unique_y_count,
        f"{x_name}_easy_avg_unique_{y_name}_per_{x_name}": x_easy_avg_unique_y_per_x,
        f"{x_name}_hard": absolute_hard,
        f"{x_name}_hard_window": absolute_hard_window,
        f"{x_name}_hard_window_captured_{y_name}_count": x_hard_window_captured_y_count,
        f"{x_name}_hard_window_uncaptured_{y_name}_count": x_hard_window_uncaptured_y_count,
        f"{x_name}_hard_window_min_{y_name}": x_hard_window_min_y,
        f"{x_name}_hard_window_max_{y_name}": x_hard_window_max_y,
        f"{x_name}_hard_window_captured_unique_{y_name}_count": x_hard_window_captured_unique_y_count,
        f"{x_name}_hard_window_uncaptured_unique_{y_name}_count": x_hard_window_uncaptured_unique_y_count,
        f"{x_name}_hard_window_uncaptured_unique_{y_name}_set": x_hard_window_uncaptured_unique_y_set,
        f"total_{x_name}_easy_{y_name}_count": total_easy_y_count,
        f"total_{x_name}_hard_captured_{y_name}_count": total_hard_captured_y_count,
        f"total_{x_name}_hard_uncaptured_{y_name}_count": total_hard_uncaptured_y_count,
        f"total_{x_name}_hard_uncaptured_unique_{y_name}_count": total_hard_uncaptured_unique_y_count,
        f"total_{y_name}_count": total_y_count,
        f"total_min_{y_name}": total_min_y,
        f"total_max_{y_name}": total_max_y,
        f"{x_name}_hard_window_suggested_size": suggested_hard_window_size,
        f"{x_name}_hard_window_final_size": real_hard_window_size,
    }


def _load_phase1_u_g_relationship_data(filepath):
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(filepath)

    rows = {0: {}, 1: {}}
    max_u = {0: -1, 1: -1}

    with filepath.open("r", encoding="utf-8", errors="replace") as f:
        total = filepath.stat().st_size
        header_seen = False
        with tqdm(total=total, unit='B', unit_scale=True, desc=f"Reading {filepath.name}") as pbar:
            for line in f:
                pbar.update(len(line.encode('utf-8')))
                parts = line.split()
                if not parts:
                    continue
                if parts[0] == "sumcheck_id":
                    header_seen = True
                    continue
                if not header_seen or len(parts) < 4:
                    continue
                try:
                    lu = int(parts[1])
                    gate_u = int(parts[2])
                    gate_g = int(parts[3])
                except ValueError:
                    continue
                if lu not in rows:
                    continue
                rows[lu].setdefault(gate_u, []).append(gate_g)
                if gate_u > max_u[lu]:
                    max_u[lu] = gate_u

    return {
        "rows": rows,
        "max_u": max_u,
    }


def summarize_phase1_u_g_relationship_from_data(data, hard_window_size=None, top_n=10):
    """
    Summarize the phase-1 gate.u -> gate.g relation for lu=0 and lu=1.

    Return a dict with the same top-level structure style as phase 2.
    """
    return {
        "unary_u_g": {
            lu: _summarize_index_to_targets(
                data["rows"][lu],
                data["max_u"][lu],
                "u",
                "g",
                desc_prefix=f"lu={lu}",
                hard_window_size=hard_window_size,
                top_n=top_n,
            )
            for lu in (0, 1)
        }
    }


def summarize_phase1_u_g_relationship(filepath, hard_window_size=None, top_n=10):
    data = _load_phase1_u_g_relationship_data(filepath)
    return summarize_phase1_u_g_relationship_from_data(
        data,
        hard_window_size=hard_window_size,
        top_n=top_n,
    )


def _load_phase2_relationship_data(filepath):
    filepath = Path(filepath)
    if not filepath.exists():
        # raise FileNotFoundError(filepath)
        print(f"Warning: {filepath} does not exist, returning empty relationship data")
        return {
            "unary_u_g": {0: {}, 1: {}},
            "unary_max_u": {0: -1, 1: -1},
            "binary_v_g": {0: {}, 1: {}},
            "binary_v_u": {0: {}, 1: {}},
            "binary_max_v": {0: -1, 1: -1},
        }

    unary_u_g = {0: {}, 1: {}}
    unary_max_u = {0: -1, 1: -1}
    binary_v_g = {0: {}, 1: {}}
    binary_v_u = {0: {}, 1: {}}
    binary_max_v = {0: -1, 1: -1}

    with filepath.open("r", encoding="utf-8", errors="replace") as f:
        total = filepath.stat().st_size
        header_seen = False
        with tqdm(total=total, unit='B', unit_scale=True, desc=f"Reading {filepath.name}") as pbar:
            for line in f:
                pbar.update(len(line.encode('utf-8')))
                parts = line.split()
                if not parts:
                    continue
                if parts[0] == "sumcheck_id":
                    header_seen = True
                    continue
                if not header_seen or len(parts) < 7:
                    continue
                try:
                    lv = int(parts[1])
                    gate_u = int(parts[2])
                    gate_g = int(parts[3])
                    gate_v = int(parts[7])
                except ValueError:
                    continue
                if lv not in (0, 1):
                    continue

                if gate_v == -1:
                    unary_u_g[lv].setdefault(gate_u, []).append(gate_g)
                    if gate_u > unary_max_u[lv]:
                        unary_max_u[lv] = gate_u
                else:
                    binary_v_g[lv].setdefault(gate_v, []).append(gate_g)
                    binary_v_u[lv].setdefault(gate_v, []).append(gate_u)
                    if gate_v > binary_max_v[lv]:
                        binary_max_v[lv] = gate_v

    return {
        "unary_u_g": unary_u_g,
        "unary_max_u": unary_max_u,
        "binary_v_g": binary_v_g,
        "binary_v_u": binary_v_u,
        "binary_max_v": binary_max_v,
    }


def summarize_phase2_relationship_from_data(data, hard_window_size=None, top_n=10):
    """
    Summarize the phase-2 log into four cases:
    1. gate.v == -1 and lv == 0: u -> g
    2. gate.v == -1 and lv == 1: u -> g
    3. gate.v != -1 and lv == 0: v -> g, v -> u
    4. gate.v != -1 and lv == 1: v -> g, v -> u
    """
    return {
        "unary_u_g": {
            lv: _summarize_index_to_targets(
                data["unary_u_g"][lv],
                data["unary_max_u"][lv],
                "u",
                "g",
                desc_prefix=f"phase2 unary lv={lv}",
                hard_window_size=hard_window_size,
                top_n=top_n,
            )
            for lv in (0, 1)
        },
        "binary_v_g": {
            lv: _summarize_index_to_targets(
                data["binary_v_g"][lv],
                data["binary_max_v"][lv],
                "v",
                "g",
                desc_prefix=f"phase2 binary lv={lv} v->g",
                hard_window_size=hard_window_size,
                top_n=top_n,
            )
            for lv in (0, 1)
        },
        "binary_v_u": {
            lv: _summarize_index_to_targets(
                data["binary_v_u"][lv],
                data["binary_max_v"][lv],
                "v",
                "u",
                desc_prefix=f"phase2 binary lv={lv} v->u",
                hard_window_size=hard_window_size,
                top_n=top_n,
            )
            for lv in (0, 1)
        },
    }


def summarize_phase2_relationship(filepath, hard_window_size=None, top_n=10):
    data = _load_phase2_relationship_data(filepath)
    return summarize_phase2_relationship_from_data(
        data,
        hard_window_size=hard_window_size,
        top_n=top_n,
    )


def _load_lasso_split_table(filepath, expected_header, expected_sumcheck_id):
    filepath = Path(filepath)
    rows = {}
    min_x = None
    max_x = -1
    header_seen = False

    with filepath.open("r", encoding="utf-8", errors="replace") as f:
        total = filepath.stat().st_size
        with tqdm(total=total, unit="B", unit_scale=True, desc=f"Reading {filepath.name}") as pbar:
            for line in f:
                pbar.update(len(line.encode("utf-8")))
                parts = line.split()
                if not parts:
                    continue

                if not header_seen:
                    if parts[:3] != list(expected_header):
                        raise ValueError(
                            f"Unexpected header in {filepath}: expected {expected_header}, got {parts[:3]}"
                        )
                    header_seen = True
                    continue

                if len(parts) < 3:
                    continue

                try:
                    sumcheck_id = int(parts[0])
                    x_value = int(parts[1])
                    y_value = int(parts[2])
                except ValueError:
                    continue

                if sumcheck_id != expected_sumcheck_id:
                    print(
                        f"Skipping row with unexpected sumcheck_id={sumcheck_id} in {filepath}; "
                        f"expected {expected_sumcheck_id}."
                    )
                    continue

                rows.setdefault(x_value, []).append(y_value)
                if min_x is None or x_value < min_x:
                    min_x = x_value
                if x_value > max_x:
                    max_x = x_value

    if not header_seen:
        raise ValueError(f"Missing header in {filepath}")

    if min_x is None:
        min_x = 0

    return rows, min_x, max_x


def _load_lasso_layer_data(model_name, squeeze_merge, layer_id):
    lasso_dir = Path(
        f"../output/{model_name}/{squeeze_merge}/{model_name}_lasso_mult_array"
    )
    if not lasso_dir.exists():
        raise FileNotFoundError(lasso_dir)
    if not lasso_dir.is_dir():
        raise NotADirectoryError(lasso_dir)

    layer_data = {}

    u_hu_path = lasso_dir / f"{layer_id}_u_hu.log"
    if u_hu_path.exists():
        layer_data["u_hu"] = _load_lasso_split_table(
            u_hu_path,
            ("sumcheck_id", "u", "hu"),
            layer_id,
        )
    else:
        print(f"Missing lasso file for sumcheck_id={layer_id}: {u_hu_path}. Skipping u->hu.")
        layer_data["u_hu"] = None

    v_hv_path = lasso_dir / f"{layer_id}_v_hv.log"
    if v_hv_path.exists():
        layer_data["v_hv"] = _load_lasso_split_table(
            v_hv_path,
            ("sumcheck_id", "v", "hv"),
            layer_id,
        )
    else:
        print(f"Missing lasso file for sumcheck_id={layer_id}: {v_hv_path}. Skipping v->hv.")
        layer_data["v_hv"] = None

    return layer_data


def _summarize_lasso_layer_from_data(layer_id, layer_data, hard_window_size=None, top_n=10):
    layer_result = {}

    if layer_data.get("u_hu") is not None:
        u_hu_rows, u_hu_min_x, u_hu_max_x = layer_data["u_hu"]
        layer_result["u_hu"] = _summarize_lasso_index_to_targets(
            u_hu_rows,
            u_hu_min_x,
            u_hu_max_x,
            "u",
            "hu",
            desc_prefix=f"lasso sumcheck_id={layer_id} u->hu",
            hard_window_size=hard_window_size,
            top_n=top_n,
        )
    else:
        layer_result["u_hu"] = _empty_relation_result("u", "hu")

    if layer_data.get("v_hv") is not None:
        v_hv_rows, v_hv_min_x, v_hv_max_x = layer_data["v_hv"]
        layer_result["v_hv"] = _summarize_lasso_index_to_targets(
            v_hv_rows,
            v_hv_min_x,
            v_hv_max_x,
            "v",
            "hv",
            desc_prefix=f"lasso sumcheck_id={layer_id} v->hv",
            hard_window_size=hard_window_size,
            top_n=top_n,
        )
    else:
        layer_result["v_hv"] = _empty_relation_result("v", "hv")

    return layer_result


def summarize_lasso_relationship(
    model_name,
    squeeze_merge,
    target_sumcheck_id,
    hard_window_size=None,
    top_n=10,
):
    """
    Summarize split lasso logs stored in:
    ../output/{model_name}/{squeeze_merge}/{model_name}_lasso_mult_array/

    Files are expected to look like:
    - {sumcheck_id}_u_hu.log
    - {sumcheck_id}_v_hv.log

    For each requested sumcheck_id, load whichever of the two files exists.
    Missing files are reported and skipped.

    Return a dict of layer_id -> {"u_hu": ..., "v_hv": ...}.
    """
    lasso_dir = Path(
        f"../output/{model_name}/{squeeze_merge}/{model_name}_lasso_mult_array"
    )
    if not lasso_dir.exists():
        raise FileNotFoundError(lasso_dir)
    if not lasso_dir.is_dir():
        raise NotADirectoryError(lasso_dir)

    if isinstance(target_sumcheck_id, (list, tuple, set)):
        target_sumcheck_ids = {int(layer_id) for layer_id in target_sumcheck_id}
    else:
        target_sumcheck_ids = {int(target_sumcheck_id)}

    results = {}
    for layer_id in sorted(target_sumcheck_ids):
        layer_data = _load_lasso_layer_data(model_name, squeeze_merge, layer_id)
        results[layer_id] = _summarize_lasso_layer_from_data(
            layer_id,
            layer_data,
            hard_window_size=hard_window_size,
            top_n=top_n,
        )

    return results


def _run_layer_tasks(layers, task_fn, desc):
    results = {}
    for layer in tqdm(layers, desc=desc):
        results[layer] = task_fn(layer)
    return results


def _strip_dump_noise(obj):
    if isinstance(obj, dict):
        cleaned = {}
        for key, value in obj.items():
            if isinstance(key, str):
                if key.count("_to_") == 1:
                    continue
                if "_hard_window_min_" in key or "_hard_window_max_" in key:
                    continue
                if key.endswith("_uncaptured_unique_y_set") or key.endswith("_uncaptured_unique_u_set") or key.endswith("_uncaptured_unique_hu_set") or key.endswith("_uncaptured_unique_hv_set") or key.endswith("_uncaptured_unique_g_set"):
                    continue
                if (
                    key.endswith("_count")
                    and not key.startswith("total_")
                    and "_window_" not in key
                    and "_easy_total_" not in key
                    and "_easy_unique_" not in key
                ):
                    continue
            cleaned[key] = _strip_dump_noise(value)
        return cleaned
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, list):
        return [_strip_dump_noise(value) for value in obj]
    return obj


def _dump_layer_results(
    model_name,
    squeeze_merge,
    hard_window_size,
    top_n,
    section_name,
    section_results,
    dump_json_dir=None,
):
    if dump_json_dir:
        output_dir = Path(dump_json_dir) / model_name / squeeze_merge
    else:
        output_dir = Path("./comp_data") / model_name / squeeze_merge
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []
    top_n_exp = top_n.bit_length() - 1

    for layer, section_result in section_results.items():
        output_path = output_dir / (
            f"{model_name}_layer_{layer}_hardwins_{hard_window_size}_topn_{top_n_exp}.json"
        )

        if output_path.exists():
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        else:
            payload = {
                "model_name": model_name,
                "squeeze_merge": squeeze_merge,
                "layer": layer,
                "hard_window_size": hard_window_size,
                "top_n": top_n,
                "top_n_exp": top_n_exp,
            }

        payload["hard_window_size"] = hard_window_size
        payload["top_n"] = top_n
        payload["top_n_exp"] = top_n_exp
        payload[section_name] = _strip_dump_noise(section_result)

        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        output_paths.append(output_path)
        print(f"Dumped {section_name} layer results to {output_path}")

    return output_paths


def _process_model_squeeze_layer(job):
    model_name = job["model_name"]
    squeeze_merge = job["squeeze_merge"]
    layer = job["layer"]
    hard_window_sizes = job["hard_window_sizes"]
    top_ns = job["top_ns"]
    dump_json_dir = job.get("dump_json_dir")

    print(f"Loading data once for {model_name} {squeeze_merge} layer={layer}...")
    phase1_path = (
        f"../output/{model_name}/{squeeze_merge}/"
        f"{model_name}_initP1_mult_array/{model_name}_initP1_mult_array_layers_{layer}.log"
    )
    phase2_path = (
        f"../output/{model_name}/{squeeze_merge}/"
        f"{model_name}_initP2_mult_array/{model_name}_initP2_mult_array_layers_{layer}.log"
    )

    phase1_data = _load_phase1_u_g_relationship_data(phase1_path)
    phase2_data = _load_phase2_relationship_data(phase2_path)
    lasso_data = _load_lasso_layer_data(model_name, squeeze_merge, layer)

    for hard_window_size, top_n in itertools.product(hard_window_sizes, top_ns):
        print(
            f"Processing {model_name} {squeeze_merge} layer={layer} "
            f"with hard_window_size={hard_window_size} and top_n={top_n}..."
        )

        phase1_result = summarize_phase1_u_g_relationship_from_data(
            phase1_data,
            hard_window_size=hard_window_size,
            top_n=top_n,
        )
        _dump_layer_results(
            model_name,
            squeeze_merge,
            hard_window_size,
            top_n,
            "phase1",
            {layer: phase1_result},
            dump_json_dir=dump_json_dir,
        )

        phase2_result = summarize_phase2_relationship_from_data(
            phase2_data,
            hard_window_size=hard_window_size,
            top_n=top_n,
        )
        _dump_layer_results(
            model_name,
            squeeze_merge,
            hard_window_size,
            top_n,
            "phase2",
            {layer: phase2_result},
            dump_json_dir=dump_json_dir,
        )

        lasso_result = _summarize_lasso_layer_from_data(
            layer,
            lasso_data,
            hard_window_size=hard_window_size,
            top_n=top_n,
        )
        _dump_layer_results(
            model_name,
            squeeze_merge,
            hard_window_size,
            top_n,
            "lasso",
            {layer: lasso_result},
            dump_json_dir=dump_json_dir,
        )

        print(
            f"Completed {model_name} {squeeze_merge} layer={layer} "
            f"with hard_window_size={hard_window_size} and top_n={top_n}."
        )

    return {
        "model_name": model_name,
        "squeeze_merge": squeeze_merge,
        "layer": layer,
    }


def _build_phase1_task(model_name, squeeze_merge, hard_window_size, top_n):
    def _task(layer):
        layers_file_name = f"{model_name}_initP1_mult_array_layers_{layer}"
        u_g_data_file = (
            f"../output/{model_name}/{squeeze_merge}/"
            f"{model_name}_initP1_mult_array/{layers_file_name}.log"
        )
        return summarize_phase1_u_g_relationship(
            u_g_data_file,
            hard_window_size=hard_window_size,
            top_n=top_n,
        )

    return _task


def _build_phase2_task(model_name, squeeze_merge, hard_window_size, top_n):
    def _task(layer):
        layers_file_name = f"{model_name}_initP2_mult_array_layers_{layer}"
        u_g_data_file = (
            f"../output/{model_name}/{squeeze_merge}/"
            f"{model_name}_initP2_mult_array/{layers_file_name}.log"
        )
        return summarize_phase2_relationship(
            u_g_data_file,
            hard_window_size=hard_window_size,
            top_n=top_n,
        )

    return _task


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hard-window-sizes",
        type=int,
        nargs="+",
        default=[256, 512, 1024, 2048], # 
        help="One or more hard window sizes to evaluate. E.g., --hard-window-sizes 256 512 1024",
    )
    parser.add_argument(
        "--top-ns",
        type=int,
        nargs="+",
        default=[2**13, 2**14, 2**15, 2**16, 2**17],
        help="One or more top_n values to evaluate. E.g., --top-ns 16384 32768 65536 131072",
    )
    parser.add_argument(
        "--multiprocess",
        action="store_true",
        help="Use multiprocessing across entries in model_squeeze_list.",
    )
    parser.add_argument(
        "--multithread",
        action="store_true",
        help="Use multithreading for per-layer P1/P2 (hard_window_size, top_n) tasks and inner window work.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=64,
        help="Maximum number of workers.",
    )
    parser.add_argument(
        "--dump-json-dir",
        type=str,
        default="./comp_data",
        help="Directory to dump JSON results. Defaults to './comp_data'.",
    )
    return parser.parse_args()



if __name__ == "__main__":
    args = _parse_args()
    hard_window_sizes = args.hard_window_sizes
    top_ns = args.top_ns
    dump_json_dir = args.dump_json_dir

    # model name, squeeze_merge, target layers to process
    model_squeeze_list = [
        ("gpt2-small", "SqueezeMerge_1", [1, 2, 3, 4, 6, 8]),  #
        ("opt-125m", "SqueezeMerge_0", [1,2,3,5,6,7,8,9,11,12,13,14,16,17,19,20,21,22,24,25,26,27,28,30,31,32,33,35,36,38]), # count=30
        ("gpt2-medium", "SqueezeMerge_1", [1, 2, 3, 4, 6, 8]),
    ]

    jobs = [
        {
            "model_name": model_name,
            "squeeze_merge": squeeze_merge,
            "layer": layer,
            "hard_window_sizes": hard_window_sizes,
            "top_ns": top_ns,
            "dump_json_dir": dump_json_dir,
        }
        for model_name, squeeze_merge, layers in model_squeeze_list
        for layer in layers[::-1]
    ]

    if args.multiprocess:
        worker_count = min(len(jobs), args.max_workers) if jobs else 0
        with ProcessPoolExecutor(max_workers=worker_count or None) as executor:
            futures = [executor.submit(_process_model_squeeze_layer, job) for job in jobs]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing layers"):
                result = future.result()
                print(
                    f"Finished {result['model_name']} {result['squeeze_merge']} "
                    f"layer={result['layer']}."
                )
    else:
        for job in jobs:
            _process_model_squeeze_layer(job)

    unique_count_jobs = [
        (model_name, squeeze_merge)
        for model_name, squeeze_merge, _ in model_squeeze_list
    ]
    unique_worker_count = min(len(unique_count_jobs), args.max_workers) if unique_count_jobs else 0
    with ProcessPoolExecutor(max_workers=unique_worker_count or None) as executor:
        futures = {
            executor.submit(
                count_unique_u_v_in_lasso_logs,
                model_name,
                squeeze_merge,
                output_root="../output",
                dump_json_dir=dump_json_dir,
            ): (model_name, squeeze_merge)
            for model_name, squeeze_merge in unique_count_jobs
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Counting unique u/v"):
            model_name, squeeze_merge = futures[future]
            result = future.result()
            print(
                f"Finished unique count for {model_name} {squeeze_merge}: "
                f"{result['total_unique_count']}."
            )

    print("Compiling gate range end...")
