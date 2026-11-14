"""Recompute raw-counter target support independently from the private CSV."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--provided-cache", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    series = defaultdict(list)
    count = Counter()
    formula_correct = True
    trajectory = args.source_root / "raw_discharge_capacity_trajectory.csv"
    with trajectory.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            count["rows"] += 1
            cycle = int(row["raw_cycle_number"])
            q = float(row["discharge_capacity_Ah"])
            soh = float(row["raw_soh_nominal"])
            counter_drop = (float(row["raw_counter_start_mAh"]) - float(row["raw_counter_end_mAh"])) / 1000
            formula_correct &= abs(counter_drop - q) < 1e-12 and abs(q / 1.1 - soh) < 1e-12
            series[row["cell_id"]].append((cycle, soh))
            count["valid"] += row["raw_capacity_valid"] == "True"
            count["upward"] += int(row["counter_upward_steps"])
            count["large_drop"] += int(row["counter_large_drop_gt10pct_nominal_steps"])
            count["negative_time"] += int(row["negative_discharge_time_steps"])
    with (args.results / "HUST_RAW_CAPACITY_TARGET_SUPPORT.csv").open(encoding="utf-8-sig", newline="") as file:
        targets = list(csv.DictReader(file))
    rates_correct = censoring_correct = True
    for row in targets:
        data = np.asarray(series[row["cell_id"]])
        cycles, soh = data[:, 0], data[:, 1]
        for name, left, right in (("r200_500", 200, 500), ("r500_1000", 500, 1000)):
            expected = (np.median(soh[np.abs(cycles - left) <= 5]) - np.median(soh[np.abs(cycles - right) <= 5])) / (right - left)
            rates_correct &= abs(expected - float(row[name])) < 1e-12
        censoring_correct &= row["eol_event"] == "False" and row["eol_censoring"] == "right" and soh.min() > .8
    provided = {}
    for line in args.provided_cache.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["eligible"]["family"] == "HUST":
            value = record["capacity_trajectory"]
            provided[record["eligible"]["cell_id"]] = dict(zip(value["cycle_number"], value["soh"]))
    boundary = json.loads((args.results / "HUST_BOUNDARY_SAMPLING_DIAGNOSTIC.json").read_text(encoding="utf-8"))
    boundary_matches = all(abs(provided[row["cell_id"]][row["raw_cycle_number"]] * 1.1 - row["right_endpoint_current_integral_Ah"]) < 1e-12 for row in boundary["rows"])
    checks = {
        "all_146122_rows_present": count["rows"] == 146122,
        "all_77_cells_present": len(series) == 77,
        "all_counter_capacities_valid": count["valid"] == 146122,
        "counter_formula_and_Ah_conversion": formula_correct,
        "zero_upward_counter_steps": count["upward"] == 0,
        "zero_large_counter_discontinuities": count["large_drop"] == 0,
        "no_decreasing_discharge_time": count["negative_time"] == 0,
        "all_77_rate_targets_independent_endpoint_recompute": rates_correct,
        "all_77_EOLevent_absent_and_censored_by_values": censoring_correct,
        "boundary_example_right_integrals_equal_current_provided_cache": boundary_matches,
        "target_support_77_unique_rows": len(targets) == len({row["cell_id"] for row in targets}) == 77,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    if not all(checks.values()):
        raise AssertionError(checks)
    digest = hashlib.sha256(trajectory.read_bytes()).hexdigest()
    result = {"status": "PASS", "checks": checks, "raw_rows": dict(count),
              "raw_trajectory_sha256": digest,
              "minimum_raw_soh": min(value for rows in series.values() for _, value in rows),
              "maximum_raw_soh": max(value for rows in series.values() for _, value in rows)}
    (args.results / "HUST_RAW_CAPACITY_VERIFICATION.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
