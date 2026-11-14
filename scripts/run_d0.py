"""Audit observed curves, source-supported cycle coordinates, and future targets."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, savgol_filter

from physics_targets import future_targets


SETTINGS = {
    "HUST": {"direction": "charge", "voltage_window": [3.1, 3.4]},
    "MATR": {"direction": "discharge", "voltage_window": [2.8, 3.3]},
    "Stanford_2": {"direction": "discharge", "voltage_window": [3.4, 4.1]},
}
PREFIXES = [10, 50, 100]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


class HashReader:
    def __init__(self, file):
        self.file = file
        self.hash = hashlib.sha256()

    def read(self, size=-1):
        block = self.file.read(size)
        self.hash.update(block)
        return block

    def readline(self, size=-1):
        block = self.file.readline(size)
        self.hash.update(block)
        return block


def load_existing_pickle(path):
    # This entry point reads the user's existing local audited dataset only.
    # Newly downloaded source archives use the independent safe source parser.
    import pickle
    with path.open("rb") as file:
        reader = HashReader(file)
        cell = pickle.load(reader)
        while reader.read(8 * 1024 * 1024):
            pass
    return cell, reader.hash.hexdigest()


def capacity(cycle, key, nominal):
    q = np.asarray(cycle.get(key), dtype=float)
    current = np.asarray(cycle.get("current_in_A"), dtype=float)
    if q.ndim != 1 or current.shape != q.shape:
        return None
    phase = current < -0.01 * nominal if key.startswith("discharge") else current > 0.01 * nominal
    active = phase & np.isfinite(q) & np.isfinite(current)
    if np.count_nonzero(active) < 15:
        return None
    valid = q[np.isfinite(q)]
    if valid.size < 2:
        return None
    value = float(q[active].max() - valid.min()) / nominal
    return value if 0 < value < 1.5 else None


def longest_run(mask):
    indices = np.flatnonzero(mask)
    if not indices.size:
        return indices
    groups = np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)
    return max(groups, key=len)


def current_reference(cycles, direction):
    samples = []
    sign = 1 if direction == "charge" else -1
    for cycle in cycles[:5]:
        current = np.asarray(cycle["current_in_A"], dtype=float) * sign
        selected = current[np.isfinite(current) & (current > 0)]
        if selected.size:
            cutoff = np.quantile(selected, 0.8)
            samples.append(float(np.median(selected[selected >= cutoff])))
    return float(np.median(samples)) if samples else None


def observed_curve(cycle, setting, reference, nominal):
    if reference is None or reference <= 0:
        return None, "current_reference_missing", None
    direction = setting["direction"]
    sign = 1 if direction == "charge" else -1
    v = np.asarray(cycle["voltage_in_V"], dtype=float)
    i = np.asarray(cycle["current_in_A"], dtype=float)
    q = np.asarray(cycle[f"{direction}_capacity_in_Ah"], dtype=float)
    if not len(v) == len(i) == len(q):
        return None, "channel_length_mismatch", None
    mask = np.isfinite(v) & np.isfinite(i) & np.isfinite(q)
    mask &= np.abs(i * sign - reference) <= 0.05 * reference
    ix = longest_run(mask)
    if len(ix) < 15:
        return None, "constant_current_segment_short", None
    voltage, charge = v[ix], q[ix]
    lo, hi = setting["voltage_window"]
    if voltage.min() > lo or voltage.max() < hi:
        return None, "voltage_window_not_covered", None
    if np.mean(np.diff(voltage) * sign < -0.002) > 0.1:
        return None, "voltage_reversal_excess", None
    if np.any(np.diff(charge) < -nominal * 1e-5):
        return None, "capacity_reset_or_reversal", None
    order = np.argsort(voltage, kind="stable")
    unique_v, unique_ix = np.unique(voltage[order], return_index=True)
    if len(unique_v) < 12:
        return None, "voltage_resolution_low", None
    grid = np.linspace(lo, hi, 64)
    curve = np.interp(grid, unique_v, charge[order][unique_ix])
    curve = (curve - curve[0]) * sign
    if curve[-1] < nominal * 0.01:
        return None, "window_capacity_too_small", None
    smoothed = savgol_filter(curve, 9, 2)
    derivative = np.gradient(smoothed, grid)
    peaks, _ = find_peaks(derivative, prominence=max(float(np.ptp(derivative)) * 0.05, 1e-8))
    proxy = {"ic_shape_peak_count": int(len(peaks)),
             "ic_shape_max": float(derivative.max()),
             "ic_shape_peak_voltage": float(grid[int(np.argmax(derivative))])}
    return curve, "observed_curve_supported", proxy


def make_features(cycle_numbers, capacities, curves, prefix):
    rows = [(n, c) for n, c in zip(cycle_numbers, capacities) if n <= prefix and c is not None]
    selected = [(n, curve) for n, curve in curves if n <= prefix]
    result = {"prefix_cutoff_supplied_cycle_number": prefix, "early_capacity_records": len(rows),
              "early_curve_records": len(selected)}
    if rows:
        number, values = np.asarray(rows, dtype=float).T
        result["early_capacity_level"] = float(np.median(values[-min(5, len(values)):]))
        result["early_capacity_slope"] = float(np.polyfit(number, values, 1)[0]) if len(values) > 2 else None
        result["early_capacity_std"] = float(np.std(values))
    if len(selected) < 2:
        return result
    first = np.median([curve for _, curve in selected[:min(3, len(selected))]], axis=0)
    last = np.median([curve for _, curve in selected[-min(3, len(selected)):]], axis=0)
    delta = last - first
    result.update({"qv_window_capacity_start_Ah": float(first[-1]),
                   "qv_window_capacity_end_Ah": float(last[-1]),
                   "delta_qv_mean_Ah": float(delta.mean()),
                   "delta_qv_std_Ah": float(delta.std()),
                   "delta_qv_log10_variance": float(np.log10(max(float(np.var(delta)), 1e-16))),
                   "delta_qv_range_Ah": float(np.ptp(delta)),
                   "delta_qv_rms_Ah": float(np.sqrt(np.mean(delta ** 2))),
                   "qv_cross_cycle_noise_Ah": float(np.median(np.std([curve for _, curve in selected], axis=0)))})
    return result


def read_audit(path):
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        records[item["relative_path"]] = item
    return records


def inspect_cell(path, root, audited, legacy_ids):
    started = time.monotonic()
    payload, digest = load_existing_pickle(path)
    if digest != audited["sha256"]:
        raise ValueError(f"Dataset hash changed for {path.name}")
    family = path.relative_to(root).parts[0]
    identifier = payload["cell_id"]
    nominal = float(payload["nominal_capacity_in_Ah"])
    cycles = payload["cycle_data"]
    numbers = [int(cycle["cycle_number"]) for cycle in cycles]
    global_cycle_order_valid = len(numbers) == len(set(numbers)) and numbers == sorted(numbers)
    setting = SETTINGS[family]
    early = [cycle for cycle in cycles if int(cycle["cycle_number"]) <= 100]
    reference = current_reference(early, setting["direction"])
    discharge = [capacity(cycle, "discharge_capacity_in_Ah", nominal) for cycle in cycles]
    curves, failures, proxies, waveforms = [], Counter(), [], []
    for cycle in early:
        curve, reason, proxy = observed_curve(cycle, setting, reference, nominal)
        failures[reason] += 1
        current = np.asarray(cycle["current_in_A"], dtype=float)
        waveforms.append(bool(np.any(current > nominal * 0.01) and np.any(current < -nominal * 0.01)))
        if curve is not None:
            curves.append((int(cycle["cycle_number"]), curve))
            proxies.append(proxy)
    prefix_complete = all(n in set(numbers) for n in range(1, 101))
    source_supported = family == "HUST" and prefix_complete and all(waveforms)
    row = {"cell_id": identifier, "family": family, "provided_file": path.relative_to(root).as_posix(),
           "provided_file_sha256": digest, "legacy_576_member": path.name in legacy_ids,
           "provided_cycles": len(cycles), "supplied_cycle_min": min(numbers), "supplied_cycle_max": max(numbers),
           "full_trajectory_cycle_order_valid": global_cycle_order_valid,
           "prefix_K100_records": len(early), "prefix_K100_numbers_complete": prefix_complete,
           "physical_cycle_source_supported": source_supported,
           "physical_cycle_axis_authenticated": False,
           "physical_cycle_evidence": "HUST_source_one_dataframe_per_cycle_and_waveform" if source_supported else "original_cycle_mapping_unverified",
           "physical_cycle_confidence": "medium" if source_supported else "low",
           "phase_role_supported": family == "HUST" and all(waveforms),
           "time_axis_authenticated": False, "original_instrument_source_available": False,
           "nominal_capacity_Ah": nominal, "response_direction": setting["direction"],
           "comparison_voltage_min_V": setting["voltage_window"][0],
           "comparison_voltage_max_V": setting["voltage_window"][1],
           "prefix_current_reference_A": reference, "curve_valid_records_K100": len(curves),
           "curve_failure_counts": dict(failures), "temperature_matching_authenticated": False,
           "N_true_ICA_records": 0, "N_true_DVA_records": 0, "resistance_truth_supported": False,
           "observed_IC_shape_proxy_records": len(proxies),
           "future_policy_status": "observed_only", "future_policy_basis": "static_design_requires_measured_suffix_check",
           "capacity_coordinate": "supplied_cycle_number", "prefix_capacity_valid": sum(q is not None for n, q in zip(numbers, discharge) if n <= 100),
           "invalid_capacity_records": sum(q is None for q in discharge),
           "feature_support_scope": "processed_observed_QV_under_early_current_matching"}
    features = [{"cell_id": identifier, "family": family, **make_features(numbers, discharge, curves, prefix)} for prefix in PREFIXES]
    valid_numbers = [n for n, q in zip(numbers, discharge) if q is not None]
    valid_soh = [q for q in discharge if q is not None]
    targets = future_targets(np.asarray(valid_numbers), np.asarray(valid_soh))
    targets.update({"cell_id": identifier, "family": family, "target_coordinate": "supplied_cycle_number",
                    "strict_physical_target_authenticated": False})
    row["target_mid_supported"] = targets.get("r200_500") is not None
    row["target_late_supported"] = targets.get("r500_1000") is not None
    row["target_EOL_event"] = bool(targets.get("eol_event", False))
    row["target_EOL_censored"] = not row["target_EOL_event"]
    row["target_knee_stable_candidate"] = bool(targets.get("knee_stable_candidate", False))
    examples = []
    for number in [10, 50, 100]:
        matches = [c for n, c in curves if n == number]
        if matches:
            examples.append({"supplied_cycle_number": number, "curve": matches[0].tolist()})
    result = {"eligible": row, "features": features, "targets": targets, "curve_examples": examples,
              "capacity_trajectory": {"cycle_number": valid_numbers, "soh": valid_soh},
              "elapsed_seconds": round(time.monotonic() - started, 3)}
    del payload, cycles, early
    gc.collect()
    return result


def aggregate(records, output):
    eligible = [r["eligible"] for r in records]
    features = [f for r in records for f in r["features"]]
    targets = [r["targets"] for r in records]
    write_csv(output / "eligible_cells.csv", eligible)
    write_csv(output / "early_observed_features.csv", features)
    write_csv(output / "future_target_support.csv", targets)
    write_csv(output / "future_condition_contract.csv", [{key: row[key] for key in ("cell_id", "family", "future_policy_status", "future_policy_basis", "temperature_matching_authenticated", "physical_cycle_axis_authenticated")} for row in eligible])
    coverage = []
    for family in SETTINGS:
        cells = [row for row in eligible if row["family"] == family]
        coverage.append({"family": family, "N_provided": len(cells),
                         "N_raw_instrument_source": sum(row["original_instrument_source_available"] for row in cells),
                         "N_K_valid_supplied_numbers": sum(row["prefix_K100_numbers_complete"] for row in cells),
                         "N_K_source_supported": sum(row["physical_cycle_source_supported"] and row["prefix_K100_numbers_complete"] for row in cells),
                         "N_K_strict_authenticated": sum(row["physical_cycle_axis_authenticated"] and row["prefix_K100_numbers_complete"] for row in cells),
                         "N_curve_observed_supported": sum(row["curve_valid_records_K100"] >= 80 for row in cells),
                         "N_ICA_valid": 0, "N_DVA_valid": 0,
                         "N_mid_slope": sum(row["target_mid_supported"] for row in cells),
                         "N_late_slope": sum(row["target_late_supported"] for row in cells),
                         "N_EOLevent": sum(row["target_EOL_event"] for row in cells),
                         "N_EOTime_censored": sum(row["target_EOL_censored"] for row in cells),
                         "N_knee_stable_candidate": sum(row["target_knee_stable_candidate"] for row in cells),
                         "N_knee_event": 0,
                         "N_policy_known": sum(row["future_policy_status"] in ("known_fixed", "known_scheduled") for row in cells),
                         "N_curve_K_mid_late_source_intersection": sum(row["physical_cycle_source_supported"] and row["prefix_K100_numbers_complete"] and row["curve_valid_records_K100"] >= 80 and row["target_mid_supported"] and row["target_late_supported"] for row in cells)})
    write_csv(output / "feature_coverage_by_family_protocol.csv", coverage)
    write_json(output / "D0_STATUS.json", {"provided_cells_completed": len(eligible), "expected": 427,
                                           "coverage": coverage, "status": "COMPLETE" if len(eligible) == 427 else "PARTIAL"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--prior-audit", type=Path, required=True)
    parser.add_argument("--legacy-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-cache", type=Path, required=True)
    args = parser.parse_args()
    args.private_cache.mkdir(parents=True, exist_ok=True)
    audit = read_audit(args.prior_audit / "standardized_cells.jsonl")
    legacy = set(json.loads(args.legacy_manifest.read_text(encoding="utf-8"))["cells"])
    cache = args.private_cache / "d0_cells.jsonl"
    signature = {"scanner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 "target_code_sha256": hashlib.sha256(Path(__file__).with_name("physics_targets.py").read_bytes()).hexdigest(),
                 "prior_audit_sha256": hashlib.sha256((args.prior_audit / "standardized_cells.jsonl").read_bytes()).hexdigest(),
                 "settings": SETTINGS, "numpy_version": np.__version__}
    signature_path = args.private_cache / "RUN_SIGNATURE.json"
    if cache.exists():
        if not signature_path.exists() or json.loads(signature_path.read_text(encoding="utf-8")) != signature:
            raise RuntimeError("Cache is bound to different code or data. Select a new private cache directory.")
    write_json(signature_path, signature)
    results = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            relative = record["eligible"]["provided_file"]
            if record["eligible"]["provided_file_sha256"] != audit[relative]["sha256"]:
                raise RuntimeError("Cached source hash differs from the audited source")
            results[record["eligible"]["cell_id"]] = record
    # The less traceable sources are audited while HUST source recovery proceeds.
    paths = [path for family in ("MATR", "Stanford_2", "HUST") for path in sorted((args.dataset_root / family).glob("*.pkl"))]
    with cache.open("a", encoding="utf-8") as file:
        for index, path in enumerate(paths, 1):
            if path.stem in results:
                continue
            relative = path.relative_to(args.dataset_root).as_posix()
            record = inspect_cell(path, args.dataset_root, audit[relative], legacy)
            file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            file.flush()
            results[path.stem] = record
            if index % 10 == 0:
                aggregate(list(results.values()), args.output)
                print(json.dumps({"completed": len(results), "expected": len(paths), "family": record["eligible"]["family"], "last": path.stem}), flush=True)
    aggregate(list(results.values()), args.output)
    print(json.dumps({"completed": len(results), "status": "COMPLETE"}), flush=True)


if __name__ == "__main__":
    main()
