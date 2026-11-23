"""Read-only audit of BatteryLife families not used in Head-aware selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


FAMILIES = ("CALCE", "CALB", "RWTH", "SNL", "MICH", "Tongji", "SDU")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric_array(cycle: dict, key: str) -> np.ndarray:
    value = cycle.get(key)
    if value is None:
        return np.empty(0, dtype=float)
    return np.asarray(value, dtype=float)


def inspect(path: Path, family: str, labels: dict) -> dict:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    cycles = payload.get("cycle_data", [])
    numbers = [int(c.get("cycle_number")) for c in cycles if c.get("cycle_number") is not None]
    early = [c for c in cycles if c.get("cycle_number") is not None and int(c["cycle_number"]) <= 100]
    valid_future = [n for n in numbers if n > 100]
    cycle_numbers = set(numbers)
    complete_prefix = set(range(1, 101)).issubset(cycle_numbers)
    curve_rows = 0
    capacity_rows = 0
    for cycle in early:
        voltage = numeric_array(cycle, "voltage_in_V")
        current = numeric_array(cycle, "current_in_A")
        discharge = numeric_array(cycle, "discharge_capacity_in_Ah")
        if voltage.size >= 15 and current.shape == voltage.shape and discharge.shape == voltage.shape:
            finite = np.isfinite(voltage) & np.isfinite(current) & np.isfinite(discharge)
            if int(finite.sum()) >= 15 and np.unique(voltage[finite]).size >= 12:
                curve_rows += 1
        if discharge.size and np.isfinite(discharge).any():
            capacity_rows += 1
    return {
        "family": family,
        "file": path.name,
        "cell_id": payload.get("cell_id"),
        "sha256": sha256(path),
        "nominal_capacity_Ah": payload.get("nominal_capacity_in_Ah"),
        "metadata_keys": json.dumps(sorted(k for k in payload if k != "cycle_data"), ensure_ascii=False),
        "cycle_count": len(cycles),
        "cycle_min": min(numbers) if numbers else None,
        "cycle_max": max(numbers) if numbers else None,
        "unique_cycle_numbers": len(set(numbers)),
        "cycle_order_valid": numbers == sorted(numbers) and len(numbers) == len(set(numbers)),
        "prefix_1_100_complete": complete_prefix,
        "prefix_records": len(early),
        "prefix_curve_rows": curve_rows,
        "prefix_capacity_rows": capacity_rows,
        "future_records_after_100": len(valid_future),
        "life_label_present": path.name in labels,
        "life_label": labels.get(path.name),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for family in FAMILIES:
        folder = args.dataset_root / family
        if not folder.exists():
            continue
        labels_path = args.labels_root / f"{family}_labels.json"
        if family == "MICH":
            labels_path = args.labels_root / "MICH_labels.json"
        labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.exists() else {}
        for path in sorted(folder.glob("*.pkl")):
            rows.append(inspect(path, family, labels))
    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, lineterminator="\n")
    summary = frame.groupby("family", as_index=False).agg(
        cells=("cell_id", "nunique"),
        prefix_complete=("prefix_1_100_complete", "sum"),
        prefix_curve_ge80=("prefix_curve_rows", lambda x: int((x >= 80).sum())),
        future_ge100=("future_records_after_100", lambda x: int((x >= 100).sum())),
        labelled=("life_label_present", "sum"),
    )
    summary.to_csv(args.output.with_name("BLIND_CANDIDATE_SUMMARY.csv"), index=False, lineterminator="\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

