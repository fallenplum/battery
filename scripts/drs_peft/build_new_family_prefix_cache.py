"""Build a prefix-only 12-channel cache for the new CALCE blind family."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


PREFIX = 100
MIN_CURVE_ROWS = 80


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def longest_run(mask: np.ndarray) -> np.ndarray:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return indices
    groups = np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)
    return max(groups, key=len)


def current_reference(cycles: list[dict], nominal: float) -> float | None:
    values = []
    for cycle in cycles[:5]:
        current = np.asarray(cycle.get("current_in_A", []), dtype=float)
        selected = -current[np.isfinite(current) & (current < -0.01 * nominal)]
        if len(selected):
            cutoff = np.quantile(selected, 0.8)
            values.append(float(np.median(selected[selected >= cutoff])))
    return float(np.median(values)) if values else None


def qv_curve(cycle: dict, reference_current: float | None, nominal: float, grid: np.ndarray, current_tolerance: float) -> tuple[np.ndarray, np.ndarray] | None:
    voltage = np.asarray(cycle.get("voltage_in_V", []), dtype=float)
    current = np.asarray(cycle.get("current_in_A", []), dtype=float)
    capacity = np.asarray(cycle.get("discharge_capacity_in_Ah", []), dtype=float)
    if not (voltage.shape == current.shape == capacity.shape) or len(voltage) < 15:
        return None
    mask = np.isfinite(voltage) & np.isfinite(current) & np.isfinite(capacity)
    mask &= current < -0.01 * nominal
    if current_tolerance > 0:
        if reference_current is None:
            return None
        mask &= np.abs(-current - reference_current) <= current_tolerance * reference_current
    run = longest_run(mask)
    if len(run) < 15:
        return None
    voltage, capacity = voltage[run], capacity[run]
    order = np.argsort(voltage, kind="stable")
    voltage, capacity = voltage[order], capacity[order]
    voltage, unique = np.unique(voltage, return_index=True)
    capacity = capacity[unique]
    if len(voltage) < 12 or voltage[0] > grid[0] or voltage[-1] < grid[-1]:
        return None
    q_grid = np.interp(grid, voltage, capacity).astype(np.float32)
    return q_grid, np.full_like(q_grid, -reference_current / nominal, dtype=np.float32)


def stats(raw: np.ndarray) -> np.ndarray:
    scales = np.asarray([5.0, 5.0, 1.0, 1.0], dtype=np.float32)
    values = raw / scales
    mean = values.mean(axis=0)
    deviation = values.std(axis=0)
    span = values[:, 2:4].max(axis=0) - values[:, 2:4].min(axis=0)
    covariance = np.asarray(
        [np.mean((values[:, 0] - mean[0]) * (values[:, j] - mean[j])) for j in (2, 3)],
        dtype=np.float32,
    )
    return np.concatenate((mean, deviation, span, covariance)).astype(np.float32)


def inspect_cell(path: Path, grid: np.ndarray, current_tolerance: float) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float] | None:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    cycles = payload["cycle_data"]
    nominal = float(payload["nominal_capacity_in_Ah"])
    prefix = [c for c in cycles if c.get("cycle_number") is not None and 1 <= int(c["cycle_number"]) <= PREFIX]
    reference_current = current_reference(prefix, nominal)
    if reference_current is None:
        return None
    curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    capacities: dict[int, float] = {}
    for cycle in prefix:
        number = int(cycle["cycle_number"])
        curve = qv_curve(cycle, reference_current, nominal, grid, current_tolerance)
        if curve is not None:
            curves[number] = curve
        q = np.asarray(cycle.get("discharge_capacity_in_Ah", []), dtype=float)
        if len(q) and np.isfinite(q).any():
            capacities[number] = float(np.nanmax(q) / nominal)
    valid = sorted(curves)
    if len(valid) < MIN_CURVE_ROWS or not set(range(1, PREFIX + 1)).issubset(curves):
        return None
    reference = np.median(np.stack([curves[n][0] for n in valid if n <= 10]), axis=0)
    if not np.isfinite(reference).all():
        return None
    real = np.zeros((PREFIX, 12), dtype=np.float32)
    reference_view = np.zeros_like(real)
    mask = np.zeros((PREFIX, 12), dtype=bool)
    cycles_out = np.zeros(PREFIX, dtype=np.int64)
    for n in range(1, PREFIX + 1):
        q_grid, current_grid = curves[n]
        drift = q_grid - reference
        raw = np.stack((grid, current_grid, q_grid / nominal, drift / nominal), axis=-1)
        raw_reference = np.stack((grid, current_grid, reference / nominal, np.zeros_like(reference)), axis=-1)
        real[n - 1] = stats(raw)
        reference_view[n - 1] = stats(raw_reference)
        mask[n - 1] = True
        cycles_out[n - 1] = n
    anchor = capacities.get(PREFIX)
    if anchor is None or not np.isfinite(anchor):
        return None
    metadata = {
        "cell_id": payload.get("cell_id"),
        "file": path.name,
        "file_sha256": sha256(path),
        "nominal_capacity_Ah": nominal,
        "prefix_curve_rows": len(valid),
        "reference_current_A": reference_current,
    }
    return metadata, real, reference_view, mask, cycles_out, float(anchor)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--family", default="SNL")
    parser.add_argument("--voltage-min", type=float, default=2.8)
    parser.add_argument("--voltage-max", type=float, default=3.3)
    parser.add_argument("--current-tolerance", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, features, references, masks, cycles, anchors = [], [], [], [], [], []
    grid = np.linspace(args.voltage_min, args.voltage_max, 64, dtype=np.float32)
    for path in sorted((args.dataset_root / args.family).glob("*.pkl")):
        result = inspect_cell(path, grid, args.current_tolerance)
        if result is None:
            continue
        metadata, real, reference, mask, cycle_ids, anchor = result
        rows.append(metadata)
        features.append(real)
        references.append(reference)
        masks.append(mask)
        cycles.append(cycle_ids)
        anchors.append(anchor)
    if not rows:
        raise RuntimeError("No CALCE cell passed the prefix qualification")
    args.output.mkdir(parents=True, exist_ok=True)
    ids = np.asarray([row["cell_id"] for row in rows])
    np.savez_compressed(
        args.output / "PAIRED_PREFIXES.npz",
        ids=ids,
        real_features=np.stack(features),
        reference_features=np.stack(references),
        feature_mask=np.stack(masks),
        cycles=np.stack(cycles),
        real_anchor=np.asarray(anchors, dtype=np.float32),
        reference_anchor=np.asarray(anchors, dtype=np.float32),
    )
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "CELL_METADATA.csv", index=False, lineterminator="\n")
    receipt = {
        "status": "COMPLETE_NEW_FAMILY_PREFIX_ONLY_CACHE",
        "domain": args.family,
        "physical_cells": len(rows),
        "input_channels": 12,
        "prefix_cycles": 100,
        "voltage_grid_V": [float(grid[0]), float(grid[-1])],
        "prefix_future_labels_read": 0,
        "target_future_labels_read": 0,
        "source_selection_labels_read": 0,
        "cache_sha256": sha256(args.output / "PAIRED_PREFIXES.npz"),
        "metadata_sha256": sha256(args.output / "CELL_METADATA.csv"),
        "preprocessing": f"BatteryLife {args.family} pickle, discharge active-segment rule with current_tolerance={args.current_tolerance}, common {args.voltage_min}-{args.voltage_max}V grid, cycle-statistics 12 channels",
        "cells": rows,
    }
    (args.output / "PREFIX_CACHE_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(frame[["cell_id", "prefix_curve_rows"]].to_string(index=False))


if __name__ == "__main__":
    main()

