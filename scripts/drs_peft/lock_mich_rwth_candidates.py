"""Prefix-only qualification and candidate-matrix lock for MICH and RWTH."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
from pathlib import Path

import pandas as pd


FAMILIES = {
    "MICH": {
        "voltage_window_V": [3.0, 4.2],
        "current_rule": "negative discharge active segment; family-specific current is retained in NativeBridge statistics",
        "group_rule": "BLForm/MCForm family, temperature token and nominal 1-1C protocol",
    },
    "RWTH": {
        "voltage_window_V": [3.5, 3.9],
        "current_rule": "negative discharge active segment; family-specific current is retained in NativeBridge statistics",
        "group_rule": "supplier RWTH file identity and top-level protocol metadata",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_prefix(path: Path, family: str) -> dict:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    cycles = [
        cycle
        for cycle in payload.get("cycle_data", [])
        if cycle.get("cycle_number") is not None and 1 <= int(cycle["cycle_number"]) <= 100
    ]
    numbers = [int(cycle["cycle_number"]) for cycle in cycles]
    voltage_min = []
    voltage_max = []
    valid_curve_rows = 0
    valid_capacity_rows = 0
    for cycle in cycles:
        voltage = cycle.get("voltage_in_V", [])
        current = cycle.get("current_in_A", [])
        capacity = cycle.get("discharge_capacity_in_Ah", [])
        if len(voltage) == len(current) == len(capacity) and len(voltage) >= 15:
            valid_curve_rows += 1
            voltage_min.append(min(float(value) for value in voltage))
            voltage_max.append(max(float(value) for value in voltage))
        if capacity:
            valid_capacity_rows += 1
    metadata = {
        key: payload.get(key)
        for key in (
            "cell_id",
            "form_factor",
            "anode_material",
            "cathode_material",
            "nominal_capacity_in_Ah",
            "depth_of_charge",
            "depth_of_discharge",
            "max_voltage_limit_in_V",
            "min_voltage_limit_in_V",
            "max_current_limit_in_A",
            "min_current_limit_in_A",
            "charge_protocol",
            "discharge_protocol",
            "SOC_interval",
        )
    }
    file_group = path.stem.split("_0-100")[0] if family == "MICH" else path.stem
    return {
        "family": family,
        "file": path.name,
        "cell_id": metadata["cell_id"],
        "file_sha256": sha256(path),
        "metadata": json.dumps(metadata, ensure_ascii=False, sort_keys=True, default=str),
        "protocol_group": file_group,
        "prefix_cycle_count": len(cycles),
        "prefix_cycle_min": min(numbers) if numbers else None,
        "prefix_cycle_max": max(numbers) if numbers else None,
        "prefix_cycle_numbers_unique_sorted": numbers == sorted(numbers) and len(numbers) == len(set(numbers)),
        "prefix_1_100_complete": set(numbers) == set(range(1, 101)),
        "prefix_curve_rows": valid_curve_rows,
        "prefix_capacity_rows": valid_capacity_rows,
        "prefix_voltage_min_V": min(voltage_min) if voltage_min else None,
        "prefix_voltage_max_V": max(voltage_max) if voltage_max else None,
        "future_labels_read": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for family in FAMILIES:
        for path in sorted((args.dataset_root / family).glob("*.pkl")):
            rows.append(inspect_prefix(path, family))
    frame = pd.DataFrame(rows)
    eligible = frame[
        frame.prefix_1_100_complete
        & (frame.prefix_curve_rows >= 80)
        & (frame.prefix_capacity_rows >= 80)
    ].copy()
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "MICH_RWTH_PREFIX_AUDIT.csv", index=False, lineterminator="\n")
    lock = {
        "status": "LOCKED_NEW_CONFIRMATORY_FAMILY_CANDIDATES_PREFIX_ONLY",
        "families": {},
        "primary_methods": ["HEAD_ONLY_TRAIN", "HEAD_LORA", "HEAD_BRIDGE_LORA"],
        "retrospective_supplement_methods": ["WARMUP_ONLY", "LORA_12", "FULL_FT_TUNED"],
        "source_selection": "MATR source fit/tune only, fixed before any MICH/RWTH future label access",
        "prediction_grid": [101, 5000],
        "target_definition": "per supplied cycle max discharge_capacity_in_Ah divided by declared nominal capacity, observed grid only",
        "target_future_labels_read": 0,
        "prefix_future_labels_read": 0,
        "candidate_matrix_status": "REGISTERED_PENDING_PREFIX_CACHE_AND_PREDICTION_LOCK",
    }
    for family, settings in FAMILIES.items():
        selected = eligible[eligible.family.eq(family)]
        lock["families"][family] = {
            **settings,
            "candidate_cells": selected.cell_id.tolist(),
            "candidate_count": int(len(selected)),
            "excluded_cells": sorted(set(frame.loc[frame.family.eq(family), "cell_id"]) - set(selected.cell_id)),
            "candidate_file_hashes": {row.cell_id: row.file_sha256 for row in selected.itertuples()},
            "prefix_qualification": "cycle IDs 1 through 100 complete, at least 80 valid curve rows and 80 capacity rows",
        }
    (args.output / "MICH_RWTH_CANDIDATE_LOCK.json").write_text(
        json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report = [
        "# MICH/RWTH 新确认性候选矩阵审计",
        "",
        "本轮只读取供应 pkl 的元数据和 1–100 周期输入字段。未读取 100 周期后的容量值，未进行目标评分。",
        "",
        "| Family | 原始文件 | 前100完整 | 资格电芯 | 目标口径 |",
        "|---|---:|---:|---:|---|",
    ]
    for family in FAMILIES:
        family_rows = frame[frame.family.eq(family)]
        selected = eligible[eligible.family.eq(family)]
        report.append(
            f"| {family} | {len(family_rows)} | {int(family_rows.prefix_1_100_complete.sum())} | {len(selected)} | supplied cycle observed SOH |"
        )
    report += [
        "",
        "主比较固定为 HEAD_ONLY_TRAIN、HEAD_LORA、HEAD_BRIDGE_LORA。WARMUP_ONLY、LORA_12、FULL_FT_TUNED 只作为未来标签公开后的回顾性补充矩阵，不参与确认性方法选择。",
        "",
        "MICH 与 RWTH 仍需分别构建 prefix-only 12 通道 cache、固定输入窗口、写入 prediction lock 后，才可进入新 family 预测。",
    ]
    (args.output / "MICH_RWTH_CANDIDATE_AUDIT_ZH.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(frame.groupby("family").agg(files=("cell_id", "count"), prefix_complete=("prefix_1_100_complete", "sum"), eligible=("cell_id", lambda s: int(s.isin(set(eligible.cell_id)).sum()))).to_string())


if __name__ == "__main__":
    main()

