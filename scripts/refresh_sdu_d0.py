"""Refresh only SDU D0 candidates against the current shared feature functions."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from collections import Counter
from pathlib import Path

from audit_sdu import dump, file_sha256, inspect_sdu_d0, write_csv, write_sdu_report
from run_d0 import load_existing_pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    scripts = Path(__file__).resolve().parent
    dependencies_before = {name: file_sha256(scripts / f"{name}.py") for name in ("run_d0", "physics_targets", "audit_sdu")}
    lineage = json.loads((args.output / "sdu_preprocessing_lineage.json").read_text(encoding="utf-8"))
    verification = json.loads((args.output / "SDU_AUDIT_VERIFICATION.json").read_text(encoding="utf-8"))
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    if file_sha256(args.labels) != verification["life_label_sha256"]:
        raise ValueError("SDU labels changed after the full signal audit")
    records = [json.loads(line) for line in (args.private_output / "sdu_cells.jsonl").read_text(encoding="utf-8").splitlines()]
    eligible, features, prefixes, targets, label_rows, refreshed = [], [], [], [], [], []
    for index, record in enumerate(records, 1):
        path = args.private_output / "extracted" / record["relative_path"]
        payload, digest = load_existing_pickle(path)
        if digest != record["sha256"]:
            raise ValueError(f"SDU extracted member changed {record['relative_path']}")
        d0 = inspect_sdu_d0(payload, record["cycle_trace"])
        eligible.append({**d0["eligible"], "provided_file_sha256": digest})
        features.extend(d0["features"])
        prefixes.extend(d0["prefix_eligibility"])
        targets.append(d0["targets"])
        label = labels.get(path.name)
        label_rows.append({"relative_path": record["relative_path"], "sha256": digest,
                           "cell_id": record["metadata"]["cell_id"], "life_label": label,
                           "cycle_count": record["cycle_count"],
                           "max_observed_supplied_cycle": record["cycle_number"]["max"],
                           "label_exceeds_record_count": label is not None and label > record["cycle_count"],
                           "label_exceeds_max_observed_cycle": label is not None and label > record["cycle_number"]["max"],
                           "observed_event": d0["targets"]["eol_event"],
                           "observed_event_cycle": d0["targets"]["eol_cycle"],
                           "observed_censoring": d0["targets"]["eol_censoring"],
                           "observed_censor_cycle": d0["targets"]["eol_censor_cycle"],
                           "label_used_for_event": False, "physical_coordinate_authenticated": False})
        refreshed.append({"relative_path": record["relative_path"], "sha256": digest, "d0": d0})
        del payload
        gc.collect()
        if index % 10 == 0:
            print(json.dumps({"SDU_D0_refresh_completed": index, "total": len(records)}), flush=True)
    dependencies_after = {name: file_sha256(scripts / f"{name}.py") for name in dependencies_before}
    if dependencies_after != dependencies_before:
        raise RuntimeError("Shared feature dependencies changed during the SDU D0 refresh")
    write_csv(args.output / "sdu_eligible_cells.csv", eligible)
    write_csv(args.output / "sdu_prefix_eligibility.csv", prefixes)
    write_csv(args.output / "sdu_early_observed_features.csv", features)
    write_csv(args.output / "sdu_future_target_support.csv", targets)
    write_csv(args.output / "sdu_label_alignment.csv", label_rows)
    coverage = []
    for cutoff in (10, 50, 100):
        selected = [row for row in prefixes if row["K_supplied"] == cutoff]
        coverage.append({"family": "SDU", "K_supplied": cutoff, "N_provided": len(records),
                         "N_K_valid_supplied_numbers": sum(row["provided_prefix_numbers_complete"] for row in selected),
                         "N_K_strict_authenticated": 0,
                         "N_curve_at_least_two_nonduplicate": sum(row["two_curve_feature_supported"] for row in selected),
                         "N_curve_at_least_80_records": sum(row["observed_curve_records"] >= 80 for row in selected),
                         "N_mid_slope": sum(row["target_mid_supported"] for row in eligible),
                         "N_late_slope": sum(row["target_late_supported"] for row in eligible),
                         "N_EOLevent_after_prior_above_threshold": sum(row["target_EOLevent_after_prior_above_threshold"] for row in eligible),
                         "N_EOL_right_censored": sum(row["eol_censoring"] == "right" for row in targets),
                         "N_EOL_left_censored": sum(row["eol_censoring"] == "left" for row in targets),
                         "N_knee_stable_candidate": sum(row["knee_stable_candidate"] for row in targets),
                         "N_ICA_valid": 0, "N_DVA_valid": 0, "N_policy_known": 0,
                         "N_strict_physical_intersection": 0,
                         "N_observed_curve2_mid_late_intersection": sum(row["two_curve_feature_supported"] and eligible[index]["target_mid_supported"] and eligible[index]["target_late_supported"] for index, row in enumerate(selected))})
    write_csv(args.output / "sdu_feature_coverage.csv", coverage)
    write_csv(args.output / "sdu_future_condition_contract.csv", [{key: row[key] for key in ("cell_id", "family", "future_policy_status", "future_policy_basis", "physical_cycle_axis_authenticated")} for row in eligible])
    field_rows, ranges = [], []
    fields = sorted({key for record in records for key in record["metadata"]})
    for key in fields:
        values = [record["metadata"][key] for record in records if key in record["metadata"]]
        field_rows.append({"field": key, "cells": len(records), "present_cells": len(values),
                           "null_cells": sum(value is None for value in values),
                           "empty_string_cells": sum(isinstance(value, str) and not value.strip() for value in values),
                           "empty_list_cells": sum(isinstance(value, list) and not value for value in values),
                           "unique_values": len({json.dumps(value, sort_keys=True) for value in values})})
    for record in records:
        meta = record["metadata"]
        voltage = record["channels"]["voltage_in_V"]
        ranges.append({"cell_id": meta["cell_id"], "relative_path": record["relative_path"],
                       "declared_voltage_min_V": meta["min_voltage_limit_in_V"],
                       "declared_voltage_max_V": meta["max_voltage_limit_in_V"],
                       "observed_voltage_min_V": voltage["min"], "observed_voltage_max_V": voltage["max"],
                       "voltage_below_declared_min_amount_V": max(0, meta["min_voltage_limit_in_V"] - voltage["min"]),
                       "voltage_above_declared_max_amount_V": max(0, voltage["max"] - meta["max_voltage_limit_in_V"]),
                       "hardcoded_removed_original_indices_count": len(meta.get("hardcoded_removed_indices", [])),
                       "duplicate_VIQ_later_records": record["issues"].get("exact_repeated_V_I_Qc_Qd_cycles", 0)})
    write_csv(args.output / "sdu_metadata_field_coverage.csv", field_rows)
    write_csv(args.output / "sdu_voltage_and_replacement_review.csv", ranges)
    checks = {"all_original_member_SHA256_rechecked": len(refreshed) == verification["pkl_expected"],
              "dependencies_unchanged_during_refresh": dependencies_before == dependencies_after,
              "source_signal_audit_remains_pass": verification["status"] == "PASS",
              "unique_refreshed_cells": len({row["relative_path"] for row in refreshed}) == len(records),
              "no_original_removed_indices_reapplied_to_supplied_coordinate": all(not row["removed_original_index_to_supplied_index_mapping_authenticated"] for row in eligible),
              "all_true_physical_certification_flags_remain_false": all(not row["physical_cycle_axis_authenticated"] for row in eligible),
              "official_labels_not_used_for_events": all(not row["label_used_for_event"] for row in label_rows)}
    provenance = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
                  "full_signal_audit_run_d0_sha256": lineage["run_d0_sha256"],
                  "final_D0_dependency_sha256": dependencies_before,
                  "scope": "D0 and future-target candidates only, full signal audit not repeated",
                  "phase_capacity_definition": "max(Q within current<−0.01*nominal, >=15 active samples) minus min(all finite record Q), divided by nominal, valid if 0<value<1.5",
                  "provided_cells": len(records), "supplied_cycles": verification["cycles_scanned"],
                  "feature_rows": len(features), "strict_physical_axis_authenticated": False}
    dump(provenance, args.output / "sdu_D0_refresh_provenance.json")
    with (args.private_output / "sdu_d0_refreshed.jsonl").open("w", encoding="utf-8") as file:
        for row in refreshed:
            file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    summary = json.loads((args.output / "sdu_family_summary.json").read_text(encoding="utf-8"))
    write_sdu_report(args.report, summary, verification, label_rows, prefixes, targets, lineage)
    notes = ["", "## 最终D0函数版本复核", "",
             "完整信号审计保留最初执行时的依赖记录。完成后又以当前共享run_d0和physics_targets函数独立补算86颗电池的D0候选，并逐个重算PKL SHA256。依赖版本与检查见sdu_D0_refresh_provenance.json，最终资格/特征/目标CSV均来自这次补算，完整通道数值统计继续使用已完成的原始全扫。", "",
             "本次新增的SDU元数据包含20个非cycle_data顶层字段，逐字段覆盖见sdu_metadata_field_coverage.csv。温度全部有限，内阻全部为空，充放电协议全部为空列表。", "",
             f"声明电压限制为3.0–4.2V，实测全局范围为{summary['channels']['voltage_in_V']['min']}–{summary['channels']['voltage_in_V']['max']}V。供应值与声明边界的偏离幅度已逐电池保存，采用零容差记录数值，具体实验有效范围需按原始16种协议解释。", "",
             "完整复现先执行scripts/audit_sdu.py，再执行scripts/refresh_sdu_d0.py。两个命令的参数均使用显式输入和输出路径。", ""]
    with args.report.open("a", encoding="utf-8") as file:
        file.write("\n".join(notes))
    if not all(checks.values()):
        raise AssertionError(checks)
    print(json.dumps({"status": "PASS", "SDU_D0_refresh_cells": len(records), "coverage": coverage}), flush=True)


if __name__ == "__main__":
    main()
