"""Bind completed D0 results to independently recovered source identities."""

import argparse
import csv
import json
from pathlib import Path

from run_d0 import aggregate, write_csv, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in (args.private_cache / "d0_cells.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(records) != 427:
        raise RuntimeError("D0 source binding requires all 427 cells")
    with (args.output / "source_identity_alignment.csv").open(encoding="utf-8-sig", newline="") as file:
        identities = {row["cell_id"].removesuffix(".pkl"): row for row in csv.DictReader(file)}
    with (args.output / "policy_scheduled_evidence.csv").open(encoding="utf-8-sig", newline="") as file:
        policies = {row["cell_id"].removesuffix(".pkl"): row for row in csv.DictReader(file)}
    checked = 0
    for record in records:
        cell = record["eligible"]
        cell["knee_event_authenticated"] = False
        cell["knee_label_reason"] = "HINGE_NOT_SPECIFIC_AGAINST_SMOOTH_CURVATURE_CONTROLS"
        if cell["family"] != "HUST":
            continue
        binding = identities[cell["cell_id"]]
        if binding["processed_file_sha256"] != cell["provided_file_sha256"]:
            raise RuntimeError("Original-source binding hash differs from D0 source")
        exact = binding["processed_prefix_vi_exact"] == "True"
        axis = binding["raw_original_cycle_axis_verified"] == "True"
        if not exact or not axis:
            raise RuntimeError("Source identity or V/I numeric replay failed")
        original_count = int(binding["original_cycle_count"])
        removed = 2 if binding["source_declared_skip_first2"] == "True" else 0
        if original_count - removed != cell["provided_cycles"]:
            raise RuntimeError("Full source cycle cardinality does not match supplied records")
        cell.update(physical_cycle_axis_authenticated=True,
                    physical_cycle_source_supported=True,
                    physical_cycle_confidence="high",
                    physical_cycle_evidence="official_archive_hash_raw_cycle_column_and_exact_prefix_VI",
                    original_instrument_source_available=True,
                    original_device_record_id=binding["original_device_record_id"],
                    original_member_sha256=binding["source_member_sha256"],
                    raw_cycle_relative_seconds_available=True,
                    phase_role_supported=binding["raw_first100_complete_axis_and_phases"] == "True",
                    capacity_coordinate="original_cycle_number")
        policy = policies[cell["cell_id"]]
        cell["future_policy_status"] = policy["future_policy_status_candidate"]
        cell["future_policy_basis"] = "official_design_and_observed_stages_final_2C_conflicts_with_metadata_1C"
        record["targets"]["target_coordinate"] = "original_cycle_number"
        record["targets"]["strict_physical_target_authenticated"] = True
        checked += 1
    aggregate(records, args.output)
    write_json(args.output / "SOURCE_BINDING_ACCEPTANCE.json", {
        "status": "PASS", "cells": len(records), "official_source_bound_HUST": checked,
        "HUST_full_cycle_mapping_rule": "original contiguous cycle IDs minus documented first two for 7-5",
        "provided_prefix_VI": "all first 100 retained records matched raw V and mA/1000 exactly",
        "knee_true_event_labels": 0,
        "capacity_target_version": "active-discharge maximum minus finite record baseline / nominal",
        "superseded_cache_versions": ["d0_cache", "d0_cache_v2"],
        "current_cache_version": "d0_cache_v3",
        "pre_model_score_implementation_repairs": [
            "Reject any excessive within-segment capacity reset",
            "Preserve prefix extraction when suffix record ordering changes",
            "Bind resumption to scanner, target code, audited data and settings hashes",
            "Restrict capacity maximum to matching current phase"
        ]
    })
    write_csv(args.output / "future_condition_contract.csv", [
        {key: row.get(key) for key in ("cell_id", "family", "future_policy_status", "future_policy_basis",
          "temperature_matching_authenticated", "physical_cycle_axis_authenticated", "original_instrument_source_available",
          "original_device_record_id", "raw_cycle_relative_seconds_available")} for row in (r["eligible"] for r in records)
    ])
    print(json.dumps({"status": "PASS", "bound_HUST": checked, "provided_cells": len(records)}))


if __name__ == "__main__":
    main()
