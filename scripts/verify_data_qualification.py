"""Independent table/source checks for completed main D0 and separate SDU D0."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def true(value):
    return value is True or value == "True"


def available(value):
    return value not in (None, "") and math.isfinite(float(value))


def equal_number(left, right):
    if left in (None, "") or right in (None, ""):
        return left in (None, "") and right in (None, "")
    return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--private-cache", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repository
    main_d0 = repo / "results/d0"
    sdu = repo / "results/sdu_audit"
    eligible = read_csv(main_d0 / "eligible_cells.csv")
    targets = read_csv(main_d0 / "future_target_support.csv")
    coverage = read_csv(main_d0 / "feature_coverage_by_family_protocol.csv")
    features = read_csv(main_d0 / "early_observed_features.csv")
    conditions = read_csv(main_d0 / "future_condition_contract.csv")
    source_binding = read_json(main_d0 / "SOURCE_BINDING_ACCEPTANCE.json")
    source_identity = read_csv(main_d0 / "source_identity_alignment.csv")
    raw_targets = read_csv(main_d0 / "HUST_RAW_CAPACITY_TARGET_SUPPORT.csv")
    target_by_id = {row["cell_id"]: row for row in targets}
    identity_by_id = {row["cell_id"].removesuffix(".pkl"): row for row in source_identity}
    cached = {row["eligible"]["cell_id"]: row for row in map(json.loads, (args.private_cache / "d0_cells.jsonl").read_text(encoding="utf-8").splitlines())}
    checks = {}
    checks["main427_unique_cells"] = len(eligible) == len({row["cell_id"] for row in eligible}) == 427
    checks["main_family_counts"] = Counter(row["family"] for row in eligible) == {"HUST": 77, "MATR": 169, "Stanford_2": 181}
    main_ids = {row["cell_id"] for row in eligible}
    checks["main427_targets_and_conditions_match_ids"] = len(targets) == len(conditions) == 427 and set(target_by_id) == main_ids == {row["cell_id"] for row in conditions}
    checks["main_features_three_prefixes_each"] = len(features) == 3 * 427 and len({(row["cell_id"], row["prefix_cutoff_supplied_cycle_number"]) for row in features}) == 3 * 427
    checks["source_binding_pass427_HUST77"] = source_binding["status"] == "PASS" and source_binding["cells"] == 427 and source_binding["official_source_bound_HUST"] == 77
    family_counts = {}
    for family in ("HUST", "MATR", "Stanford_2"):
        rows = [row for row in eligible if row["family"] == family]
        target_rows = [target_by_id[row["cell_id"]] for row in rows]
        computed = {
            "N_provided": len(rows),
            "N_raw_instrument_source": sum(true(row["original_instrument_source_available"]) for row in rows),
            "N_K_valid_supplied_numbers": sum(true(row["prefix_K100_numbers_complete"]) for row in rows),
            "N_K_strict_authenticated": sum(true(row["physical_cycle_axis_authenticated"]) and true(row["prefix_K100_numbers_complete"]) for row in rows),
            "N_curve_observed_supported": sum(int(row["curve_valid_records_K100"]) >= 80 for row in rows),
            "N_mid_slope": sum(available(row["r200_500"]) for row in target_rows),
            "N_late_slope": sum(available(row["r500_1000"]) for row in target_rows),
            "N_EOLevent": sum(true(row["eol_event"]) for row in target_rows),
            "N_EOTime_censored": sum(not true(row["eol_event"]) for row in target_rows),
            "N_knee_stable_candidate": sum(true(row["knee_stable_candidate"]) for row in target_rows),
            "N_knee_event": sum(true(row["knee_event_authenticated"]) for row in rows),
            "N_policy_known": sum(row["future_policy_status"] in ("known_fixed", "known_scheduled") for row in rows),
        }
        computed["N_curve_K_mid_late_source_intersection"] = sum(
            true(row["physical_cycle_axis_authenticated"]) and true(row["prefix_K100_numbers_complete"])
            and int(row["curve_valid_records_K100"]) >= 80
            and available(target_by_id[row["cell_id"]]["r200_500"])
            and available(target_by_id[row["cell_id"]]["r500_1000"]) for row in rows)
        recorded = next(row for row in coverage if row["family"] == family)
        checks[family + "_coverage_recomputed"] = all(int(recorded[key]) == value for key, value in computed.items())
        family_counts[family] = computed
    checks["HUST_physical_K100_76_intersection76"] = family_counts["HUST"]["N_K_strict_authenticated"] == family_counts["HUST"]["N_curve_K_mid_late_source_intersection"] == 76
    checks["MATR_Stanford_strict_axis_zero"] = all(not true(row["physical_cycle_axis_authenticated"]) for row in eligible if row["family"] != "HUST")
    checks["true_knee_event_zero"] = all(not true(row["knee_event_authenticated"]) for row in eligible)
    source_checks = []
    for row in eligible:
        if row["family"] != "HUST":
            continue
        binding = identity_by_id[row["cell_id"]]
        source_checks.append(row["provided_file_sha256"] == binding["processed_file_sha256"] and true(binding["processed_prefix_vi_exact"]) and true(binding["raw_original_cycle_axis_verified"]) and int(binding["raw_first100_cycles_with_complete_charge_discharge"]) == 100)
    checks["HUST_all77_source_hash_axis_VI_phase_bindings"] = len(source_checks) == 77 and all(source_checks)
    special = next(row for row in eligible if row["cell_id"] == "HUST_7-5")
    checks["HUST7_5_has98_records_within_physical100"] = not true(special["prefix_K100_numbers_complete"]) and int(special["prefix_K100_records"]) == 98 and json.loads(identity_by_id["HUST_7-5"]["processed_prefix_cycle_numbers"]) == list(range(3, 103))
    checks["main_targets_numeric_source_is_provided_cache"] = all(equal_number(target_by_id[cell][name], cached[cell]["targets"][name]) for cell in main_ids for name in ("r200_500", "r500_1000", "eol_cycle", "knee_cycle"))
    checks["rawcounter_targets_are_separate77_authenticated"] = len(raw_targets) == 77 and all(true(row["strict_physical_target_authenticated"]) and row["target_coordinate"] == "original_official_cycle_number" and available(row["r200_500"]) and available(row["r500_1000"]) and not true(row["eol_event"]) and row["eol_censoring"] == "right" for row in raw_targets)
    checks["rawcounter_independent_verification_pass"] = read_json(main_d0 / "HUST_RAW_CAPACITY_VERIFICATION.json")["status"] == "PASS"

    summary = read_json(sdu / "sdu_family_summary.json")
    verification = read_json(sdu / "SDU_AUDIT_VERIFICATION.json")
    independent = read_json(sdu / "sdu_independent_verification.json")
    refreshed = read_json(sdu / "sdu_D0_refresh_provenance.json")
    metadata = read_csv(sdu / "sdu_cell_metadata.csv")
    channel_quality = read_csv(sdu / "sdu_channel_quality.csv")
    sdu_eligible = read_csv(sdu / "sdu_eligible_cells.csv")
    sdu_targets = read_csv(sdu / "sdu_future_target_support.csv")
    sdu_prefix = read_csv(sdu / "sdu_prefix_eligibility.csv")
    sdu_coverage = read_csv(sdu / "sdu_feature_coverage.csv")
    sdu_labels = read_csv(sdu / "sdu_label_alignment.csv")
    sdu_members = read_csv(sdu / "sdu_archive_members.csv")
    checks["SDU_source_audit_and_independent_checks_pass"] = verification["status"] == independent["status"] == refreshed["status"] == "PASS" and all(independent["checks"].values()) and all(refreshed["checks"].values())
    checks["SDU86_unique_metadata_and_members"] = len(metadata) == len(sdu_members) == len({row["cell_id"] for row in metadata}) == summary["files"] == 86
    checks["SDU49146_cycle_count_recomputed"] = sum(int(row["cycle_count"]) for row in metadata) == summary["cycles"] == verification["cycles_scanned"] == 49146
    checks["SDU9199_duplicate_fingerprints_recomputed"] = sum(json.loads(row["issues"]).get("exact_repeated_V_I_Qc_Qd_cycles", 0) for row in metadata) == sum(int(row["duplicate_VIQ_records_excluded"]) for row in sdu_eligible) == summary["issues"]["exact_repeated_V_I_Qc_Qd_cycles"] == 9199
    checks["SDU70_life_labels"] = len(sdu_labels) == 86 and sum(available(row["life_label"]) for row in sdu_labels) == verification["life_label_entries"] == 70
    sdu_ids = {row["cell_id"] for row in sdu_eligible}
    checks["SDU86_eligible_and_targets_match"] = len(sdu_eligible) == len(sdu_targets) == 86 and sdu_ids == {row["cell_id"] for row in sdu_targets}
    checks["SDU_separate_from427"] = not (main_ids & sdu_ids) and verification["SDU_excluded_from_frozen_main_cohort"]
    checks["SDU73_K100_and258_prefix_rows"] = len(sdu_prefix) == 258 and sum(true(row["prefix_K100_numbers_complete"]) for row in sdu_eligible) == 73
    checks["SDU36mid24late_recomputed"] = sum(available(row["r200_500"]) for row in sdu_targets) == sum(true(row["target_mid_supported"]) for row in sdu_eligible) == 36 and sum(available(row["r500_1000"]) for row in sdu_targets) == sum(true(row["target_late_supported"]) for row in sdu_eligible) == 24
    censoring = Counter(row["eol_censoring"] for row in sdu_targets)
    checks["SDU77_observed_events9right_censored"] = sum(true(row["eol_event"]) for row in sdu_targets) == 77 and censoring == {"none": 76, "interval": 1, "right": 9}
    checks["SDU_no_official_label_used_for_event"] = all(not true(row["official_label_used_for_event"]) for row in sdu_targets) and all(not true(row["label_used_for_event"]) for row in sdu_labels)
    checks["SDU_no_strict_axis_or_true_ICA_DVA"] = all(not true(row["physical_cycle_axis_authenticated"]) and not true(row["strict_physical_target_authenticated"]) and int(row["N_true_ICA_records"]) == int(row["N_true_DVA_records"]) == 0 for row in sdu_eligible)
    checks["SDU_all86_policy_unknown"] = all(row["future_policy_status"] == "unknown" for row in sdu_eligible)
    channel_sums = defaultdict(Counter)
    for row in channel_quality:
        for field in ("points", "finite_points", "nan_points", "positive_inf_points", "negative_inf_points", "present_cycles", "none_cycles", "absent_cycles"):
            channel_sums[row["channel"]][field] += int(row[field])
    checks["SDU_channel_counts_reaggregate"] = all(summary["channels"][channel][field] == value for channel, values in channel_sums.items() for field, value in values.items())
    checks["SDU_voltage_points73264808"] = channel_sums["voltage_in_V"]["points"] == channel_sums["voltage_in_V"]["finite_points"] == 73264808
    for recorded in sdu_coverage:
        prefix = int(recorded["K_supplied"])
        selected = [row for row in sdu_prefix if int(row["K_supplied"]) == prefix]
        calculated = {"N_provided": 86, "N_K_valid_supplied_numbers": sum(true(row["provided_prefix_numbers_complete"]) for row in selected),
                      "N_curve_at_least_two_nonduplicate": sum(int(row["observed_curve_records"]) >= 2 for row in selected),
                      "N_curve_at_least_80_records": sum(int(row["observed_curve_records"]) >= 80 for row in selected),
                      "N_mid_slope": 36, "N_late_slope": 24, "N_EOLevent_after_prior_above_threshold": 77,
                      "N_EOL_right_censored": 9, "N_EOL_left_censored": 0, "N_strict_physical_intersection": 0,
                      "N_ICA_valid": 0, "N_DVA_valid": 0, "N_policy_known": 0}
        checks[f"SDU_K{prefix}_coverage_recomputed"] = all(int(recorded[key]) == value for key, value in calculated.items())
    report = (repo / "report/SDU_DATA_AUDIT_ZH.md").read_text(encoding="utf-8")
    checks["SDU_report_counts_and_scope_match_tables"] = all(token in report for token in ("86", "49,146", "9199", "70/86", "36颗", "24颗", "Protocol_ID", "真实物理"))
    checks = {key: bool(value) for key, value in checks.items()}
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
              "main_family_counts": family_counts,
              "SDU_counts": {"files": 86, "cycles": 49146, "duplicates": 9199, "labels": 70,
                             "K100_complete": 73, "mid_slope": 36, "late_slope": 24,
                             "observed_events": 77, "right_censored": 9, "true_physical": 0},
              "capacity_source_disclosure": "Main D0 future_target_support retains provided-integral values. HUST_RAW_CAPACITY_TARGET_SUPPORT separately provides raw-counter values for the registered experiment B primary target.",
              "true_knee_event_count": 0,
              "method": "Independent CSV/JSON reaggregation and source binding checks, no 4GB raw redecoding",
              "experiment_B_checked": False,
              "code_sha256": sha(Path(__file__))}
    output = repo / "results/FINAL_DATA_QUALIFICATION_QA.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": len(checks), "failed": [key for key, value in checks.items() if not value]}))
    if not all(checks.values()):
        raise AssertionError("Qualification cross-check failed")


if __name__ == "__main__":
    main()
