"""Independent HUST two-view data QA, with symbolic raw parsing and no fitting."""

from __future__ import annotations

import argparse
import collections
import gc
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(REPOSITORY / "scripts/unified_matr"))
from scripts.hust_safe_pickle import dataframe_columns, parse_pickle_data
from scripts.hust_raw_v2.measurements import VIEWS, measure_view
from physics_features import PhysicsConfig, extract_prefix_features
from models.unified_matr import UnifiedMATRModel
import train_seed42 as model_entrypoint


ARMS = ("M0", "M1", "M2", "M3", "M3-NL", "M3-NP", "M3-NR")
DISCHARGE = tuple(f"Constant current discharge_{index}" for index in range(4))


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def same_numeric(left, right, atol=1e-10, rtol=1e-7):
    first, second = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    return first.shape == second.shape and bool(np.allclose(first, second, atol=atol, rtol=rtol, equal_nan=True))


class Audit:
    def __init__(self):
        self.checks = []

    def check(self, name, passed, detail=None):
        self.checks.append({"id": name, "status": "PASS" if bool(passed) else "FAIL", "detail": json_ready(detail)})
        return bool(passed)


def source_blocks(columns):
    cursor = 0
    output = []
    for phase, members in itertools.groupby(columns["Status"]):
        points = sum(1 for _ in members)
        output.append((str(phase), cursor, cursor + points))
        cursor += points
    return output


def independent_full_capacity(columns):
    phases = [block for block in source_blocks(columns) if block[0] in DISCHARGE]
    if tuple(block[0] for block in phases) != DISCHARGE:
        return {"valid": False, "capacity_Ah": np.nan, "origin_mAh": np.nan, "origin_point": None}
    if any(left[2] != right[1] for left, right in zip(phases[:-1], phases[1:])):
        return {"valid": False, "capacity_Ah": np.nan, "origin_mAh": np.nan, "origin_point": None}
    first, stop = phases[0][1], phases[-1][2]
    charge = np.asarray(columns["Capacity (mAh)"][first:stop], dtype=float)
    current = np.asarray(columns["Current (mA)"][first:stop], dtype=float) / 1000
    amount = float(charge[0] - charge[-1]) / 1000
    valid = np.isfinite(charge).all() and np.isfinite(current).all() and np.all(current < 0)
    valid = valid and np.all(np.diff(charge) <= 1e-6) and np.all(np.diff(charge) >= -110)
    valid = valid and 0 < amount < 1.65
    return {"valid": bool(valid), "capacity_Ah": amount if valid else np.nan,
            "origin_mAh": float(charge[0]), "origin_point": first}


def independent_view(columns, original_cycle, definition):
    """Derive the two Q(V) views independently from source phase and units."""
    grid = np.linspace(definition.voltage_min_V, definition.voltage_max_V, 64)
    output = {"valid": False, "q_absolute": np.full(64, np.nan), "q_native_Ah": np.full(64, np.nan),
              "raw_viq": np.full((64, 3), np.nan), "origin_mAh": np.nan, "origin_point": None}
    selected = [block for block in source_blocks(columns) if block[0] == definition.phase]
    if len(selected) != 1:
        return output
    _, start, stop = selected[0]
    voltage = np.asarray(columns["Voltage (V)"][start:stop], dtype=float)
    current = np.asarray(columns["Current (mA)"][start:stop], dtype=float) / 1000
    charge = np.asarray(columns["Capacity (mAh)"][start:stop], dtype=float)
    cycle = np.asarray(columns["Cycle number"][start:stop])
    supported = np.isfinite(voltage).all() and np.isfinite(current).all() and np.isfinite(charge).all()
    supported = supported and bool(np.all(cycle == original_cycle))
    direction = 1 if definition.direction == "charge" else -1
    supported = supported and bool(np.all(direction * np.diff(charge) >= -1e-6)) and bool(np.all(np.abs(np.diff(charge)) <= 110))
    if definition.direction == "charge":
        output.update(origin_mAh=float(charge[0]), origin_point=start)
        supported = supported and abs(charge[0]) <= 1e-6
        absolute = charge / 1000
    else:
        discharge = independent_full_capacity(columns)
        output.update(origin_mAh=discharge["origin_mAh"], origin_point=discharge["origin_point"])
        supported = supported and discharge["valid"]
        absolute = (discharge["origin_mAh"] - charge) / 1000
    native = (voltage >= grid[0]) & (voltage <= grid[-1])
    supported = supported and voltage.min() <= grid[0] and voltage.max() >= grid[-1]
    supported = supported and int(native.sum()) >= definition.minimum_window_native_points
    supported = supported and len(set(voltage[native])) >= definition.minimum_distinct_voltage_points
    positions = np.flatnonzero(native)
    if len(positions) > 1:
        changes = np.diff(voltage[positions[0]:positions[-1] + 1])
        supported = supported and bool(np.all(-direction * changes <= definition.maximum_voltage_reversal_V))
    retained = [0]
    for index, potential in enumerate(voltage[1:], 1):
        if direction * (potential - voltage[retained[-1]]) > 1e-8:
            retained.append(index)
    if definition.direction == "discharge":
        retained.reverse()
    retained = np.asarray(retained)
    increasing = voltage[retained]
    native_retained = (increasing >= grid[0]) & (increasing <= grid[-1])
    supported = supported and int(native_retained.sum()) >= definition.minimum_window_native_points
    supported = supported and len(set(increasing[native_retained])) >= definition.minimum_distinct_voltage_points
    supported = supported and increasing[0] <= grid[0] and increasing[-1] >= grid[-1]
    if not supported:
        return output
    # Include native history and both actual interpolation brackets. A later
    # excitation change outside this support cannot alter the earlier Q(V).
    lower = int(np.searchsorted(increasing, grid[0], side="right") - 1)
    upper = int(np.searchsorted(increasing, grid[-1], side="left"))
    history_stop = int(np.max(retained[lower:upper + 1])) + 1
    if not np.all(np.abs(current[:history_stop] - definition.current_A) <= definition.current_tolerance_A):
        return output
    q = np.interp(grid, increasing, absolute[retained])
    if np.ptp(q) < definition.minimum_window_charge_Ah:
        return output
    output.update(valid=True, q_absolute=q, q_native_Ah=np.interp(grid, increasing, charge[retained] / 1000),
                  raw_viq=np.column_stack((grid, np.interp(grid, increasing, current[retained]), q)))
    return output


def synthetic_controls(audit):
    count = [101, 12, 20, 20, 20, 131]
    phases = ["Constant current charge", "Constant current-constant voltage charge", *DISCHARGE]
    status = np.concatenate([np.repeat(phase, size) for phase, size in zip(phases, count)])
    voltage = np.concatenate([np.linspace(3.0, 3.6, count[0]), np.full(count[1], 3.6),
                              np.linspace(3.6, 3.5, count[2]), np.linspace(3.5, 3.4, count[3]),
                              np.linspace(3.4, 3.3, count[4]), np.linspace(3.3, 2.0, count[5])])
    current = np.concatenate([np.full(size, value) for size, value in zip(count, (5500, 1100, -3300, -2200, -1100, -2200))])
    charge = np.concatenate([np.linspace(0, 800, count[0]), np.linspace(800, 1000, count[1]),
                             np.linspace(1000, 900, count[2]), np.linspace(900, 800, count[3]),
                             np.linspace(800, 600, count[4]), np.linspace(600, 0, count[5])])
    columns = {"Status": status, "Voltage (V)": voltage, "Current (mA)": current,
               "Capacity (mAh)": charge, "Time (s)": np.arange(len(status)) * 5,
               "Cycle number": np.ones(len(status), dtype=int)}
    full = independent_full_capacity(columns)
    audit.check("synthetic_full_net_capacity_units_and_origin", full["valid"] and full["capacity_Ah"] == 1.0 and full["origin_mAh"] == 1000)
    for view in VIEWS:
        independent = independent_view(columns, 1, view)
        canonical, arrays = measure_view(columns, 1, view)
        audit.check("synthetic_independent_view_" + view.name, independent["valid"] and canonical["curve_valid"]
                    and same_numeric(independent["q_absolute"], arrays["q_absolute"])
                    and same_numeric(independent["q_native_Ah"], arrays["q_native_Ah"]))
    changed = {key: value.copy() for key, value in columns.items()}
    changed["Capacity (mAh)"][:count[0]] += 1
    audit.check("synthetic_charge_origin_failure", not independent_view(changed, 1, VIEWS[1])["valid"])
    changed = {key: value.copy() for key, value in columns.items()}
    changed["Current (mA)"][-110] = -3000
    audit.check("synthetic_discharge_current_failure", not independent_view(changed, 1, VIEWS[0])["valid"])
    changed = {key: value.copy() for key, value in columns.items()}
    changed["Voltage (V)"][-count[-1]:] -= 0.3
    audit.check("synthetic_no_voltage_extrapolation", not independent_view(changed, 1, VIEWS[0])["valid"])
    changed = {key: value.copy() for key, value in columns.items()}
    changed["Current (mA)"][90:count[0]] = 3000
    late = independent_view(changed, 1, VIEWS[1])
    baseline = independent_view(columns, 1, VIEWS[1])
    audit.check("synthetic_later_outside_window_current_isolated", late["valid"] and same_numeric(late["q_absolute"], baseline["q_absolute"]))
    changed["Current (mA)"][2] = 3000
    audit.check("synthetic_earlier_history_current_conflict_masks_view", not independent_view(changed, 1, VIEWS[1])["valid"])
    changed = {key: value.copy() for key, value in columns.items()}
    changed["Current (mA)"][45] = 3000
    audit.check("synthetic_interpolation_support_current_conflict_masks_view", not independent_view(changed, 1, VIEWS[1])["valid"])
    return {"scope": "synthetic source-unit/origin/phase/interpolation controls", "scientific_data_read": False}


def verify_sources(audit, root, registration):
    identity_path = REPOSITORY / "results/hust_raw_idea_audit_v2/CURRENT_ARCHIVE_IDENTITY.json"
    identity = read_json(identity_path)
    receipt_path = REPOSITORY.parent / "private/raw_sources/HUST/SOURCE_RECOVERY_RECEIPT.json"
    recovery = read_json(receipt_path)
    audit.check("current_outer_inner_and_official_source_SHA", sha256(Path(identity["outer_path"])) == identity["outer_sha256"]
                and sha256(Path(identity["replay_inner_cache_path"])) == identity["inner_sha256"] == recovery["official_sha256"] == recovery["actual_sha256"]
                and registration["source_outer_sha256"] == identity["outer_sha256"] and registration["source_inner_sha256"] == identity["inner_sha256"])
    checks = []
    for member in recovery["members"]:
        path = Path(member["path"])
        checks.append(path.stat().st_size == member["bytes"] and sha256(path) == member["sha256"])
    audit.check("all77_original_member_bytes_and_SHA", len(checks) == 77 and all(checks), {"members": len(checks)})
    audit.check("source_recovery_and_identity_bound_to_registration", sha256(receipt_path) == registration["source_recovery_receipt_sha256"]
                and sha256(identity_path) == registration["source_archive_identity_sha256"])
    for source in registration["code"]:
        audit.check("processing_source_" + source["path"], sha256(REPOSITORY / source["path"]) == source["sha256"])
    support = read_json(root / "prefix_support/PREFIX_SUPPORT_COMPLETE.json")
    for source in support["files"]:
        audit.check("prefix_audit_artifact_" + source["path"], sha256(root / "prefix_support" / source["path"]) == source["sha256"])
    audit.check("prefix_support_does_not_consume_future", support["future_capacity_values_used"] == 0 and support["future_prediction_results_used"] == 0
                and support["source_cells"] == 77 and support["original_prefix_cycles"] == 7700)
    matr_freeze = read_json(REPOSITORY / "data/unified_matr_seed42/DATA_FREEZE.json")
    old_unchanged = all(sha256(REPOSITORY / "data/unified_matr_seed42" / row["path"]) == row["sha256"] for row in matr_freeze["files"])
    audit.check("existing44_frozen_artifacts_unchanged", old_unchanged)
    return recovery


def verify_targets(audit, root, cells, trajectory, future, manifest):
    columns = ["cell_uid", "original_cycle_id"]
    audit.check("all77_source_axes_and_unique_rows", cells.cell_uid.nunique() == len(cells) == 77
                and not trajectory.duplicated(columns).any()
                and all(np.array_equal(frame.original_cycle_id, np.arange(1, len(frame) + 1))
                        for _, frame in trajectory.sort_values(columns).groupby("cell_uid", sort=False)))
    audit.check("all_actual_future_source_records_retained", future.equals(trajectory.loc[trajectory.original_cycle_id > 100].reset_index(drop=True)))
    quality_path = REPOSITORY.parent / "private/raw_sources/HUST_IDEA_V2/all_cycle_quality.csv"
    quality = pd.read_csv(quality_path)
    expected = quality.rename(columns={"cell_id": "cell_uid", "cycle": "original_cycle_id"}).sort_values(columns).reset_index(drop=True)
    actual = trajectory.sort_values(columns).reset_index(drop=True)
    independent = expected.net_discharge_mAh.to_numpy(float) / 1000
    audit.check("all146122_source_net_capacity_values_independently_match", actual[columns].equals(expected[columns])
                and len(actual) == 146122 and same_numeric(actual.capacity_Ah, independent, atol=1e-12, rtol=1e-12)
                and actual.capacity_valid.all() and actual.source_cycle_column_matches_key.all(),
                {"source_cycle_quality_sha256": sha256(quality_path), "capacity_rows": len(actual)})
    audit.check("capacity_units_and_nominal_SOH", same_numeric(trajectory.soh_nominal, trajectory.capacity_Ah / 1.1, atol=1e-12)
                and same_numeric(trajectory.capacity_Ah, (trajectory.source_capacity_origin_mAh - trajectory.source_capacity_end_mAh) / 1000, atol=1e-12)
                and trajectory.capacity_Ah.between(0, 1.65, inclusive="neither").all())
    counts = collections.Counter()
    targets_correct = []
    for cell in cells.itertuples(index=False):
        rows = trajectory.loc[(trajectory.cell_uid == cell.cell_uid) & trajectory.capacity_valid].sort_values("original_cycle_id")
        lookup = rows.set_index("original_cycle_id").soh_nominal.to_dict()
        early_anchor = np.median([value for cycle, value in lookup.items() if 96 <= cycle <= 100])
        correct = same_numeric(cell.anchor_soh100, early_anchor, atol=1e-7, rtol=1e-6)
        correct = correct and cell.source_cycle_count == len(rows) and cell.original_cycle_last == rows.original_cycle_id.iloc[-1]
        correct = correct and cell.last_observed_valid_cycle == rows.original_cycle_id.iloc[-1]
        correct = correct and cell.future_observed_capacity_count == int(rows.original_cycle_id.gt(100).sum())
        events = rows.loc[rows.soh_nominal <= 0.8]
        event = len(events) > 0
        expected_event = events.original_cycle_id.iloc[0] if event else np.nan
        correct = correct and bool(cell.eol80_event_observed) == event and bool(cell.right_censored) == (not event)
        correct = correct and same_numeric(cell.eol80_first_observed_crossing_cycle, expected_event)
        if not event:
            correct = correct and cell.eol80_censoring == "right" and cell.eol80_censor_cycle == rows.original_cycle_id.iloc[-1]
        counts["observed_eol80_events"] += int(event)
        for left, right in ((200, 500), (500, 1000)):
            name = f"r{left}_{right}"
            supported = left in lookup and right in lookup
            target = (lookup[left] - lookup[right]) / (right - left) if supported else np.nan
            correct = correct and bool(getattr(cell, name + "_supported")) == supported and same_numeric(getattr(cell, name), target, atol=1e-12)
            counts[name] += int(supported)
            start = [value for cycle, value in lookup.items() if abs(cycle - left) <= 5]
            end = [value for cycle, value in lookup.items() if abs(cycle - right) <= 5]
            audit_supported = len(start) >= 5 and len(end) >= 5
            audit_rate = (np.median(start) - np.median(end)) / (right - left) if audit_supported else np.nan
            correct = correct and bool(getattr(cell, name + "_window5_audit_supported")) == audit_supported
            correct = correct and same_numeric(getattr(cell, name + "_window5_audit"), audit_rate, atol=1e-12)
        targets_correct.append(correct)
    audit.check("all77_exact_rates_EOL_and_prefix_capacity_anchors", all(targets_correct), dict(counts))
    audit.check("future_support_and_manifest_denominators", manifest["future_observed_points"] == int(future.capacity_valid.sum())
                and manifest["observed_eol80_events"] == counts["observed_eol80_events"]
                and all(manifest["rate_support"][name] == counts[name] for name in ("r200_500", "r500_1000")),
                {"future_rows": len(future), "valid_future": int(future.capacity_valid.sum()), **dict(counts)})
    return {"capacity_rows": len(trajectory), "future_rows": len(future), "target_support": dict(counts)}


def verify_view(audit, root, cells, manifest, view):
    location = root / "views" / view.name
    with np.load(location / "early_inputs.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    qualification = pd.read_parquet(location / "view_qualification.parquet")
    diagnostic = pd.read_parquet(location / "feature_diagnostics.parquet")
    name = view.name
    identifiers = cells.cell_uid.tolist()
    audit.check(name + "/all77_identities_and7700_rows", arrays["cell_uid"].astype(str).tolist() == identifiers == qualification.cell_uid.tolist()
                and len(diagnostic) == 7700 and not diagnostic.duplicated(["cell_uid", "original_cycle_id"]).any())
    geometry = {"raw_viq": (77, 100, 64, 3), "reference_viq": (77, 100, 64, 3), "point_mask": (77, 100, 64),
                "cycle_ids": (77, 100), "q_absolute": (77, 100, 64), "q_native_Ah": (77, 100, 64),
                "global_features": (77, 100, 20), "global_mask": (77, 100, 20), "local_residual": (77, 100, 64),
                "local_mask": (77, 100, 64), "local_features": (77, 100, 8, 26), "local_feature_mask": (77, 100, 8, 26),
                "fit_weights": (77, 100, 64), "policy": (77, 12), "policy_mask": (77, 12)}
    mask_keys = ("point_mask", "policy_mask", "global_mask", "local_mask", "local_feature_mask", "curve_valid_mask", "decomposition_valid", "input_eligible", "reference_cycle_mask")
    safe = all(arrays[key].shape == shape for key, shape in geometry.items()) and all(arrays[key].dtype == np.bool_ for key in mask_keys)
    if not audit.check(name + "/exact_geometry_and_boolean_masks", safe):
        raise ValueError("Unsafe mask geometry or dtype")
    valid = arrays["curve_valid_mask"]
    grid = np.linspace(view.voltage_min_V, view.voltage_max_V, 64)
    audit.check(name + "/original_prefix_axis_and_voltage_grid", np.array_equal(arrays["cycle_ids"], np.tile(np.arange(1, 101), (77, 1)))
                and np.array_equal(arrays["voltage_grid_V"], grid))
    audit.check(name + "/raw_VIQ_and_absolute_native_Q_coordinate", np.array_equal(arrays["point_mask"], np.broadcast_to(valid[:, :, None], (77, 100, 64)))
                and np.array_equal(np.isfinite(arrays["q_absolute"]).all(axis=-1), valid)
                and same_numeric(arrays["raw_viq"][..., 2], arrays["q_absolute"].astype(np.float32), atol=0, rtol=0)
                and same_numeric(arrays["raw_viq"][..., 0][arrays["point_mask"]], np.broadcast_to(grid, (77, 100, 64))[arrays["point_mask"]], atol=1e-6)
                and same_numeric(arrays["q_shape"], arrays["q_absolute"] - arrays["q_absolute"][:, :, -1, None], atol=1e-12))
    if view.direction == "discharge":
        absolute_origin = arrays["q_origin_capacity_mAh"][:, :, None] / 1000 - arrays["q_native_Ah"]
    else:
        absolute_origin = arrays["q_native_Ah"]
    audit.check(name + "/mAh_to_Ah_origin_identity", same_numeric(arrays["q_absolute"], absolute_origin, atol=1e-12)
                and (view.direction != "charge" or np.all(np.abs(arrays["q_origin_capacity_mAh"][valid]) <= 1e-6)))
    for value, mask in (("global_features", "global_mask"), ("local_residual", "local_mask"), ("local_features", "local_feature_mask")):
        audit.check(name + "/finite_support/" + value, np.array_equal(np.isfinite(arrays[value]), arrays[mask]))
    support = pd.read_csv(root / "prefix_support/prefix_view_support.csv")
    selected = support.loc[support.view == name].set_index(["cell_uid", "original_cycle_id"])
    matching = []
    physics = PhysicsConfig(voltage_min=view.voltage_min_V, voltage_max=view.voltage_max_V)
    for index, uid in enumerate(identifiers):
        rows = diagnostic.loc[diagnostic.cell_uid == uid].sort_values("original_cycle_id")
        original = selected.loc[uid].sort_index()
        reference = np.flatnonzero(valid[index, :10]) + 1
        rebuilt = extract_prefix_features(arrays["q_absolute"][index], valid[index], np.arange(1, 101), grid, physics)
        fields = ("global_features", "global_mask", "local_residual", "local_mask", "local_features", "local_feature_mask", "fit_weights",
                  "reference_q", "reference_derivative_Ah_per_V", "decomposition_valid", "anchor_observed_cycles", "anchor_support",
                  "anchor_global_features", "anchor_global_mask", "anchor_local_features", "anchor_local_mask")
        feature_equal = all(same_numeric(arrays[field][index], rebuilt[field], atol=1e-7, rtol=2e-6) for field in fields)
        reference_correct = np.array_equal(np.flatnonzero(arrays["reference_cycle_mask"][index]) + 1, reference)
        scale = np.median([np.quantile(arrays["q_absolute"][index, cycle - 1], .95) - np.quantile(arrays["q_absolute"][index, cycle - 1], .05) for cycle in reference]) if len(reference) else np.nan
        qrow = qualification.iloc[index]
        eligibility = valid[index].any() and len(reference) >= 5 and rebuilt["diagnostics"]["reference_valid"] and rebuilt["decomposition_valid"].any()
        eligibility = eligibility and np.isfinite(scale) and scale > 1e-8 and np.isfinite(cells.iloc[index].anchor_soh100)
        equal = feature_equal and reference_correct and bool(arrays["input_eligible"][index]) == bool(eligibility) == bool(qrow.input_eligible)
        equal = equal and same_numeric(arrays["anchor_soh100"][index], cells.iloc[index].anchor_soh100, atol=1e-7)
        equal = equal and qrow.reference_curve_count == len(reference) and qrow.prefix_curve_valid_count == valid[index].sum()
        equal = equal and np.array_equal(original.curve_valid.to_numpy(bool), valid[index])
        equal = equal and np.array_equal(rows.curve_valid.to_numpy(bool), valid[index])
        equal = equal and np.array_equal(rows.decomposition_valid.to_numpy(bool), arrays["decomposition_valid"][index])
        equal = equal and same_numeric(rows.global_a_Ah, arrays["global_features"][index, :, 0], atol=1e-7)
        equal = equal and same_numeric(rows.global_b_V, arrays["global_features"][index, :, 1], atol=1e-7)
        expected_reference = arrays["raw_viq"][index].astype(float)
        if np.isfinite(scale) and scale > 1e-8:
            expected_reference[..., 1:] /= scale
        equal = equal and same_numeric(arrays["reference_viq"][index], expected_reference, atol=5e-6, rtol=2e-6)
        projection = rows[["weighted_constant_projection", "weighted_derivative_projection"]].to_numpy(float)
        finite_projection = projection[np.isfinite(projection)]
        equal = equal and (not len(finite_projection) or np.max(np.abs(finite_projection)) < 1e-6)
        reconstruction = rows.reconstruction_max_absolute_error_Ah.dropna().to_numpy(float)
        equal = equal and (not len(reconstruction) or np.max(np.abs(reconstruction)) < 1e-10)
        matching.append(equal)
    audit.check(name + "/all77_physics_reconstruction_reference_mask_and_prefix_qualification", all(matching),
                {"cells": len(matching), "eligible": int(arrays["input_eligible"].sum()), "valid_curves": int(valid.sum()),
                 "eligibility_uses_view_endcurve": False, "coverage80_is_descriptive": True})
    audit.check(name + "/actual_native_interpolation_support", selected.loc[selected.curve_valid, "interpolation_native_points"].ge(15).all()
                and selected.loc[selected.curve_valid, "interpolation_distinct_voltage_points"].ge(12).all()
                and selected.loc[selected.curve_valid, "waveform_prefix_current_max_deviation_A"].le(.055).all())
    summary = manifest["views"][name]
    audit.check(name + "/manifest_and_independent_view_denominators", summary["valid_curves"] == int(valid.sum())
                and summary["input_eligible_cells"] == int(arrays["input_eligible"].sum())
                and np.array_equal(cells["input_eligible_" + name].to_numpy(bool), arrays["input_eligible"]))
    return arrays, qualification


def model_forward_checks(audit, arrays, view_name, device):
    rows = np.flatnonzero(arrays["input_eligible"])
    inputs = {key: arrays[key] for key in model_entrypoint.INPUT_KEYS}
    forbidden = {"rul", "dq", "source_rul", "source_dq", "future_soh", "eol80", "last_observed_valid_cycle"}
    audit.check(view_name + "/normal_input_keys_exclude_source_labels", not forbidden & inputs.keys())
    records = []
    for arm in ARMS:
        model = UnifiedMATRModel(arm, policy_dim=12, global_dim=20, local_feature_dim=26, seed=42).to(device)
        prediction, trend = model_entrypoint.predictions(model, inputs, rows, np.asarray([101, 200, 500, 1000], np.float32), device, 8)
        valid = prediction.shape == trend.shape == (len(rows), 4) and np.isfinite(prediction).all() and np.isfinite(trend).all()
        valid = valid and np.min(prediction) >= 0 and np.max(prediction) <= 1.2 + 1e-6
        audit.check(view_name + "/normal_model_forward/" + arm, valid, {"seed": 42, "eligible_cells": len(rows), "query_points": 4,
                    "optimizer_updates": 0, "model_fits": 0, "performance_scoring": False})
        records.append({"view": view_name, "arm": arm, "input_cells": len(rows), "normal_entrypoint_forward_pass": bool(valid)})
        del model
    return records


def serialized_precision_checks(audit, arrays, diagnostic, view_name):
    eps32 = np.finfo(np.float32).eps
    maximum_error = maximum_bound = maximum_ratio = maximum_projection = maximum_residual_rounding = 0.0
    rows_checked = 0
    for index, uid in enumerate(arrays["cell_uid"].astype(str)):
        rows = diagnostic.loc[diagnostic.cell_uid == uid].sort_values("original_cycle_id")
        valid = arrays["decomposition_valid"][index]
        if not valid.any():
            continue
        derivative = arrays["reference_derivative_Ah_per_V"][index].astype(float)
        delta = arrays["delta_q"][index, valid].astype(float)
        a = arrays["global_features"][index, valid, 0].astype(float)[:, None]
        b = arrays["global_features"][index, valid, 1].astype(float)[:, None]
        residual = arrays["local_residual"][index, valid].astype(float)
        error = np.abs(delta - a - b * derivative - residual)
        bound = 2 * eps32 * (np.abs(a) + np.abs(b * derivative) + np.abs(residual)) + 1e-12
        maximum_error = max(maximum_error, float(error.max()))
        maximum_bound = max(maximum_bound, float(bound.max()))
        maximum_ratio = max(maximum_ratio, float((error / bound).max()))
        a_high = rows.global_a_Ah.to_numpy(float)[valid, None]
        b_high = rows.global_b_V.to_numpy(float)[valid, None]
        high_residual = delta - a_high - b_high * derivative
        maximum_residual_rounding = max(maximum_residual_rounding, float(np.abs(residual - high_residual).max()))
        basis = np.column_stack((np.ones(64), derivative / np.sqrt(np.mean(derivative**2))))
        weights = arrays["fit_weights"][index, valid].astype(float)
        numerator = np.einsum("nk,nk,kj->nj", weights, residual, basis)
        residual_norm = np.maximum(np.sqrt(np.sum(weights * residual**2, axis=1)), 1e-6 * np.sqrt(weights.sum(axis=1)))
        basis_norm = np.sqrt(np.einsum("nk,kj->nj", weights, basis**2))
        projection = numerator / (basis_norm * residual_norm[:, None])
        maximum_projection = max(maximum_projection, float(np.max(np.abs(projection))))
        rows_checked += int(valid.sum())
    audit.check(view_name + "/serialized_FP32_decomposition_roundoff_bound", maximum_ratio <= 1,
                {"checked_curves": rows_checked, "maximum_reconstruction_error_Ah": maximum_error,
                 "maximum_coordinate_roundoff_bound_Ah": maximum_bound, "maximum_error_over_its_coordinate_bound": maximum_ratio,
                 "bound": "2*eps32*(abs(a32)+abs(b32*dref64)+abs(r32))+1e-12 Ah",
                 "maximum_r32_error_vs_high_precision_ab_reconstruction_Ah": maximum_residual_rounding})
    audit.check(view_name + "/serialized_FP32_residual_weighted_projection", maximum_projection <= 8 * eps32,
                {"maximum_normalized_weighted_projection": maximum_projection, "FP32_tolerance": 8 * eps32,
                 "precision_scope": "Actual serialized residual32, reference derivative64 and final IRLS weights64"})


def replay_raw_prefixes(audit, recovery, all_arrays, trajectory):
    selected_ids = ("HUST_1-5", "HUST_7-5", "HUST_7-7")
    receipts = []
    for uid in selected_ids:
        member = next(row for row in recovery["members"] if "HUST_" + Path(row["path"]).stem == uid)
        path = Path(member["path"])
        graph, globals_seen = parse_pickle_data(path.read_bytes())
        data = graph[path.stem]["data"]
        prefix = {cycle: dataframe_columns(data[cycle]) for cycle in range(1, 101)}
        cap_expected = []
        for cycle in sorted(data):
            columns = prefix[cycle] if cycle <= 100 else dataframe_columns(data[cycle])
            cap_expected.append(independent_full_capacity(columns)["capacity_Ah"])
        saved_cap = trajectory.loc[trajectory.cell_uid == uid].sort_values("original_cycle_id")
        audit.check("raw_full_capacity_replay/" + uid, same_numeric(saved_cap.capacity_Ah, cap_expected, atol=1e-12, rtol=1e-12),
                    {"source_cycles": len(data), "source_labels_used": False})
        for view in VIEWS:
            arrays = all_arrays[view.name]
            index = arrays["cell_uid"].astype(str).tolist().index(uid)
            comparisons = []
            for cycle, columns in prefix.items():
                derived = independent_view(columns, cycle, view)
                good = derived["valid"] == bool(arrays["curve_valid_mask"][index, cycle - 1])
                good = good and same_numeric(arrays["q_absolute"][index, cycle - 1], derived["q_absolute"], atol=1e-12)
                good = good and same_numeric(arrays["q_native_Ah"][index, cycle - 1], derived["q_native_Ah"], atol=1e-12)
                good = good and same_numeric(arrays["raw_viq"][index, cycle - 1], derived["raw_viq"].astype(np.float32), atol=0, rtol=0)
                comparisons.append(good)
            audit.check("independent_raw_phase_interpolation/" + uid + "/" + view.name, all(comparisons),
                        {"original_prefix_cycles": 100, "safe_symbolic_globals": len(globals_seen), "global_constructor_invocations": 0})
            valid = arrays["curve_valid_mask"][index]
            q = arrays["q_absolute"][index]
            config = PhysicsConfig(voltage_min=view.voltage_min_V, voltage_max=view.voltage_max_V)
            baseline = extract_prefix_features(q, valid, np.arange(1, 101), arrays["voltage_grid_V"], config)
            polluted = extract_prefix_features(np.vstack((q, np.full((20, 64), 1e20))), np.r_[valid, np.ones(20, bool)],
                                               np.arange(1, 121), arrays["voltage_grid_V"], config)
            fields = ("global_features", "global_mask", "local_residual", "local_mask", "local_features", "local_feature_mask", "fit_weights")
            audit.check("future_suffix_perturbation/" + uid + "/" + view.name,
                        all(same_numeric(baseline[field], polluted[field], atol=1e-10, rtol=1e-10) for field in fields),
                        {"added_future_curves": 20, "future_value": 1e20, "source_rul_or_dq_entered": False})
        # Direct dependencies are checked on a true mixed-current charge cycle.
        if uid == "HUST_7-7":
            independent = independent_view(prefix[2], 2, VIEWS[1])
            audit.check("actual_7p7_cycle2_later_charge_change_does_not_mask_window", independent["valid"])
        if uid in ("HUST_1-5", "HUST_7-5"):
            audit.check("actual_missing_D3_upper_bracket_is_masked/" + uid, not independent_view(prefix[1], 1, VIEWS[0])["valid"])
        receipts.append({"cell_uid": uid, "source_sha256": member["sha256"], "prefix_cycles": 100, "all_source_capacity_cycles": len(cap_expected),
                         "symbolic_globals_never_executed": globals_seen})
        del data, graph, prefix
        gc.collect()
    return receipts


def verify_dataset(audit, root, device):
    manifest = read_json(root / "dataset_manifest.json")
    registration = read_json(root / "PROCESSING_REGISTRATION.json")
    complete = read_json(root / "DATASET_COMPLETE.json")
    config = read_json(REPOSITORY / "configs/hust_raw_idea_v2.json")
    audit.check("processing_registration_and_fixed_views", complete["registration_sha256"] == sha256(root / "PROCESSING_REGISTRATION.json")
                and manifest["registration_sha256"] == sha256(root / "PROCESSING_REGISTRATION.json") and registration["config"] == config
                and registration["config_sha256"] == sha256(REPOSITORY / "configs/hust_raw_idea_v2.json"))
    for entry in complete["files"]:
        path = root / entry["path"]
        audit.check("derived_file/" + entry["path"], path.stat().st_size == entry["bytes"] and sha256(path) == entry["sha256"])
    recovery = verify_sources(audit, root, registration)
    cells = pd.read_parquet(root / "cells.parquet")
    trajectory = pd.read_parquet(root / "source_capacity_trajectory.parquet")
    future = pd.read_parquet(root / "future_soh.parquet")
    target_summary = verify_targets(audit, root, cells, trajectory, future, manifest)
    expected_ids = ["HUST_" + Path(member["path"]).stem for member in recovery["members"]]
    audit.check("all77_source_identities_match_cells", cells.cell_uid.tolist() == expected_ids and cells.historical_exposure.all())
    flags = ("source_capacity_algorithm_certified", "temporal_sampling_certified", "ica_dva_certified", "dynamic_resistance_certified", "polarization_parameter_certified")
    audit.check("uncertified_measurements_remain_explicitly_uncertified", not cells[list(flags)].to_numpy(bool).any()
                and config["measurement"]["independent_instrument_calibration_certified"] is False)
    audit.check("source_RUL_dq_and_future_are_excluded_from_inference_recipe", manifest["source_rul_dq_used_as_model_inputs_or_future_targets"] is False
                and manifest["population_statistics_fitted"] is False and manifest["model_fits"] == manifest["optimizer_updates"] == 0
                and not any(any(term in name.lower() for term in ("rul", "source_dq", "future", "life_label", "eol")) for name in manifest["policy_columns"]))
    policy = pd.read_parquet(root / "policy_context.parquet")
    all_arrays, forward_records, view_summary = {}, [], {}
    for view in VIEWS:
        arrays, qualification = verify_view(audit, root, cells, manifest, view)
        all_arrays[view.name] = arrays
        diagnostic = pd.read_parquet(root / "views" / view.name / "feature_diagnostics.parquet")
        serialized_precision_checks(audit, arrays, diagnostic, view.name)
        audit.check(view.name + "/actual_policy_schema_and_values", arrays["policy_columns"].astype(str).tolist() == config["policy_columns"]
                    and policy.cell_uid.tolist() == cells.cell_uid.tolist()
                    and same_numeric(arrays["policy"], policy[config["policy_columns"]].to_numpy(np.float32), atol=0, rtol=0)
                    and arrays["policy_mask"].all())
        forward_records.extend(model_forward_checks(audit, arrays, view.name, device))
        view_summary[view.name] = {"qualification_denominator": 77, "prefix_rows": 7700,
                                   "valid_curves": int(arrays["curve_valid_mask"].sum()), "input_eligible_cells": int(arrays["input_eligible"].sum()),
                                   "reference_at_least5": int(qualification.reference_curve_count.ge(5).sum()),
                                   "view_endcurve_is_an_admission_requirement": False}
    raw_receipts = replay_raw_prefixes(audit, recovery, all_arrays, trajectory)
    audit.check("all14_normal_model_entrypoint_forward_probes", len(forward_records) == 14 and all(row["normal_entrypoint_forward_pass"] for row in forward_records))
    audit.check("derived_complete_receipt_unchanged_after_QA", all(sha256(root / item["path"]) == item["sha256"] for item in complete["files"]))
    return {"views": view_summary, "targets": target_summary, "raw_replays": raw_receipts,
            "normal_model_forward_probes": forward_records, "normal_entrypoint": "models.unified_matr.UnifiedMATRModel and train_seed42.predictions",
            "normal_forward_scope": "Fresh seed42 models, existing physical inputs, no population scaler fitted, no optimizer and no performance score.",
            "limitations": ["Source net Capacity internal generation algorithm and independent instrument calibration remain uncertified.",
                            "Temporal sampling and unrecorded rest intervals are not certified for time-based physical parameter inference.",
                            "Declared final discharge1C and recorded approximately2C remain separate fields.",
                            "Secondary charge view has separately reported limited native support.",
                            "Raw-source interpolation is independently replayed for three selected physical cells; full source target values are compared against the independent all-cycle raw audit."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPOSITORY / "data/hust_raw_idea_v2")
    parser.add_argument("--output", type=Path, default=REPOSITORY / "results/hust_raw_idea_audit_v2/FINAL_DATASET_QA.json")
    parser.add_argument("--selfcheck", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    audit = Audit()
    summary = synthetic_controls(audit)
    error = None
    if not args.selfcheck:
        try:
            if not (args.data_root / "DATASET_COMPLETE.json").is_file():
                raise RuntimeError("Completed HUST derived dataset is pending")
            torch.set_num_threads(4)
            torch.use_deterministic_algorithms(True)
            summary = verify_dataset(audit, args.data_root, torch.device(args.device))
        except Exception as exception:
            error = {"type": type(exception).__name__, "message": str(exception)}
            audit.check("complete_independent_QA_execution", False, error)
    passed = error is None and all(row["status"] == "PASS" for row in audit.checks)
    result = {"status": "PASS" if passed else "PENDING_OR_FAILED", "checks": audit.checks, "summary": summary,
              "error": error, "verifier_sha256": sha256(Path(__file__)), "model_fits": 0, "optimizer_updates": 0,
              "performance_scoring_executed": False, "selfcheck_only": args.selfcheck}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json_ready(result), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": len(audit.checks), "failed": sum(row["status"] == "FAIL" for row in audit.checks), "error": error}))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
