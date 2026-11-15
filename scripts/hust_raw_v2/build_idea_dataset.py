"""Build separately qualified HUST two-view prefix features and observed labels."""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))
from scripts.hust_safe_pickle import dataframe_columns, parse_pickle_data
from scripts.hust_raw_v2.measurements import CHARGE, DISCHARGE, NOMINAL_AH, VIEWS, full_discharge_measurement, measure_view, phase_blocks, view_definitions
from scripts.hust_raw_v2.prefix_support import json_ready, sha256
from scripts.physics_targets import endpoint_rate_target, eol80_target
from scripts.unified_matr.physics_features import GLOBAL_COLUMNS, LOCAL_COLUMNS, PhysicsConfig, extract_prefix_features


FEATURE_ARRAYS = (
    "global_features", "global_mask", "local_residual", "local_mask", "local_features", "local_feature_mask",
    "reference_q", "reference_derivative_Ah_per_V", "delta_q", "decomposition_valid", "failure_code",
    "fit_weights", "anchor_query_cycles", "anchor_observed_cycles", "anchor_support",
    "anchor_global_features", "anchor_global_mask", "anchor_local_features", "anchor_local_mask",
)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(json_ready(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_parquet(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    pd.DataFrame(rows).to_parquet(temporary, index=False)
    os.replace(temporary, path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def protocol_rates(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "DISCHARGE_RATES" for target in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("Declared HUST discharge-rate mapping is missing")


def policy_context(prefix_columns, rates):
    medians = {CHARGE: [], DISCHARGE[-1]: []}
    for columns in prefix_columns.values():
        for name, left, right in phase_blocks(columns):
            if name in medians:
                current = np.asarray(columns["Current (mA)"][left:right], dtype=float) / 1000
                if np.isfinite(current).all():
                    medians[name].append(float(np.median(current)))
    charge = float(np.median(medians[CHARGE])) / NOMINAL_AH if medians[CHARGE] else np.nan
    discharge = -float(np.median(medians[DISCHARGE[-1]])) / NOMINAL_AH if medians[DISCHARGE[-1]] else np.nan
    values = [5.0, 1.0, 3.6, *map(float, rates), 1.0, discharge, charge, 30.0, NOMINAL_AH,
              float(abs(discharge - 1.0) > .05) if np.isfinite(discharge) else np.nan]
    policy = np.asarray(values, dtype=np.float32)
    protocol_key = json.dumps({"variable_discharge_C": list(rates), "common_declared_controls": [5, 1, 3.6, 1, 30, 1.1]}, sort_keys=True)
    protocol_uid = "HUST_protocol_" + hashlib.sha256(protocol_key.encode("utf-8")).hexdigest()[:12]
    return policy, np.isfinite(policy), protocol_uid


def build_view(uid, prefix_columns, prefix_capacities, policy, policy_mask, view):
    q = np.full((100, 64), np.nan)
    native_q = np.full_like(q, np.nan)
    viq = np.full((100, 64, 3), np.nan)
    valid = np.zeros(100, dtype=bool)
    origin = np.full(100, np.nan)
    origin_points = np.full(100, -1, dtype=np.int64)
    native_points = np.zeros(100, dtype=np.int16)
    retained_points = np.zeros(100, dtype=np.int16)
    audit = []
    for cycle, columns in prefix_columns.items():
        measured, values = measure_view(columns, cycle, view)
        position = cycle - 1
        origin[position] = measured["q_origin_capacity_mAh"]
        if measured["q_origin_point"] is not None:
            origin_points[position] = measured["q_origin_point"]
        native_points[position] = measured["window_native_points"]
        retained_points[position] = measured["interpolation_native_points"]
        if measured["curve_valid"]:
            q[position] = values["q_absolute"]
            native_q[position] = values["q_native_Ah"]
            viq[position] = values["raw_viq"]
            valid[position] = True
        audit.append(measured)
    voltage = np.linspace(view.voltage_min_V, view.voltage_max_V, 64)
    physics = PhysicsConfig(voltage_min=view.voltage_min_V, voltage_max=view.voltage_max_V)
    features = extract_prefix_features(q, valid, np.arange(1, 101), voltage, physics)
    reference_cycles = features["reference_cycles"]
    spans = [np.quantile(q[cycle - 1], .95) - np.quantile(q[cycle - 1], .05) for cycle in reference_cycles]
    scale = float(np.median(spans)) if spans else np.nan
    reference = viq.copy()
    if np.isfinite(scale) and scale > 1e-8:
        reference[..., 1:] /= scale
    anchor_values = [record["capacity_Ah"] / NOMINAL_AH for cycle, record in prefix_capacities.items()
                     if 96 <= cycle <= 100 and record["capacity_valid"]]
    anchor = float(np.median(anchor_values)) if anchor_values else np.nan
    failures = []
    if not valid.any():
        failures.append("NO_QUALIFIED_PREFIX_VIEW_CURVE")
    if len(reference_cycles) < 5:
        failures.append("FEWER_THAN5_REFERENCE_CURVES_WITHIN1_TO10")
    if not features["diagnostics"]["reference_valid"] or not features["decomposition_valid"].any():
        failures.append("REFERENCE_DECOMPOSITION_UNIDENTIFIABLE")
    if not np.isfinite(scale) or scale <= 1e-8:
        failures.append("EARLY_REFERENCE_SPAN_UNAVAILABLE")
    if not np.isfinite(anchor):
        failures.append("SOURCE_CAPACITY_ANCHOR96_TO100_UNAVAILABLE")
    arrays = {"raw_viq": viq.astype(np.float32), "reference_viq": reference.astype(np.float32),
              "point_mask": np.broadcast_to(valid[:, None], (100, 64)).copy(),
              "cycle_ids": np.arange(1, 101, dtype=np.float32), "q_absolute": q,
              "q_native_Ah": native_q, "q_shape": q - q[:, -1:], "curve_valid_mask": valid,
              "q_origin_capacity_mAh": origin, "q_origin_point_index": origin_points,
              "window_native_points": native_points, "retained_window_native_points": retained_points,
              "policy": np.where(policy_mask, policy, 0).astype(np.float32), "policy_mask": policy_mask,
              "anchor_soh100": np.float32(anchor), "input_eligible": np.bool_(not failures),
              "reference_cycle_mask": np.isin(np.arange(1, 101), reference_cycles),
              "reference_qspan_95_05_Ah": np.float64(scale)}
    for name in FEATURE_ARRAYS:
        value = features[name]
        dtype = bool if "mask" in name or name == "decomposition_valid" else value.dtype
        if name in ("global_features", "local_residual", "local_features"):
            dtype = np.float32
        arrays[name] = np.asarray(value, dtype=dtype)
    diagnostics = features["diagnostics"]
    diagnostic_rows = []
    for position, measurement in enumerate(audit):
        diagnostic_rows.append({"cell_uid": uid, "view": view.name, "original_cycle_id": position + 1,
                                "curve_valid": bool(valid[position]), "decomposition_valid": bool(features["decomposition_valid"][position]),
                                "failure_code": int(features["failure_code"][position]),
                                "global_a_Ah": features["global_features"][position, 0],
                                "global_b_V": features["global_features"][position, 1],
                                "weighted_constant_projection": diagnostics["weighted_normalized_projection"][position, 0],
                                "weighted_derivative_projection": diagnostics["weighted_normalized_projection"][position, 1],
                                "reconstruction_max_absolute_error_Ah": diagnostics["reconstruction_max_absolute_error_Ah"][position],
                                "weighted_scaled_basis_condition": diagnostics["weighted_scaled_basis_condition"][position],
                                "irls_iterations": int(diagnostics["irls_iterations"][position]),
                                "irls_converged": bool(diagnostics["irls_converged"][position]),
                                "global_supported_dimensions": int(features["global_mask"][position].sum()),
                                "local_supported_coordinates": int(features["local_feature_mask"][position].sum()),
                                "source_curve_failure_reasons": "|".join(measurement["failure_reasons"])})
    qualification = {"cell_uid": uid, "view": view.name, "input_eligible": not failures,
                     "input_failure_reasons": "|".join(failures), "prefix_curve_valid_count": int(valid.sum()),
                     "coverage_ge80": bool(valid.sum() >= 80), "reference_curve_count": len(reference_cycles),
                     "reference_original_cycles": json.dumps(reference_cycles.tolist()),
                     "reference_valid": bool(diagnostics["reference_valid"]),
                     "reference_scaled_basis_condition": diagnostics["reference_scaled_basis_condition"],
                     "reference_derivative_rms_Ah_per_V": diagnostics["reference_derivative_rms_Ah_per_V"],
                     "decomposition_valid_count": int(features["decomposition_valid"].sum()),
                     "anchor_soh100": anchor, "anchor_capacity_support96_to100": len(anchor_values),
                     "anchor_curve_support96_to100": int(valid[95:].sum()), "reference_qspan_95_05_Ah": scale}
    return arrays, qualification, diagnostic_rows


def capacity_row(uid, cycle, columns, source_sha):
    measured = full_discharge_measurement(columns)
    reasons = measured["failure_reasons"]
    coordinate_valid = bool(np.all(np.asarray(columns["Cycle number"], dtype=float) == cycle))
    if not coordinate_valid:
        reasons.append("SOURCE_CYCLE_COLUMN_DIFFERS_FROM_KEY")
        measured["capacity_valid"] = False
        measured["capacity_Ah"] = measured["soh_nominal"] = np.nan
    return {"cell_uid": uid, "original_cycle_id": int(cycle), "capacity_valid": bool(measured["capacity_valid"]),
            "capacity_Ah": measured["capacity_Ah"], "soh_nominal": measured["soh_nominal"],
            "source_capacity_origin_point": measured["origin_point"], "source_capacity_origin_mAh": measured["origin_capacity_mAh"],
            "source_capacity_end_point": measured["end_point"], "source_capacity_end_mAh": measured["end_capacity_mAh"],
            "source_cycle_column_matches_key": coordinate_valid, "failure_reasons": "|".join(reasons),
            "source_member_sha256": source_sha}


def target_fields(rows):
    cycles = np.asarray([row["original_cycle_id"] for row in rows])
    soh = np.asarray([row["soh_nominal"] for row in rows])
    valid = np.asarray([row["capacity_valid"] for row in rows], dtype=bool)
    lookup = {row["original_cycle_id"]: row["soh_nominal"] for row in rows if row["capacity_valid"]}
    event = eol80_target(cycles, soh, valid)
    output = {"original_cycle_last": int(cycles[-1]), "source_cycle_count": len(rows),
              "valid_source_capacity_count": int(valid.sum()), "future_observed_capacity_count": int(np.sum(valid & (cycles > 100))),
              "last_observed_valid_cycle": int(cycles[valid][-1]) if valid.any() else None,
              "eol80_event_observed": bool(event["has_event"]),
              "eol80_first_observed_crossing_cycle": event["event_cycle"],
              "eol80_crossing_interval_lower": event["crossing_interval_lower"],
              "eol80_crossing_interval_upper": event["crossing_interval_upper"],
              "eol80_censoring": event["censoring"], "eol80_censor_cycle": event["censor_cycle"],
              "right_censored": event["censoring"] == "right"}
    for left, right in ((200, 500), (500, 1000)):
        name = f"r{left}_{right}"
        output[name] = (lookup[left] - lookup[right]) / (right - left) if left in lookup and right in lookup else None
        output[name + "_supported"] = left in lookup and right in lookup
        audit = endpoint_rate_target(cycles, soh, left, right, valid)
        output[name + "_window5_audit"] = audit["value"]
        output[name + "_window5_audit_supported"] = audit["available"]
    return output


def register(args, config, recovery, support):
    identity = read_json(args.source_identity)
    if identity["inner_sha256"] != recovery["actual_sha256"] or not identity["replay_cache_matches_current_inner"]:
        raise RuntimeError("New downloaded archive and replay source identity differ")
    if config["views"] != view_definitions() or config["source_cells"] != 77:
        raise RuntimeError("HUST view definitions differ from the fixed prefix audit")
    for entry in support["files"]:
        if sha256(args.output / "prefix_support" / entry["path"]) != entry["sha256"]:
            raise RuntimeError("Prefix-only support audit artifact changed")
    for entry in support["code"]:
        if sha256(REPOSITORY / entry["path"]) != entry["sha256"]:
            raise RuntimeError("Prefix-only support code changed after the actual audit")
    files = [Path(__file__), Path(__file__).with_name("measurements.py"), Path(__file__).with_name("prefix_support.py"),
             REPOSITORY / "scripts/hust_safe_pickle.py", REPOSITORY / "scripts/physics_targets.py",
             REPOSITORY / "scripts/unified_matr/physics_features.py"]
    registration = {"study_id": config["study_id"], "config": config, "config_sha256": sha256(args.config),
                    "source_archive_identity_sha256": sha256(args.source_identity),
                    "source_outer_sha256": identity["outer_sha256"], "source_inner_sha256": identity["inner_sha256"],
                    "source_recovery_receipt_sha256": sha256(args.source_root / "SOURCE_RECOVERY_RECEIPT.json"),
                    "prefix_support_sha256": sha256(args.output / "prefix_support/PREFIX_SUPPORT_COMPLETE.json"),
                    "protocol_source_sha256": sha256(args.protocol_source),
                    "independent_source_audit": {"all_cycle_quality_file": args.cycle_audit.name,
                                                 "all_cycle_quality_sha256": sha256(args.cycle_audit),
                                                 "raw_audit_summary_sha256": sha256(args.source_identity.with_name("HUST_RAW_AUDIT.json")),
                                                 "full_audit_verification_sha256": sha256(args.source_identity.with_name("FULL_AUDIT_VERIFICATION.json"))},
                    "code": [{"path": str(path.relative_to(REPOSITORY)).replace("\\", "/"), "sha256": sha256(path)} for path in files],
                    "model_fits": 0, "future_prediction_results_used": 0,
                    "selection_basis": "Fixed source phases and two voltage windows, approved before full label derivation and checked only against original cycles1..100"}
    path = args.output / "PROCESSING_REGISTRATION.json"
    if path.exists() and read_json(path) != registration:
        raise RuntimeError("Existing HUST processing registration differs")
    atomic_json(path, registration)
    return registration


def write_report(path, cells, view_summaries, registration):
    lines = ["# HUST 原始数据双视图物理特征处理", "", "## 数据与计量来源", "",
             f"处理保留 {len(cells)} 颗电池。新下载外层归档与既有官方内层77成员通过SHA/CRC闭合，内层SHA为 `{registration['source_inner_sha256']}`。",
             "", "容量使用作者公开表内 source-provided 净 Capacity(mAh)。其内部生成算法和独立仪器校准尚未认证。完整放电容量为 first-D0 至 last-D3 的原字段差，除1000后以1.1Ah定义名义SOH。时间积分保持审计用途。阶段边界容量继承、时间前进及未完整记录的静置区间均保留。", "",
             "## 固定的两个独立视图", "", "| 视图 | 电流与电压窗 | 有效前缀曲线 | 输入合格电池 | 前缀覆盖≥80电池 |", "|---|---|---:|---:|---:|"]
    for name, summary in view_summaries.items():
        definition = summary["definition"]
        lines.append(f"| {name} | {definition['current_A']} A，{definition['voltage_min_V']}–{definition['voltage_max_V']} V | {summary['valid_curves']}/7700 | {summary['input_eligible_cells']}/77 | {summary['coverage_ge80_cells']}/77 |")
    lines += ["", "主视图为实际2C末段放电。原文末段设计1C与前100圈实测约2C的冲突分列保存。前面三段变电流造成完整放电电压回升，完整四段用于容量与源审计，单阶段Q(V)沿原采集顺序构造。5C充电视图的原生采样较稀，保留独立资格与缺失掩码。64个插值坐标的原生支撑点数、去除点数和电压间隙单独记录。", "",
             "D3绝对Q使用D0首个原字段容量减去当前原字段容量。charge绝对Q直接由原字段换算并逐圈认证首值零。两者同时保留插值原始净Q及原点值/位置。shape仅另存窗口上端归零表示，a/b/r分解保留绝对Q。", "",
             "## 特征与输入资格", "", "各视图独立使用原始1..10圈至少5条合格曲线建立逐点中位参考。在64点电压轴拟合 ΔQ=a+b·dQref/dV+r，复用已登记的Huber IRLS数学算子。保存G20、8窗L26、64点残差与相同形状布尔掩码、权重、正交投影、重建误差和条件数。动态量使用最后10次有效观测及真实循环距离，锚点为10/50/100圈。", "",
             "SOH100输入锚点独立取完整放电容量在96..100圈的可用中位数。某视图末五圈曲线缺失只计入覆盖描述。输入资格取至少一条视图曲线、前10圈至少5条参考、可辨识分解、正参考跨度及合法容量锚点。77颗母表全部保留，80圈覆盖只作分层描述。数据处理阶段未拟合跨电池标准化统计。", "",
             "## 未来观测与支持", "", f"有效未来观测点 {int(cells.future_observed_capacity_count.sum())}。exact r200_500支持 {int(cells.r200_500_supported.sum())}/77，exact r500_1000支持 {int(cells.r500_1000_supported.sum())}/77。±5圈中位速率以 `_window5_audit` 后缀保留。",
             "", f"精确名义SOH≤0.8事件 {int(cells.eol80_event_observed.sum())}/77，右删失 {int(cells.right_censored.sum())}/77。源rul/dq进入独立审计，未来监督从相同净Capacity字段重新构造。", "",
             "## 优势与劣势（风险）", "", "优势是原始阶段、单位、Q原点、插值支撑、全局/局部分解与未来标签形成同一来源链，两视图按相同资格原则保留支持差异。", "",
             "风险包括容量内部生成算法、独立仪器校准和时序完整性证据缺口，末段设计/实测冲突，以及充电视图较稀原生采样。a/b/r是固定激励条件下的加载响应分解，ICA/DVA、动态内阻和极化参数认证标记为缺失。历史77颗已参与探索，后续结论应保留这一暴露信息。", "",
             "## 置信度评级", "", "阶段取样、单位换算、前缀隔离和数组计量的置信度为高，需与独立QA结果共同阅读。具体电化学机制与独立仪器计量的置信度为低，当前源记录缺少对应认证。", "",
             "## 拓展思考", "", "同一电池的两个可比视图可以检验加载响应在不同激励阶段的稳定性，比较时应共享电池和目标支持。", "",
             "## 你可能想知道", "", "本次交付包含派生数据与审计，模型拟合次数和优化器更新次数均为0。"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(args):
    config = read_json(args.config)
    recovery = read_json(args.source_root / "SOURCE_RECOVERY_RECEIPT.json")
    support = read_json(args.output / "prefix_support/PREFIX_SUPPORT_COMPLETE.json")
    if len(recovery["members"]) != 77 or recovery["actual_sha256"] != recovery["official_sha256"]:
        raise RuntimeError("The HUST source recovery identity differs")
    completion = args.output / "DATASET_COMPLETE.json"
    if completion.exists():
        raise RuntimeError("The derived HUST dataset is already complete")
    registration = register(args, config, recovery, support)
    rates = protocol_rates(args.protocol_source)
    view_arrays = {view.name: [] for view in VIEWS}
    qualifications = {view.name: [] for view in VIEWS}
    diagnostics = {view.name: [] for view in VIEWS}
    cells, trajectories, policies = [], [], []
    for index, member in enumerate(recovery["members"], 1):
        path = Path(member["path"])
        if sha256(path) != member["sha256"]:
            raise RuntimeError("Source member bytes differ from the authenticated recovery")
        obj, _ = parse_pickle_data(path.read_bytes())
        data = obj[path.stem]["data"]
        uid = "HUST_" + path.stem
        prefix = {cycle: dataframe_columns(data[cycle]) for cycle in range(1, 101)}
        prefix_capacity = {cycle: capacity_row(uid, cycle, columns, member["sha256"]) for cycle, columns in prefix.items()}
        policy, policy_mask, protocol_uid = policy_context(prefix, rates[path.stem])
        policy_row = {"cell_uid": uid, "protocol_uid": protocol_uid,
                      **{name: float(value) for name, value in zip(config["policy_columns"], policy)}}
        policies.append(policy_row)
        eligibility = {}
        for view in VIEWS:
            arrays, qualification, diagnostic = build_view(uid, prefix, prefix_capacity, policy, policy_mask, view)
            view_arrays[view.name].append(arrays)
            qualifications[view.name].append(qualification)
            diagnostics[view.name].extend(diagnostic)
            eligibility[view.name] = qualification["input_eligible"]
        # Suffix values are read only after inference-time arrays are complete.
        rows = [capacity_row(uid, cycle, dataframe_columns(data[cycle]), member["sha256"]) for cycle in sorted(data)]
        trajectories.extend(rows)
        anchor = view_arrays[VIEWS[0].name][-1]["anchor_soh100"]
        cells.append({"cell_uid": uid, "family": "HUST", "source_device_record_id": path.stem,
                      "legacy_standardized_filename": uid + ".pkl", "source_archive_member": member["member"],
                      "source_member_sha256": member["sha256"], "protocol_uid": protocol_uid,
                      "nominal_capacity_Ah": NOMINAL_AH, "anchor_soh100": float(anchor),
                      "historical_exposure": True, "input_eligible": eligibility[config["primary_view"]],
                      **{"input_eligible_" + name: value for name, value in eligibility.items()},
                      **target_fields(rows), "source_capacity_algorithm_certified": False,
                      "temporal_sampling_certified": False, "ica_dva_certified": False,
                      "dynamic_resistance_certified": False, "polarization_parameter_certified": False})
        del obj, data, prefix
        gc.collect()
        print(json.dumps({"status": "HUST_DERIVING", "completed_cells": index, "total_cells": 77, "cell_uid": uid,
                          "view_input_eligibility": eligibility}), flush=True)
    trajectory = pd.DataFrame(trajectories)
    atomic_parquet(args.output / "source_capacity_trajectory.parquet", trajectory)
    atomic_parquet(args.output / "future_soh.parquet", trajectory.loc[trajectory.original_cycle_id > 100])
    atomic_parquet(args.output / "cells.parquet", cells)
    atomic_parquet(args.output / "policy_context.parquet", policies)
    view_summaries = {}
    for view in VIEWS:
        root = args.output / "views" / view.name
        root.mkdir(parents=True, exist_ok=True)
        stacked = {name: np.stack([cell[name] for cell in view_arrays[view.name]]) for name in view_arrays[view.name][0]}
        stacked.update(cell_uid=np.asarray([cell["cell_uid"] for cell in cells]), voltage_grid_V=np.linspace(view.voltage_min_V, view.voltage_max_V, 64),
                       policy_columns=np.asarray(config["policy_columns"]), global_columns=np.asarray(GLOBAL_COLUMNS), local_columns=np.asarray(LOCAL_COLUMNS))
        path = root / "early_inputs.npz"
        temporary = path.with_suffix(".npz.pending")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **stacked)
        os.replace(temporary, path)
        atomic_parquet(root / "view_qualification.parquet", qualifications[view.name])
        atomic_parquet(root / "feature_diagnostics.parquet", diagnostics[view.name])
        view_summaries[view.name] = {"definition": asdict(view), "cells": 77,
                                    "valid_curves": int(stacked["curve_valid_mask"].sum()),
                                    "decomposition_valid_curves": int(stacked["decomposition_valid"].sum()),
                                    "input_eligible_cells": int(stacked["input_eligible"].sum()),
                                    "coverage_ge80_cells": int(np.sum(stacked["curve_valid_mask"].sum(axis=1) >= 80)),
                                    "reference_valid_cells": int(sum(row["reference_valid"] for row in qualifications[view.name])),
                                    "physics_config": asdict(PhysicsConfig(voltage_min=view.voltage_min_V, voltage_max=view.voltage_max_V)),
                                    "array_shapes": {name: list(value.shape) for name, value in stacked.items()},
                                    "q_shape_zero_at_voltage_V": view.voltage_max_V,
                                    "loaded_shape_certification": "response decomposition only; ICA/DVA and electrochemical parameter certification unavailable"}
    table = pd.DataFrame(cells)
    manifest = {"status": "DERIVED_DATASET_COMPLETE_PENDING_INDEPENDENT_QA", "study_id": config["study_id"],
                "registration_sha256": sha256(args.output / "PROCESSING_REGISTRATION.json"),
                "source_cells": len(cells), "source_capacity_rows": len(trajectory),
                "valid_source_capacity_rows": int(trajectory.capacity_valid.sum()),
                "future_observed_points": int((trajectory.capacity_valid & (trajectory.original_cycle_id > 100)).sum()),
                "rate_support": {name: int(table[name + "_supported"].sum()) for name in ("r200_500", "r500_1000")},
                "observed_eol80_events": int(table.eol80_event_observed.sum()), "right_censored_cells": int(table.right_censored.sum()),
                "views": view_summaries, "policy_columns": config["policy_columns"],
                "global_columns": list(GLOBAL_COLUMNS), "local_columns": list(LOCAL_COLUMNS),
                "capacity_provenance": config["measurement"], "policy_provenance": config["policy_provenance"],
                "independent_source_audit": registration["independent_source_audit"],
                "population_statistics_fitted": False, "model_fits": 0, "optimizer_updates": 0,
                "two_view_qualification_is_independent": True, "source_rul_dq_used_as_model_inputs_or_future_targets": False,
                "temporal_features_inferred_from_missing_measurements": False,
                "training_or_model_selection_performed": False,
                "shared_operator_source": "scripts/unified_matr/physics_features.py, reused with per-view registered physical voltage windows. HUST field names and capacity provenance are declared here.",
                "historical_exposure": "All77 HUST cells participated in previous exploration"}
    atomic_json(args.output / "dataset_manifest.json", manifest)
    write_report(args.report, table, view_summaries, registration)
    paths = [args.output / name for name in ("PROCESSING_REGISTRATION.json", "dataset_manifest.json", "source_capacity_trajectory.parquet", "future_soh.parquet", "cells.parquet", "policy_context.parquet")]
    paths += sorted((args.output / "views").rglob("*.npz")) + sorted((args.output / "views").rglob("*.parquet"))
    paths += [args.output / "prefix_support" / entry["path"] for entry in support["files"]]
    paths.append(args.output / "prefix_support/PREFIX_SUPPORT_COMPLETE.json")
    atomic_json(completion, {"status": "DERIVED_DATASET_COMPLETE_PENDING_INDEPENDENT_QA", "model_fits": 0, "optimizer_updates": 0,
                             "files": [{"path": str(path.relative_to(args.output)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in paths],
                             "report": {"path": str(args.report.relative_to(REPOSITORY)).replace("\\", "/"), "sha256": sha256(args.report)},
                             "registration_sha256": sha256(args.output / "PROCESSING_REGISTRATION.json"),
                             "summary": {key: manifest[key] for key in ("source_cells", "source_capacity_rows", "future_observed_points", "rate_support", "observed_eol80_events", "right_censored_cells")}})
    print(json.dumps({"status": manifest["status"], "views": {name: {key: value[key] for key in ("valid_curves", "input_eligible_cells")} for name, value in view_summaries.items()}, "model_fits": 0}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=REPOSITORY.parent / "private/raw_sources/HUST")
    parser.add_argument("--source-identity", type=Path, default=REPOSITORY / "results/hust_raw_idea_audit_v2/CURRENT_ARCHIVE_IDENTITY.json")
    parser.add_argument("--protocol-source", type=Path, default=Path(r"D:\develop\patchtst\BatteryLife\process_scripts\preprocess_HUST.py"))
    parser.add_argument("--cycle-audit", type=Path, default=REPOSITORY.parent / "private/raw_sources/HUST_IDEA_V2/all_cycle_quality.csv")
    parser.add_argument("--config", type=Path, default=REPOSITORY / "configs/hust_raw_idea_v2.json")
    parser.add_argument("--output", type=Path, default=REPOSITORY / "data/hust_raw_idea_v2")
    parser.add_argument("--report", type=Path, default=REPOSITORY / "report/HUST_RAW_IDEA_PROCESSING_ZH.md")
    args = parser.parse_args()
    for name in ("source_root", "source_identity", "protocol_source", "cycle_audit", "config", "output", "report"):
        setattr(args, name, getattr(args, name).resolve())
    build(args)


if __name__ == "__main__":
    main()
