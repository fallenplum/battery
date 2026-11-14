"""Standalone SDU audit adapted from the previously executed full BatteryLife auditor."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import pickle
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(".")
OUT = Path(__file__).resolve().parent
ALIGNED = (
    "voltage_in_V", "current_in_A", "charge_capacity_in_Ah",
    "discharge_capacity_in_Ah", "time_in_s",
)


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.generic):
        return safe_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": str(value)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"python_type": type(value).__name__, "representation": repr(value)}


def dump(value, path):
    path.write_text(json.dumps(safe_json(value), ensure_ascii=False, indent=2), encoding="utf-8")


class HashingReader:
    def __init__(self, file):
        self.file = file
        self.digest = hashlib.sha256()

    def read(self, size=-1):
        data = self.file.read(size)
        self.digest.update(data)
        return data

    def readline(self, size=-1):
        data = self.file.readline(size)
        self.digest.update(data)
        return data


def new_channel():
    return {
        "present_cycles": 0, "none_cycles": 0, "empty_cycles": 0,
        "nonnumeric_cycles": 0, "scalar_cycles": 0, "array_cycles": 0,
        "points": 0, "finite_points": 0, "nan_points": 0,
        "positive_inf_points": 0, "negative_inf_points": 0,
        "all_nonfinite_cycles": 0, "constant_finite_cycles": 0,
        "all_zero_cycles": 0, "negative_points": 0, "positive_points": 0,
        "zero_points": 0, "min": None, "max": None,
        "length_min": None, "length_max": None,
        "types": Counter(), "shapes": Counter(),
    }


def extrema(target, low, high, low_key="min", high_key="max"):
    if target[low_key] is None or low < target[low_key]:
        target[low_key] = float(low)
    if target[high_key] is None or high > target[high_key]:
        target[high_key] = float(high)


def channel_values(value, stats):
    stats["present_cycles"] += 1
    stats["types"][type(value).__name__] += 1
    if value is None:
        stats["none_cycles"] += 1
        return None
    try:
        array = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError):
        stats["nonnumeric_cycles"] += 1
        return None
    stats["shapes"]["scalar" if array.ndim == 0 else f"{array.ndim}d"] += 1
    stats["scalar_cycles" if array.ndim == 0 else "array_cycles"] += 1
    array = array.reshape(-1)
    size = array.size
    stats["points"] += int(size)
    extrema(stats, size, size, "length_min", "length_max")
    if not size:
        stats["empty_cycles"] += 1
        return array
    finite = np.isfinite(array)
    count = int(np.count_nonzero(finite))
    stats["finite_points"] += count
    stats["nan_points"] += int(np.count_nonzero(np.isnan(array)))
    stats["positive_inf_points"] += int(np.count_nonzero(np.isposinf(array)))
    stats["negative_inf_points"] += int(np.count_nonzero(np.isneginf(array)))
    stats["negative_points"] += int(np.count_nonzero(array < 0))
    stats["positive_points"] += int(np.count_nonzero(array > 0))
    stats["zero_points"] += int(np.count_nonzero(array == 0))
    if not count:
        stats["all_nonfinite_cycles"] += 1
        return array
    valid = array if count == size else array[finite]
    low, high = float(valid.min()), float(valid.max())
    extrema(stats, low, high)
    stats["constant_finite_cycles"] += int(low == high)
    stats["all_zero_cycles"] += int(count == size and low == 0 and high == 0)
    return array


def summarize(values):
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0}
    return {"count": len(array), "min": float(array.min()),
            "median": float(np.median(array)), "max": float(array.max()),
            "p05": float(np.quantile(array, 0.05)), "p95": float(np.quantile(array, 0.95))}


def inspect(path):
    before = path.stat()
    result = {"relative_path": path.relative_to(ROOT).as_posix(),
              "family": path.relative_to(ROOT).parts[0], "size_bytes": before.st_size,
              "mtime_ns": before.st_mtime_ns, "status": "OK"}
    start = time.monotonic()
    try:
        with path.open("rb") as file:
            reader = HashingReader(file)
            obj = pickle.load(reader)
            trailing_bytes = 0
            while block := reader.read(8 * 1024 * 1024):
                trailing_bytes += len(block)
        result["sha256"] = reader.digest.hexdigest()
        result["trailing_bytes"] = trailing_bytes
        result["object_type"] = type(obj).__name__
        if not isinstance(obj, dict):
            raise TypeError(f"Expected dict, got {type(obj).__name__}")
        metadata = {key: value for key, value in obj.items() if key != "cycle_data"}
        result["metadata"] = safe_json(metadata)
        result["top_level_keys"] = list(obj)
        cycles = obj.get("cycle_data")
        if not isinstance(cycles, list):
            raise TypeError(f"Expected cycle_data list, got {type(cycles).__name__}")
        result["cycle_count"] = len(cycles)
        channels = defaultdict(new_channel)
        schemas = Counter()
        issues = Counter()
        examples = defaultdict(list)
        distributions = defaultdict(list)
        fingerprints = {}
        numbers = []
        cycle_trace = []
        arrays = {}
        previous_end = None
        previous_number = None
        nominal = metadata.get("nominal_capacity_in_Ah")
        nominal = float(nominal) if isinstance(nominal, (float, int)) and nominal > 0 else None

        def issue(name, index, number):
            issues[name] += 1
            if len(examples[name]) < 5:
                examples[name].append({"list_index": index, "cycle_number": safe_json(number)})

        for index, cycle in enumerate(cycles):
            if not isinstance(cycle, dict):
                issue("cycle_is_not_dictionary", index, None)
                continue
            schemas[json.dumps(sorted(cycle))] += 1
            number = cycle.get("cycle_number")
            if isinstance(number, (int, float, np.number)) and np.isfinite(number):
                number = float(number)
                numbers.append(number)
                if previous_number is not None:
                    if number <= previous_number:
                        issue("cycle_number_nonincreasing", index, number)
                    elif number - previous_number != 1:
                        issue("cycle_number_gap", index, number)
                previous_number = number
            else:
                issue("cycle_number_invalid", index, number)
            cycle_row = {"list_index": index, "cycle_number": safe_json(number), "duplicate_VIQ_first_index": None}
            arrays = {}
            for key, value in cycle.items():
                if key != "cycle_number":
                    arrays[key] = channel_values(value, channels[key])
            lengths = {key: None if arrays.get(key) is None else len(arrays[key]) for key in ALIGNED}
            if any(lengths[key] is None or lengths[key] == 0 for key in ALIGNED):
                issue("required_channel_missing_or_empty", index, number)
            if len({length for length in lengths.values() if length is not None}) > 1:
                issue("required_channel_length_mismatch", index, number)
            temp = arrays.get("temperature_in_C")
            if temp is not None and lengths["voltage_in_V"] is not None and len(temp) != lengths["voltage_in_V"]:
                issue("temperature_length_mismatch", index, number)
            for key in ALIGNED:
                array = arrays.get(key)
                if array is not None and np.any(~np.isfinite(array)):
                    issue(f"nonfinite_{key}", index, number)
            voltage = arrays.get("voltage_in_V")
            if voltage is not None and np.any(voltage <= 0):
                issue("nonpositive_voltage", index, number)
            current = arrays.get("current_in_A")
            if current is not None:
                negative_current = current[(current < 0) & np.isfinite(current)]
                positive_current = current[(current > 0) & np.isfinite(current)]
                cycle_row["negative_current_mean_A"] = float(negative_current.mean()) if len(negative_current) else None
                cycle_row["positive_current_mean_A"] = float(positive_current.mean()) if len(positive_current) else None
            t = arrays.get("time_in_s")
            dt = None
            if t is not None and len(t) > 1:
                dt = np.diff(t)
                negative = int(np.count_nonzero(dt < -1e-9))
                repeated = int(np.count_nonzero(dt == 0))
                issues["negative_time_steps"] += negative
                issues["repeated_time_steps"] += repeated
                if negative:
                    issue("time_decreasing_cycles", index, number)
                if repeated:
                    issue("time_repeated_cycles", index, number)
                positive = dt[(dt > 0) & np.isfinite(dt)]
                if positive.size:
                    distributions["cycle_median_positive_dt_s"].append(float(np.median(positive)))
                    distributions["cycle_max_positive_dt_s"].append(float(positive.max()))
                if np.isfinite(t[0]) and np.isfinite(t[-1]):
                    distributions["cycle_time_span_s"].append(float(t[-1] - t[0]))
                    if previous_end is not None:
                        distributions["between_cycle_time_gap_s"].append(float(t[0] - previous_end))
                        if t[0] < previous_end - 1e-9:
                            issue("cross_cycle_time_reset", index, number)
                    previous_end = float(t[-1])
            for direction, qkey in (("charge", "charge_capacity_in_Ah"), ("discharge", "discharge_capacity_in_Ah")):
                q = arrays.get(qkey)
                if q is None or not q.size:
                    continue
                finite_q = q[np.isfinite(q)]
                if not finite_q.size:
                    continue
                qmax = float(finite_q.max())
                qspan = float(qmax - finite_q.min())
                cycle_row[f"{direction}_max_Ah"] = qmax
                cycle_row[f"{direction}_span_Ah"] = qspan
                cycle_row[f"{direction}_capacity_nonfinite_points"] = int(len(q) - len(finite_q))
                distributions[f"{direction}_capacity_max_Ah"].append(qmax)
                if nominal:
                    distributions[f"{direction}_capacity_max_over_nominal"].append(qmax / nominal)
                if qmax <= 0:
                    issue(f"{direction}_capacity_max_nonpositive", index, number)
                if dt is None or current is None or len(current) != len(t):
                    continue
                if np.any(~np.isfinite(current)) or np.any(~np.isfinite(dt)) or np.any(dt < -1e-9):
                    continue
                signed = current if direction == "charge" else -current
                clipped = np.maximum(signed, 0)
                integral = float(np.sum((clipped[:-1] + clipped[1:]) * 0.5 * dt) / 3600)
                distributions[f"{direction}_current_integral_Ah"].append(integral)
                if integral > 1e-12:
                    distributions[f"{direction}_qspan_over_current_integral"].append(qspan / integral)
            fingerprint = hashlib.blake2b(digest_size=16)
            fingerprint_ok = True
            for key in ALIGNED[:4]:
                array = arrays.get(key)
                if array is None:
                    fingerprint_ok = False
                    break
                fingerprint.update(key.encode())
                fingerprint.update(str(array.shape).encode())
                fingerprint.update(array.tobytes())
            if fingerprint_ok:
                digest = fingerprint.digest()
                if digest in fingerprints:
                    issue("exact_repeated_V_I_Qc_Qd_cycles", index, number)
                    cycle_row["duplicate_VIQ_first_index"] = fingerprints[digest]
                else:
                    fingerprints[digest] = index
            cycle_trace.append(cycle_row)
        result["cycle_trace"] = cycle_trace
        for stats in channels.values():
            stats["absent_cycles"] = len(cycles) - stats["present_cycles"]
            stats["types"] = dict(stats["types"])
            stats["shapes"] = dict(stats["shapes"])
        result["channels"] = dict(channels)
        result["cycle_schemas"] = [{"keys": json.loads(keys), "cycles": count} for keys, count in schemas.items()]
        result["cycle_number"] = {"min": min(numbers) if numbers else None,
                                  "max": max(numbers) if numbers else None,
                                  "unique_count": len(set(numbers)), "numeric_count": len(numbers)}
        result["issues"] = dict(issues)
        result["issue_examples"] = dict(examples)
        result["distributions"] = {key: summarize(values) for key, values in distributions.items()}
        result["protocol_issues"] = []
        for direction in ("charge_protocol", "discharge_protocol"):
            for step_index, step in enumerate(metadata.get(direction) or []):
                if not isinstance(step, dict):
                    result["protocol_issues"].append(f"{direction}[{step_index}] is not dict")
                    continue
                for key in ("start_soc", "end_soc"):
                    value = step.get(key)
                    if isinstance(value, (int, float)) and not 0 <= value <= 1:
                        result["protocol_issues"].append(f"{direction}[{step_index}].{key}={value} outside [0,1]")
                if step.get("rate_in_C") == "":
                    result["protocol_issues"].append(f"{direction}[{step_index}].rate_in_C is empty string")
        result["d0"] = inspect_sdu_d0(obj, cycle_trace)
        del obj, cycles, arrays, fingerprints
        gc.collect()
    except Exception as exc:
        result["status"] = "ERROR"
        result["error"] = f"{type(exc).__name__}: {exc}"
    after = path.stat()
    result["stat_unchanged_during_read"] = before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns
    result["elapsed_seconds"] = round(time.monotonic() - start, 3)
    return result


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(safe_json(value), ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})



REFERENCE_AUDITOR_SHA256 = "196617c517005c9e19589faefd27b9969d65a75bc40a5daed14dcec300814d0d"


def inspect_sdu_d0(payload, cycle_trace):
    """Keep this new cohort on its supplied coordinate and exclude repeated VIQ."""
    from run_d0 import capacity, current_reference, make_features, observed_curve
    from physics_targets import future_targets

    cycles = payload["cycle_data"]
    nominal = float(payload["nominal_capacity_in_Ah"])
    identifier = payload["cell_id"]
    numbers = [float(cycle["cycle_number"]) for cycle in cycles]
    repeated = {int(row["list_index"]) for row in cycle_trace if row.get("duplicate_VIQ_first_index") is not None}
    capacities = [None if index in repeated else capacity(cycle, "discharge_capacity_in_Ah", nominal) for index, cycle in enumerate(cycles)]
    valid_numbers = [n for n, q in zip(numbers, capacities) if q is not None]
    valid_soh = [q for q in capacities if q is not None]
    targets = future_targets(np.asarray(valid_numbers), np.asarray(valid_soh))
    targets.update(cell_id=identifier, family="SDU", target_coordinate="supplied_cycle_number",
                   strict_physical_target_authenticated=False,
                   duplicate_VIQ_records_excluded=len(repeated),
                   capacity_definition="phase_confirmed_discharge_max_minus_global_finite_baseline_over_declared_nominal",
                   official_label_used_for_event=False)
    setting = {"direction": "discharge", "voltage_window": [3.4, 4.1]}
    features, prefix_rows = [], []
    for prefix in (10, 50, 100):
        early = [cycle for index, cycle in enumerate(cycles) if numbers[index] <= prefix and index not in repeated]
        reference = current_reference(early, "discharge")
        curves, reasons, shape_count = [], Counter(), 0
        for cycle in early:
            curve, reason, proxy = observed_curve(cycle, setting, reference, nominal)
            reasons[reason] += 1
            if curve is not None:
                curves.append((float(cycle["cycle_number"]), curve))
                shape_count += int(proxy is not None)
        feature = {"cell_id": identifier, "family": "SDU", **make_features(numbers, capacities, curves, prefix)}
        feature.update(prefix_reference_current_A=reference, duplicate_VIQ_records_excluded=True,
                       feature_scope="supplied_coordinate_observed_QV_shape_only")
        features.append(feature)
        prefix_rows.append({"cell_id": identifier, "family": "SDU", "K_supplied": prefix,
                            "provided_prefix_numbers_complete": set(range(1, prefix + 1)).issubset(set(numbers)),
                            "provided_prefix_records": sum(n <= prefix for n in numbers),
                            "nonduplicate_prefix_records": len(early),
                            "valid_capacity_prefix_records": feature["early_capacity_records"],
                            "observed_curve_records": len(curves),
                            "observed_IC_shape_proxy_records": shape_count,
                            "two_curve_feature_supported": len(curves) >= 2,
                            "curve_failure_counts": dict(reasons), "reference_current_A": reference,
                            "physical_cycle_axis_authenticated": False, "N_true_ICA": 0, "N_true_DVA": 0})
    k100 = prefix_rows[-1]
    eligible = {"cell_id": identifier, "family": "SDU", "provided_cycles": len(cycles),
                "supplied_cycle_min": min(numbers), "supplied_cycle_max": max(numbers),
                "nominal_capacity_Ah": nominal, "duplicate_VIQ_records_excluded": len(repeated),
                "hardcoded_removed_original_indices": payload.get("hardcoded_removed_indices", []),
                "median_removed_original_indices": payload.get("median_removed_indices", []),
                "removed_original_index_to_supplied_index_mapping_authenticated": False,
                "provided_replacement_index_metadata_available": False,
                "prefix_K100_numbers_complete": k100["provided_prefix_numbers_complete"],
                "curve_valid_records_K100": k100["observed_curve_records"],
                "physical_cycle_axis_authenticated": False, "original_instrument_source_available": False,
                "physical_cycle_source_supported": False,
                "physical_cycle_evidence": "processed_cycles_reindexed_and_diagnostic_replacements_unmapped",
                "time_axis_authenticated": False, "temperature_matching_authenticated": False,
                "future_policy_status": "unknown", "future_policy_basis": "Protocol_ID_and_raw_policy_mapping_absent_in_processed_payload",
                "N_true_ICA_records": 0, "N_true_DVA_records": 0,
                "strict_physical_target_authenticated": False,
                "target_mid_supported": targets["r200_500"] is not None,
                "target_late_supported": targets["r500_1000"] is not None,
                "target_EOLevent_after_prior_above_threshold": targets["eol_event"] is True and targets["eol_censoring"] in ("none", "interval"),
                "target_EOL_censoring": targets["eol_censoring"],
                "target_knee_stable_candidate": bool(targets["knee_stable_candidate"])}
    return {"eligible": eligible, "features": features, "prefix_eligibility": prefix_rows, "targets": targets}


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def aggregate_sdu(records):
    fields, presence = defaultdict(Counter), Counter()
    channels, ranges, issues, schemas = defaultdict(Counter), {}, Counter(), Counter()
    total_cycles = sum(r.get("cycle_count", 0) for r in records)
    for row in records:
        for key, value in row.get("metadata", {}).items():
            fields[key][json.dumps(value, ensure_ascii=False, sort_keys=True)] += 1
            presence[key] += 1
        issues.update(row.get("issues", {}))
        for schema in row.get("cycle_schemas", []):
            schemas[json.dumps(schema["keys"])] += schema["cycles"]
        for name, stats in row.get("channels", {}).items():
            for key, value in stats.items():
                if isinstance(value, int) and key not in ("length_min", "length_max"):
                    channels[name][key] += value
            if stats["finite_points"]:
                interval = ranges.setdefault(name, {"min": None, "max": None})
                extrema(interval, stats["min"], stats["max"])
    for counts in channels.values():
        counts["absent_cycles"] = total_cycles - counts["present_cycles"]
    return {"family": "SDU", "files": len(records), "read_ok": sum(r["status"] == "OK" for r in records),
            "cycles": total_cycles, "uncompressed_bytes": sum(r["size_bytes"] for r in records),
            "cycles_per_cell": summarize([r["cycle_count"] for r in records]),
            "metadata_fields": {key: {"present_cells": presence[key], "values": [{"value": json.loads(value), "cells": count} for value, count in counts.items()]} for key, counts in fields.items()},
            "channels": {key: {**dict(counts), **ranges.get(key, {})} for key, counts in channels.items()},
            "issues": dict(issues),
            "cycle_schemas": [{"keys": json.loads(keys), "cycles": count} for keys, count in schemas.items()]}


def write_sdu_report(report_path, summary, coverage, label_rows, prefixes, targets, source_info):
    lines = ["# SDU 新增快照完整元数据、信号与独立D0资格审计", "",
             "审计日期为2026-11-15，数据来自本次新增的SDU.zip。该快照独立于已冻结的427个主要D0电池。", "",
             "## 覆盖", "",
             f"压缩包有{summary['files']}个标准化电池PKL，共{summary['cycles']:,}圈，未压缩{summary['uncompressed_bytes']:,}字节。全部文件完整SHA-256、zip成员CRC、全部顶层元数据、每循环schema与全部数值点均完成检查。压缩包本次读取前后SHA-256一致性为{coverage['source_sha256_unchanged']}。", "",
             f"精确寿命标签命中{sum(r['life_label'] is not None for r in label_rows)}/{len(label_rows)}。其余文件仍保留信号，标签生成版本与当前快照的关联尚未独立证实。", "",
             "## 元数据与来源", "",
             "当前标准化文件中的材料、容量、电压限制、SOC和删圈列表属于供应元数据，逐文件完整值见sdu_cell_metadata.csv，所有取值及计数见sdu_family_summary.json。", "",
             "|字段|完整取值及电池计数|", "|---|---|"]
    for key, info in summary["metadata_fields"].items():
        if key in ("cell_id", "hardcoded_removed_indices", "median_removed_indices"):
            text = f"{len(info['values'])}种取值，完整逐电池列表在CSV/JSON"
        else:
            text = json.dumps(info["values"], ensure_ascii=False).replace("|", "/")
        lines.append(f"|{key}|{text}|")
    lines += ["", "官方源研究包含初次使用和第二寿命两个阶段。初次使用包含86颗NMC532/graphite电池、16种协议，诊断测试每4个工作循环执行，原始CSV含Battery_ID与Protocol_ID。标准化PKL中未发现Protocol_ID，IR字段的实际覆盖以本次扫描为准。官方文件名编号与论文Battery_ID不同，原始文件名61/62缺失的说明无法直接用于否定标准化Battery_61/62身份。[SDU官方数据记录](https://zenodo.org/records/14859405)", "",
              "本地preprocess_SDU.py按Test_Time排序并重组Cycle_Index，按硬编码容量/循环窗口删圈，将指定诊断圈替换为最近普通圈的完整V/I/Q/t/T，再连续重编号并调用统一时间修复。hardcoded_removed_indices是删圈前列表位置，替换圈位置未单独保存。当前源码与该zip的生成版本关联仍未证实。源码行号和SHA见sdu_preprocessing_lineage.json。", "",
              "## 全通道数值质量", "", "|通道|有限/总点数|有限值范围|None/缺键圈|NaN点|", "|---|---|---|---|---|"]
    for name, stats in summary["channels"].items():
        lines.append(f"|{name}|{stats['finite_points']:,}/{stats['points']:,}|{stats.get('min')}至{stats.get('max')}|{stats['none_cycles'] + stats['absent_cycles']:,}|{stats['nan_points']:,}|")
    lines += ["", "循环与时间问题的精确计数如下。", "", "```json", json.dumps(summary["issues"], ensure_ascii=False, indent=2), "```", "",
              "重复VIQ指纹使用完整V/I/Qc/Qd数组，时间与温度未加入。后续同指纹记录从本次D0特征和目标候选中排除。首个保留记录的原始身份仍可能受源代码诊断替换影响，因此该排除不能认证真实物理循环。", "",
              "## 寿命标签、事件与删失", "",
              f"寿命标量大于当前循环记录条数的电池为{sum(r['label_exceeds_record_count'] for r in label_rows)}个，大于最大供应循环编号的电池为{sum(r['label_exceeds_max_observed_cycle'] for r in label_rows)}个。标签不用于生成已观察事件。", "",
              "未来容量由放电电流阶段确认后的容量最大值减去有限全圈容量基线，再除以声明容量。排除重复VIQ后计算200–500、500–1000供应循环号的端点窗口平均斜率，EOL80及稳定膝点候选。EOL字段区分none、interval、right、left、unknown，首次观测已低于阈值单独按left报告。", "",
              "## 独立D0资格与支持", "", "|供应前缀K|编号1至K齐全/电池|至少2条合格观测曲线/电池|真实物理循环认证|真实ICA/DVA|", "|---|---|---|---|---|"]
    for cutoff in (10, 50, 100):
        rows = [row for row in prefixes if row["K_supplied"] == cutoff]
        lines.append(f"|{cutoff}|{sum(r['provided_prefix_numbers_complete'] for r in rows)}|{sum(r['two_curve_feature_supported'] for r in rows)}|0|0/0|")
    event_counts = Counter(r["eol_censoring"] for r in targets)
    lines += ["", f"独立新cohort的200–500斜率候选有{sum(r['r200_500'] is not None for r in targets)}颗，500–1000斜率候选有{sum(r['r500_1000'] is not None for r in targets)}颗，稳定膝点候选有{sum(r['knee_stable_candidate'] for r in targets)}颗。EOL删失类别为{dict(event_counts)}。同一电池的严格物理认证交集为0，完整观测候选交集见sdu_feature_coverage.csv。", "",
              "曲线设置为放电、共同电压窗口3.4–4.1V、前缀K10/50/100。每个K的参考电流仅由该K内未重复供应记录的前五圈估计。V/I/Q形状与IC形状代理按run_d0已有逻辑计算，所有输入均限制在对应前缀。未来目标留在独立结果列，严格物理目标和真实ICA/DVA均未认证。", "",
              "## 优势与劣势（风险）", "",
              "优势为同一NMC532/graphite体系、温度时序和较长轨迹，且新增86个电池的字段与有效支持可完整追踪。劣势（风险）为Protocol_ID、诊断身份和原始循环映射缺失，删圈与整圈替换改变供应坐标，时间修复影响真实持续时长。当前文件未包含原始两阶段CSV，未来政策状态记为unknown。", "",
              "## 置信度评级", "",
              "当前zip数量、字段、数值、哈希、标签精确匹配和供应坐标候选支持的置信度为高。源码行为与zip生成关联、真实物理循环和未来策略认证的置信度为低，需要原始CSV对照。", "",
              "## 复现与边界", "",
              "使用scripts/audit_sdu.py传入source、labels、preprocessor、output、private-output和report参数。扫描逻辑来自上一轮完整BatteryLife审计，记录了参考实现SHA及当前脚本/依赖SHA。提取的PKL和逐循环中间记录放在private-output，公开结果包含全部元数据、质量、资格和目标支持表。SDU未并入已冻结427个主要D0或主实验B。", "",
              "## 拓展思考", "",
              "原始诊断循环包含与日常工作循环不同的测量条件。恢复其身份和Protocol_ID后，可以独立审视诊断容量、工况与早期响应的对应关系。", "",
              "## 你可能想知道", "",
              "官方提供初次使用与第二寿命两个原始压缩包。当前新增SDU.zip是标准化电池文件集合，已核验的86个PKL本身不能建立两阶段原始轨迹的一一对应。", ""]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    import importlib
    import zipfile
    from pathlib import PurePosixPath

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--preprocessor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.private_output.mkdir(parents=True, exist_ok=True)
    source_before = args.source.stat()
    source_sha_before = file_sha256(args.source)
    label_before = args.labels.stat()
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    source_info = {"preprocessor_file": args.preprocessor.name,
                   "preprocessor_sha256": file_sha256(args.preprocessor),
                   "reference_auditor_sha256": REFERENCE_AUDITOR_SHA256,
                   "sdu_auditor_sha256": file_sha256(Path(__file__)),
                   "source_to_zip_generation_linkage": "UNVERIFIED",
                   "evidence_lines": {"input_columns_and_counter": [23, 90], "hardcoded_deletions": [134, 187],
                                      "diagnostic_whole_cycle_replacement": [192, 243],
                                      "renumber_and_time_normalization": [240, 265], "capacity_and_materials": [267, 288]},
                   "official_original_source": "https://zenodo.org/records/14859405",
                   "original_archives": [{"file": "Primary_use_phase.zip", "reported_size_GB": 2.7, "md5": "15d94a62d47dd9036899889b3f36992d"},
                                         {"file": "Second_life_phase.zip", "reported_size_GB": 1.8, "md5": "0ba11864d2c1017c84d96dfe578956fc"}],
                   "independent_cohort": True, "frozen_main_D0_cohort_changed": False}
    for module_name in ("run_d0", "physics_targets"):
        module = importlib.import_module(module_name)
        source_info[module_name + "_sha256"] = file_sha256(Path(module.__file__))
    dump(source_info, args.output / "sdu_preprocessing_lineage.json")
    dump({"cohort": "SDU_ADDED_SNAPSHOT_20261115", "prefix_cutoffs_supplied": [10, 50, 100],
          "response": {"direction": "discharge", "voltage_window_V": [3.4, 4.1]},
          "duplicates": "exclude later exact VIQ fingerprints, retain identity uncertainty for first record",
          "removed_indices": "persist original pre-removal indices, never apply as supplied indices",
          "feature_and_target_functions": source_info,
          "qualification_scope": "supplied-coordinate observed candidates, strict authentication remains false"},
         args.output / "sdu_audit_contract.json")
    extracted = args.private_output / "extracted"
    extracted.mkdir(parents=True, exist_ok=True)
    global ROOT
    ROOT = extracted
    records, inventory = [], []
    with zipfile.ZipFile(args.source) as archive, (args.private_output / "sdu_cells.jsonl").open("w", encoding="utf-8") as log:
        members = [entry for entry in archive.infolist() if not entry.is_dir()]
        if len({entry.filename for entry in members}) != len(members):
            raise ValueError("Archive has duplicate member names")
        for index, entry in enumerate(members, 1):
            relative = PurePosixPath(entry.filename)
            if relative.is_absolute() or ".." in relative.parts or any(":" in p for p in relative.parts):
                raise ValueError(f"Unsafe archive member {entry.filename}")
            member_meta = {"member": entry.filename, "bytes": entry.file_size,
                           "compressed_bytes": entry.compress_size, "CRC32": f"{entry.CRC:08x}",
                           "extension": Path(entry.filename).suffix.lower()}
            if not entry.filename.lower().endswith(".pkl"):
                member_meta["scope"] = "inventory_only_unhandled_non_pickle"
                inventory.append(member_meta)
                continue
            path = extracted.joinpath(*relative.parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with archive.open(entry) as stream, path.open("wb") as output:
                while block := stream.read(8 * 1024 * 1024):
                    digest.update(block)
                    output.write(block)
            member_meta.update(sha256=digest.hexdigest(), CRC_complete=True)
            inventory.append(member_meta)
            record = inspect(path)
            if record.get("sha256") != member_meta["sha256"]:
                raise ValueError(f"Extracted member hash changed {entry.filename}")
            records.append(record)
            log.write(json.dumps(safe_json(record), ensure_ascii=False) + "\n")
            log.flush()
            print(json.dumps({"completed": index, "total": len(members), "member": entry.filename,
                              "status": record["status"], "cycles": record.get("cycle_count")}), flush=True)
    summary = aggregate_sdu(records)
    dump(summary, args.output / "sdu_family_summary.json")
    write_csv(args.output / "sdu_archive_members.csv", inventory)
    metadata_rows, channel_rows, label_rows, cycle_rows = [], [], [], []
    eligible, features, prefixes, targets = [], [], [], []
    for row in records:
        if row["status"] != "OK":
            continue
        common = {"relative_path": row["relative_path"], "sha256": row["sha256"], "cell_id": row["metadata"]["cell_id"]}
        metadata_rows.append({**common, "cycle_count": row["cycle_count"], **row["metadata"], "issues": row["issues"]})
        for name, channel in row["channels"].items():
            channel_rows.append({**common, "channel": name, **channel})
        label = labels.get(Path(row["relative_path"]).name)
        target = row["d0"]["targets"]
        label_rows.append({**common, "life_label": label, "cycle_count": row["cycle_count"],
                           "max_observed_supplied_cycle": row["cycle_number"]["max"],
                           "label_exceeds_record_count": label is not None and label > row["cycle_count"],
                           "label_exceeds_max_observed_cycle": label is not None and label > row["cycle_number"]["max"],
                           "observed_event": target["eol_event"], "observed_event_cycle": target["eol_cycle"],
                           "observed_censoring": target["eol_censoring"], "observed_censor_cycle": target["eol_censor_cycle"],
                           "label_used_for_event": False, "physical_coordinate_authenticated": False})
        cycle_rows.extend({"cell_id": common["cell_id"], **cycle} for cycle in row["cycle_trace"])
        eligible.append({**row["d0"]["eligible"], "provided_file_sha256": row["sha256"]})
        features.extend(row["d0"]["features"])
        prefixes.extend(row["d0"]["prefix_eligibility"])
        targets.append(target)
    write_csv(args.output / "sdu_cell_metadata.csv", metadata_rows)
    write_csv(args.output / "sdu_channel_quality.csv", channel_rows)
    write_csv(args.output / "sdu_label_alignment.csv", label_rows)
    write_csv(args.private_output / "sdu_cycle_trace.csv", cycle_rows)
    write_csv(args.output / "sdu_eligible_cells.csv", eligible)
    write_csv(args.output / "sdu_prefix_eligibility.csv", prefixes)
    write_csv(args.output / "sdu_early_observed_features.csv", features)
    write_csv(args.output / "sdu_future_target_support.csv", targets)
    feature_coverage = []
    for cutoff in (10, 50, 100):
        selected = [row for row in prefixes if row["K_supplied"] == cutoff]
        feature_coverage.append({"family": "SDU", "K_supplied": cutoff, "N_provided": len(records),
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
    write_csv(args.output / "sdu_feature_coverage.csv", feature_coverage)
    write_csv(args.output / "sdu_future_condition_contract.csv", [{key: row[key] for key in ("cell_id", "family", "future_policy_status", "future_policy_basis", "physical_cycle_axis_authenticated")} for row in eligible])
    source_after = args.source.stat()
    label_after = args.labels.stat()
    checks = {"all_members_audited": len(records) == len(inventory) and all(row["status"] == "OK" for row in records),
              "all_CRC_complete": all(row.get("CRC_complete", False) for row in inventory),
              "all_pkl_sha_matches_extracted_stream": all(row["sha256"] == item["sha256"] for row, item in zip(records, inventory)),
              "all_channel_partitions": all(c["finite_points"] + c["nan_points"] + c["positive_inf_points"] + c["negative_inf_points"] == c["points"] for row in records for c in row["channels"].values()),
              "all_cycle_schema_counts": all(sum(s["cycles"] for s in row["cycle_schemas"]) == row["cycle_count"] for row in records),
              "all_family_presence_denominators": all(c["present_cycles"] + c["absent_cycles"] == summary["cycles"] for c in summary["channels"].values()),
              "source_stat_unchanged": (source_before.st_size, source_before.st_mtime_ns) == (source_after.st_size, source_after.st_mtime_ns),
              "source_sha256_unchanged": source_sha_before == file_sha256(args.source),
              "label_stat_unchanged": (label_before.st_size, label_before.st_mtime_ns) == (label_after.st_size, label_after.st_mtime_ns),
              "all_label_entries_match_local_member": all(key in {Path(row["relative_path"]).name for row in records} for key in labels),
              "unique_cell_ids": len({row["metadata"]["cell_id"] for row in records}) == len(records),
              "SDU_excluded_from_frozen_main_cohort": True}
    coverage = {"status": "PASS" if all(checks.values()) else "FAIL", **checks,
                "source_archive": args.source.name, "source_sha256": source_sha_before,
                "source_bytes": source_before.st_size, "pkl_expected": len(inventory), "pkl_audited": len(records),
                "cycles_scanned": summary["cycles"], "life_label_entries": len(labels),
                "life_label_sha256": file_sha256(args.labels),
                "numeric_scope": "Every numeric point in every channel of all supplied PKLs",
                "schema_scope": "All top-level metadata and every cycle schema",
                "true_physical_axis_scope": "Unverified original-cycle mapping and diagnostic replacements"}
    dump(coverage, args.output / "SDU_AUDIT_VERIFICATION.json")
    write_sdu_report(args.report, summary, coverage, label_rows, prefixes, targets, source_info)
    if not all(checks.values()):
        raise AssertionError(checks)
    print(json.dumps({"status": "PASS", "files": len(records), "cycles": summary["cycles"], "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
