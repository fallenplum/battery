"""Analyze four independent M4 ablations after all formal and repeat fits."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import struct
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
from scripts.unified_matr import train_methodology_ablations as training
from scripts.unified_matr import analyze_results as statistics
from scripts import analyze_methodology_full_seed42 as full

ARMS, SPLITS = training.ARMS, training.SPLITS
FULL = "M4_FULL_METHOD"
read_json, sha256, write_json = training.read_json, training.sha256, training.write_json


def strict_nested_equal(first, second):
    """Compare checkpoint trees directly, preserving types and floating bits."""
    if type(first) is not type(second):
        return False
    if isinstance(first, torch.Tensor):
        if first.dtype != second.dtype or first.shape != second.shape:
            return False
        first_bytes = first.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        second_bytes = second.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        return first_bytes == second_bytes
    if isinstance(first, np.ndarray):
        if first.dtype.hasobject or second.dtype.hasobject:
            raise TypeError("Object checkpoint arrays require explicit value comparison")
        return (first.dtype.str == second.dtype.str and first.dtype.descr == second.dtype.descr
                and first.shape == second.shape
                and np.ascontiguousarray(first).tobytes() == np.ascontiguousarray(second).tobytes())
    if isinstance(first, dict):
        if len(first) != len(second):
            return False
        unmatched = list(second.items())
        for key, value in first.items():
            match = next((index for index, (other_key, other_value) in enumerate(unmatched)
                          if strict_nested_equal(key, other_key) and strict_nested_equal(value, other_value)), None)
            if match is None:
                return False
            unmatched.pop(match)
        return True
    if isinstance(first, (list, tuple)):
        return len(first) == len(second) and all(strict_nested_equal(a, b) for a, b in zip(first, second))
    if isinstance(first, np.generic):
        return first.dtype.str == second.dtype.str and first.tobytes() == second.tobytes()
    if isinstance(first, float):
        return struct.pack(">d", first) == struct.pack(">d", second)
    if first is None or isinstance(first, (str, bytes, int, bool)):
        return first == second
    raise TypeError("Unsupported checkpoint leaf type " + type(first).__qualname__)


def strict_frame_equal(first, second):
    if not first.columns.equals(second.columns) or not first.dtypes.equals(second.dtypes) or first.shape != second.shape:
        return False
    for name in first.columns:
        a, b = first[name].to_numpy(), second[name].to_numpy()
        if a.dtype.hasobject:
            if not strict_nested_equal(a.tolist(), b.tolist()):
                return False
        elif not strict_nested_equal(a, b):
            return False
    return True


def completed_units(output_root, data_root):
    contract = read_json(output_root / "MODEL_CONTRACT.json")
    run_root = output_root / "runs"
    registration = read_json(run_root / "RUN_REGISTRATION.json")
    matrix = read_json(run_root / "MATRIX_COMPLETE.json")
    if (contract != registration or matrix["status"] != "COMPLETE" or matrix["completed_units"] != 24
            or matrix["successful_updates"] != 49800 or tuple(contract["arms"]) != ARMS
            or matrix["registration_sha256"] != sha256(run_root / "RUN_REGISTRATION.json")
            or matrix["model_contract_sha256"] != sha256(output_root / "MODEL_CONTRACT.json")
            or contract["data_freeze_sha256"] != sha256(data_root / "DATA_FREEZE.json")):
        raise RuntimeError("The complete same-data24-unit delivery is required")
    if sha256(REPOSITORY / contract["ablation_contract"]["path"]) != contract["ablation_contract"]["sha256"]:
        raise RuntimeError("Registered ablation research contract changed")
    for item in contract["sources"]:
        if sha256(REPOSITORY / item["path"]) != item["sha256"]:
            raise RuntimeError("Registered source changed at " + item["path"])
    for item in contract["frozen_files"]:
        if sha256(data_root / item["path"]) != item["sha256"]:
            raise RuntimeError("Frozen data changed at " + item["path"])
    if any(not (run_root / split / arm / "COMPLETE.json").exists() for arm in ARMS for split in SPLITS):
        raise RuntimeError("All24 formal units must complete before test analysis")
    units = []
    for arm in ARMS:
        for split in SPLITS:
            root = run_root / split / arm
            unit = read_json(root / "UNIT_CONTRACT.json")
            selection = read_json(root / "SELECTION_FROZEN.json")
            complete = read_json(root / "COMPLETE.json")
            if (unit["arm"] != arm or unit["split_name"] != split
                    or unit["registration_sha256"] != matrix["registration_sha256"]
                    or complete["status"] != "EVALUATED"
                    or complete["successful_updates"] != contract["expected_updates_by_split"][split]
                    or selection["checkpoint_sha256"] != sha256(root / "SELECTED.pt")):
                raise RuntimeError("Formal unit identity or selection differs")
            for item in complete["files"]:
                if sha256(root / item["path"]) != item["sha256"]:
                    raise RuntimeError("Completed artifact changed")
            units.append({"arm": arm, "split": split, "root": root, "contract": unit, "selection": selection})
    return contract, units


def reproduction_comparisons(output_root, contract):
    reproduction = output_root / "training_reproduction"
    marker = read_json(reproduction / "REPRODUCTION_COMPLETE.json")
    if (marker["status"] != "EVALUATED" or tuple(marker["arms"]) != ARMS
            or marker["additional_optimizer_updates"] != 8400
            or read_json(reproduction / "RUN_REGISTRATION.json") != contract
            or marker["registration_sha256"] != sha256(reproduction / "RUN_REGISTRATION.json")):
        raise RuntimeError("All four registered full training reproductions are required")
    results = []
    for arm in ARMS:
        formal = output_root / "runs" / SPLITS[0] / arm
        repeated = reproduction / SPLITS[0] / arm
        complete = read_json(repeated / "COMPLETE.json")
        if complete["status"] != "EVALUATED" or complete["successful_updates"] != 2100:
            raise RuntimeError("Full reproduction update budget differs")
        for item in complete["files"]:
            if sha256(repeated / item["path"]) != item["sha256"]:
                raise RuntimeError("Reproduction artifact changed")
        checks = []
        def record(name, equal):
            checks.append({"comparison": name, "exact_equal": bool(equal)})
        core = ("epoch", "train_loss", "validation_mae", "selected", "successful_updates")
        histories = [[{key: row[key] for key in core} for row in (
            json.loads(line) for line in (path / "training.jsonl").read_text(encoding="utf-8").splitlines())] for path in (formal, repeated)]
        if any(len(history) != 150 for history in histories):
            raise RuntimeError("Complete reproduction must execute150 epochs")
        record("all150_epoch_history_core", strict_nested_equal(histories[0], histories[1]))
        states = [torch.load(path / "TRAIN_STATE.pt", map_location="cpu", weights_only=False) for path in (formal, repeated)]
        for key in ("model", "optimizer", "random_state", "selected_checkpoint", "next_epoch", "best_mae", "successful_updates", "contract"):
            record("final_" + key, strict_nested_equal(states[0][key], states[1][key]))
        selected = [torch.load(path / "SELECTED.pt", map_location="cpu", weights_only=False) for path in (formal, repeated)]
        record("selected_checkpoint_all_tensors_and_metadata", strict_nested_equal(selected[0], selected[1]))
        for name in ("UNIT_CONTRACT.json", "TRAIN_SCALER.json", "test_metrics.json"):
            record(name, strict_nested_equal(read_json(formal / name), read_json(repeated / name)))
        choices = [{key: value for key, value in read_json(path / "SELECTION_FROZEN.json").items() if key != "checkpoint_sha256"} for path in (formal, repeated)]
        record("selection_validation_and_epoch", strict_nested_equal(choices[0], choices[1]))
        for name in ("test_predictions.parquet", "test_cell_metrics.csv", "test_rate_predictions.csv", "test_eol_predictions.csv", "test_mechanism_summary.csv"):
            tables = [pd.read_parquet(path / name) if name.endswith("parquet") else pd.read_csv(path / name) for path in (formal, repeated)]
            record(name, strict_frame_equal(tables[0], tables[1]))
        with np.load(formal / "test_mechanism_diagnostics.npz", allow_pickle=False) as first, np.load(repeated / "test_mechanism_diagnostics.npz", allow_pickle=False) as second:
            record("all_saved_mechanism_tensors", set(first.files) == set(second.files) and all(strict_nested_equal(first[key], second[key]) for key in first.files))
        results.append({"arm": arm, "status": "PASS" if all(row["exact_equal"] for row in checks) else "FAIL",
                        "epochs_per_fit": 150, "additional_optimizer_updates": 2100,
                        "comparison_checks": checks, "complete_sha256": sha256(repeated / "COMPLETE.json"),
                        "first_divergent_epoch": next((i + 1 for i, (a, b) in enumerate(zip(*histories)) if not strict_nested_equal(a, b)), None)})
    return results


def load_m4_reference(contract, data_root, cells, future, split):
    root = REPOSITORY / contract["m4_reference"]["root"]
    for name, key in (("MODEL_CONTRACT.json", "model_contract_sha256"),
                      ("RUN_VERIFICATION.json", "verification_sha256"),
                      ("analysis/ANALYSIS_COMPLETE.json", "analysis_sha256")):
        if sha256(root / name) != contract["m4_reference"][key]:
            raise RuntimeError("Sealed M4 reference changed")
    registration, units = full.completed_units(root, data_root)
    for item in read_json(root / "analysis/ANALYSIS_COMPLETE.json")["files"]:
        if sha256(root / "analysis" / item["path"]) != item["sha256"]:
            raise RuntimeError("Sealed M4 analysis artifact changed")
    results = [statistics.load_unit_results(unit, cells, future, split, registration) for unit in units]
    return results, root


def paired_comparisons(metrics):
    records, cells = [], []
    for scope in ("main_oof", "unseen_protocol_sensitivity"):
        for metric in sorted(metrics.metric.unique()):
            scoped = metrics.loc[(metrics.scope == scope) & (metrics.metric == metric)]
            wide = scoped.pivot(index=["cell_uid", "protocol_uid"], columns="arm", values="value")
            for arm in ARMS:
                pair = wide[[arm, FULL]].rename(columns={arm: "reference_error", FULL: "candidate_error"}).reset_index()
                if not pair.reference_error.notna().equals(pair.candidate_error.notna()):
                    raise RuntimeError("Full-versus-ablation support differs")
                pair["improvement"] = pair.reference_error - pair.candidate_error
                pair = pair.assign(scope=scope, metric=metric, candidate=FULL, reference=arm, contrast=FULL + "_vs_" + arm)
                cells.extend(pair.to_dict("records"))
                valid = pair.dropna(subset=["improvement"]).sort_values("cell_uid")
                groups = valid.groupby("protocol_uid", sort=True)[["reference_error", "candidate_error", "improvement"]].mean()
                for estimator, frame in (("cell_mean", valid), ("protocol_macro", groups)):
                    first, last = frame.reference_error.mean(), frame.candidate_error.mean()
                    records.append({"scope": scope, "metric": metric, "candidate": FULL, "reference": arm,
                                    "contrast": FULL + "_vs_" + arm, "estimator": estimator,
                                    "reference_metric": first, "candidate_metric": last,
                                    **statistics.paired_mean_bootstrap(frame.improvement.to_numpy(float)),
                                    "relative_full_improvement_percent": (first - last) / first * 100 if first > 0 else np.nan,
                                    "paired_cells": len(valid), "protocol_groups": len(groups),
                                    "full_better_cells": int(valid.improvement.gt(0).sum()),
                                    "ablation_better_cells": int(valid.improvement.lt(0).sum()),
                                    "tied_cells": int(valid.improvement.eq(0).sum()),
                                    "bootstrap_seed": 42, "bootstrap_replicates": 2000,
                                    "direction": "ablation_error_minus_full_error_positive_favors_full"})
    return pd.DataFrame(records), pd.DataFrame(cells)


def eol_statistics(eol, cells):
    truth = cells[["protocol_uid", "eol80_event_observed", "eol80_first_observed_crossing_cycle", "last_observed_valid_cycle"]].copy()
    truth["outcome_cycle"] = np.where(truth.eol80_event_observed, truth.eol80_first_observed_crossing_cycle, truth.last_observed_valid_cycle)
    joined = eol.merge(truth, left_on="cell_uid", right_index=True, validate="many_to_one")
    scores, comparisons = [], []
    for scope, frame in joined.groupby("scope", sort=True):
        risk = frame.loc[frame.outcome_cycle > 100]
        wide = risk.pivot(index="cell_uid", columns="arm", values="predicted_trend_eol80_cycle").sort_index()
        target = truth.loc[wide.index]
        times, events = target.outcome_cycle.to_numpy(float), target.eol80_event_observed.to_numpy(bool)
        predictions = {arm: wide[arm].to_numpy(float) for arm in (FULL,) + ARMS}
        current = {}
        for arm, predicted in predictions.items():
            value = statistics.harrell_concordance(times, events, predicted)
            current[arm] = value["harrell_c"]
            sample = frame.loc[frame.arm == arm]
            scores.append({"scope": scope, "arm": arm, **value, "executed_test_cells": len(sample),
                           "at_risk100_cells": len(target), "observed_events": int(events.sum()),
                           "right_censored": int((~events).sum()),
                           "predicted_no_crossing_through5000": int(sample.no_crossing_through_registered_horizon.sum())})
        sampled = statistics.eol_bootstrap(times, events, predictions, target.protocol_uid.to_numpy(str))
        for unit, draws in sampled.items():
            for arm in ARMS:
                delta = draws[FULL] - draws[arm]
                valid = delta[np.isfinite(delta)]
                low, high = np.percentile(valid, [2.5, 97.5]) if len(valid) else (np.nan, np.nan)
                comparisons.append({"scope": scope, "candidate": FULL, "reference": arm,
                                    "sampling_unit": unit, "reference_metric": current[arm], "candidate_metric": current[FULL],
                                    "improvement": current[FULL] - current[arm], "ci95_low": low, "ci95_high": high,
                                    "bootstrap_supported": len(valid), "bootstrap_requested": 2000, "bootstrap_seed": 42,
                                    "paired_cells": len(target), "direction": "full_C_minus_ablation_C_positive_favors_full"})
    return pd.DataFrame(scores), pd.DataFrame(comparisons)


def monotonic_diagnostics(units):
    records = []
    for unit in units:
        prediction = pd.read_parquet(unit["root"] / "test_predictions.parquet")
        with np.load(unit["root"] / "test_mechanism_diagnostics.npz", allow_pickle=False) as archive:
            identifiers = archive["cell_uid"].astype(str).tolist()
            raw, rate = archive["flow__raw_trend"], archive["flow__rate_per_1000_cycles"]
            for uid, frame in prediction.groupby("cell_uid", sort=True):
                frame = frame.sort_values("original_cycle_id")
                positions = frame.original_cycle_id.to_numpy(int) - 101
                number = identifiers.index(uid)
                record = {"arm": unit["arm"], "split": unit["split"], "cell_uid": uid,
                          "scope": "main_oof" if unit["split"].startswith("cell_fold_") else "unseen_protocol_sensitivity",
                          "observed_queries": len(frame), "observed_adjacent_pairs": max(len(frame) - 1, 0),
                          "negative_signed_rate_points": int((rate[number, positions] < 0).sum()),
                          "nonnegative_rate_structure": unit["arm"] != ARMS[0]}
                for name, values in (("raw_trend", raw[number, positions]), ("trend", frame.trend.to_numpy(float)),
                                     ("prediction", frame.prediction.to_numpy(float))):
                    delta = np.diff(values)
                    record[name + "_rise_count"] = int((delta > 0).sum())
                    record[name + "_rise_over1e6_count"] = int((delta > 1e-6).sum())
                    record[name + "_largest_positive_step"] = float(max(delta.max(initial=0), 0))
                records.append(record)
    return pd.DataFrame(records)


def query_diagnostics(data_root, units, contract, device):
    training.configure_execution()
    data = training.load_frozen_data(data_root)
    inputs, identifiers, eligible, cells, _, _, observed, _, _, splits, _ = data
    lookup = {uid: i for i, uid in enumerate(identifiers)}
    queries = np.arange(100, 5001, dtype=np.float32)
    records = []
    for unit in units:
        roles = splits["splits"][unit["split"]]
        indices = {role: np.array([lookup[uid] for uid in roles[role] if eligible[lookup[uid]]]) for role in ("train", "test")}
        scaled, _ = training.fold_scaler(inputs, indices["train"])
        model = training.make_model(unit["arm"], inputs, contract["training"]).to(device).eval()
        model.load_state_dict(torch.load(unit["root"] / "SELECTED.pt", map_location=device, weights_only=False)["model"])
        scope = "main_oof" if unit["split"].startswith("cell_fold_") else "unseen_protocol_sensitivity"
        with torch.no_grad():
            for start in range(0, len(indices["test"]), 8):
                part = indices["test"][start:start + 8]
                output = model(**training.to_batch(scaled, part, device), query_cycles=torch.as_tensor(queries, device=device))
                arrays = {name: output[name].cpu().numpy() for name in ("raw_trend", "observation_residual", "rate_per_1000_cycles")}
                for local, number in enumerate(part):
                    supported = np.zeros(len(queries), bool)
                    supported[1:1 + observed.shape[1]] = observed[number]
                    rows = full.query_region_rows(identifiers[number], scope, unit["split"], queries, supported,
                                                  int(cells.loc[identifiers[number], "last_observed_valid_cycle"]),
                                                  {name: value[local] for name, value in arrays.items()})
                    records.extend({**row, "arm": unit["arm"]} for row in rows)
        del model, scaled
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return pd.DataFrame(records)


def query_summary(frame):
    return pd.concat([full.summarize_query_regions(group).assign(arm=arm) for arm, group in frame.groupby("arm", sort=True)], ignore_index=True)


def render_report(path, summary, units, paired, eol, eol_paired, reproductions, monotonic, query, counterexample):
    lines = ["# M4独立消融seed42正式结果", "",
             "A1至A4各完成原六个划分，共24项正式fit与49800次优化更新。四个fold0完整重复另8400次，四个单epoch工程smoke另56次，实际总更新58256次。对照M4使用已封存预测。", "",
             "训练目标沿用trajectory MAE + 0.1 × (端点rate MAE × 1000) + 0.001 × persistence penalty。选权使用validation全轨迹cell-mean MAE的strict first-min，训练固定150epochs。", "",
             "## 全轨迹与四时距", "", "|范围|模型|指标|单位|cell mean|protocol macro|电芯|协议|观测点|",
             "|---|---|---|---|---|---|---|---|---|"]
    for row in summary.itertuples(index=False):
        lines.append(f"|{row.scope}|{row.arm}|{row.metric}|{row.unit}|{row.cell_mean:.9g}|{row.protocol_macro:.9g}|{row.supported_cells}|{row.supported_protocols}|{row.observations}|")
    lines.extend(["", "## 训练与选权", "", "|模型|split|epoch|selected|val MAE|末epoch loss|selected epoch loss|updates|", "|---|---|---|---|---|---|---|---|"])
    for row in units.itertuples(index=False):
        lines.append(f"|{row.arm}|{row.split}|{row.epochs}|{row.selected_epoch}|{row.validation_mae:.8g}|{row.final_epoch_train_loss:.8g}|{row.selected_epoch_train_loss:.8g}|{row.successful_updates}|")
    lines.extend(["", "## 全轨迹配对差异", "",
                  "误差差值为消融误差减完整M4误差，正值支持完整M4。区间为条件于已拟合seed42预测的配对bootstrap区间。", "",
                  "|范围|消融|指标|估计量|M4改进|95%区间|配对电芯|协议组|", "|---|---|---|---|---|---|---|---|"])
    for row in paired.loc[paired.metric.isin(("trajectory_mae", "trajectory_rmse"))].itertuples(index=False):
        lines.append(f"|{row.scope}|{row.reference}|{row.metric}|{row.estimator}|{row.improvement:.8g}|[{row.ci95_low:.8g}, {row.ci95_high:.8g}]|{row.paired_cells}|{row.protocol_groups}|")
    lines.extend(["", "各时距及rate的完整配对区间、逐cell正负差异保存在full_vs_ablation_paired_comparisons.csv和full_vs_ablation_paired_cells.csv。", ""])
    lines.extend(["", "## EOL80", "", "|范围|模型|Harrell C|可比较对|风险电芯|事件|删失|", "|---|---|---|---|---|---|---|"])
    for row in eol.itertuples(index=False):
        lines.append(f"|{row.scope}|{row.arm}|{row.harrell_c:.8g}|{row.comparable_pairs}|{row.at_risk100_cells}|{row.observed_events}|{row.right_censored}|")
    lines.extend(["", "|范围|消融|重采样单位|M4 C改进|95%区间|风险集电芯|", "|---|---|---|---|---|---|"])
    for row in eol_paired.itertuples(index=False):
        lines.append(f"|{row.scope}|{row.reference}|{row.sampling_unit}|{row.improvement:.8g}|[{row.ci95_low:.8g}, {row.ci95_high:.8g}]|{row.paired_cells}|")
    lines.extend(["", "## 重复、趋势与查询限制", ""])
    for result in reproductions:
        lines.append(f"- {result['arm']}完整fold0重复为{result['status']}，150epoch核心历史（epoch、loss、validation MAE、selected、updates）及最终model/optimizer/RNG/selected与预测共20项严格比较，运行耗时独立记录。")
    rises = monotonic.groupby(["scope", "arm"], sort=True)[["observed_adjacent_pairs", "raw_trend_rise_count", "trend_rise_count", "prediction_rise_count"]].sum().reset_index()
    lines.extend(["", "A1保留原解析Flow，其signed rate可为负，趋势回升按原预测记录。A2至A4保留非负rate积分。观测残差会影响最终prediction的局部变化。", "",
                  "|范围|模型|实际相邻观测对|raw trend上升|clamped trend上升|prediction上升|", "|---|---|---|---|---|---|"])
    for row in rises.itertuples(index=False):
        lines.append(f"|{row.scope}|{row.arm}|{row.observed_adjacent_pairs}|{row.raw_trend_rise_count}|{row.trend_rise_count}|{row.prediction_rise_count}|")
    lines.extend(["", "查询诊断分别统计真实observed future点、每cell最后有效随访之后至5000，以及all100..5000。率单位为SOH/1000cycles，除以1000为SOH/cycle。A1的signed loss兼容字段来自解析decoder差值，其signed rate来自该差值导数。", "",
                  "未截断raw趋势和raw加观测残差的越界clip统计保存在query_range_summary.csv。有限性与单调性属于数值性质，超出实际随访的曲线仍缺少物理校准。MATR_b1c2的逐模型完整轨迹误差与clip分母保存在known_counterexample_MATR_b1c2.csv。", "",
                  "|范围|模型|查询区域|查询点|raw低于0比例|观测clip比例|rate最小/1000|rate最大/1000|", "|---|---|---|---|---|---|---|---|"])
    for row in query.itertuples(index=False):
        lines.append(f"|{row.scope}|{row.arm}|{row.query_region}|{row.query_count}|{row.raw_trend_below_zero_fraction:.6g}|{row.observation_clipped_fraction:.6g}|{row.rate_min_per1000_cycles:.6g}|{row.rate_max_per1000_cycles:.6g}|")
    lines.extend(["", "|MATR_b1c2范围|模型|MAE|RMSE|实际观测点|观测clip点|", "|---|---|---|---|---|---|"])
    for row in counterexample.itertuples(index=False):
        lines.append(f"|{row.scope}|{row.arm}|{row.trajectory_mae:.8g}|{row.trajectory_rmse:.8g}|{row.query_count}|{row.observation_clipped_count}|")
    lines.extend(["",
                  "## 比较口径、优势与风险", "",
                  "配对误差改进定义为消融误差减完整M4误差，正值支持完整M4。EOL改进定义为完整M4的C减少融C。保留逐cell、协议组及2000次seed42配对bootstrap。bootstrap条件于已拟合预测。", "",
                  "优势来自共同输入、标签、split、训练与选权预算以及独立的单项机制替换。风险包括单seed方差未覆盖、结构性参数量差异、169电芯与M4历史结果已经暴露，以及A4保留raw/reference VIQ中的局部曲线信息。内部网络写入诊断的电化学因果解释仍需独立测量。", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze(data_root, output_root, device):
    contract, units = completed_units(output_root, data_root)
    reproductions = reproduction_comparisons(output_root, contract)
    cells, future, split = statistics.frozen_tables(data_root)
    current = [statistics.load_unit_results(unit, cells, future, split, contract) for unit in units]
    reference, reference_root = load_m4_reference(contract, data_root, cells, future, split)
    combined = current + reference
    metrics = pd.DataFrame([row for result in combined for row in result["metrics"]])
    summary, protocols, splits_summary = statistics.metric_summaries(metrics)
    paired, paired_cells = paired_comparisons(metrics)
    eol = pd.concat([result["eol"] for result in combined], ignore_index=True)
    eol_summary, eol_paired = eol_statistics(eol, cells)
    training_units = pd.DataFrame([{**result["training"], "final_epoch_train_loss": result["history"][-1]["train_loss"],
                                    "selected_epoch_train_loss": result["history"][result["training"]["selected_epoch"] - 1]["train_loss"]} for result in current])
    monotonic = monotonic_diagnostics(units)
    query_cells = pd.concat([query_diagnostics(data_root, units, contract, device),
                             pd.read_csv(reference_root / "analysis/query_range_cell_diagnostics.csv")], ignore_index=True)
    query = query_summary(query_cells)
    mechanism = pd.concat([pd.read_csv(unit["root"] / "test_mechanism_summary.csv").assign(split=unit["split"], arm=unit["arm"]) for unit in units], ignore_index=True)
    known = metrics.loc[(metrics.cell_uid == "MATR_b1c2") & metrics.metric.isin(("trajectory_mae", "trajectory_rmse"))]
    known = known.pivot(index=["scope", "split", "arm", "cell_uid"], columns="metric", values="value").reset_index()
    known = known.merge(query_cells.loc[(query_cells.cell_uid == "MATR_b1c2") & (query_cells.query_region == "observed_future")],
                        on=["scope", "split", "arm", "cell_uid"], validate="one_to_one")
    root = output_root / "analysis"
    root.mkdir(parents=True, exist_ok=True)
    if (root / "ANALYSIS_COMPLETE.json").exists():
        raise RuntimeError("Completed analysis is immutable")
    tables = {"cell_metrics.csv": metrics, "summary_metrics.csv": summary, "protocol_metrics.csv": protocols,
              "split_metrics.csv": splits_summary, "full_vs_ablation_paired_comparisons.csv": paired,
              "full_vs_ablation_paired_cells.csv": paired_cells,
              "rate_predictions_and_errors.csv": pd.DataFrame([row for result in combined for row in result["rates"]]),
              "eol80_predictions.csv": eol, "eol80_summary.csv": eol_summary, "eol80_paired_comparisons.csv": eol_paired,
              "training_units.csv": training_units, "mechanism_summary.csv": mechanism,
              "observed_monotonic_diagnostics.csv": monotonic, "query_range_cell_diagnostics.csv": query_cells,
              "query_range_summary.csv": query, "known_counterexample_MATR_b1c2.csv": known}
    for name, frame in tables.items():
        frame.to_csv(root / name, index=False)
    write_json(root / "TRAINING_REPRODUCTIONS.json", {"arms": reproductions})
    render_report(root / "ABLATION_RESULTS_ZH.md", summary, training_units, paired, eol_summary, eol_paired, reproductions, monotonic, query, known)
    status = "ANALYZED" if all(result["status"] == "PASS" for result in reproductions) else "ANALYZED_WITH_REPRODUCTION_FAILURE"
    metadata = {"status": status, "arms": list(ARMS), "seed": 42, "completed_units": 24,
                "formal_optimizer_updates": 49800, "full_reproduction_optimizer_updates": 8400,
                "engineering_optimizer_updates": 56, "total_executed_optimizer_updates": 58256,
                "model_contract_sha256": sha256(output_root / "MODEL_CONTRACT.json"),
                "analysis_source_sha256": sha256(__file__),
                "statistics_sources": [{"path": name, "sha256": sha256(REPOSITORY / name)} for name in (
                    "scripts/unified_matr/analyze_results.py", "scripts/analyze_methodology_full_seed42.py")],
                "reference_m4": contract["m4_reference"], "bootstrap": {"replicates": 2000, "seed": 42, "conditional_on_fitted_predictions": True},
                "files": [{"path": name, "sha256": sha256(root / name)} for name in sorted(tables)] + [
                    {"path": name, "sha256": sha256(root / name)} for name in ("TRAINING_REPRODUCTIONS.json", "ABLATION_RESULTS_ZH.md")]}
    write_json(root / "ANALYSIS_COMPLETE.json", metadata)
    print(json.dumps({"status": status, "formal_units": 24, "actual_optimizer_updates": 58256}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    analyze(args.data.resolve(), args.output_root.resolve(), args.device)


if __name__ == "__main__":
    main()
