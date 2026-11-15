"""Analyze the completed single-arm M4 run and fixed historical comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
from scripts.unified_matr import analyze_results as historical
from scripts.unified_matr.train_seed42 import sha256, write_json

ARM = "M4_FULL_METHOD"
REFERENCES = ("M1", "M2", "M3")
SPLITS = tuple(f"cell_fold_{number}" for number in range(5)) + ("unseen_protocol_sensitivity",)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def nested_exact(first, second):
    import torch

    if isinstance(first, torch.Tensor):
        return isinstance(second, torch.Tensor) and first.dtype == second.dtype and torch.equal(first, second)
    if isinstance(first, np.ndarray):
        return isinstance(second, np.ndarray) and first.dtype == second.dtype and np.array_equal(first, second)
    if isinstance(first, dict):
        return isinstance(second, dict) and first.keys() == second.keys() and all(nested_exact(value, second[key]) for key, value in first.items())
    if isinstance(first, (tuple, list)):
        return type(first) is type(second) and len(first) == len(second) and all(nested_exact(a, b) for a, b in zip(first, second))
    return type(first) is type(second) and first == second


def reproduction_comparison(output_root, contract):
    """Retain a failed complete repeat as a reported finding while analyzing."""
    import torch

    root = output_root / "training_reproduction"
    marker = read_json(root / "REPRODUCTION_COMPLETE.json")
    repeated = root / SPLITS[0] / ARM
    formal = output_root / "runs" / SPLITS[0] / ARM
    complete = read_json(repeated / "COMPLETE.json")
    if (marker.get("status") != "EVALUATED" or marker.get("additional_optimizer_updates") != 2100
            or marker["registration_sha256"] != sha256(root / "RUN_REGISTRATION.json")
            or read_json(root / "RUN_REGISTRATION.json") != contract
            or complete.get("successful_updates") != 2100):
        raise RuntimeError("The registered full fold0 reproduction is incomplete")
    for entry in complete["files"]:
        if sha256(repeated / entry["path"]) != entry["sha256"]:
            raise RuntimeError("Full reproduction artifact changed at " + entry["path"])
    checks = []
    def record(name, equal):
        checks.append({"comparison": name, "exact_equal": bool(equal)})
    history_columns = ("epoch", "train_loss", "validation_mae", "selected", "successful_updates")
    histories = [[{key: row[key] for key in history_columns} for row in (
        json.loads(line) for line in (path / "training.jsonl").read_text(encoding="utf-8").splitlines())] for path in (formal, repeated)]
    if len(histories[1]) != 150:
        raise RuntimeError("The full duplicate must execute all150 epochs")
    record("all150_epoch_history_core", histories[0] == histories[1])
    final = [torch.load(path / "TRAIN_STATE.pt", map_location="cpu", weights_only=False) for path in (formal, repeated)]
    for key in ("model", "optimizer", "random_state", "selected_checkpoint", "next_epoch", "best_mae", "successful_updates", "contract"):
        record("final_" + key, nested_exact(final[0][key], final[1][key]))
    selected = [torch.load(path / "SELECTED.pt", map_location="cpu", weights_only=False) for path in (formal, repeated)]
    record("selected_checkpoint_all_tensors_and_metadata", nested_exact(selected[0], selected[1]))
    for name in ("UNIT_CONTRACT.json", "TRAIN_SCALER.json", "test_metrics.json"):
        record(name, read_json(formal / name) == read_json(repeated / name))
    selections = [read_json(path / "SELECTION_FROZEN.json") for path in (formal, repeated)]
    record("selection_validation_and_epoch", {k: v for k, v in selections[0].items() if k != "checkpoint_sha256"}
           == {k: v for k, v in selections[1].items() if k != "checkpoint_sha256"})
    for name in ("test_predictions.parquet", "test_cell_metrics.csv", "test_rate_predictions.csv", "test_eol_predictions.csv", "test_mechanism_summary.csv"):
        tables = [pd.read_parquet(path / name) if name.endswith(".parquet") else pd.read_csv(path / name) for path in (formal, repeated)]
        record(name, tables[0].equals(tables[1]))
    with np.load(formal / "test_mechanism_diagnostics.npz", allow_pickle=False) as first, np.load(repeated / "test_mechanism_diagnostics.npz", allow_pickle=False) as second:
        record("all_saved_mechanism_tensors", set(first.files) == set(second.files) and all(np.array_equal(first[key], second[key]) for key in first.files))
    return {"status": "PASS" if all(row["exact_equal"] for row in checks) else "FAIL",
            "split": SPLITS[0], "seed": 42, "epochs_per_fit": 150,
            "additional_optimizer_updates": 2100, "formal_optimizer_updates": 12450,
            "comparison_checks": checks, "repeat_complete_sha256": sha256(repeated / "COMPLETE.json"),
            "first_divergent_epoch": next((i + 1 for i, (a, b) in enumerate(zip(*histories)) if a != b), None),
            "scope_limitation": "One full fold0 duplicate in one registered environment. Cross-seed variance is unmeasured."}


def completed_units(output_root, data_root):
    """Authenticate the complete six-unit delivery before reading test tables."""
    run_root = output_root / "runs"
    matrix = read_json(run_root / "MATRIX_COMPLETE.json")
    contract = read_json(output_root / "MODEL_CONTRACT.json")
    registration = read_json(run_root / "RUN_REGISTRATION.json")
    if (matrix.get("status") != "COMPLETE" or matrix.get("completed_units") != 6
            or matrix.get("successful_updates") != 12450 or contract != registration
            or matrix["registration_sha256"] != sha256(run_root / "RUN_REGISTRATION.json")
            or matrix["model_contract_sha256"] != sha256(output_root / "MODEL_CONTRACT.json")
            or contract["data_freeze_sha256"] != sha256(data_root / "DATA_FREEZE.json")):
        raise RuntimeError("The complete registered six-unit delivery is required")
    for entry in contract["sources"]:
        if sha256(REPOSITORY / entry["path"]) != entry["sha256"]:
            raise RuntimeError("Registered M4 source changed at " + entry["path"])
    for entry in contract["frozen_files"]:
        if sha256(data_root / entry["path"]) != entry["sha256"]:
            raise RuntimeError("Frozen data changed at " + entry["path"])
    missing = [split for split in SPLITS if not (run_root / split / ARM / "COMPLETE.json").is_file()]
    if missing:
        raise RuntimeError("Formal unit completion is pending for " + ", ".join(missing))
    units = []
    for split in SPLITS:
        root = run_root / split / ARM
        unit = read_json(root / "UNIT_CONTRACT.json")
        selection = read_json(root / "SELECTION_FROZEN.json")
        complete = read_json(root / "COMPLETE.json")
        if (unit["arm"] != ARM or unit["split_name"] != split
                or unit["registration_sha256"] != matrix["registration_sha256"]
                or complete["status"] != "EVALUATED"
                or complete["successful_updates"] != contract["expected_updates_by_split"][split]
                or selection["checkpoint_sha256"] != sha256(root / "SELECTED.pt")):
            raise RuntimeError("Unit identity, selection or budget differs at " + split)
        for entry in complete["files"]:
            if sha256(root / entry["path"]) != entry["sha256"]:
                raise RuntimeError("Completed unit artifact changed at " + split + "/" + entry["path"])
        units.append({"split": split, "arm": ARM, "root": root, "contract": unit, "selection": selection})
    return contract, units


def historical_units(history_root, cells, future, split_contract, new_contract):
    registration = read_json(history_root / "RUN_REGISTRATION.json")
    matrix = read_json(history_root / "MATRIX_COMPLETE.json")
    if (matrix["status"] != "COMPLETE" or matrix["completed_units"] != 42
            or matrix["registration_sha256"] != sha256(history_root / "RUN_REGISTRATION.json")
            or registration["data_freeze_sha256"] != new_contract["data_freeze_sha256"]):
        raise RuntimeError("Historical reference must be the sealed completed same-data matrix")
    results, bound = [], []
    for split in SPLITS:
        for arm in REFERENCES:
            root = history_root / split / arm
            complete = read_json(root / "COMPLETE.json")
            for entry in complete["files"]:
                if sha256(root / entry["path"]) != entry["sha256"]:
                    raise RuntimeError("Historical completed artifact changed")
            unit = {"split": split, "arm": arm, "root": root,
                    "contract": read_json(root / "UNIT_CONTRACT.json"),
                    "selection": read_json(root / "SELECTION_FROZEN.json")}
            results.append(historical.load_unit_results(unit, cells, future, split_contract, registration))
            bound.append({"split": split, "arm": arm, "complete_sha256": sha256(root / "COMPLETE.json")})
    return results, bound


def fixed_comparisons(metrics):
    rows, pairs = [], []
    for scope in ("main_oof", "unseen_protocol_sensitivity"):
        scoped = metrics.loc[metrics.scope == scope]
        for metric in sorted(scoped.metric.unique()):
            wide = scoped.loc[scoped.metric == metric].pivot(index=["cell_uid", "protocol_uid"], columns="arm", values="value")
            for reference in REFERENCES:
                pair = wide[[reference, ARM]].rename(columns={reference: "reference_error", ARM: "candidate_error"}).reset_index()
                if not pair.reference_error.notna().equals(pair.candidate_error.notna()):
                    raise RuntimeError("Historical paired support differs")
                pair["improvement"] = pair.reference_error - pair.candidate_error
                pair = pair.assign(scope=scope, metric=metric, reference=reference, candidate=ARM)
                pairs.extend(pair.to_dict("records"))
                valid = pair.dropna(subset=["improvement"]).sort_values("cell_uid")
                protocols = valid.groupby("protocol_uid", sort=True)[["reference_error", "candidate_error", "improvement"]].mean()
                for estimator, sample in (("cell_mean", valid), ("protocol_macro", protocols)):
                    estimate = historical.paired_mean_bootstrap(sample.improvement.to_numpy(float))
                    first, last = sample.reference_error.mean(), sample.candidate_error.mean()
                    rows.append({"scope": scope, "metric": metric, "candidate": ARM, "reference": reference,
                                 "estimator": estimator, "reference_metric": first, "candidate_metric": last,
                                 **estimate,
                                 "relative_improvement_percent": (first - last) / first * 100 if first > 0 else None,
                                 "paired_cells": len(valid), "protocol_groups": len(protocols),
                                 "improved_cells": int(valid.improvement.gt(0).sum()),
                                 "worsened_cells": int(valid.improvement.lt(0).sum()),
                                 "tied_cells": int(valid.improvement.eq(0).sum()),
                                 "bootstrap_seed": 42, "bootstrap_replicates": 2000,
                                 "comparison_status": "historical_same_data_exposed_internal_single_seed"})
    return pd.DataFrame(rows), pd.DataFrame(pairs)


def eol_summary(eol, cells):
    truth = cells[["protocol_uid", "eol80_event_observed", "eol80_first_observed_crossing_cycle", "last_observed_valid_cycle"]].copy()
    truth["outcome_cycle"] = np.where(truth.eol80_event_observed, truth.eol80_first_observed_crossing_cycle, truth.last_observed_valid_cycle)
    merged = eol.merge(truth, left_on="cell_uid", right_index=True, validate="many_to_one")
    rows, contrasts = [], []
    for scope, frame in merged.groupby("scope", sort=True):
        risk = frame.loc[frame.outcome_cycle > 100]
        wide = risk.pivot(index="cell_uid", columns="arm", values="predicted_trend_eol80_cycle").sort_index()
        outcome = truth.loc[wide.index]
        times = outcome.outcome_cycle.to_numpy(float)
        events = outcome.eol80_event_observed.to_numpy(bool)
        predictions = {arm: wide[arm].to_numpy(float) for arm in (ARM,) + REFERENCES}
        scores = {}
        for arm, predicted in predictions.items():
            result = historical.harrell_concordance(times, events, predicted)
            scores[arm] = result["harrell_c"]
            arm_rows = frame.loc[frame.arm == arm]
            rows.append({"scope": scope, "arm": arm, **result,
                         "executed_test_cells": len(arm_rows), "at_risk100_cells": len(outcome),
                         "observed_events": int(events.sum()), "right_censored": int((~events).sum()),
                         "predicted_no_crossing_through5000": int(arm_rows.no_crossing_through_registered_horizon.sum())})
        bootstrap = historical.eol_bootstrap(times, events, predictions, outcome.protocol_uid.to_numpy(str))
        for unit, sampled in bootstrap.items():
            for reference in REFERENCES:
                delta = sampled[ARM] - sampled[reference]
                valid = delta[np.isfinite(delta)]
                low, high = np.percentile(valid, [2.5, 97.5]) if len(valid) else (np.nan, np.nan)
                contrasts.append({"scope": scope, "candidate": ARM, "reference": reference,
                                  "sampling_unit": unit, "reference_metric": scores[reference], "candidate_metric": scores[ARM],
                                  "improvement": scores[ARM] - scores[reference], "ci95_low": low, "ci95_high": high,
                                  "bootstrap_supported": len(valid), "bootstrap_requested": 2000,
                                  "bootstrap_seed": 42, "paired_cells": len(outcome)})
    return pd.DataFrame(rows), pd.DataFrame(contrasts)


def query_region_rows(uid, scope, split, queries, observed, last_followup, output):
    raw = output["raw_trend"]
    unclamped = raw + output["observation_residual"]
    rate = output["rate_per_1000_cycles"]
    rows = []
    for region, mask in (("observed_future", observed),
                         ("beyond_cell_last_followup", queries > last_followup),
                         ("all_queries_100_5000", np.ones(len(queries), dtype=bool))):
        count = int(mask.sum())
        selected_raw, selected_observation, selected_rate = raw[mask], unclamped[mask], rate[mask]
        raw_negative = int((selected_raw < 0).sum())
        trend_clipped = int(((selected_raw < 0) | (selected_raw > 1.2)).sum())
        observation_clipped = int(((selected_observation < 0) | (selected_observation > 1.2)).sum())
        rows.append({"scope": scope, "split": split, "arm": ARM, "cell_uid": uid, "query_region": region,
                     "query_count": count, "query_min_cycle": float(queries[mask].min()) if count else None,
                     "query_max_cycle": float(queries[mask].max()) if count else None,
                     "last_observed_valid_cycle": last_followup,
                     "raw_trend_below_zero_count": raw_negative,
                     "raw_trend_below_zero_fraction": raw_negative / count if count else None,
                     "trend_clipped_count": trend_clipped,
                     "trend_clipped_fraction": trend_clipped / count if count else None,
                     "observation_clipped_count": observation_clipped,
                     "observation_clipped_fraction": observation_clipped / count if count else None,
                     "raw_trend_min": float(selected_raw.min()) if count else None,
                     "raw_trend_max": float(selected_raw.max()) if count else None,
                     "unclamped_observation_min": float(selected_observation.min()) if count else None,
                     "unclamped_observation_max": float(selected_observation.max()) if count else None,
                     "rate_min_per1000_cycles": float(selected_rate.min()) if count else None,
                     "rate_max_per1000_cycles": float(selected_rate.max()) if count else None,
                     "rate_min_per_cycle": float(selected_rate.min() / 1000) if count else None,
                     "rate_max_per_cycle": float(selected_rate.max() / 1000) if count else None})
    return rows


def summarize_query_regions(frame):
    rows = []
    for (scope, region), part in frame.groupby(["scope", "query_region"], sort=True):
        count = int(part.query_count.sum())
        record = {"scope": scope, "query_region": region, "supported_cells": int(part.query_count.gt(0).sum()),
                  "query_count": count, "query_min_cycle": part.query_min_cycle.min(), "query_max_cycle": part.query_max_cycle.max(),
                  "raw_trend_min": part.raw_trend_min.min(), "raw_trend_max": part.raw_trend_max.max(),
                  "unclamped_observation_min": part.unclamped_observation_min.min(),
                  "unclamped_observation_max": part.unclamped_observation_max.max(),
                  "rate_min_per1000_cycles": part.rate_min_per1000_cycles.min(),
                  "rate_max_per1000_cycles": part.rate_max_per1000_cycles.max(),
                  "rate_min_per_cycle": part.rate_min_per_cycle.min(), "rate_max_per_cycle": part.rate_max_per_cycle.max()}
        for name in ("raw_trend_below_zero", "trend_clipped", "observation_clipped"):
            affected = int(part[name + "_count"].sum())
            record[name + "_count"] = affected
            record[name + "_fraction"] = affected / count if count else np.nan
            record[name + "_affected_cells"] = int(part[name + "_count"].gt(0).sum())
        rows.append(record)
    return pd.DataFrame(rows)


def query_range_diagnostics(data_root, units, contract, device):
    import torch
    from scripts.unified_matr import train_methodology_full as training

    training.configure_execution()
    inputs, identifiers, eligible, cells, _, _, observed, _, _, splits, _ = training.load_frozen_data(data_root)
    index = {uid: number for number, uid in enumerate(identifiers)}
    queries = np.arange(100, 5001, dtype=np.float32)
    records = []
    for unit in units:
        roles = splits["splits"][unit["split"]]
        rows = {role: np.asarray([index[uid] for uid in roles[role] if eligible[index[uid]]]) for role in ("train", "test")}
        scaled, _ = training.fold_scaler(inputs, rows["train"])
        model = training.make_model(inputs, contract["training"]).to(device).eval()
        checkpoint = torch.load(unit["root"] / "SELECTED.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        scope = "main_oof" if unit["split"].startswith("cell_fold_") else "unseen_protocol_sensitivity"
        with torch.no_grad():
            for start in range(0, len(rows["test"]), 8):
                part = rows["test"][start:start + 8]
                output = model(**training.to_batch(scaled, part, device), query_cycles=torch.as_tensor(queries, device=device))
                arrays = {key: output[key].cpu().numpy() for key in ("raw_trend", "observation_residual", "rate_per_1000_cycles")}
                for local, number in enumerate(part):
                    uid = identifiers[number]
                    support = np.zeros(len(queries), dtype=bool)
                    support[1:1 + observed.shape[1]] = observed[number]
                    records.extend(query_region_rows(uid, scope, unit["split"], queries, support,
                                                     int(cells.loc[uid, "last_observed_valid_cycle"]),
                                                     {key: value[local] for key, value in arrays.items()}))
        del model, scaled
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    frame = pd.DataFrame(records)
    return frame, summarize_query_regions(frame)


def render_report(root, summary, units, eol, mechanism, reproduction, query_summary):
    lines = ["# M4 完整 Methodology 的 seed42 正式结果", "",
             "六个固定划分各运行150 epochs，正式更新12450次。完整fold0复验额外2100次，单epoch工程smoke另14次，实际优化更新合计14564次。fresh-batch重复与数学oracle均为0次优化更新。验证集全轨迹 cell-mean MAE 首次最小值选择权重，选权冻结后评分测试集。", "",
             "## 完整轨迹与时距", "",
             "|范围|指标|单位|cell mean|protocol macro|支持电芯|协议|观测数|", "|---|---|---|---|---|---|---|---|"]
    for row in summary.loc[summary.arm == ARM].itertuples(index=False):
        lines.append(f"|{row.scope}|{row.metric}|{row.unit}|{row.cell_mean:.9g}|{row.protocol_macro:.9g}|{row.supported_cells}|{row.supported_protocols}|{row.observations}|")
    lines.extend(["", "## 六单元与选权", "",
                  "训练目标固定为trajectory MAE + 0.1 × (rate MAE × 1000) + 0.001 × persistence penalty。trajectory项先在每颗电芯的全部有效未来观测上求均值。rate项先在每颗电芯的受支持窗口上求均值，再在有窗口支持的电芯间平均。日志train_loss为该epoch训练batch总目标的均值。", "",
                  "|split|epochs|selected epoch|validation MAE|最终epoch train loss|selected epoch train loss|updates|", "|---|---|---|---|---|---|---|"])
    for row in units.itertuples(index=False):
        lines.append(f"|{row.split}|{row.epochs}|{row.selected_epoch}|{row.validation_mae:.9g}|{row.final_epoch_train_loss:.9g}|{row.selected_epoch_train_loss:.9g}|{row.successful_updates}|")
    lines.extend(["", "## 历史同数据对照", "", "|范围|模型|全轨迹MAE|全轨迹RMSE|", "|---|---|---|---|"])
    for scope in ("main_oof", "unseen_protocol_sensitivity"):
        selected = summary.loc[(summary.scope == scope) & summary.metric.isin(("trajectory_mae", "trajectory_rmse"))]
        wide = selected.pivot(index="arm", columns="metric", values="cell_mean")
        for arm, row in wide.iterrows():
            lines.append(f"|{scope}|{arm}|{row.trajectory_mae:.9g}|{row.trajectory_rmse:.9g}|")
    lines.extend(["", "## EOL80", "", "|范围|模型|Harrell C|可比较对|风险集电芯|观测事件|右删失|", "|---|---|---|---|---|---|---|"])
    for row in eol.itertuples(index=False):
        lines.append(f"|{row.scope}|{row.arm}|{row.harrell_c:.9g}|{row.comparable_pairs}|{row.at_risk100_cells}|{row.observed_events}|{row.right_censored}|")
    lines.extend(["", "## 查询范围与截断", "",
                  "速率极值覆盖固定100–5000查询轴，单位为SOH/1000 cycles，除以1000得到SOH/cycle。观测监督点按每颗电芯的真实future mask选择。超出随访范围的查询始于该电芯最后有效容量观测之后。all查询还包含100锚点及随访内缺测位置。", "",
                  "表中比例按查询点计数。raw负值表示未截断积分趋势小于0，观测截断表示raw趋势加小幅观测残差后超出[0,1.2]。", "",
                  "|范围|查询区域|电芯|查询点|raw<0比例|观测截断比例|rate最小/1000|rate最大/1000|", "|---|---|---|---|---|---|---|---|"])
    for row in query_summary.itertuples(index=False):
        lines.append(f"|{row.scope}|{row.query_region}|{row.supported_cells}|{row.query_count}|{row.raw_trend_below_zero_fraction:.6g}|{row.observation_clipped_fraction:.6g}|{row.rate_min_per1000_cycles:.6g}|{row.rate_max_per1000_cycles:.6g}|")
    lines.extend(["", "100–5000范围的有限、非负速率和单调积分属于数值性质。超过实际随访的速率与趋势缺少同范围的测量校准。", ""])
    lines.extend(["", "## 机制与限制", "",
                  f"保存{len(mechanism)}个测试角色电芯的机制回执。rho、写入率、真实间隔、漂移方向、两层B和Bprime的L2变化、速率极值及查询一致性见mechanism_summary.csv。", "",
                  "全部169颗电芯已经参与历史研究。本轮结果为固定问题下的内部探索性证据，历史M1/M2/M3对照沿用已保存预测。固定seed42没有测量训练种子方差。配对bootstrap仅条件于这些已拟合预测。", "",
                  f"六个正式划分完成后，fold0从seed42重新初始化重复完整150epochs，额外2100次优化更新独立计数。完整训练逐值复现检查为{reproduction['status']}，逐项结果见TRAINING_REPRODUCTION.json。该检查覆盖同环境同seed的一组划分，跨种子方差仍未测量。旧七臂矩阵M2与NR的完整训练逐值重复为0/6，根因仍未认证。", "",
                  "M4首次工程首批检查曾出现预测与loss逐值一致、25组参数梯度逐值失配。失败源码、合同与回执已归档。后续工程改动、重新注册及首批检查结论见FAILURE_HISTORY.json和SMOKE_QA.json。", "",
                  "loaded Q(V)及其导数表示带负载响应。物理分解与写入诊断属于模型内部量，电化学LLI/LAM机制归因仍需独立测量。", ""])
    (root / "FULL_MODEL_FINAL_RESULTS_ZH.md").write_text("\n".join(lines), encoding="utf-8")


def analyze(data_root, output_root, history_root, device="cuda"):
    registration, units = completed_units(output_root, data_root)
    reproduction = reproduction_comparison(output_root, registration)
    cells, future, split = historical.frozen_tables(data_root)
    results = [historical.load_unit_results(unit, cells, future, split, registration) for unit in units]
    old_results, historical_binding = historical_units(history_root, cells, future, split, registration)
    all_results = results + old_results
    metrics = pd.DataFrame([row for result in all_results for row in result["metrics"]])
    summaries, protocols, folds = historical.metric_summaries(metrics)
    comparisons, pairs = fixed_comparisons(metrics)
    eol = pd.concat([result["eol"] for result in all_results], ignore_index=True)
    eol_scores, eol_comparisons = eol_summary(eol, cells)
    mechanism = pd.concat([pd.read_csv(unit["root"] / "test_mechanism_summary.csv").assign(split=unit["split"], arm=ARM) for unit in units], ignore_index=True)
    training = pd.DataFrame([{**result["training"],
                              "final_epoch_train_loss": result["history"][-1]["train_loss"],
                              "selected_epoch_train_loss": result["history"][result["training"]["selected_epoch"] - 1]["train_loss"]}
                             for result in results])
    query_cells, query_summary = query_range_diagnostics(data_root, units, registration, device)
    root = output_root / "analysis"
    root.mkdir(parents=True, exist_ok=True)
    if (root / "ANALYSIS_COMPLETE.json").exists():
        raise RuntimeError("Completed analysis is immutable")
    tables = {
        "cell_metrics.csv": metrics, "summary_metrics.csv": summaries,
        "protocol_metrics.csv": protocols, "split_metrics.csv": folds,
        "historical_paired_comparisons.csv": comparisons, "historical_paired_cells.csv": pairs,
        "rate_predictions_and_errors.csv": pd.DataFrame([row for result in all_results for row in result["rates"]]),
        "eol80_predictions.csv": eol, "eol80_summary.csv": eol_scores,
        "eol80_historical_comparisons.csv": eol_comparisons,
        "training_units.csv": training, "mechanism_summary.csv": mechanism,
        "query_range_cell_diagnostics.csv": query_cells, "query_range_summary.csv": query_summary,
    }
    for name, table in tables.items():
        table.to_csv(root / name, index=False)
    write_json(root / "TRAINING_REPRODUCTION.json", reproduction)
    render_report(root, summaries, training, eol_scores, mechanism, reproduction, query_summary)
    metadata = {
        "status": "ANALYZED" if reproduction["status"] == "PASS" else "ANALYZED_WITH_REPRODUCTION_FAILURE", "arm": ARM, "seed": 42, "completed_units": 6,
        "successful_updates": int(training.successful_updates.sum()),
        "formal_optimizer_updates": int(training.successful_updates.sum()),
        "full_reproduction_optimizer_updates": 2100, "engineering_optimizer_updates": 14,
        "total_executed_optimizer_updates": int(training.successful_updates.sum()) + 2100 + 14,
        "model_contract_sha256": sha256(output_root / "MODEL_CONTRACT.json"),
        "analysis_source": {"path": str(Path(__file__).relative_to(REPOSITORY)).replace("\\", "/"), "sha256": sha256(__file__)},
        "reused_statistics_source_sha256": sha256(REPOSITORY / "scripts/unified_matr/analyze_results.py"),
        "historical_reference_bindings": historical_binding,
        "historical_exposure": "internal exploratory same169 cells, seed42 fitted predictions only",
        "complete_duplicate_training_executed": True,
        "complete_duplicate_training_status": reproduction["status"],
        "additional_reproduction_optimizer_updates": 2100,
        "old_matrix_duplicate_training_exact_matches": "0/6 for M2 versus NR, unresolved",
        "engineering_failure_history": {"path": "FAILURE_HISTORY.json", "sha256": sha256(output_root / "FAILURE_HISTORY.json")}
            if (output_root / "FAILURE_HISTORY.json").exists() else None,
        "bootstrap": {"replicates": 2000, "seed": 42, "scope": "conditional on fitted predictions"},
        "files": [{"path": name, "sha256": sha256(root / name)} for name in sorted(tables)] + [
            {"path": "TRAINING_REPRODUCTION.json", "sha256": sha256(root / "TRAINING_REPRODUCTION.json")},
            {"path": "FULL_MODEL_FINAL_RESULTS_ZH.md", "sha256": sha256(root / "FULL_MODEL_FINAL_RESULTS_ZH.md")}],
    }
    write_json(root / "ANALYSIS_COMPLETE.json", metadata)
    print(json.dumps({"status": metadata["status"], "units": 6, "updates": metadata["successful_updates"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--historical-runs", type=Path, default=REPOSITORY / "results/unified_matr_seed42/runs")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    analyze(args.data, args.output_root, args.historical_runs, args.device)


if __name__ == "__main__":
    main()
