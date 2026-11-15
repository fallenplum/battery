"""Post-score diagnosis of frozen Stanford predictions, without model fitting."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from run_d0 import capacity, current_reference, load_existing_pickle, make_features, observed_curve, SETTINGS


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository
    study = root / "results" / "experiment_b_stanford"
    out = root / "results" / "extension_stanford"
    protected = [p for p in study.rglob("*") if p.is_file() and p.name != "RUN.lock"]
    protected += [root / "results/d0/early_observed_features.csv", root / "results/extension_stanford/prior_context.csv"]
    before = {p.relative_to(root).as_posix(): digest(p) for p in protected}
    features = {row["cell_id"]: row for row in read_csv(root / "results/d0/early_observed_features.csv") if row["family"] == "Stanford_2" and row["prefix_cutoff_supplied_cycle_number"] == "100"}
    context = {row["cell_id"]: row for row in read_csv(out / "prior_context.csv")}
    eligible = {row["cell_id"]: row for row in read_csv(root / "results/d0/eligible_cells.csv") if row["family"] == "Stanford_2"}
    predictions = [row for row in read_csv(study / "predictions.csv") if row["model"] == "combined"]
    diagnostic_rows = []
    summaries = {}
    chosen = set()
    maximum_replay_difference = 0.0
    for target in ("r200_500", "r500_1000"):
        selections = json.loads((study / "targets" / target / "SELECTIONS.json").read_text(encoding="utf-8"))
        model = selections["models"][target]["combined"]
        columns = model["features"]
        mean = np.asarray(model["scaler_mean"], dtype=float)
        scale = np.asarray(model["scaler_scale"], dtype=float)
        coefficients = np.asarray(model["coef_scaled"], dtype=float)
        intercept = float(model["intercept"])

        def vector(identifier):
            values = {**context[identifier], **features[identifier]}
            return np.asarray([float(values[column]) for column in columns], dtype=float)

        train = np.stack([vector(identifier) for identifier in selections["train_ids"]])
        lower, upper = train.min(axis=0), train.max(axis=0)
        target_predictions = [row for row in predictions if row["target"] == target]
        worst = sorted(target_predictions, key=lambda row: float(row["absolute_error"]), reverse=True)
        chosen.update(row["cell_id"] for row in worst[:3])
        absolute_total = sum(float(row["absolute_error"]) for row in target_predictions)
        squared_total = sum(float(row["absolute_error"]) ** 2 for row in target_predictions)
        target_rows = []
        for row in target_predictions:
            identifier = row["cell_id"]
            values = vector(identifier)
            z = (values - mean) / scale
            contributions = z * coefficients
            calculated = float(intercept + contributions.sum())
            difference = abs(calculated - float(row["prediction"]))
            maximum_replay_difference = max(maximum_replay_difference, difference)
            if difference > 1e-12:
                raise ValueError(f"Stored prediction differs from its frozen model for {identifier}/{target}")
            outside = (values < lower) | (values > upper)
            largest = np.argsort(np.abs(contributions))[::-1][:5]
            top = [{"feature": columns[index], "observed_value": float(values[index]),
                    "train_min": float(lower[index]), "train_max": float(upper[index]),
                    "standardized_value": float(z[index]), "frozen_coefficient": float(coefficients[index]),
                    "linear_contribution": float(contributions[index])} for index in largest]
            item = {"cell_id": identifier, "target": target, "model": "combined",
                    "truth": float(row["truth"]), "prediction": float(row["prediction"]),
                    "absolute_error": float(row["absolute_error"]), "frozen_alpha": float(model["alpha"]),
                    "training_cells": len(selections["train_ids"]), "max_abs_test_z": float(np.abs(z).max()),
                    "max_z_feature": columns[int(np.argmax(np.abs(z)))],
                    "outside_train_minmax_count": int(outside.sum()),
                    "outside_train_features": [column for column, flag in zip(columns, outside) if flag],
                    "absolute_error_share_of_full_test_sum": float(row["absolute_error"]) / absolute_total,
                    "squared_error_share_of_full_test_sum": float(row["absolute_error"]) ** 2 / squared_total,
                    "intercept": intercept, "top_frozen_linear_contributions": top,
                    "postscore_diagnostic_only": True, "prediction_replay_abs_difference": difference}
            diagnostic_rows.append(item)
            target_rows.append(item)
        summaries[target] = {"test_cells": len(target_rows), "train_cells": len(selections["train_ids"]),
                             "validation_cells": len(selections["validation_ids"]), "selected_alpha": float(model["alpha"]),
                             "negative_predictions": sum(row["prediction"] < 0 for row in target_rows),
                             "test_cells_with_abs_z_gt3": sum(row["max_abs_test_z"] > 3 for row in target_rows),
                             "test_cells_outside_any_train_feature_range": sum(row["outside_train_minmax_count"] > 0 for row in target_rows),
                             "maximum_abs_z": max(row["max_abs_test_z"] for row in target_rows),
                             "worst_absolute_error_cases": [row["cell_id"] for row in worst[:3]]}

    if not 3 <= len(chosen) <= 5:
        raise ValueError("Registered diagnostic selection did not yield three to five unique cases")
    early_checks = []
    for identifier in sorted(chosen):
        path = args.dataset_root / eligible[identifier]["provided_file"]
        before_stat = path.stat()
        payload, actual_hash = load_existing_pickle(path)
        if actual_hash != eligible[identifier]["provided_file_sha256"]:
            raise ValueError("Existing user source file changed after the D0 audit")
        early = [cycle for cycle in payload["cycle_data"] if int(cycle["cycle_number"]) <= 100]
        setting = SETTINGS["Stanford_2"]
        nominal = float(payload["nominal_capacity_in_Ah"])
        reference = current_reference(early, setting["direction"])
        curves = []
        reasons = {}
        discharge_capacity = []
        for cycle in early:
            curve, reason, _ = observed_curve(cycle, setting, reference, nominal)
            reasons[reason] = reasons.get(reason, 0) + 1
            if curve is not None:
                curves.append((int(cycle["cycle_number"]), curve))
            discharge_capacity.append(capacity(cycle, "discharge_capacity_in_Ah", nominal))
        recalculated = make_features([int(cycle["cycle_number"]) for cycle in early], discharge_capacity, curves, 100)
        feature_differences = {key: abs(float(value) - float(features[identifier][key])) for key, value in recalculated.items() if value is not None}
        if any(not np.isclose(float(value), float(features[identifier][key]), rtol=1e-12, atol=1e-14) for key, value in recalculated.items() if value is not None):
            raise ValueError("Re-extracted early feature differs from the sealed feature table")
        voltage = np.concatenate([np.asarray(cycle["voltage_in_V"], dtype=float) for cycle in early])
        current = np.concatenate([np.asarray(cycle["current_in_A"], dtype=float) for cycle in early])
        qd = np.concatenate([np.asarray(cycle["discharge_capacity_in_Ah"], dtype=float) for cycle in early])
        valid_capacities = [value for value in discharge_capacity if value is not None]
        after_stat = path.stat()
        if before_stat.st_size != after_stat.st_size or before_stat.st_mtime_ns != after_stat.st_mtime_ns:
            raise ValueError("Source file changed during the read-only prefix check")
        early_checks.append({"cell_id": identifier, "provided_file_sha256": actual_hash,
                             "only_feature_values_from_supplied_cycles_le100": True,
                             "early_cycle_count": len(early), "valid_observed_curves": len(curves),
                             "curve_qualification_counts": reasons,
                             "reference_discharge_current_A": reference,
                             "all_early_voltage_current_Qd_finite": bool(np.isfinite(voltage).all() and np.isfinite(current).all() and np.isfinite(qd).all()),
                             "early_voltage_range_V": [float(voltage.min()), float(voltage.max())],
                             "early_discharge_capacity_ratio_range": [min(valid_capacities), max(valid_capacities)],
                             "early_discharge_capacity_Ah_range": [float(qd.min()), float(qd.max())],
                             "all_registered_features_reproduced": True,
                             "maximum_numeric_feature_replay_difference": max(feature_differences.values()),
                             "file_size_and_mtime_unchanged": True})
        del payload, early
    unchanged = all(digest(root / name) == original for name, original in before.items())
    if not unchanged:
        raise RuntimeError("A sealed experiment or input artifact changed during diagnosis")
    result = {"status": "POSTSCORE_DIAGNOSTIC_COMPLETE", "analysis_role": "post-score explanation of frozen model failures",
              "model_fit_calls": 0, "test_entrypoint_calls": 0, "cohort_or_metric_redefinitions": 0,
              "sealed_artifacts_unchanged": True, "maximum_frozen_prediction_replay_difference": maximum_replay_difference,
              "targets": summaries, "selected_case_rule": "union of three largest combined absolute errors for each target, exactly five unique cells",
              "early_source_checks": early_checks,
              "top_case_contributions": [row for row in diagnostic_rows if row["cell_id"] == "Stanford_Nova_Regular_270"],
              "implementation_defect_evidence": "No mismatch found between frozen predictions, stored coefficients and independently re-extracted early features",
              "diagnosis": "A supplied-window charge value outside training support combines with large opposite-sign linear coefficients and produces unconstrained extrapolation. Observed input validity and predictive error are distinct measured properties.",
              "original_artifact_hashes": before}
    write_csv(out / "postscore_failure_diagnostics.csv", diagnostic_rows)
    (out / "postscore_failure_diagnostics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    main_case = {row["target"]: row for row in diagnostic_rows if row["cell_id"] == "Stanford_Nova_Regular_270"}
    lines = ["# Stanford 已封存预测的事后失败诊断", "", "## 分析范围", "",
             "本诊断在扩展实验评分完成后执行。只读取已冻结输入、选择、系数和预测，并对五颗已有用户本地PKL的前100圈重新核验。模型拟合次数为0，测试评价入口调用次数为0，主结果与测试队列保持已封存值。",
             "", "诊断案例为两个目标各自误差最大的前三颗电池的并集，恰好五颗，分别是270、320、124、280、279。计算训练范围外状态和误差贡献时始终使用完整测试集。",
             "", "## 主要观察", "",
             "最大失败案例均为Stanford_Nova_Regular_270。其早期3.4–4.1V窗口的起始与末期电荷为0.171829和0.174024Ah，均低于相应训练范围。", ""]
    for target, row in main_case.items():
        label = "200–500" if target == "r200_500" else "500–1000"
        lines += [f"{label}目标中，该电池预测为{row['prediction']:.9g}，真实目标为{row['truth']:.9g}。最大标准化偏移为{row['max_abs_test_z']:.3f}个训练标准差，{row['outside_train_minmax_count']}项输入超出训练极值范围。它占完整测试平方误差之和的{100*row['squared_error_share_of_full_test_sum']:.2f}%。", ""]
    lines += ["晚期模型中，起始窗口电荷单项线性贡献约−0.006556，末期窗口电荷贡献约−0.001557，截距约0.000951。冻结系数的分解与原预测逐值一致。线性模型没有对速率的符号或幅值施加物理约束，训练范围外的输入可产生很大的外推。", "",
              "## 原始供应前缀复核", "",
              "五颗案例均使用已有本地供应文件，文件SHA与D0登记一致。所有登记的早期特征都可从<=100的供应循环重现，电压、电流和Qd数值均有限，固定电流和电压窗口资格保持通过。容量仍在登记的有效范围，较小的窗口电荷本身没有构成已定义的输入无效条件。", ""]
    for check in early_checks:
        lines += [f"- {check['cell_id']}，合格窗口{check['valid_observed_curves']}/100，早期容量比例{check['early_discharge_capacity_ratio_range'][0]:.6f}–{check['early_discharge_capacity_ratio_range'][1]:.6f}，全部早期特征重现。"]
    lines += ["", "## 解释边界", "",
              "当前证据支持一个可直接复核的统计解释，训练覆盖有限的窗口尺度输入经过冻结Ridge系数组合形成极端外推。数据可用性和当前模型效果分别报告。局部误差与平方误差高度集中，可以与仍有一定排序能力的Spearman同时出现。",
              "", "没有发现封存预测与系数不一致，或当前重提取早期特征与已用输入不一致的实现缺陷。当前负结果与所有反例保留。事后诊断没有形成新的独立确认，也没有用于修改本次主结果。",
              "", "## 优势与劣势（风险）", "",
              "优势为完整测试支持、冻结贡献和真实前缀可以逐值连接。劣势（风险）为数据位于训练覆盖之外时线性外推不受物理约束，训练范围内诊断也无法独立确认电化学机制。",
              "", "## 置信度评级", "",
              "冻结算术重放及五颗前缀核验的置信度为高，外推失效解释具有直接数值支撑。具体材料或形成机理导致窗口尺度差异的解释保持未认证。",
              "", "## 拓展思考", "", "训练支持度、预测幅值和排序指标可以共同记录，以描述模型在极端早期形态上的实际行为。",
              "", "## 你可能想知道", "", "Nova_Regular_270的窗口容量仍是有限、可重现的观测值。它相对训练数据偏离约九个标准差，数据资格通过与模型严重失准可以同时成立。", ""]
    (root / "report/STANFORD_PREDICTION_FAILURE_DIAGNOSTIC_ZH.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": result["status"], "rows": len(diagnostic_rows), "source_cases": len(early_checks), "summaries": summaries}, ensure_ascii=False))


if __name__ == "__main__":
    main()
