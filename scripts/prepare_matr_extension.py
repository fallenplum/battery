"""Prepare prediction-time MATR design context and supplied-coordinate qualification."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path


SEVERSON_URL = "https://www.nature.com/articles/s41560-019-0356-8"
ATTIA_URL = "https://escholarship.org/content/qt117547r6/qt117547r6.pdf"
AUTHOR_LOADER_URL = "https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation/blob/master/Load%20Data.ipynb"
B4_DESIGNS = {
    (3.6, 6.0, 5.6, 4.755), (4.4, 5.6, 5.2, 4.252),
    (4.8, 5.2, 5.2, 4.160), (5.2, 5.2, 4.8, 4.160),
    (6.0, 5.6, 4.4, 3.834), (7.0, 4.8, 4.8, 3.652),
    (8.0, 4.4, 4.4, 3.940), (8.0, 6.0, 4.8, 3.000),
    (8.0, 7.0, 5.2, 2.680),
}


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def normalized_soc(value):
    value = float(value)
    converted = value / 100.0 if value > 1 else value
    if not 0 <= converted <= 1:
        raise ValueError("SOC is outside the supported percentage/fraction range")
    return converted


def parse_original_policy(text):
    stages = [part for part in text.split("-") if "new" not in part]
    if len(stages) != 2:
        raise ValueError("Recovered original MATR policy is not a two-step design")
    match = re.fullmatch(r"([0-9.]+)C\(([0-9.]+)%\)", stages[0])
    if match is None:
        raise ValueError("Original MATR policy string is unsupported")
    rate1, percent = map(float, match.groups())
    rate2 = float(stages[1].removesuffix("C"))
    return [rate1, rate2], [percent / 100.0, 0.8]


def count_support(cells, eligible, targets):
    rows = []
    for target in ("r200_500", "r500_1000"):
        supported = [name for name in cells if targets[name][target] != ""]
        legacy = sum(eligible[name]["legacy_576_member"] == "True" for name in supported)
        rows.append({"target": target, "N_supported": len(supported), "N_legacy_576": legacy,
                     "N_new_to_576": len(supported) - legacy,
                     "by_batch": dict(Counter(name.split("_")[1].split("c")[0] for name in supported))})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--d0-root", type=Path, required=True)
    parser.add_argument("--policy-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    metadata_rows = [row for row in read_rows(args.metadata_csv) if row["family"] == "MATR"]
    metadata = {json.loads(row["metadata"])["cell_id"]: json.loads(row["metadata"]) for row in metadata_rows}
    eligible = {row["cell_id"]: row for row in read_rows(args.d0_root / "eligible_cells.csv") if row["family"] == "MATR"}
    targets = {row["cell_id"]: row for row in read_rows(args.d0_root / "future_target_support.csv") if row["family"] == "MATR"}
    feature_rows = [row for row in read_rows(args.d0_root / "early_observed_features.csv") if row["family"] == "MATR" and row["prefix_cutoff_supplied_cycle_number"] == "100"]
    if not len(metadata) == len(eligible) == len(targets) == len(feature_rows) == 169:
        raise ValueError("MATR evidence does not contain the same 169-cell cohort")
    policies, source_files = {}, []
    for batch in (1, 2, 3):
        path = args.policy_root / f"batch{batch}_original_policies.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload["static_policy_only"] or payload["summary_curves_cycle_life_targets_or_model_metrics_read"]:
            raise ValueError("Policy recovery has unexpected future-value access")
        source_files.append({"file": path.name, "sha256": file_hash(path), "policies": len(payload["policies"]),
                             "source_full_file_bytes": payload["range_source"]["full_file_bytes"],
                             "whole_original_file_sha256_verified": payload["range_source"]["whole_file_sha256_verified"]})
        for entry in payload["policies"]:
            name = entry["processed_cell_id"].removesuffix(".pkl")
            if name in metadata:
                if name in policies:
                    raise ValueError("More than one recovered policy maps to the same supplied cell")
                policies[name] = entry
    contexts, details, source_counter = [], [], Counter()
    normalized_cells = 0
    b4_profiles = Counter()
    for name, meta in sorted(metadata.items()):
        provided = meta["charge_protocol"]
        rates = [float(step["rate_in_C"]) for step in provided]
        source_entry = policies.get(name)
        if source_entry:
            original_rates, fast_ends = parse_original_policy(source_entry["policy_original"])
            provided_switch = normalized_soc(provided[0]["end_soc"])
            if rates != original_rates or not math.isclose(provided_switch, fast_ends[0], abs_tol=1e-12):
                raise ValueError("Supplied charge rates or normalized first switch differ from the recovered original policy")
            normalized_cells += int(float(provided[0]["end_soc"]) > 1)
            source_kind = "original_HDF5_policy_recovered_and_numeric_match"
            original_text = source_entry["policy_original"]
            source_url = source_entry["source_url"]
            publication = SEVERSON_URL
        else:
            if not name.startswith("MATR_b4") or len(rates) != 4:
                raise ValueError("MATR prior has neither recovered two-step policy nor supported four-step design")
            if tuple(rates) not in B4_DESIGNS:
                raise ValueError("Batch4 supplied rate vector does not match the original published validation design")
            fast_ends = [0.2, 0.4, 0.6, 0.8]
            if not all(math.isclose(float(step["end_soc"]), end, abs_tol=1e-12) for step, end in zip(provided, fast_ends)):
                raise ValueError("Batch4 supplied boundaries differ from published equal-SOC windows")
            b4_profiles[tuple(rates)] += 1
            source_kind = "provided_four_step_vector_matches_primary_validation_design"
            original_text = ""
            source_url = "https://data.matr.io/1/projects/5d80e633f405260001c0b60a/batches/5dcef1fe110002c7215b2c94"
            publication = ATTIA_URL
        discharge = meta["discharge_protocol"]
        discharge_rate = float(discharge[0]["rate_in_C"])
        if discharge_rate != 4:
            raise ValueError("Unexpected supplied discharge design")
        row = {"cell_id": name, "family": "MATR",
               "protocol_group": json.dumps({"fast_charge_rates_C": rates, "fast_charge_end_soc": fast_ends, "discharge_rate_C": discharge_rate}, separators=(",", ":")),
               "prior_charge_stage_count": len(rates)}
        for index in range(4):
            row[f"prior_charge_rate_{index+1}_C"] = rates[index] if index < len(rates) else 0.0
            row[f"prior_charge_rate_{index+1}_missing"] = 0 if index < len(rates) else 1
        row.update(prior_charge_first_switch_soc=fast_ends[0],
                   prior_charge_first_switch_soc_missing=0,
                   prior_fast_charge_end_soc=0.8,
                   prior_common_topoff_rate_C=1.0,
                   prior_discharge_rate_C=discharge_rate,
                   source_kind=source_kind, source_policy_string=original_text,
                   source_batch=int(re.search(r"MATR_b([0-9]+)", name).group(1)),
                   source_policy_archive_recovered=bool(source_entry),
                   source_url=source_url, primary_design_reference=publication,
                   source_future_values_used=False,
                   future_policy_status="observed_only",
                   context_scope="prior_assigned_design_parameters_without_future_measured_signals")
        for key, value in row.items():
            if key.startswith("prior_") and (not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise ValueError("A model-facing prior column is nonnumeric or nonfinite")
        contexts.append(row)
        source_counter[source_kind] += 1
        details.append({"cell_id": name, "provided_charge_protocol": provided,
                        "provided_discharge_protocol": discharge,
                        "corrected_fast_stage_end_soc": fast_ends,
                        "source_kind": source_kind,
                        "original_policy": source_entry,
                        "supplied_max_voltage_limit_V": meta["max_voltage_limit_in_V"],
                        "primary_design_max_voltage_V": 3.6,
                        "provided_cycle_axis_authenticated": eligible[name]["physical_cycle_axis_authenticated"] == "True"})
    if len(policies) != 124 or len(b4_profiles) != 9 or set(b4_profiles.values()) != {5}:
        raise ValueError("MATR prior recovery/published-profile counts do not match the source contract")
    support = count_support(metadata, eligible, targets)
    summary = {
        "status": "READY_FOR_SUPPLIED_COORDINATE_EXPLORATORY_EXTENSION",
        "family": "MATR", "provided_cells": 169,
        "supplied_prefix_K100_complete": sum(row["prefix_K100_numbers_complete"] == "True" for row in eligible.values()),
        "observed_curve_K100_at_least80": sum(int(row["curve_valid_records_K100"]) >= 80 for row in eligible.values()),
        "legacy_576_cells": sum(row["legacy_576_member"] == "True" for row in eligible.values()),
        "new_to_576_cells": sum(row["legacy_576_member"] != "True" for row in eligible.values()),
        "supplied_EOL80_events": sum(row["eol_event"] == "True" for row in targets.values()),
        "supplied_EOL80_right_censored": sum(row["eol_censoring"] == "right" for row in targets.values()),
        "strict_original_cycle_axis_authenticated_cells": 0,
        "strict_true_knee_events": 0,
        "model_eligibility_uses": ["supplied cycle number and measurement definition", "per-target observed support", "prefix-only features", "static design context"],
        "metadata_history_gap_is_model_exclusion_rule": False,
        "future_target_measurement": "supplied discharge-active maximum minus finite record baseline / declared1.1Ah",
        "future_target_coordinate": "supplied_cycle_number",
        "Qv_measurement": "discharge,2.8-3.3V,64-point oriented window charge relative to lower voltage endpoint",
        "target_support": support,
        "source_policy_counts": dict(source_counter),
        "percent_SOC_first_switch_normalized_cells": normalized_cells,
        "b4_published_four_step_profiles": [{"rates_C": list(k), "cells": v} for k, v in sorted(b4_profiles.items())],
        "prior_columns": [key for key in contexts[0] if key.startswith("prior_")],
        "context_missingness": "absent charge stages use0 and missing1, explicit stage_count separates structure",
        "SOC_semantics": {"percent_conversion": "for first switch values>1 and<=100 divide by100", "fast_charge_terminal_soc": 0.8, "common_topoff": "published1C CC-CV from0.8 to1.0", "supplied_two_step_end_soc1": "retained as supplied in details but excluded from physical fast-stage interpretation"},
        "physical_identity_continuations": {"b2c7": "b1c0", "b2c8": "b1c1", "b2c9": "b1c2", "b2c15": "b1c3", "b2c16": "b1c4"},
        "source_files": source_files,
        "inputs": {"metadata_sha256": file_hash(args.metadata_csv), "eligible_sha256": file_hash(args.d0_root / "eligible_cells.csv"), "targets_sha256": file_hash(args.d0_root / "future_target_support.csv"), "features_sha256": file_hash(args.d0_root / "early_observed_features.csv")},
        "primary_sources": [SEVERSON_URL, ATTIA_URL, AUTHOR_LOADER_URL],
        "boundary": "Static design values do not certify every actual future temperature/current or restore original instrument cycle mapping.",
        "qualifications": details,
    }
    write_rows(args.output / "prior_context.csv", contexts)
    write_rows(args.output / "target_support_by_legacy.csv", support)
    write_json(args.output / "source_qualification.json", summary)
    lines = ["# MATR扩展实验资格与预测时上下文", "", "核验日期为2026-11-15。", "",
             "## 可执行范围", "",
             "MATR的169颗供应电池均具有完整供应K100编号及至少80条合格观测曲线，可以执行早期Q(V)窗口特征预测未来容量斜率的探索性留出。当前目标坐标为供应循环编号，测量为放电电流阶段确认后的容量最大值减去全记录有限容量基线，再除以1.1Ah。原始生成日志的缺口需要作为解释边界保留。", "",
             "## 实际支持", "", "|目标|可用电池|旧576成员|新增成员|", "|---|---|---|---|"]
    lines += [f"|{row['target']}|{row['N_supported']}|{row['N_legacy_576']}|{row['N_new_to_576']}|" for row in support]
    lines += ["", "全部169颗中103颗属于旧576队列，66颗为新增成员。供应容量80%阈值事件为88颗，右删失81颗。不同速率目标有不同观测长度，按目标使用实际支持并保留原ID角色，可以覆盖中期151颗和晚期40颗。", "",
              "## 快充策略来源与SOC修正", "",
              "前三批原始HDF5策略回收JSON精确对应124颗现存文件，逐电池的前两段倍率和首切换SOC均与供应元数据一致。首切换SOC的百分数字段全部124颗通过除以100恢复到比例值，原值保留在source_qualification.json。", "",
              "原作者的方法说明把两段可变快充安排在0至80%SOC，再采用共同1C CC-CV补足到100%。因此当前供应两段对象中的第二段end_soc=1.0与原始可变快充窗口含义存在差异。预测上下文采用原始两段倍率、首切换SOC及0.8快充终点，并记录共同1C补充阶段。[Severson原始论文](https://www.nature.com/articles/s41560-019-0356-8)", "",
              "b4的45颗文件具有9种四段倍率，每种5颗。全部9种向量逐组匹配Attia原文验证实验的9项设计，各段覆盖20%SOC至80%，随后共同1C CC-CV。原始b4逐电池HDF5策略尚未在本地回收，现有供应设计与原文设计的值级对应已记录。[Attia原始论文](https://escholarship.org/content/qt117547r6/qt117547r6.pdf)", "",
              "prior_context.csv前3列为cell_id、family和protocol_group。所有prior_列均为有限数值，包括4个分段倍率及缺失标记、阶段数、首切换SOC、快充SOC终点、共同补充倍率和放电倍率。两段设计的第3/4段采用0与missing=1编码，阶段数保存结构。恢复状态、批次、置信来源及URL采用非prior_列，模型上下文只使用prior_数值列。", "",
              "## 坐标与物理身份", "",
              "供应编号与原始仪器循环尚未逐电池重放认证。当前预处理源码删除原始第0条记录，部分电池跳过坏记录并移位，b1c0至b1c4还拼接了下一批续测。原作者读取代码明确保留这五组物理延续关系，现存MATR目录未发现被归并的b2副本。[原作者数据读取代码](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation/blob/master/Load%20Data.ipynb)", "",
              "按当前可追溯cell_id和别名关系进行电池互斥，目标结果标注supplied_cycle_number。独立的仪器循环、严格ICA/DVA、真实电阻和knee事件认证尚有缺口。现有观测Q(V)、容量变化和供应阈值事件可以保留作相应坐标下的模型任务。", "",
              "## 实验条件与测量边界", "",
              "未来条件继续记录observed_only。已发布的固定策略是预测前可用设计信息，未来真实温度与执行电流没有由本轮元数据审计逐点认证。源研究采用30°C温箱并记录电芯温度，电芯温度随运行变化。[Severson实验方法](https://www.nature.com/articles/s41560-019-0356-8)", "",
              "供应电压上限全部为3.5V，而原作者实验上限为3.6V。该差异保留为元数据问题，当前Q(V)采用2.8至3.3V的共同放电窗口。原有Qdlin千量级极值和内阻零值没有作为本轮窗口模型的特征输入。", "",
              "## 优势与劣势（风险）", "",
              "优势为169颗同一LFP体系、丰富快充策略、共同放电窗口，以及124颗直接策略回收和45颗原文设计匹配。劣势（风险）为部分策略仅有少量重复，晚期目标支持40颗，历史选择暴露、跨批续测、供应编号和插值处理会限制解释与外推。", "",
              "## 置信度评级", "",
              "当前169个输入、策略数值与支持计数的置信度为高。b4值级设计匹配置信度为高，逐电池原始HDF5绑定置信度为中。实际未来执行条件和原始仪器循环认证的置信度为低，模型使用供应坐标与已发布设计上下文。", "",
              "## 拓展思考", "",
              "两段与四段快充具有不同策略结构，模型和误差汇总可同时保留策略组与批次支持数。", "",
              "## 你可能想知道", "",
              "b4恰有9个策略各5颗，具备重复设计支持。它的45颗并不因前三批策略回收文件未覆盖就失去已有的观测曲线与预测上下文。", ""]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "cells": 169, "original_policy_matches": len(policies), "b4_primary_matches": sum(b4_profiles.values()), "target_support": support}, ensure_ascii=False))


if __name__ == "__main__":
    main()
