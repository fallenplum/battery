"""Bind Stanford supplied cells to released formation-design metadata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


FORMATION_COLUMNS = (
    "formation_temperature", "formation_charge_current_1",
    "formation_cutoff_voltage_1", "formation_charge_current_2",
    "formation_verification_repeat", "ocv_time",
)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def fingerprint(path):
    return {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--formation-parameters", type=Path, required=True)
    parser.add_argument("--preprocessor", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository
    output = root / "results" / "extension_stanford"
    eligible = [row for row in read_csv(root / "results/d0/eligible_cells.csv") if row["family"] == "Stanford_2"]
    targets = {row["cell_id"]: row for row in read_csv(root / "results/d0/future_target_support.csv") if row["family"] == "Stanford_2"}
    source_rows = read_csv(args.formation_parameters)
    source = {int(row["seq_num"]): row for row in source_rows}
    if len(source) != len(source_rows):
        raise ValueError("Author metadata contains duplicate sequence IDs")
    metadata = {json.loads(row["metadata"])["cell_id"]: json.loads(row["metadata"]) for row in read_csv(root / "data/cell_metadata.csv") if row["family"] == "Stanford_2"}
    contexts, groups = [], Counter()
    device_hashes = set()
    for cell in sorted(eligible, key=lambda item: item["cell_id"]):
        identifier = cell["cell_id"]
        sequence = int(identifier.rsplit("_", 1)[1])
        author = source[sequence]
        if author["regular_project_name"] != "Nova_Regular":
            raise ValueError("Unexpected author aging project")
        nominal = float(metadata[identifier]["nominal_capacity_in_Ah"])
        if nominal != 0.24:
            raise ValueError("Unexpected Stanford nominal capacity")
        parameters = [float(author[name]) for name in FORMATION_COLUMNS]
        group = "formation_tuple:" + json.dumps(parameters, separators=(",", ":"))
        groups[group] += 1
        identity_hash = hashlib.sha256(author["cell_id"].encode("utf-8")).hexdigest()
        if identity_hash in device_hashes:
            raise ValueError("Two supplied cells map to one author hardware identifier")
        device_hashes.add(identity_hash)
        contexts.append({
            "cell_id": identifier, "family": "Stanford_2", "protocol_group": group,
            "prior_charge_rate_C": float(author["regular_charge_current"]) / nominal,
            "prior_discharge_rate_C": float(author["regular_discharge_current"]) / nominal,
            "prior_nominal_capacity_Ah": nominal,
            "prior_upper_voltage_V": float(author["regular_charge_cutoff_voltage"]),
            "prior_lower_voltage_V": float(author["regular_discharge_cutoff_voltage"]),
            "prior_aging_chamber_setpoint_C": 30.0,
            "prior_formation_temperature_C": parameters[0],
            "prior_formation_charge1_C": parameters[1] / nominal,
            "prior_formation_transition_voltage_V": parameters[2],
            "prior_formation_charge2_C": parameters[3] / nominal,
            "prior_verification_cycles": parameters[4],
            "prior_postformation_rest_hours": parameters[5],
            "prior_missing_aging_protocol": 0,
            "prior_missing_formation_protocol": 0,
            "source_seq_num": sequence,
            "source_device_id_sha256": identity_hash,
            "source_project_name": author["regular_project_name"],
            "metadata_context_available_before_aging": True,
            "protocol_group_definition": "released six formation parameters, not an author protocol-ID field",
            "future_policy_status": "observed_only",
            "future_policy_design_status": "documented_fixed_aging_design",
            "temperature_context_scope": "formation and aging chamber setpoints, supplied cell-temperature channel has no finite values",
            "source_parameter_sha256": fingerprint(args.formation_parameters)["sha256"],
        })
    counts = {}
    support_rows = []
    for name in ("r200_500", "r500_1000"):
        supported = [cell for cell in eligible if targets[cell["cell_id"]][name] != ""]
        legacy = Counter(cell["legacy_576_member"] for cell in supported)
        counts[name] = {"supported": len(supported), "legacy_576": legacy["True"], "new_to_576": legacy["False"],
                        "unavailable_reasons": dict(Counter(targets[cell["cell_id"]][name + "_reason"] for cell in eligible if targets[cell["cell_id"]][name] == ""))}
        support_rows.extend({"cell_id": cell["cell_id"], "target": name, "available": targets[cell["cell_id"]][name] != "",
                             "reason": targets[cell["cell_id"]][name + "_reason"], "legacy_576_member": cell["legacy_576_member"]} for cell in eligible)
    source_meta = {"family": "Stanford_2", "qualification": "USABLE_FOR_SUPPLIED_CYCLE_OBSERVATIONAL_PREDICTION",
                   "provided_cells": len(eligible), "K100_complete": sum(cell["prefix_K100_numbers_complete"] == "True" for cell in eligible),
                   "K100_at_least80_observed_curves": sum(int(cell["curve_valid_records_K100"]) >= 80 for cell in eligible),
                   "curve_reference_current_A_range": [min(float(cell["prefix_current_reference_A"]) for cell in eligible), max(float(cell["prefix_current_reference_A"]) for cell in eligible)],
                   "legacy_576_members": sum(cell["legacy_576_member"] == "True" for cell in eligible),
                   "target_support": counts, "EOL_candidate_supplied_nominal80_events": sum(targets[cell["cell_id"]]["eol_event"] == "True" for cell in eligible),
                   "source_identity": {"local_name_rule": "Stanford_Nova_Regular_<seq_num>", "metadata_matched_cells": len(contexts),
                                       "source_parameter_rows": len(source_rows), "source_unique_device_hashes": len(device_hashes),
                                       "source_table_extra_sequence_numbers": sorted(set(source) - {row["source_seq_num"] for row in contexts}),
                                       "physical_cycle_axis_authenticated": False, "full_BEEP_payload_numeric_replay": "unperformed"},
                   "prior_context": {"available_before_aging": True, "shared_by_all_models": True, "formation_parameters": list(FORMATION_COLUMNS),
                                     "formation_group_count": len(groups), "formation_group_size_distribution": dict(Counter(groups.values())),
                                     "group_definition": "exact released six-parameter tuple, current table distinct count retained",
                                     "aging_protocol": {"charge_C": 1.0, "discharge_C": 0.75, "upper_V": 4.4, "lower_V": 3.0, "aging_chamber_setpoint_C": 30},
                                     "working_temperature_is_not_formation_temperature": True,
                                     "ocv_time_unit": "hours inferred from 0/72/168 released values and paper 0/3/7 days"},
                   "observation_contract": {"coordinate": "supplied_cycle_number after preprocessing renumbering",
                                            "prefix": "supplied number <=100", "phase": "discharge current sign",
                                            "common_voltage_window_V": [3.4, 4.1], "current_reference": "prefix-only, approximately0.18A",
                                            "target_capacity_definition": "phase-confirmed Qd maximum minus record finite baseline, divided by declared0.24Ah",
                                            "supplied_EOL_reference": "0.8 times declared nominal capacity",
                                            "paper_EOL_reference": "0.8 times initial aging capacity",
                                            "actual_EOLEvent_scope": "observed supplied capacity threshold event, physical original-cycle mapping unverified",
                                            "ICA_DVA_scope": "observed geometry/proxy diagnostics, no independent electrode-mechanism certification"},
                   "preprocessing": {"source": fingerprint(args.preprocessor), "interpolated_input_key": "cycles_interpolated",
                                     "formation_or_initial_record_removed": True, "cycle_renumbered": True,
                                     "small_negative_current_rows_removed": "-0.001A < current <0",
                                     "inactive_capacity_channel_zeroed_by_current_sign": True,
                                     "unified_time_normalization_called": True, "source_to_current_generation_link": "unverified"},
                   "evidence": {"formation_parameter_csv": {"url": "https://raw.githubusercontent.com/cx26/formation-analysis/main/data/Formation_2022-Parameter.csv", **fingerprint(args.formation_parameters)},
                                "dataset": "https://data.matr.io/8/", "paper": "https://doi.org/10.1016/j.joule.2024.07.024",
                                "author_code": "https://github.com/cx26/formation-analysis"},
                   "confidence": {"local_counts_and_parameter_join": "high", "aging_design_and_formation_priors": "high",
                                  "strict_original_cycle_alignment": "unverified"},
                   "eligibility_rule": "Use qualified observed curves and actually supported supplied-coordinate outcomes. Source certification limits claims and does not prohibit prediction."}
    write_csv(output / "prior_context.csv", contexts)
    write_csv(output / "per_target_support.csv", support_rows)
    write_json(output / "source_qualification.json", source_meta)
    lines = ["# Stanford_2 扩展实验来源与资格", "", "## 当前支持", "",
             "181颗供应电池均可进入合格观测曲线预测流程。供应K100完整181颗，173颗有100条、8颗有99条合格共同电压窗口曲线。窗口为放电3.4–4.1V，前缀电流参考为约0.18A。",
             "", "中期200–500斜率181颗，旧576成员101、新成员80。晚期500–1000斜率98颗，旧成员53、新成员45。其余83颗晚期端点缺少足够实际观测。两个目标应各自保留有效分母，同一目标内各模型共用精确电池集合。",
             "", "供应容量80%阈值事件181颗。当前定义使用声明0.24Ah，原论文寿命使用初始老化容量的80%。供应阈值与原论文寿命参考分别记录。",
             "", "## 原作者实验设计与新增先验", "",
             "官方数据说明包含BEEP结构化和插值老化数据。原论文说明186颗、62种形成设计，后续老化采用共同条件，充电1C CC-CV、放电0.75C、3–4.4V，老化温箱30°C，每100圈有诊断测试。[官方数据](https://data.matr.io/8/) [原论文](https://doi.org/10.1016/j.joule.2024.07.024)",
             "", "已读取原作者公开形成参数表，183个唯一seq_num与本地181个Nova_Regular编号全部精确匹配，额外两项为197和298。181个硬件标识均唯一，公开连接表仅保存硬件标识SHA。该连接恢复了形成前已指定的温度、两段电流、切换电压、验证次数和形成后静置时间。[原作者参数表](https://github.com/cx26/formation-analysis/blob/main/data/Formation_2022-Parameter.csv)",
             "", "当前181颗按已释放六参数的精确元组有63组，55组三重复、8组两重复。该计数反映当前参数表与本地交集，保留与论文62种设计的版本差异。分组字段明确为参数元组。",
             "", "形成温度为25–55°C，老化温箱为30°C，两者在字段名中分开。静置表值0/72/168依据论文0/3/7天解释为小时。所有新增prior字段在老化前已可获得，所有模型使用相同先验上下文。",
             "", "## 循环坐标、插值与可用任务", "",
             "当前预处理源码读取cycles_interpolated，将周期编号重新映射、删去初始圈并再次重编号，按电流符号将非活动容量置零，还会移除−0.001A至0A的小负电流点并调用时间修复。源码与现存文件的具体生成版本仍缺逐文件绑定。",
             "", "这些证据支持供应循环坐标下的早期Q(V)、跨循环变化与未来容量斜率研究。高频瞬态、真实日历、原始物理循环绝对编号及电极级IC/DVA机理需要更多原始记录。当前扩展实验按供应编号和已插值观测曲线定义测量对象。",
             "", "共同老化条件有原始研究设计支持，完整当前后缀与控制器脚本尚未逐点对照。future_policy_status记录observed_only，固定设计证据另存future_policy_design_status。当前供应温度通道全部缺少有限值，温箱设定进入上下文时应保留其设计来源。",
             "", "## 优势与劣势（风险）", "",
             "优势为181颗共同老化条件、完整早期窗口、181颗中期目标及可追溯的形成参数。劣势（风险）为插值和重编号影响局部导数及坐标解释，形成组可能在按电池划分的训练与测试中重叠，真实微观机理和严格工况族泛化仍需相应验证。",
             "", "## 置信度评级", "",
             "本地计数、设备参数连接及共同老化设计的置信度为高。严格原始循环映射保留未认证状态。上述边界用于限制结论范围，当前数据具备开展观测曲线预测的资格。",
             "", "## 拓展思考", "", "形成参数元组提供可核验的物理分组，可报告组内重复支持与按形成组留出的敏感性结果。",
             "", "## 你可能想知道", "", "BatteryLife README另列的Stanford动态工况项目属于另一数据记录，本地181个Nova_Regular文件与形成研究参数表一一对应。", ""]
    report = root / "report" / "STANFORD_EXTENSION_QUALIFICATION_ZH.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "STANFORD_CONTEXT_READY", "cells": len(contexts), "formation_groups": len(groups), "target_support": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
