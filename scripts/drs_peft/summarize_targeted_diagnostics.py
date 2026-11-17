"""Build the combined source-only diagnostic report from completed artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/drs_peft/diagnostics_v1"
REPORT = ROOT / "report/drs_peft/diagnostics_v1/TARGETED_DIAGNOSTICS_FINAL_ZH.md"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def table(frame):
    lines = ["| " + " | ".join(frame.columns) + " |"]
    lines.append("| " + " | ".join("---" for _ in frame.columns) + " |")
    for row in frame.itertuples(index=False, name=None):
        values = [
            f"{value:.6f}" if isinstance(value, float) else str(value) for value in row
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main():
    paths = {
        "registration": ROOT / "configs/drs_peft/diagnostics_v1.json",
        "D1_summary": RESULTS / "D1_specificity/D1_SUMMARY.json",
        "D1_pairs": RESULTS / "D1_specificity/D1_AUGMENTED_PAIR_SUMMARY.csv",
        "D1_capture": RESULTS / "D1_specificity/D1_TRUE_GRADIENT_CAPTURE_SUMMARY.csv",
        "D1_cross_cell": RESULTS / "D1_specificity/D1_CROSS_CELL_GEOMETRY_SUMMARY.csv",
        "D2_geometry": RESULTS / "D2_structure/D2_GEOMETRY_SUMMARY.csv",
        "D2_errors": RESULTS / "D2_structure/D2_ERROR_SUMMARY.csv",
        "D2_replay": RESULTS / "D2_structure/D2_INDEPENDENT_REPLAY.json",
        "D3_gate": RESULTS / "D3/D3_GATE.json",
        "D3_metrics": RESULTS / "D3/D3_METRICS.csv",
        "D3_cells": RESULTS / "D3/D3_CELL_METRICS.csv",
        "D3_comparisons": RESULTS / "D3/D3_COMPARISONS.json",
        "D3_audit": RESULTS / "D3/D3_INDEPENDENT_AUDIT.json",
        "QA_log": RESULTS / "validation/QA_FULL.log",
        "parent_preservation": RESULTS / "validation/PARENT_PRESERVATION.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(
            "The three diagnostics are incomplete: " + ", ".join(missing)
        )
    registration = read_json(paths["registration"])
    d1 = read_json(paths["D1_summary"])
    d3 = read_json(paths["D3_gate"])
    preservation = read_json(paths["parent_preservation"])
    audit = read_json(paths["D3_audit"])
    replay = read_json(paths["D2_replay"])
    if (
        d1["new_autograd_evaluations"] != 1044
        or d1["optimizer_updates"] != 0
        or d1["final_target_Query_future_labels_accessed"] is not False
        or d3["completed_units"] != 18
        or d3["new_optimizer_updates"] != 19800
        or d3["final_target_query_labels_used"] != 0
        or preservation["status"] != "PASSED"
        or audit["status"] != "PASSED"
        or audit["final_target_query_labels_used"] != 0
        or replay["status"] != "PASSED"
        or replay["final_target_query_labels_used"] != 0
    ):
        raise RuntimeError(
            "A diagnostic completion or parent preservation guard failed"
        )
    pairs = pd.read_csv(paths["D1_pairs"])
    directions = pairs.loc[
        pairs.left.eq("true_common") & pairs.right.eq("shuffled_common"),
        [
            "seed",
            "mean_residual_sign_agreement",
            "gradient_cosine",
            "absolute_gradient_cosine",
        ],
    ].copy()
    directions.columns = ["种子", "残差同号率", "带符号梯度余弦", "绝对梯度余弦"]
    capture = pd.read_csv(paths["D1_capture"])
    capture = capture[
        capture.role.eq("meta_validation")
        & capture.basis_kind.eq("gradient_cov")
        & capture["rank"].eq(4)
        & capture.basis_supervision.isin(["true_full", "flat_full", "linear_full"])
    ]
    capture_table = capture.pivot(
        index="basis_supervision", columns="seed", values="capture"
    ).reset_index()
    capture_table.columns = ["构基监督", "42", "123", "2026"]
    cross_cell = pd.read_csv(paths["D1_cross_cell"])
    cross_cell = cross_cell.loc[
        cross_cell.supervision.eq("true_full"),
        [
            "seed",
            "within_protocol_absolute_cosine_mean",
            "cross_protocol_absolute_cosine_mean",
            "participation_effective_rank",
            "top1_normalized_vector_covariance_energy",
        ],
    ].copy()
    cross_cell.columns = [
        "种子",
        "同工况绝对余弦",
        "跨工况绝对余弦",
        "参与比有效秩",
        "首方向能量",
    ]
    geometry = pd.read_csv(paths["D2_geometry"])
    geometry = geometry[geometry.origin.eq("relative_W0")]
    errors = pd.read_csv(paths["D2_errors"])
    errors = (
        errors[errors.window.isin(["all", "1001_plus"])]
        .pivot(index=["method", "rank"], columns="window", values="mae")
        .reset_index()
    )
    structure = geometry.merge(errors, on=["method", "rank"], validate="one_to_one")
    structure = structure[
        [
            "method",
            "rank",
            "mean_update_to_weight_ratio",
            "mean_stable_rank",
            "all",
            "1001_plus",
        ]
    ]
    structure.columns = [
        "方法",
        "rank",
        "更新与原权重范数比",
        "stable rank",
        "完整未来MAE",
        "1001+ MAE",
    ]
    units = pd.read_csv(paths["D3_metrics"])
    cells = pd.read_csv(paths["D3_cells"])
    if len(units) != 18 or units.groupby("mode").size().ne(3).any():
        raise RuntimeError("The D3 report requires all six arms and all three seeds")
    means = units.groupby("mode", as_index=False).agg(
        source_meta_mae=("mae", "mean"),
        trainable_parameters=("trainable_parameters", "first"),
    )
    tail = cells[cells.window.eq("1001_plus") & cells.n_obs.gt(0)]
    tail_means = tail.groupby("mode", as_index=False).agg(
        tail_mae=("mae", "mean"), tail_unique_cells=("cell_uid", "nunique")
    )
    means = means.merge(tail_means, on="mode", validate="one_to_one")
    order = registration["D3"]["arms"]
    means = means.set_index("mode").loc[order].reset_index()
    means.columns = ["实验臂", "完整未来MAE", "可训练参数", "1001+ MAE", "远期独立电芯"]
    comparisons = read_json(paths["D3_comparisons"])
    full_contrasts = {
        item["reference"]: item for item in comparisons if item["window"] == "all"
    }
    tail_contrasts = {
        item["reference"]: item for item in comparisons if item["window"] == "1001_plus"
    }
    static_seed_gains = "、".join(
        f"{100 * full_contrasts['static']['per_seed_relative_improvement'][str(seed)]:.2f}%"
        for seed in registration["training_seeds"]
    )
    contrasts = []
    for item in comparisons:
        if item["window"] != "all":
            continue
        contrasts.append(
            {
                "dual对照": item["reference"],
                "相对改善百分比": 100 * item["relative_mae_improvement"],
                "三种子均改善": item["all_seeds_improved"],
                "电芯区间下界": item["crossed_seed_cell_interval"][0],
                "电芯区间上界": item["crossed_seed_cell_interval"][1],
                "工况区间下界": item["crossed_seed_protocol_interval"][0],
                "工况区间上界": item["crossed_seed_protocol_interval"][1],
                "登记判据满足": d3["contrast_support"][item["reference"]],
            }
        )
    state = d3["state_conditioning_supported"]
    complementary = d3["complementary_views_supported"]
    if state and complementary:
        decision = "本轮支持源开发集上的状态条件化核心和双视角互补性，下一步需要独立留出验证与同预算的基旋转对照。"
    elif state:
        decision = "本轮支持源开发集上的状态条件化核心，双视角互补性尚未满足登记判据。后续需要优先核查有效状态信息和更强的单视角对照。"
    else:
        decision = (
            f"双视角完整未来MAE较静态核心改善{100 * full_contrasts['static']['relative_mae_improvement']:.2f}%，"
            "严格支持判据未通过。当前应优先复核状态条件化系数与绝对视角的源开发信号，"
            "旋转U/V的必要性仍待直接验证。退化特异固定基的解释不足，"
            "固定核心已经产生可测量的权重、激活及预测变化。"
        )
    qa_lines = (
        paths["QA_log"].read_text(encoding="utf-8", errors="replace").splitlines()
    )
    qa_summary = next((line for line in reversed(qa_lines) if " passed" in line), "")
    if not qa_summary or "failed" in qa_summary.lower():
        raise RuntimeError("The full regression test run has not passed")
    report = f"""# 三项针对性诊断综合结果

## 结论与证据范围

**{decision}**

采用同一 MATR 源训练85颗、47工况组与源验证30颗、15工况组，种子42、123、2026。实际执行D1监督替换、D2已训练适配器结构测量和D3六臂对照。新训练为15个单元，每单元120个epoch、1320次优化更新，共19800次更新。旧static三单元复用原P2结果，工程两epoch另计22次更新。最终目标Query未来标签使用量为0。

## 退化特异性

{table(directions)}

标签置换改变带符号梯度，同一电芯的梯度仍几乎沿同一条直线。共同掩码消除了旧置换同时搬运donor观测掩码的混杂。以下为源方向对真实源验证梯度的rank4二阶矩基捕获。

{table(capture_table)}

容量flat和前缀OLS外推都保持较高留出捕获。标签会改变更新符号与幅度，当前共享几何缺乏退化结局特异性的证明。容量外推包含完整放电容量的前100圈摘要，信息预算高于原局部电压窗Q(V)，只用作诊断伪监督。无截断外推在5000圈可产生负SOH，不能赋予这些标签实测物理含义。

观察到同一电芯的近直线梯度后，追加了只读缓存的跨电芯探索性检查。逐层将85颗电芯的完整矩阵梯度展开并归一化，再求电芯Gram谱。以下为12层均值，该对象与前述左右因子协方差不同。

{table(cross_cell)}

每层同工况无序对75组，跨工况3495组，这些对共享电芯，仅报告描述性统计。真实监督也存在强共享几何，当前组合中的输入、骨干、桥、头与损失尚未被逐项隔离。剩余方向是否携带关键退化信息，需要独立对照验证。

## 适配结构

{table(structure)}

Gradient-Core实际更新与激活影响更大，rank4单位权重更新坐标的局部输出敏感性也较高。较大更新与更高尾段误差共同出现，当前比较没有隔离其因果关系。1001+误差仅有5颗独立电芯。Jacobian选取六颗源验证电芯，1500与3000查询均缺真实观测，属于局部外推敏感性测量。

两方法沿用各自上一轮选定的核心学习率，Gradient-Core为0.001，LoRA-XS为0.0001。更新幅度差同时包含优化尺度和共同适配状态的影响，单位权重坐标换算只消除了XS奇异值造成的坐标尺度差异。

## 双视角最小实验

{table(means)}

{table(pd.DataFrame(contrasts))}

状态条件化登记支持结果为 `{state}`，双视角互补性登记支持结果为 `{complementary}`。对照同时要求至少1%完整未来改善、三个种子方向为正、物理电芯与工况区间下界为正，以及1001+平均误差没有增加。所有比较均保留，源验证选模和既有学习率条件化使区间属于探索性描述。

区间以对照MAE减去dual的MAE表示，正值代表dual改善。工况区间先在组内求电芯均值，再对工况等权重抽样，其权重与电芯等权的主指标不同。

本轮完整未来均值改善与严格门槛的结果需同时保留。双视角相对静态核心三个种子的改善分别为{static_seed_gains}，相对常数门平均改善{100 * full_contrasts["constant"]["relative_mae_improvement"]:.2f}%。各工况配对区间均跨零，1001+相对常数门与置换门的平均误差分别增加{-100 * tail_contrasts["constant"]["relative_mae_improvement"]:.2f}%与{-100 * tail_contrasts["shuffled_dual"]["relative_mae_improvement"]:.2f}%。绝对视角单独获得{full_contrasts["absolute"]["reference_mae"]:.6f}的MAE，双视角较其再改善{100 * full_contrasts["absolute"]["relative_mae_improvement"]:.2f}%，该比较的两类区间均跨零。201–500与501–1000两段误差均值也降低，远期支持仅{tail_contrasts["constant"]["cells"]}颗、{tail_contrasts["constant"]["protocol_groups"]}工况组。

绝对VIQ与相对响应各12维，其中4个V相关统计在本队列固定，实际变化维度各8维。新增五臂都有1248个门网络参数和192个核心参数，共同Bridge/Head为63169个参数，合计64609。static合计63361。constant臂输入常数，其有效函数自由度低于输入条件化臂。标准化仅拟合源训练85颗。

所有门满足固定U/V下的对角核心调制。更新始终位于同一线性空间，当前实验能够评价系数条件化，基方向旋转的必要性需要额外直接对照。

## 优势、风险与置信度

优势是三种子、同桥同头、真实监督替换、同掩码对照、实际权重谱、尺度校正Jacobian以及门参数匹配和置换控制可以相互约束解释。风险包括源验证反复使用、有限远期支持、电压网格常量、压缩I/Q统计的信息损失与桥头共同适配。

执行完整性置信度高，依据缓存与checkpoint指纹、端点重算、D2独立CPU重放、D3落盘预测重放及独立审计。具体物理因果归因置信度低。当前结果适用于已登记的骨干、数据、特征与预算。

## 产物与核验

- D1新增1044次梯度前后向，另255次单电芯预测前向，优化器更新0。
- D2主GPU测量360次标量反向，独立CPU重放另360次，共720次，优化器更新0。
- D3正式新增19800次优化更新，工程两epoch22次更新单独保存。
- 完整回归测试结果为 `{qa_summary.strip()}`。
- 父版本 `{preservation["parent_commit"]}` 的{preservation["checked_files"]}个文件、{preservation["checked_bytes"]}字节均保持原字节。

完整运行命令与现场资源记录保存在[运行与核验记录](../../../research/drs_peft/diagnostics_v1/RUN_COMMANDS_AND_VALIDATION_ZH.md)。

单项详细数值、对照定义与局限分别保存在[D1监督诊断](D1_SPECIFICITY_ZH.md)、[跨电芯补充检查](D1_CROSS_CELL_GEOMETRY_ZH.md)、[D2结构诊断](D2_STRUCTURE_REPORT_ZH.md)与[D3双视角实验](D3_DUAL_VIEW_REPORT_ZH.md)。

方法来源保留LoRA-XS的权重SVD核心结构与LoRA-SB的梯度更新近似先例，本轮围绕电池监督与条件化开展机制诊断。[LoRA-XS](https://arxiv.org/abs/2405.17604)、[LoRA-SB](https://arxiv.org/abs/2411.19557)

## 拓展思考

后续比较应同时控制方向空间、系数自由度、有效更新幅度与预测头。动态基方法需要直接测量每颗电芯新增方向的独立价值，并保留当前固定基与强单视角对照。

## 你可能想知道

Stanford、HUST及最终MATR目标Query的未来轨迹均未参与本轮诊断、训练与评分，原目标Episode及P2阴性结论继续保留。
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")
    result = {
        "status": "COMPLETE_SOURCE_ONLY_DIAGNOSTICS",
        "final_target_query_labels_used": 0,
        "new_formal_optimizer_updates": 19800,
        "engineering_optimizer_updates": 22,
        "D1_new_gradient_forward_backward_evaluations": 1044,
        "D2_primary_and_independent_replay_scalar_backward_evaluations": 720,
        "degradation_specificity_established": False,
        "state_conditioning_supported": state,
        "complementary_views_supported": complementary,
        "rotating_dynamic_subspace_established": False,
        "D3_independent_audit_status": audit.get("status"),
        "qa_summary": qa_summary.strip(),
        "parent_preservation": preservation,
        "inputs": [
            {
                "name": name,
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": file_hash(path),
            }
            for name, path in paths.items()
        ],
        "report": {
            "path": REPORT.relative_to(ROOT).as_posix(),
            "sha256": file_hash(REPORT),
        },
        "formatter_sha256": file_hash(__file__),
    }
    (RESULTS / "TARGETED_DIAGNOSTICS_COMPLETE.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": result["status"], "report": str(REPORT)}))


if __name__ == "__main__":
    main()
