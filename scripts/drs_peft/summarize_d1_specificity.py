"""Derive unsigned-direction summaries and the D1 report from sealed diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .data import ROOT, file_hash, write_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def markdown_table(frame):
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def summarize(root, report):
    inputs = []
    all_gradients, all_coefficients = [], []
    for seed in (42, 123, 2026):
        folder = root / f"seed{seed}"
        complete = read_json(folder / "D1_COMPLETE.json")
        for record in complete["files"]:
            path = folder / record["path"]
            if file_hash(path) != record["sha256"]:
                raise RuntimeError("D1 formatter encountered changed completed outputs")
            inputs.append({"path": str(path.resolve()), "sha256": record["sha256"]})
        inputs.append(
            {
                "path": str((folder / "D1_COMPLETE.json").resolve()),
                "sha256": file_hash(folder / "D1_COMPLETE.json"),
            }
        )
        all_gradients.append(pd.read_csv(folder / "D1_GRADIENT_PAIR_METRICS.csv"))
        all_coefficients.append(pd.read_csv(folder / "D1_OUTPUT_COEFFICIENTS.csv"))
    gradients = pd.concat(all_gradients, ignore_index=True)
    gradients["absolute_gradient_cosine"] = gradients.gradient_cosine.abs()
    gradients["squared_gradient_cosine"] = gradients.gradient_cosine.pow(2)
    gradients["reversed_nearly_collinear"] = gradients.gradient_cosine.lt(-0.95)
    gradients["nearly_collinear"] = gradients.gradient_cosine.abs().gt(0.95)
    columns = [
        "gradient_cosine",
        "absolute_gradient_cosine",
        "squared_gradient_cosine",
        "right_to_left_norm_ratio",
        "reversed_nearly_collinear",
        "nearly_collinear",
    ]
    cell = gradients.groupby(
        ["seed", "left", "right", "cell_uid", "protocol_uid"], as_index=False
    )[columns].mean()
    summary = cell.groupby(["seed", "left", "right"], as_index=False)[columns].mean()
    coefficients = pd.concat(all_coefficients, ignore_index=True)
    coefficient_summary = coefficients.groupby(
        ["seed", "left", "right"], as_index=False
    ).agg(
        mean_output_coefficient_cosine=("output_coefficient_cosine", "mean"),
        mean_residual_sign_agreement=("common_residual_sign_agreement", "mean"),
        identical_output_coefficient_cells=("output_coefficients_identical", "sum"),
        cells=("cell_uid", "nunique"),
    )
    summary = summary.merge(
        coefficient_summary, on=["seed", "left", "right"], validate="one_to_one"
    )
    summary.to_csv(root / "D1_AUGMENTED_PAIR_SUMMARY.csv", index=False)
    directions = summary.loc[
        summary.left.eq("true_common") & summary.right.eq("shuffled_common"),
        [
            "seed",
            "mean_residual_sign_agreement",
            "mean_output_coefficient_cosine",
            "gradient_cosine",
            "absolute_gradient_cosine",
        ],
    ].copy()
    directions.columns = [
        "Seed",
        "输出残差同号率",
        "输出系数cos",
        "带符号梯度cos",
        "梯度绝对cos",
    ]
    for column in directions.columns[1:]:
        directions[column] = directions[column].map(lambda value: f"{value:.6f}")
    capture = pd.read_csv(root / "D1_TRUE_GRADIENT_CAPTURE_SUMMARY.csv")
    meta = capture.loc[
        capture.role.eq("meta_validation")
        & capture.basis_kind.eq("gradient_cov")
        & capture["rank"].eq(4)
    ]
    capture_table = meta.pivot(
        index="basis_supervision", columns="seed", values="capture"
    ).reset_index()
    capture_table.columns = ["构基监督", "Seed42", "Seed123", "Seed2026"]
    for column in capture_table.columns[1:]:
        capture_table[column] = capture_table[column].map(lambda value: f"{value:.6f}")
    overlaps = pd.read_csv(root / "D1_SUBSPACE_OVERLAP_SUMMARY.csv")
    matched = overlaps.loc[
        overlaps.left.eq("true_common")
        & overlaps.right.eq("shuffled_common")
        & overlaps.basis_kind.eq("gradient_cov")
        & overlaps["rank"].eq(4),
        ["seed", "left_overlap", "right_overlap", "tensor_product_overlap"],
    ].copy()
    matched.columns = ["Seed", "左基重合", "右基重合", "张量积子空间重合"]
    for column in matched.columns[1:]:
        matched[column] = matched[column].map(lambda value: f"{value:.6f}")
    slope = pd.read_csv(root / "seed42/D1_PREFIX_CAPACITY_SLOPES.csv")
    linear_endpoint = slope.anchor_soh100 + slope.slope_SOH_per_cycle * 4900
    summary_state = read_json(root / "D1_SUMMARY.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# D1 退化特异性诊断

## 运行与信息预算

三seed 42、123、2026均已完成。每seed使用相同的P0 B0源开发checkpoint，源训练85颗/47工况，源meta验证30颗/15工况，QV共12个384×384矩阵。新收集四种监督，每种85次单cell前后向及首尾2次独立重放，合计{summary_state["new_autograd_evaluations"]}次autograd，另有255次单cell预测前向，优化器更新为0。最终目标Query未来标签保持封存。

真实全轨迹、原始置换全轨迹与真实meta梯度从P1只读复用。所有缓存数组逐SHA核验。原始置换同时搬运donor观测mask，本轮新增的true_common和shuffled_common严格共用receiver与donor的实际观测交集，消除两者间的mask差异。

容量flat监督对每颗电芯取SOH100常数。容量linear监督先用该电芯认证的前1–100真实循环完整放电容量拟合OLS斜率，再以SOH100为锚点外推。读取raw_record_audit.parquet时先以Arrow predicate限定源85颗及1–100圈，共8487条记录、8486条有效容量，每颗99或100个有效点。完整放电容量仅用于构造伪监督，模型输入保留原VIQ和anchor。

斜率范围为{slope.slope_SOH_per_cycle.min():.9f}至{slope.slope_SOH_per_cycle.max():.9f} SOH/cycle。固定5000圈处linear伪标签范围为{linear_endpoint.min():.6f}至{linear_endpoint.max():.6f}。保留无截断外推可检查监督方向的敏感性，这些数值属于诊断伪标签。

## 同mask置换改变了什么

{markdown_table(directions)}

所有数值先按12层取每颗电芯均值，再对85颗等权。带符号cos保留上升与下降方向，绝对cos描述同一直线程度。绝对cos与cov子空间重合均对整体符号翻转不敏感，须与带符号统计联合解释。

## 同mask梯度协方差子空间

{markdown_table(matched)}

子空间由源47个工况等权、工况内电芯等权估计。左、右基各取rank4，以主角度余弦平方均值定义重合率。张量积重合为左、右重合率的乘积。rank56与mean-SVD控制保留在原始CSV及SUMMARY表中。

## 对真实源meta梯度的捕获

{markdown_table(capture_table)}

表中全部评价对象均为30颗源meta电芯的真实未来轨迹损失梯度。每种构基监督只改变源方向，meta监督始终使用实际观测SOH。表中capture为二维投影能量比例，工况宏均值另存在同一CSV。

## 梯度方向的解释

固定单颗输入与模型参数时，MAE梯度满足 g=Jᵀd，其中J为固定网格预测对QV参数的Jacobian，d=mask×sign(预测−监督)/实际观测数。同mask且d完全相同的两种监督得到相同梯度。监督差异还可能被J映射成同一条方向上的幅度与符号变化。

本轮直接改变监督并测量g和d。标签对带符号更新的作用、共享梯度几何和子空间捕获是不同层面的观测。高绝对cos或高捕获本身无法建立退化结局特异性。输出系数与梯度cos的逐层相关、完全相同输出系数的分层对照均保存在D1_SIGN_DIRECTION_ASSOCIATION.csv与D1_SIGN_CONDITIONED_GRADIENT_SUMMARY.csv。

## 优势与局限

- 优势在于相同模型与电芯输入下直接替换监督，三seed、12层完整保留带符号方向、无符号子空间与留出真实梯度捕获。共同mask对照使标签差异的解释更明确。
- 风险包括源开发队列已用于多轮选型、容量linear无约束远期外推、逐cell时间序列误差相关，以及row/column因子子空间对整体符号的固有不敏感。静态checkpoint上的方向诊断不能等同于实际训练增益，也无法独立识别物理因果机制。
- 执行完整性置信度高，依据完整缓存SHA、独立端点重放、模型前后指纹与冻结角色。退化机制归因置信度低，当前证据仅覆盖该骨干、桥与预测头的源开发状态。

## 复现

运行scripts.drs_peft.diagnostic_specificity，指定configs/drs_peft/diagnostics_v1.json、官方backbone目录及device。三seed采集后自动汇总。随后运行scripts.drs_peft.summarize_d1_specificity产生本文及绝对cos摘要。每seed D1_SOURCE_SNAPSHOT.json与D1_REGISTRATION.json保存源码、注册文件、源数据及checkpoint SHA。

## 拓展思考

本轮与适配结构诊断共同约束后续方法解释。若调制改善预测，只能先建立状态条件化在当前共享子空间中的作用，再检验旋转或扩展子空间是否带来独立增益。

## 你可能想知道

本轮容量控制没有训练新预测模型。所有新增前后向都用于梯度诊断，最终目标域预测性能没有进入本报告。
"""
    report.write_text(text, encoding="utf-8")
    outputs = [root / "D1_AUGMENTED_PAIR_SUMMARY.csv", report]
    write_json(
        root / "D1_REPORT_MANIFEST.json",
        {
            "status": "COMPLETE_DERIVED_SOURCE_ONLY_DIAGNOSTIC",
            "inputs": inputs,
            "formatter_sha256": file_hash(__file__),
            "outputs": [
                {"path": str(path.resolve()), "sha256": file_hash(path)}
                for path in outputs
            ],
            "final_target_Query_future_labels_accessed": False,
            "posthoc_direction_summary": "absolute cosine and thresholds |cos|>0.95 are descriptive additions derived from frozen signed cosine arrays",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "results/drs_peft/diagnostics_v1/D1_specificity",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "report/drs_peft/diagnostics_v1/D1_SPECIFICITY_ZH.md",
    )
    args = parser.parse_args()
    summarize(args.root, args.report)


if __name__ == "__main__":
    main()
