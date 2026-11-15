"""Compile the registered MATR and Stanford extension results without refitting."""

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FAMILIES = (("MATR", "experiment_b_matr"), ("Stanford_2", "experiment_b_stanford"))


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"] + ["| " + " | ".join(str(value) for value in row) + " |" for row in rows])


def main():
    supports, headline, all_metrics, intervals, decisions = [], [], [], [], []
    names = {"mean": "训练均值", "context_only": "已知工况上下文", "context_capacity": "上下文＋早期容量", "context_qv_shape": "上下文＋Q(V)", "combined": "上下文＋容量＋Q(V)"}
    for family, directory in FAMILIES:
        location = ROOT / "results" / directory
        contract = json.loads((location / "CONTRACT.json").read_text(encoding="utf-8"))
        receipt = json.loads((location / "EVALUATED.json").read_text(encoding="utf-8"))
        if receipt["completed_targets"] != ["r200_500", "r500_1000"]:
            raise RuntimeError("Both registered extension targets must be completed")
        metrics = read(location / "metrics.csv")
        indexed = {(row["target"], row["model"]): row for row in metrics}
        macro = {row["target"]: row for row in read(location / "protocol_macro_paired_improvement.csv")}
        comparisons = {row["target"]: row for row in read(location / "paired_mae_improvement.csv")}
        negative = Counter(row["target"] for row in read(location / "negative_cases.csv"))
        for target in ("r200_500", "r500_1000"):
            horizon = "200–500" if target == "r200_500" else "500–1000"
            support = contract["target_support"][target]["role_counts"]
            supports.append([family, horizon, sum(support.values()), support["train"], support["val"], support["test"]])
            baseline, combined = indexed[target, "context_capacity"], indexed[target, "combined"]
            base, candidate = float(baseline["MAE"]), float(combined["MAE"])
            cell, group = comparisons[target], macro[target]
            headline.append([family, horizon, support["test"], f"{base*1e4:.4f}", f"{candidate*1e4:.4f}",
                             f"{(base-candidate)/base*100:.2f}%", f"{float(baseline['R2']):.4f}", f"{float(combined['R2']):.4f}"])
            intervals.append([family, horizon,
                              f"{float(cell['MAE_improvement'])*1e4:.5f} [{float(cell['CI95_low'])*1e4:.5f}, {float(cell['CI95_high'])*1e4:.5f}]",
                              f"{float(group['protocol_macro_MAE_improvement'])*1e4:.5f} [{float(group['CI95_low'])*1e4:.5f}, {float(group['CI95_high'])*1e4:.5f}]",
                              group["protocol_groups"], f"{negative[target]}/{support['test']}"])
            decisions.append({"family": family, "target": target, "test_cells": support["test"],
                              "relative_MAE_reduction": (base-candidate)/base,
                              "cell_CI_excludes_zero_in_positive_direction": float(cell["CI95_low"]) > 0,
                              "protocol_macro_CI_excludes_zero_in_positive_direction": float(group["CI95_low"]) > 0,
                              "scope": "one registered historical exploratory supplied-coordinate holdout"})
        for row in metrics:
            all_metrics.append([family, row["target"], names[row["model"]], row["n"],
                                f"{float(row['MAE'])*1e4:.5f}", f"{float(row['RMSE'])*1e4:.5f}",
                                f"{float(row['R2']):.5f}" if row["R2"] else "无定义",
                                f"{float(row['Spearman']):.5f}" if row["Spearman"] else "无定义"])
    lines = ["# MATR与Stanford早期曲线预测实验扩展", "", "执行日期为2026-11-15。",
             "", "## 覆盖修正与当前结论", "",
             "上一轮D0包含MATR和Stanford，模型实验B仅执行HUST，覆盖不完整。本轮对MATR全部169颗和Stanford全部181颗完成资格连接、预注册、选模、留出评分与独立复核。",
             "", "预测任务采用可追溯的供应循环编号和观测曲线。原始物理循环标定、插值及阶段语义作为证据字段保留。原始生成记录的缺口用于限定解释范围，现有合格观测支持可以进入相应坐标下的预测实验。",
             "", "MATR的200–500目标显示明确的当前留出增量证据。MATR晚期目标及Stanford两个目标的联合模型改善区间跨0。Stanford最大失败是已核验的分布外线性外推，真实早期特征和冻结预测均可重放。",
             "", "## 数据支持与固定拆分", "",
             table(["来源", "供应循环目标", "总支持", "训练", "验证", "测试"], supports),
             "", "先在完整合格base队列固定电池ID角色，再对每个目标使用实际已观察的端点。同一目标内五个模型使用完全相同电池。MATR base为169颗，Stanford base为181颗。",
             "", "主目标为200–500供应循环的平均容量比退化速率，次目标为500–1000。容量定义为放电电流阶段确认的最大容量减去有限全记录基线，再除以供应标称容量。端点窗口沿用±5供应循环、至少5个有效观测。晚期缺少观察的电池保留支持不足原因。",
             "", "## 工况与早期输入合同", "",
             "所有特征均来自供应编号不超过100的前缀。MATR采用2.8–3.3V共同放电窗口，Stanford采用3.4–4.1V。容量组三项和Q(V)组七项沿用已登记提取逻辑，曲线每圈以下界电压处电荷为零，64点网格。",
             "", "五组模型均可获得同一份预测前上下文，包括训练均值、上下文、上下文＋容量、上下文＋Q(V)、上下文＋容量＋Q(V)。组标识仅用于分层和重采样。",
             "", "MATR有14列有限先验。前三批124颗的原始快充policy已回收，SOC百分数转为比例，两段快充截止80%后共同1C补充。b4的45颗对应九种四段策略各五颗，依据供应值与原作者设计匹配。零占位与missing标志共同记录结构性缺段。[MATR来源核验](MATR_EXTENSION_QUALIFICATION_ZH.md)、[原始两段策略研究](https://www.nature.com/articles/s41560-019-0356-8)",
             "", "Stanford有14列有限先验。原作者183行形成参数表与181颗供应文件全部精确匹配，包含六项形成设计参数和共同老化条件。当前交集形成参数元组63组，55组三重复、8组两重复。老化条件为1C充电、0.75C放电、3–4.4V、30°C温箱，电芯实测温度缺少有限值。[Stanford来源核验](STANFORD_EXTENSION_QUALIFICATION_ZH.md)、[官方数据](https://data.matr.io/8/)、[原作者参数表](https://github.com/cx26/formation-analysis/blob/main/data/Formation_2022-Parameter.csv)",
             "", "新增的工况与形成先验扩展了VIQ-only信息预算，所有对照使用相同上下文。供应曲线与坐标的资格均已登记，严格IC/DVA、电阻和knee事件的认证状态延续前轮。",
             "", "## 模型选择与封存", "",
             "旧576成员按固定SHA排序以75%/25%分训练和验证，其他成员为留出测试。StandardScaler仅使用各目标的训练电池，Ridge的alpha为0.01、0.1、1、10、100，按验证MAE选择。全部已支持目标的系数先冻结，再计算测试预测。",
             "", "本轮两个来源各10组测试预测，共1010条逐电池/目标/模型预测。选模覆盖均值控制与Ridge网格，历史HUST结果及其实现保持原提交内容。",
             "", "## 主要留出结果", "",
             "MAE单位为10⁻⁴容量比例/供应循环。相对MAE下降的正值表示误差减少，负值表示误差增加。", "",
             table(["来源", "目标", "测试N", "上下文＋容量MAE", "联合Q(V) MAE", "相对MAE下降", "容量对照R²", "联合R²"], headline),
             "", "MATR中期MAE由1.71256e-4降为8.68552e-5，降低49.28%，R²由0.2122升至0.7236。其电池配对改善95%区间为[6.32254e-5,1.06060e-4]，工况组宏均值改善区间为[4.67487e-5,1.05998e-4]，两者下界均为正。该结论对应当前数据、窗口、模型和拆分。",
             "", "MATR晚期支持只有40颗，测试20颗。联合增量区间跨0，完整五组结果保留QV单独组与联合组的差异。",
             "", "Stanford两个目标中，联合模型的MAE均高于容量对照，区间均跨0。数据的观测支持、排序关联与当前模型的误差表现分别记录。",
             "", "## 电池与工况组不确定性", "",
             "下表改善单位同为10⁻⁴容量比例/供应循环。电池统计量采用测试电池等权均值，工况统计量先计算每个工况的均值再对组等权，二者分别以2000次重采样形成95%区间。", "",
             table(["来源", "目标", "电池改善及95%区间", "工况宏改善及95%区间", "测试工况组", "误差恶化电池"], intervals),
             "", "![联合特征的配对改善和重采样区间](figures/MULTIFAMILY_PAIRED_IMPROVEMENT.png)",
             "", "## 工况覆盖与Stanford失败诊断", "",
             "以下为封存后的描述性诊断。MATR中期测试57颗分为28个设计参数组，其中22颗所在组在相应训练中出现，35颗属于未在训练出现的组。两层配对改善分别为6.38128e-5和9.73421e-5，完整支持数和反例保存在extension_analysis。",
             "", "Stanford中期测试80颗的28个形成参数组、晚期测试45颗的21个组，均与训练/验证的完整六参数元组无交集。原拆分依据legacy身份注册，当前分组差异是实际数据属性。",
             "", "Stanford_Nova_Regular_270的窗口电荷比训练均值低约9个标准差。晚期联合预测为−0.0088779，真实观察目标为0.0002381。该例占中期/晚期完整测试平方误差约72.04%/86.23%。",
             "", "对270、320、124、280、279五颗重新读取现有源文件的前100圈，登记特征逐值重现，窗口全部合格、相关通道有限，容量比也在登记有效范围内。125条联合预测使用冻结参数重放的最大差为8.67e-19。结果支持当前线性模型在有限训练覆盖下发生极端外推，诊断保留全部原评分和样本。详见STANFORD_PREDICTION_FAILURE_DIAGNOSTIC_ZH.md。",
             "", "## 五组完整结果", "",
             table(["来源", "目标", "模型", "测试N", "MAE", "RMSE", "R²", "Spearman"], all_metrics),
             "", "## 独立核验与历史记录", "",
             "MULTIFAMILY_EXTENSION_QA.json包含106项独立检查，覆盖每目标ID与mask、输入和核心代码SHA、训练标准化、验证alpha、Ridge正规方程、1010条预测、全部指标、两类bootstrap与负例，以及源policy/形成参数逐值转换。核验调用模型拟合和测试评价入口次数为0。",
             "", "旧HUST模型、输入、D0和SDU产物对应历史提交e8a7295，保持原科学记录。README新增本轮索引，新交付清单单独封存本轮仓库快照，历史DELIVERY_MANIFEST继续锚定旧提交。",
             "", "## 优势与劣势（风险）", "",
             "优势为两个来源全部早期合格电池纳入base，按各目标保留151/40与181/98的有效支持，元数据与曲线派生特征的增量关系可直接检验，并同时保留电池和工况组层面统计。",
             "", "劣势（风险）为历史数据开发暴露、单次固定拆分、晚期支持较少、观测长度条件化，以及供应编号和插值处理的解释限制。工况先验均在预测前可用，当前结果的误差统计仍以已观察测试队列为条件。",
             "", "## 置信度评级", "",
             "当前可用性、元数据连接、预测和区间重算的置信度为高。MATR中期增量作用在当前留出条件下的证据评级为中，依据是电池与工况组区间方向一致，但缺少独立新数据和重复拆分。MATR晚期与Stanford联合增量证据评级为低，依据是配对区间跨0及已记录外推失败。",
             "", "## 拓展思考", "",
             "Stanford的共同老化条件与形成历史分别影响任务分层。形成参数组的覆盖结构能够解释一部分训练和测试分布差异，已纳入后续研究记录。",
             "", "## 你可能想知道", "",
             "MATR的45颗b4具有九种四段快充设计，每种五颗。它们的观测输入和已发表设计已全部接入本轮上下文与预测任务。", ""]
    (ROOT / "report" / "MULTIFAMILY_EXTENSION_REPORT_ZH.md").write_text("\n".join(lines), encoding="utf-8")
    summary = {"status": "COMPLETE_EXPLORATORY_MULTIFAMILY_EXTENSION", "base_cells": {"MATR": 169, "Stanford_2": 181},
               "targets": ["r200_500", "r500_1000"], "decisions": decisions,
               "coordinate": "supplied_cycle_number", "historical_HUST_snapshot": "e8a7295d5199775006f7a0b750f58cb96488d5df",
               "prior_context": "All compared models share source-confirmed pre-aging design parameters"}
    (ROOT / "results" / "MULTIFAMILY_STAGE_DECISION.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
