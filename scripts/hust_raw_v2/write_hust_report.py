"""Render the source/measurement audit from the completed full-read artifacts."""

import csv
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "hust_raw_idea_audit_v2"


def read_csv(name):
    with (OUT / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    audit = json.loads((OUT / "HUST_RAW_AUDIT.json").read_text(encoding="utf-8"))
    qa = json.loads((OUT / "FULL_AUDIT_VERIFICATION.json").read_text(encoding="utf-8"))
    identity = audit["current_archive_identity"]
    integration = read_csv("capacity_integration_closure_global.csv")
    phases = []
    for name, values in qa["phase_aggregate"].items():
        phases.append(f"| {name} | {values['points']:,} | {values['cycle_median_min_mA']:.2f} 至 {values['cycle_median_max_mA']:.2f} | {values['voltage_min_V']:.4f} 至 {values['voltage_max_V']:.4f} |")
    closure = []
    for phase in qa["phase_aggregate"]:
        rows = {r["time_grid"]: r for r in integration if r["phase"] == phase and r["quadrature"] == "trapezoid"}
        closure.append(f"| {phase} | {float(rows['recorded_dt']['residual_MAE_mAh']):.8f} | {float(rows['recorded_dt']['residual_max_abs_mAh']):.8f} | {float(rows['fixed_5s']['residual_MAE_mAh']):.8f} |")
    report = f"""# HUST 新下载原始包的全量来源与计量审计

## 审计范围与结论

2026-11-15 实际核验当前 `BatteryLife/dataset/hust_raw.zip`，并安全符号解析其全部 **77 电芯、146,122 圈、91,736,659 个采样点**。六通道、全部原始 `rul/dq` 标签、六阶段顺序、原始循环轴、缺失和数值质量均有全量覆盖。此轮数据读取约 {audit['elapsed_s']:.1f} 秒。

**新下载包与既有官方 HUST 数据内容一致。** 原始循环编号和作者公开容量口径闭合。物理测量的独立性、设备校准和实际未来控制可知性仍缺认证。77 电芯均有原始前100圈，具体曲线窗口存在稀疏和端点缺口，需要保留逐圈 mask。

## 当前文件身份与完整性

| 层级 | 实际内容 | 字节数 | SHA256 |
|---|---|---|---|
| 当前外 ZIP | 唯一 member `nsc7hnsg4s-2/our_data.zip` | {identity['outer_bytes']:,} | `{identity['outer_sha256']}` |
| 内 ZIP | 77 个 `our_data/<cell>.pkl` | {identity['inner_bytes']:,} | `{identity['inner_sha256']}` |

当前外 ZIP 已完整 SHA256。其内 member 已流式读取至 EOF 并通过 CRC32 `b887fff0`，内流 SHA256 与旧官方归档及本地缓存一致。随后使用这一逐字节同一的内 ZIP 缓存，重新解压读取全部77 member的字节、CRC32、SHA256并解析。成员集合、大小和 SHA256 与旧官方回执逐项相同。

所有 GLOBAL、REDUCE、NEWOBJ、BUILD 保持为符号数据，未调用 pickle 对象构造器。数字 buffer 经非 object dtype 和长度验证后显式读取。全部10个 GLOBAL 名称保存在审计 JSON。完整来源身份、逐 member 复验及校验结果分别保存在 `CURRENT_ARCHIVE_IDENTITY.json`、`archive_member_verification.csv` 和 `FULL_AUDIT_VERIFICATION.json`。

## 完整字段与元数据

每个文件只有本电芯 root，内部包含 `rul`、`dq`、`data` 三个字典，三者 key 在每颗电芯均完全相同。`data` 的146,122个 DataFrame 全部为以下同一 schema。全部字段和标签缺失或非有限值为 **0**。DataFrame `attrs` 全部为空，`_metadata` 全部为空。

| 字段 | dtype | 单位与实际语义 |
|---|---|---|
| Status | object，字符串字面值 | 六个作者定义阶段 |
| Cycle number | int64 | 原始公开循环号，与字典 key 每点一致 |
| Current (mA) | float64 | 带符号电流，充电正、放电负 |
| Voltage (V) | float64 | 端电压 |
| Capacity (mAh) | float64 | 作者公开的带符号净容量坐标，循环开始归零 |
| Time (s) | int64 | 每圈相对时间，所有循环首点为0 |
| rul | 原始 NumPy 整数 scalar | 所有圈精确等于 `N_recorded - cycle + 1` |
| dq | 原始 NumPy 浮点 scalar | mAh，所有圈等于 D0首点 Capacity 减 D3末点 Capacity，误差小于1e-6mAh |

全部77条原始循环轴从1开始连续，`Cycle number` 全列核验一致。146,122圈六阶段全部齐全且顺序一致。排除循环号和标签后，完整 I/V/Q/Time 内容及阶段长度指纹的精确重复圈为 **0**。未发现额外形成、诊断、休息阶段的专用标识，形成史和诊断角色无法从这批字段认证。

原始文件没有测得温度、日历时间戳、仪器校准和导出头、电极载量/几何、电解液或阻抗。材料和额定参数来自官方记录及论文。官方说明为77颗 LFP/石墨，1.1Ah、3.3V，30℃恒温环境。论文方法列出 A123 APR18650M1A、Neware CT-4008，初测80颗、故障排除3颗。30℃属于实验环境声明，电芯温度轨迹缺失。[官方数据记录](https://data.mendeley.com/datasets/nsc7hnsg4s/2) [原论文方法](https://pubs.rsc.org/en/content/articlehtml/2022/ee/d2ee01676a)

## 六阶段与工况冲突

| 原始阶段 | 全源采样点数 | 每圈电流中位数范围 mA | 电压范围 V |
|---|---|---|---|
{chr(10).join(phases)}

论文写相同充电过程，5C后1C再CV至C/20，末段放电1C至2V，步骤之间休息30s。全量实测末段约−2.2A，对1.1Ah为 **2C**，77颗均与论文和 BatteryLife 写死1C冲突。前三放电阶段的逐电芯实测中位工况与 BatteryLife 三段表在0.05C内一致。原始末点最低电压为1.8291V，最高末点为1.9999V，实际截止采样需要保留原值。[原论文方法](https://pubs.rsc.org/en/content/articlehtml/2022/ee/d2ee01676a)

原始 `HUST_7-7` 第2圈的 `Constant current charge` 包含约5C至约3A的变化。该 phase 0-based rows `[0,162)`，162点中107点偏离5C±0.05C，电流范围2999.06–5498.88mA，中位2999.44mA，容量从0增至858.2553778mAh。这是前100圈内的混合阶段，完整 phase 的共同5C假设有实证例外。其3.1–3.4V窗口位于约5C部分，19个窗口点的电流最大偏差1.49mA，因此该局部窗口的激励条件仍得到支持。

原始表没有显式休息行，phase边界时间差5–32s，Capacity 在全部730,610个阶段边界继承前值。单凭相邻 phase 时间差无法恢复实际休息脚本。论文的工况设计、实际源记录和推理时已知未来控制是不同证据需求。此审计保留实际记录与冲突，未来工况可知性仍未认证。

## 容量计量口径与电流积分

146,122圈 Capacity 均从0开始，充电阶段非降、放电阶段非升，阶段内方向异常为0，阶段边界跳变在1e-6mAh内全部为0。每圈放电量按 `(Q_D0_first - Q_D3_last)/1000` Ah 定义。它与作者 `dq` 标签逐圈闭合，与旧私有146,122行轨迹 **逐值完全相同**。两个冻结斜率目标的154个数值重算误差全部为0，目标支持文件 SHA256 为 `{qa['old_target_support_sha256']}`。

这证明公开源字段和标签的内部一致性。Capacity 的独立设备测量来源、设备固件积分算法、隐藏高频采样及作者预处理没有独立材料认证。此轮将其命名为 **source-provided net Capacity(mAh)**。由于 dq 与这一字段存在直接算术关系，两者不能作为两份独立测量证据。

针对全部同 phase 相邻采样，对实际记录 dt 和固定5s两种时间格，分别比较左端、右端和梯形电流积分，共2772组电芯/阶段/规则统计。下表显示梯形法的逐步容量增量残差，单位mAh。

| 阶段 | recorded-dt MAE | recorded-dt最大绝对残差 | fixed-5s MAE |
|---|---|---|---|
{chr(10).join(closure)}

多数恒流阶段的每圈有效时间中位数 `3600*ΔQ/I_pair_mean` 接近5s，恒流阶段的全源中位数范围约4.9994–5.0006s。六种简单规则均有未闭合步数。数值与库仑积分相容，但这些统计不足以确定唯一生成算法或证明独立计量。phase边界不参与上述积分比较，避免把缺记录休息区间按活动电流填满。

`HUST_1-8` 第168圈 D3，phase rows `[563,668)`，row630到631的 Time 从3160跳至40083s，差36923s。两端电流均−2199.59mA，Capacity 从106.8494667到104.4057333mAh，变化−2.4437333mAh，同时电压从2.9923升至3.1529V。按记录时间梯形填充会得到−22559.8504mAh，残差22557.4067mAh。该间隔体现未认证的中断或记录问题，缺少证据判定其休息身份。另4个重复 Time 步位于 `4-5/341`、`9-5/185`、`9-5/1188`、`9-7/1323` 的 D2末端，容量增量均非0。

这5个时间问题均发生在第100圈之后。时间积分特征须单独标记对应步无支持，source Capacity/Q(V)仍保留。1-8/168的容量目标增加测量可比性待核标记，保留其原值和原目标标签，研究时可做预先声明的敏感性分析。源数据没有足够信息把异常定位为电流、容量或时间中某一通道错误。完整 before/after 点、row bounds、I范围及零容量增量统计保存在 `source_anomaly_details.json`。

## 前100圈局部窗口支持

| 原始窗口 | 7700圈窗口内native点数 | phase覆盖两端 | 逐点电流支持 | 其他质量 |
|---|---|---|---|---|
| D3，2.8–3.1V | 43–85，中位64 | 7698圈 | 所有点距−2200mA最大2.83mA | 电压7次轻微逆向，全部小于10mV，容量方向异常0 |
| CC，3.1–3.4V | 1–23，中位15 | 7700圈 | 所有点距5500mA最大6.70mA | 电压/容量方向异常0 |

D3缺上端的两圈为 `HUST_1-5/1`，phase最高3.0998V，及 `HUST_7-5/1`，phase最高3.0109V。固定3.1V上端严格插值需要对这两圈置mask。CC `HUST_7-5/2` 整个CC阶段仅5点，窗口内仅1点3.2533V。阶段两端能包住窗口，窗口内部形态缺少充分实测点支持。

CC窗口要求每圈至少2/5/8/10个窗口内点时，前100圈全部满足的电芯数分别为76/66/41/30。D3窗口每圈至少30点由77颗全部满足，端点mask仍需要独立保留。插值网格长度和原生测量分辨率分别记录。审计提供逐圈支持和阈值分布，最终feature配置由后续登记流程冻结。

## 退化与未来标签边界

全部77颗都有200–500及500–1000原始循环范围的容量斜率支持。按上述源容量除以1.1Ah，SOH范围为 **0.8000045025252525–1.120186**。精确0.8阈值下实际观察到的事件为0，右删失为77。`rul` 的记录终点剩余计数和论文80%终止声明不足以补造未记录的阈值交叉。

全源相邻圈容量回升51,022步，所有77颗均有回升。因此“持续退化”需要明确时间尺度及趋势定义。源标签无法支持把逐圈容量当作严格单调序列。真实膝点人工/物理标注为0，旧拟合候选仍需保留其统计候选身份。未来标签由后缀容量计算，前缀特征提取只使用原始循环号不大于K的记录。

## 研究使用判断

| 维度 | 优势 | 劣势与风险 | 置信度 |
|---|---|---|---|
| 来源身份和循环轴 | 新旧包逐字节同一，77条轴原始键与列逐点一致 | 未取得仪器直接导出头及形成史 | 高，基于完整字节和全量字段核验 |
| 全局—局部Q(V) | 原始带符号容量、电压、阶段和实测电流可对齐，D3采样密 | 少数端点无覆盖，CC稀点，阶段内轻微电压回摆 | 高，支持范围见逐圈表 |
| 物理容量目标 | 作者dq和净放电口径全量闭合，旧目标完整复现 | 容量通道独立计量未认证，个别中断需额外flag | 中，来源算术闭合，外部计量材料缺失 |
| 已知未来工况 | 各电芯放电阶段实测中位条件稳定 | 末段1C/2C冲突，前缀7-7/2充电混合，控制脚本和未来可知证据缺失 | 低，实际观察无法直接认证推理时可知性 |

## 交付与复核

全部新结果位于 `results/hust_raw_idea_audit_v2/`，包括77行电芯元数据、616行字段质量、462行阶段质量、2772组积分闭合、15400行前缀窗口支持、154行未来斜率重算、逐成员来源校验及异常点见证。146,122行逐圈质量表位于仓库外 `private/raw_sources/HUST_IDEA_V2/all_cycle_quality.csv`，SHA256为 `{audit['private_cycle_quality_sha256']}`。

新脚本位于 `scripts/hust_raw_v2/`。`cell_qualification_flags.csv` 为77颗电芯逐项保留原容量目标、源中断计量复核、Time积分支持、前缀端点及稀点标记。全量复验和逐数值交叉检查为PASS，此状态对应审计覆盖和计算闭合。计量独立性、形成/诊断角色、未来已知控制均在JSON中明确保留未认证状态。此轮采用逐字节相同内缓存重放的实现选择，已完整保存当前外包到内缓存的哈希链。

## 拓展思考

可以把“局部曲线的测量支持”与“容量退化目标的时间尺度”分别写入实验契约，在前100圈的真实采样点及端点范围内定义特征，并让逐圈mask随数据输出。

## 你可能想知道

原始 `HUST_7-5` 确实包含第1和第2圈。旧标准化数据删去这两圈有真实异常背景，新原始处理仍需显式记录第1圈D3上端不足和第2圈CC稀点，恢复原始编号并不自动修复曲线支持。
"""
    (REPO / "report" / "HUST_RAW_IDEA_AUDIT_ZH.md").write_text(report, encoding="utf-8")
    expected_rows = {"archive_member_verification.csv": 77, "raw_cell_metadata.csv": 77, "raw_field_quality.csv": 616,
        "phase_quality.csv": 462, "capacity_integration_closure.csv": 2772, "capacity_integration_closure_global.csv": 36,
        "early_window_support.csv": 15400, "future_target_replay.csv": 154, "degradation_trajectory_quality.csv": 77,
        "declared_vs_observed_protocol.csv": 308, "cell_qualification_flags.csv": 77}
    checks = [{"file": name, "expected_rows": count, "actual_rows": len(read_csv(name)), "pass": len(read_csv(name)) == count} for name, count in expected_rows.items()]
    if not all(row["pass"] for row in checks) or qa["status"] != "PASS":
        raise ValueError("Final audit artifact verification failed")
    current_identity = json.loads((OUT / "CURRENT_ARCHIVE_IDENTITY.json").read_text(encoding="utf-8"))
    if current_identity["outer_sha256"] != identity["outer_sha256"] or current_identity["inner_sha256"] != identity["inner_sha256"]:
        raise ValueError("Final archive identity and full-source replay differ")
    artifacts = []
    for path in sorted([*OUT.glob("*"), *(REPO / "scripts" / "hust_raw_v2").glob("*.py"), REPO / "report" / "HUST_RAW_IDEA_AUDIT_ZH.md"]):
        if path.is_file() and path.name != "AUDIT_ARTIFACTS.json":
            artifacts.append({"path": path.relative_to(REPO).as_posix(), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = {"status": "PASS", "row_count_checks": checks, "artifacts": artifacts,
        "large_raw_arrays_or_models_in_public_artifacts": False, "small_anomaly_point_witnesses_in_public": True,
        "source_capacity_independence_certified": False, "future_control_known_at_prediction_certified": False}
    (OUT / "AUDIT_ARTIFACTS.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("HUST_RAW_IDEA_AUDIT_ZH.md written")


if __name__ == "__main__":
    main()
