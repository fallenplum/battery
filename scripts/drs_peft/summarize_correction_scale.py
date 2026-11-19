"""Recompute the E2 scale audit and selected source-only horizon scores."""

import hashlib
import json
from argparse import Namespace

import numpy as np
import pandas as pd

from scripts.drs_peft.run_correction_scale import (
    CACHE,
    CONFIG,
    METHODS,
    PRIVATE,
    PUBLIC,
    REFERENCES,
    checkpoint,
    csv,
    load_prefixes,
    prepare_episode,
    registered,
    trajectory_paths,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fmt(value):
    return f"{value:.7g}" if np.isfinite(value) else "NA"


def table(lines, headers, rows):
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    lines.append("")


def signed(value):
    return f"{value:+.6g}"


def masked_metrics(a, y, mask):
    eligible = mask.any(axis=1)
    if not eligible.any():
        return (float("nan"), float("nan"), 0, 0)
    mask = mask[eligible]
    count = mask.sum(axis=1)
    err = np.where(mask, a[eligible].astype(np.float64) - y[eligible], 0.0)
    return (
        float(np.mean(np.abs(err).sum(axis=1) / count)),
        float(np.mean(np.square(err).sum(axis=1) / count)),
        int(eligible.sum()),
        int(count.sum()),
    )


def main():
    args = Namespace(
        config=CONFIG,
        public_root=PUBLIC,
        private_root=PRIVATE,
        reference_root=REFERENCES,
        data_root=CACHE,
    )
    cfg, parent, registration, identity = registered(args)
    complete = json.loads((PUBLIC / "E2_COMPLETE.json").read_bytes())
    assert complete["identity"] == identity
    assert (
        complete["trajectories"] == 72
        and complete["snapshots"] == 216
        and complete["matched_pairs"] == 108
    )
    assert (
        complete["optimizer_updates"] == 4320
        and complete["target_label_rows_loaded"] == 0
    )
    for name, expected in complete["result_sha256"].items():
        assert digest(PUBLIC / name) == expected
    anchors = []
    for entry in complete["receipts"]:
        path = PUBLIC / entry["path"]
        assert digest(path) == entry["sha256"]
        receipt = json.loads(path.read_bytes())
        if "anchor" in receipt["files"]:
            item = receipt["files"]["anchor"]
            anchor = PRIVATE / item["path"]
            assert (
                anchor.is_relative_to(PRIVATE / "anchors")
                and digest(anchor) == item["sha256"]
            )
            anchors.append(anchor)
    assert len(anchors) == len(set(anchors)) == 18
    curves = pd.read_csv(PUBLIC / "SCALE_CURVES.csv", float_precision="round_trip")
    pairs = pd.read_csv(
        PUBLIC / "MATCHED_FUNCTION_SCALE.csv", float_precision="round_trip"
    )
    assert len(curves) == 216 and len(pairs) == 108
    assert not curves.duplicated(["fold", "K", "method", "learning_rate", "step"]).any()
    assert not pairs.duplicated(["fold", "K", "learning_rate", "step"]).any()
    assert all(
        np.allclose(
            pairs[f"{m}_radius"] * pairs[f"{m}_contraction"],
            pairs.common_radius,
            atol=1e-12,
            rtol=1e-10,
        )
        for m in ("abba", "lora")
    )
    assert curves.parameters.nunique() == 1
    sample_reindex = (
        (curves.support_objective_mse - curves.support_sampled_mse).abs().max()
    )
    assert sample_reindex <= 2e-7

    old = PUBLIC.parent.parent / "abba_moirai_v1/source"
    audit = []
    for row in curves.itertuples():
        if row.step != 30 or row.learning_rate not in (1e-4, 1e-3):
            continue
        prior = pd.read_csv(old / f"fold{row.fold}" / row.method / "TRIALS.csv")
        trial = prior[
            (prior.K == row.K)
            & (prior.draw == row.draw)
            & (prior.learning_rate == row.learning_rate)
        ]
        assert len(trial) == 1
        audit.append(abs(float(trial.mae.iloc[0]) - row.query_mae))
    assert len(audit) == 36 and max(audit) < 2e-6

    selected_rates = (3e-5, 1e-4)
    data = load_prefixes("MATR", CACHE)
    windows = [
        ("101–200", 101, 200),
        ("201–500", 201, 500),
        ("501–1000", 501, 1000),
        ("1001–5000", 1001, 5000),
    ]
    selected_windows = []
    for fold in cfg["folds"]:
        ref = checkpoint(args, fold, cfg["seed"])
        for k in cfg["shots"]:
            scope = prepare_episode(args, cfg, parent, registration, ref, data, fold, k)
            for rate in selected_rates:
                p = pairs[
                    (pairs.fold == fold)
                    & (pairs.K == k)
                    & (pairs.step == 30)
                    & (pairs.learning_rate == rate)
                ].iloc[0]
                for method, arm in zip(METHODS, ("abba", "lora"), strict=True):
                    receipt_dir, folder = trajectory_paths(args, fold, k, method, rate)
                    predictions_path = folder / "PREDICTIONS.npz"
                    receipt = json.loads((receipt_dir / "COMPLETE.json").read_bytes())
                    if digest(predictions_path) != receipt["files"]["PREDICTIONS.npz"]:
                        raise ValueError(
                            f"Prediction hash mismatch for {predictions_path}"
                        )
                    with np.load(predictions_path, allow_pickle=False) as archive:
                        array = archive["prediction"].astype(np.float64)
                        step = archive["steps"].tolist().index(30)
                    raw = array[step]
                    contracted = array[0] + float(p[f"{arm}_contraction"]) * (
                        raw - array[0]
                    )
                    for label, low, high in windows:
                        index = slice(low - 101, high - 100)
                        fields = {}
                        for group, subset, true_key, mask_key in (
                            ("support_full", slice(0, k), "support_y", "support_mask"),
                            ("support_sampled", slice(0, k), "support_y", "sampled"),
                            ("source_query", slice(k, None), "query_y", "query_mask"),
                        ):
                            y, mask = (
                                scope[true_key][:, index],
                                scope[mask_key][:, index],
                            )
                            for coefficient, predictions in (
                                ("initial", array[0]),
                                ("raw", raw),
                                ("matched", contracted),
                            ):
                                mae, mse, n_cell, n_obs = masked_metrics(
                                    predictions[subset, index], y, mask
                                )
                                fields[f"{group}_{coefficient}_mae"] = mae
                                fields[f"{group}_{coefficient}_mse"] = mse
                            fields[f"{group}_cells"] = n_cell
                            fields[f"{group}_points"] = n_obs
                        selected_windows.append(
                            dict(
                                fold=fold,
                                K=k,
                                method=method,
                                learning_rate=rate,
                                step=30,
                                window=label,
                                start=low,
                                end=high,
                                **fields,
                            )
                        )
    window_frame = pd.DataFrame(selected_windows)
    assert len(window_frame) == 144
    csv(PUBLIC / "SELECTED_WINDOW_DIAGNOSTIC.csv", selected_windows)

    minutes = complete["training_seconds"] / 60
    timing_path = PRIVATE / "EXECUTION_STATUS.json"
    wall_time = "。"
    if timing_path.is_file():
        elapsed = json.loads(timing_path.read_text(encoding="utf-8-sig"))
        elapsed_wall = (
            pd.Timestamp(elapsed["finish"]) - pd.Timestamp(elapsed["start"])
        ).total_seconds() / 60
        wall_time = f"，含评价、存盘和汇总的墙钟 {elapsed_wall:.2f} 分钟。"
    lines = [
        "# E2 学习率、更新预算与函数改变量",
        "",
        "在 MATR 已登记的三个源域工况折上，固定 seed 42、draw 0、K=1/3/5，共比较 **72 条训练轨迹、216 个检查点、108 组配对**。每条轨迹连续训练 60 步，在第 10、30、60 步评价。Bridge、Decoder 和参考权重冻结，ABBA 与 LoRA 各有 "
        + f"**{int(curves.parameters.iloc[0]):,}** 个可训练参数。",
        "",
        "每折按固定哈希顺序取参考模型 fit 集的 8 颗电芯，只用早期前缀。在 101–5000 圈固定网格计算模型输出相对共同冻结参考的 RMS 改变量 R。等幅诊断取两臂 R 的较小值，以 f₀ + ρ(f−f₀) 收缩原始输出。对照在输出层进行，沿用已训练的权重。源 Query 标签只用于打分，**目标未来标签读取数为 0**。",
        "",
        "## 实际学习曲线",
        "",
        "各格对三个折及三个 K 的九个配对取均值。MAE 与 MSE 用 SOH 分数及其平方表示。",
        "",
    ]
    rows = []
    for (rate, step), group in curves.groupby(["learning_rate", "step"]):
        means = group.groupby("method")[
            [
                "query_mae",
                "query_mse",
                "support_sampled_mse",
                "support_full_mse",
                "function_radius",
            ]
        ].mean()
        rows.append(
            (
                f"{rate:.0e}",
                step,
                *[
                    " / ".join(
                        fmt(number)
                        for number in (
                            means.loc[METHODS[0], col],
                            means.loc[METHODS[1], col],
                        )
                    )
                    for col in means.columns
                ],
            )
        )
    table(
        lines,
        [
            "学习率",
            "步数",
            "Query MAE（ABBA / LoRA）",
            "Query MSE（ABBA / LoRA）",
            "采样 Support MSE（ABBA / LoRA）",
            "完整 Support MSE（ABBA / LoRA）",
            "源锚 RMS（ABBA / LoRA）",
        ],
        rows,
    )

    lines.extend(
        [
            "## 第 30 步的等幅对照",
            "",
            "差值为 ABBA MAE 减 LoRA MAE，负值支持 ABBA。每个学习率含九个配对，保留每个折与 K 的方向。",
            "",
        ]
    )
    summary = []
    detail = []
    for rate in selected_rates:
        subset = pairs[(pairs.learning_rate == rate) & (pairs.step == 30)].sort_values(
            ["fold", "K"]
        )
        ratio = subset.abba_radius / subset.lora_radius.replace(0, np.nan)
        a, b = (
            subset.raw_abba_minus_lora_query_mae,
            subset.matched_abba_minus_lora_query_mae,
        )
        summary.append(
            (
                f"{rate:.0e}",
                fmt(ratio.median()),
                fmt(subset.common_radius.mean()),
                signed(a.mean()),
                f"{(a < 0).sum()}/{len(a)}",
                signed(b.mean()),
                f"{(b < 0).sum()}/{len(b)}",
            )
        )
    for fold in cfg["folds"]:
        for k in cfg["shots"]:
            pair = pairs[
                (pairs.fold == fold) & (pairs.K == k) & (pairs.step == 30)
            ].set_index("learning_rate")
            detail.append(
                (
                    fold,
                    k,
                    *[
                        signed(float(pair.loc[rate, col]))
                        for rate in selected_rates
                        for col in (
                            "raw_abba_minus_lora_query_mae",
                            "matched_abba_minus_lora_query_mae",
                        )
                    ],
                )
            )
    table(
        lines,
        [
            "学习率",
            "ABBA/LoRA R 中位比",
            "平均共同 R",
            "原始平均差",
            "原始 ABBA 胜",
            "等幅平均差",
            "等幅 ABBA 胜",
        ],
        summary,
    )
    table(
        lines, ["折", "K", "3e-5 原始", "3e-5 等幅", "1e-4 原始", "1e-4 等幅"], detail
    )

    lines.extend(
        [
            "## 分时距与计算量",
            "",
            "仅列第 30 步两档学习率。各窗口按实际观测筛选电芯，再取电芯宏平均。分母为跨折/K 的电芯实例数，同一电芯可重复计入不同 K。",
            "",
        ]
    )
    spans = []
    for (rate, low, label), group in window_frame.groupby(
        ["learning_rate", "start", "window"]
    ):
        keys = [
            "source_query_raw_mae",
            "source_query_matched_mae",
            "support_full_raw_mse",
            "support_full_matched_mse",
        ]
        arms = group.groupby("method")[keys].mean()
        q_cells = int(group[group.method == METHODS[0]].source_query_cells.sum())
        s_cells = int(group[group.method == METHODS[0]].support_full_cells.sum())
        spans.append(
            (
                f"{rate:.0e}",
                label,
                f"{q_cells} / {s_cells}",
                *[" / ".join(fmt(arms.loc[m, col]) for m in METHODS) for col in keys],
            )
        )
    table(
        lines,
        [
            "学习率",
            "窗口（圈）",
            "Query / Support 电芯实例",
            "Query 原始 MAE（ABBA / LoRA）",
            "Query 等幅 MAE（ABBA / LoRA）",
            "完整 Support 原始 MSE（ABBA / LoRA）",
            "完整 Support 等幅 MSE（ABBA / LoRA）",
        ],
        spans,
    )

    costs = []
    for method in METHODS:
        group = curves[curves.method == method]
        chosen = group[(group.step == 30) & (group.learning_rate == 1e-4)]
        costs.append(
            (
                method,
                fmt(chosen.update_frobenius.mean()),
                fmt(chosen.layer_mean_stable_rank.mean()),
                fmt(chosen.layer_mean_energy_rank_99.mean()),
                int(group.layer_max_numerical_rank.max()),
                fmt(group.fit_peak_allocated_bytes.max() / 1024**2),
                fmt(group.fit_peak_reserved_bytes.max() / 1024**2),
                fmt(group[group.step == 60].training_seconds.sum()),
            )
        )
    table(
        lines,
        [
            "方法",
            "1e-4/30 平均更新 Frobenius 范数",
            "1e-4/30 平均层稳定秩",
            "1e-4/30 平均层 99% 能量秩",
            "全矩阵最高数值秩",
            "峰值 PyTorch 分配 MiB",
            "峰值 PyTorch 保留 MiB",
            "累计训练秒",
        ],
        costs,
    )
    lines.append(
        f"本机 RTX 4060 实际执行 **4,320** 步，累计优化 {minutes:.2f} 分钟{wall_time}"
        f"36 个与上一轮重叠的源选参格，Query MAE 重放差最大为 {max(audit):.3g}。"
        f"全轨迹重索引与训练实际采样 MSE 最大差 {sample_reindex:.3g}。"
    )
    lines.append("")
    lines.extend(["## 可支持的判断", ""])
    for method in METHODS:
        g = curves[(curves.method == method) & (curves.step == 30)]
        old_scale = g[g.learning_rate == 1e-4].set_index(["fold", "K"])
        mild = g[g.learning_rate == 3e-5].set_index(["fold", "K"])
        q = mild.query_mae - old_scale.query_mae
        s = mild.support_sampled_mse - old_scale.support_sampled_mse
        lines.append(
            f"- {method} 从 1e-4 减到 3e-5，30 步 Query MAE 平均变化 {signed(q.mean())}，{(q < 0).sum()}/9 改善。"
            f"采样 Support MSE 平均变化 {signed(s.mean())}，{(s < 0).sum()}/9 改善。"
            f"源锚 R 均值从 {fmt(old_scale.function_radius.mean())} 变为 {fmt(mild.function_radius.mean())}。"
        )
    for rate in selected_rates:
        subset = pairs[(pairs.learning_rate == rate) & (pairs.step == 30)]
        raw = subset.raw_abba_minus_lora_query_mae.mean()
        matched = subset.matched_abba_minus_lora_query_mae.mean()
        direction = "符号翻转" if np.sign(raw) != np.sign(matched) else "符号保持"
        lines.append(
            f"- {rate:.0e} 等幅前后 ABBA 减 LoRA 的均值为 {signed(raw)} 和 {signed(matched)}，{direction}。"
            f"配对的等幅后绝对差均值是 {fmt(subset.matched_abba_minus_lora_query_mae.abs().mean())}。"
        )
    lines.append(
        "- 两档均出现时距反例。原始 ABBA 在 101–500 圈 MAE 较低，在 501–5000 圈的两个分窗 MAE 较高。"
        "1001–5000 圈仅有 12 个折/K 电芯实例，等幅后的差异应连同分母读取。"
    )
    n0 = int((curves.function_radius == 0).sum())
    n_noise = int((~curves.radius_above_initial_noise).sum())
    lines.append(
        f"- 零步数值噪声 RMS 最大 {fmt(curves.initial_function_noise.max())}。{n0} 个训练后检查点 R 等于零，"
        f"{n_noise} 个检查点 R 未超出初始数值噪声或 1e-12 计算阈值。{int(pairs.resolvable_common_radius.sum())}/108 组有非零可解析共同半径。"
    )
    lines.append(
        "- 优势是固定电芯、训练数据和参考后可重算有符号的幅度对照。局限是每折只选择一个支持抽样，采用既往已用的源折。"
        "等幅方向比较发生在两臂较小的函数半径，较大半径上的方向优劣仍需相应数据。"
    )
    lines.append("")
    lines.append(
        "完整指标和配对见 `SCALE_CURVES.csv`、`MATCHED_FUNCTION_SCALE.csv`。分窗逐组 MAE、MSE 和分母见 `SELECTED_WINDOW_DIAGNOSTIC.csv`。"
    )
    (PUBLIC / "E2_REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(PUBLIC / "E2_REPORT_ZH.md")


if __name__ == "__main__":
    main()
