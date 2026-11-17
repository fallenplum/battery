"""Independent CPU audit of completed source-only dual-view experiments."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.drs_peft.data import file_hash, write_json
from scripts.drs_peft.diagnostic_dualview import ALL_MODES, SEEDS, prepare_views
from scripts.drs_peft.train import ROOT, load_source, read_json

WINDOWS = (
    ("all", 101, 5000),
    ("101_200", 101, 200),
    ("201_500", 201, 500),
    ("501_1000", 501, 1000),
    ("1001_plus", 1001, 5000),
)


def _independent_prediction_metrics(folder, data, validation_rows):
    with np.load(
        folder / "SOURCE_VALIDATION_PREDICTIONS.npz", allow_pickle=False
    ) as archive:
        ids = archive["ids"].astype(str).tolist()
        query = archive["query"]
        predictions = archive["prediction"].astype(np.float64)
        targets = archive["target"].astype(np.float64)
        observed = archive["observed"]
    expected_ids = [data.ids[row] for row in validation_rows]
    if ids != expected_ids or not np.array_equal(query, data.query):
        raise RuntimeError("D3 source validation identities or fixed query grid differ")
    if not np.array_equal(
        targets, data.targets[validation_rows].astype(np.float64)
    ) or not np.array_equal(observed, data.observed[validation_rows]):
        raise RuntimeError("D3 source validation truth or observation mask changed")
    rows = []
    residual = predictions - targets
    for position, uid in enumerate(ids):
        for window, lower, upper in WINDOWS:
            mask = observed[position] & (query >= lower) & (query <= upper)
            values = residual[position, mask]
            rows.append(
                {
                    "cell_uid": uid,
                    "protocol_uid": str(data.protocol_ids[validation_rows[position]]),
                    "window": window,
                    "n_obs": len(values),
                    "mae": float(np.abs(values).mean()) if len(values) else np.nan,
                    "rmse": float(np.sqrt(np.square(values).mean()))
                    if len(values)
                    else np.nan,
                    "bias": float(values.mean()) if len(values) else np.nan,
                }
            )
    independent = pd.DataFrame(rows)
    original = pd.read_csv(folder / "SOURCE_VALIDATION_CELL_METRICS.csv")
    keys = ["cell_uid", "protocol_uid", "window"]
    paired = independent.merge(
        original, on=keys, suffixes=("_audit", "_original"), validate="one_to_one"
    )
    if len(paired) != len(independent) or len(paired) != len(original):
        raise RuntimeError("D3 source error rows lack exact pairing")
    if not np.array_equal(paired.n_obs_audit, paired.n_obs_original):
        raise RuntimeError("D3 source observation counts differ")
    max_difference = 0.0
    for metric in ("mae", "rmse", "bias"):
        before = paired[metric + "_original"].to_numpy()
        after = paired[metric + "_audit"].to_numpy()
        if not np.allclose(before, after, rtol=0, atol=2e-15, equal_nan=True):
            raise RuntimeError("D3 NPZ metrics do not reproduce the stored CSV")
        max_difference = max(max_difference, float(np.nanmax(np.abs(before - after))))
    receipt = read_json(folder / "SOURCE_DEVELOPMENT_COMPLETE.json")
    all_rows = independent[independent.window == "all"]
    mae = float(all_rows.mae.mean())
    if abs(mae - receipt["validation"]["mae"]) > 2e-15:
        raise RuntimeError("D3 independently recomputed MAE differs from the receipt")
    return independent, max_difference


def _gate_parameters_and_values(
    folder, mode, seed, state, layer_names, arrays, data, episodes
):
    source_rows = data.rows(episodes["source_cell_roles"]["source_train"])
    meta_rows = data.rows(episodes["source_cell_roles"]["source_meta_validation"])
    selected = np.concatenate((source_rows, meta_rows))
    donors = read_json(folder / "D3_SHUFFLE_DONORS.json")
    if mode == "shuffled_dual":
        for role in ("source_train", "source_meta_validation"):
            mapping = donors[role]
            expected = set(episodes["source_cell_roles"][role])
            if (
                set(mapping) != expected
                or set(mapping.values()) != expected
                or any(uid == donor for uid, donor in mapping.items())
            ):
                raise RuntimeError(
                    "D3 shuffled features violate the role-local derangement"
                )
        donor_rows = data.rows(
            [
                donors[
                    "source_train"
                    if row in set(source_rows)
                    else "source_meta_validation"
                ][data.ids[row]]
                for row in selected
            ]
        )
    else:
        if donors:
            raise RuntimeError("An unshuffled arm contains donor mappings")
        donor_rows = selected
    absolute = arrays["absolute"][donor_rows].astype(np.float64)
    relative = arrays["relative"][donor_rows].astype(np.float64)
    if mode in ("constant", "relative"):
        absolute = np.ones_like(absolute)
    if mode in ("constant", "absolute"):
        relative = np.ones_like(relative)
    records, summaries = [], []
    selected_nonzero, final_nonzero = 0, 0
    final = torch.load(
        folder / "TRAIN_STATE.pt", map_location="cpu", weights_only=False
    )["model"]
    for index, name in enumerate(layer_names):
        gates = {}
        for view, values in (("absolute", absolute), ("relative", relative)):
            prefix = f"bridge.backbone.module.core_gates.{view}.{index}."
            weight, bias = state[prefix + "weight"], state[prefix + "bias"]
            selected_nonzero += int(bool((weight != 0).any())) + int(
                bool((bias != 0).any())
            )
            final_nonzero += int(bool((final[prefix + "weight"] != 0).any())) + int(
                bool((final[prefix + "bias"] != 0).any())
            )
            logits = values @ weight.numpy().astype(np.float64).T + bias.numpy().astype(
                np.float64
            )
            gates[view] = 1.0 + 0.5 * np.tanh(logits)
            if (
                not np.isfinite(gates[view]).all()
                or (gates[view] < 0.5).any()
                or (gates[view] > 1.5).any()
            ):
                raise RuntimeError("D3 gate output escaped its registered range")
            for role, segment in (
                ("source", slice(0, len(source_rows))),
                ("meta", slice(len(source_rows), None)),
            ):
                chosen = gates[view][segment]
                summaries.append(
                    {
                        "mode": mode,
                        "seed": seed,
                        "layer": name,
                        "view": view,
                        "role": role,
                        "physical_cells": len(chosen),
                        "mean": float(chosen.mean()),
                        "std_across_cell_and_rank": float(chosen.std()),
                        "mean_rankwise_std_across_cells": float(chosen.std(0).mean()),
                        "min": float(chosen.min()),
                        "max": float(chosen.max()),
                        "fraction_within_0_005_of_bound": float(
                            ((chosen <= 0.505) | (chosen >= 1.495)).mean()
                        ),
                        "max_abs_weight": float(weight.abs().max()),
                        "max_abs_bias": float(bias.abs().max()),
                    }
                )
        core = (
            state["bridge.backbone.module." + name + ".core"].numpy().astype(np.float64)
        )
        core_norm = float(np.linalg.norm(core))
        effective = (
            gates["relative"][:, :, None] * core[None] * gates["absolute"][:, None, :]
        )
        norms = np.linalg.norm(effective, axis=(1, 2))
        if mode == "constant" and np.ptp(norms) > 1e-12:
            raise RuntimeError(
                "The constant arm acquired cell-dependent update amplitudes"
            )
        for position, row in enumerate(selected):
            record = {
                "mode": mode,
                "seed": seed,
                "layer": name,
                "cell_uid": data.ids[row],
                "role": "source" if position < len(source_rows) else "meta",
                "feature_donor_uid": data.ids[donor_rows[position]],
                "unmodulated_core_frobenius": core_norm,
                "effective_update_frobenius": float(norms[position]),
                "effective_to_unmodulated_ratio": float(norms[position] / core_norm)
                if core_norm
                else None,
            }
            record.update(
                {
                    f"{view}_rank{rank + 1}": float(gates[view][position, rank])
                    for view in ("absolute", "relative")
                    for rank in range(4)
                }
            )
            records.append(record)
    return (
        records,
        summaries,
        {
            "selected_nonzero_gate_tensors": selected_nonzero,
            "final_nonzero_gate_tensors": final_nonzero,
            "expected_gate_tensors": 48,
        },
    )


def _recompute_comparisons(table, comparisons):
    maximum_difference = 0.0
    for comparison in comparisons:
        window = comparison["window"]
        subset = table[(table.window == window) & table.n_obs.gt(0)]
        candidate = subset[subset["mode"] == comparison["candidate"]]
        reference = subset[subset["mode"] == comparison["reference"]]
        paired = candidate.merge(
            reference,
            on=["seed", "cell_uid", "protocol_uid"],
            suffixes=("_candidate", "_reference"),
            validate="one_to_one",
        )
        paired["improvement"] = paired.mae_reference - paired.mae_candidate
        relative_gain = float(paired.improvement.mean() / paired.mae_reference.mean())
        if abs(relative_gain - comparison["relative_mae_improvement"]) > 2e-13:
            raise RuntimeError(
                "D3 contrast point estimate differs from raw predictions"
            )
        key = comparison["candidate"] + "/" + comparison["reference"] + "/" + window
        base_seed = (42 + int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)) % (
            2**32 - 1
        )
        for offset, unit in enumerate(("cell_uid", "protocol_uid")):
            values = (
                paired.groupby(["seed", unit])
                .improvement.mean()
                .unstack()
                .sort_index()
                .to_numpy()
            )
            rng = np.random.default_rng(base_seed + offset)
            draws = []
            for _ in range(2000):
                seeds = rng.integers(0, len(values), len(values))
                units = rng.integers(0, values.shape[1], values.shape[1])
                draws.append(float(values[np.ix_(seeds, units)].mean()))
            recomputed = np.quantile(draws, [0.025, 0.975])
            field = (
                "crossed_seed_cell_interval"
                if offset == 0
                else "crossed_seed_protocol_interval"
            )
            difference = float(np.max(np.abs(recomputed - comparison[field])))
            maximum_difference = max(maximum_difference, difference)
            if difference > 2e-15:
                raise RuntimeError(
                    "D3 paired bootstrap interval differs from its raw-prediction recomputation"
                )
    return {
        "comparisons_recomputed": len(comparisons),
        "paired_intervals_recomputed": len(comparisons) * 2,
        "maximum_interval_absolute_difference": maximum_difference,
    }


def _write_report(path, table, gate_summaries, audit_result, gate_result, comparisons):
    names = {
        "static": "静态核心",
        "constant": "常数双门",
        "absolute": "绝对视角",
        "relative": "相对视角",
        "dual": "双视角",
        "shuffled_dual": "置换双视角",
    }
    full_comparisons = {
        item["reference"]: item for item in comparisons if item["window"] == "all"
    }
    tail_scope = next(item for item in comparisons if item["window"] == "1001_plus")
    lines = [
        "# 双视角调制的源开发最小实验",
        "",
        "## 执行与判定",
        "",
        "本轮完成 15 项新增真实训练，另复用 3 项 P2 静态核心结果。新增训练共 19800 次优化更新，训练 seed 为 42、123 和 2026。源训练使用 85 颗独立电芯，源开发使用 30 颗电芯和 15 个工况组。最终目标 Query 的未来标签使用数为 0。",
        "",
        f"**状态条件化判定为 {gate_result['status']}，双视角互补性判据为 {gate_result['complementary_views_supported']}。**",
        "",
        "各单元保持相同 rank4 U/V、初始 P0 B0、桥、头和源监督，120 epoch 后取源开发总体 cell-mean MAE 首次严格最低的权重。核心与门学习率为 0.001，桥/头为 0.0003。新增学习率搜索次数为 0。",
        "",
        "## 源开发误差",
        "",
        "| 方法 | 总可训练参数 | 全未来 MAE | 全未来电芯数 | 1001+ MAE | 1001+ 电芯数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode in ALL_MODES:
        full = table[
            (table["mode"] == mode) & (table.window == "all") & table.n_obs.gt(0)
        ]
        tail = table[
            (table["mode"] == mode) & (table.window == "1001_plus") & table.n_obs.gt(0)
        ]
        parameters = {
            unit["trainable_parameters"]
            for unit in audit_result["units_checked"]
            if unit["mode"] == mode
        }
        if len(parameters) != 1:
            raise RuntimeError("D3 parameter counts vary across seeds of the same arm")
        lines.append(
            f"| {names[mode]} | {next(iter(parameters))} | {full.mae.mean():.8f} | {full.cell_uid.nunique()} | {tail.mae.mean():.8f} | {tail.cell_uid.nunique()} |"
        )
    lines += [
        "",
        "每颗电芯先在实际观测位置计算误差，再跨电芯等权平均。1001+ 分母随实际记录覆盖保留，跨 seed 记录仍来自同一组物理电芯。",
        f"1001+ 远期误差仅覆盖 {tail_scope['cells']} 颗物理电芯和 {tail_scope['protocol_groups']} 个工况组。该小分母限定了远期总体解释。",
        "",
        "## 完整未来改善与严格门槛",
        "",
        f"双视角完整未来 MAE 为 {full_comparisons['static']['candidate_mae']:.8f}，相对静态核心改善 {full_comparisons['static']['relative_mae_improvement'] * 100:.3f}%，相对常数双门改善 {full_comparisons['constant']['relative_mae_improvement'] * 100:.3f}%，相对置换双视角改善 {full_comparisons['shuffled_dual']['relative_mae_improvement'] * 100:.3f}%。全部五个完整未来对照均在三种 seed 上同向。电芯状态条件化具有一致的源开发点估计收益。",
        "",
        f"绝对单视角 MAE 为 {full_comparisons['absolute']['reference_mae']:.8f}，双视角在此基础上额外改善 {full_comparisons['absolute']['relative_mae_improvement'] * 100:.3f}%。相对视角单独使用的改善较弱，双视角互补性仍受到配对区间与远期表现限制。",
        "",
        "预注册的联合门槛同时要求工况稳定性与远期不恶化。当前 NOT_SUPPORTED 保留对这些联合条件的原始判定。下面逐项列出未满足的条件，完整未来的收益继续按实际数值保留。",
        "",
        "| 对照 | 全未来改善至少 1% | 完整未来三 seed 同向 | 电芯区间下界高于零 | 工况区间下界高于零 | 1001+ 无恶化 | 未满足条件 |",
        "|---|---|---|---|---|---|---|",
    ]
    for reference, comparison in full_comparisons.items():
        tail = next(
            item
            for item in comparisons
            if item["window"] == "1001_plus" and item["reference"] == reference
        )
        checks = {
            "全未来改善幅度": comparison["relative_mae_improvement"] >= 0.01,
            "完整未来seed方向": comparison["all_seeds_improved"],
            "电芯区间": comparison["crossed_seed_cell_interval"][0] > 0,
            "工况组区间": comparison["crossed_seed_protocol_interval"][0] > 0,
            "1001+无恶化": tail["relative_mae_improvement"] >= 0,
        }
        flags = " | ".join("通过" if passed else "未满足" for passed in checks.values())
        failures = "、".join(name for name, passed in checks.items() if not passed)
        lines.append(f"| {names[reference]} | {flags} | {failures or '全部满足'} |")
    lines += [
        "",
        "## 双视角的全部注册对照",
        "",
        "正收益表示双视角的 MAE 较低。区间对应参考方法 MAE 减双视角 MAE，按 seed 与电芯、seed 与工况组分别交叉重抽样 2000 次。工况组区间使用组宏均值。",
        "",
        "| 对照 | 全未来相对改善 | seed42 / 123 / 2026 改善 | 电芯配对 95% 区间 | 工况组配对 95% 区间 | 1001+ 相对改善 |",
        "|---|---:|---|---|---|---:|",
    ]
    for comparison in comparisons:
        if comparison["window"] != "all":
            continue
        tail = next(
            item
            for item in comparisons
            if item["window"] == "1001_plus"
            and item["reference"] == comparison["reference"]
        )
        seed_gains = " / ".join(
            f"{comparison['per_seed_relative_improvement'][str(seed)] * 100:+.3f}%"
            for seed in SEEDS
        )
        cell_interval = ", ".join(
            f"{value:+.6f}" for value in comparison["crossed_seed_cell_interval"]
        )
        group_interval = ", ".join(
            f"{value:+.6f}" for value in comparison["crossed_seed_protocol_interval"]
        )
        lines.append(
            f"| {names[comparison['reference']]} | {comparison['relative_mae_improvement'] * 100:+.3f}% | {seed_gains} | [{cell_interval}] | [{group_interval}] | {tail['relative_mae_improvement'] * 100:+.3f}% |"
        )
    lines += [
        "",
        "状态条件化要求双视角相对常数和置换臂分别达到至少 1% 的全未来改善、三种 seed 同向、两类区间下界高于零，以及 1001+ 平均误差无恶化。互补性额外要求两个单视角对照满足相同条件。",
        "",
        "## 实际门与更新幅度",
        "",
        "实际门由选权中的线性权重与前缀特征在 CPU 上独立计算。近饱和比例指门值距离 0.5 或 1.5 的边界不超过 0.005。电芯间变化按每个 rank 坐标的跨电芯标准差衡量。",
        "",
        "| 方法 | 门视角 | 源开发门值最小 / 最大 | 平均跨电芯标准差 | 近饱和比例 |",
        "|---|---|---|---:|---:|",
    ]
    gate_table = pd.DataFrame(gate_summaries)
    for mode in ALL_MODES[1:]:
        for view in ("absolute", "relative"):
            selected = gate_table[
                (gate_table["mode"] == mode)
                & (gate_table.view == view)
                & (gate_table.role == "meta")
            ]
            lines.append(
                f"| {names[mode]} | {'绝对' if view == 'absolute' else '相对'} | {selected['min'].min():.5f} / {selected['max'].max():.5f} | {selected.mean_rankwise_std_across_cells.mean():.6f} | {selected.fraction_within_0_005_of_bound.mean() * 100:.3f}% |"
            )
    lines += [
        "",
        "逐电芯、逐层门值及调制后的有效更新 Frobenius 范数保存在 D3_GATE_VALUES_AUDIT.csv。恒定臂的有效更新幅度在电芯间相同。",
        "",
        "## 独立核验",
        "",
        f"独立 CPU 审计重新计算 {audit_result['raw_npz_cell_window_metrics_recomputed']} 条电芯窗口误差，完成 {audit_result['frozen_uv_tensor_comparisons']} 次 U/V 张量比较、{audit_result['selected_gate_tensor_checks']} 个门状态检查，以及 10 个对照和 20 个配对区间重算。源开发原始目标与观测 mask 均与冻结缓存一致。各单元的初始化恒等、落盘重构与预测重放凭据完整。",
        "",
        "## 优势与风险",
        "",
        "优势是同一低维核心、U/V、桥、头、数据、初始化、优化次数与选权规则共同固定，常数和置换对照分别审查额外可训练门结构及电芯状态对应关系。",
        "",
        "风险包括源开发集与旧学习率选择的反复使用、有限远期电芯分母、恒定输入导致的有效函数自由度差异、标准化视角与常数输入的优化条件差异，以及门缩放与核心幅度的冗余。两视角各 12 维中有 4 个电压相关统计量在该固定网格内恒定，名义参数匹配仍存在有效容量差异。置信度对执行和审计完整性为高，机制解释受上述设计限制。",
        "",
        "## 拓展思考",
        "",
        "对角门保持固定 U/V 的行列空间。性能改善所支持的对象是电池状态条件化的核心系数。随状态旋转 U/V 的动态子空间仍需要对应的直接对照。",
        "",
        "## 你可能想知道",
        "",
        "本轮训练、选权、诊断与配对统计均限定在源训练和源开发角色。最终目标 Query 的评分仍处于封存状态。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def audit(args):
    data, inputs, _, episodes = load_source(args.data_root)
    arrays, manifest = prepare_views(data, inputs, episodes)
    if read_json(args.output_root / "D3_FEATURE_MANIFEST.json") != manifest:
        raise RuntimeError("D3 independently rebuilt source feature manifest differs")
    meta = data.rows(episodes["source_cell_roles"]["source_meta_validation"])
    rows, gates, gate_summaries, checks = [], [], [], []
    uv_checks = 0
    bridge = read_json(args.prior_public / "P0/BRIDGE_SELECTION.json")[
        "selected_bridge"
    ]
    for seed in SEEDS:
        p0_checkpoint = (
            args.prior_public
            / "P0/formal"
            / bridge
            / "B0_pretrained_frozen"
            / f"seed{seed}"
            / "CHECKPOINT_SELECTED.pt"
        )
        expected_initial_sha256 = file_hash(p0_checkpoint)
        basis_path = args.p1_private / f"seed{seed}" / "P1_BASES.pt"
        public_basis = read_json(
            args.prior_public / "P1" / f"seed{seed}" / "P1_PUBLIC_COMPLETE.json"
        )
        if file_hash(basis_path) != public_basis["private_basis"]["sha256"]:
            raise RuntimeError("D3 P1 basis source file changed")
        bundle = torch.load(basis_path, map_location="cpu", weights_only=False)
        for mode in ALL_MODES:
            folder = (
                args.p2_private / "formal/grad_core/rank4" / f"seed{seed}"
                if mode == "static"
                else args.private_root / mode / f"seed{seed}"
            )
            receipt = read_json(folder / "SOURCE_DEVELOPMENT_COMPLETE.json")
            for item in receipt["files"]:
                if file_hash(folder / item["path"]) != item["sha256"]:
                    raise RuntimeError("D3 source training file hash changed")
            checkpoint = torch.load(
                folder / "CHECKPOINT_SELECTED.pt",
                map_location="cpu",
                weights_only=False,
            )
            if (
                checkpoint["identity"]["initial_checkpoint_sha256"]
                != expected_initial_sha256
                or receipt["target_query_labels_used"] != 0
                or not receipt["frozen_parameters_unchanged"]
            ):
                raise RuntimeError(
                    "D3 lost its common P0 initialization or source-only frozen-parameter guarantees"
                )
            state = checkpoint["model"]
            layer_names = [
                key[len("bridge.backbone.module.") : -2]
                for key in state
                if key.startswith("bridge.backbone.module.") and key.endswith(".U")
            ]
            if layer_names != list(bundle["bases"]):
                raise RuntimeError(
                    "D3 adapter layer ordering differs from the source basis manifest"
                )
            for name, (u, v) in bundle["bases"].items():
                for key, basis in (("U", u), ("V", v)):
                    saved = state["bridge.backbone.module." + name + "." + key]
                    if not torch.equal(saved, basis[:, :4].to(saved)):
                        raise RuntimeError(
                            "D3 selected U/V differs from the frozen P1 basis"
                        )
                    uv_checks += 1
            recomputed, difference = _independent_prediction_metrics(folder, data, meta)
            recomputed["mode"], recomputed["seed"] = mode, seed
            rows.append(recomputed)
            gate_check = {}
            if mode != "static":
                if sum(".core_gates." in key for key in state) != 48:
                    raise RuntimeError(
                        "D3 selected state has an unexpected number of gate tensors"
                    )
                initial = read_json(folder / "D3_INITIALIZATION_CHECK.json")
                if (
                    initial["status"] != "PASSED"
                    or initial["selected_state_gate_tensors"] != 48
                    or initial["initial_static_prediction_max_difference"] != 0
                ):
                    raise RuntimeError(
                        "D3 lacks its exact initial prediction equality check"
                    )
                records, summary, gate_check = _gate_parameters_and_values(
                    folder, mode, seed, state, layer_names, arrays, data, episodes
                )
                gates.extend(records)
                gate_summaries.extend(summary)
            checks.append(
                {
                    "mode": mode,
                    "seed": seed,
                    "npz_recomputed_metric_max_difference": difference,
                    "checkpoint_sha256": file_hash(folder / "CHECKPOINT_SELECTED.pt"),
                    "source_initial_checkpoint_sha256": checkpoint["identity"][
                        "initial_checkpoint_sha256"
                    ],
                    "uv_tensor_checks": 24,
                    "source_meta_physical_cells": len(meta),
                    "trainable_parameters": receipt["trainable_parameters"],
                    "target_query_labels_used": receipt["target_query_labels_used"],
                    **gate_check,
                }
            )
    table = pd.concat(rows, ignore_index=True)
    stored = pd.read_csv(args.output_root / "D3_CELL_METRICS.csv")
    paired = table.merge(
        stored,
        on=["mode", "seed", "cell_uid", "protocol_uid", "window"],
        suffixes=("_audit", "_original"),
        validate="one_to_one",
    )
    if len(paired) != 2700 or not np.allclose(
        paired.mae_audit, paired.mae_original, rtol=0, atol=2e-15, equal_nan=True
    ):
        raise RuntimeError("D3 final summary CSV does not match raw NPZ evidence")
    comparison_check = _recompute_comparisons(
        table, read_json(args.output_root / "D3_COMPARISONS.json")
    )
    pd.DataFrame(gates).to_csv(
        args.output_root / "D3_GATE_VALUES_AUDIT.csv", index=False
    )
    pd.DataFrame(gate_summaries).to_csv(
        args.output_root / "D3_GATE_SUMMARY_AUDIT.csv", index=False
    )
    audit_result = {
        "status": "PASSED",
        "units": 18,
        "new_training_units": 15,
        "raw_npz_cell_window_metrics_recomputed": len(table),
        "frozen_uv_tensor_comparisons": uv_checks,
        "selected_gate_tensor_checks": 15 * 48,
        "data_cache_sha256": file_hash(args.data_root / "MATR/COMPLETE.json"),
        "source_feature_manifest_reproduced": True,
        "final_target_query_labels_used": 0,
        "audit_device": "CPU",
        "optimizer_updates": 0,
        "script_sha256": file_hash(__file__),
        "units_checked": checks,
        "comparison_check": comparison_check,
        "geometry_scope": "Gate amplitude diagnostics preserve original U/V spans and do not establish rotating subspaces",
    }
    write_json(args.output_root / "D3_INDEPENDENT_AUDIT.json", audit_result)
    _write_report(
        args.report,
        table,
        gate_summaries,
        audit_result,
        read_json(args.output_root / "D3_GATE.json"),
        read_json(args.output_root / "D3_COMPARISONS.json"),
    )
    return audit_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path, default=ROOT / "runtime/drs_peft/data_v1"
    )
    parser.add_argument("--prior-public", type=Path, default=ROOT / "results/drs_peft")
    parser.add_argument("--p1-private", type=Path, default=ROOT / "runtime/drs_peft/P1")
    parser.add_argument("--p2-private", type=Path, default=ROOT / "runtime/drs_peft/P2")
    parser.add_argument(
        "--private-root", type=Path, default=ROOT / "runtime/drs_peft/diagnostics_v1/D3"
    )
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "results/drs_peft/diagnostics_v1/D3"
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "report/drs_peft/diagnostics_v1/D3_DUAL_VIEW_REPORT_ZH.md",
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = audit(args)
    print({key: value for key, value in result.items() if key != "units_checked"})


if __name__ == "__main__":
    main()
