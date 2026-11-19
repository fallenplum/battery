"""Test whether legal early prefixes predict the transfer of a source correction."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from scripts.drs_peft.data import CACHE, ROOT, file_hash
from scripts.drs_peft.e123_protocol import WINDOWS
from scripts.drs_peft.transfer_scope_data import load_prefixes


PUBLIC = ROOT / "results/drs_peft/correction_transfer_v1"
PRIVATE = ROOT / "runtime/drs_peft/correction_transfer_v1"
REGISTRATION = ROOT / "configs/drs_peft/onesided_fewshot_v1/EPISODES.json"
REFERENCE = ROOT / "results/drs_peft/onesided_fewshot_v1/references"
CORE_METHODS = ("ABBA_R4X4", "LORA_R8")
HORIZONS = tuple(window for window, _, _ in WINDOWS)
BANDS = ((1, 10), (81, 100))
FRACTIONS = np.linspace(.1, .9, 9)
PHYSICAL = {
    "current": ("median_abs_I_C", "median_within_curve_I_range_C", "late_minus_early_abs_I_C"),
    "voltage": ("median_observed_V_low", "median_observed_V_high", "median_observed_V_span"),
    "capacity": ("prefix_anchor", "early_observed_Q_span_SOH", "late_observed_Q_span_SOH",
                 "late_minus_early_Q_span_SOH", "Q_span_slope_SOH_per_100cycle"),
    "local_deformation": ("late_minus_early_V_at_relative_Q_10_30",
                          "late_minus_early_V_at_relative_Q_40_60",
                          "late_minus_early_V_at_relative_Q_70_90"),
}
DISTANCES = (*PHYSICAL, "representation")
VIEWS = {
    "horizon_prevalence": (), "representation": ("representation",),
    "physical": tuple(PHYSICAL), "physical_and_representation": DISTANCES,
}
LABELS = ("initial_direction_helpful", "full_step_helpful")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_bytes())


def save_csv(path, value):
    tmp = path.with_suffix(".pending.csv")
    value.to_csv(tmp, index=False, lineterminator="\n")
    tmp.replace(path)


def save_json(path, value):
    pending = path.with_suffix(".pending.json")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def band_median(values, cycles, usable):
    output = []
    for lower, upper in BANDS:
        available = usable & (cycles >= lower) & (cycles <= upper)
        output.append(np.median(values[available], axis=0) if available.sum() >= 5 else np.full(values.shape[1:], np.nan))
    return output


def normalized_voltage_curve(voltage, charge):
    """Inverse Q(V) on empirical relative capacity within the observed curve."""
    order = np.argsort(charge, kind="stable")
    q, indices = np.unique(charge[order], return_index=True)
    if len(q) < 5 or q[-1] - q[0] <= 1e-8:
        return np.full(len(FRACTIONS), np.nan)
    fraction = (q - q[0]) / (q[-1] - q[0])
    return np.interp(FRACTIONS, fraction, voltage[order][indices])


def describe_prefix(x, mask, cycles, anchor):
    good = mask.sum(axis=1) >= 5
    require(x.shape == (100, 64, 4) and mask.shape == (100, 64) and cycles.shape == (100,)
            and 0 < good.sum() and np.isfinite(x[..., :3][mask]).all()
            and ((cycles[mask.any(axis=1)] >= 1) & (cycles[mask.any(axis=1)] <= 100)).all()
            and np.isfinite(anchor), "Prefix contains unsupported chronology, shape, or units")
    i_level, i_range, q_span = np.full((3, len(cycles)), np.nan)
    v_bounds = np.full((len(cycles), 3), np.nan)
    deformation = np.full((len(cycles), len(FRACTIONS)), np.nan)
    for index in np.flatnonzero(good):
        voltage, current, charge = x[index, mask[index], :3].astype(np.float64).T
        i_level[index] = np.median(np.abs(current))
        i_range[index] = np.ptp(np.quantile(current, (.05, .95)))
        v_bounds[index] = voltage.min(), voltage.max(), voltage.max() - voltage.min()
        q_span[index] = np.ptp(charge)
        deformation[index] = normalized_voltage_curve(voltage, charge)
    early_i, late_i = band_median(i_level, cycles, good)
    early_q, late_q = band_median(q_span, cycles, good)
    time = cycles[good].astype(np.float64)
    centered = time - time.mean()
    slope = 100 * np.dot(centered, q_span[good]) / np.dot(centered, centered) if np.ptp(time) else np.nan
    early_v, late_v = band_median(deformation, cycles, np.isfinite(deformation).all(axis=1))
    local_shift = (late_v - early_v).reshape(3, 3).mean(axis=1)
    return np.r_[np.median(i_level[good]), np.median(i_range[good]), late_i - early_i,
                 np.median(v_bounds[good], axis=0), anchor, early_q, late_q, late_q - early_q, slope, local_shift]


def source_prefix_features(prefixes, source_ids):
    source_ids = sorted(set(source_ids))
    names = [name for names in PHYSICAL.values() for name in names]
    data = prefixes.inputs
    values = [describe_prefix(data["x"][i], data["point_mask"][i], data["cycle_ids"][i], prefixes.anchor[i])
              for i in prefixes.rows(source_ids)]
    frame = pd.DataFrame(values, index=source_ids, columns=names)
    audit = {name: {"finite_cells": int(frame[name].count()),
                    "unique_at_8_decimal_places": int(frame[name].round(8).nunique()),
                    "sample_sd": float(frame[name].std(ddof=0)) if frame[name].count() else None}
             for name in frame}
    return frame, audit


def load_pair_rows(path, registration, reference):
    raw = pd.read_csv(path, float_precision="round_trip")
    needed = {"domain", "fold", "seed", "method", "K", "support_label_view", "population",
              "cell_uid", "support_uid", "window", "n_obs", "dot_ed", "energy_dd", "delta_mse", "eta",
              "parent_reference_sha256"}
    require(needed <= set(raw), f"Source artifact is missing {sorted(needed - set(raw))}")
    rows = raw[raw.K.eq(1) & raw.domain.eq("MATR") & raw.population.eq("QUERY")
               & raw.support_label_view.eq("observed_full") & raw.method.isin(CORE_METHODS)].copy()
    require(len(rows) > 0 and set(rows.seed) == {42} and set(rows.method) == set(CORE_METHODS)
            and set(rows.window) == set(HORIZONS) and rows.eta.eq(1).all()
            and rows.n_obs.ge(0).all() and rows.n_obs.eq(np.floor(rows.n_obs)).all(), "Source single-Support axes differ")
    protocols = registration["source_protocols"]
    rows["support_protocol"] = rows.support_uid.map(protocols)
    rows["query_protocol"] = rows.cell_uid.map(protocols)
    roles = {task["fold_id"]: task for task in registration["source_tasks"]["inner"]}
    sealed = set(registration["sealed_final_MATR_ids"])
    key = ["fold", "seed", "method", "support_uid", "cell_uid", "window"]
    require(not rows.duplicated(key).any(), "Physical Support/Query pairs were repeated")
    require(not (set(rows.support_uid) | set(rows.cell_uid)) & sealed, "A sealed target cell reached source E3")
    for fold, task in roles.items():
        group = rows[rows.fold.eq(fold)]
        donors = set(group.support_uid)
        query = set(task["query_ids"])
        require(donors == set(task["support_pool_ids"]) and set(group.cell_uid) == query and not donors & query,
                f"Wrong frozen source role membership in fold {fold}")
        require(len(group) == len(donors) * len(query) * len(CORE_METHODS) * len(HORIZONS),
                f"Incomplete frozen pair grid for fold {fold}")
        require(group.parent_reference_sha256.eq(reference[str(fold)]["reference_checkpoint_sha256"]).all(),
                f"Source correction and frozen representation reference differ in fold {fold}")
    require(set(rows.fold) == set(roles), "Unexpected source fold axis")
    valid = rows.n_obs.gt(0)
    require(np.isfinite(rows.loc[valid, ["dot_ed", "energy_dd", "delta_mse"]]).to_numpy().all()
            and rows.loc[valid, "energy_dd"].ge(0).all(), "Invalid correction geometry in observed windows")
    residual = rows.loc[valid, "delta_mse"] - 2 * rows.loc[valid, "dot_ed"] - rows.loc[valid, "energy_dd"]
    require(np.abs(residual).max() <= 1e-10, "Source finite-step MSE does not reproduce from the directional decomposition")
    active = rows[valid & rows.energy_dd.gt(0)].copy()
    active["initial_direction_helpful"] = active.dot_ed.lt(0)
    active["full_step_helpful"] = active.delta_mse.lt(0)
    active["max_helpful_positive_scale"] = -2 * active.dot_ed / active.energy_dd
    active["same_protocol"] = active.support_protocol.eq(active.query_protocol)
    return active.reset_index(drop=True), dict(scored_pair_windows=len(rows), unsupported_windows=int((~valid).sum()),
                                               zero_update_windows=int((valid & rows.energy_dd.eq(0)).sum()),
                                               decomposition_max_abs=float(np.abs(residual).max()))


def read_latent(path, rows):
    latent = {}
    registered = set(zip(rows.fold, rows.support_uid)) | set(zip(rows.fold, rows.cell_uid))
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            value = json.loads(line)
            identity = value["fold"], value["cell_uid"]
            z = np.asarray(value["z"], dtype=np.float64)
            require(identity not in latent and z.shape == (64,) and np.isfinite(z).all(), "Bad frozen representation record")
            if identity in registered:
                latent[identity] = z
    require(set(latent) == registered, "Frozen representation roster does not cover every active source pair")
    return latent


def physical_scale(values):
    """Scale from unique predictor-fit cells and discard constant channels."""
    count = values.count()
    scale = values.std(ddof=0)
    return scale.where(count.ge(2) & scale.gt(1e-8))


def distances(rows, features, latent, scale):
    left = features.loc[rows.support_uid].to_numpy(dtype=np.float64)
    right = features.loc[rows.cell_uid].to_numpy(dtype=np.float64)
    difference = pd.DataFrame((left - right) / scale.to_numpy(), columns=features.columns, index=rows.index)
    output = {name: difference.loc[:, list(columns)].pow(2).mean(axis=1).pow(.5)
              for name, columns in PHYSICAL.items()}
    output["representation"] = [np.linalg.norm(latent[fold, support] - latent[fold, query]) / 8
                                for fold, support, query in zip(rows.fold, rows.support_uid, rows.cell_uid, strict=True)]
    return pd.DataFrame(output, index=rows.index)


def fit_predict(train_x, test_x, target, weights):
    baseline = np.average(target, weights=weights)
    supported = train_x.columns[train_x.count().gt(1) & train_x.std(ddof=0).gt(1e-8)]
    if not len(supported) or np.unique(target).size == 1:
        return np.full(len(test_x), baseline), len(supported), True
    train_x, test_x = train_x[supported], test_x[supported]
    median = train_x.median()
    train_x, test_x = train_x.fillna(median), test_x.fillna(median)
    scaler = StandardScaler().fit(train_x, sample_weight=weights)
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500)
    model.fit(scaler.transform(train_x), target, sample_weight=weights)
    return model.predict_proba(scaler.transform(test_x))[:, 1], len(supported), False


def validate_reference(registration, reference_root):
    protocols = registration["source_protocols"]
    ancestry = {}
    for task in registration["source_tasks"]["inner"]:
        fold = task["fold_id"]
        path = reference_root / f"fold{fold}" / "seed42/REFERENCE_COMPLETE.json"
        frozen = read_json(path)
        heldout = set(task["query_ids"]) | set(task["support_pool_ids"])
        fit = set(frozen["fit_ids"])
        require(not fit & heldout and not {protocols[uid] for uid in fit} & {protocols[uid] for uid in heldout}
                and frozen["final_target_query_labels_used"] == 0, f"Fold {fold} reference fit crosses source inner roles")
        ancestry[str(fold)] = {"fit_cells": len(fit), "reference_checkpoint_sha256": frozen["reference_checkpoint_sha256"],
                               "reference_receipt_sha256": file_hash(path)}
    return ancestry


def grouped_predict(rows, features, latent):
    observations, associations, coverage = [], [], []
    audit_cells = ["fold", "seed", "method", "window", "support_uid", "cell_uid", "support_protocol", "query_protocol", "same_protocol"]
    outcomes = [*LABELS, "max_helpful_positive_scale", "dot_ed", "energy_dd", "delta_mse", "n_obs"]
    for (fold, method, window), block in rows.groupby(["fold", "method", "window"], sort=True):
        for group in sorted(block.query_protocol.unique()):
            test = block[block.query_protocol.eq(group)]
            query_ids = set(test.cell_uid)
            train = block[~(block.query_protocol.eq(group) | block.support_protocol.eq(group)
                            | block.cell_uid.isin(query_ids) | block.support_uid.isin(query_ids))]
            require(not ({group} & (set(train.query_protocol) | set(train.support_protocol))),
                    f"Protocol purity differs in fold {fold}/{method}/{window}/{group}")
            train_ids = sorted(set(train.support_uid) | set(train.cell_uid))
            require(not set(train_ids) & query_ids, "A held-out Query cell entered preprocessing")
            scale = physical_scale(features.loc[train_ids])
            record = dict(fold=int(fold), method=method, window=window, query_protocol=group,
                          train_pairs=len(train), test_pairs=len(test), heldout_query_cells=len(query_ids),
                          train_query_groups=train.query_protocol.nunique(), train_unique_physical_cells=len(train_ids),
                          excluded_support_protocol_rows=int(block[~block.query_protocol.eq(group)].support_protocol.eq(group).sum()),
                          active_physical_cell_columns=int(scale.count()),
                          status="SCORED" if len(train) else "NO_LEGAL_TRAIN_GROUP_FOR_WINDOW")
            coverage.append(record)
            test_x = distances(test, features, latent, scale)
            test_data = test[audit_cells + outcomes].join(test_x)
            test_data["alignment_but_overshoot"] = test_data.initial_direction_helpful & ~test_data.full_step_helpful
            train_x = distances(train, features, latent, scale)
            for name in DISTANCES:
                median = train_x[name].median() if train_x[name].count() >= 2 else np.nan
                variability = train_x[name].std(ddof=0)
                if np.isfinite(median) and variability > 1e-8:
                    test_data[f"{name}_stratum"] = np.where(test_x[name].isna(), "unavailable",
                                                            np.where(test_x[name].le(median), "near", "far"))
                else:
                    test_data[f"{name}_stratum"] = "unavailable"
            associations.append(test_data)
            if not len(train):
                continue
            weights = (1 / train.groupby("query_protocol").fold.transform("size")).to_numpy()
            weights /= weights.mean()
            for outcome, (view, columns) in product(LABELS, VIEWS.items()):
                y = train[outcome].to_numpy(dtype=int)
                probability, n_features, used_prevalence = fit_predict(train_x[list(columns)], test_x[list(columns)], y, weights)
                val = test_data[audit_cells].assign(
                    outcome=outcome, view=view, y=test[outcome].astype(int), probability=probability,
                    usable_distance_features=n_features, used_prevalence_fallback=used_prevalence,
                )
                observations.append(val)
    require(len(observations) > 0, "No source prediction folds were scored")
    return pd.concat(observations, ignore_index=True), pd.concat(associations, ignore_index=True), pd.DataFrame(coverage)


def evaluate(predictions):
    predicted = predictions.assign(
        brier=(predictions.y - predictions.probability) ** 2,
        correct=predictions.y.eq(predictions.probability.ge(.5)),
    )
    score_axes = ["fold", "query_protocol", "method", "window", "outcome", "view"]
    heldout = predicted.groupby(score_axes, as_index=False).agg(
        pair_count=("y", "size"), useful_rate=("y", "mean"), brier=("brier", "mean"), accuracy=("correct", "mean"),
        physical_query_cells=("cell_uid", "nunique"), prevalence_fallback=("used_prevalence_fallback", "all"),
    )
    group_axes = score_axes[:-1]
    baseline = heldout[heldout.view.eq("horizon_prevalence")][group_axes + ["brier"]].rename(columns={"brier": "constant_brier"})
    heldout = heldout.merge(baseline, on=group_axes, validate="many_to_one")
    heldout["brier_gain_over_prevalence"] = heldout.constant_brier - heldout.brier
    rep = heldout[heldout.view.eq("representation")][group_axes + ["brier"]].rename(columns={"brier": "representation_brier"})
    heldout = heldout.merge(rep, on=group_axes, validate="many_to_one")
    heldout["brier_gain_over_representation"] = heldout.representation_brier - heldout.brier
    axis = ["method", "window", "outcome", "view"]
    summaries = []
    for key, fold_groups in heldout.groupby(axis, sort=True):
        method, window, outcome, view = key
        selected = predicted[predicted.method.eq(method) & predicted.window.eq(window)
                             & predicted.outcome.eq(outcome) & predicted.view.eq(view)]
        fold_gain = fold_groups.groupby("fold").brier_gain_over_prevalence.mean()
        fold_increment = fold_groups.groupby("fold").brier_gain_over_representation.mean()
        both = selected.y.nunique() == 2
        summaries.append(dict(zip(axis, key, strict=True)) | {
            "n_fold_protocol_groups": len(fold_groups), "n_distinct_protocols": fold_groups.query_protocol.nunique(),
            "physical_query_cells": selected.cell_uid.nunique(), "physical_support_cells": selected.support_uid.nunique(),
            "n_pair_windows": len(selected), "brier_group_macro": fold_groups.brier.mean(),
            "acc_group_macro": fold_groups.accuracy.mean(),
            "balanced_accuracy_pooled_descriptive": balanced_accuracy_score(selected.y, selected.probability.ge(.5)) if both else np.nan,
            "roc_auc_pooled_descriptive": roc_auc_score(selected.y, selected.probability) if both else np.nan,
            "brier_gain_over_prevalence": fold_groups.brier_gain_over_prevalence.mean(),
            "brier_gain_over_representation": fold_groups.brier_gain_over_representation.mean(),
            "folds": len(fold_gain), "folds_improved_over_prevalence": int(fold_gain.gt(0).sum()),
            "folds_improved_over_representation": int(fold_increment.gt(0).sum()),
            "groups_with_prevalence_fallback": int(fold_groups.prevalence_fallback.sum()),
        })
    return heldout, pd.DataFrame(summaries)


def descriptor_associations(rows):
    statistics = []
    for feature in ("same_protocol", *DISTANCES):
        strata = "same_protocol" if feature == "same_protocol" else f"{feature}_stratum"
        for key, group in rows.groupby(["method", "window", strata], sort=True):
            method, window, band = key
            statistics.append(dict(
                method=method, window=window, descriptor=feature, stratum=band,
                pairs=len(group), query_cells=group.cell_uid.nunique(), support_cells=group.support_uid.nunique(),
                initial_helpful_rate=group.initial_direction_helpful.mean(), full_step_helpful_rate=group.full_step_helpful.mean(),
                alignment_but_overshoot=int(group.alignment_but_overshoot.sum()),
                median_MSE_improvement=float((-group.delta_mse).median()),
            ))
    return pd.DataFrame(statistics)


def format_markdown(rows):
    return "\n".join("| " + " | ".join(map(str, row)) + " |" for row in rows)


def render_report(scores, associations, pair_details, predictions, feature_audit, coverage):
    core = scores[scores.view.isin(("representation", "physical", "physical_and_representation"))]
    display = core[core.window.isin(("all", "1001_plus"))]
    table = [["方法", "时距", "目标", "预测量", "Brier", "比训练组先验改善", "改善折数", "比表征距离增量"],
             ["---"] * 8]
    for row in display.itertuples(index=False):
        table.append([row.method, row.window, row.outcome, row.view, f"{row.brier_group_macro:.6f}",
                      f"{row.brier_gain_over_prevalence:+.6f}", f"{row.folds_improved_over_prevalence}/{row.folds}",
                      f"{row.brier_gain_over_representation:+.6f}"])
    same = associations[associations.descriptor.eq("same_protocol") & associations.window.eq("all")]
    strata = [["方法", "同协议", "电芯对", "有利方向率", "全步有利率", "初始有利但全步未改善"], ["---"] * 6]
    for row in same.itertuples(index=False):
        strata.append([row.method, row.stratum, row.pairs, f"{row.initial_helpful_rate:.3f}",
                       f"{row.full_step_helpful_rate:.3f}", row.alignment_but_overshoot])
    robust = core[(core.brier_gain_over_prevalence > 0) & (2 * core.folds_improved_over_prevalence > core.folds)]
    focus = robust[robust.window.eq("all")]
    midterm = robust[robust.window.isin(("101_200", "201_500", "501_1000"))]
    scored_far = predictions[predictions.window.eq("1001_plus")]
    observed_far = pair_details[pair_details.window.eq("1001_plus")]
    n_protocols = pair_details.query_protocol.nunique()
    available = [name for name, audit in feature_audit.items() if audit["unique_at_8_decimal_places"] > 1]
    examples = pair_details[pair_details.alignment_but_overshoot & pair_details.window.eq("all")]
    support_quotes = []
    for method, rows in examples.groupby("method"):
        row = rows.loc[rows.max_helpful_positive_scale.idxmin()]
        support_quotes.append(f"- {method} 的一个完整窗例子来自折 {row.fold}。"
                              f"{row.support_uid} 学得的方向对 {row.cell_uid} 有利，正缩放上界为 "
                              f"{row.max_helpful_positive_scale:.4g}，全步 MSE 变化为 {row.delta_mse:+.6g}。")
    focus_text = "、".join(f"{r.method}/{r.outcome}/{r.view}" for r in focus.itertuples(index=False)) or "无"
    midterm_text = "\n".join(
        f"- {row.method} 的 {row.window} 窗使用 {row.view} 预测 {row.outcome}，Brier 改善 "
        f"{row.brier_gain_over_prevalence:+.6f}，{row.folds_improved_over_prevalence}/{row.folds} 折同向。"
        for row in midterm.itertuples(index=False)
    ) or "其他分窗也无符合该条件的结果。"
    sections = [
        "# 实验 3 源工况留出上的修正可预测性",
        f"源域同一折内逐个留出 Query 协议组，合计 {n_protocols} 个不同协议。"
        f"冻结参考分别来自三个源折。每次只用同折其他组拟合，且把留出协议的 Support 和留出 Query 电芯排除于模型拟合和标准化。"
        f"共核对 {len(coverage)} 个方法、窗口与协议组组合，其中 {int(coverage.status.eq('SCORED').sum())} 个可拟合留出预测。"
        "源内条件在前期试验中有历史使用，本诊断为回顾性验证。",
        r"方向有利用 $\langle e,d\rangle<0$。全步有利用 $2\langle e,d\rangle+\|d\|^2<0$。"
        r"方向有利时，MSE 的正缩放有利范围为 $0<\eta<-2\langle e,d\rangle/\|d\|^2$。"
        "两项分别用每个时距的训练组先验或固定 L2 Logistic 预测。Brier 先在评估协议组内平均，再对组取平均。"
        "单个电芯会被多个 Support 反复打分。窗口支持性参与评分范围确定，未来 mask、观测数与协议不进入任何预测特征。",
        f"在完整窗中，同时满足组宏平均 Brier 降低和多于一半源折同向的对照为 {focus_text}。"
        "这些是该轮稳定性线索，模型与超参数全程固定。",
        "## 留出预测评分",
        format_markdown(table),
        f"1001 圈以后共有 {observed_far.cell_uid.nunique()} 颗不同物理 Query 有观测。"
        f"其中 {scored_far.fold.nunique()} 折、{scored_far.cell_uid.nunique()} 颗物理 Query、"
        f"{scored_far.query_protocol.nunique()} 个协议的合法训练组能产出留出预测。远期结果需结合这些分母解释。",
        midterm_text,
        "## 协议与物理信息诊断",
        format_markdown(strata),
        f"{len(available)}/{len(feature_audit)} 项物理描述在 8 位小数处有变化。"
        "电流来自可用的早期放电 VIQ 网格，覆盖不到网格以外的激励。电压范围可能因预处理采用共同网格而接近常数。"
        "容量响应来自观测电压窗口内的 Q 范围与早期容量锚点。局部形变比较相对容量坐标上的早晚电压，"
        "其中已按每条曲线的可测 Q 窗口归一化。表征距离使用冻结的 64 维编码。"
        "逐描述量的近远分层阈值仅从该轮训练对估计。描述的关联与源域预测结果共同保留，置信度受协议与电芯数限制。",
        "## 对齐后仍变差的例子",
        "\n".join(support_quotes) if support_quotes else "完整窗没有这类可评分的例子。",
    ]
    return "\n\n".join(sections) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pairs", type=Path, default=PUBLIC / "e1/SOURCE_CELL.csv")
    parser.add_argument("--latent", type=Path, default=PRIVATE / "e1/SOURCE_LATENT.jsonl")
    parser.add_argument("--registration", type=Path, default=REGISTRATION)
    parser.add_argument("--data-root", type=Path, default=CACHE)
    parser.add_argument("--reference-root", type=Path, default=REFERENCE)
    parser.add_argument("--output", type=Path, default=PUBLIC / "e3")
    args = parser.parse_args()
    source_receipt_path = args.source_pairs.with_name("SOURCE_DIRECTION_COMPLETE.json")
    source_receipt = read_json(source_receipt_path)
    require(source_receipt["status"] == "COMPLETE" and source_receipt["scope"] == "SOURCE_DEVELOPMENT_FIRST"
            and source_receipt["target_rows_read"] == 0 and file_hash(args.source_pairs) == source_receipt["source_cells_sha256"]
            and file_hash(args.latent) == source_receipt["source_latent_sha256"],
            "Upstream source correction arrays are not complete or differ from the score receipt")
    registration = read_json(args.registration)
    reference = validate_reference(registration, args.reference_root)
    rows, found = load_pair_rows(args.source_pairs, registration, reference)
    ids = set(rows.support_uid) | set(rows.cell_uid)
    require(ids <= set(registration["source_ids"]), "Source pair metadata contain non-source physical IDs")
    prefixes = load_prefixes("MATR", args.data_root)
    features, feature_audit = source_prefix_features(prefixes, ids)
    latent = read_latent(args.latent, rows)
    predicted, pair_details, coverage = grouped_predict(rows, features, latent)
    keys = ["fold", "method", "window", "support_uid", "cell_uid"]
    require(len(pair_details) == len(rows) and not pair_details.duplicated(keys).any()
            and not predicted.duplicated([*keys, "outcome", "view"]).any()
            and predicted.probability.between(0, 1).all(), "Invalid out-of-fold record identity or probability")
    group_scores, scores = evaluate(predicted)
    association = descriptor_associations(pair_details)
    args.output.mkdir(parents=True, exist_ok=True)
    outputs = {
        "PAIR_DETAILS.csv": pair_details, "OOF_PREDICTIONS.csv": predicted,
        "SCORES_BY_HELDOUT_GROUP.csv": group_scores, "SCORES_BY_WINDOW.csv": scores,
        "DESCRIPTOR_ASSOCIATIONS.csv": association, "GROUP_HOLDOUT_COVERAGE.csv": coverage,
    }
    for filename, table in outputs.items():
        save_csv(args.output / filename, table)
    report = args.output / "E3_ZH.md"
    report.write_text(render_report(scores, association, pair_details, predicted, feature_audit, coverage), encoding="utf-8", newline="\n")
    receipt = {
        "status": "COMPLETE_SOURCE_PROTOCOL_HOLDOUT_DIAGNOSTIC",
        "registration_sha256": file_hash(args.registration), "source_pair_scores_sha256": file_hash(args.source_pairs),
        "source_direction_receipt_sha256": file_hash(source_receipt_path),
        "source_latent_sha256": file_hash(args.latent), "source_prefix_cache_manifest_sha256": prefixes.cache_manifest_sha256,
        "analysis_code_sha256": file_hash(Path(__file__)), "fold_references": reference,
        "source_pair_window_coverage": found, "oof_prediction_rows": len(predicted),
        "group_holdout_plans": len(coverage), "group_holdout_scored": int(coverage.status.eq("SCORED").sum()),
        "source_descriptor_cells": len(ids), "source_query_physical_cells": rows.cell_uid.nunique(),
        "source_support_physical_cells": rows.support_uid.nunique(), "source_query_protocol_groups": rows.query_protocol.nunique(),
        "source_fold_protocol_groups": len(rows[["fold", "query_protocol"]].drop_duplicates()),
        "adaptation_states": len(rows[["fold", "method", "support_uid"]].drop_duplicates()),
        "feature_audit": feature_audit, "early_cycle_interval": [1, 100], "fixed_band_intervals": list(BANDS),
        "prediction_designs": VIEWS, "model_C": 1., "test_window_bounds_given": True,
        "training_only_physical_scale": True, "n_obs_is_model_feature": False, "protocol_is_model_feature": False,
        "heldout_query_protocol_in_predictor_training": False, "historically_exposed_source": True,
        "final_target_query_label_reads": 0, "source_future_array_reads_by_E3": 0,
        "cuda_training_updates_by_E3": 0,
        "ablation_scores": {name: file_hash(args.output / name) for name in (*outputs, report.name)},
    }
    save_json(args.output / "COMPLETE.json", receipt)
    print(f"E3 source grouped holdout completed for {len(rows)} pair-window observations across {receipt['source_query_protocol_groups']} distinct protocols.")


if __name__ == "__main__":
    main()
