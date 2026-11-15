#!/usr/bin/env python3
"""Source-only exploratory test of cross-prefix Q(V) persistence; NEVER evaluates test cells.
Run from the battery repository root. No raw waveform restoration is claimed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
CAPACITY = ("early_capacity_level", "early_capacity_slope", "early_capacity_std")
QV = ("qv_window_capacity_start_Ah", "qv_window_capacity_end_Ah",
      "delta_qv_mean_Ah", "delta_qv_std_Ah", "delta_qv_log10_variance",
      "delta_qv_range_Ah", "delta_qv_rms_Ah")
PERSISTENCE = ("delta_qv_mean_Ah", "delta_qv_std_Ah",
               "delta_qv_range_Ah", "delta_qv_rms_Ah")
SPLIT = {
    "HUST": "results/experiment_b/split.csv",
    "MATR": "results/experiment_b_matr/split.csv",
    "Stanford_2": "results/experiment_b_stanford/split.csv",
}
CONTEXT = {
    "MATR": "results/extension_matr/prior_context.csv",
    "Stanford_2": "results/extension_stanford/prior_context.csv",
}

def read_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def make_table(root, family, target):
    split = read_csv(root / SPLIT[family])
    split = split.loc[split["role"].isin(("train", "val"))].copy()
    assert split.cell_id.is_unique and len(split) > 0
    # Strong guard: no test cell ID, feature, or label enters this exploratory fit.
    assert not (split.role == "test").any()

    features = read_csv(root / "results/d0/early_observed_features.csv")
    features = features.loc[features.family.eq(family)].copy()
    keep = list(dict.fromkeys(CAPACITY + QV + PERSISTENCE))
    for k in (10, 50, 100):
        part = features.loc[features.prefix_cutoff_supplied_cycle_number.eq(k),
                            ["cell_id"] + keep].copy()
        assert part.cell_id.is_unique
        part = part.rename(columns={p: f"{p}_K{k}" for p in keep})
        split = split.merge(part, on="cell_id", how="left", validate="one_to_one")

    if family == "HUST":
        # Historical protocol_group contains the three authenticated prior rates.
        rates = split.protocol_group.map(json.loads)
        assert rates.map(len).eq(3).all()
        for j in range(3):
            split[f"prior_discharge_{j+1}"] = rates.map(lambda a: float(a[j]))
        context = [f"prior_discharge_{j+1}" for j in range(3)]
        target_file = root / "results/d0/HUST_RAW_CAPACITY_TARGET_SUPPORT.csv"
    else:
        meta = read_csv(root / CONTEXT[family])
        context = sorted(p for p in meta.columns if p.startswith("prior_"))
        assert meta.cell_id.is_unique and context
        split = split.merge(meta[["cell_id"] + context], on="cell_id",
                            how="left", validate="one_to_one")
        target_file = root / "results/d0/future_target_support.csv"

    # Filter target source by approved family, and join ONLY train/val IDs.
    tgt = read_csv(target_file)
    if "family" in tgt:
        tgt = tgt.loc[tgt.family.eq(family)]
    tgt = tgt[["cell_id", target, f"{target}_reason"]].copy()
    assert tgt.cell_id.is_unique
    split = split.merge(tgt, on="cell_id", how="left", validate="one_to_one")
    split = split.loc[split[f"{target}_reason"].eq("AVAILABLE")].copy()
    assert split.cell_id.is_unique
    for c in context:
        split[c] = pd.to_numeric(split[c].replace(
            {"True": 1, "False": 0, "true": 1, "false": 0}), errors="raise")

    cap = [f"{k}_K100" for k in CAPACITY]
    shape = [f"{k}_K100" for k in QV]
    persistence = []
    for k in PERSISTENCE:
        a, b = f"{k}_K50_minus_K10", f"{k}_K100_minus_K50"
        split[a] = split[f"{k}_K50"] - split[f"{k}_K10"]
        split[b] = split[f"{k}_K100"] - split[f"{k}_K50"]
        persistence.extend((a, b))

    arms = {
        "capacity": context + cap,
        "qv_K100": context + cap + shape,
        "qv_K100_plus_drift": context + cap + shape + persistence,
    }
    used = list(dict.fromkeys(x for cols in arms.values() for x in cols))
    for c in used + [target]:
        split[c] = pd.to_numeric(split[c], errors="coerce")
    before = len(split)
    split = split.replace([np.inf, -np.inf], np.nan).dropna(subset=used + [target])
    excluded = before - len(split)
    assert not (split.role == "test").any()
    return split, arms, excluded, target_file

def fit(alpha, X, y):
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, fit_intercept=True))
    model.fit(X, y)
    return model

def choose_alpha(X, y):
    # Same deterministic interleaved 3-fold TRAIN-only screening used for
    # the supplemental JS diagnostic; validation stays untouched until selection.
    n = len(y)
    best = None
    for alpha in ALPHAS:
        errors = []
        for fold in range(3):
            mask = np.arange(n) % 3 == fold
            if mask.sum() == 0 or (~mask).sum() < 3:
                raise ValueError("Insufficient source-train CV support")
            f = fit(alpha, X[~mask], y[~mask])
            errors.append(float(np.mean(np.abs(f.predict(X[mask]) - y[mask]))))
        cv = float(np.mean(errors))
        if best is None or cv < best[1]:
            best = (alpha, cv)
    return best

def evaluate(root, out):
    results, predictions, identities = [], [], {}
    for family in SPLIT:
        for target in ("r200_500", "r500_1000"):
            d, arms, excluded, target_file = make_table(root, family, target)
            tr = d.loc[d.role.eq("train")].sort_values("cell_id").reset_index(drop=True)
            va = d.loc[d.role.eq("val")].sort_values("cell_id").reset_index(drop=True)
            assert len(tr) >= 12 and len(va) >= 5
            for arm, columns in arms.items():
                X = tr[columns].to_numpy(dtype=float)
                y = tr[target].to_numpy(dtype=float) * 10000.0
                Xv = va[columns].to_numpy(dtype=float)
                yv = va[target].to_numpy(dtype=float) * 10000.0
                alpha, train_cv_mae = choose_alpha(X, y)
                model = fit(alpha, X, y)
                pred = model.predict(Xv)
                mae = float(np.mean(np.abs(pred - yv)))
                denom = np.sum((yv - yv.mean()) ** 2)
                r2 = float(1.0 - np.sum((pred - yv) ** 2) / denom) if denom > 0 else None
                results.append(dict(family=family, target=target, arm=arm,
                                    train_cells=len(tr), val_cells=len(va),
                                    excluded_missing=excluded, n_features=len(columns),
                                    alpha=alpha, train3fold_mae_times_1e4=train_cv_mae,
                                    source_val_mae_times_1e4=mae, source_val_r2=r2))
                for ident, truth, yhat in zip(va.cell_id, yv, pred):
                    predictions.append(dict(family=family, target=target,
                                            arm=arm, cell_id=ident,
                                            true_times_1e4=float(truth),
                                            predicted_times_1e4=float(yhat),
                                            abs_error_times_1e4=float(abs(yhat-truth))))
            identities[f"{family}:{target}"] = dict(target_sha256=sha(target_file),
                         source_train_ids=tr.cell_id.tolist(),
                         source_val_ids=va.cell_id.tolist())
    out.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(results).to_csv(out / "SOURCE_VALIDATION_METRICS.csv", index=False)
    pd.DataFrame(predictions).to_csv(out / "SOURCE_VALIDATION_PREDICTIONS.csv", index=False)
    (out / "SOURCE_VALIDATION_CONTRACT.json").write_text(json.dumps({
        "status": "EXPLORATORY_TRAIN_VAL_ONLY",
        "statement": "These validation cells and source choices were previously used in research; no independent blind test or raw-local-QV modeling performed.",
        "data_identity": identities,
        "regression": "standardized Ridge, train-internal interleaved 3 folds, fixed alpha grid",
        "targets": "HUST raw-counter; MATR/Stanford provided-capacity on supplied-cycle coordinate",
        "model_arms": list(arms), "feature_scope": "published K10/K50/K100 scalar summaries",
        "test_scoring_count": 0,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(results).to_string(index=False))
    print("NO TEST CELLS EVALUATED; exploratory source-validation only")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    evaluate(a.root.resolve(), a.output.resolve())

if __name__ == "__main__":
    main()
