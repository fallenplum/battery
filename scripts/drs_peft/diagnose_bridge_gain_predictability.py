"""Exploratory post-hoc test of whether prefix-only features predict Bridge gain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def prefix_summary(cache_path: Path) -> tuple[list[str], np.ndarray, list[str]]:
    with np.load(cache_path, allow_pickle=False) as archive:
        ids = archive["ids"].astype(str).tolist()
        features = archive["real_features"].astype(np.float64)
    means = features.mean(axis=1)
    stds = features.std(axis=1)
    first = features[:, :20].mean(axis=1)
    last = features[:, -20:].mean(axis=1)
    delta = last - first
    values = np.concatenate((means, stds, first, last, delta), axis=1)
    names = [f"{prefix}_{channel}" for prefix in ("mean", "std", "first20", "last20", "late_minus_early") for channel in range(features.shape[-1])]
    return ids, values, names


def evaluate(family: str, public: Path, cache: Path) -> tuple[pd.DataFrame, dict]:
    ids, x, names = prefix_summary(cache)
    cell = pd.read_csv(public / "BLIND_CELL.csv")
    wide = cell.groupby(["cell_uid", "method"], as_index=False).mae.mean().pivot(index="cell_uid", columns="method", values="mae")
    wide = wide.reindex(ids)
    rows, summary = [], {"family": family, "cells": len(ids), "features": len(names), "comparisons": {}}
    for baseline in ("HEAD_ONLY_TRAIN", "HEAD_LORA"):
        gain = wide[baseline].to_numpy() - wide["HEAD_BRIDGE_LORA"].to_numpy()
        correlations = []
        for index, name in enumerate(names):
            feature = x[:, index]
            if np.std(feature) == 0 or np.std(gain) == 0:
                continue
            correlations.append((name, float(pearsonr(feature, gain).statistic), float(spearmanr(feature, gain).statistic)))
        correlations.sort(key=lambda item: abs(item[2]), reverse=True)
        loo_pred = np.zeros_like(gain)
        loo = LeaveOneOut()
        for train_index, test_index in loo.split(x):
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            model.fit(x[train_index], gain[train_index])
            loo_pred[test_index] = model.predict(x[test_index])
        sign_accuracy = float(np.mean(np.sign(loo_pred) == np.sign(gain)))
        summary["comparisons"][baseline] = {
            "gain_definition": f"{baseline}_MAE_minus_HEAD_BRIDGE_LORA_MAE",
            "gain_mean": float(np.mean(gain)),
            "gain_positive_fraction": float(np.mean(gain > 0)),
            "loo_r2": float(r2_score(gain, loo_pred)) if len(gain) > 1 else float("nan"),
            "loo_sign_accuracy": sign_accuracy,
            "top_spearman_features": correlations[:5],
        }
        for uid, value, pred in zip(ids, gain, loo_pred):
            rows.append({"family": family, "baseline": baseline, "cell_uid": uid, "bridge_gain": float(value), "loo_predicted_gain": float(pred)})
    result = pd.DataFrame(rows)
    (public / "BRIDGE_GAIN_SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    result.to_csv(public / "BRIDGE_GAIN_PER_CELL.csv", index=False, lineterminator="\n")
    return result, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    _, summary = evaluate(args.family, args.public, args.cache)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

