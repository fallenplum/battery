"""Build the Stage A validation curves from its locked results ledger."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from scripts.drs_peft.data import file_hash
from scripts.drs_peft.run_abba_moirai import csv
from scripts.drs_peft.spectral_objective_protocol import PUBLIC
from scripts.drs_peft.train import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, default=PUBLIC)
    args = parser.parse_args()
    stage = read_json(args.public / "STAGE_A_DECISION.json")
    path = args.public / "VALIDATION_TRIALS.csv"
    if file_hash(path) != stage["files"][path.name] or stage["validation_rows"] != 399:
        raise RuntimeError("The Stage A validation grid is not authentic")
    rows = pd.read_csv(path, float_precision="round_trip")
    keys = ["fold", "model", "objective", "alpha", "lr", "steps", "ridge", "eta"]
    curves = rows.groupby(keys, dropna=False, as_index=False).agg(
        validation_mae=("query_mae", "mean"),
        validation_mse=("query_mse", "mean"),
        support_sampled_mse=("support_sampled_mse", "mean"),
        anchor_rms=("anchor_rms", "mean"),
        validation_draws=("draw", "size"),
    )
    if len(curves) != 171:
        raise RuntimeError("The Stage A validation curves have an incorrect number of points")
    k0 = curves.loc[curves.model.eq("K0")].set_index("fold").validation_mae
    curves["mae_minus_k0"] = curves.validation_mae - curves.fold.map(k0)
    csv(args.public / "VALIDATION_CURVES.csv", curves)
    graph, axes = plt.subplots(1, 3, figsize=(10.4, 3.55), sharey=True, layout="constrained")
    alpha = [1e-4, 1e-3, 1e-2, 1e-1]
    for fold, axis in enumerate(axes):
        for method, tag, color, marker in (
            ("ROTATION_TRUE_MM", "MM (Query MSE)", "#16758D", "s"),
            ("ROTATION_TRUE_MA", "MA (Query MAE)", "#A7472E", "o"),
        ):
            arm = curves.loc[curves.fold.eq(fold) & curves.model.eq(method)].sort_values("alpha")
            axis.plot(arm.alpha, arm.mae_minus_k0, color=color, marker=marker, linewidth=1.5, label=tag)
        axis.axhline(0, color="#353535", linewidth=0.9, label="K0")
        axis.axhline(-stage["mae_margin"], color="#527052", linewidth=1, linestyle="--", label="Stage A gate")
        axis.set_xscale("log")
        axis.set_xticks(alpha, [r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$"])
        axis.set_yscale("symlog", linthresh=1e-5)
        axis.set_title(f"Source fold {fold}")
        axis.set_xlabel("Shared Support step size")
        axis.grid(axis="y", alpha=0.17)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Validation MAE minus K0")
    graph.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=4, frameon=False)
    graph.savefig(args.public / "VALIDATION_CURVES.png", dpi=180, facecolor="white")
    plt.close(graph)


if __name__ == "__main__":
    main()
