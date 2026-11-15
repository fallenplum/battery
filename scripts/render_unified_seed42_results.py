"""Render the fixed seed42 analysis tables after checking their completion receipt."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ARMS = ("M0", "M1", "M2", "M3", "M3-NL", "M3-NP", "M3-NR")
CONTRASTS = ("M3_vs_M0", "M1_vs_M0", "M2_vs_M1", "M3_vs_M2",
             "M3_vs_M3-NL", "M3_vs_M3-NP", "M3_vs_M3-NR")
COLORS = ("#24607b", "#bd6528")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_analysis(root):
    complete_path = root / "ANALYSIS_COMPLETE.json"
    receipt = json.loads(complete_path.read_text(encoding="utf-8"))
    if receipt["status"] not in ("ANALYZED", "ANALYZED_WITH_REPRODUCTION_FAILURE") or receipt["training_seed"] != 42:
        raise RuntimeError("Figures require the complete seed42 analysis and its recorded qualification")
    for item in receipt["files"]:
        if sha256(root / item["path"]) != item["sha256"]:
            raise RuntimeError("Analysis output changed after completion")
    return receipt


def tidy_axis(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="x", color="#d6dce0", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.tick_params(length=3)


def trajectory_panel(axis, table, title):
    selected = table.loc[table.metric == "trajectory_mae"].set_index("arm").loc[list(ARMS)]
    y = np.arange(len(ARMS))
    axis.plot(selected.cell_mean * 100, y - 0.13, "o", color=COLORS[0], label="Cell mean", markersize=5)
    axis.plot(selected.protocol_macro * 100, y + 0.13, "s", color=COLORS[1], label="Protocol macro", markersize=4.5)
    axis.set_yticks(y, ARMS)
    axis.invert_yaxis()
    axis.set_xlim(left=0)
    axis.set_xlabel("Trajectory MAE (SOH percentage points)")
    axis.set_title(f"{title}\n{int(selected.supported_cells.iloc[0])} cells, "
                   f"{int(selected.supported_protocols.iloc[0])} supported protocols",
                   loc="left", fontweight="bold")
    tidy_axis(axis)


def contrast_panel(axis, table, title):
    selected = table.loc[table.metric == "trajectory_mae"]
    y = np.arange(len(CONTRASTS))
    for offset, sampling, color, marker, label in (
        (-0.13, "paired_cell", COLORS[0], "o", "Paired cells"),
        (0.13, "protocol_group", COLORS[1], "s", "Protocol macro"),
    ):
        rows = selected.loc[selected.sampling_unit == sampling].set_index("contrast").loc[list(CONTRASTS)]
        mean = rows.improvement.to_numpy() * 100
        lower = rows.ci95_low.to_numpy() * 100
        upper = rows.ci95_high.to_numpy() * 100
        axis.hlines(y + offset, lower, upper, color=color, linewidth=1.3)
        axis.scatter(mean, y + offset, marker=marker, color=color, s=21, label=label, zorder=3)
    axis.axvline(0, color="#64717a", linewidth=0.8, linestyle="--")
    axis.set_yticks(y, [name.replace("_vs_", " vs ") for name in CONTRASTS])
    axis.invert_yaxis()
    axis.set_xlabel("Reference MAE minus candidate MAE (SOH pp)")
    axis.set_title(title, loc="left", fontweight="bold")
    tidy_axis(axis)


def save_figure(figure, output, name):
    files = []
    for suffix in ("png", "pdf"):
        path = output / f"{name}.{suffix}"
        figure.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
        files.append(path)
    plt.close(figure)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analysis_receipt = checked_analysis(args.analysis_root)
    repeat_warning = analysis_receipt["status"] == "ANALYZED_WITH_REPRODUCTION_FAILURE"
    footer = "Intervals use 2,000 paired bootstrap samples. Training-seed variation is unmeasured."
    if repeat_warning:
        footer += "\nM2 / M3-NR bitwise training-repeat check failed in all six splits."
    args.output.mkdir(parents=True, exist_ok=True)
    main_metrics = pd.read_csv(args.analysis_root / "main_oof_metrics.csv")
    protocol_metrics = pd.read_csv(args.analysis_root / "protocol_heldout_metrics.csv")
    main_contrasts = pd.read_csv(args.analysis_root / "main_oof_fixed_contrasts.csv")
    protocol_contrasts = pd.read_csv(args.analysis_root / "protocol_heldout_fixed_contrasts.csv")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.titlesize": 10, "axes.labelsize": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    figure, axes = plt.subplots(2, 2, figsize=(12, 8.5), layout="constrained")
    trajectory_panel(axes[0, 0], main_metrics, "a  Main five-fold OOF")
    trajectory_panel(axes[0, 1], protocol_metrics, "b  Unseen-protocol sensitivity")
    contrast_panel(axes[1, 0], main_contrasts, "c  Main OOF fixed contrasts")
    contrast_panel(axes[1, 1], protocol_contrasts, "d  Unseen-protocol fixed contrasts")
    axes[0, 0].legend(loc="lower left", frameon=False, fontsize=8)
    axes[1, 0].legend(loc="lower right", frameon=False, fontsize=8)
    figure.suptitle("MATR early K100 to observed future SOH, training seed 42", fontsize=12)
    figure.supxlabel(footer, fontsize=8)
    files = save_figure(figure, args.output, "trajectory_and_fixed_contrasts")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for axis, metric, title in zip(axes, ("r200_500_absolute_error", "r500_1000_absolute_error"),
                                   ("a  Observed endpoints 200 to 500", "b  Observed endpoints 500 to 1000")):
        y = np.arange(len(ARMS))
        counts = []
        for offset, table, color, marker, label in (
            (-0.13, main_metrics, COLORS[0], "o", "Main OOF"),
            (0.13, protocol_metrics, COLORS[1], "s", "Unseen protocol"),
        ):
            rows = table.loc[table.metric == metric].set_index("arm").loc[list(ARMS)]
            axis.scatter(rows.cell_mean * 1000, y + offset, color=color, marker=marker, s=24, label=label)
            counts.append(f"{label} n={int(rows.supported_cells.iloc[0])}")
        axis.set_yticks(y, ARMS)
        axis.invert_yaxis()
        axis.set_xlim(left=0)
        axis.set_xlabel("Rate MAE (SOH per 1,000 cycles)")
        axis.set_title(title + "\n" + ", ".join(counts), loc="left", fontweight="bold")
        tidy_axis(axis)
    axes[0].legend(loc="upper left", frameon=False, fontsize=8)
    figure.suptitle("Secondary observed average degradation rates, training seed 42", fontsize=12)
    if repeat_warning:
        figure.supxlabel("M2 / M3-NR bitwise training-repeat check failed in all six splits.", fontsize=8)
    files.extend(save_figure(figure, args.output, "secondary_rate_errors"))
    receipt = {"status": "RENDERED_WITH_REPRODUCTION_FAILURE" if repeat_warning else "PASS", "training_seed": 42,
               "analysis_status": analysis_receipt["status"],
               "analysis_complete_sha256": sha256(args.analysis_root / "ANALYSIS_COMPLETE.json"),
               "renderer_source_sha256": sha256(Path(__file__)),
               "files": [{"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in files],
               "interpretation": "All seven fixed arms and all seven fixed contrasts retained. Intervals quantify paired cell or protocol resampling, conditional on seed42 and historical cohort exposure."}
    (args.output / "FIGURE_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "figures": len(files), "training_seed": 42}))


if __name__ == "__main__":
    main()
