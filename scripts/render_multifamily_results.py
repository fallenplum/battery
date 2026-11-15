"""Render paired-error uncertainty for the completed family extension."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def main():
    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False,
                         "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for column, (family, directory) in enumerate((("MATR", "experiment_b_matr"), ("Stanford", "experiment_b_stanford"))):
        path = ROOT / "results" / directory
        cells = {row["target"]: row for row in read(path / "paired_mae_improvement.csv")}
        groups = {row["target"]: row for row in read(path / "protocol_macro_paired_improvement.csv")}
        for row_index, target in enumerate(("r200_500", "r500_1000")):
            axis = axes[row_index, column]
            for position, row, value_name, color in ((1, cells[target], "MAE_improvement", "#235a87"),
                                                      (0, groups[target], "protocol_macro_MAE_improvement", "#b35d26")):
                point = float(row[value_name]) * 1e4
                low = float(row["CI95_low"]) * 1e4
                high = float(row["CI95_high"]) * 1e4
                axis.errorbar(point, position, xerr=[[point - low], [high - point]], fmt="o", color=color,
                              markersize=6, capsize=4, lw=1.8)
            axis.axvline(0, color="#555555", ls="--", lw=1)
            axis.set_yticks([0, 1], ["工况组宏均值", "电池均值"])
            axis.set_ylim(-0.7, 1.7)
            interval = "200–500" if row_index == 0 else "500–1000"
            axis.set_title(f"{family}，{interval}供应循环\n测试{cells[target]['n_test_cells']}颗，{groups[target]['protocol_groups']}个工况组")
            axis.set_xlabel("MAE改善 / " + r"$10^{-4}$" + "容量比例每供应循环")
    figure.suptitle("加入Q(V)窗口特征的配对MAE改善与95%重采样区间\n正值表示联合模型误差较低，两个统计量使用各自的重采样单位", fontsize=12)
    output = ROOT / "report" / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "MULTIFAMILY_PAIRED_IMPROVEMENT.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
