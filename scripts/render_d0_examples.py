"""Render diagnostic examples from real audited curves and method controls."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    wanted = {"HUST_1-1", "MATR_b1c0", "Stanford_Nova_Regular_100"}
    examples = {}
    for line in (args.private_cache / "d0_cells.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["eligible"]["cell_id"] in wanted:
            examples[row["eligible"]["cell_id"]] = row
    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False,
                         "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    colors = {10: "#1f77b4", 50: "#e08028", 100: "#8253a0"}
    for column, name in enumerate(("HUST_1-1", "MATR_b1c0", "Stanford_Nova_Regular_100")):
        record = examples[name]
        cell = record["eligible"]
        grid = np.linspace(cell["comparison_voltage_min_V"], cell["comparison_voltage_max_V"], 64)
        for example in record["curve_examples"]:
            number = example["supplied_cycle_number"]
            axes[0, column].plot(grid, example["curve"], color=colors[number], label=f"供应编号 {number}")
        axes[0, column].set(title=f"{name}\n同电流窗口内观测 Q(V)", xlabel="电压 / V", ylabel="定向窗口电荷 / Ah")
        axes[0, column].legend(frameon=False)
        trajectory = record["capacity_trajectory"]
        x = np.asarray(trajectory["cycle_number"])
        y = np.asarray(trajectory["soh"])
        axes[1, column].plot(x, y, lw=0.9, color="#355a73")
        axes[1, column].axvline(100, ls="--", lw=0.9, color="#a14b35", label="前缀截止100")
        axes[1, column].axhline(0.8, ls=":", lw=0.9, color="#666666", label="80%参考阈值")
        axes[1, column].set(xlabel="供应循环编号", ylabel="分相容量 / 声明容量",
                            title="当前供应容量轨迹与前缀截止点")
        axes[1, column].legend(frameon=False)
    figure.suptitle("D0 实测曲线示例\nHUST 循环轴已由官方原始数据认证，MATR/Stanford 坐标保留供应编号", fontsize=13)
    args.output.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output / "D0_OBSERVED_CURVE_EXAMPLES.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
