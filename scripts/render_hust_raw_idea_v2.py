"""Render a descriptive HUST audit from verified source and derived data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
from scripts.hust_safe_pickle import dataframe_columns, parse_pickle_data


VIEWS = ("actual2C_loaded_response", "common5C_charge_response")
COLORS = ("#47758e", "#bd793f", "#817897")


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def checked_data(data_root, qa_path):
    complete = read_json(data_root / "DATASET_COMPLETE.json")
    for item in complete["files"]:
        if digest(data_root / item["path"]) != item["sha256"]:
            raise RuntimeError("Derived HUST data changed after completion")
    qa = read_json(qa_path)
    if qa["status"] != "PASS" or any(row["status"] != "PASS" for row in qa["checks"]):
        raise RuntimeError("The descriptive figure requires successful independent data QA")
    return complete, qa


def native_discharge(columns, voltage_grid):
    status = np.asarray(columns["Status"])
    positions = np.flatnonzero(status == "Constant current discharge_3")
    origin_positions = np.flatnonzero(status == "Constant current discharge_0")
    if not len(positions) or not len(origin_positions) or np.any(np.diff(positions) != 1):
        raise RuntimeError("Native D3 source phase or D0 origin differs")
    voltage = np.asarray(columns["Voltage (V)"], dtype=float)[positions]
    capacity = np.asarray(columns["Capacity (mAh)"], dtype=float)
    absolute = (capacity[origin_positions[0]] - capacity[positions]) / 1000
    inside = (voltage >= voltage_grid[0]) & (voltage <= voltage_grid[-1])
    return voltage[inside], absolute[inside]


def style_axis(axis, xlabel, ylabel, title):
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title, loc="left", fontweight="bold", pad=9)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(length=3, width=.8)
    axis.grid(color="#e3e7e9", linewidth=.5)
    axis.set_axisbelow(True)


def render(args):
    complete, qa = checked_data(args.data, args.qa)
    arrays = {}
    for view in VIEWS:
        with np.load(args.data / "views" / view / "early_inputs.npz", allow_pickle=False) as archive:
            arrays[view] = {name: archive[name].copy() for name in (
                "cell_uid", "input_eligible", "cycle_ids", "voltage_grid_V", "q_absolute", "local_residual",
                "curve_valid_mask", "anchor_observed_cycles", "anchor_query_cycles")}
    primary, secondary = arrays[VIEWS[0]], arrays[VIEWS[1]]
    if not np.array_equal(primary["cell_uid"], secondary["cell_uid"]):
        raise RuntimeError("The two views no longer share the same physical-cell order")
    eligible = np.flatnonzero(primary["input_eligible"])
    if not len(eligible):
        raise RuntimeError("The primary view has no qualified prefix exemplar")
    # The NPZ UID order is the authenticated original archive-member order.
    number = int(eligible[0])
    uid = str(primary["cell_uid"][number])
    metadata = pd.read_parquet(args.data / "cells.parquet", columns=[
        "cell_uid", "source_device_record_id", "source_archive_member", "source_member_sha256"]).set_index("cell_uid").loc[uid]
    source_path = args.source_root / "our_data" / (str(metadata.source_device_record_id) + ".pkl")
    if digest(source_path) != metadata.source_member_sha256:
        raise RuntimeError("The plotted raw source differs from the registered member")
    source, _ = parse_pickle_data(source_path.read_bytes())
    native_data = source[str(metadata.source_device_record_id)]["data"]
    grid = primary["voltage_grid_V"]
    anchors = primary["anchor_observed_cycles"][number]
    anchor_queries = primary["anchor_query_cycles"][number]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8.6), layout="constrained")
    anchor_records = []
    for color, query, cycle in zip(COLORS, anchor_queries, anchors):
        if not np.isfinite(cycle):
            continue
        cycle = int(cycle)
        position = int(np.flatnonzero(primary["cycle_ids"][number] == cycle)[0])
        native_v, native_q = native_discharge(dataframe_columns(native_data[cycle]), grid)
        axes[0, 0].plot(grid, primary["q_absolute"][number, position], color=color, linewidth=1.4,
                        label=f"原始第 {cycle} 圈")
        axes[0, 0].scatter(native_v, native_q, s=13, facecolors="none", edgecolors=color,
                           linewidths=.65, alpha=.75)
        axes[0, 1].plot(grid, primary["local_residual"][number, position] * 1000,
                        color=color, linewidth=1.4, label=f"原始第 {cycle} 圈")
        anchor_records.append({"query_cycle": int(query), "original_observed_cycle": cycle,
                               "native_window_points": len(native_v), "interpolated_grid_points": len(grid)})
    axes[0, 0].legend(loc="lower left", frameon=False, fontsize=9)
    axes[0, 1].axhline(0, color="#7b858a", linewidth=.75, linestyle="--")
    style_axis(axes[0, 0], "电压 V (V)", "绝对放电量 Q (Ah)", f"a  {uid} 的末段放电 Q(V)")
    style_axis(axes[0, 1], "电压 V (V)", "局部残差 r (mAh)", "b  相同原始循环的加载响应残差")
    for axis in axes[0]:
        axis.set_xlim(grid[0], grid[-1])
    support_counts = {}
    for view, color, label in zip(VIEWS, COLORS[:2], ("末段放电 2C", "恒流充电 5C")):
        counts = arrays[view]["curve_valid_mask"].sum(axis=1)
        thresholds = np.arange(101)
        supported_cells = np.asarray([np.sum(counts >= threshold) for threshold in thresholds])
        axes[1, 0].step(thresholds, supported_cells, where="post", color=color, linewidth=1.8, label=label)
        support_counts[view] = {"qualification_cells": len(counts), "valid_prefix_curves": int(counts.sum()),
                                "input_eligible_cells": int(arrays[view]["input_eligible"].sum()),
                                "per_cell_prefix_support": counts.tolist()}
    axes[1, 0].set_xlim(0, 100)
    axes[1, 0].set_ylim(0, 82)
    axes[1, 0].set_yticks([0, 20, 40, 60, 77])
    axes[1, 0].legend(loc="lower left", frameon=False, fontsize=9)
    style_axis(axes[1, 0], "至少具备的有效前缀曲线数", "达到该覆盖的电池数", "c  每视图均保留 77 颗电池")
    future = pd.read_parquet(args.data / "future_soh.parquet")
    valid_future = future.loc[future.capacity_valid]
    for _, frame in valid_future.groupby("cell_uid", sort=False):
        frame = frame.sort_values("original_cycle_id")
        axes[1, 1].plot(frame.original_cycle_id, frame.soh_nominal, color="#7f929c", alpha=.4, linewidth=.65)
    axes[1, 1].axhline(.8, color="#ad6b4d", linestyle="--", linewidth=1.1, label="名义 SOH 0.8")
    axes[1, 1].set_ylim(.78, 1.13)
    axes[1, 1].set_xlim(100, int(valid_future.original_cycle_id.max()))
    axes[1, 1].legend(loc="upper right", frameon=False, fontsize=9)
    style_axis(axes[1, 1], "原始循环号", "源 Capacity / 1.1 Ah", "d  全部 77 颗电池的未来实际观测")
    figure.suptitle("HUST 原始数据与物理特征输入的描述性审计", fontsize=14)
    events = int(complete["summary"]["observed_eol80_events"])
    censored = int(complete["summary"]["right_censored_cells"])
    cells = int(complete["summary"]["source_cells"])
    minimum_soh = float(valid_future.soh_nominal.min())
    figure.supxlabel("a 中空心圆为原生采样点，彩色实线为64点插值。样例仅按归档顺序及前缀资格选取。\n"
                     f"模型拟合0次。实际SOH最低{minimum_soh:.7f}，80%阈值事件{events}/{cells}，右删失{censored}/{cells}。", fontsize=9)
    args.output.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("png", "pdf"):
        path = args.output / f"overview.{suffix}"
        figure.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(figure)
    receipt = {"status": "RENDERED_DESCRIPTIVE_AUDIT", "models_fitted": 0, "optimizer_updates": 0,
               "renderer_source_sha256": digest(Path(__file__)), "dataset_complete_sha256": digest(args.data / "DATASET_COMPLETE.json"),
               "independent_qa_sha256": digest(args.qa), "independent_qa_checks": len(qa["checks"]),
               "selected_cell_uid": uid, "selected_source_member": str(metadata.source_archive_member),
               "selected_source_member_sha256": str(metadata.source_member_sha256),
               "selection_rule": "first primary-view input-eligible UID in authenticated archive order; future outcomes never enter selection",
               "secondary_view_eligible_for_selected_cell": bool(secondary["input_eligible"][number]),
               "anchor_observations": anchor_records, "support": support_counts,
               "future_valid_observations": len(valid_future), "future_cells": valid_future.cell_uid.nunique(),
               "future_soh_minimum": minimum_soh, "observed_eol80_events": events, "right_censored_cells": censored,
               "source_pickle_global_invocations": 0, "capacity_source": "source-provided net Capacity; independent instrument calibration and generation algorithm uncertified",
               "figure_scope": "descriptive native samples, interpolation, loaded-response residual, prefix support and observed future capacity; model performance and mechanism validation are outside this figure",
               "files": [{"path": path.name, "bytes": path.stat().st_size, "sha256": digest(path)} for path in paths]}
    (args.output / "FIGURE_RECEIPT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "selected_cell_uid": uid, "figures": len(paths), "models_fitted": 0}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPOSITORY / "data/hust_raw_idea_v2")
    parser.add_argument("--qa", type=Path, default=REPOSITORY / "results/hust_raw_idea_audit_v2/FINAL_DATASET_QA.json")
    parser.add_argument("--source-root", type=Path, default=REPOSITORY.parent / "private/raw_sources/HUST")
    parser.add_argument("--output", type=Path, default=REPOSITORY / "report/figures/hust_raw_idea_v2")
    args = parser.parse_args()
    font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    font_manager.fontManager.addfont(str(font_path))
    font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({"font.family": font_name, "font.size": 10, "axes.titlesize": 11,
                         "axes.labelsize": 10, "axes.linewidth": .8, "axes.unicode_minus": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    render(args)


if __name__ == "__main__":
    main()
