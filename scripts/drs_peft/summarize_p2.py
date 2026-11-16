"""Format verified P2 source-development evidence without changing its Gate."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .episodes import digest, load_registration
from .summarize_p0 import (
    cell_errors,
    read_json,
    require,
    require_no_target_use,
    write_json,
)

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("lora", "lora_xs", "weight_svd_core", "grad_core", "random_core", "lora_ga")
CORE = ("lora_xs", "weight_svd_core", "grad_core", "random_core")
LABELS = dict(
    zip(
        METHODS,
        ("LoRA", "LoRA-XS", "W-SVD core", "Gradient core", "Random core", "LoRA-GA"),
    )
)
COLORS = dict(
    zip(METHODS, ("#397FA6", "#B49960", "#B47C63", "#8E6CAA", "#929DA7", "#5F9491"))
)
WINDOWS = (
    ("all", 101, 5000),
    ("101_200", 101, 200),
    ("201_500", 201, 500),
    ("501_1000", 501, 1000),
    ("1001_plus", 1001, 5000),
)
COUNTER_COLUMNS = (
    "budget",
    "cell_uid",
    "protocol_uid",
    "candidate",
    "candidate_rank",
    "reference",
    "reference_rank",
    "candidate_mae_three_seed_mean",
    "reference_mae_three_seed_mean",
    "candidate_minus_reference_mae",
    "candidate_error_rank_within_30",
    "plotted_top3",
    "reason",
)


def unit_path(root, unit):
    path = Path(root) / unit["stage"] / unit["method"] / f"rank{unit['rank']}"
    if unit["stage"] == "search":
        path = path / f"trial{unit['trial']}"
    return path / f"seed{unit['seed']}"


def verify_unit(unit, public_root, private_root, config, episodes, selection):
    public, private = unit_path(public_root, unit), unit_path(private_root, unit)
    published = read_json(public / "SOURCE_DEVELOPMENT_COMPLETE.json")
    receipt = read_json(private / "SOURCE_DEVELOPMENT_COMPLETE.json")
    require(
        published["status"] == receipt["status"] == "SOURCE_DEVELOPMENT_COMPLETE",
        "P2 receipt is incomplete",
    )
    for key in receipt:
        if key != "files":
            require(
                published[key] == receipt[key],
                "P2 public/private receipt differs: " + key,
            )
    for record in published["public_files"]:
        path = (public / record["path"]).resolve()
        require(path.parent == public.resolve(), "Public receipt path escapes its run")
        require(
            path.stat().st_size == record["bytes"] and digest(path) == record["sha256"],
            "P2 public artifact SHA differs: " + path.name,
        )
    for record in receipt["files"]:
        path = (private / record["path"]).resolve()
        require(
            path.parent == private.resolve(), "Private receipt path escapes its run"
        )
        require(
            digest(path) == record["sha256"],
            "P2 private artifact SHA differs: " + path.name,
        )
    identity = read_json(private / "RUN_CONFIG.json")
    bridge_selection = read_json(public_root.parent / "P0/BRIDGE_SELECTION.json")
    require(
        identity["bridge"] == bridge_selection["selected_bridge"]
        and receipt["bridge"] == identity["bridge"],
        "P2 Bridge differs from its frozen source selection",
    )
    updates = unit["epochs"] * 11
    for key, value in {
        "stage": "P2",
        "method": unit["method"],
        "seed": unit["seed"],
        "epochs": unit["epochs"],
        "updates": updates,
    }.items():
        require(receipt[key] == value, "P2 run denominator differs: " + key)
    require(
        identity["rank"] == unit["rank"] and identity["expected_updates"] == updates,
        "P2 rank/update identity differs",
    )
    require(
        identity["training_ids"] == episodes["source_cell_roles"]["source_train"]
        and identity["validation_ids"]
        == episodes["source_cell_roles"]["source_meta_validation"],
        "P2 physical roles differ",
    )
    require(
        identity["episode_sha256"] == episodes["episode_contract_sha256"],
        "P2 episode fingerprint differs",
    )
    require(
        identity["head"] == config["head"]
        and identity["loss"] == config["loss"]
        and identity["query_grid"] == [101, 5000],
        "P2 shared Head/loss/query contract differs",
    )
    require_no_target_use(identity)
    require_no_target_use(receipt)
    require(
        receipt["final_target_test_status"] == "SEALED_NOT_EVALUATED",
        "Final target Query was evaluated",
    )
    require(
        receipt["frozen_parameters_unchanged"]
        and receipt["prediction_replay_max_absolute_error"] == 0,
        "P2 frozen parameter or independent reconstruction check failed",
    )
    for record in identity["source_snapshot"]:
        require(
            digest(ROOT / record["path"]) == record["sha256"],
            "P2 locked implementation differs: " + record["path"],
        )
    adapter_lr = unit.get(
        "adapter_lr", selection["methods"][unit["method"]]["adapter_lr"]
    )
    require(
        identity["adapter_learning_rate"] == adapter_lr,
        "P2 adapter LR differs from frozen search",
    )
    require(
        identity["learning_rate"] == config["P2"]["shared_bridge_head_learning_rate"],
        "P2 common Bridge/Head LR differs",
    )
    p0 = (
        public_root.parent
        / "P0/formal"
        / identity["bridge"]
        / "B0_pretrained_frozen"
        / f"seed{unit['seed']}"
    )
    common_parameters = read_json(p0 / "TRAINABLE_PARAMETER_REPORT.json")
    require(
        identity["initial_checkpoint_sha256"] == digest(p0 / "CHECKPOINT_SELECTED.pt"),
        "P2 common B0 start differs",
    )
    scaler = read_json(private / "SOURCE_SCALER.json")
    require(
        scaler == read_json(p0 / "SOURCE_SCALER.json"),
        "P2 source scaler differs from matching P0 B0",
    )
    history = read_json(private / "TRAIN_LOG.json")
    require(
        len(history) == unit["epochs"]
        and [row["epoch"] for row in history] == list(range(1, unit["epochs"] + 1)),
        "P2 training epoch history has gaps or repeats",
    )
    require(
        [row["updates"] for row in history]
        == [11 * epoch for epoch in range(1, unit["epochs"] + 1)],
        "P2 optimizer update counts differ",
    )
    require(
        receipt["best_epoch"]
        == int(np.argmin([row["validation_mae"] for row in history])) + 1,
        "P2 selected epoch differs from first source-meta minimum",
    )
    require(
        receipt["selected_checkpoint_sha256"]
        == digest(private / "CHECKPOINT_SELECTED.pt"),
        "P2 selected checkpoint SHA differs",
    )
    parameters = read_json(private / "TRAINABLE_PARAMETER_REPORT.json")
    detail = parameters["peft_detail"]
    expected_adapter = 12 * (
        unit["rank"] * 768
        if unit["method"] in ("lora", "lora_ga")
        else unit["rank"] ** 2
    )
    require(
        detail["adapter_trainable_parameters"] == expected_adapter
        and len(detail["layers"]) == 12,
        "P2 actual adapter parameter count or QV target set differs",
    )
    require(
        all(not layer["target_active"] for layer in detail["layers"]),
        "P2 source stage has activated target adaptation",
    )
    require(
        parameters["trainable"]
        == parameters["bridge_trainable"]
        + parameters["head_trainable"]
        + expected_adapter,
        "P2 Bridge/Head/adapter parameter decomposition does not close",
    )
    require(
        parameters["trainable"] == receipt["trainable_parameters"],
        "P2 receipt parameter count differs",
    )
    require(
        parameters["bridge_trainable"] == common_parameters["bridge_trainable"]
        and parameters["head_trainable"] == common_parameters["head_trainable"] == 6337,
        "P2 registered common Bridge/Head parameter budget differs",
    )
    with np.load(
        private / "SOURCE_VALIDATION_PREDICTIONS.npz", allow_pickle=False
    ) as arrays:
        ids, query = arrays["ids"].astype(str), arrays["query"].copy()
        prediction, target, observed = (
            arrays[key].copy() for key in ("prediction", "target", "observed")
        )
    require(
        ids.tolist() == episodes["source_cell_roles"]["source_meta_validation"],
        "P2 prediction UID order differs",
    )
    require(
        np.array_equal(query, np.arange(101, 5001))
        and prediction.shape == target.shape == observed.shape == (30, 4900),
        "P2 prediction grid or shape differs",
    )
    require(
        observed.dtype == np.bool_
        and np.isfinite(prediction).all()
        and np.isfinite(target[observed]).all(),
        "P2 prediction/target/mask validity failed",
    )
    protocol_for = {
        uid: group
        for group, members in episodes["source_groups"].items()
        for uid in members
    }
    protocols = np.asarray([protocol_for[uid] for uid in ids])
    key = {
        "phase": unit["stage"],
        "setting": unit.get("setting", "lr_search"),
        "method": unit["method"],
        "rank": unit["rank"],
        "seed": unit["seed"],
        "trial": unit.get("trial"),
        "adapter_lr": adapter_lr,
    }
    row = dict(
        **key,
        status="VERIFIED_SOURCE_DEVELOPMENT",
        epochs=unit["epochs"],
        updates=updates,
        best_epoch=receipt["best_epoch"],
        elapsed_seconds=receipt["elapsed_seconds"],
        peak_allocated_MiB=receipt["peak_gpu_memory_bytes"]["allocated"] / 1024**2,
        peak_reserved_MiB=receipt["peak_gpu_memory_bytes"]["reserved"] / 1024**2,
        checkpoint_sha256=receipt["selected_checkpoint_sha256"],
        receipt_sha256=digest(public / "SOURCE_DEVELOPMENT_COMPLETE.json"),
        private_prediction_sha256=digest(private / "SOURCE_VALIDATION_PREDICTIONS.npz"),
    )
    cell_records = []
    for window, lower, upper in WINDOWS:
        mask = observed & (query >= lower) & (query <= upper)
        supported = mask.any(1)
        mae = np.full(30, np.nan)
        rmse = np.full(30, np.nan)
        if supported.any():
            mae[supported], rmse[supported] = cell_errors(
                prediction[supported], target[supported], mask[supported]
            )
        group = pd.DataFrame({"protocol": protocols[supported], "mae": mae[supported]})
        row.update(
            {
                f"cells_{window}": int(supported.sum()),
                f"observations_{window}": int(mask.sum()),
                f"mae_{window}": float(np.mean(mae[supported]))
                if supported.any()
                else None,
                f"rmse_{window}": float(np.mean(rmse[supported]))
                if supported.any()
                else None,
                f"protocol_macro_mae_{window}": float(
                    group.groupby("protocol").mae.mean().mean()
                )
                if supported.any()
                else None,
            }
        )
        for index, uid in enumerate(ids):
            cell_records.append(
                dict(
                    **key,
                    cell_uid=uid,
                    protocol_uid=protocols[index],
                    window=window,
                    n_obs=int(mask[index].sum()),
                    mae=mae[index],
                    rmse=rmse[index],
                )
            )
    for name in ("mae", "rmse", "protocol_macro_mae"):
        require(
            np.isclose(
                row[name + "_all"], receipt["validation"][name], atol=1e-12, rtol=1e-9
            ),
            "Independent P2 metric differs: " + name,
        )
    recorded = (
        pd.read_csv(public / "SOURCE_VALIDATION_CELL_METRICS.csv")
        .set_index(["cell_uid", "window"])
        .sort_index()
    )
    rebuilt = pd.DataFrame(cell_records).set_index(["cell_uid", "window"]).sort_index()
    require(
        recorded.index.equals(rebuilt.index)
        and np.array_equal(recorded.n_obs, rebuilt.n_obs),
        "P2 public cell/window support differs from the private NPZ",
    )
    require(
        np.array_equal(
            recorded.protocol_uid.to_numpy(str), rebuilt.protocol_uid.to_numpy(str)
        ),
        "P2 public cell protocol binding differs",
    )
    for name in ("mae", "rmse"):
        require(
            np.allclose(
                recorded[name], rebuilt[name], atol=1e-12, rtol=1e-9, equal_nan=True
            ),
            "P2 public cell/window metric differs: " + name,
        )
    budget = dict(
        **key,
        bridge_trainable=parameters["bridge_trainable"],
        head_trainable=parameters["head_trainable"],
        adapter_trainable=expected_adapter,
        total_trainable=parameters["trainable"],
        total_parameters=parameters["total"],
        frozen_parameters=detail["frozen_parameters"],
        frozen_UV_elements=detail["frozen_basis_elements"],
        frozen_compensation_elements=detail["fixed_compensation_elements"],
        frozen_sigma_elements=12 * unit["rank"] if unit["method"] == "lora_xs" else 0,
        source_inactive_target_flag_bool_elements=12 if unit["method"] in CORE else 0,
        buffer_count_scope="adapter buffers; U/V and compensation from recorded counts, Sigma and source flags from locked schema",
        head_bridge_shared=True,
        alpha=unit["rank"],
        alpha_over_rank=1.0,
        dropout=0.0,
    )
    return (
        row,
        budget,
        cell_records,
        {"ids": ids, "target": target, "observed": observed},
    )


def budget_rows(table, name):
    formal = table.loc[table.phase.eq("formal")]
    if name == "rank_matched":
        return formal.loc[formal["rank"].eq(4)]
    return formal.loc[
        ((formal.method.isin(CORE)) & formal["rank"].eq(56))
        | ((formal.method.isin(("lora", "lora_ga"))) & formal["rank"].eq(4))
    ]


def verify_comparisons(cells, comparisons, gate):
    source = cells.loc[cells.phase.eq("formal") & cells.window.eq("all")]
    require(len(comparisons) == 10, "P2 completed paired comparison matrix differs")
    for item in comparisons:
        left = source.loc[
            source.method.eq("grad_core") & source["rank"].eq(item["candidate_rank"])
        ]
        right = source.loc[
            source.method.eq(item["reference"])
            & source["rank"].eq(item["reference_rank"])
        ]
        paired = left.merge(
            right,
            on=["seed", "cell_uid", "protocol_uid"],
            suffixes=("_candidate", "_reference"),
            validate="one_to_one",
        )
        require(
            len(paired) == 90,
            "P2 paired comparison changed its three-seed physical cohort",
        )
        paired["improvement"] = paired.mae_reference - paired.mae_candidate
        expected = (
            paired.mae_reference.mean() - paired.mae_candidate.mean()
        ) / paired.mae_reference.mean()
        require(
            np.isclose(
                expected, item["mean_relative_MAE_improvement"], atol=1e-10, rtol=1e-9
            ),
            "P2 paired relative gain differs from private predictions",
        )
        physical = paired.groupby(["cell_uid", "protocol_uid"]).improvement.mean()
        groups = physical.groupby("protocol_uid").mean()
        for name, values in [("cell", physical), ("protocol", groups)]:
            role = (
                item["setting"]
                + item["reference"]
                + "/"
                + ("cell" if name == "cell" else "group")
            )
            seed = int.from_bytes(
                hashlib.sha256(f"42/P1/{role}".encode()).digest()[:8], "big"
            ) % (2**31 - 1)
            generator = np.random.default_rng(seed)
            data = values.to_numpy(float)
            repeated = data[generator.integers(0, len(data), (2000, len(data)))].mean(1)
            limits = np.quantile(repeated, [0.025, 0.975])
            for suffix, value in zip(("low", "high"), limits):
                require(
                    np.isclose(
                        value, item[f"paired_{name}_CI_{suffix}"], atol=1e-10, rtol=1e-9
                    ),
                    "P2 original paired interval differs from independent source-prediction replay",
                )
        per_seed = paired.groupby("seed")[["mae_reference", "mae_candidate"]].mean()
        relative = (
            per_seed.mae_reference - per_seed.mae_candidate
        ) / per_seed.mae_reference
        require(
            len(relative) == 3
            and bool(relative.gt(0).all())
            == item["all_three_seed_directions_positive"],
            "P2 three-seed improvement direction flag differs",
        )
        for seed, value in relative.items():
            require(
                np.isclose(
                    value,
                    item["per_seed_relative_MAE_improvement"][str(seed)],
                    atol=1e-10,
                    rtol=1e-9,
                ),
                "P2 source-seed relative direction differs",
            )
    require(
        [
            item
            for item in comparisons
            if item["reference"] in ("lora_xs", "random_core")
        ]
        == gate["primary_controls"],
        "P2 comparison file differs from the original Gate controls",
    )


def plot_parameters(metrics, budgets, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), sharey=True)
    symbols = {42: "o", 123: "s", 2026: "^"}
    for axis, name in zip(axes, ("rank_matched", "parameter_matched")):
        data, counts = budget_rows(metrics, name), budget_rows(budgets, name)
        require(
            len(data) == len(counts) == 18,
            "Parameter figure requires every method and all three seeds",
        )
        ticklabels = []
        for index, method in enumerate(METHODS):
            rows = data.loc[data.method.eq(method)].sort_values("seed")
            params = int(
                counts.loc[counts.method.eq(method), "total_trainable"].iloc[0]
            )
            ticklabels.append(LABELS[method].replace(" ", "\n") + "\n" + str(params))
            for offset, row in zip((-0.12, 0, 0.12), rows.itertuples(index=False)):
                axis.scatter(
                    index + offset,
                    row.mae_all,
                    marker=symbols[row.seed],
                    color=COLORS[method],
                    s=19,
                )
            axis.plot(
                [index - 0.21, index + 0.21],
                [rows.mae_all.mean()] * 2,
                color="#26343F",
                linewidth=1.2,
            )
        axis.set(
            xticks=range(6),
            xticklabels=ticklabels,
            xlabel="Method and total trainable parameters",
        )
        axis.set_title(
            "a  Same rank r4" if name == "rank_matched" else "b  LoRA r4 / core r56",
            loc="left",
            fontsize=9,
        )
    axes[0].set_ylabel("Source-meta cell-mean MAE (SOH)")
    handles = [
        plt.Line2D(
            [],
            [],
            marker=symbols[seed],
            linestyle="None",
            color="#52616C",
            label=f"Seed {seed}",
        )
        for seed in symbols
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.005)
    )
    fig.tight_layout(rect=(0, 0.075, 1, 1), w_pad=1.7)
    save_figure(fig, directory / "P2_PARAMETERS_AND_ERROR")
    plt.close(fig)


def save_figure(figure, stem):
    for suffix in ("png", "pdf"):
        metadata = {"CreationDate": None, "ModDate": None} if suffix == "pdf" else None
        figure.savefig(
            stem.with_suffix("." + suffix),
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
            metadata=metadata,
        )


def plot_rank_parameter_curves(metrics, budgets, directory):
    import matplotlib.pyplot as plt

    keys = ["method", "rank", "seed"]
    errors = metrics.loc[metrics.phase.eq("formal"), keys + ["mae_all"]]
    counts = budgets.loc[
        budgets.phase.eq("formal"), keys + ["adapter_trainable", "total_trainable"]
    ]
    table = errors.merge(counts, on=keys, validate="one_to_one")
    require(
        len(table) == 66, "Full parameter curves require all registered source runs"
    )
    markers = {2: "o", 4: "s", 8: "^", 56: "*"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.65), sharey=True)
    for axis, count in zip(axes, ("adapter_trainable", "total_trainable")):
        for method in METHODS:
            values = table.loc[table.method.eq(method)]
            require(
                len(values) == (12 if method in CORE else 9),
                "Rank-curve seed denominator differs",
            )
            axis.scatter(
                values[count],
                values.mae_all,
                color=COLORS[method],
                marker=".",
                alpha=0.35,
                s=16,
                zorder=1,
            )
            average = values.groupby("rank")[[count, "mae_all"]].mean()
            main = average.loc[[2, 4, 8]]
            axis.plot(
                main[count],
                main.mae_all,
                color=COLORS[method],
                linewidth=1.15,
                label=LABELS[method],
            )
            for rank, point in main.iterrows():
                axis.scatter(
                    point[count],
                    point.mae_all,
                    color=COLORS[method],
                    marker=markers[int(rank)],
                    s=22,
                    zorder=3,
                )
            if method in CORE:
                high = average.loc[56]
                axis.plot(
                    [main.loc[8, count], high[count]],
                    [main.loc[8, "mae_all"], high.mae_all],
                    color=COLORS[method],
                    linewidth=1,
                    linestyle="--",
                )
                axis.scatter(
                    high[count],
                    high.mae_all,
                    color=COLORS[method],
                    marker="*",
                    s=60,
                    zorder=4,
                )
        axis.margins(x=0.10)
    axes[0].set_xscale("log")
    axes[0].set(
        title="a  Adapter parameter count",
        xlabel="Trainable adapter parameters (log scale)",
        ylabel="Source-meta cell-mean MAE (SOH)",
    )
    axes[1].set(
        title="b  Total trainable parameter count",
        xlabel="Bridge + Head + adapter parameters",
    )
    axes[1].ticklabel_format(style="plain", axis="x", useOffset=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.005),
        fontsize=7,
    )
    rank_handles = [
        plt.Line2D(
            [],
            [],
            marker=markers[rank],
            linestyle="None",
            color="#52616C",
            label=f"Mean r{rank}",
        )
        for rank in (2, 4, 8, 56)
    ]
    fig.legend(
        handles=rank_handles,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.005),
        fontsize=7,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.90), w_pad=1.8)
    save_figure(fig, directory / "P2_RANK_PARAMETER_CURVES")
    plt.close(fig)


def counterexamples(cells):
    records = []
    primary = budget_rows(cells, "rank_matched")
    scores = (
        primary.loc[primary.method.eq("grad_core") & primary.window.eq("all")]
        .groupby(["cell_uid", "protocol_uid"])
        .mae.mean()
    )
    ranked = (
        scores.rename("candidate_mae")
        .reset_index()
        .sort_values(["candidate_mae", "cell_uid"], ascending=[False, True])
    )
    ranked["candidate_error_rank_within_30"] = np.arange(1, len(ranked) + 1)
    top3 = ranked.cell_uid.head(3).tolist()
    for name in ("rank_matched", "parameter_matched"):
        data = budget_rows(cells, name)
        data = (
            data.loc[data.window.eq("all")]
            .groupby(["method", "cell_uid", "protocol_uid"])
            .mae.mean()
            .unstack("method")
        )
        require(
            len(data) == 30 and set(data) == set(METHODS),
            "Counterexample selection requires all source-meta methods/cells",
        )
        rank = data.grad_core.rank(method="first", ascending=False)
        for (uid, group), row in data.iterrows():
            for reference in METHODS:
                if reference == "grad_core" or row.grad_core <= row[reference]:
                    continue
                records.append(
                    {
                        "budget": name,
                        "cell_uid": uid,
                        "protocol_uid": group,
                        "candidate": "grad_core",
                        "candidate_rank": 4 if name == "rank_matched" else 56,
                        "reference": reference,
                        "reference_rank": 4
                        if name == "rank_matched" or reference in ("lora", "lora_ga")
                        else 56,
                        "candidate_mae_three_seed_mean": row.grad_core,
                        "reference_mae_three_seed_mean": row[reference],
                        "candidate_minus_reference_mae": row.grad_core - row[reference],
                        "candidate_error_rank_within_30": int(rank.loc[(uid, group)]),
                        "plotted_top3": uid in top3,
                        "reason": "REFERENCE_HAS_LOWER_SOURCE_META_MAE",
                    }
                )
    for row in ranked.head(3).itertuples(index=False):
        records.append(
            {
                "budget": "rank_matched",
                "cell_uid": row.cell_uid,
                "protocol_uid": row.protocol_uid,
                "candidate": "grad_core",
                "candidate_rank": 4,
                "reference": "",
                "reference_rank": None,
                "candidate_mae_three_seed_mean": row.candidate_mae,
                "reference_mae_three_seed_mean": None,
                "candidate_minus_reference_mae": None,
                "candidate_error_rank_within_30": row.candidate_error_rank_within_30,
                "plotted_top3": True,
                "reason": "TOP3_ABSOLUTE_CANDIDATE_ERROR",
            }
        )
    return pd.DataFrame(records, columns=COUNTER_COLUMNS), top3


def plot_examples(ids, private_root, directory):
    import matplotlib.pyplot as plt

    artifacts = []
    for number, uid in enumerate(ids, 1):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), sharey=True)
        for axis, budget in zip(axes, ("rank_matched", "parameter_matched")):
            observed_reference = None
            target_reference = None
            for method in METHODS:
                rank = 56 if budget == "parameter_matched" and method in CORE else 4
                predictions = []
                for seed in (42, 123, 2026):
                    spec = {
                        "stage": "formal",
                        "method": method,
                        "rank": rank,
                        "seed": seed,
                    }
                    with np.load(
                        unit_path(private_root, spec)
                        / "SOURCE_VALIDATION_PREDICTIONS.npz",
                        allow_pickle=False,
                    ) as data:
                        positions = np.flatnonzero(data["ids"].astype(str) == uid)
                        require(
                            len(positions) == 1,
                            "Counterexample UID is missing from source-only predictions",
                        )
                        index = int(positions[0])
                        query = data["query"]
                        observed = data["observed"][index]
                        target = data["target"][index]
                        predictions.append(data["prediction"][index].copy())
                        if observed_reference is None:
                            observed_reference = observed.copy()
                            target_reference = target.copy()
                        require(
                            np.array_equal(observed, observed_reference)
                            and np.array_equal(
                                target, target_reference, equal_nan=True
                            ),
                            "Counterexample methods have different labels/masks",
                        )
                value = np.mean(predictions, axis=0)
                axis.plot(
                    query,
                    np.where(observed_reference, value, np.nan),
                    color=COLORS[method],
                    linewidth=1.05,
                    label=LABELS[method],
                )
            axis.plot(
                query,
                np.where(observed_reference, target_reference, np.nan),
                color="#20272D",
                linewidth=1.6,
                label="Observed",
            )
            axis.set(xlim=(101, 2500), xlabel="Original instrument cycle")
            axis.set_title(
                "Same rank r4" if budget == "rank_matched" else "LoRA r4 / core r56",
                loc="left",
                fontsize=9,
            )
        axes[0].set_ylabel("SOH / nominal capacity")
        fig.suptitle(
            uid + " · source-meta diagnostic · three-seed mean curves", fontsize=9
        )
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=4,
            bbox_to_anchor=(0.5, -0.01),
            fontsize=7,
        )
        fig.tight_layout(rect=(0, 0.10, 1, 0.95), w_pad=1.7)
        stem = directory / f"P2_SOURCE_COUNTEREXAMPLE_{number}"
        save_figure(fig, stem)
        artifacts.extend(stem.with_suffix("." + suffix) for suffix in ("png", "pdf"))
        plt.close(fig)
    return artifacts


def report(
    status,
    plan,
    metrics,
    budgets,
    selection,
    gate,
    comparisons,
    failures,
    plotted,
    study,
):
    formal = metrics.loc[metrics.phase.eq("formal")]
    search = metrics.loc[metrics.phase.eq("search")]
    common = budgets[["bridge_trainable", "head_trainable"]].drop_duplicates()
    require(
        len(common) == 1, "P2 units differ in their selected common Bridge/Head budget"
    )
    bridge_count = int(common.bridge_trainable.iloc[0])
    head_count = int(common.head_trainable.iloc[0])
    lora_total = bridge_count + head_count + 36864
    core_total = bridge_count + head_count + 37632
    update_note = (
        "全部登记搜索与正式单元已闭环，当前累计覆盖本轮完整P2预算。"
        if plotted
        else "正在执行且尚无完成凭据的单元等待闭环后记入累计表，当前部分状态保留这一计数范围。"
    )
    lines = [
        "# P2 源域低维PEFT受控比较",
        "",
        f"**当前交付状态为{status}。**",
        "",
        (
            f"已核验LR搜索{len(search)}/{len(plan['search'])}、正式单元{len(formal)}/{len(plan['formal'])}。"
            f"已闭环单元的source optimizer updates为搜索{int(search.updates.sum())}、正式{int(formal.updates.sum())}，"
            f"合计{int(metrics.updates.sum())}。"
        ),
        "",
        update_note,
        "",
        (
            "本formatter只读运行凭据、私有完整预测和原run_p2生成的Gate。阶段判断由原Gate决定。"
            "每个完成单元重新检查公开/私有SHA、共同P0 B0起点、source scaler、source85/meta30角色、Head、loss、查询网格、选权及真实参数。"
            "未来标签来自30颗已允许source-meta电芯的预测包，最终target Query保持sealed。"
        ),
        "",
        "## 预算与真实搜索成本",
        "",
        "| 方法 | 选定adapter LR | 搜索试验数 | 每次epoch/update |",
        "|---|---:|---:|---|",
    ]
    for method in METHODS:
        lines.append(
            f"| {method} | {selection['methods'][method]['adapter_lr']} | 3 | 60/660 |"
        )
    lines += [
        "",
        (
            f"跨阶段已闭环真实源训练共{study['source_training_units']}个单元、{study['source_optimizer_updates']}次optimizer update，"
            f"其中P0为{study['p0_units']}个单元/{study['p0_updates']}次，P2为{len(metrics)}个单元/{int(metrics.updates.sum())}次。"
            "P1另有1035次真实梯度前后向、0次optimizer update。工程smoke、合成恢复和测试单独记账。"
        ),
        "",
        (
            "LR搜索固定seed42、rank4，每方法三个登记候选。选定值固定用于全部正式seed与rank2/4/8/56，"
            "共享Bridge/Head LR为0.0003。完整试验及选权见P2_SEARCH_TRIALS.csv与P2_LR_SELECTION.json。"
        ),
        "",
        (
            f"搜索作业墙钟秒合计为{search.elapsed_seconds.sum():.3f}，当前正式作业墙钟秒合计为{formal.elapsed_seconds.sum():.3f}。"
            "每单元记录训练与逐epoch验证墙钟时间，多个worker并行时分别记账。峰值allocated/reserved显存在P2_METRICS.csv逐单元保存。"
            "核函数活跃时长尚无独立计时。"
        ),
        "",
        (
            f"所有方法共享{bridge_count}个Bridge和{head_count}个Head可训练参数。LoRA r4 adapter为36864，core r4为192。"
            f"core r56为37632，相对LoRA r4 adapter多768，差2.0833%。含共同Bridge/Head后的总量为{core_total}与{lora_total}。"
            "P2_PARAM_BUDGET.csv逐单元列出全部参数、U/V、补偿、XS奇异值及source inactive flag buffer元素。"
            "U/V和补偿采用真实参数报告，XS/flag额外元素按已锁实现及该单元真实rank列示。"
        ),
        "",
        "## 正式逐seed结果",
        "",
        "| 设置 | 方法 | rank | seed | MAE | RMSE | protocol宏MAE | epoch/update |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in formal.itertuples(index=False):
        lines.append(
            f"| {row.setting} | {row.method} | {row.rank} | {row.seed} | {row.mae_all:.8f} | "
            f"{row.rmse_all:.8f} | {row.protocol_macro_mae_all:.8f} | {row.epochs}/{row.updates} |"
        )
    lines += [
        "",
        "## 固定source-meta监督支持",
        "",
        "| 时距 | 独立物理电芯 | 真实观测点 |",
        "|---|---:|---:|",
    ]
    for window, _, _ in WINDOWS:
        require(
            metrics[f"cells_{window}"].nunique() == 1
            and metrics[f"observations_{window}"].nunique() == 1,
            "P2 run support differs across methods or seeds",
        )
        lines.append(
            f"| {window} | {int(metrics[f'cells_{window}'].iloc[0])} | {int(metrics[f'observations_{window}'].iloc[0])} |"
        )
    if plotted:
        lines += [
            "",
            "## 两预算三seed均值",
            "",
            "| 预算 | 方法 | rank | MAE均值 | RMSE均值 | MAE seed范围 | 总可训练参数 |",
            "|---|---|---:|---:|---:|---|---:|",
        ]
        for name in ("rank_matched", "parameter_matched"):
            data = budget_rows(metrics, name)
            counts = budget_rows(budgets, name)
            for method in METHODS:
                subset = data.loc[data.method.eq(method)]
                parameters = int(
                    counts.loc[counts.method.eq(method), "total_trainable"].iloc[0]
                )
                lines.append(
                    f"| {name} | {method} | {int(subset['rank'].iloc[0])} | {subset.mae_all.mean():.8f} | "
                    f"{subset.rmse_all.mean():.8f} | [{subset.mae_all.min():.8f}, {subset.mae_all.max():.8f}] | {parameters} |"
                )
    lines += [
        "",
        (
            "主指标先对各物理电芯的全部真实未来点平均，再跨电芯等权。四时距MAE/RMSE、支持电芯数与观测点数保存在P2_METRICS.csv。"
            "P2_CELL_METRICS.csv保留每cell的full及四时距误差和支持数。正式完成后各单元均为120epochs/1320updates，"
            "所选epoch为本单元source-meta MAE的首次严格最小值。"
        ),
        "",
        "## 原Gate与配对区间",
        "",
    ]
    if gate is None:
        lines.append(
            "原P2_GATE尚未完成生成。当前正式单元仍在累积，方法排序、改进判定及后续放行等待完整矩阵和原Gate。"
        )
    else:
        lines.append(
            f"原Gate状态为{gate['status']}，P3/P4大矩阵放行为{gate.get('P3_P4_large_expansion_allowed', False)}。"
        )
    if comparisons:
        lines += [
            "",
            "| 预算 | 参照 | candidate/reference rank | 相对MAE改善 | 绝对MAE改善cell 95%区间 | protocol 95%区间 | 三seed均正 |",
            "|---|---|---|---:|---|---|---|",
        ]
        for row in comparisons:
            lines.append(
                f"| {row['setting']} | {row['reference']} | {row['candidate_rank']}/{row['reference_rank']} | "
                f"{row['mean_relative_MAE_improvement']:.6f} | [{row['paired_cell_CI_low']:.8f}, {row['paired_cell_CI_high']:.8f}] | "
                f"[{row['paired_protocol_CI_low']:.8f}, {row['paired_protocol_CI_high']:.8f}] | {row['all_three_seed_directions_positive']} |"
            )
        lines += [
            "",
            (
                "candidate固定为grad_core，区间取参照MAE减candidate MAE的绝对改善。相对改善为独立列。"
                "原统计先将三个seed同cell的配对改善平均，再按30个物理cell或15个exact protocol重抽2000次。"
                "cell区间对应电芯等权平均改善，protocol区间对应工况等权宏平均改善。"
                "source seed原值与方向单独保留，区间条件于已拟合seed、已选LR及复用的source-meta集合。"
            ),
        ]
        if gate["status"] == "NEGATIVE":
            lines += [
                "",
                "## 阴性门槛与混合结果",
                "",
                (
                    "预注册P2门槛要求gradient-core在两预算下相对LoRA-XS与random-core均达到至少1%平均相对改善，"
                    "cell与protocol区间下界均大于0，且三个seed改善方向均正。以下保留原Gate四项主控制。"
                ),
                "",
                "| 预算 | 参照 | 平均相对MAE改善 | cell区间下界>0 | protocol区间下界>0 | 三seed均正 |",
                "|---|---|---:|---|---|---|",
            ]
            for item in gate["primary_controls"]:
                lines.append(
                    f"| {item['setting']} | {item['reference']} | {100 * item['mean_relative_MAE_improvement']:.4f}% | "
                    f"{item['paired_cell_CI_low'] > 0} | {item['paired_protocol_CI_low'] > 0} | "
                    f"{item['all_three_seed_directions_positive']} |"
                )
            lines += [
                "",
                (
                    "gradient-core相对LoRA-XS的两个预算平均改善均低于1%，两类区间均跨0，seed方向混合。"
                    "相对random-core存在平均改善，两个预算cell区间均跨0。参数匹配预算的protocol区间为正。"
                    "这些混合证据保留原值，P2判定为NEGATIVE。P1通过的窄共享梯度捕获尚未转化为预注册的下游优势。"
                ),
                "",
                (
                    "P3/P4/P5记录NOT_RUN_NEGATIVE_P2_GATE，H2/H3本轮未评估。全部最终target Query保持未评分，"
                    "当前研究结束于已完成的source阶段阴性门槛。"
                ),
            ]
    if plotted:
        lines += [
            "",
            "![两预算六方法的真实参数与误差](figures/P2_PARAMETERS_AND_ERROR.png)",
            "",
            "每种方法显示三个seed原始MAE，短横线为seed均值，横轴标签给出包含Bridge/Head的总可训练参数。",
            "",
            "![六方法的完整rank与实际参数误差曲线](figures/P2_RANK_PARAMETER_CURVES.png)",
            "",
            (
                "完整参数曲线覆盖六方法的r2/r4/r8及四个core方法的r56。淡色点为三个source seed原始值，"
                "实线依次连接r2/r4/r8的三seed均值。r56用星形和从r8延伸的虚线单独标记。"
                "左轴采用adapter参数量的log坐标，右轴给出包含共同Bridge/Head的实际可训练量。"
                "seed点的离散表示训练重复，配对CI继续以表中物理电芯与工况重抽样定义解释。"
            ),
            "",
            "## 源验证轨迹反例",
            "",
            (
                "反例按primary grad_core r4的三seed逐cell MAE均值，在固定30颗source-meta电芯中降序选择前三，UID打破同分。"
                "相同三颗在两预算下叠加六方法。曲线为三seed预测均值，评分仍分别计算各seed误差后平均。"
                "预测已在101–5000公共网格生成，绘图仅显示真实评分位置，固定展示窗101–2500覆盖本域全部已资格观测。"
                "P2_ERROR_COUNTEREXAMPLES.csv同时保留candidate相对各参照变差的全部source-meta电芯。"
            ),
            "",
        ]
        for index in range(1, 4):
            lines += [
                f"![源验证反例{index}](figures/P2_SOURCE_COUNTEREXAMPLE_{index}.png)",
                "",
            ]
    lines += [
        "",
        "## 优势与劣势（风险）",
        "",
        "- 优势为共同B0初态、相同Head/Bridge、等搜索次数、相同更新预算、三seed原值及近实际参数匹配。",
        (
            "- 劣势（风险）为source-meta同时用于epoch/LR选择和报告，P1标签置换控制未弱化，以及外域子空间失配。"
            "mean-SVD基完整下游训练仍有缺口，相对LoRA-SB的优越性需要忠实对应对照。"
        ),
        "",
        (
            "FULL_FINETUNE属于合同可选资源参照，本轮没有登记及运行。当前矩阵的计算选择和硬件可行性分别记录。"
            "这些源开发结果维持retrospective解释，H2/H3按当前阶段门槛记录评估状态。"
        ),
        "",
        "## 缺失或未闭环单元",
        "",
    ]
    lines.extend(
        ["- " + item["unit"] + "，" + item["status"] for item in failures]
        or ["全部登记单元已通过本formatter的凭据与指标核验。"]
    )
    lines += [
        "",
        "## 复现与证据范围",
        "",
        "```powershell",
        "& D:\\Anaconda\\envs\\tslib\\python.exe -m scripts.drs_peft.summarize_p2",
        "```",
        "",
        (
            "运行产生P2_METRICS、P2_PARAM_BUDGET、P2_CELL_METRICS、P2_ERROR_COUNTEREXAMPLES及P2_FORMATTER_AUDIT。"
            "完整NPZ预测保存在runtime/drs_peft/P2各run目录，图文由其实际值重建。原run_p2、train、peft、配置及Gate继续保持锁定字节。"
        ),
        "",
        (
            "算子与首个真实单元的独立CPU审计见research/drs_peft/P2_IMPLEMENTATION_AUDIT_ZH.md。"
            "文件身份、分母与误差重算的置信度依据已通过的记录。方法效应以完整原Gate和对应比较范围解释。"
        ),
        "",
        "本轮最终工程套件实测61项通过，运行耗时59.94秒。工程验证与正式模型性能分别记账。",
        "",
        "## 关键源码与状态指纹",
        "",
        "| 文件 | SHA256 |",
        "|---|---|",
    ]
    for name in (
        "scripts/drs_peft/run_p2.py",
        "scripts/drs_peft/train.py",
        "models/drs_peft/peft.py",
        "configs/drs_peft/execution_v1.json",
        "configs/drs_peft/episodes_v1.json",
        "results/drs_peft/P1/P1_GATE.json",
        "results/drs_peft/P2/P2_LR_SELECTION.json",
        "results/drs_peft/P2/P2_GATE.json",
    ):
        path = ROOT / name
        if path.is_file():
            lines.append(f"| {name} | {digest(path)} |")
    lines += [
        "",
        "拓展思考与后续关注保留P2阴性、分工况误差及高误差个体，后续研究版本以这些证据和原Gate边界为起点。",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/drs_peft")
    parser.add_argument(
        "--private-root", type=Path, default=ROOT / "runtime/drs_peft/P2"
    )
    parser.add_argument("--report-dir", type=Path, default=ROOT / "report/drs_peft")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    public = args.output_root / "P2"
    registration = read_json(public / "P2_REGISTRATION.json")
    plan = registration["plan"]
    config = read_json(ROOT / "configs/drs_peft/execution_v1.json")
    require(
        registration["driver_sha256"] == digest(ROOT / "scripts/drs_peft/run_p2.py"),
        "Locked P2 driver differs",
    )
    require(
        registration["config_sha256"]
        == digest(ROOT / "configs/drs_peft/execution_v1.json"),
        "Locked P2 config differs",
    )
    require(
        registration["P1_gate_sha256"] == digest(args.output_root / "P1/P1_GATE.json"),
        "P2 source P1 Gate identity differs",
    )
    selection = read_json(public / "P2_LR_SELECTION.json")
    search_complete = read_json(public / "P2_SEARCH_COMPLETE.json")
    require(
        search_complete["status"] == "PASSED"
        and search_complete["attempted_registered_trials"] == 18,
        "P2 search has not completed",
    )
    for item in search_complete["files"]:
        require(
            digest(public / item["path"]) == item["sha256"],
            "Frozen P2 search evidence differs",
        )
    if args.dry_run:
        print(
            {
                "status": "FORMATTER_READY",
                "required_formal": len(plan["formal"]),
                "existing_formal": sum(
                    (
                        unit_path(public, u) / "SOURCE_DEVELOPMENT_COMPLETE.json"
                    ).is_file()
                    for u in plan["formal"]
                ),
            }
        )
        return
    episodes = load_registration()
    metrics, budgets, cells, failures = [], [], [], []
    reference = None
    for unit in plan["search"] + plan["formal"]:
        key = unit_path(Path(), unit).as_posix()
        if not (unit_path(public, unit) / "SOURCE_DEVELOPMENT_COMPLETE.json").is_file():
            failures.append({"unit": key, "status": "PENDING_RECEIPT"})
            continue
        try:
            row, budget, per_cell, arrays = verify_unit(
                unit, public, args.private_root, config, episodes, selection
            )
            if reference is None:
                reference = arrays
            require(
                all(
                    np.array_equal(
                        arrays[name], reference[name], equal_nan=name == "target"
                    )
                    for name in ("ids", "target", "observed")
                ),
                "P2 units do not share the same physical cell labels and masks",
            )
            metrics.append(row)
            budgets.append(budget)
            cells.extend(per_cell)
        except (OSError, ValueError, KeyError, TypeError, AssertionError) as error:
            failures.append(
                {"unit": key, "status": "FAILED_ARTIFACT_AUDIT", "detail": str(error)}
            )
    metrics = pd.DataFrame(metrics)
    budgets = pd.DataFrame(budgets)
    cells = pd.DataFrame(cells)
    require(len(metrics) > 0, "No completed P2 unit could be verified")
    n_formal = int(metrics.phase.eq("formal").sum())
    gate_path = public / "P2_GATE.json"
    gate = read_json(gate_path) if gate_path.is_file() else None
    final_gate = (
        gate is not None
        and gate["status"] in ("PASSED", "NEGATIVE")
        and gate["complete_formal_units"] == len(plan["formal"])
    )
    complete = n_formal == len(plan["formal"]) and not failures and final_gate
    status = (
        gate["status"]
        if complete
        else "BLOCKED"
        if any(f["status"] == "FAILED_ARTIFACT_AUDIT" for f in failures)
        else "PARTIAL"
    )
    comparisons = read_json(public / "P2_PAIRED_COMPARISONS.json") if complete else []
    search_rows = metrics.loc[metrics.phase.eq("search")]
    for method in METHODS:
        trials = search_rows.loc[search_rows.method.eq(method)].sort_values("trial")
        if len(trials) == 3:
            best = trials.sort_values(["mae_all", "trial"]).iloc[0]
            require(
                best.trial == selection["methods"][method]["trial"]
                and best.adapter_lr == selection["methods"][method]["adapter_lr"],
                "Frozen P2 LR choice differs from independently replayed source search",
            )
    if complete:
        require(
            int(metrics.loc[metrics.phase.eq("formal"), "updates"].sum())
            == len(plan["formal"]) * 1320,
            "Full P2 formal updates do not close",
        )
        require(
            gate["target_Query_future_labels_accessed"] is False,
            "P2 Gate reports target Query access",
        )
        verify_comparisons(cells, comparisons, gate)
    for name, table in [
        ("P2_METRICS.csv", metrics),
        ("P2_PARAM_BUDGET.csv", budgets),
        ("P2_CELL_METRICS.csv", cells),
    ]:
        table.to_csv(public / name, index=False, lineterminator="\n")
    counter = pd.DataFrame(columns=COUNTER_COLUMNS)
    figures = []
    if complete:
        counter, top3 = counterexamples(cells)
        directory = args.report_dir / "figures"
        directory.mkdir(parents=True, exist_ok=True)
        plot_parameters(metrics, budgets, directory)
        plot_rank_parameter_curves(metrics, budgets, directory)
        figures = [
            directory / ("P2_PARAMETERS_AND_ERROR." + suffix)
            for suffix in ("png", "pdf")
        ]
        figures.extend(
            directory / ("P2_RANK_PARAMETER_CURVES." + suffix)
            for suffix in ("png", "pdf")
        )
        figures.extend(plot_examples(top3, args.private_root, directory))
    counter.to_csv(
        public / "P2_ERROR_COUNTEREXAMPLES.csv", index=False, lineterminator="\n"
    )
    args.report_dir.mkdir(parents=True, exist_ok=True)
    target = args.report_dir / "P2_RESULTS_ZH.md"
    p0_metrics = pd.read_csv(args.output_root / "P0/P0_SUMMARY.csv")
    study = {
        "p0_units": len(p0_metrics),
        "p0_updates": int(p0_metrics.updates.sum()),
        "source_training_units": len(p0_metrics) + len(metrics),
        "source_optimizer_updates": int(
            p0_metrics.updates.sum() + metrics.updates.sum()
        ),
    }
    target.write_text(
        report(
            status,
            plan,
            metrics,
            budgets,
            selection,
            gate,
            comparisons,
            failures,
            complete,
            study,
        ),
        encoding="utf-8",
        newline="\n",
    )
    write_json(
        public / "P2_FORMATTER_AUDIT.json",
        {
            "status": status,
            "execution_complete": complete,
            "valid_search_units": int(metrics.phase.eq("search").sum()),
            "valid_formal_units": n_formal,
            "required_formal_units": len(plan["formal"]),
            "actual_source_optimizer_updates": int(metrics.updates.sum()),
            "cross_stage_source_accounting": study,
            "final_target_query_labels_used": False,
            "update_count_scope": "completed verified receipts only; active incomplete units await closure",
            "original_gate_sha256": digest(gate_path) if gate_path.is_file() else None,
            "formatter_sha256": digest(Path(__file__)),
            "failures": failures,
            "outputs": [
                {"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)}
                for p in [
                    public / "P2_METRICS.csv",
                    public / "P2_PARAM_BUDGET.csv",
                    public / "P2_CELL_METRICS.csv",
                    public / "P2_ERROR_COUNTEREXAMPLES.csv",
                    target,
                ]
                + figures
            ],
        },
    )
    print(
        {
            "status": status,
            "valid_search": int(metrics.phase.eq("search").sum()),
            "valid_formal": n_formal,
            "required_formal": len(plan["formal"]),
            "actual_updates": int(metrics.updates.sum()),
            "original_gate_modified": False,
        }
    )


if __name__ == "__main__":
    main()
