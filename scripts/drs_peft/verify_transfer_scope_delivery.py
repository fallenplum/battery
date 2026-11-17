"""Independently reconstruct V2 scientific comparisons from scored public facts."""

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "results/drs_peft/transfer_scope_v2"
TOTAL = ROOT / "results/drs_peft/specialization_comparison_v1/target_total_K"
REG = ROOT / "configs/drs_peft/specialization_comparison_v1/EPISODES_TOTAL_K.json"
ARMS = ("G-CAL2", "L-CAL2", "G-JOINT18", "L-JOINT18")
NEW_LORA = ("PRETRAINED_LORA_ZERO_RETAINING", "SPECIALIZED_LORA_ZERO_RETAINING")
OLD = ("PRETRAINED_LORA", "SPECIALIZED_LORA", "SPECIALIZED_OUT")
ZERO = ("WARMUP_ZERO", "SPECIALIZED_ZERO")
LEGACY = "HUST_LEGACY_CAL2"
WINDOWS = dict(
    all=(101, 5000),
    **{
        "101_200": (101, 200),
        "201_500": (201, 500),
        "501_1000": (501, 1000),
        "1001_plus": (1001, 5000),
    },
)
SLOT = ["domain", "method", "K", "seed", "draw"]
PAIR_AXIS = ["domain", "K", "seed", "draw", "cell_uid", "window"]
METRIC = ["domain", "method", "K", "window"]
PAIRS = (
    ("G-JOINT18", "G-CAL2"),
    ("L-JOINT18", "L-CAL2"),
    ("L-CAL2", "G-CAL2"),
    ("L-JOINT18", "G-JOINT18"),
)


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def signs(values):
    return dict(
        positive=int((values > 1e-12).sum()),
        negative=int((values < -1e-12).sum()),
        tie=int((np.abs(values) <= 1e-12).sum()),
    )


def register(episodes):
    expected, tasks, shots = set(), {}, {}
    for task in episodes["target_tasks"]:
        domain = task["domain"]
        tasks[domain] = task
        for shot in task["shots"]:
            k = shot["k"]
            for draw in shot["draws"]:
                ids = draw["support_ids"]
                assert len(ids) == len(set(ids)) == k and set(ids).issubset(
                    task["support_pool_ids"]
                )
                assert (
                    set(ids).isdisjoint(task["query_ids"])
                    and draw["query_ids"] == task["query_ids"]
                )
                methods = (
                    ZERO
                    if not k
                    else (
                        *ARMS,
                        *NEW_LORA,
                        *OLD,
                        *((LEGACY,) if domain == "HUST" else ()),
                    )
                )
                shots[domain, k, draw["draw"]] = ids
                expected.update(
                    (domain, m, k, s, draw["draw"])
                    for m, s in itertools.product(methods, episodes["training_seeds"])
                )
        for draw in range(5):
            assert (
                set(shots[domain, 1, draw])
                <= set(shots[domain, 3, draw])
                <= set(shots[domain, 5, draw])
            )
    return expected, tasks, shots


def load_raw(tasks, shots):
    lock = sha(PUBLIC / "TARGET_TOTAL_K_LOCK.json")
    prior = pd.read_csv(TOTAL / "CELL_METRICS.csv", low_memory=False)
    frames = [prior[prior.method.isin((*ZERO, *OLD))].copy()]
    for path in sorted((PUBLIC / "target").glob("*/*/METRICS.csv")):
        receipt = read(path.with_name("SCORED.json"))
        assert receipt["status"] == "SCORED" and receipt["target_lock_sha256"] == lock
        assert receipt["metrics_sha256"] == sha(path)
        assert receipt["target_query_label_uses_for_selection_or_adaptation"] == 0
        record = read(path.with_name("PREDICTIONS_LOCKED.json"))
        assert receipt["prediction_manifest_sha256"] == sha(
            path.with_name("PREDICTIONS_LOCKED.json")
        )
        task = tasks[receipt["domain"]]
        assert record["query_ids"] == task["query_ids"]
        slot_keys = {
            (slot["method"], slot["K"], slot["draw"]) for slot in record["slots"]
        }
        assert len(record["slots"]) == 60 and len(slot_keys) == 60
        for slot in record["slots"]:
            assert set(slot["support_ids"]) == set(
                shots[receipt["domain"], slot["K"], slot["draw"]]
            )
        rows = pd.read_csv(path)
        assert (
            rows.domain.eq(receipt["domain"]).all()
            and rows.seed.eq(receipt["seed"]).all()
        )
        assert (
            set(rows[["method", "K", "draw"]].itertuples(index=False, name=None))
            == slot_keys
        )
        frames.append(rows)
    for path in sorted((PUBLIC / "e0/units").rglob("CELL_METRICS.csv")):
        receipt = read(path.with_name("RECEIPT.json"))
        assert receipt["status"] == "SCORED" and receipt["target_lock_sha256"] == lock
        assert receipt["cell_metrics_sha256"] == sha(path)
        assert receipt["target_Query_labels_used_for_adaptation_or_selection"] == 0
        assert receipt["query_ids"] == tasks[receipt["domain"]]["query_ids"]
        assert (
            receipt["support_ids"]
            == shots[receipt["domain"], receipt["K"], receipt["draw"]]
        )
        rows = pd.read_csv(path)
        assert all(rows[column].eq(receipt[column]).all() for column in SLOT)
        if receipt["K"]:
            frames.append(rows)
    legacy = read(PUBLIC / "HUST_LEGACY_CAL2_REUSE.json")
    assert (
        legacy["status"] == "AUTHENTICATED_LEGACY_ANCHOR" and len(legacy["slots"]) == 45
    )
    for slot in legacy["slots"]:
        rpath = ROOT / slot["receipt_path"]
        path = rpath.with_name("CELL_METRICS.csv")
        receipt = read(rpath)
        assert slot["receipt_sha256"] == sha(rpath)
        assert (
            slot["cell_metrics_sha256"] == sha(path) == receipt["cell_metrics_sha256"]
        )
        assert receipt["query_ids"] == tasks["HUST"]["query_ids"]
        assert receipt["support_ids"] == shots["HUST", slot["K"], slot["draw"]]
        zero_path = (
            TOTAL
            / "units/C_HUST"
            / f"seed{slot['seed']}/SPECIALIZED_ZERO/K0_draw0/RECEIPT.json"
        )
        assert (
            receipt["source_checkpoint_sha256"]
            == read(zero_path)["source_checkpoint_sha256"]
        )
        rows = pd.read_csv(path)
        assert rows.method.eq("CAL2").all()
        rows["method"] = LEGACY
        rows["reference_family"] = "SPECIALIZED"
        frames.append(rows)
    return pd.concat(frames, ignore_index=True)


def main():
    summary = read(PUBLIC / "SUMMARY_MANIFEST.json")
    assert summary["status"] == "COMPLETE" and summary["new_bootstrap_draws"] == 0
    assert summary["scored_slots"] == summary["requested_slots"] == 867
    assert summary["new_method_scored_slots"] == 540 and summary["missing_slots"] == 0
    for record in summary["inputs"] + summary["files"]:
        assert sha(ROOT / record["path"]) == record["sha256"], record["path"]
    ledger = read(PUBLIC / "EXECUTION_LEDGER.json")
    assert ledger["status"] == "COMPLETE"
    assert ledger["new_source_reference_fits"] == ledger["new_optimizer_updates"] == 0
    expected, tasks, shots = register(read(REG))
    fact = load_raw(tasks, shots)
    assert len(expected) == 867
    assert set(fact[SLOT].itertuples(index=False, name=None)) == expected
    assert len(fact) == summary["scored_cell_window_rows"] == 105360
    assert not fact.duplicated(SLOT + ["cell_uid", "window"]).any()
    for key, rows in fact.groupby(SLOT):
        assert set(zip(rows.cell_uid, rows.window)) == set(
            itertools.product(tasks[key[0]]["query_ids"], WINDOWS)
        )
    observed = fact[fact.n_obs > 0].copy()
    assert fact.loc[fact.n_obs.eq(0), "mae"].isna().all()
    windows = fact[["domain", "cell_uid", "window", "n_obs"]].drop_duplicates()
    assert not windows.duplicated(["domain", "cell_uid", "window"]).any()
    denominators = {}
    for (domain, window), rows in windows.groupby(["domain", "window"]):
        denominators.setdefault(domain, {})[window] = dict(
            cycle_bounds=list(WINDOWS[window]),
            observed_points=int(rows.n_obs.sum()),
            physical_cells=int(rows.n_obs.gt(0).sum()),
        )
    split_n = fact[fact.window.ne("all")].groupby(SLOT + ["cell_uid"]).n_obs.sum()
    total = fact[fact.window.eq("all")].set_index(SLOT + ["cell_uid"]).n_obs
    np.testing.assert_array_equal(split_n, total.reindex(split_n.index))
    part = observed[observed.window.ne("all")].assign(sae=lambda x: x.n_obs * x.mae)
    reduced = part.groupby(SLOT + ["cell_uid"])[["sae", "n_obs"]].sum()
    whole = observed[observed.window.eq("all")].set_index(SLOT + ["cell_uid"]).mae
    np.testing.assert_allclose(
        reduced.sae / reduced.n_obs, whole.reindex(reduced.index), atol=2e-8, rtol=1e-6
    )
    baseline = (
        observed[observed.K.eq(0)]
        .set_index(["domain", "seed", "reference_family", "cell_uid", "window"])
        .mae
    )
    own = baseline.reindex(
        pd.MultiIndex.from_frame(
            observed[["domain", "seed", "reference_family", "cell_uid", "window"]]
        )
    )
    np.testing.assert_allclose(observed.zero_mae, own, atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(
        observed.gain_mae, observed.zero_mae - observed.mae, atol=1e-12, rtol=1e-8
    )
    main_table = pd.read_csv(PUBLIC / "METRICS.csv", low_memory=False)
    columns = ["mae", "zero_mae", "gain_mae"]
    for level, details in [("aggregate", []), ("seed_draw", ["seed", "draw"])]:
        scored = main_table[main_table.level.eq(level)].set_index(METRIC + details)
        raw = observed.groupby(METRIC + details)[columns].mean()
        assert scored.index.is_unique and set(raw.index) == set(scored.index)
        np.testing.assert_allclose(
            raw, scored.reindex(raw.index)[columns], atol=1e-12, rtol=1e-8
        )
    reported_pairs = pd.read_csv(PUBLIC / "CAL2_INCREMENTAL.csv", low_memory=False)
    slices, paired_windows, worst = [], [], []
    methods = list(PAIRS)
    methods += list(
        itertools.product(ARMS, ("PRETRAINED_LORA", "SPECIALIZED_LORA", LEGACY))
    )
    for left, right in methods:
        scores = observed[observed.method.eq(left)].set_index(PAIR_AXIS).mae
        control = observed[observed.method.eq(right)].set_index(PAIR_AXIS).mae
        index = scores.index.intersection(control.index)
        dif = (
            (control.reindex(index) - scores.reindex(index))
            .rename("gain")
            .reset_index()
        )
        agg = dif.groupby(["domain", "K", "window"]).gain.mean()
        if (left, right) in PAIRS:
            directly = reported_pairs[
                (reported_pairs.level == "aggregate")
                & (reported_pairs.method == left)
                & (reported_pairs.comparison == right)
            ].set_index(["domain", "K", "window"])
            assert set(agg.index) == set(directly.index)
            np.testing.assert_allclose(
                agg, directly.reindex(agg.index).mae_improvement, rtol=1e-8, atol=1e-12
            )
        for (domain, k, window), rows in dif.groupby(["domain", "K", "window"]):
            if window not in {"all", "1001_plus"}:
                continue
            by_seed = rows.groupby("seed").gain.mean()
            by_draw = rows.groupby(["seed", "draw"]).gain.mean()
            paired_windows.append(
                dict(
                    domain=domain,
                    K=int(k),
                    window=window,
                    method=left,
                    comparison=right,
                    improvement=float(rows.gain.mean()),
                    seed_signs=signs(by_seed),
                    seed_draw_signs=signs(by_draw),
                    physical_cells=int(rows.cell_uid.nunique()),
                )
            )
    for (domain, method, k, window), rows in observed.groupby(METRIC):
        if window not in {"all", "1001_plus"}:
            continue
        by_seed = rows.groupby("seed").gain_mae.mean()
        by_draw = rows.groupby(["seed", "draw"]).gain_mae.mean()
        slices.append(
            dict(
                domain=domain,
                method=method,
                K=int(k),
                window=window,
                mae=float(rows.mae.mean()),
                own_ZERO_mae=float(rows.zero_mae.mean()),
                delta_K=float(rows.gain_mae.mean()),
                seed_gains={str(seed): float(gain) for seed, gain in by_seed.items()},
                seed_signs=signs(by_seed),
                seed_draw_signs=signs(by_draw),
                negative_cell_seed_draw_percent=float(
                    rows.gain_mae.lt(-1e-12).mean() * 100
                ),
                physical_cells=int(rows.cell_uid.nunique()),
            )
        )
    columns = ["cell_uid", "seed", "draw", "K", "n_obs", "mae", "zero_mae", "gain_mae"]
    for (domain, method, window), rows in observed[observed.method.isin(ARMS)].groupby(
        ["domain", "method", "window"]
    ):
        if window in {"all", "1001_plus"}:
            row = rows.nsmallest(1, "gain_mae")[columns].iloc[0]
            if row.gain_mae < -1e-12:
                worst.append(
                    dict(domain=domain, method=method, window=window, **row.to_dict())
                )
    diagnostic_paths = sorted((PUBLIC / "target").glob("*/*/TRANSFER_DIAGNOSTICS.csv"))
    diagnostics = pd.concat(
        [pd.read_csv(path) for path in diagnostic_paths], ignore_index=True
    )
    axis = SLOT + ["cell_uid"]
    full = observed[observed.method.isin(ARMS) & observed.window.eq("all")].set_index(
        axis
    )
    diagnostic_axis = pd.MultiIndex.from_frame(diagnostics[axis])
    assert len(diagnostic_paths) == 6 and diagnostic_axis.is_unique
    assert set(diagnostic_axis) == set(full.index)
    np.testing.assert_allclose(
        diagnostics.query_gain_mae, full.reindex(diagnostic_axis).gain_mae, atol=1e-12
    )
    mechanism = []
    for (domain, method, k), rows in diagnostics.groupby(["domain", "method", "K"]):
        active = (rows.calibration_norm > 0) | (rows.out_norm > 0)
        lower_support = rows.support_mse_after < rows.support_mse_before
        higher_query = rows.query_gain_mae < -1e-12
        mechanism.append(
            dict(
                domain=domain,
                method=method,
                K=int(k),
                seed_draw_Query_rows=len(rows),
                actively_updated_rows=int(active.sum()),
                support_MSE_improved_rows=int(lower_support.sum()),
                support_improved_while_Query_MAE_worsened_rows=int(
                    (lower_support & higher_query).sum()
                ),
                mean_absolute_support_weight=float(rows.mean_weight.mean()),
                mean_effective_neighbors=float(rows.effective_neighbors.mean()),
            )
        )
    eta = read(PUBLIC / "TARGET_TOTAL_K_LOCK.json")["selected_hyperparameters"]
    refused = [
        [method, int(k)]
        for method in eta
        for k, choice in eta[method].items()
        if not choice["eta"]
    ]
    result = dict(
        status="COMPLETE_INDEPENDENT_SCORED_FACT_REVIEW",
        target_slots=len(expected),
        cell_window_rows=len(fact),
        full_and_window_means_reproduced=True,
        own_family_ZERO_delta_verified=True,
        paired_cal2_incremental_reproduced=True,
        strict_total_K_support_and_fixed_query_verified=True,
        window_denominator_reconciliation_verified=True,
        new_bootstrap_draws=0,
        new_model_forwards=0,
        new_original_future_label_array_reads=0,
        data_scope="Historical target Query pools; seed and draw are repeated evaluation axes.",
        provenance=dict(
            summary_manifest_sha256=sha(PUBLIC / "SUMMARY_MANIFEST.json"),
            target_lock_sha256=sha(PUBLIC / "TARGET_TOTAL_K_LOCK.json"),
            independent_replay_script_sha256=sha(Path(__file__)),
            historical_hust_cal2_reuse_sha256=sha(
                PUBLIC / "HUST_LEGACY_CAL2_REUSE.json"
            ),
        ),
        window_denominators=denominators,
        own_family_full_and_tail=slices,
        paired_full_and_tail=paired_windows,
        worst_geometry_cell_seed_draw=worst,
        support_fitting_and_query_transfer=mechanism,
        source_selected_eta_zero_method_K=refused,
    )
    path = PUBLIC / "INDEPENDENT_REVIEW.json"
    pending = path.with_suffix(".pending.json")
    pending.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    pending.replace(path)
    print(
        json.dumps(
            dict(
                status=result["status"],
                slots=len(expected),
                rows=len(fact),
                aggregate_means=len(observed.groupby(METRIC)),
                proof_sha256=sha(path),
            )
        )
    )


if __name__ == "__main__":
    main()
