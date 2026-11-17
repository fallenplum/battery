"""Score locked V2 nonlinear predictions and account for measured execution."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.drs_peft.data import ROOT, file_hash, write_json
from scripts.drs_peft.train import evaluate_arrays, read_json
from scripts.drs_peft.transfer_scope_data import load_future_rows
from scripts.drs_peft.transfer_scope_reference import old_zero

GRID = np.arange(101, 5001, dtype=np.float32)
DIAGNOSTIC_KEYS = ["domain", "seed", "method", "K", "draw", "cell_uid"]
SOURCE_PHASES = ("SCREEN", "NONLINEAR")


def _write_csv(path, frame):
    pending = path.with_suffix(".pending.csv")
    frame.to_csv(pending, index=False)
    pending.replace(path)


def _validate_score_receipt(folder, prediction_marker_sha, lock_sha):
    marker = folder / "SCORED.json"
    if not marker.exists():
        return None
    receipt = read_json(marker)
    if (
        receipt["status"] != "SCORED"
        or receipt["prediction_manifest_sha256"] != prediction_marker_sha
        or receipt["target_lock_sha256"] != lock_sha
        or receipt["metrics_sha256"] != file_hash(folder / "METRICS.csv")
        or receipt["transfer_diagnostics_sha256"] != file_hash(folder / "TRANSFER_DIAGNOSTICS.csv")
        or receipt["scorer_sha256"] != file_hash(Path(__file__))
    ):
        raise RuntimeError(f"An existing geometry score receipt changed {marker}")
    return receipt


def _prediction(slot, query_ids):
    path = ROOT / slot["path"]
    if file_hash(path) != slot["sha256"]:
        raise RuntimeError("A locked geometry prediction changed")
    state = read_json(path.with_name("STATE.json"))
    if (
        state["sha256"] != slot["sha256"]
        or state["state_id"] != slot["adaptation_id"]
        or state["identity"]["query_ids"] != query_ids
        or state["identity"]["support_ids"] != slot["support_ids"]
    ):
        raise RuntimeError("A private geometry state changed its frozen identity")
    with np.load(path, allow_pickle=False) as bundle:
        ids = bundle["ids"].astype(str).tolist()
        grid = bundle["grid"]
        prediction = bundle["prediction"]
        gap = bundle["gap"]
        coeff = bundle["coefficients"]
        weights = bundle["weights"]
    count = len(query_ids) if slot["method"].startswith("L-") else 1
    dimension = 18 if slot["method"].endswith("18") else 2
    if (
        ids != query_ids
        or not np.array_equal(grid, GRID)
        or prediction.shape != (len(query_ids), len(GRID))
        or gap.shape != prediction.shape
        or coeff.shape != (count, dimension)
        or weights.shape != (count, slot["K"])
        or not all(np.isfinite(array).all() for array in (prediction, gap, coeff, weights))
        or (gap < 0).any()
        or (weights < 0).any()
        or (weights > 1).any()
        or len(set(slot["support_ids"])) != slot["K"]
        or set(slot["support_ids"]) & set(query_ids)
        or slot["coefficient_states"] != len(coeff)
        or slot["active_nonzero_coefficient_states"] != int(np.any(coeff != 0, axis=1).sum())
    ):
        raise RuntimeError("A geometry prediction has invalid axes, values or states")
    return state["identity"], prediction, gap, coeff


def _gap_mean(gap, mask):
    if not mask.any():
        return None
    return float(gap[mask].mean())


def _coefficient_state_ids(identity, slot, coeff, query_ids):
    local = slot["method"].startswith("L-")
    active = {}
    for index, uid in enumerate(query_ids):
        vector = coeff[index if local else 0]
        digest = hashlib.sha256(identity["checkpoint_sha256"].encode("ascii"))
        if np.any(vector):
            digest.update(identity["dictionary_sha256"].encode("ascii"))
            canonical = np.zeros(18, dtype="<f8")
            canonical[:len(vector)] = vector
            canonical[canonical == 0] = 0
            digest.update(canonical.tobytes())
        effective = digest.hexdigest()
        identifier = (
            hashlib.sha256(f"{slot['adaptation_id']}|{uid}".encode()).hexdigest()
            if local else slot["adaptation_id"]
        )
        active[uid] = identifier, effective, bool(np.any(vector))
    return active


def _score_slot(prediction, target, observed, task, seed, slot, baseline, identities):
    ids = task["query_ids"]
    protocols = task["protocol_by_id"]
    table, _ = evaluate_arrays(
        prediction, target, observed, GRID, ids, [protocols[uid] for uid in ids], "original_cycle_id"
    )
    axes = ["cell_uid", "window", "n_obs"]
    if not table[axes].equals(baseline[axes]):
        raise RuntimeError("Geometry and same-reference baseline scoring axes differ")
    table["prediction_scope"] = "HISTORICALLY_EXPOSED_FIXED_TARGET_QUERY"
    table["zero_mae"] = baseline.mae
    table["gain_mae"] = table.zero_mae - table.mae
    table["negative_transfer"] = table.gain_mae.lt(0).where(table.n_obs.gt(0))
    table["stage"] = "TRANSFER_SCOPE_V2"
    table["task_id"] = task["task_id"]
    table["domain"] = task["domain"]
    table["fold_id"] = "final"
    table["seed"] = seed
    for field in ("method", "K", "draw", "is_duplicate"):
        table[field] = slot[field]
    table["reference_family"] = "SPECIALIZED"
    table["episode_adaptation_id"] = slot["adaptation_id"]
    for index, name in enumerate(("adaptation_id", "effective_state_id", "active_coefficients")):
        table[name] = table.cell_uid.map({uid: value[index] for uid, value in identities.items()})
    counts = {
        uid: sum(protocols[uid] == protocols[support] for support in slot["support_ids"])
        for uid in ids
    }
    table["same_protocol_support_count"] = table.cell_uid.map(counts)
    return table


def _score_diagnostics(prefix, scores, gaps):
    if prefix.duplicated(DIAGNOSTIC_KEYS).any():
        raise RuntimeError("Prefix-only geometry diagnostics repeated a Query identity")
    full = scores.loc[scores.window.eq("all"), DIAGNOSTIC_KEYS + [
        "mae", "gain_mae", "n_obs", "adaptation_id", "episode_adaptation_id", "effective_state_id",
        "active_coefficients",
    ]].rename(columns={"mae": "query_mae", "gain_mae": "query_gain_mae", "n_obs": "query_observations"})
    observed_gap = pd.DataFrame(gaps)
    combined = prefix.merge(observed_gap, on=DIAGNOSTIC_KEYS, how="outer", validate="one_to_one", indicator=True)
    if not combined["_merge"].eq("both").all():
        raise RuntimeError("Locked prefix diagnostics and scored gaps differ in their physical Query identities")
    if not np.allclose(
        combined.linearization_gap_mae_full_grid, combined.verified_full_grid_gap, rtol=1e-7, atol=1e-10,
    ):
        raise RuntimeError("Prefix linearization error changed before the observed-position diagnostic")
    combined = combined.drop(columns=["_merge", "verified_full_grid_gap"]).merge(
        full, on=DIAGNOSTIC_KEYS, how="outer", validate="one_to_one", indicator=True,
    )
    if not combined["_merge"].eq("both").all():
        raise RuntimeError("Locked prefix diagnostics and scoring facts disagree")
    return combined.drop(columns="_merge")


def score_geometry(args, records):
    """Score all validated shards after the shared geometry and LoRA barrier."""
    lock_sha = file_hash(args.public_root / "TARGET_TOTAL_K_LOCK.json")
    tasks, pending, results = {}, [], []
    for folder, task, seed, record in records:
        key = task["domain"], seed
        tasks[key] = task
        marker_sha = file_hash(folder / "PREDICTIONS_LOCKED.json")
        if (
            record["status"] != "PREDICTIONS_LOCKED"
            or record["lock_sha256"] != lock_sha
            or record["domain"] != task["domain"]
            or record["task_id"] != task["task_id"]
            or record["seed"] != seed
            or record["query_ids"] != task["query_ids"]
            or record["target_query_future_rows_loaded"] != 0
            or record["diagnostic_sha256"] != file_hash(folder / "DIAGNOSTICS_PREFIX_ONLY.csv")
        ):
            raise RuntimeError("Scoring requires the frozen pre-label-access prediction envelopes")
        prior = _validate_score_receipt(folder, marker_sha, lock_sha)
        if prior:
            results.append(prior)
        else:
            pending.append((folder, task, seed, record, marker_sha))
    if len(tasks) != len(records):
        raise RuntimeError("A geometry target domain and seed appeared twice")
    query_values = {}
    for folder, task, seed, record, marker_sha in pending:
        domain, ids = task["domain"], task["query_ids"]
        load_key = (domain, tuple(ids))
        freshly_loaded = load_key not in query_values
        if freshly_loaded:
            query_values[load_key] = load_future_rows(domain, args.data_root, ids)
        target, observed = query_values[load_key]
        zero, zero_receipt = old_zero(task, seed)
        baseline, _ = evaluate_arrays(
            zero, target, observed, GRID, ids, [task["protocol_by_id"][uid] for uid in ids], "original_cycle_id"
        )
        scores, gaps, unique = [], [], {}
        for slot in record["slots"]:
            key = slot["adaptation_id"]
            if key not in unique:
                unique[key] = _prediction(slot, ids)
            identity, prediction, gap, coeff = unique[key]
            if (
                identity["parent_zero_sha256"] != zero_receipt["prediction_sha256"]
                or identity["checkpoint_sha256"] != zero_receipt["source_checkpoint_sha256"]
                or identity["lock_sha256"] != lock_sha
                or identity["domain"] != domain
                or identity["seed"] != seed
                or identity["method"] != slot["method"]
                or identity["K"] != slot["K"]
                or identity["support_ids"] != slot["support_ids"]
            ):
                raise RuntimeError("Geometry prediction and paired baseline changed their identity")
            if not coeff.any() and not np.array_equal(prediction, zero):
                raise RuntimeError("Zero effective coordinates must recover the saved baseline")
            identities = _coefficient_state_ids(identity, slot, coeff, ids)
            scores.append(_score_slot(prediction, target, observed, task, seed, slot, baseline, identities))
            for uid, diff, mask in zip(ids, gap, observed, strict=True):
                gaps.append({
                    "domain": domain, "seed": seed, "method": slot["method"],
                    "K": slot["K"], "draw": slot["draw"], "cell_uid": uid,
                    "verified_full_grid_gap": float(diff.mean()),
                    "linearization_gap_mae_observed": _gap_mean(diff, mask),
                    "linearization_gap_mae_1001_plus": _gap_mean(diff, mask & (GRID >= 1001)),
                })
        metrics = pd.concat(scores, ignore_index=True)
        diagnostics = _score_diagnostics(pd.read_csv(folder / "DIAGNOSTICS_PREFIX_ONLY.csv"), metrics, gaps)
        metrics_path = folder / "METRICS.csv"
        diagnostics_path = folder / "TRANSFER_DIAGNOSTICS.csv"
        _write_csv(metrics_path, metrics)
        _write_csv(diagnostics_path, diagnostics)
        receipt = {
            "status": "SCORED", "domain": domain, "task_id": task["task_id"], "seed": seed,
            "expected_and_scored_slots": len(record["slots"]),
            "physical_query_cells": len(ids),
            "scored_cell_window_rows": len(metrics),
            "transfer_diagnostic_rows": len(diagnostics),
            "unique_episode_state_payloads": len(unique),
            "recorded_coefficient_states": int(metrics.adaptation_id.nunique()),
            "effective_state_fingerprints": int(metrics.effective_state_id.nunique()),
            "target_lock_sha256": lock_sha,
            "prediction_manifest_sha256": marker_sha,
            "metrics_sha256": file_hash(metrics_path),
            "transfer_diagnostics_sha256": file_hash(diagnostics_path),
            "specialized_zero_prediction_sha256": zero_receipt["prediction_sha256"],
            "query_scoring_only_after_locked_predictions": True,
            "query_future_rows_loaded": len(ids) if freshly_loaded else 0,
            "target_query_label_uses_for_selection_or_adaptation": 0,
            "new_model_forwards": 0,
            "new_optimizer_updates": 0,
            "scorer_sha256": file_hash(Path(__file__)),
        }
        write_json(folder / "SCORED.json", receipt)
        results.append(receipt)
        print(f"target scored {domain} seed={seed} slots={len(record['slots'])}", flush=True)
    return results


def _account(path, counters, scope):
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": file_hash(path),
        "scope": scope,
        "counters": counters,
    }


def collect_execution_ledger(args, records):
    """Count measured work once by stage and preserve cache/reuse boundaries."""
    public = args.public_root
    source_records, final_fit, geometry_target = [], [], []
    for receipt_path in sorted((public / "source").glob("f*s*/*_RECEIPT.json")):
        phase = receipt_path.name.removesuffix("_RECEIPT.json")
        if phase not in SOURCE_PHASES:
            continue
        receipt = read_json(receipt_path)
        if receipt["sha256"] != file_hash(receipt_path.with_name(f"{phase}.csv")):
            raise RuntimeError("Source candidate scores differ from their frozen cost receipts")
        source_records.append(_account(
            receipt_path, receipt["costs"] | {"candidate_score_slots": receipt["rows"]}, phase,
        ))
    for path in sorted((public / "fit_cost").glob("s*/COST.json")):
        counters = {name: value for name, value in read_json(path).items() if name != "parent_jacobians"}
        final_fit.append(_account(path, counters, "final_source_scale_fit"))
    lock_sha = file_hash(public / "TARGET_TOTAL_K_LOCK.json")
    for folder, task, seed, record in records:
        path = folder / "PREDICT_COST.json"
        score_receipt = _validate_score_receipt(folder, file_hash(folder / "PREDICTIONS_LOCKED.json"), lock_sha)
        counters = {name: value for name, value in read_json(path).items() if name != "parent_jacobians"}
        if score_receipt is None or counters["request_slots"] != len(record["slots"]):
            raise RuntimeError("Geometry target requires an intact scoring and cost receipt")
        measured = counters | {
            "scored_cell_window_rows": score_receipt["scored_cell_window_rows"],
            "logical_coefficient_state_requests": sum(slot["coefficient_states"] for slot in record["slots"]),
            "active_coefficient_state_requests": sum(slot["active_nonzero_coefficient_states"] for slot in record["slots"]),
            "unique_solution_payloads": score_receipt["unique_episode_state_payloads"],
            "unique_recorded_coefficient_states_within_shard": score_receipt["recorded_coefficient_states"],
            "unique_effective_state_fingerprints_within_shard": score_receipt["effective_state_fingerprints"],
            "sum_episode_seconds": float(sum(slot["seconds"] for slot in record["slots"])),
            "scored_query_future_rows_loaded": score_receipt["query_future_rows_loaded"],
        }
        geometry_target.append(_account(path, measured, f"{task['domain']}/seed{seed}"))
    lora_path = public / "LORA_SOURCE_SELECTION.json"
    lora_source = read_json(lora_path)
    lora_source_rows = []
    for block in lora_source["source_blocks"]:
        path = ROOT / block["path"]
        if file_hash(path) != block["sha256"]:
            raise RuntimeError("LoRA source cost block changed")
        lora_source_rows.append(_account(
            path,
            block["counters"] | {
                "reused_source_candidate_scores": block["old_score_slots_reused"],
                "new_nonlinear_source_prediction_states": block["new_nonzero_prediction_states"],
                "snapshot_cache_hits": block["snapshot_cache_hits"],
            },
            "LoRA_source_shrinkage",
        ))
    lora_target = []
    lora_origins = Counter()
    effective_lora = {}
    for path in sorted((public / "e0/units").rglob("RECEIPT.json")):
        receipt = read_json(path)
        if (
            receipt["status"] != "SCORED"
            or receipt["target_lock_sha256"] != lock_sha
            or receipt["cell_metrics_sha256"] != file_hash(path.with_name("CELL_METRICS.csv"))
            or receipt["target_Query_labels_used_for_adaptation_or_selection"] != 0
        ):
            raise RuntimeError("LoRA target scoring lifecycle remains unfinished or changed")
        lora_target.append(_account(path, receipt["new_execution_counters"], receipt["origin"]))
        lora_origins[receipt["origin"]] += 1
        effective_lora.setdefault(receipt["method"], set()).add(receipt["state_sha256"])
    counts = (len(source_records), len(final_fit), len(geometry_target), len(lora_source_rows), len(lora_target))
    if counts != (18, 3, 6, 18, 192):
        raise RuntimeError(f"Incomplete geometry or LoRA execution ledger coverage {counts}")

    def total(entries):
        counters = Counter()
        for item in entries:
            counters.update(item["counters"])
        return dict(counters)

    stages = {
        "geometry_source_screen": [r for r in source_records if r["scope"] == "SCREEN"],
        "geometry_source_nonlinear": [r for r in source_records if r["scope"] == "NONLINEAR"],
        "geometry_final_source_scale_fit": final_fit,
        "geometry_target_predict_and_score": geometry_target,
        "lora_source_shrinkage": lora_source_rows,
        "lora_target_shrinkage": lora_target,
    }
    totals = {name: total(entries) for name, entries in stages.items()}
    count_fields = ("optimizer_updates", "new_optimizer_updates", "nonlinear_optimizer_updates")
    total_new_fits = sum(counters.get("source_reference_fits", 0) for counters in totals.values())
    total_updates = sum(counters.get(field, 0) for counters in totals.values() for field in count_fields)
    if total_new_fits or total_updates:
        raise RuntimeError("The V2 geometry or LoRA shrinkage ran unregistered optimization")
    ledger = {
        "status": "COMPLETE",
        "target_lock_sha256": lock_sha,
        "new_source_reference_fits": total_new_fits,
        "new_optimizer_updates": total_updates,
        "counts_by_stage": totals,
        "source_stage_receipts": stages["geometry_source_screen"] + stages["geometry_source_nonlinear"],
        "final_source_geometry_receipts": final_fit,
        "target_geometry_receipts": geometry_target,
        "lora_source_receipts": lora_source_rows,
        "lora_target_receipts": lora_target,
        "lora_target_prediction_origins": dict(lora_origins),
        "unique_lora_target_state_payloads_by_method": {key: len(value) for key, value in effective_lora.items()},
        "adaptation_coordinate_dimensions": {
            "G-CAL2": 2, "L-CAL2": 2, "G-JOINT18": 18, "L-JOINT18": 18,
            "LoRA": 9216, "OUT_dictionary": 16,
        },
        "OUT_frozen_dictionary_bytes_per_reference_for_adapter_factors": 1253376,
        "OUT_dictionary_storage_scope": "Registered L/R directional factor storage from the unchanged parent OUT16.",
        "timing_scope": "Geometry episode seconds include solving, prediction, state writes and diagnostics. "
        "Cache reuse changes the measured duration. LoRA per-state walltimes were absent from these receipts.",
        "new_Query_label_array_reads_by_ledger": 0,
    }
    write_json(public / "EXECUTION_LEDGER.json", ledger)
    return ledger
