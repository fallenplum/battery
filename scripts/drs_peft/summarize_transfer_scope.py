"""Audit historical anchors and summarize the registered transfer-scope V2 results."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.drs_peft.e123_protocol import WINDOWS
from scripts.drs_peft.summarize_onesided import aggregate, parse_boolean

ROOT = Path(__file__).resolve().parents[2]
EPISODES = ROOT / "configs/drs_peft/specialization_comparison_v1/EPISODES_TOTAL_K.json"
PARENT = ROOT / "results/drs_peft/onesided_fewshot_v1"
TOTAL = ROOT / "results/drs_peft/specialization_comparison_v1/target_total_K"
PUBLIC = ROOT / "results/drs_peft/transfer_scope_v2"
REPORT = ROOT / "report/drs_peft/transfer_scope_v2"
SPLIT = ROOT / "research/drs_peft/DATA_SPLIT_LOCK.json"
MATR_CACHE = ROOT / "runtime/drs_peft/data_v1/MATR"
GEOMETRY = ("G-CAL2", "L-CAL2", "G-JOINT18", "L-JOINT18")
LORA_ZERO = ("PRETRAINED_LORA_ZERO_RETAINING", "SPECIALIZED_LORA_ZERO_RETAINING")
NEW = GEOMETRY + LORA_ZERO
ZERO = ("WARMUP_ZERO", "SPECIALIZED_ZERO")
OLD = ("PRETRAINED_LORA", "SPECIALIZED_LORA", "SPECIALIZED_OUT")
LEGACY_CAL2 = "HUST_LEGACY_CAL2"
FAMILY = {
    method: "WARMUP" if method.startswith(("WARMUP_", "PRETRAINED_")) else "SPECIALIZED"
    for method in (*NEW, *ZERO, *OLD, LEGACY_CAL2)
}
DIMENSION = dict(
    zip(
        (*GEOMETRY, *LORA_ZERO, *ZERO, *OLD, LEGACY_CAL2),
        (2, 2, 18, 18, 9216, 9216, 0, 0, 9216, 9216, 16, 2),
        strict=True,
    )
)
SLOT = ["domain", "seed", "method", "K", "draw"]
PAIR = ["domain", "seed", "K", "draw", "cell_uid", "window"]
QUERY = PAIR + ["method"]
SCORES = ["mae", "rmse", "bias", "zero_mae", "gain_mae"]
WINDOW_LABELS = tuple(label for label, _, _ in WINDOWS)
PAIR_METHODS = (
    ("G-JOINT18", "G-CAL2"),
    ("L-JOINT18", "L-CAL2"),
    ("L-CAL2", "G-CAL2"),
    ("L-JOINT18", "G-JOINT18"),
)


def read_json(path):
    path = Path(path)
    return json.loads(path.read_bytes()) if path.exists() else None


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def artifact_reference(path):
    path = Path(path).resolve()
    return {
        "path": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),
        "sha256": sha256(path),
    }


def build_reuse_manifest(public, evidence, ledger):
    """Gather identity and measured reuse provenance from verified public receipts."""
    public = Path(public)
    lock_path = public / "TARGET_TOTAL_K_LOCK.json"
    lock = read_json(lock_path)
    if ledger["status"] != "COMPLETE" or ledger["target_lock_sha256"] != sha256(lock_path):
        raise ValueError("A complete execution ledger for the current lock is required")
    fits = []
    for path in sorted((public / "fit").glob("*.json")):
        if lock["fit_sha256"].get(path.name) != sha256(path):
            raise ValueError("A frozen source fit fingerprint changed before delivery")
        fit = read_json(path)
        identity = fit["identity"]
        fold, seed = path.stem.removeprefix("f").split("s")
        folder = "final" if fold == "final" else f"fold{fold}"
        parent_dir = PARENT / "references" / folder / f"seed{seed}"
        reference = parent_dir / "static/REFERENCE_COMPLETE.json"
        if not reference.exists():
            reference = parent_dir / "REFERENCE_COMPLETE.json"
        parent = read_json(reference)
        dictionary_path = parent_dir / "geometry/DICTIONARIES_COMPLETE.json"
        dictionary = read_json(dictionary_path)
        registered_dictionary = next(item["sha256"] for item in dictionary["files"] if item["path"] == "DICTIONARIES.pt")
        if (
            parent["reference_checkpoint_sha256"] != identity["checkpoint_sha256"]
            or parent["fit_ids"] != identity["fit_ids"]
            or registered_dictionary != identity["dictionary_sha256"]
        ):
            raise ValueError("The final V2 fit lineage changed relative to the parent reference")
        fits.append({
            "fold": fold, "seed": int(seed), "fit_cell_count": len(identity["fit_ids"]),
            "frozen_specialized_checkpoint_sha256": identity["checkpoint_sha256"],
            "official_pretrained_with_source_interface_checkpoint_sha256": parent["warmup_checkpoint_sha256"],
            "out16_dictionary_sha256": identity["dictionary_sha256"],
            "source_prefix_data_manifest_sha256": identity["data_manifest_sha256"],
            "new_fit_statistics_payload_sha256": fit["array_sha256"],
            "reused_reference_receipt": artifact_reference(reference),
            "reused_dictionary_receipt": artifact_reference(dictionary_path),
            "v2_fit_receipt": artifact_reference(path),
        })
    if len(fits) != 12 or len(lock["fit_sha256"]) != 12:
        raise ValueError("The reuse manifest requires nine fold and three final fit fingerprints")
    cache = []
    cache_groups = ledger["source_stage_receipts"] + ledger["final_source_geometry_receipts"] + ledger["target_geometry_receipts"]
    for item in cache_groups:
        path = ROOT / item["path"]
        if item["sha256"] != sha256(path):
            raise ValueError("A Jacobian execution receipt changed")
        receipt = read_json(path)
        counters = item["counters"]
        hits = counters["parent_jacobian_hits"]
        cached = counters["new_cache_jacobian_hits"]
        built = counters["jacobian_builds"]
        cache.append({
            "scope": item["scope"], "receipt": artifact_reference(path),
            "parent_jacobian_hits": hits, "v2_jacobian_cache_hits": cached,
            "newly_constructed_jacobians_after_cache_misses": built,
            "distinct_parent_payload_checksums": sorted(set(receipt["parent_jacobians"].values())),
        })
    split = read_json(SPLIT)
    data_manifests = {}
    for domain in ("MATR", "HUST"):
        registered = split["data_caches"][domain]
        manifest_path = ROOT / f"runtime/drs_peft/data_v1/{domain}/COMPLETE.json"
        if sha256(manifest_path) != registered["manifest_sha256"]:
            raise ValueError("A data cache manifest changed after its split freeze")
        data_manifests[domain] = {
            "frozen_manifest_sha256": registered["manifest_sha256"],
            "frozen_payload_sha256": {
                value["path"]: value["sha256"]
                for value in read_json(manifest_path)["files"]
            },
        }
    lora_proof = public / "LORA_SOURCE_SELECTION.json"
    lora_preflight = public / "LORA_TARGET_REUSE_PREFLIGHT.json"
    lora_source_replay = public / "E0_SOURCE_LABEL_REPLAY.json"
    legacy_cal2_proof = public / "HUST_LEGACY_CAL2_REUSE.json"
    cal2 = read_json(legacy_cal2_proof)
    if cal2["status"] != "AUTHENTICATED_LEGACY_ANCHOR" or len(cal2["slots"]) != evidence["historical_HUST_CAL2_slots"]:
        raise ValueError("The independent legacy CAL2 authenticity receipt differs")
    preflight = read_json(lora_preflight)
    if preflight and (
        preflight["status"] != "COMPLETE_EXACT_STATE_AND_PREDICTION_REUSE_PREFLIGHT"
        or preflight["target_nonzero_draw_slots_authenticated"] != 180
        or not preflight["unit_scale_factor_snapshots_verified"]
        or preflight["target_future_rows_decoded"] != 0
    ):
        raise ValueError("The LoRA unit-scale fit reuse preflight changed")
    supplemental_evidence = {}
    if lora_source_replay.exists():
        supplemental_evidence["whitelisted_source_replay"] = artifact_reference(lora_source_replay)
    if preflight:
        supplemental_evidence.update(
            parent_unit_scale_fit_payload_preflight=artifact_reference(lora_preflight),
            preflight_target_nonzero_slots=preflight["target_nonzero_draw_slots_authenticated"],
            preflight_unique_old_state_files=preflight["unique_old_state_files"],
        )
    origins = Counter(ledger["lora_target_prediction_origins"])
    return {
        "status": "COMPLETE_V2_REUSE_PROVENANCE",
        "target_lock": artifact_reference(lock_path),
        "inherited_total_K_episode": artifact_reference(EPISODES),
        "historically_exposed_query_cohorts": True,
        "data_split_lock": artifact_reference(SPLIT),
        "data_manifest_identities": data_manifests,
        "frozen_source_reference_and_fit_groups": fits,
        "jacobian_activity_by_execution": cache,
        "jacobian_totals": {
            name: sum(entry[name] for entry in cache)
            for name in (
                "parent_jacobian_hits", "v2_jacobian_cache_hits",
                "newly_constructed_jacobians_after_cache_misses",
            )
        },
        "historical_total_K_anchors": {
            "methods": [*ZERO, *OLD],
            "CELL_METRICS": artifact_reference(TOTAL / "CELL_METRICS.csv"),
            "target_lock": artifact_reference(TOTAL / "TARGET_LOCK.json"),
            "coverage_verification": artifact_reference(TOTAL / "DELIVERY_VERIFICATION.json"),
        },
        "historical_HUST_CAL2_anchor": {
            "verified_slots": evidence["historical_HUST_CAL2_slots"],
            "identity_audit": artifact_reference(legacy_cal2_proof),
            "original_source_lock": artifact_reference(PARENT / "TARGET_LOCK.json"),
            "source_selection_is_original_support_Gram_regularized_CAL2": True,
        },
        "zero_retaining_LoRA": {
            "original_checkpoint_and_fit_budget_preserved": True,
            "source_selection": artifact_reference(lora_proof),
            **supplemental_evidence,
            "target_prediction_origins": dict(origins),
            "target_unique_saved_or_reused_states": ledger["unique_lora_target_state_payloads_by_method"],
            "source_optimizer_updates_added": 0,
            "target_optimizer_updates_added": 0,
        },
        "execution_ledger": artifact_reference(public / "EXECUTION_LEDGER.json"),
        "public_record_scope": "Identity, receipt and payload checksums only. Arrays and checkpoints remain in the private runtime.",
    }


def registered_draws(episodes):
    draws = {}
    for task in episodes["target_tasks"]:
        for shot in task["shots"]:
            for draw in shot["draws"]:
                k = shot["k"]
                uids = draw["support_ids"]
                query = task["query_ids"]
                if (
                    len(set(uids)) != k
                    or len(uids) != k
                    or not set(uids) <= set(task["support_pool_ids"])
                    or set(uids) & set(query)
                    or draw["query_ids"] != query
                ):
                    raise ValueError("Registered domain-total-K Support/Query identity changed")
                draws[task["domain"], k, draw["draw"]] = (task, draw)
    return draws


def matr_extension_identity(episodes):
    """Use frozen UID membership and legal prefix fields only."""
    split = read_json(SPLIT)
    roles = split["source_cell_roles"]
    source = set(roles["source_train"]) | set(roles["source_meta_validation"])
    heldout = set(roles["target_test"])
    task = next(task for task in episodes["target_tasks"] if task["domain"] == "MATR")
    support = set(task["support_pool_ids"])
    old = set(task["query_ids"])
    if source & heldout or not support <= heldout or not old <= heldout or support & old:
        raise ValueError("MATR expansion identity overlaps a source or Support role")
    candidate = sorted(heldout - support)
    references = []
    for seed in episodes["training_seeds"]:
        receipt_path = PARENT / "references" / "final" / f"seed{seed}" / "REFERENCE_COMPLETE.json"
        receipt = read_json(receipt_path)
        if set(receipt["fit_ids"]) != source or len(receipt["fit_ids"]) != len(source):
            raise ValueError("A final reference fit roster differs from the 115 locked source cells")
        references.append({
            "seed": seed,
            "path": receipt_path.relative_to(ROOT).as_posix(),
            "sha256": sha256(receipt_path),
            "reference_checkpoint_sha256": receipt["reference_checkpoint_sha256"],
        })
    cache_manifest_path = MATR_CACHE / "COMPLETE.json"
    cache_manifest = read_json(cache_manifest_path)
    if sha256(cache_manifest_path) != split["data_caches"]["MATR"]["manifest_sha256"]:
        raise ValueError("MATR cache manifest differs from the frozen split lock")
    artifacts = {entry["path"]: entry["sha256"] for entry in cache_manifest["files"]}
    archive = MATR_CACHE / "arrays.npz"
    metadata = MATR_CACHE / "cell_metadata.csv"
    for path in (archive, metadata):
        if sha256(path) != artifacts[path.name]:
            raise ValueError(f"Frozen MATR cache differs from manifest {path.name}")
    fields = (
        "ids", "raw_viq", "point_mask", "cycle_ids", "anchor",
        "drift_q", "drift_mask", "nominal_capacity_Ah",
    )
    with np.load(archive, allow_pickle=False) as bundle:
        all_ids = bundle["ids"].tolist()
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("MATR prefix cache has repeated physical UID")
        lookup = {uid: position for position, uid in enumerate(all_ids)}
        present = [uid for uid in candidate if uid in lookup]
        positions = [lookup[uid] for uid in present]
        raw, mask, cycles, anchor, drift, dmask, nominal = (
            bundle[name][positions] for name in fields[1:]
        )
    qualification = pd.read_csv(
        metadata, usecols=["cell_uid", "input_eligible", "protocol_uid", "historical_exposure"]
    ).set_index("cell_uid", verify_integrity=True)
    protocol = {uid: group for group, ids in split["source_groups"].items() for uid in ids}
    entries = {}
    for index, uid in enumerate(present):
        active_cycles = cycles[index, mask[index].any(axis=1)]
        prefix_valid = bool(
            active_cycles.size
            and np.isfinite(active_cycles).all()
            and np.equal(active_cycles, np.floor(active_cycles)).all()
            and ((active_cycles >= 1) & (active_cycles <= 100)).all()
            and np.isfinite(raw[index][mask[index]]).all()
            and np.isfinite(drift[index][dmask[index]]).all()
            and np.isfinite(anchor[index])
            and np.isfinite(nominal[index])
            and nominal[index] > 0
        )
        row = qualification.loc[uid]
        if row.protocol_uid != protocol[uid]:
            raise ValueError("MATR prefix metadata differs from registered protocol identity")
        entries[uid] = {
            "input_eligible": bool(prefix_valid and parse_boolean(row.input_eligible)),
            "protocol_uid": protocol[uid],
            "historical_exposure": str(row.historical_exposure),
            "already_in_main8": uid in old,
        }
    ids = [uid for uid in candidate if entries.get(uid, {}).get("input_eligible")]
    added = [uid for uid in ids if uid not in old]
    all_eligible = len(ids) == len(candidate) and old <= set(ids)
    return {
        "status": "PREFIX_IDENTITY_ELIGIBLE" if all_eligible else "PREFIX_IDENTITY_INCOMPLETE",
        "evaluation_status": "OPTIONAL_EXPANSION_NOT_EVALUATED_BY_IDENTITY_AUDIT",
        "source_fit_ids": len(source),
        "heldout_ids": len(heldout),
        "fixed_support_pool_ids": sorted(support),
        "main_query_ids": task["query_ids"],
        "eligible_complement_query_ids": ids,
        "eligible_additional_query_ids": added,
        "ineligible_or_missing_uids": sorted(set(candidate) - set(ids)),
        "main_protocol_groups": len({protocol[uid] for uid in old}),
        "additional_protocol_groups": len({protocol[uid] for uid in added}),
        "complement_protocol_groups": len({protocol[uid] for uid in ids}),
        "prefix_members_read": list(fields),
        "metadata_columns_read": list(qualification.columns.insert(0, "cell_uid")),
        "future_target_or_observation_arrays_deserialized": 0,
        "historical_query_exposure": True,
        "qualification": entries,
        "split_lock_sha256": sha256(SPLIT),
        "total_K_episodes_sha256": sha256(EPISODES),
        "frozen_cache_manifest_sha256": sha256(MATR_CACHE / "COMPLETE.json"),
        "frozen_prefix_archive_sha256": artifacts[archive.name],
        "final_reference_fit_rosters": references,
    }


def historical_anchors(episodes):
    """Authenticate exact HUST CAL2 identity and load only scored CSV facts."""
    draws = registered_draws(episodes)
    previous = read_json(PARENT / "TARGET_LOCK.json")
    current = read_json(TOTAL / "TARGET_LOCK.json")
    prior_metrics = pd.read_csv(TOTAL / "CELL_METRICS.csv", low_memory=False)
    prior_metrics = prior_metrics[prior_metrics.method.isin((*ZERO, *OLD))].copy()
    prior_metrics["result_origin"] = "historical_domain_total_K"
    prior_metrics["effective_state_id"] = prior_metrics.adaptation_id
    prior_metrics["active_coefficients"] = prior_metrics.K.gt(0)
    legacy_rows, receipts = [], []
    for (domain, k, draw), (task, registered) in draws.items():
        if domain != "HUST" or k == 0:
            continue
        for seed in episodes["training_seeds"]:
            prefix = Path(task["task_id"]) / f"seed{seed}"
            slot_name = f"K{k}_draw{draw}"
            folder = PARENT / "units/D4" / prefix / "CAL2" / slot_name
            receipt_path = folder / "RECEIPT.json"
            receipt = read_json(receipt_path)
            zero_path = TOTAL / "units" / prefix / "SPECIALIZED_ZERO/K0_draw0/RECEIPT.json"
            zero = read_json(zero_path)
            out = read_json(TOTAL / "units" / prefix / "SPECIALIZED_OUT" / slot_name / "RECEIPT.json")
            metrics_path = folder / "CELL_METRICS.csv"
            if (
                receipt["status"] != "SCORED"
                or receipt["method"] != "CAL2"
                or receipt["seed"] != seed
                or receipt["K"] != k
                or receipt["draw"] != draw
                or receipt["task_id"] != task["task_id"]
                or receipt["support_ids"] != registered["support_ids"]
                or receipt["query_ids"] != task["query_ids"]
                or zero["query_ids"] != task["query_ids"]
                or out["support_ids"] != registered["support_ids"]
                or receipt["config_sha256"] != previous["config_sha256"]
                or receipt["config_sha256"] != current["parent_config_sha256"]
                or receipt["source_checkpoint_sha256"] != zero["source_checkpoint_sha256"]
                or receipt["source_checkpoint_sha256"] != out["source_checkpoint_sha256"]
                or receipt["dictionary_sha256"] != out["adapter_dictionary_sha256"]
                or receipt["hyperparameters"] != previous["selected_hyperparameters"]["CAL2"][str(k)]
                or sha256(metrics_path) != receipt["cell_metrics_sha256"]
            ):
                raise ValueError(f"Historical HUST CAL2 identity does not match {receipt_path}")
            payload_checks = {}
            for role in ("prediction", "state"):
                artifact = ROOT / receipt[f"{role}_path"]
                expected = receipt[f"{role}_sha256"]
                if artifact.exists() and sha256(artifact) != expected:
                    raise ValueError(f"Historical CAL2 {role} checksum differs {artifact}")
                payload_checks[role] = artifact.exists()
            scores = pd.read_csv(metrics_path)
            scores["method"] = LEGACY_CAL2
            scores["reference_family"] = "SPECIALIZED"
            scores["result_origin"] = "historical_HUST_CAL2"
            scores["effective_state_id"] = receipt["state_sha256"]
            scores["active_coefficients"] = True
            legacy_rows.append(scores)
            receipts.append({
                "seed": seed, "K": k, "draw": draw,
                "receipt_path": receipt_path.relative_to(ROOT).as_posix(),
                "receipt_sha256": sha256(receipt_path),
                "metrics_sha256": receipt["cell_metrics_sha256"],
                "checkpoint_sha256": receipt["source_checkpoint_sha256"],
                "hyperparameters": receipt["hyperparameters"],
                "local_payloads_sha256_verified": payload_checks,
            })
    legacy = pd.concat(legacy_rows, ignore_index=True)
    evidence = {
        "status": "HISTORICAL_ANCHORS_IDENTITY_VERIFIED",
        "total_K_metrics_sha256": sha256(TOTAL / "CELL_METRICS.csv"),
        "total_K_lock_sha256": sha256(TOTAL / "TARGET_LOCK.json"),
        "historical_source_only_CAL2_lock_sha256": sha256(PARENT / "TARGET_LOCK.json"),
        "total_K_episodes_sha256": sha256(EPISODES),
        "historical_HUST_CAL2_slots": len(receipts),
        "historical_HUST_CAL2_K5_full_mean_mae": float(
            legacy.loc[legacy.K.eq(5) & legacy.window.eq("all"), "mae"].mean()
        ),
        "comparison_scope": "Same fixed HUST Support/Query and SPECIALIZED f0. "
        "Legacy CAL2 uses the original Support-Gram scale and original source hyperparameters.",
        "historical_MATR_four_task_adaptations_used": 0,
        "new_model_forwards": 0,
        "new_Query_label_array_reads": 0,
        "HUST_CAL2_receipts": receipts,
    }
    return pd.concat([prior_metrics, legacy], ignore_index=True), evidence


def validate_cells(frame, episodes):
    frame = frame.copy()
    mandatory = set(QUERY + SCORES + [
        "adaptation_id", "task_id", "protocol_uid", "n_obs", "negative_transfer",
        "same_protocol_support_count", "is_duplicate", "reference_family", "result_origin",
    ])
    missing = mandatory - set(frame)
    if missing:
        raise ValueError(f"Scored cell facts omit required columns {sorted(missing)}")
    if frame.duplicated(QUERY).any():
        raise ValueError("A scored physical cell-window repeats within its slot")
    if not frame.reference_family.eq(frame.method.map(FAMILY)).all():
        raise ValueError("A method changed its registered reference family")
    for name in ("seed", "K", "draw", "n_obs", "same_protocol_support_count"):
        values = pd.to_numeric(frame[name], errors="raise")
        if not (np.isfinite(values) & values.ge(0) & values.eq(np.floor(values))).all():
            raise ValueError(f"An identity or count is invalid {name}")
        frame[name] = values.astype(int)
    if not frame.window.isin(WINDOW_LABELS).all():
        raise ValueError("A scored window differs from the fixed grid")
    frame["is_duplicate"] = frame.is_duplicate.map(parse_boolean)
    frame["negative_transfer"] = frame.negative_transfer.map(
        lambda value: parse_boolean(value, allow_missing=True)
    )
    for name in SCORES:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    observed = frame.n_obs.gt(0)
    if (
        not np.isfinite(frame.loc[observed, SCORES]).all(axis=None)
        or frame.loc[~observed, SCORES].notna().any(axis=None)
        or frame.loc[observed, ["mae", "rmse", "zero_mae"]].lt(0).any(axis=None)
        or not np.allclose(frame.gain_mae, frame.zero_mae - frame.mae, equal_nan=True, atol=1e-12)
        or not frame.negative_transfer.eq(frame.gain_mae.lt(0)).all()
    ):
        raise ValueError("Scored MAE, Delta K, or unsupported window semantics changed")
    observed_counts = frame.groupby(["domain", "cell_uid", "window"]).n_obs.nunique()
    if observed_counts.gt(1).any():
        raise ValueError("Compared methods used different physical observation support")
    draws = registered_draws(episodes)
    for (domain, seed, method, k, draw), rows in frame.groupby(SLOT, sort=False):
        if seed not in episodes["training_seeds"] or (k == 0) != (method in ZERO):
            raise ValueError("Scored method, K, or seed escaped the frozen contract")
        task, registered = draws[domain, k, draw]
        query_windows = set(product(task["query_ids"], WINDOW_LABELS))
        if set(zip(rows.cell_uid, rows.window)) != query_windows:
            raise ValueError(f"Scored Query membership incomplete {domain} {seed} {method} {k} {draw}")
        protocols = task["protocol_by_id"]
        same_protocol = {
            uid: sum(protocols[sid] == protocols[uid] for sid in registered["support_ids"])
            for uid in task["query_ids"]
        }
        if (
            not rows.task_id.eq(task["task_id"]).all()
            or not rows.protocol_uid.eq(rows.cell_uid.map(protocols)).all()
            or not rows.same_protocol_support_count.eq(rows.cell_uid.map(same_protocol)).all()
            or not rows.is_duplicate.eq(registered["duplicate_of_draw"] is not None).all()
            or rows.adaptation_id.isna().any()
            or rows.groupby("cell_uid").adaptation_id.nunique().gt(1).any()
            or (not method.startswith("L-") and rows.adaptation_id.nunique() != 1)
        ):
            raise ValueError(f"A scored slot changed its descriptive identity {domain} {seed} {method} {k} {draw}")
    zero_keys = ["domain", "seed", "reference_family", "cell_uid", "window"]
    zero = frame[frame.K.eq(0)][zero_keys + ["mae"]].rename(columns={"mae": "audit_zero"})
    audit = frame.merge(zero, on=zero_keys, how="left", validate="many_to_one")
    if not np.allclose(audit.zero_mae, audit.audit_zero, atol=1e-7, rtol=1e-6, equal_nan=True):
        raise ValueError("Own-family ZERO differs from its paired physical-cell scoring fact")
    frame["stage"] = "TARGET"
    return frame


def coverage(frame, episodes):
    draws = registered_draws(episodes)
    actual = set(frame[SLOT].itertuples(index=False, name=None))
    rows = []
    for (domain, k, draw), (_, registered) in draws.items():
        methods = ZERO if k == 0 else (*NEW, *OLD, *((LEGACY_CAL2,) if domain == "HUST" else ()))
        for seed, method in product(episodes["training_seeds"], methods):
            key = (domain, seed, method, k, draw)
            rows.append(dict(zip(SLOT, key, strict=True)) | {
                "status": "SCORED" if key in actual else "MISSING",
                "support_uids": json.dumps(registered["support_ids"]),
                "physical_support_cells": len(registered["support_ids"]),
                "duplicate_of_draw": registered["duplicate_of_draw"],
                "physical_query_cells": len(registered["query_ids"]),
            })
    table = pd.DataFrame(rows)
    if actual - set(table[SLOT].itertuples(index=False, name=None)):
        raise ValueError("An unregistered V2 target slot was scored")
    return table


def match_scores(frame, left, right):
    scores = frame[frame.n_obs.gt(0)]
    left_rows = scores[scores.method.eq(left)]
    right_rows = scores[scores.method.eq(right)][PAIR + ["mae", "gain_mae", "n_obs"]]
    pairs = left_rows.merge(
        right_rows, on=PAIR, how="inner", suffixes=("", "_comparison"), validate="one_to_one"
    )
    if not pairs.n_obs.eq(pairs.n_obs_comparison).all():
        raise ValueError("A paired method contrast changed the observation mask")
    pairs["comparison"] = right
    pairs["mae_improvement"] = pairs.mae_comparison - pairs.mae
    pairs["delta_K_difference"] = pairs.gain_mae - pairs.gain_mae_comparison
    return pairs


def cal2_incremental(frame):
    pieces = [match_scores(frame, left, right) for left, right in PAIR_METHODS]
    selected = pd.concat(pieces, ignore_index=True)
    keys = ["domain", "method", "comparison", "K", "window"]
    if selected.empty:
        return pd.DataFrame(columns=keys + ["level", "seed", "draw", "mae_improvement"])
    output = []
    for level, details in (("cell", ["seed", "draw", "cell_uid"]),
                           ("seed_draw", ["seed", "draw"]), ("aggregate", [])):
        group = selected.groupby(keys + details, sort=True)
        summary = group.agg(
            method_mae=("mae", "mean"),
            comparison_mae=("mae_comparison", "mean"),
            mae_improvement=("mae_improvement", "mean"),
            delta_K_difference=("delta_K_difference", "mean"),
            physical_query_cells=("cell_uid", "nunique"),
            observations_with_repeated_evaluation=("n_obs", "sum"),
            scored_cell_seed_draw_rows=("mae", "size"),
            positive_cell_seed_draw_rows=("mae_improvement", lambda v: int(v.gt(0).sum())),
        ).reset_index()
        summary["level"] = level
        output.append(summary)
    return pd.concat(output, ignore_index=True)


def metric_outputs(frame):
    cells = frame.copy()
    cells["level"] = "cell"
    cells["physical_query_cells"] = cells.n_obs.gt(0).astype(int)
    cells["observations_with_repeated_evaluation"] = cells.n_obs
    cells["relative_gain_percent"] = 100 * cells.gain_mae / cells.zero_mae.replace(0, np.nan)
    cells["negative_transfer_ratio"] = cells.negative_transfer.where(cells.n_obs.gt(0)).astype(float)
    aggregate_rows = aggregate(frame, ["reference_family", "result_origin"])
    aggregate_rows["level"] = "aggregate"
    observation_support = frame.loc[frame.n_obs.gt(0), ["domain", "cell_uid", "window", "n_obs"]]
    observation_support = observation_support.drop_duplicates().groupby(["domain", "window"]).n_obs.sum()
    aggregate_rows["physical_observation_points"] = [
        observation_support.get((row.domain, row.window), 0)
        for row in aggregate_rows.itertuples()
    ]
    seed_draw = aggregate(frame, ["reference_family", "result_origin", "seed", "draw"])
    seed_draw["level"] = "seed_draw"
    summaries = pd.concat([aggregate_rows, seed_draw], ignore_index=True)
    for right, output_name in (
        ("G-CAL2", "mae_minus_G_CAL2"),
        ("L-CAL2", "mae_minus_L_CAL2"),
        ("PRETRAINED_LORA", "mae_minus_pretrained_LoRA"),
        ("SPECIALIZED_LORA", "mae_minus_specialized_LoRA"),
        (LEGACY_CAL2, "mae_minus_historical_HUST_CAL2"),
    ):
        matched = pd.concat(
            [match_scores(frame, left, right) for left in frame.method.unique() if left not in ZERO],
            ignore_index=True,
        )
        cells = cells.merge(
            matched[QUERY + ["mae_improvement"]].rename(columns={"mae_improvement": output_name}),
            on=QUERY, how="left", validate="one_to_one",
        )
        cells[output_name] *= -1
        summaries[output_name] = np.nan
        for level, details in (("aggregate", []), ("seed_draw", ["seed", "draw"])):
            keys = ["domain", "method", "K", "window", *details]
            if not matched.empty:
                differences = -matched.groupby(keys)["mae_improvement"].mean()
                query_index = pd.MultiIndex.from_frame(summaries.loc[summaries.level.eq(level), keys])
                summaries.loc[summaries.level.eq(level), output_name] = differences.reindex(query_index).to_numpy()
    output = pd.concat([cells, summaries], ignore_index=True, sort=False)
    output["adaptation_coordinates_per_state"] = output.method.map(DIMENSION)
    output["state_scope"] = np.where(output.method.str.startswith("L-"), "per_Query", "shared_episode")
    output["unique_effective_state_fingerprints"] = np.nan
    for level, details in (("aggregate", []), ("seed_draw", ["seed", "draw"]), ("cell", ["seed", "draw", "cell_uid"])):
        keys = ["domain", "method", "K", "window", *details]
        derived = frame.groupby(keys, sort=False).effective_state_id.nunique(dropna=True)
        selected = output.level.eq(level)
        index = pd.MultiIndex.from_frame(output.loc[selected, keys])
        output.loc[selected, "unique_effective_state_fingerprints"] = derived.reindex(index).to_numpy()
    return output


def load_new_cells(public, historical):
    paths = sorted((public / "target").glob("*/*/METRICS.csv"))
    lora_paths = sorted((public / "e0" / "units").rglob("CELL_METRICS.csv"))
    paths.extend(lora_paths)
    lora_receipts = {path: path.with_name("RECEIPT.json") for path in lora_paths}
    lock_sha = sha256(public / "TARGET_TOTAL_K_LOCK.json") if paths else None
    draws = registered_draws(read_json(EPISODES))
    pieces = []
    for path in paths:
        rows = pd.read_csv(path)
        unexpected = set(rows.method) - set((*NEW, *ZERO))
        if unexpected:
            raise ValueError(f"Unexpected method in a V2 scoreboard shard {path} {sorted(unexpected)}")
        if path in lora_receipts:
            receipt = read_json(lora_receipts[path])
            task, draw = draws[receipt["domain"], receipt["K"], receipt["draw"]]
            if (
                receipt["status"] != "SCORED"
                or receipt["target_lock_sha256"] != lock_sha
                or receipt["cell_metrics_sha256"] != sha256(path)
                or receipt["target_Query_labels_used_for_adaptation_or_selection"] != 0
                or receipt["support_ids"] != draw["support_ids"]
                or receipt["query_ids"] != task["query_ids"]
                or any(not rows[name].eq(receipt[name]).all() for name in SLOT)
                or not rows.adaptation_id.eq(receipt["adaptation_id"]).all()
            ):
                raise ValueError(f"V2 LoRA scoreboard shard does not match its locked receipt {path}")
            rows["effective_state_id"] = receipt["state_sha256"]
            rows["active_coefficients"] = bool(receipt["details"]["trainable_parameters"])
        else:
            receipt_path = path.with_name("SCORED.json")
            receipt = read_json(receipt_path)
            if (
                receipt["status"] != "SCORED"
                or receipt["target_lock_sha256"] != lock_sha
                or receipt["metrics_sha256"] != sha256(path)
                or receipt["transfer_diagnostics_sha256"] != sha256(path.with_name("TRANSFER_DIAGNOSTICS.csv"))
                or receipt["prediction_manifest_sha256"] != sha256(path.with_name("PREDICTIONS_LOCKED.json"))
                or receipt["target_query_label_uses_for_selection_or_adaptation"] != 0
                or not rows.domain.eq(receipt["domain"]).all()
                or not rows.seed.eq(receipt["seed"]).all()
                or len(rows) != receipt["scored_cell_window_rows"]
            ):
                raise ValueError(f"V2 geometry scoreboard shard does not match its locked receipt {path}")
        rows["result_origin"] = "transfer_scope_v2"
        k0 = rows[rows.K.eq(0)].copy()
        if not k0.empty:
            if not k0.reference_family.eq(k0.method.map(FAMILY)).all():
                raise ValueError(f"An additional K0 shard changed reference family {path}")
            k0["method"] = k0.reference_family.map(
                {"WARMUP": "WARMUP_ZERO", "SPECIALIZED": "SPECIALIZED_ZERO"}
            )
            expected = historical[historical.method.isin(ZERO)][QUERY + ["n_obs", "mae"]]
            compared = k0.merge(expected, on=QUERY, suffixes=("", "_historical"), validate="one_to_one")
            if (
                not k0.K.eq(0).all()
                or len(compared) != len(k0)
                or not compared.n_obs.eq(compared.n_obs_historical).all()
                or not np.allclose(compared.mae, compared.mae_historical, atol=1e-7, rtol=1e-6, equal_nan=True)
            ):
                raise ValueError(f"An additional K0 shard changed its reference {path}")
        pieces.append(rows[rows.K.ne(0)])
    combined = pd.concat([historical, *pieces], ignore_index=True)
    return combined, paths


def collect_diagnostics(public, cells, complete):
    paths = sorted((public / "target").glob("*/*/TRANSFER_DIAGNOSTICS.csv"))
    paths.extend(sorted((public / "e0" / "units").rglob("TRANSFER_DIAGNOSTICS.csv")))
    keys = ["domain", "seed", "method", "K", "draw", "cell_uid"]
    if paths:
        diagnostics = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    else:
        diagnostics = pd.DataFrame(columns=keys)
    if diagnostics.duplicated(keys).any():
        raise ValueError("Transfer diagnostics repeated an evaluation identity")
    full = cells[cells.window.eq("all")][keys + ["mae", "gain_mae", "n_obs", "adaptation_id"]]
    mapping = {
        "mae": "scored_query_mae",
        "gain_mae": "scored_query_gain_mae",
        "n_obs": "scored_query_observations",
        "adaptation_id": "scored_adaptation_id",
    }
    diagnostics = diagnostics.merge(
        full.rename(columns=mapping), on=keys, how="left", validate="one_to_one", indicator=True,
    )
    if not diagnostics["_merge"].eq("both").all():
        raise ValueError("Transfer diagnostics differ from the scored Query identities")
    if complete:
        geom = full[full.method.isin(GEOMETRY)][keys]
        expected = set(geom.itertuples(index=False, name=None))
        actual = set(diagnostics.loc[diagnostics.method.isin(GEOMETRY), keys].itertuples(index=False, name=None))
        if expected != actual:
            raise ValueError("Transfer diagnostics do not cover all scored geometry Query slots")
    for check in ("query_mae", "query_gain_mae"):
        if check in diagnostics and not np.allclose(
            diagnostics[check], diagnostics["scored_" + check], atol=1e-8, rtol=1e-6, equal_nan=True
        ):
            raise ValueError(f"Transfer diagnostic and metric disagree {check}")
    return diagnostics.drop(columns="_merge"), paths


def format_number(value):
    return f"{value:.6f}" if pd.notna(value) else "—"


def render_full_table(metrics):
    header = [
        "| 域 | 方法 | K | MAE | 家族 K0 | ΔK | 相对改善 % | MAE − G-CAL2 | MAE − 预训练 LoRA | MAE − 专门化 LoRA | 有效 Query | 观测点 | 每状态坐标 | 登记状态数 | 有效状态指纹数 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    full = metrics.loc[metrics.level.eq("aggregate") & metrics.window.eq("all")]
    for row in full.itertuples():
        values = (
            row.domain, row.method, int(row.K), format_number(row.mae), format_number(row.zero_mae),
            format_number(row.gain_mae), f"{row.relative_gain_percent:.2f}" if pd.notna(row.relative_gain_percent) else "—",
            format_number(row.mae_minus_G_CAL2), format_number(row.mae_minus_pretrained_LoRA),
            format_number(row.mae_minus_specialized_LoRA), int(row.physical_query_cells),
            int(row.physical_observation_points), row.adaptation_coordinates_per_state,
            int(row.unique_adapted_states), int(row.unique_effective_state_fingerprints),
        )
        header.append("| " + " | ".join(map(str, values)) + " |")
    return "\n".join(header)


def render_contrasts(contrasts):
    header = [
        "| 域 | K | 待比较方法 | 对照 | 完整观测 MAE 改善 | 完整改善种子 | 远期 MAE 改善 | 远期改善种子 |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    keys = ["domain", "K", "method", "comparison"]
    selected = contrasts[contrasts.window.isin(("all", "1001_plus"))]
    aggregates = selected[selected.level.eq("aggregate")]
    seed_draw = selected[selected.level.eq("seed_draw")]
    signs = seed_draw.groupby(keys + ["window", "seed"]).mae_improvement.mean()
    for key, group in aggregates.groupby(keys, sort=True):
        observed = group.set_index("window").mae_improvement
        values = list(key)
        for window in ("all", "1001_plus"):
            values.append(format_number(observed.get(window, np.nan)))
            try:
                per_seed = signs.loc[(*key, window)]
            except KeyError:
                values.append("—")
            else:
                values.append(f"{int(per_seed.gt(0).sum())}/{len(per_seed)}")
        header.append("| " + " | ".join(map(str, values)) + " |")
    return "\n".join(header)


def render_focal_evidence(metrics, paired, selection):
    """Report signed effects and source choices without an outcome-based gate."""
    view = metrics[metrics.level.eq("aggregate") & metrics.window.eq("all")]
    sections = []
    for domain, rows in view.groupby("domain", sort=True):
        chosen = rows[rows.method.isin(GEOMETRY) & rows.K.gt(0)].set_index(["method", "K"])
        specs = rows[rows.method.eq("SPECIALIZED_ZERO")]
        if chosen.empty or specs.empty:
            continue
        selected = paired[(paired.domain == domain) & paired.level.eq("aggregate") & paired.window.eq("all")]
        key = selected.set_index(["method", "comparison", "K"]).mae_improvement
        for k in (1, 3, 5):
            if ("G-CAL2", k) not in chosen.index:
                continue
            gain = chosen.loc[("G-CAL2", k), "gain_mae"]
            columns = [
                (("L-CAL2", "G-CAL2", k), "局部CAL2相对全局CAL2"),
                (("G-JOINT18", "G-CAL2", k), "全局联合相对对应CAL2"),
                (("L-JOINT18", "L-CAL2", k), "局部联合相对对应CAL2"),
                (("L-JOINT18", "G-JOINT18", k), "局部联合相对全局联合"),
            ]
            effects = "，".join(
                f"{label} {key.loc[contrast]:+.6f}"
                for contrast, label in columns if contrast in key.index
            )
            sections.append(f"- {domain} K={k} 的全局 CAL2 ΔK 为 {gain:+.6f}。{effects}。" if effects else
                            f"- {domain} K={k} 的全局 CAL2 ΔK 为 {gain:+.6f}。")
    if selection:
        settings = selection.get("selected_hyperparameters", {})
        if settings:
            zero = [
                f"{method} K={k}"
                for method, values in settings.items()
                for k, choice in values.items()
                if choice.get("eta") == 0.0
            ]
            count = sum(len(values) for values in settings.values())
            formatted = "、".join(zero)
            sections.append(
                f"源内锁定的 {count} 组方法与K组合中有 {len(zero)} 组选择 η=0。"
                + (f"对应组合为 {formatted}。这些组合直接保留各自 f0。" if zero else "")
            )
    return "\n".join(sections)


def render_summary(metrics, contrasts, diagnostics, slot_coverage, evidence, optional, ledger, selection):
    missing = int(slot_coverage.status.ne("SCORED").sum())
    status = "PARTIAL" if missing else "COMPLETE"
    title = "# 总 K 校准与迁移范围 V2 复核"
    sections = [
        title,
        "## 执行范围",
        f"已核 {len(slot_coverage) - missing}/{len(slot_coverage)} 个评分槽，状态为 **{status}**。"
        "MATR 从24颗候选总共抽取 K 颗并评价8颗固定 Query。HUST 从38颗候选总共抽取 K 颗并评价39颗固定 Query。"
        "K 为0、1、3、5，种子42、123、2026，非零 K 保留五次冻结抽样。MATR K1 对应3个不同 Support 集合。"
        "既有 MATR 四个任务合计4K的适配结果保留在历史报告中。",
        "两类参考共享源训练 Bridge/Head，主机制矩阵使用 SPECIALIZED f0。"
        "HUST 历史 CAL2 与当前主矩阵使用相同 Support、Query 和参考函数。历史 CAL2 使用原锁定参数与Support Gram正则尺度。"
        f"其 K5 完整观测 MAE 为 {evidence['historical_HUST_CAL2_K5_full_mean_mae']:.6f}。"
        "当前 G-CAL2 使用新源内锁定参数与固定源 Gram 尺度。",
        "## 校准、迁移范围与收缩",
        render_focal_evidence(metrics, contrasts, selection) or "已登记母实验锚点。主矩阵比较仍待评分。",
        "## 完整已观测轨迹",
        "ΔK 为各方法自己的 K0 减去适配 MAE。表中的观测点按物理电芯去重。"
        "误差先按电芯真实观测点计算，再按固定电芯、种子和登记 draw 汇总。"
        "L 方法为每颗 Query 求系数，每状态列只描述一颗 Query 的坐标规模。"
        "登记状态按适配请求身份计数，有效状态指纹合并相同的源起点与系数。"
        "LoRA 使用保存状态哈希，历史总 K 锚点沿用适配身份。",
        render_full_table(metrics),
        "## 局部信息与内部更新的配对增量",
        "表中改善为对照 MAE 减去待比较方法 MAE，使用相同域、K、种子、draw、电芯和窗口。"
        "种子方向先在每个种子内平均五次抽样。逐电芯及逐 seed/draw 的数值保存在 CAL2_INCREMENTAL.csv。",
        render_contrasts(contrasts),
    ]
    if missing:
        sections.extend([
            "## 未完成槽",
            "当前汇总保留已经评分的槽。待完成槽的合同身份保存在 TARGET_SLOT_COVERAGE.csv。",
        ])
    observed = metrics[metrics.level.eq("cell") & metrics.window.eq("all")]
    checked = observed[observed.n_obs.gt(0) & observed.method.isin(GEOMETRY)]
    if len(checked):
        sections.append("## 负迁移与远期观测")
        for domain, rows in checked.groupby("domain", sort=True):
            worst = rows.loc[rows.gain_mae.idxmin()]
            sections.append(
                f"- {domain} 的主矩阵中，{int(rows.negative_transfer.sum())}/{len(rows)} 个电芯、种子和抽样记录发生负迁移。"
                f"最低 ΔK 出现在 {worst['cell_uid']} 的 {worst['method']}，K={int(worst['K'])}，"
                f"seed={int(worst['seed'])}，draw={int(worst['draw'])}，ΔK={worst['gain_mae']:+.6f}。"
            )
    baseline = metrics[metrics.level.eq("aggregate") & metrics.method.eq("SPECIALIZED_ZERO")]
    windows = ["| 域 | 时距 | 有效物理 Query | 真实观测点 |", "| --- | --- | ---: | ---: |"]
    for row in baseline.itertuples():
        windows.append(
            f"| {row.domain} | {row.window} | {int(row.physical_query_cells)} | {int(row.physical_observation_points)} |"
        )
    sections.extend([
        "\n".join(windows),
        "HUST 源记录包含 EOL80 右删失。上表只计各时距有真实容量观测的电芯与点，主指标为实际观测未来的 MAE。",
        "## 诊断与证据边界",
        f"TRANSFER_DIAGNOSTICS.csv 保留 {len(diagnostics)} 条配对诊断。"
        "近邻权重和有效邻居数描述 Support 对单颗 Query 的覆盖。"
        "线性化偏差分别按固定网格和实际观测位置报告。",
        "共同参考、固定总 K、相同 Support 和同物理电芯配对有利于解释矩阵差异。"
        "目标 Query 有历史暴露，种子和抽样是重复评价轴。"
        "本次配对结果的置信度取决于覆盖与方向一致性，向独立新队列外推的置信度较低。",
    ])
    if optional:
        sections.append(
            f"MATR 扩展身份核验确认 {len(optional['eligible_complement_query_ids'])} 颗补集电芯的前缀资格，"
            f"其中 {len(optional['eligible_additional_query_ids'])} 颗在主8颗之外。"
            "本轮先完成使用原固定8颗 Query 的主矩阵。扩展仍处于资格核验阶段，所有候选均有历史暴露记录。"
        )
    if ledger:
        sections.append("训练、状态、求解与前向开销见 EXECUTION_LEDGER.json。源权重、OUT 字典与实际缓存复用见 REUSE_MANIFEST.json。")
    return "\n\n".join(sections) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, default=PUBLIC)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    episodes = read_json(EPISODES)
    historical, evidence = historical_anchors(episodes)
    validate_cells(historical, episodes)
    anchor_path = args.public / "HISTORICAL_ANCHORS.json"
    write_json(anchor_path, evidence)
    extension_path = args.public / "MATR_EXTENSION_IDENTITY.json"
    if args.audit_only:
        optional = matr_extension_identity(episodes)
        write_json(extension_path, optional)
        print(json.dumps({
            "status": "HISTORICAL_ANCHORS_AND_MATR_PREFIX_IDENTITY_VERIFIED",
            "historical_HUST_CAL2_slots": evidence["historical_HUST_CAL2_slots"],
            "MATR_eligible_complement": len(optional["eligible_complement_query_ids"]),
            "MATR_additional_prefix_eligible": len(optional["eligible_additional_query_ids"]),
            "new_Query_label_array_reads": 0,
        }))
        return 0
    cells, paths = load_new_cells(args.public, historical)
    cells = validate_cells(cells, episodes)
    slots = coverage(cells, episodes)
    pending = int(slots.status.ne("SCORED").sum())
    diagnostics, diagnostic_paths = collect_diagnostics(args.public, cells, complete=pending == 0)
    metrics = metric_outputs(cells)
    contrasts = cal2_incremental(cells)
    outputs = {
        args.public / "METRICS.csv": metrics,
        args.public / "CAL2_INCREMENTAL.csv": contrasts,
        args.public / "TRANSFER_DIAGNOSTICS.csv": diagnostics,
        args.public / "TARGET_SLOT_COVERAGE.csv": slots,
    }
    for output, table in outputs.items():
        table.to_csv(output, index=False, encoding="utf-8", lineterminator="\n")
    args.report.mkdir(parents=True, exist_ok=True)
    report_path = args.report / "FINAL_REVIEW_ZH.md"
    ledger = read_json(args.public / "EXECUTION_LEDGER.json")
    optional = read_json(extension_path)
    selection_path = args.public / "TARGET_TOTAL_K_LOCK.json"
    selection = read_json(selection_path)
    reuse_path = args.public / "REUSE_MANIFEST.json"
    if not pending:
        if not ledger:
            raise ValueError("The V2 scoring matrix requires its complete execution ledger before final delivery")
        write_json(reuse_path, build_reuse_manifest(args.public, evidence, ledger))
    report_path.write_text(
        render_summary(metrics, contrasts, diagnostics, slots, evidence, optional, ledger, selection),
        encoding="utf-8",
    )
    manifest = {
        "status": "PARTIAL" if pending else "COMPLETE",
        "scored_slots": int(len(slots) - pending),
        "requested_slots": int(len(slots)),
        "missing_slots": pending,
        "new_method_scored_slots": int((slots.status.eq("SCORED") & slots.method.isin(NEW)).sum()),
        "scored_cell_window_rows": len(cells),
        "new_model_forwards": 0,
        "new_Query_label_array_reads": 0,
        "new_training": 0,
        "new_bootstrap_draws": 0,
        "summarizer_sha256": sha256(Path(__file__)),
        "inputs": [
            artifact_reference(path)
            for path in [
                EPISODES, TOTAL / "CELL_METRICS.csv", *paths, *diagnostic_paths,
                *((selection_path,) if selection else ()),
                *((args.public / "EXECUTION_LEDGER.json",) if ledger else ()),
            ]
        ],
        "files": [
            artifact_reference(path)
            for path in [
                *outputs, anchor_path, report_path, *((extension_path,) if optional else ()),
                *((reuse_path,) if not pending else ()),
            ]
        ],
    }
    write_json(args.public / "SUMMARY_MANIFEST.json", manifest)
    print(json.dumps({k: manifest[k] for k in (
        "status", "scored_slots", "requested_slots", "new_method_scored_slots", "missing_slots",
    )}))
    return int(bool(pending and not args.allow_partial))


if __name__ == "__main__":
    raise SystemExit(main())
