"""Generate source corrections and replay authenticated target Support outputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from scripts.drs_peft.abba_engine import AdapterExperiment, restore, support_batch
from scripts.drs_peft.data import CACHE, ROOT, file_hash, write_json
from scripts.drs_peft.onesided_protocol import payload_hash
from scripts.drs_peft.run_abba_moirai import (
    BACKBONE,
    REFERENCES,
    checkpoint,
    csv,
    models,
    parents,
    qpredict,
    target_slots,
    unit_identity,
)
from scripts.drs_peft.run_abba_moirai import (
    CONFIG as ABBACONFIG,
)
from scripts.drs_peft.run_abba_moirai import (
    PRIVATE as PARENT_PRIVATE,
)
from scripts.drs_peft.run_abba_moirai import (
    PUBLIC as PARENT_PUBLIC,
)
from scripts.drs_peft.run_onesided import shot_for, task_for
from scripts.drs_peft.train import configure, early_batch, read_json
from scripts.drs_peft.transfer_scope_data import load_future_rows, load_prefixes
from scripts.drs_peft.transfer_scope_reference import save_npz

PUBLIC = ROOT / "results/drs_peft/correction_transfer_v1/e1"
PRIVATE = ROOT / "runtime/drs_peft/correction_transfer_v1/e1"
METHODS = ("ABBA_R4X4", "LORA_R8")


@torch.no_grad()
def sampled_prediction(model, batch):
    early, cycles, _, _ = batch
    return model(early, cycles).cpu().numpy()


@torch.no_grad()
def latent(model, inputs, rows, device):
    return np.concatenate(
        [
            model.encode(early_batch(inputs, rows[i : i + 8], device)).cpu().numpy()
            for i in range(0, len(rows), 8)
        ]
    )


def record_arrays(
    case, data, ref, op, ids, query, batch, base_sample, base_full, device
):
    save_npz(
        case,
        support_ids=np.asarray(ids),
        query_ids=np.asarray(query),
        grid=data.query,
        sample_grid=batch[1].cpu().numpy(),
        sample_mask=batch[3].cpu().numpy(),
        sample_base=base_sample,
        sample_adapted=sampled_prediction(op.model, batch),
        full_support_base=base_full,
        full_support_adapted=qpredict(op.model, data, ids, device),
        query_base=qpredict(ref, data, query, device)
        if query
        else np.empty((0, len(data.query))),
        query_adapted=qpredict(op.model, data, query, device)
        if query
        else np.empty((0, len(data.query))),
    )


def source_predict(args, cfg, source):
    prefix = load_prefixes("MATR", args.data_root)
    lock = read_json(PARENT_PUBLIC / "TARGET_LOCK.json")
    all_latents, records = [], []
    for fold in range(3):
        task = task_for(source, "inner", fold)
        query, pool = task["query_ids"], task["support_pool_ids"]
        fit = checkpoint(args, fold, 42)["fit_ids"]
        if (set(query) | set(pool)) - set(source["source_ids"]) or set(
            query + pool
        ) & set(fit):
            raise RuntimeError("Source scope overlaps model fit or target cohort")
        original, reference = models(args, fold, 42)
        original.cpu()
        reference.to(args.device)
        embeddings = latent(
            reference, prefix.inputs, prefix.rows(pool + query), args.device
        )
        all_latents.extend(
            dict(fold=fold, cell_uid=uid, z=vector.tolist())
            for uid, vector in zip(pool + query, embeddings, strict=True)
        )
        reference.cpu()
        schedule = [(1, i, [uid]) for i, uid in enumerate(sorted(pool))]
        schedule += [
            (k, 0, sorted(shot_for(task, k)["draws"][0]["support_ids"])) for k in (3, 5)
        ]
        for method in METHODS:
            op = AdapterExperiment(reference, method, 42, cfg, args.device)
            for k, draw, ids in schedule:
                case = (
                    args.private_root
                    / "source"
                    / f"f{fold}"
                    / method
                    / f"{k}_{draw}.npz"
                )
                marker = case.with_suffix(".json")
                meta = dict(
                    domain="MATR",
                    fold=fold,
                    seed=42,
                    method=method,
                    K=k,
                    draw=draw,
                    episode_id=f"f{fold}_{method}_{k}_{draw}",
                    support_ids=ids,
                    query_ids=query,
                    support_uid=ids[0] if k == 1 else "",
                    parent_reference_sha256=checkpoint(args, fold, 42)[
                        "reference_checkpoint_sha256"
                    ],
                    parent_target_lock_sha256=file_hash(
                        PARENT_PUBLIC / "TARGET_LOCK.json"
                    ),
                    scope="all_inner_support_pool_singletons"
                    if k == 1
                    else "registered_draw0",
                )
                if marker.exists():
                    original_record = read_json(marker)
                    if original_record["metadata"] != meta or original_record[
                        "sha256"
                    ] != file_hash(case):
                        raise RuntimeError("Source prediction receipt changed")
                    records.append(original_record)
                    continue
                batch = support_batch(prefix, args.data_root, ids, args.device, 512)
                restore(op.model, op.initial)
                base_sample = sampled_prediction(op.model, batch)
                base_full = qpredict(op.model, prefix, ids, args.device)
                cost = op.fit(batch, lock["learning_rates"][method][str(k)])
                reference.to(args.device)
                record_arrays(
                    case,
                    prefix,
                    reference,
                    op,
                    ids,
                    query,
                    batch,
                    base_sample,
                    base_full,
                    args.device,
                )
                reference.cpu()
                rec = dict(
                    metadata=meta,
                    path=case.relative_to(args.private_root).as_posix(),
                    sha256=file_hash(case),
                    optimizer_steps=cost["actual_optimizer_steps"],
                )
                write_json(marker, rec)
                records.append(rec)
            print(f"source correction outputs fold={fold} method={method}", flush=True)
            del op
    zpath = args.private_root / "SOURCE_LATENT.jsonl"
    zpath.write_text(
        "\n".join(json.dumps(row) for row in all_latents) + "\n", encoding="utf-8"
    )
    write_json(
        args.private_root / "SOURCE_PREDICTIONS.json",
        {
            "status": "COMPLETE",
            "cases": records,
            "source_latent_sha256": file_hash(zpath),
            "target_rows_read": 0,
        },
    )


def cases_to_metrics(args, cases, *, is_source):
    from scripts.drs_peft.correction_decomposition import decompose_corrections

    cell, episode = [], []
    for record in cases:
        path = args.private_root / record["path"]
        if file_hash(path) != record["sha256"]:
            raise RuntimeError("Private prediction bytes changed")
        meta = record["metadata"]
        domain, ids, query = meta["domain"], meta["support_ids"], meta["query_ids"]
        source_meta = {
            key: value
            for key, value in meta.items()
            if key not in ("support_ids", "query_ids")
        }
        with np.load(path, allow_pickle=False) as arrays:
            if (
                arrays["support_ids"].tolist() != ids
                or arrays["query_ids"].tolist() != query
            ):
                raise RuntimeError(
                    "Prediction membership differs from its immutable receipt"
                )
            grid = arrays["grid"]
            targets, observed = load_future_rows(domain, args.data_root, ids)
            labels = np.zeros_like(arrays["sample_base"])
            cycles, mask = arrays["sample_grid"], arrays["sample_mask"]
            for i in range(len(ids)):
                positions = np.searchsorted(grid, cycles[i, mask[i]])
                if (
                    not np.array_equal(grid[positions], cycles[i, mask[i]])
                    or not observed[i, positions].all()
                ):
                    raise RuntimeError(
                        "Training sample differs from physical observed cycles"
                    )
                labels[i, mask[i]] = targets[i, positions]
            comparisons = [
                (
                    "SUPPORT",
                    "sampled_train",
                    ids,
                    cycles,
                    mask,
                    arrays["sample_base"],
                    arrays["sample_adapted"],
                    labels,
                ),
                (
                    "SUPPORT",
                    "observed_full",
                    ids,
                    grid,
                    observed,
                    arrays["full_support_base"],
                    arrays["full_support_adapted"],
                    targets,
                ),
            ]
            if is_source:
                y, observed = load_future_rows(domain, args.data_root, query)
                comparisons.append(
                    (
                        "QUERY",
                        "observed_full",
                        query,
                        grid,
                        observed,
                        arrays["query_base"],
                        arrays["query_adapted"],
                        y,
                    )
                )
            for pop, view, uids, q, obs, b, a, y in comparisons:
                x, z = decompose_corrections(
                    uids,
                    q,
                    obs,
                    b,
                    a,
                    y,
                    metadata=dict(source_meta, population=pop, support_label_view=view),
                )
                cell.extend(x)
                episode.extend(z)
    return cell, episode


def source_score(args):
    manifest = read_json(args.private_root / "SOURCE_PREDICTIONS.json")
    cases = manifest["cases"]
    if len(cases) != 56 or manifest["status"] != "COMPLETE":
        raise RuntimeError("Expected 56 distinct source adaptation cases")
    cell, episode = cases_to_metrics(args, cases, is_source=True)
    cell_path, episode_path = (
        args.public_root / "SOURCE_CELL.csv",
        args.public_root / "SOURCE_EPISODE.csv",
    )
    csv(cell_path, cell)
    csv(episode_path, episode)
    write_json(
        args.public_root / "SOURCE_DIRECTION_COMPLETE.json",
        {
            "status": "COMPLETE",
            "scope": "SOURCE_DEVELOPMENT_FIRST",
            "fit_cases": len(cases),
            "optimizer_steps_to_recover_unsaved_source_trajectories": sum(
                r["optimizer_steps"] for r in cases
            ),
            "target_rows_read": 0,
            "source_cells_sha256": file_hash(cell_path),
            "source_episodes_sha256": file_hash(episode_path),
            "source_latent_sha256": manifest["source_latent_sha256"],
        },
    )


def target_replay(args, cfg, episodes):
    parent = argparse.Namespace(public_root=PARENT_PUBLIC, private_root=PARENT_PRIVATE)
    lock_path = PARENT_PUBLIC / "TARGET_LOCK.json"
    lock = read_json(lock_path)
    lock_sha = file_hash(lock_path)
    parent_seal = read_json(PARENT_PUBLIC / "TARGET_PREDICTIONS_COMPLETE.json")
    if parent_seal["lock_sha256"] != lock_sha or parent_seal["slots"] != 546:
        raise RuntimeError("Parent predictions are not sealed")
    if (
        read_json(args.public_root / "SOURCE_DIRECTION_COMPLETE.json")["status"]
        != "COMPLETE"
    ):
        raise RuntimeError(
            "Source direction diagnostics must be complete before target replay"
        )
    plans = list(target_slots(parent, cfg, episodes))
    receipts = []
    for folder, task, seed, method, k, draw in plans:
        row = read_json(folder / "RECEIPT.json")
        expected = unit_identity(lock_sha, lock, task, seed, method, k, draw)
        if row["status"] != "SCORED" or any(
            row[key] != value for key, value in expected.items()
        ):
            raise RuntimeError("Parent target slot identity differs")
        if row["metrics_sha256"] != file_hash(folder / "METRICS.csv"):
            raise RuntimeError("Parent target scores differ from their receipt")
        if k:
            receipts.append(row)
    if len(receipts) != 540:
        raise RuntimeError("Target nonzero coverage differs")
    records = []
    max_support_before = 0.0
    max_support_after = 0.0
    for domain in ("MATR", "HUST"):
        prefix = load_prefixes(domain, args.data_root)
        for seed in (42, 123, 2026):
            _, ref = models(args, "final", seed)
            for method in cfg["methods"]:
                op = AdapterExperiment(ref, method, seed, cfg, args.device)
                selected = [
                    r
                    for r in receipts
                    if (r["domain"], r["seed"], r["method"]) == (domain, seed, method)
                ]
                for row in selected:
                    ids = row["support_ids"]
                    state_path = PARENT_PRIVATE / row["state_path"]
                    if file_hash(state_path) != row["state_sha256"]:
                        raise RuntimeError("Parent adapter weight bytes changed")
                    case = (
                        args.private_root
                        / "target"
                        / (row["adaptation_id"][:24] + ".npz")
                    )
                    meta = {
                        k: row[k]
                        for k in (
                            "domain",
                            "seed",
                            "method",
                            "K",
                            "draw",
                            "adaptation_id",
                            "is_duplicate",
                            "support_ids",
                        )
                    }
                    meta["query_ids"] = []
                    rec = dict(
                        metadata=meta,
                        path=case.relative_to(args.private_root).as_posix(),
                        sha256=file_hash(case) if case.exists() else "",
                        optimizer_steps=0,
                    )
                    cache_marker = case.with_suffix(".json")
                    if cache_marker.exists():
                        marker = read_json(cache_marker)
                        if (
                            marker["sha256"] != rec["sha256"]
                            or marker["state_sha256"] != row["state_sha256"]
                        ):
                            raise RuntimeError("Support replay cache changed")
                        before, after = marker["training_sample_mse_replay_difference"]
                        max_support_before = max(max_support_before, before)
                        max_support_after = max(max_support_after, after)
                        records.append(rec)
                        continue
                    payload = torch.load(
                        state_path, map_location="cpu", weights_only=True
                    )
                    if payload_hash(payload["identity"]) != row["adaptation_id"]:
                        raise RuntimeError(
                            "Adapter state belongs to another physical episode"
                        )
                    batch = support_batch(prefix, args.data_root, ids, args.device, 512)
                    restore(op.model, op.initial)
                    base_sample = sampled_prediction(op.model, batch)
                    base_full = qpredict(op.model, prefix, ids, args.device)
                    restore(op.model, payload["adapter_state"])
                    after = sampled_prediction(op.model, batch)
                    mask = batch[3].cpu().numpy()
                    labels = batch[2].cpu().numpy()
                    before_mse = float(
                        np.mean(
                            [
                                np.mean((a[m] - y[m]) ** 2)
                                for a, y, m in zip(
                                    base_sample, labels, mask, strict=True
                                )
                            ]
                        )
                    )
                    after_mse = float(
                        np.mean(
                            [
                                np.mean((a[m] - y[m]) ** 2)
                                for a, y, m in zip(after, labels, mask, strict=True)
                            ]
                        )
                    )
                    diff_before = abs(before_mse - row["support_mse_before"])
                    diff_after = abs(after_mse - row["support_mse_after"])
                    if max(diff_before, diff_after) > 2e-6:
                        raise RuntimeError(
                            "Sampled Support MSE replay differs from optimized parent"
                        )
                    max_support_before = max(max_support_before, diff_before)
                    max_support_after = max(max_support_after, diff_after)
                    record_arrays(
                        case,
                        prefix,
                        ref,
                        op,
                        ids,
                        [],
                        batch,
                        base_sample,
                        base_full,
                        args.device,
                    )
                    rec["sha256"] = file_hash(case)
                    write_json(
                        case.with_suffix(".json"),
                        dict(
                            sha256=rec["sha256"],
                            state_sha256=row["state_sha256"],
                            training_sample_mse_replay_difference=[
                                diff_before,
                                diff_after,
                            ],
                        ),
                    )
                    records.append(rec)
                del op
            print(f"target support real forward {domain} seed={seed}", flush=True)
    cell, episode = cases_to_metrics(args, records, is_source=False)
    csv(args.public_root / "TARGET_SUPPORT_CELL.csv", cell)
    csv(args.public_root / "TARGET_SUPPORT_EPISODE.csv", episode)
    write_json(
        args.public_root / "TARGET_SUPPORT_COMPLETE.json",
        {
            "status": "COMPLETE",
            "registered_nonzero_slots": len(records),
            "unique_replayed_states": len({r["path"] for r in records}),
            "new_optimizer_steps": 0,
            "target_query_future_labels_used": 0,
            "max_observed_new_replay_abs_delta_sampled_mse_before": max_support_before,
            "max_observed_new_replay_abs_delta_sampled_mse_after": max_support_after,
            "cell_sha256": file_hash(args.public_root / "TARGET_SUPPORT_CELL.csv"),
            "episode_sha256": file_hash(
                args.public_root / "TARGET_SUPPORT_EPISODE.csv"
            ),
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--phase",
        choices=("source-predict", "source-score", "target-support"),
        required=True,
    )
    p.add_argument("--config", type=Path, default=ABBACONFIG)
    p.add_argument("--public-root", type=Path, default=PUBLIC)
    p.add_argument("--private-root", type=Path, default=PRIVATE)
    p.add_argument("--data-root", type=Path, default=CACHE)
    p.add_argument("--reference-root", type=Path, default=REFERENCES)
    p.add_argument("--backbone-dir", type=Path, default=BACKBONE)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    args.private_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    configure()
    cfg, source, episodes = parents(args)
    with threadpool_limits(limits=2):
        if args.phase == "source-predict":
            source_predict(args, cfg, source)
        elif args.phase == "source-score":
            source_score(args)
        else:
            target_replay(args, cfg, episodes)


if __name__ == "__main__":
    main()
