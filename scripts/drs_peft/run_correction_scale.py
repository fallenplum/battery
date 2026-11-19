"""Audit ABBA and LoRA-8 optimization scale on registered source episodes."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from scripts.drs_peft.abba_engine import (
    AdapterExperiment,
    restore,
    snapshot,
    support_batch,
    support_mse,
    synchronized,
)
from scripts.drs_peft.data import CACHE, ROOT, file_hash, write_json
from scripts.drs_peft.onesided_calibration import observed_positions
from scripts.drs_peft.onesided_protocol import load_registration
from scripts.drs_peft.run_abba_moirai import (
    checkpoint,
    code_identity,
    csv,
    models,
    qpredict,
    task_for,
)
from scripts.drs_peft.run_transfer_scope import draws
from scripts.drs_peft.train import bind, configure, read_json, save_torch
from scripts.drs_peft.transfer_scope_data import load_future_rows, load_prefixes
from scripts.drs_peft.transfer_scope_reference import BACKBONE, REFERENCES, save_npz

CONFIG = ROOT / "configs/drs_peft/correction_transfer_v1/E2_CONFIG.json"
PUBLIC = ROOT / "results/drs_peft/correction_transfer_v1/e2"
PRIVATE = ROOT / "runtime/drs_peft/correction_transfer_v1/e2"
METHODS = ("ABBA_R4X4", "LORA_R8")
EPSILON = 1e-12


def episode_ids(registration, fit_ids, fold, k, draw):
    task = task_for(registration, "inner", fold)
    records = [row for shot, row in draws(task) if shot == k and row["draw"] == draw]
    if len(records) != 1:
        raise ValueError("Expected one registered source support draw")
    support = sorted(records[0]["support_ids"])
    query = task["query_ids"]
    source = set(registration["source_ids"])
    eligible = set(fit_ids)
    pools = set(query) | set(task["support_pool_ids"])
    if (
        len(support) != k
        or len(set(support)) != k
        or len(query) != len(set(query))
        or set(support) & set(query)
        or (pools | eligible) - source
        or pools & eligible
        or (pools | eligible) & set(registration["sealed_final_MATR_ids"])
    ):
        raise RuntimeError("The source fit, support or query roles changed")
    return support, query


def anchor_ids(fit_ids, fold, count):
    if len(set(fit_ids)) != len(fit_ids) or len(fit_ids) < count:
        raise ValueError("Insufficient distinct source-fit prefix anchors")
    return sorted(
        fit_ids,
        key=lambda uid: (
            hashlib.sha256(f"correction_scale|{fold}|{uid}".encode()).digest(),
            uid,
        ),
    )[:count]


def errors(prediction, label, observed):
    if (
        prediction.shape != label.shape
        or label.shape != observed.shape
        or not observed.any(axis=1).all()
    ):
        raise ValueError("Equal-cell errors require at least one observation per cell")
    if not np.isfinite(prediction).all() or not np.isfinite(label[observed]).all():
        raise FloatingPointError("Nonfinite prediction or observed trajectory label")
    difference = np.where(observed, prediction.astype(np.float64) - label, 0.0)
    counts = observed.sum(axis=1)
    return {
        "mse": float(np.mean(np.square(difference).sum(axis=1) / counts)),
        "mae": float(np.mean(np.abs(difference).sum(axis=1) / counts)),
    }


def score_predictions(
    prediction, support_count, support_y, support_mask, query_y, query_mask, sampled
):
    support = errors(prediction[:support_count], support_y, support_mask)
    support_sampled = errors(prediction[:support_count], support_y, sampled)
    query = errors(prediction[support_count:], query_y, query_mask)
    return {
        "support_full_mse": support["mse"],
        "support_full_mae": support["mae"],
        "support_sampled_mse": support_sampled["mse"],
        "support_sampled_mae": support_sampled["mae"],
        "query_mse": query["mse"],
        "query_mae": query["mae"],
    }


def function_radius(prediction, base):
    return float(np.sqrt(np.mean(np.square(prediction.astype(np.float64) - base))))


def matched_radius(first, second, noise):
    radius = min(first, second)
    if radius <= max(EPSILON, noise):
        return 0.0, (0.0, 0.0), False
    return radius, (radius / first, radius / second), True


def optimization_snapshots(operator, support, rate, checkpoints):
    """Train one uninterrupted trajectory and pause only for evaluation."""
    restore(operator.model, operator.initial)
    parameters = [p for p in operator.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=rate, weight_decay=operator.opt["weight_decay"]
    )
    device = torch.device(operator.device)
    previous = 0
    training_seconds = 0.0
    peak_allocated = peak_reserved = 0
    for boundary in checkpoints:
        operator.model.train()
        synchronized(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        segment_start = time.perf_counter()
        for _ in range(previous, boundary):
            optimizer.zero_grad(set_to_none=True)
            loss = support_mse(operator.model, support)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                parameters, operator.opt["clip_norm"], error_if_nonfinite=True
            )
            optimizer.step()
        synchronized(device)
        training_seconds += time.perf_counter() - segment_start
        if device.type == "cuda":
            peak_allocated = max(
                peak_allocated, torch.cuda.max_memory_allocated(device)
            )
            peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved(device))
        operator.model.eval()
        with torch.no_grad():
            sampled_mse = float(support_mse(operator.model, support))
        if not math.isfinite(sampled_mse):
            raise FloatingPointError("Nonfinite sampled support objective")
        yield (
            boundary,
            {
                "training_seconds": training_seconds,
                "fit_peak_allocated_bytes": peak_allocated
                if device.type == "cuda"
                else None,
                "fit_peak_reserved_bytes": peak_reserved
                if device.type == "cuda"
                else None,
                "support_objective_mse": sampled_mse,
                "gradient_norm_before_clip": float(norm),
            },
        )
        previous = boundary


def evaluate_model(model, data, ids, device):
    synchronized(device)
    started = time.perf_counter()
    prediction = qpredict(model, data, ids, device)
    synchronized(device)
    return prediction, time.perf_counter() - started


def registered(args):
    cfg = read_json(args.config)
    parent_config = ROOT / cfg["parent_config"]
    parent = read_json(parent_config)
    reference = read_json(ROOT / cfg["parent_reproduction"])
    original = argparse.Namespace(config=parent_config)
    if reference["status"] != "PASS_REAL_MOIRAI" or reference[
        "implementation"
    ] != code_identity(original):
        raise RuntimeError("The registered original adapter code or P0 receipt changed")
    if (
        cfg["domain"] != "MATR"
        or tuple(cfg["methods"]) != METHODS
        or cfg["seed"] not in parent["source"]["seeds"]
        or not set(cfg["folds"]) <= set(parent["source"]["folds"])
        or len(set(cfg["folds"])) != len(cfg["folds"])
        or sorted(set(cfg["snapshots"])) != cfg["snapshots"]
        or min(cfg["snapshots"]) <= 0
        or sorted(set(cfg["learning_rates"])) != cfg["learning_rates"]
        or min(cfg["learning_rates"]) <= 0
        or cfg["anchor_snapshot"]["step"] not in cfg["snapshots"]
        or cfg["anchor_snapshot"]["learning_rate"] not in cfg["learning_rates"]
    ):
        raise ValueError("The E2 matrix or source-only scope changed")
    registration = load_registration(ROOT / parent["source_episodes"])
    identity = {
        "config_sha256": file_hash(args.config),
        "implementation_sha256": file_hash(Path(__file__)),
        "parent_config_sha256": file_hash(parent_config),
        "parent_p0_sha256": file_hash(ROOT / cfg["parent_reproduction"]),
        "episodes_sha256": file_hash(ROOT / parent["source_episodes"]),
        "target_label_rows_loaded": 0,
    }
    bind(args.public_root / "REGISTRATION.json", identity)
    return cfg, parent, registration, identity


def prepare_episode(args, cfg, parent, source, reference, data, fold, k):
    support, query = episode_ids(source, reference["fit_ids"], fold, k, cfg["draw"])
    anchors = anchor_ids(reference["fit_ids"], fold, cfg["anchor_count"])
    support_y, support_mask = load_future_rows("MATR", args.data_root, support)
    query_y, query_mask = load_future_rows("MATR", args.data_root, query)
    maximum = parent["input"]["maximum_observations_per_support_cell"]
    sampled = np.zeros(support_mask.shape, dtype=bool)
    for row, valid in enumerate(support_mask):
        sampled[row, observed_positions(valid, maximum)] = True
    return {
        "support": support,
        "query": query,
        "anchors": anchors,
        "ids": support + query + anchors,
        "support_y": support_y,
        "support_mask": support_mask,
        "query_y": query_y,
        "query_mask": query_mask,
        "sampled": sampled,
        "maximum": maximum,
        "data_manifest_sha256": data.cache_manifest_sha256,
    }


def score_episode(prediction, episode, k):
    return score_predictions(
        prediction,
        k,
        episode["support_y"],
        episode["support_mask"],
        episode["query_y"],
        episode["query_mask"],
        episode["sampled"],
    )


def trajectory_paths(args, fold, k, method, rate):
    relative = Path("curves") / f"fold{fold}" / f"k{k}" / method / f"lr_{rate:.8g}"
    return args.public_root / relative, args.private_root / relative


def anchor_path(args, fold, k, method):
    return (
        args.private_root / "anchors" / f"fold{fold}" / f"k{k}" / method / "ADAPTER.pt"
    )


def episode_identity(identity, cfg, episode, reference, fold, k):
    return dict(
        identity,
        fold=fold,
        seed=cfg["seed"],
        K=k,
        draw=cfg["draw"],
        support_ids=episode["support"],
        source_query_ids=episode["query"],
        source_fit_anchor_ids=episode["anchors"],
        reference_sha256=reference["reference_checkpoint_sha256"],
        data_manifest_sha256=episode["data_manifest_sha256"],
    )


def verify_trajectory(public, private, private_root, identity):
    receipt = read_json(public / "COMPLETE.json")
    if receipt["identity"] != identity:
        raise RuntimeError(f"Completed trajectory has different provenance: {public}")
    for name, root in (
        ("CURVE.csv", public),
        ("UPDATE_SPECTRUM.csv", public),
        ("PREDICTIONS.npz", private),
    ):
        if file_hash(root / name) != receipt["files"][name]:
            raise RuntimeError(f"Completed trajectory has changed: {root / name}")
    anchor = receipt["files"].get("anchor")
    if (
        anchor is not None
        and file_hash(private_root / anchor["path"]) != anchor["sha256"]
    ):
        raise RuntimeError("The registered step-30 adapter checkpoint changed")
    return receipt


def fit_trajectory(
    args, cfg, operator, batch, data, episode, baseline, identity, method, rate
):
    fold, k = identity["fold"], identity["K"]
    public, private = trajectory_paths(args, fold, k, method, rate)
    unit = dict(identity, method=method, learning_rate=rate)
    if (public / "COMPLETE.json").is_file():
        return verify_trajectory(public, private, args.private_root, unit)

    other = k + len(episode["query"])
    initial, _ = evaluate_model(operator.model, data, episode["ids"], args.device)
    replay_max = float(np.max(np.abs(initial - baseline)))
    if replay_max > 2e-6:
        raise RuntimeError(
            "Initial model does not replay the identical frozen function"
        )
    noise = function_radius(initial[other:], baseline[other:])
    base_metrics = score_episode(baseline[:other], episode, k)
    curve, spectrum, predictions = [], [], [baseline[:other]]
    anchor = None
    for step, telemetry in optimization_snapshots(
        operator, batch, rate, cfg["snapshots"]
    ):
        prediction, inference_seconds = evaluate_model(
            operator.model, data, episode["ids"], args.device
        )
        radius = function_radius(prediction[other:], baseline[other:])
        values = score_episode(prediction[:other], episode, k)
        layers, active = operator.spectrum({"allocated_rank_by_module": {}})
        if active != operator.parameter_count:
            raise RuntimeError("The fixed-rank adapter parameter budget changed")
        row_identity = {
            "fold": fold,
            "K": k,
            "draw": cfg["draw"],
            "seed": cfg["seed"],
            "method": method,
            "learning_rate": rate,
            "step": step,
        }
        curve.append(
            dict(
                row_identity,
                **values,
                **telemetry,
                inference_seconds=inference_seconds,
                function_radius=radius,
                initial_function_noise=noise,
                radius_above_initial_noise=radius > max(noise, EPSILON),
                support_full_points=int(episode["support_mask"].sum()),
                support_sampled_points=int(episode["sampled"].sum()),
                source_query_points=int(episode["query_mask"].sum()),
                source_query_cells=len(episode["query"]),
                update_frobenius=math.sqrt(
                    sum(layer["frobenius"] ** 2 for layer in layers)
                ),
                layer_mean_stable_rank=float(
                    np.mean([layer["stable_rank"] for layer in layers])
                ),
                layer_mean_energy_rank_99=float(
                    np.mean([layer["energy_rank_99"] for layer in layers])
                ),
                layer_max_numerical_rank=max(
                    layer["numerical_rank_rtol_1e_5"] for layer in layers
                ),
                parameters=active,
                **{f"initial_{name}": value for name, value in base_metrics.items()},
            )
        )
        spectrum.extend(dict(row_identity, **layer) for layer in layers)
        predictions.append(prediction[:other])
        if (
            rate == cfg["anchor_snapshot"]["learning_rate"]
            and step == cfg["anchor_snapshot"]["step"]
        ):
            path = anchor_path(args, fold, k, method)
            save_torch(
                path,
                {
                    "identity": unit | {"step": step},
                    "adapter_state": snapshot(operator.model),
                },
            )
            anchor = {
                "path": path.relative_to(args.private_root).as_posix(),
                "sha256": file_hash(path),
            }

    csv(public / "CURVE.csv", curve)
    csv(public / "UPDATE_SPECTRUM.csv", spectrum)
    save_npz(
        private / "PREDICTIONS.npz",
        prediction=np.stack(predictions),
        steps=np.asarray([0, *cfg["snapshots"]]),
    )
    receipt = {
        "identity": unit,
        "status": "COMPLETE",
        "optimizer_updates": cfg["snapshots"][-1],
        "initial_prediction_max_abs": replay_max,
        "hardware": torch.cuda.get_device_name(torch.device(args.device))
        if torch.device(args.device).type == "cuda"
        else "CPU",
        "files": {
            "CURVE.csv": file_hash(public / "CURVE.csv"),
            "UPDATE_SPECTRUM.csv": file_hash(public / "UPDATE_SPECTRUM.csv"),
            "PREDICTIONS.npz": file_hash(private / "PREDICTIONS.npz"),
        },
    }
    if anchor is not None:
        receipt["files"]["anchor"] = anchor
    write_json(public / "COMPLETE.json", receipt)
    print(
        f"E2 fold={fold} K={k} method={method} lr={rate}: {cfg['snapshots'][-1]} updates",
        flush=True,
    )
    return receipt


def execute(args, cfg, parent, source, identity, data):
    for fold in args.folds:
        reference = checkpoint(args, fold, cfg["seed"])
        original, model = models(args, fold, cfg["seed"])
        del original
        for k in cfg["shots"]:
            episode = prepare_episode(
                args, cfg, parent, source, reference, data, fold, k
            )
            contract = episode_identity(identity, cfg, episode, reference, fold, k)
            baseline, _ = evaluate_model(
                model.to(args.device), data, episode["ids"], args.device
            )
            # Match the original runner's CPU initialization, including ABBA's SVD.
            model.cpu()
            batch = support_batch(
                data,
                args.data_root,
                episode["support"],
                args.device,
                episode["maximum"],
            )
            for method in cfg["methods"]:
                operator = AdapterExperiment(
                    model, method, cfg["seed"], parent, args.device
                )
                for rate in cfg["learning_rates"]:
                    restore(operator.model, operator.initial)
                    fit_trajectory(
                        args,
                        cfg,
                        operator,
                        batch,
                        data,
                        episode,
                        baseline,
                        contract,
                        method,
                        rate,
                    )
                del operator
        del model


def read_curve(args, cfg, fold, k, method, rate, identity):
    public, private = trajectory_paths(args, fold, k, method, rate)
    receipt = verify_trajectory(
        public,
        private,
        args.private_root,
        dict(identity, method=method, learning_rate=rate),
    )
    rows = pd.read_csv(public / "CURVE.csv", float_precision="round_trip").to_dict(
        "records"
    )
    with np.load(private / "PREDICTIONS.npz", allow_pickle=False) as values:
        if values["steps"].tolist() != [0, *cfg["snapshots"]]:
            raise RuntimeError(
                "Curve prediction snapshots differ from the registration"
            )
        prediction = values["prediction"]
    if [row["step"] for row in rows] != cfg["snapshots"] or len(prediction) != len(
        rows
    ) + 1:
        raise RuntimeError("Incomplete curve")
    return rows, prediction, receipt


def compile_results(args, cfg, parent, source, identity, data):
    raw, matched, receipts = [], [], []
    for fold, k in itertools.product(cfg["folds"], cfg["shots"]):
        reference = checkpoint(args, fold, cfg["seed"])
        episode = prepare_episode(args, cfg, parent, source, reference, data, fold, k)
        contract = episode_identity(identity, cfg, episode, reference, fold, k)
        for rate in cfg["learning_rates"]:
            abba, lora = [
                read_curve(args, cfg, fold, k, method, rate, contract)
                for method in METHODS
            ]
            baseline = abba[1][0].astype(np.float64)
            if not np.array_equal(baseline, lora[1][0]):
                raise RuntimeError(
                    "ABBA and LoRA must match the exact same baseline outputs"
                )
            raw.extend(abba[0] + lora[0])
            for method, result in zip(METHODS, (abba, lora), strict=True):
                public, _ = trajectory_paths(args, fold, k, method, rate)
                receipts.append(
                    {
                        "path": (public / "COMPLETE.json")
                        .relative_to(args.public_root)
                        .as_posix(),
                        "sha256": file_hash(public / "COMPLETE.json"),
                        "updates": result[2]["optimizer_updates"],
                        "training_seconds": result[0][-1]["training_seconds"],
                    }
                )
            for idx, (a, b) in enumerate(zip(abba[0], lora[0], strict=True), start=1):
                noise = max(a["initial_function_noise"], b["initial_function_noise"])
                radius, (alpha_a, alpha_b), resolvable = matched_radius(
                    a["function_radius"], b["function_radius"], noise
                )
                abba_scaled = score_episode(
                    baseline + alpha_a * (abba[1][idx] - baseline), episode, k
                )
                lora_scaled = score_episode(
                    baseline + alpha_b * (lora[1][idx] - baseline), episode, k
                )
                matched.append(
                    dict(
                        fold=fold,
                        K=k,
                        seed=cfg["seed"],
                        draw=cfg["draw"],
                        step=a["step"],
                        learning_rate=rate,
                        common_radius=radius,
                        resolvable_common_radius=resolvable,
                        initial_radius_noise_floor=noise,
                        abba_radius=a["function_radius"],
                        lora_radius=b["function_radius"],
                        abba_contraction=alpha_a,
                        lora_contraction=alpha_b,
                        raw_abba_minus_lora_query_mae=a["query_mae"] - b["query_mae"],
                        raw_abba_minus_lora_query_mse=a["query_mse"] - b["query_mse"],
                        matched_abba_minus_lora_query_mae=abba_scaled["query_mae"]
                        - lora_scaled["query_mae"],
                        matched_abba_minus_lora_query_mse=abba_scaled["query_mse"]
                        - lora_scaled["query_mse"],
                        **{
                            f"abba_matched_{name}": value
                            for name, value in abba_scaled.items()
                        },
                        **{
                            f"lora_matched_{name}": value
                            for name, value in lora_scaled.items()
                        },
                    )
                )
    if (
        len(receipts) != cfg["registered_trajectories"]
        or len(raw) != cfg["registered_snapshots"]
        or 2 * len(matched) != len(raw)
    ):
        raise RuntimeError("Incomplete registered optimization scale matrix")
    csv(args.public_root / "SCALE_CURVES.csv", raw)
    csv(args.public_root / "MATCHED_FUNCTION_SCALE.csv", matched)
    write_json(
        args.public_root / "E2_COMPLETE.json",
        {
            "identity": identity,
            "trajectories": len(receipts),
            "snapshots": len(raw),
            "matched_pairs": len(matched),
            "pairs_with_resolvable_radius": sum(
                row["resolvable_common_radius"] for row in matched
            ),
            "optimizer_updates": sum(row["updates"] for row in receipts),
            "training_seconds": sum(row["training_seconds"] for row in receipts),
            "receipts": receipts,
            "result_sha256": {
                name: file_hash(args.public_root / name)
                for name in ("SCALE_CURVES.csv", "MATCHED_FUNCTION_SCALE.csv")
            },
            "target_label_rows_loaded": 0,
        },
    )
    print(
        f"E2 complete: {len(receipts)} fits, {len(raw)} snapshots, {len(matched)} matched pairs",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("run", "summarize"), default="run")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--public-root", type=Path, default=PUBLIC)
    parser.add_argument("--private-root", type=Path, default=PRIVATE)
    parser.add_argument("--reference-root", type=Path, default=REFERENCES)
    parser.add_argument("--backbone-dir", type=Path, default=BACKBONE)
    parser.add_argument("--data-root", type=Path, default=CACHE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--folds", type=int, nargs="+")
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    configure()
    cfg, parent, source, identity = registered(args)
    args.folds = args.folds or cfg["folds"]
    if not set(args.folds) <= set(cfg["folds"]) or len(set(args.folds)) != len(
        args.folds
    ):
        raise ValueError("Fold filters may only narrow the registered source matrix")
    data = load_prefixes("MATR", args.data_root)
    with threadpool_limits(limits=2):
        if args.phase == "run":
            execute(args, cfg, parent, source, identity, data)
        else:
            compile_results(args, cfg, parent, source, identity, data)


if __name__ == "__main__":
    main()
