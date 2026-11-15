"""Run the single, complete M4 methodology on the six frozen MATR splits.

Engineering smoke and formal optimization have separate contracts and states.
The original seven-arm entry point and its registered artifacts stay read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))
from scripts.unified_matr import train_seed42 as baseline
from scripts.unified_matr.train_seed42 import (
    capture_random_state, execution_lock, fold_scaler, load_frozen_data,
    masked_cell_mae, predictions, restore_random_state, same_selection,
    save_checkpoint, seed_everything, sha256, to_batch, trajectory_metrics,
    write_history, write_json,
)

ARM = "M4_FULL_METHOD"
SPLITS = tuple(f"cell_fold_{number}" for number in range(5)) + ("unseen_protocol_sensitivity",)
DEPENDENCIES = (
    "models/unified_matr/__init__.py",
    "models/unified_matr/model.py",
    "models/unified_matr/legacy_operators.py",
    "models/unified_matr/methodology_full.py",
    "models/unified_matr/methodology_operators.py",
    "scripts/run_experiment_b.py",
    "scripts/unified_matr/train_seed42.py",
    "scripts/unified_matr/train_methodology_full.py",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def make_model(inputs, settings):
    from models.unified_matr.methodology_full import MethodologyFullModel

    return MethodologyFullModel(
        policy_dim=inputs["policy"].shape[-1],
        global_dim=inputs["global_features"].shape[-1],
        local_feature_dim=inputs["local_features"].shape[-1], seed=42,
        observation_bound=settings["observation_residual_amplitude"],
        soh_bounds=tuple(settings["observation_soh_bounds"]),
    )


def configure_execution():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False


def model_contract(data, data_root, config_path, methodology_path):
    from models.unified_matr.model import state_hash
    from models.unified_matr.methodology_full import model_registry

    config = read_json(config_path)
    baseline.validate_config(config)
    if sha256(config_path) not in {row["sha256"] for row in data[-1]["files"]}:
        raise RuntimeError("The reused training settings are outside the frozen data fingerprint")
    splits = data[-2]["splits"]
    if tuple(splits) != SPLITS:
        raise RuntimeError("M4 requires the original ordered six splits")
    inputs, identifiers, eligible = data[:3]
    if len(identifiers) != 169 or not eligible.all():
        raise RuntimeError("The full methodology contract fixes the eligible MATR169 cohort")
    seed_everything()
    model = make_model(inputs, config["training"])
    initial = {"arm": ARM, "seed": 42,
               "parameters": sum(p.numel() for p in model.parameters()),
               "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
               "initial_state_sha256": state_hash(model),
               "policy_dim": inputs["policy"].shape[-1],
               "global_dim": inputs["global_features"].shape[-1],
               "local_feature_dim": inputs["local_features"].shape[-1]}
    updates = {name: 150 * math.ceil(len(roles["train"]) / 8) for name, roles in splits.items()}
    if sum(updates.values()) != 12450:
        raise RuntimeError("The fixed six-unit optimizer budget must be 12450 updates")
    return {
        "study_id": "unified_matr_methodology_full_seed42", "arm": ARM, "seed": 42,
        "planned_units": 6, "planned_optimizer_updates": sum(updates.values()),
        "expected_updates_by_split": updates,
        "methodology_contract": {"path": str(methodology_path.relative_to(REPOSITORY)).replace("\\", "/"),
                                 "sha256": sha256(methodology_path)},
        "config_sha256": sha256(config_path),
        "data_freeze_sha256": sha256(data_root / "DATA_FREEZE.json"),
        "data_fingerprint_sha256": sha256(data_root / "dataset_fingerprint.sha256"),
        "split_contract_sha256": sha256(data_root / "split_contract.json"),
        "frozen_files": data[-1]["files"],
        "sources": [{"path": name, "sha256": sha256(REPOSITORY / name)} for name in DEPENDENCIES],
        "initialization": {"models": {ARM: initial}},
        "model_implementation_registry": model_registry(model),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "cuda_runtime": torch.version.cuda,
                        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                        "cuda_matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                        "torch_num_threads": torch.get_num_threads(), "precision": "float32"},
        "training": config["training"], "reporting": config["reporting"],
        "config": {**config, "study_id": "unified_matr_methodology_full_seed42", "models": [ARM]},
        "computational_graph": {
            "I1": "frozen robust a_Ah/b_V/residual64 plus PhysicalResponseEncoder global/local routes",
            "I2": "past-only response-conditioned dynamic persistence and valid-only actual history B modulation",
            "I3": "nonnegative effective rate per1000 cycles integrated on fixed physical unit-cycle grid",
            "inputs": list(baseline.INPUT_KEYS), "state_dim": 16, "condition_dim": 16,
            "rho_bounds": [0.01, 0.99], "observation_amplitude": 0.03,
            "rho_network_dimensions": [172, 64, 64],
            "rho_final_weight_initialization_std": 0.001,
            "rho_final_bias_initialization": "logit(0.9)",
            "rho_context": "phi64 + previous_global20 + previous_global_mask20 + quality32 + policy15 + policy_mask15 + gap1 + seen1 + robust_a_b_drift2 + drift_mask2",
            "robust_global_drift_columns": [2, 11],
            "quality_dimensions": "local_residual_valid_fraction8 + local_feature_valid_fraction8 + scaled_noise_proxy8 + noise_mask8",
            "local_noise_feature_column": 6,
            "write_gate_dimensions": [96, 16],
            "write_gate_context": "persistent_state64 + quality32",
            "write_projection_dimensions": [64, 16],
            "write_projection_initialization": "exact_zero_weight_without_bias",
            "history_routing": "Both history mixers modify post-convolution B and then call dense_ssd with Bprime",
            "cycle_gap": "actual current cycle minus previous valid cycle, with first valid initialization and invalid hold",
            "persistence_penalty": "mean squared persistent-state change per actual cycle over valid transitions and64 channels",
            "persistence_penalty_coefficient": 0.001,
            "precision": "float32", "fresh_formal_initialization_after_smoke": True,
        },
        "historical_exposure": "All169 cells have been exposed in earlier exploratory experiments. This is internal single-seed evidence.",
        "full_duplicate_training_planned": True,
        "full_duplicate_training_scope": {"split": "cell_fold_0", "repeats_after_formal": 1,
                                          "epochs": 150, "additional_optimizer_updates": 2100,
                                          "role": "complete fresh-seed42 training reproduction",
                                          "included_in_formal_12450_updates": False},
        "full_duplicate_training_limitation": "One complete fold0 repeat checks repeatability under the registered environment. Cross-seed variability remains unmeasured.",
    }


def bind_contract(path, value):
    if path.exists() and read_json(path) != value:
        raise RuntimeError("Existing immutable contract differs at " + str(path))
    write_json(path, value)


def training_loss(output, rows, targets, observed, rate_targets, rate_mask, settings, device):
    mask = torch.as_tensor(observed[rows], device=device)
    target = torch.as_tensor(targets[rows], device=device)
    trajectory_loss = masked_cell_mae(output["trajectory"], target, mask)
    rate_prediction = torch.stack((
        (output["trajectory"][:, 200 - 101] - output["trajectory"][:, 500 - 101]) / 300,
        (output["trajectory"][:, 500 - 101] - output["trajectory"][:, 1000 - 101]) / 500,
    ), dim=1)
    auxiliary_mask = torch.as_tensor(rate_mask[rows], device=device)
    auxiliary_count = auxiliary_mask.sum(dim=1)
    auxiliary_loss = trajectory_loss.new_zeros(())
    if auxiliary_mask.any():
        auxiliary_error = torch.abs(rate_prediction - torch.as_tensor(rate_targets[rows], device=device))
        auxiliary_cell = (auxiliary_error * auxiliary_mask).sum(dim=1) / auxiliary_count.clamp_min(1)
        auxiliary_loss = auxiliary_cell[auxiliary_count > 0].mean() * settings["auxiliary_rate_scale"]
    loss = trajectory_loss + settings["auxiliary_rate_loss_weight"] * auxiliary_loss
    loss = loss + settings["persistence_soft_penalty_weight"] * output["persistence_penalty"]
    return loss


def flatten_tensors(value, prefix=""):
    rows = {}
    if isinstance(value, torch.Tensor):
        rows[prefix] = value.detach().cpu().numpy()
    elif isinstance(value, dict):
        for key, item in value.items():
            rows.update(flatten_tensors(item, prefix + "__" + str(key) if prefix else str(key)))
    elif isinstance(value, (list, tuple)):
        for number, item in enumerate(value):
            rows.update(flatten_tensors(item, prefix + "__" + str(number)))
    return rows


def mechanism_diagnostics(model, inputs, rows, identifiers, query, device, batch_size, root):
    """Store full selected-model diagnostic tensors plus per-cell summaries."""
    model.eval()
    collected = {}
    summary_rows = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            part = rows[start:start + batch_size]
            batch = to_batch(inputs, part, device)
            output = model(**batch, query_cycles=torch.as_tensor(query, device=device), return_diagnostics=True)
            diagnostic = flatten_tensors(output["diagnostics"])
            for key, value in diagnostic.items():
                if value.ndim and value.shape[0] == len(part):
                    collected.setdefault(key, []).append(value)
            sparse_query = np.array([100, 101, 200, 500, 1000, 2238, 5000], dtype=np.float32)
            sparse = model(**batch, query_cycles=torch.as_tensor(sparse_query, device=device))
            dense_query = np.arange(100, 5001, dtype=np.float32)
            dense = model(**batch, query_cycles=torch.as_tensor(dense_query, device=device))
            query_slots = torch.as_tensor(sparse_query.astype(int) - 100, device=device)
            consistency = {}
            for key in ("trend", "raw_trend", "integrated_loss"):
                selected = dense[key][:, query_slots]
                consistency[key] = (selected - sparse[key]).abs().amax(dim=1).cpu().numpy()
            rate = dense["rate_per_1000_cycles"].cpu().numpy()
            trend = dense["trend"].cpu().numpy()
            trajectory = output["trajectory"].cpu().numpy()
            persistence = output["diagnostics"]["persistence"]
            valid = persistence["valid"].cpu().numpy().astype(bool)
            rho = persistence["rho"].cpu().numpy()
            write = persistence["write_weight"].cpu().numpy()
            gap = persistence["gap"].cpu().numpy()
            drift = persistence["drift_per_cycle"].cpu().numpy()
            for local, number in enumerate(part):
                use = valid[local]
                record = {
                    "cell_uid": identifiers[number], "valid_prefix_cycles": int(use.sum()),
                    "rho_min": float(rho[local][use].min()), "rho_max": float(rho[local][use].max()),
                    "rho_mean": float(rho[local][use].mean()), "write_mean": float(write[local][use].mean()),
                    "max_actual_cycle_gap": float(gap[local][use].max()),
                    "drift_mean_abs_per_cycle": float(np.abs(drift[local][use]).mean()),
                    "drift_positive_fraction": float((drift[local][use] > 0).mean()),
                    "rate_min_per1000": float(rate[local].min()), "rate_max_per1000": float(rate[local].max()),
                    "trend_min": float(trend[local].min()), "trend_max": float(trend[local].max()),
                    "trajectory_min": float(trajectory[local].min()), "trajectory_max": float(trajectory[local].max()),
                    "trend_increase_count_over_1e_6": int((np.diff(trend[local]) > 1e-6).sum()),
                    "anchor_abs_error": float(abs(trend[local, 0] - inputs["anchor_soh100"][number])),
                    "cross_query_max_abs_error": float(consistency["trend"][local]),
                    "cross_query_raw_trend_max_abs_error": float(consistency["raw_trend"][local]),
                    "cross_query_integrated_loss_max_abs_error": float(consistency["integrated_loss"][local]),
                    "raw_trend_min": float(dense["raw_trend"][local].min()),
                    "raw_trend_max": float(dense["raw_trend"][local].max()),
                    "raw_trend_increase_count_over_1e_6": int((np.diff(dense["raw_trend"][local].cpu().numpy()) > 1e-6).sum()),
                }
                robust_mask = persistence["robust_drift_mask"][local].cpu().numpy()
                robust_drift = persistence["robust_drift_per_cycle"][local].cpu().numpy()
                directional = persistence["drift_direction_consistency"][local].cpu().numpy()
                record["robust_drift_supported_coordinates"] = int(robust_mask.sum())
                record["robust_drift_direction_consistency_mean"] = float(directional[robust_mask].mean()) if robust_mask.any() else 0.0
                record["robust_drift_positive_fraction"] = float((robust_drift[robust_mask] > 0).mean()) if robust_mask.any() else 0.0
                for layer, history in enumerate(output["diagnostics"]["history"]):
                    original = history["B"][local].cpu().numpy()
                    altered = history["Bprime"][local].cpu().numpy()
                    delta = altered[use] - original[use]
                    record[f"history{layer}_B_l2_mean"] = float(np.linalg.norm(original[use], axis=-1).mean())
                    record[f"history{layer}_Bprime_l2_mean"] = float(np.linalg.norm(altered[use], axis=-1).mean())
                    record[f"history{layer}_B_delta_l2_mean"] = float(np.linalg.norm(delta, axis=-1).mean())
                summary_rows.append(record)
    saved = {key: np.concatenate(parts, axis=0) for key, parts in collected.items()}
    saved["cell_uid"] = np.asarray([identifiers[number] for number in rows])
    np.savez_compressed(root / "test_mechanism_diagnostics.npz", **saved)
    pd.DataFrame(summary_rows).to_csv(root / "test_mechanism_summary.csv", index=False)


def gradient_receipt(model):
    gradients = {}
    joined = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        gradient = parameter.grad
        raw = b"NONE" if gradient is None else gradient.detach().cpu().contiguous().numpy().tobytes()
        digest = hashlib.sha256(raw).hexdigest()
        joined.update(name.encode())
        joined.update(raw)
        gradients[name] = {"sha256": digest, "present": gradient is not None,
                           "finite": gradient is None or bool(torch.isfinite(gradient).all()),
                           "l2": None if gradient is None else float(gradient.detach().norm())}
    return {"sha256": joined.hexdigest(), "parameters": gradients}


def real_smoke(data, config, registration, output_root, device):
    inputs, identifiers, eligible, _, query, targets, observed, rate_targets, rate_mask, splits, _ = data
    split_name = SPLITS[0]
    roles = splits["splits"][split_name]
    index = {uid: number for number, uid in enumerate(identifiers)}
    training = np.asarray([index[uid] for uid in roles["train"] if eligible[index[uid]]])
    scaled, _ = fold_scaler(inputs, training)
    rows = np.random.default_rng(42).permutation(training)[:8]
    repeats = []
    initial_B_delta = []
    for repeat in range(2):
        seed_everything()
        model = make_model(inputs, config["training"]).to(device)
        model.train()
        batch = to_batch(scaled, rows, device)
        output = model(**batch, query_cycles=torch.as_tensor(query, device=device), return_diagnostics=True)
        loss = training_loss(output, rows, targets, observed, rate_targets, rate_mask, config["training"], device)
        loss.backward()
        gradients = gradient_receipt(model)
        history = output["diagnostics"]["history"]
        initial_B_delta.append([float((item["Bprime"] - item["B"]).abs().max()) for item in history])
        repeats.append({"loss": float(loss.detach()),
                        "prediction_sha256": hashlib.sha256(output["trajectory"].detach().cpu().numpy().tobytes()).hexdigest(),
                        "gradient": gradients})
        del output, model
    exact = repeats[0] == repeats[1]
    write_gradients = {name: item for name, item in repeats[0]["gradient"]["parameters"].items()
                       if "write_projection" in name or "W_B" in name or "write_to_b" in name}
    checks = {
        "fresh_seed42_forward_backward_exact": exact,
        "all_gradients_finite": all(item["finite"] for item in repeats[0]["gradient"]["parameters"].values()),
        "initial_B_modulation_zero": all(value == 0 for values in initial_B_delta for value in values),
        "B_write_projection_gradient_nonzero": bool(write_gradients) and all(
            item["present"] and item["l2"] > 0 for item in write_gradients.values()),
    }
    smoke_root = output_root / "engineering_smoke"
    smoke_root.mkdir(parents=True, exist_ok=True)
    write_json(smoke_root / "FIRST_BATCH_REPRODUCIBILITY.json", {
        "checks": checks, "cell_uid": [identifiers[number] for number in rows],
        "observed_targets": int(observed[rows].sum()), "rate_target_support": rate_mask[rows].sum(0).tolist(),
        "initial_B_delta_max": initial_B_delta, "repeats": repeats,
        "W_B_gradients": write_gradients, "optimizer_updates": 0,
    })
    if not all(checks.values()):
        raise RuntimeError("First-batch M4 reproducibility/gradient smoke failed")
    smoke_registration = {**registration, "phase": "engineering_smoke", "planned_units": 1, "epochs": 1}
    bind_contract(smoke_root / "RUN_REGISTRATION.json", smoke_registration)
    smoke_config = {**config, "training": {**config["training"], "epochs": 1}}
    train_unit(ARM, split_name, roles, data, smoke_config, None, smoke_root, device,
               smoke_registration, evaluate_test=False)
    smoke_files = [path for path in smoke_root.rglob("*") if path.is_file() and not path.name.startswith(".")]
    receipt = {
        "status": "PASS", "model_contract_sha256": sha256(output_root / "MODEL_CONTRACT.json"),
        "checks": checks, "engineering_epochs": 1,
        "engineering_optimizer_updates": math.ceil(len(training) / 8),
        "formal_optimizer_updates": 0, "test_observations_consumed": 0,
        "fresh_formal_seed42_initialization_required": True,
        "first_batch_reproducibility": "FIRST_BATCH_REPRODUCIBILITY.json",
        "files": [{"path": str(path.relative_to(output_root)).replace("\\", "/"), "sha256": sha256(path)} for path in sorted(smoke_files)],
        "model_mathematical_checks": "External engineering controls must also be reviewed before formal launch.",
    }
    write_json(output_root / "SMOKE_QA.json", receipt)
    return receipt


def train_unit(arm, split_name, roles, data, config, data_root, run_root, device, registration, evaluate_test=True):
    from models.unified_matr.model import state_hash

    inputs, identifiers, eligible, cells, query, targets, observed, rate_targets, rate_mask, _, _ = data
    identifier_index = {uid: number for number, uid in enumerate(identifiers)}
    indices = {role: np.array([identifier_index[uid] for uid in roles[role]
                              if eligible[identifier_index[uid]]], dtype=int)
               for role in ("train", "validation", "test")}
    for role, rows in indices.items():
        if not len(rows) or not observed[rows].sum():
            raise RuntimeError("No observed supervision in split role " + role)
    unit_root = run_root / split_name / arm
    unit_root.mkdir(parents=True, exist_ok=True)
    unit_path = unit_root / "UNIT_CONTRACT.json"
    unit_contract = {"arm": arm, "seed": 42, "split_name": split_name,
                     "registration_sha256": sha256(run_root / "RUN_REGISTRATION.json"),
                     "roles": {role: [identifiers[i] for i in rows] for role, rows in indices.items()},
                     "training": config["training"],
                     "expected_updates": config["training"]["epochs"] * math.ceil(len(indices["train"]) / config["training"]["batch_size"]),
                     "initial_model": registration["initialization"]["models"][arm], "device": str(device)}
    if unit_path.exists() and json.loads(unit_path.read_text(encoding="utf-8")) != unit_contract:
        raise RuntimeError("Existing unit has a different frozen contract")
    write_json(unit_path, unit_contract)
    complete_path = unit_root / "COMPLETE.json"
    if complete_path.exists():
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if complete.get("status") != "EVALUATED" or complete.get("successful_updates") != unit_contract["expected_updates"]:
            raise RuntimeError("Completed unit has an inconsistent evaluation state or update budget")
        for item in complete["files"]:
            if sha256(unit_root / item["path"]) != item["sha256"]:
                raise RuntimeError("Completed unit artifact mismatch")
        print(json.dumps({"status": "VERIFIED_COMPLETE", "split": split_name, "arm": arm}), flush=True)
        return
    scaled, scaler = fold_scaler(inputs, indices["train"])
    write_json(unit_root / "TRAIN_SCALER.json", {"training_ids": unit_contract["roles"]["train"], "statistics": scaler})
    seed_everything()
    model = make_model(inputs, config["training"]).to(device)
    if state_hash(model) != registration["initialization"]["models"][arm]["initial_state_sha256"]:
        raise RuntimeError("Actual training model differs from the frozen common initialization")
    settings = config["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                 weight_decay=settings["weight_decay"], betas=tuple(settings["betas"]),
                                 eps=settings["eps"])
    checkpoint = unit_root / "TRAIN_STATE.pt"
    best_path = unit_root / "SELECTED.pt"
    epoch_start, best_mae, successful_updates = 0, math.inf, 0
    history, selected_checkpoint = [], None
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        if state["contract"] != unit_contract:
            raise RuntimeError("Resume checkpoint contract mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        epoch_start, best_mae, successful_updates = state["next_epoch"], state["best_mae"], state["successful_updates"]
        history, selected_checkpoint = state["history"], state["selected_checkpoint"]
        expected_updates = epoch_start * math.ceil(len(indices["train"]) / settings["batch_size"])
        if len(history) != epoch_start or successful_updates != expected_updates:
            raise RuntimeError("Resume checkpoint has an incomplete epoch or optimizer-update history")
        restore_random_state(state["random_state"])
        actual = torch.load(best_path, map_location="cpu", weights_only=False) if best_path.exists() else None
        if actual is None or not same_selection(actual, selected_checkpoint):
            if (unit_root / "SELECTION_FROZEN.json").exists():
                raise RuntimeError("A frozen selected checkpoint differs from the atomic training state")
            save_checkpoint(best_path, selected_checkpoint)
        write_history(unit_root / "training.jsonl", history)
    tensor_query = torch.as_tensor(query, device=device)
    started = time.time() - (history[-1]["elapsed_seconds"] if history else 0)
    for epoch in range(epoch_start, settings["epochs"]):
        model.train()
        epoch_rng = np.random.default_rng(42 + epoch)
        order = epoch_rng.permutation(indices["train"])
        losses = []
        for start in range(0, len(order), settings["batch_size"]):
            rows = order[start:start + settings["batch_size"]]
            batch = to_batch(scaled, rows, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch, query_cycles=tensor_query)
            loss = training_loss(output, rows, targets, observed, rate_targets, rate_mask, settings, device)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            successful_updates += 1
            losses.append(float(loss.detach()))
        val_prediction, _ = predictions(model, scaled, indices["validation"], query, device, settings["batch_size"])
        val_metrics, _, _ = trajectory_metrics(val_prediction, targets[indices["validation"]], observed[indices["validation"]])
        if not np.isfinite(val_metrics["cell_mean_mae"]):
            raise FloatingPointError("Validation selection criterion is nonfinite")
        selected = val_metrics["cell_mean_mae"] < best_mae
        if selected:
            best_mae = val_metrics["cell_mean_mae"]
            selected_checkpoint = {"model": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                                   "epoch": epoch + 1, "validation_mae": best_mae, "contract": unit_contract}
            save_checkpoint(best_path, selected_checkpoint)
        record = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                  "validation_mae": val_metrics["cell_mean_mae"], "selected": selected,
                  "successful_updates": successful_updates, "elapsed_seconds": time.time() - started}
        history.append(record)
        save_checkpoint(checkpoint, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                    "next_epoch": epoch + 1, "best_mae": best_mae,
                                    "successful_updates": successful_updates, "contract": unit_contract,
                                    "random_state": capture_random_state(), "history": history,
                                    "selected_checkpoint": selected_checkpoint})
        write_history(unit_root / "training.jsonl", history)
        if (epoch + 1) % 10 == 0:
            print(json.dumps({"split": split_name, "arm": arm, **record}), flush=True)
    if successful_updates != unit_contract["expected_updates"]:
        raise RuntimeError("Final optimizer updates differ from the fixed epoch/batch budget")
    selected = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(selected["model"])
    # Freeze the chosen weights and validation criterion before any test prediction.
    selection = {"selected_epoch": selected["epoch"],
               "validation_mae": selected["validation_mae"], "checkpoint_sha256": sha256(best_path),
               "successful_updates": successful_updates, "seed": 42,
               "parameters": sum(parameter.numel() for parameter in model.parameters()),
               "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)}
    selection_path = unit_root / "SELECTION_FROZEN.json"
    if selection_path.exists() and json.loads(selection_path.read_text(encoding="utf-8")) != selection:
        raise RuntimeError("Test-exposed checkpoint selection is immutable")
    write_json(selection_path, selection)
    if not evaluate_test:
        write_json(unit_root / "ENGINEERING_EPOCH_COMPLETE.json", {
            "status": "ENGINEERING_ONLY", "epochs": settings["epochs"],
            "successful_updates": successful_updates, "validation_mae": selected["validation_mae"],
            "test_evaluation_performed": False, "selected_checkpoint_sha256": sha256(best_path),
        })
        return
    test = indices["test"]
    test_prediction, test_trend = predictions(model, scaled, test, query, device, settings["batch_size"])
    metrics, per_cell_mae, per_cell_rmse = trajectory_metrics(test_prediction, targets[test], observed[test])
    rows = []
    for position, number in enumerate(test):
        uid = identifiers[number]
        valid = observed[number]
        for slot in np.flatnonzero(valid):
            rows.append({"cell_uid": uid, "original_cycle_id": int(query[slot]), "target": float(targets[number, slot]),
                         "prediction": float(test_prediction[position, slot]), "trend": float(test_trend[position, slot])})
    pd.DataFrame(rows).to_parquet(unit_root / "test_predictions.parquet", index=False)
    per_cell = [{"cell_uid": identifiers[number], "mae": float(per_cell_mae[position]),
                 "rmse": float(per_cell_rmse[position]), "observations": int(observed[number].sum()),
                 "protocol_uid": str(cells.loc[identifiers[number], "protocol_uid"])}
                for position, number in enumerate(test) if observed[number].any()]
    pd.DataFrame(per_cell).to_csv(unit_root / "test_cell_metrics.csv", index=False)
    rate_rows = []
    query_index = {int(cycle): position for position, cycle in enumerate(query)}
    for position, number in enumerate(test):
        for window, (left, right) in enumerate(((200, 500), (500, 1000))):
            first = float(test_prediction[position, query_index[left]])
            last = float(test_prediction[position, query_index[right]])
            rate_rows.append({"cell_uid": identifiers[number], "protocol_uid": str(cells.loc[identifiers[number], "protocol_uid"]),
                              "target_name": f"r{left}_{right}", "start_cycle": left, "end_cycle": right,
                              "predicted_start_soh": first, "predicted_end_soh": last,
                              "predicted_rate": (first - last) / (right - left),
                              "target_supported": bool(rate_mask[number, window]),
                              "target_rate": float(rate_targets[number, window]) if rate_mask[number, window] else None})
    pd.DataFrame(rate_rows).to_csv(unit_root / "test_rate_predictions.csv", index=False)
    eol_limit = config["reporting"]["eol80_prediction_query_max_cycle"]
    eol_queries = np.arange(100, eol_limit + 1, dtype=np.float32)
    _, eol_trends = predictions(model, scaled, test, eol_queries, device, settings["batch_size"])
    eol_rows = []
    for position, number in enumerate(test):
        crossings = np.flatnonzero(eol_trends[position] <= 0.8)
        predicted_cycle = float(eol_limit)
        if len(crossings):
            right = int(crossings[0])
            predicted_cycle = float(eol_queries[right])
            if right > 0:
                y0, y1 = map(float, eol_trends[position, right - 1:right + 1])
                if y0 > y1:
                    predicted_cycle = float(eol_queries[right - 1] + (y0 - 0.8) / (y0 - y1))
        eol_rows.append({"cell_uid": identifiers[number], "predicted_trend_eol80_cycle": predicted_cycle,
                         "no_crossing_through_registered_horizon": not bool(len(crossings)),
                         "prediction_query_max_cycle": eol_limit})
    pd.DataFrame(eol_rows).to_csv(unit_root / "test_eol_predictions.csv", index=False)
    write_json(unit_root / "test_metrics.json", {**metrics, "arm": arm, "split": split_name, "seed": 42})
    mechanism_diagnostics(model, scaled, test, identifiers, query, device, settings["batch_size"], unit_root)
    files = ["test_mechanism_diagnostics.npz", "test_mechanism_summary.csv", "UNIT_CONTRACT.json", "TRAIN_SCALER.json", "SELECTION_FROZEN.json", "training.jsonl",
             "test_predictions.parquet", "test_cell_metrics.csv", "test_metrics.json", "test_eol_predictions.csv", "test_rate_predictions.csv",
             "SELECTED.pt", "TRAIN_STATE.pt"]
    write_json(complete_path, {"status": "EVALUATED", "seed": 42, "successful_updates": successful_updates, "files": [
        {"path": name, "sha256": sha256(unit_root / name)} for name in files]})
    print(json.dumps({"status": "EVALUATED", "split": split_name, "arm": arm, **metrics}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("register", "smoke", "train", "reproduce"), required=True)
    parser.add_argument("--config", type=Path, default=REPOSITORY / "configs/unified_matr_seed42.json")
    parser.add_argument("--methodology-contract", type=Path,
                        default=REPOSITORY / "research/M4_FULL_METHODOLOGY_RUN_ONLY_ZH.md")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.data = args.data.resolve()
    args.output_root = args.output_root.resolve()
    args.config = args.config.resolve()
    args.methodology_contract = args.methodology_contract.resolve()
    configure_execution()
    config = read_json(args.config)
    data = load_frozen_data(args.data)
    registration = model_contract(data, args.data, args.config, args.methodology_contract)
    args.output_root.mkdir(parents=True, exist_ok=True)
    with execution_lock(args.output_root / ".methodology.lock"):
        bind_contract(args.output_root / "MODEL_CONTRACT.json", registration)
        if args.mode == "register":
            print(json.dumps({"status": "REGISTERED", "units": 6, "planned_updates": 12450}), flush=True)
            return
        if args.mode == "smoke":
            if any((args.output_root / "runs").glob("*/" + ARM + "/TRAIN_STATE.pt")):
                raise RuntimeError("Engineering smoke cannot restart after formal optimization has begun")
            real_smoke(data, config, registration, args.output_root, args.device)
            print(json.dumps({"status": "SMOKE_PASS", "formal_optimizer_updates": 0}), flush=True)
            return
        smoke = read_json(args.output_root / "SMOKE_QA.json")
        if (smoke.get("status") != "PASS" or smoke.get("model_contract_sha256") != sha256(args.output_root / "MODEL_CONTRACT.json")
                or not all(smoke["checks"].values())):
            raise RuntimeError("The current registered model requires its separate passing engineering smoke")
        for entry in smoke["files"]:
            if sha256(args.output_root / entry["path"]) != entry["sha256"]:
                raise RuntimeError("Engineering smoke artifact mismatch")
        run_root = args.output_root / "runs"
        run_root.mkdir(parents=True, exist_ok=True)
        bind_contract(run_root / "RUN_REGISTRATION.json", registration)
        if args.mode == "reproduce":
            matrix = read_json(run_root / "MATRIX_COMPLETE.json")
            if (matrix.get("status") != "COMPLETE" or matrix.get("completed_units") != 6
                    or matrix.get("successful_updates") != 12450
                    or matrix.get("registration_sha256") != sha256(run_root / "RUN_REGISTRATION.json")):
                raise RuntimeError("The full fold0 reproduction follows all six completed formal fits")
            for split_name in SPLITS:
                unit_root = run_root / split_name / ARM
                complete = read_json(unit_root / "COMPLETE.json")
                if complete.get("status") != "EVALUATED":
                    raise RuntimeError("Formal unit evaluation is incomplete")
                for item in complete["files"]:
                    if sha256(unit_root / item["path"]) != item["sha256"]:
                        raise RuntimeError("Formal artifact changed before complete training reproduction")
            reproduction = args.output_root / "training_reproduction"
            reproduction.mkdir(parents=True, exist_ok=True)
            bind_contract(reproduction / "RUN_REGISTRATION.json", registration)
            train_unit(ARM, SPLITS[0], data[-2]["splits"][SPLITS[0]], data,
                       config, args.data, reproduction, args.device, registration)
            write_json(reproduction / "REPRODUCTION_COMPLETE.json", {
                "status": "EVALUATED", "split": SPLITS[0], "arm": ARM, "seed": 42,
                "additional_optimizer_updates": 2100, "formal_optimizer_updates": 12450,
                "comparison_status": "pending_independent_full_state_and_history_comparison",
                "registration_sha256": sha256(reproduction / "RUN_REGISTRATION.json"),
            })
            return
        for split_name in SPLITS:
            train_unit(ARM, split_name, data[-2]["splits"][split_name], data,
                       config, args.data, run_root, args.device, registration)
        write_json(run_root / "MATRIX_COMPLETE.json", {
            "status": "COMPLETE", "seed": 42, "arm": ARM, "completed_units": 6,
            "successful_updates": 12450, "registration_sha256": sha256(run_root / "RUN_REGISTRATION.json"),
            "model_contract_sha256": sha256(args.output_root / "MODEL_CONTRACT.json"),
        })


if __name__ == "__main__":
    main()

