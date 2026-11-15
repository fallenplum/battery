"""Diagnose registered first-batch repeatability without optimizer updates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(REPOSITORY / "scripts/unified_matr"))
import train_seed42 as training
from models.unified_matr import UnifiedMATRModel
from models.unified_matr.model import state_hash


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def tensor_digest(values):
    checksum = hashlib.sha256()
    for name, value in sorted(values.items()):
        array = value.detach().cpu().contiguous().numpy()
        checksum.update(name.encode("utf-8"))
        checksum.update(str(array.shape).encode("ascii"))
        checksum.update(str(array.dtype).encode("ascii"))
        checksum.update(array.tobytes())
    return checksum.hexdigest()


def rng_fingerprint():
    state = training.capture_random_state()
    values = {"cpu": state["torch_cpu"]}
    values.update({f"cuda_{number}": value for number, value in enumerate(state["torch_cuda"])})
    return tensor_digest(values)


def compare_tensors(first, last):
    if first.keys() != last.keys():
        raise RuntimeError("Compared tensor identities differ")
    rows = []
    for name, left in first.items():
        right = last[name]
        if left is None or right is None:
            rows.append({"name": name, "exact_equal": left is None and right is None,
                         "both_missing": left is None and right is None, "max_absolute_difference": None})
            continue
        difference = (left.to(torch.float64) - right.to(torch.float64)).abs()
        rows.append({"name": name, "exact_equal": torch.equal(left, right),
                     "different_elements": int(torch.count_nonzero(left != right)),
                     "elements": left.numel(), "max_absolute_difference": float(difference.max()),
                     "left_max_absolute_value": float(left.abs().max()),
                     "right_max_absolute_value": float(right.abs().max())})
    return {"exact_equal": all(row["exact_equal"] for row in rows),
            "max_absolute_difference": max((row["max_absolute_difference"] or 0.0) for row in rows),
            "tensors": rows}


def objective(output, targets, observed, rate_targets, rate_mask, settings):
    trajectory_loss = training.masked_cell_mae(output["trajectory"], targets, observed)
    prediction = torch.stack((
        (output["trajectory"][:, 200 - 101] - output["trajectory"][:, 500 - 101]) / 300,
        (output["trajectory"][:, 500 - 101] - output["trajectory"][:, 1000 - 101]) / 500,
    ), dim=1)
    count = rate_mask.sum(dim=1)
    auxiliary_loss = trajectory_loss.new_zeros(())
    if rate_mask.any():
        error = torch.abs(prediction - rate_targets)
        cell_error = (error * rate_mask).sum(dim=1) / count.clamp_min(1)
        auxiliary_loss = cell_error[count > 0].mean() * settings["auxiliary_rate_scale"]
    penalty = settings["persistence_soft_penalty_weight"] * output["persistence_penalty"]
    total = trajectory_loss + settings["auxiliary_rate_loss_weight"] * auxiliary_loss + penalty
    return total, {"trajectory_loss": trajectory_loss, "scaled_rate_loss": auxiliary_loss,
                   "weighted_persistence_penalty": penalty, "total_loss": total}


def run_case(model, batch, query, target, observed, rate_target, rate_mask, settings, rng_state):
    training.restore_random_state(rng_state)
    model.train()
    model.zero_grad(set_to_none=True)
    before = rng_fingerprint()
    output = model(**batch, query_cycles=query, return_context=True)
    loss, components = objective(output, target, observed, rate_target, rate_mask, settings)
    loss.backward()
    if query.device.type == "cuda":
        torch.cuda.synchronize(query.device)
    after = rng_fingerprint()
    outputs = {name: output[name].detach().cpu().clone() for name in
               ("trajectory", "trend", "observation_residual", "persistence_penalty")}
    outputs.update({"context_" + name: output["context"][name].detach().cpu().clone()
                    for name in ("state", "condition", "raw_trend")})
    outputs.update({name: value.detach().cpu().clone() for name, value in components.items()})
    gradients = {name: parameter.grad.detach().cpu().clone() if parameter.grad is not None else None
                 for name, parameter in model.named_parameters()}
    finite = all(value is None or bool(torch.isfinite(value).all()) for value in gradients.values())
    if not finite or not all(bool(torch.isfinite(value).all()) for value in outputs.values()):
        raise FloatingPointError("A diagnostic output or gradient is nonfinite")
    return {"outputs": outputs, "gradients": gradients, "rng_before": before,
            "rng_after": after, "loss_components": {name: float(value) for name, value in components.items()}}


def artifact_integrity(registration, freeze, data_root):
    return {
        "registered_sources": all(digest(REPOSITORY / item["path"]) == item["sha256"]
                                  for item in registration["sources"]),
        "frozen_data": all(digest(data_root / item["path"]) == item["sha256"] for item in freeze["files"]),
    }


def diagnose(data_root, run_root, device):
    registration = read_json(run_root / "RUN_REGISTRATION.json")
    matrix = read_json(run_root / "MATRIX_COMPLETE.json")
    if matrix["status"] != "COMPLETE" or matrix["completed_units"] != 42 or registration["seed"] != 42:
        raise RuntimeError("The diagnostic requires the completed registered seed42 matrix")
    if matrix["registration_sha256"] != digest(run_root / "RUN_REGISTRATION.json"):
        raise RuntimeError("Matrix registration binding differs")
    data = training.load_frozen_data(data_root)
    inputs, ids, eligible, _, query, targets, observed, rate_targets, rate_mask, split, freeze = data
    before_integrity = artifact_integrity(registration, freeze, data_root)
    if not all(before_integrity.values()):
        raise RuntimeError("A registered or frozen artifact changed")
    roles = split["splits"]["cell_fold_0"]
    indices = np.asarray([ids.index(uid) for uid in roles["train"] if eligible[ids.index(uid)]], dtype=int)
    scaled, statistics = training.fold_scaler(inputs, indices)
    contracts = {arm: read_json(run_root / "cell_fold_0" / arm / "UNIT_CONTRACT.json") for arm in ("M2", "M3-NR")}
    scalers = {arm: read_json(run_root / "cell_fold_0" / arm / "TRAIN_SCALER.json") for arm in ("M2", "M3-NR")}
    if (contracts["M2"]["roles"] != contracts["M3-NR"]["roles"]
            or scalers["M2"] != scalers["M3-NR"]
            or statistics != scalers["M2"]["statistics"]):
        raise RuntimeError("Diagnostic inputs or scaler differ from the formal first batch")
    order = np.random.default_rng(42).permutation(indices)
    rows = order[:8]
    batch = training.to_batch(scaled, rows, device)
    tensor_query = torch.as_tensor(query, device=device)
    tensor_target = torch.as_tensor(targets[rows], device=device)
    tensor_observed = torch.as_tensor(observed[rows], device=device)
    tensor_rates = torch.as_tensor(rate_targets[rows], device=device)
    tensor_rate_mask = torch.as_tensor(rate_mask[rows], device=device)
    settings = registration["config"]["training"]
    models = {}
    hashes = {}
    for name, arm in (("M2_a", "M2"), ("M2_b", "M2"), ("NR", "M3-NR")):
        training.seed_everything()
        model = UnifiedMATRModel(arm, inputs["policy"].shape[-1], inputs["global_features"].shape[-1],
                                 inputs["local_features"].shape[-1], seed=42,
                                 observation_bound=settings["observation_residual_amplitude"],
                                 soh_bounds=tuple(settings["observation_soh_bounds"])).to(device)
        hashes[name] = state_hash(model)
        if hashes[name] != registration["initialization"]["models"][arm]["initial_state_sha256"]:
            raise RuntimeError("Initial model differs from the registered state")
        models[name] = model
    training.seed_everything()
    common_rng = training.capture_random_state()
    cases = {}
    for name, model_name in (("M2_a_first", "M2_a"), ("M2_a_repeat", "M2_a"), ("M2_b", "M2_b"), ("NR", "NR")):
        cases[name] = run_case(models[model_name], batch, tensor_query, tensor_target, tensor_observed,
                               tensor_rates, tensor_rate_mask, settings, common_rng)
    comparisons = []
    for name in ("M2_a_repeat", "M2_b", "NR"):
        first, last = cases["M2_a_first"], cases[name]
        comparisons.append({"reference": "M2_a_first", "candidate": name,
                            "rng_before_equal": first["rng_before"] == last["rng_before"],
                            "rng_after_equal": first["rng_after"] == last["rng_after"],
                            "forward_and_loss": compare_tensors(first["outputs"], last["outputs"]),
                            "parameter_gradients": compare_tensors(first["gradients"], last["gradients"])})
    after_integrity = artifact_integrity(registration, freeze, data_root)
    unchanged = all(after_integrity.values()) and all(state_hash(model) == hashes[name] for name, model in models.items())
    if not unchanged:
        raise RuntimeError("An artifact or model parameter changed during the zero-update diagnostic")
    return {
        "status": "DIAGNOSED", "scope": "engineering_first_training_batch_zero_optimizer_updates",
        "seed": 42, "split": "cell_fold_0", "epoch": 1, "batch_size": len(rows),
        "query_count": len(query), "query_range": [float(query[0]), float(query[-1])],
        "first_batch_cell_uids": [ids[number] for number in rows],
        "batch_input_sha256": tensor_digest(batch), "query_sha256": tensor_digest({"query": tensor_query}),
        "observed_training_target_sha256": tensor_digest({"targets": tensor_target, "observed": tensor_observed,
                                                        "rate_targets": tensor_rates, "rate_mask": tensor_rate_mask}),
        "scaler_sha256": {arm: digest(run_root / "cell_fold_0" / arm / "TRAIN_SCALER.json") for arm in ("M2", "M3-NR")},
        "initial_state_sha256": hashes,
        "cases": {name: {key: value for key, value in result.items() if key not in ("outputs", "gradients")}
                  for name, result in cases.items()},
        "comparisons": comparisons, "optimizer_updates": 0, "parameters_unchanged": unchanged,
        "artifact_integrity_before": before_integrity, "artifact_integrity_after": after_integrity,
        "registration_sha256": digest(run_root / "RUN_REGISTRATION.json"),
        "data_freeze_sha256": digest(data_root / "DATA_FREEZE.json"),
        "environment": {"torch": torch.__version__, "numpy": np.__version__, "device": str(device),
                        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                        "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
                        "strict_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"), "dtype": "FP32"},
        "interpretation": "This local first-batch diagnostic examines identical model states, inputs and restored RNG with forward/backward only. A matching diagnostic does not establish full-training bitwise repeatability. The sealed six-split repeat failure remains unchanged. No underlying cause is certified by this result alone.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPOSITORY / "data/unified_matr_seed42")
    parser.add_argument("--run-root", type=Path, default=REPOSITORY.parent / "private/unified_matr_seed42/runs")
    parser.add_argument("--output", type=Path, default=REPOSITORY / "results/unified_matr_seed42/REPRODUCTION_DIAGNOSTIC.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
    result = diagnose(args.data, args.run_root, torch.device(args.device))
    result["diagnostic_source_sha256"] = digest(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".pending")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    summary = [{"candidate": row["candidate"], "forward_exact": row["forward_and_loss"]["exact_equal"],
                "gradient_exact": row["parameter_gradients"]["exact_equal"],
                "maximum_gradient_difference": row["parameter_gradients"]["max_absolute_difference"]}
               for row in result["comparisons"]]
    print(json.dumps({"status": result["status"], "optimizer_updates": 0, "comparisons": summary}))


if __name__ == "__main__":
    main()
