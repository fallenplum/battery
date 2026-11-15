"""Zero-update numerical, information-deletion and repeatability checks."""

from __future__ import annotations

import argparse
import copy
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
from models.unified_matr.legacy_operators import LegacyFlowHead
from models.unified_matr.methodology_ablations import (
    ABLATION_ARMS, LegacyFlowWithSignedRate, MethodologyAblationModel,
    NoWriteMixer, model_registry,
)
from models.unified_matr.methodology_full import MethodologyFullModel
from models.unified_matr.model import initialize
from scripts.unified_matr import train_methodology_full as runner
from scripts.unified_matr import train_seed42 as baseline


def tensor_sha(value):
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def synthetic_batch(device):
    generator = torch.Generator().manual_seed(42)
    def normal(*shape):
        return torch.randn(*shape, generator=generator).to(device)
    point_mask = torch.ones(3, 100, 64, dtype=torch.bool, device=device)
    point_mask[0, :3] = False
    point_mask[0, 30:35] = False
    point_mask[1, 11:18] = False
    point_mask[2] = False
    return {"raw_viq": normal(3, 100, 64, 3), "point_mask": point_mask,
            "cycle_ids": torch.arange(1, 101, device=device).float().expand(3, -1),
            "reference_viq": normal(3, 100, 64, 3), "policy": normal(3, 15),
            "policy_mask": torch.ones(3, 15, dtype=torch.bool, device=device),
            "global_features": normal(3, 100, 20),
            "global_mask": torch.ones(3, 100, 20, dtype=torch.bool, device=device),
            "local_residual": normal(3, 100, 64),
            "local_mask": torch.ones(3, 100, 64, dtype=torch.bool, device=device),
            "local_features": normal(3, 100, 8, 26),
            "local_feature_mask": torch.ones(3, 100, 8, 26, dtype=torch.bool, device=device),
            "anchor_soh100": torch.full((3,), .98, device=device)}


def check(checks, name, condition, **detail):
    row = {"name": name, "status": "PASS" if bool(condition) else "FAIL", **detail}
    checks.append(row)
    if not condition:
        raise AssertionError(name + " " + repr(detail))


def close(checks, name, first, second, atol=2e-6, rtol=2e-5):
    error = float((first - second).abs().max()) if first.numel() else 0.0
    check(checks, name, torch.allclose(first, second, atol=atol, rtol=rtol), max_absolute_error=error)


def legacy_oracle(checks, device):
    for dtype in (torch.float64, torch.float32):
        head = initialize(42, "legacy_head", LegacyFlowWithSignedRate).to(device=device, dtype=dtype)
        original = initialize(42, "legacy_head", LegacyFlowHead).to(device=device, dtype=dtype)
        generator = torch.Generator().manual_seed(42)
        state = torch.randn(2, 16, generator=generator, dtype=dtype).to(device).tanh()
        condition = torch.randn(2, 16, generator=generator, dtype=dtype).to(device)
        tau = torch.tensor([[0., .001, .1, .4, .9], [0., .002, .2, .5, 2.]], dtype=dtype, device=device)
        loss, signed = head(state, condition, tau * 1000 + 100)
        # Use exactly the coordinates recovered from the normal head interface.
        recovered_tau = ((tau * 1000 + 100) - 100) / 1000
        expected = original(state, condition, recovered_tau)
        check(checks, "legacy_prediction_bitwise/" + str(dtype), torch.equal(-loss, expected))
        variable = recovered_tau.detach().clone().requires_grad_(True)
        expected_gradient, = torch.autograd.grad(-original(state, condition, variable).sum(), variable)
        tolerance = 1e-10 if dtype == torch.float64 else 2e-7
        close(checks, "legacy_signed_derivative_positive_tau/" + str(dtype), signed[:, 1:], expected_gradient[:, 1:], atol=tolerance)
        check(checks, "legacy_signed_rate_detached/" + str(dtype), not signed.requires_grad)
        if dtype == torch.float64:
            epsilon = 1e-6
            upper = -original(state, condition, recovered_tau + epsilon)
            lower = -original(state, condition, recovered_tau - epsilon)
            close(checks, "legacy_centered_finite_difference", signed[:, 1:], ((upper - lower) / (2 * epsilon))[:, 1:], atol=2e-10)
            zero = torch.zeros_like(tau[:, :1])
            right = (-original(state, condition, zero + epsilon) + original(state, condition, zero)) / epsilon
            close(checks, "legacy_anchor_right_derivative", signed[:, :1], right, atol=2e-7)


def static_loop(value, rho, valid, cycles):
    state = torch.zeros_like(value[:, 0])
    previous = torch.zeros_like(cycles[:, 0])
    seen = torch.zeros(len(value), dtype=torch.bool, device=value.device)
    outputs, penalties, transitions = [], [], 0
    for step in range(value.shape[1]):
        observed = valid[:, step]
        gap = (cycles[:, step] - previous).clamp_min(1)
        retention = rho[:, step].pow(gap[:, None])
        proposal = torch.where(seen[:, None], retention * state + (1 - retention) * value[:, step], value[:, step])
        transition = observed & seen
        penalties.append(torch.where(transition[:, None], ((proposal - state) / gap[:, None]).square(), 0.0).sum())
        transitions = transitions + transition.sum()
        state = torch.where(observed[:, None], proposal, state)
        previous = torch.where(observed, cycles[:, step], previous)
        seen = seen | observed
        outputs.append(torch.where(observed[:, None], state, 0.0))
    return torch.stack(outputs, 1), torch.stack(penalties).sum() / (transitions.clamp_min(1) * value.shape[-1])


def structural_checks(checks, device):
    batch = synthetic_batch(device)
    queries = torch.tensor([100., 101., 200., 500., 1000.], device=device)
    parent = MethodologyFullModel(15, 20, 26).to(device).eval()
    baseline_output = parent(**batch, query_cycles=queries, return_diagnostics=True)
    registries = []
    for arm in ABLATION_ARMS:
        model = MethodologyAblationModel(arm, 15, 20, 26).to(device).eval()
        output = model(**batch, query_cycles=queries, return_diagnostics=True)
        registry = model_registry(model)
        registries.append(registry)
        check(checks, arm + "/finite_outputs", all(torch.isfinite(output[key]).all() for key in ("trajectory", "raw_trend", "rate_per_cycle")))
        parent_state = parent.state_dict()
        shared = [(key, value) for key, value in model.state_dict().items()
                  if key in parent_state and value.shape == parent_state[key].shape]
        check(checks, arm + "/shared_initial_tensors", all(torch.equal(value, parent_state[key]) for key, value in shared), shared_tensors=len(shared))
        diagnostic = output["diagnostics"]["persistence"]
        check(checks, arm + "/missing_state_zero", (diagnostic["state"][~diagnostic["valid"]] == 0).all())
        if arm == "A2_M4_NO_B_WRITE":
            check(checks, arm + "/no_write_parameters", not any("write_" in name for name, _ in model.named_parameters()))
            for layer, item in enumerate(output["diagnostics"]["history"]):
                check(checks, arm + f"/Bprime_is_B_layer{layer}", torch.equal(item["B"], item["Bprime"]) and not item["write_modulation"].any())
            for key in ("current_response", "rho", "state"):
                check(checks, arm + "/initial_" + key, torch.equal(diagnostic[key], baseline_output["diagnostics"]["persistence"][key]))
            check(checks, arm + "/initial_prediction_matches_M4", torch.equal(output["trajectory"], baseline_output["trajectory"]))
            raw = torch.randn(2, 7, 64, device=device)
            valid = torch.ones(2, 7, dtype=torch.bool, device=device)
            response, quality = torch.randn_like(raw), torch.randn(2, 7, 32, device=device)
            mixer = copy.deepcopy(parent.history[0].mixer)
            first = NoWriteMixer(mixer)(raw, valid, response, quality, valid)[0]
            with torch.no_grad():
                mixer.write_projection.weight.add_(1.)
                mixer.write_gate.weight.add_(1.)
            last = NoWriteMixer(mixer)(raw, valid, response, quality, valid)[0]
            check(checks, arm + "/controlled_WB_gate_perturbation_invariant", torch.equal(first, last))
        elif arm == "A3_M4_STATIC_RHO":
            check(checks, arm + "/rho_initial_point9", torch.allclose(diagnostic["rho"], torch.full_like(diagnostic["rho"], .9)))
            expected, penalty = static_loop(diagnostic["current_response"], diagnostic["rho"], diagnostic["valid"], batch["cycle_ids"])
            close(checks, arm + "/actual_gap_loop_state", diagnostic["state"], expected, atol=5e-6)
            close(checks, arm + "/actual_gap_loop_penalty", output["persistence_penalty"], penalty)
            changed = {**batch, "global_features": batch["global_features"] + 10, "policy": batch["policy"] + 10}
            altered = model(**changed, query_cycles=queries, return_diagnostics=True)
            check(checks, arm + "/rho_response_policy_invariance", torch.equal(diagnostic["rho"], altered["diagnostics"]["persistence"]["rho"]))
        elif arm == "A4_M4_NO_LOCAL_PHYSICS":
            changed = {**batch, "local_residual": torch.full_like(batch["local_residual"], float("nan")),
                       "local_features": torch.full_like(batch["local_features"], float("nan")),
                       "local_mask": ~batch["local_mask"], "local_feature_mask": ~batch["local_feature_mask"]}
            altered = model(**changed, query_cycles=queries, return_diagnostics=True)
            check(checks, arm + "/all_local_values_and_masks_invariant", torch.equal(output["trajectory"], altered["trajectory"]))
            check(checks, arm + "/quality_zero", not diagnostic["quality"].any())
            supported = batch["global_mask"] & batch["point_mask"].any(-1)[..., None]
            global_value = torch.where(supported, batch["global_features"], 0.)
            token = parent.physics.global_encoder(torch.cat((global_value, supported.to(global_value.dtype)), -1))
            expected = parent.physics.combine(torch.cat((token, torch.zeros_like(token)), -1))
            available = batch["point_mask"].any(-1) & supported.any(-1)
            expected = torch.where(available[..., None], expected, 0.)
            close(checks, arm + "/global_path_equals_M4_zero_local_token", diagnostic["current_response"], expected)
            absent_local = torch.zeros_like(batch["local_mask"])
            absent_features = torch.zeros_like(batch["local_features"])
            absent_feature_mask = torch.zeros_like(batch["local_feature_mask"])
            safe_cycles = torch.where(batch["point_mask"].any(-1), batch["cycle_ids"], 0.)
            old_state, old_penalty, old_quality, old_diagnostic = parent.persistence(
                diagnostic["current_response"], available, safe_cycles, batch["global_features"], batch["global_mask"],
                absent_local, absent_features, absent_feature_mask, batch["policy"], batch["policy_mask"], True)
            close(checks, arm + "/rho_equals_M4_zero_quality", diagnostic["rho"], old_diagnostic["rho"])
            close(checks, arm + "/state_equals_M4_zero_quality", diagnostic["state"], old_state)
            close(checks, arm + "/penalty_equals_M4_zero_quality", output["persistence_penalty"], old_penalty)
            for layer, block in enumerate(model.history):
                full_input = torch.cat((diagnostic["state"], diagnostic["quality"]), -1)
                close(checks, arm + f"/gate_zero_quality_equivalence_layer{layer}",
                      block.mixer.write_gate(full_input), parent.history[layer].mixer.write_gate(full_input))
            differentiable = {key: value.detach().clone().requires_grad_(True) if key in ("raw_viq", "reference_viq", "global_features", "local_residual", "local_features") else value for key, value in batch.items()}
            current = model(**differentiable, query_cycles=queries)
            (current["trajectory"].mean() + .001 * current["persistence_penalty"]).backward()
            for key in ("raw_viq", "reference_viq", "global_features"):
                gradient = differentiable[key].grad
                check(checks, arm + "/retained_gradient/" + key, gradient is not None and bool(gradient.abs().sum() > 0))
            for key in ("local_residual", "local_features"):
                check(checks, arm + "/removed_gradient/" + key, differentiable[key].grad is None)
        del output, model
    return registries


def first_batch_repeats(checks, device, repeats):
    inputs, ids, eligible, _, query, targets, observed, rate_targets, rate_mask, splits, _ = baseline.load_frozen_data(REPOSITORY / "data/unified_matr_seed42")
    roles = splits["splits"]["cell_fold_0"]
    index = {uid: number for number, uid in enumerate(ids)}
    train = np.asarray([index[uid] for uid in roles["train"] if eligible[index[uid]]])
    scaled, scaler = baseline.fold_scaler(inputs, train)
    rows = np.random.default_rng(42).permutation(train)[:8]
    settings = json.loads((REPOSITORY / "configs/unified_matr_seed42.json").read_text(encoding="utf-8"))["training"]
    result = {"training_cell_uids": [ids[number] for number in rows], "query_count": len(query),
              "input_sha256": {key: tensor_sha(torch.as_tensor(value[rows])) for key, value in scaled.items()},
              "train_scaler_sha256": hashlib.sha256(json.dumps(scaler, sort_keys=True).encode()).hexdigest(), "arms": {}}
    for arm in ABLATION_ARMS:
        records = []
        for _ in range(repeats):
            baseline.seed_everything()
            model = MethodologyAblationModel(arm, 15, 20, 26, observation_bound=settings["observation_residual_amplitude"], soh_bounds=tuple(settings["observation_soh_bounds"])).to(device).train()
            output = model(**baseline.to_batch(scaled, rows, device), query_cycles=torch.as_tensor(query, device=device))
            loss = runner.training_loss(output, rows, targets, observed, rate_targets, rate_mask, settings, device)
            loss.backward()
            receipt = runner.gradient_receipt(model)
            check(checks, arm + f"/repeat{len(records) + 1}_finite_gradients", all(item["finite"] for item in receipt["parameters"].values()))
            records.append({"prediction_sha256": tensor_sha(output["trajectory"]), "loss_sha256": tensor_sha(loss),
                            "gradients_sha256": receipt["sha256"], "gradient_tensors": len(receipt["parameters"])})
            del model, output, loss
        check(checks, arm + "/fresh_first_batch_exact", all(record == records[0] for record in records), repeats=repeats)
        result["arms"][arm] = records
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--real-first-batch", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runner.configure_execution()
    checks = []
    legacy_oracle(checks, args.device)
    registries = structural_checks(checks, args.device)
    repeats = first_batch_repeats(checks, args.device, args.repeats) if args.real_first_batch else None
    receipt = {"status": "PASS", "device": args.device, "strict_determinism": torch.are_deterministic_algorithms_enabled(),
               "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"], "optimizer_updates": 0, "formal_fits": 0,
               "checks": checks, "model_registries": registries, "real_training_first_batch": repeats,
               "sources": [{"path": str(path.relative_to(REPOSITORY)).replace("\\", "/"), "sha256": baseline.sha256(path)} for path in
                           (Path(__file__), REPOSITORY / "models/unified_matr/methodology_ablations.py")]}
    if args.output.exists():
        raise RuntimeError("An existing engineering receipt is immutable")
    baseline.write_json(args.output, receipt)
    print(json.dumps({"status": "PASS", "device": args.device, "checks": len(checks), "repeats_per_arm": args.repeats if repeats else 0, "optimizer_updates": 0}))


if __name__ == "__main__":
    main()
