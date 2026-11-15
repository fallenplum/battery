"""Independent M5 decoder equations and frozen-prefix engineering controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
from scipy.special import erf, expit
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.unified_matr.m5_bounded import BoundedRelativeHazardHead, M5BoundedModel
from models.unified_matr.methodology_full import MethodologyFullModel
from scripts.unified_matr import train_m5
from scripts.unified_matr import train_methodology_full as base


def numpy_head(head, state, condition, query):
    first, last = head.net[0], head.net[2]
    x = np.concatenate((state, condition), axis=-1)
    hidden = x @ first.weight.detach().cpu().numpy().T + first.bias.detach().cpu().numpy()
    hidden = .5 * hidden * (1 + erf(hidden / np.sqrt(2)))
    p = hidden @ last.weight.detach().cpu().numpy().T + last.bias.detach().cpu().numpy()
    grid = np.arange(100, max(101, int(np.ceil(query.max()))) + 1)
    tau = (grid - 100) / 1000
    a, b, c, d = [bound * np.tanh(p[:, i]) for i, bound in enumerate((2, 1.5, 1, 2))]
    center, width = .1 + 3.5 * expit(p[:, 4]), .05 + .8 * expit(p[:, 5])
    local = expit((tau[None] - center[:, None]) / width[:, None]) - expit(-center[:, None] / width[:, None])
    smooth = np.tanh(tau / 2)[None]
    logits = float(head.baseline_logit.cpu()) + a[:, None] + b[:, None] * smooth + c[:, None] * smooth**2 + d[:, None] * local
    rate = head.lambda_max * expit(logits)
    cumulative = np.concatenate((np.zeros((len(state), 1)), np.cumsum((rate[:, 1:] + rate[:, :-1]) / 2000, axis=1)), axis=1)
    index = np.minimum(np.floor(query - 100).astype(int), len(grid) - 2)
    fraction = query - 100 - index
    row = np.arange(len(state))[:, None]
    integral = cumulative[row, index] + fraction * (cumulative[row, index + 1] - cumulative[row, index])
    values = rate[row, index] + fraction * (rate[row, index + 1] - rate[row, index])
    return integral, values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    base.configure_execution()
    checks = []

    def require(name, passed, **details):
        checks.append({"name": name, "status": "PASS" if bool(passed) else "FAIL", **details})
        if not passed:
            raise AssertionError(name)

    result = {"status": "FAIL", "device": args.device, "optimizer_updates": 0, "formal_fits": 0}
    try:
        before = (base.ARM, base.make_model, base.model_contract, base.DEPENDENCIES)
        with train_m5.m5_runner_scope():
            require("scoped_factory_is_M5", base.ARM == train_m5.ARM and base.make_model is train_m5.build_model)
        require("shared_runner_restored", before == (base.ARM, base.make_model, base.model_contract, base.DEPENDENCIES))
        torch.manual_seed(42)
        head = BoundedRelativeHazardHead().double().to(args.device)
        state = torch.randn(2, 16, dtype=torch.float64, device=args.device, requires_grad=True)
        condition = torch.randn_like(state, requires_grad=True)
        q = torch.tensor([[100., 100.25, 101., 207.5, 500., 1000., 2238., 5000., 10000.]], dtype=torch.float64, device=args.device).expand(2, -1)
        hazard, rate = head(state, condition, q)
        oracle_h, oracle_rate = numpy_head(head, state.detach().cpu().numpy(), condition.detach().cpu().numpy(), q.cpu().numpy())
        for key, tensor, expected in (("integral", hazard, oracle_h), ("relative_intensity", rate, oracle_rate)):
            error = float(np.max(np.abs(tensor.detach().cpu().numpy() - expected)))
            require("numpy_float64_" + key, np.allclose(tensor.detach().cpu().numpy(), expected, atol=2e-11, rtol=2e-11), max_abs_error=error)
        require("relative_intensity_bound", bool(((rate > 0) & (rate < 2)).all()))
        anchor = torch.tensor([.97, 1.03], dtype=torch.float64, device=args.device)
        trend = anchor[:, None] * (-hazard).exp()
        require("head_anchor_exact", torch.equal(trend[:, 0], anchor))
        require("head_nonnegative_nonincreasing", bool((trend >= 0).all() and (trend[:, 1:] <= trend[:, :-1]).all()))
        order = [8, 3, 0, 6, 1, 4, 7, 2, 5]
        reordered, _ = head(state, condition, q[:, order])
        require("unordered_queries", torch.equal(reordered, hazard[:, order]))
        for slot in (0, 1, 3, 5, 7):
            single, _ = head(state, condition, q[:, slot:slot + 1])
            require("query_alone_" + str(slot), torch.allclose(single[:, 0], hazard[:, slot], atol=2e-12, rtol=2e-12))
        trend.mean().backward()
        require("head_input_gradients", bool(torch.isfinite(state.grad).all() and torch.isfinite(condition.grad).all() and state.grad.norm() > 0 and condition.grad.norm() > 0))
        require("head_parameter_gradients", all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in head.parameters()))
        constants = BoundedRelativeHazardHead().double().to(args.device)
        with torch.no_grad():
            for parameter in constants.net.parameters():
                parameter.zero_()
        constant_h, constant_r = constants(state.detach(), condition.detach(), q)
        require("constant_intensity_integral", torch.allclose(constant_h, constant_r[:, :1] * (q - 100) / 1000, atol=2e-12, rtol=2e-12))

        data = base.load_frozen_data(args.data)
        inputs, identifiers, eligible, _, query, targets, observed, rates, rate_mask, split, _ = data
        require("frozen_cohort169", len(identifiers) == 169 and eligible.all())
        indices = {uid: i for i, uid in enumerate(identifiers)}
        training = np.array([indices[uid] for uid in split["splits"]["cell_fold_0"]["train"]])
        scaled, _ = base.fold_scaler(inputs, training)
        rows = np.random.default_rng(42).permutation(training)[:8]
        batch = base.to_batch(scaled, rows, args.device)
        base.seed_everything()
        full = MethodologyFullModel(15, 20, 26).to(args.device).eval()
        model = M5BoundedModel(15, 20, 26).to(args.device).eval()
        shared = [name for name in full.state_dict() if not name.startswith("flow_head.")]
        require("unchanged_shared_initialization", all(torch.equal(full.state_dict()[name], model.state_dict()[name]) for name in shared), tensors=len(shared))
        coordinates = torch.tensor([100., 101., 200., 500., 1000., 2238., 5000.], device=args.device)
        with torch.no_grad():
            old = full(**batch, query_cycles=coordinates, return_diagnostics=True)
            new = model(**batch, query_cycles=coordinates, return_diagnostics=True)
        require("same_encoder_state", torch.equal(old["diagnostics"]["state"], new["diagnostics"]["state"]))
        require("same_encoder_condition", torch.equal(old["diagnostics"]["condition"], new["diagnostics"]["condition"]))
        f = new["diagnostics"]["flow"]
        require("full_model_anchor_exact", torch.equal(new["raw_trend"][:, 0], batch["anchor_soh100"]))
        require("relative_integral_exponential_map", torch.equal(new["raw_trend"], batch["anchor_soh100"][:, None] * torch.exp(-f["relative_hazard_integral"])))
        require("absolute_rate_conversion", torch.equal(new["rate_per_1000_cycles"], new["raw_trend"] * f["relative_hazard_per_1000"]))
        require("per_cycle_units", torch.equal(new["rate_per_cycle"], new["rate_per_1000_cycles"] / 1000))
        require("trend_nonnegative_nonincreasing", bool((new["raw_trend"] >= 0).all() and (new["raw_trend"][:, 1:] <= new["raw_trend"][:, :-1]).all()))
        require("soft_center_bounds", bool(((f["soft_transition_center_cycle"] > 200) & (f["soft_transition_center_cycle"] < 3700)).all()))
        require("soft_width_bounds", bool(((f["soft_transition_width_cycles"] > 50) & (f["soft_transition_width_cycles"] < 850)).all()))
        masked = {key: value.clone() for key, value in batch.items()}
        masked["raw_viq"] = torch.where(batch["point_mask"][..., None], batch["raw_viq"], torch.nan)
        masked["reference_viq"] = torch.where(batch["point_mask"][..., None], batch["reference_viq"], torch.nan)
        masked["global_features"] = torch.where(batch["global_mask"], batch["global_features"], torch.nan)
        with torch.no_grad():
            changed = model(**masked, query_cycles=coordinates)
        require("invalid_value_invariance", torch.equal(new["trajectory"], changed["trajectory"]))
        del full, model, new, old, changed

        repetitions = []
        for _ in range(12 if args.device == "cuda" else 2):
            base.seed_everything()
            model = train_m5.build_model(inputs, json.loads((ROOT / "configs/unified_matr_seed42.json").read_text(encoding="utf-8"))["training"]).to(args.device).train()
            output = model(**batch, query_cycles=torch.as_tensor(query, device=args.device), return_diagnostics=True)
            loss = base.training_loss(output, rows, targets, observed, rates, rate_mask,
                                      json.loads((ROOT / "configs/unified_matr_seed42.json").read_text(encoding="utf-8"))["training"], args.device)
            loss.backward()
            gradients = base.gradient_receipt(model)
            require("all_real_gradients_finite_" + str(len(repetitions)), all(v["finite"] for v in gradients["parameters"].values()))
            for layer, block in enumerate(output["diagnostics"]["history"]):
                require("initial_B_write_zero_" + str(len(repetitions)) + "_" + str(layer), torch.equal(block["B"], block["Bprime"]))
                parameter = dict(model.named_parameters())[f"history.{layer}.mixer.write_projection.weight"]
                require("B_write_gradient_nonzero_" + str(len(repetitions)) + "_" + str(layer), parameter.grad is not None and bool(parameter.grad.norm() > 0))
            repetitions.append({"prediction_sha256": hashlib.sha256(output["trajectory"].detach().cpu().numpy().tobytes()).hexdigest(),
                                "loss": float(loss.detach()), "gradient_sha256": gradients["sha256"]})
            del model, output, loss
        require("fresh_seed42_real_batch_exact", all(item == repetitions[0] for item in repetitions), repetitions=len(repetitions))
        result.update(status="PASS", repetitions=repetitions, shared_initial_tensors=len(shared), checks=checks,
                      training_cell_uid=[identifiers[i] for i in rows], test_observations_consumed=0,
                      sources=[{"path": name, "sha256": base.sha256(ROOT / name)} for name in
                               ("models/unified_matr/m5_bounded.py", "scripts/unified_matr/train_m5.py", "scripts/verify_m5_engineering.py")])
    except Exception as exc:
        result.update(exception=type(exc).__name__, message=str(exc), checks=checks)
        raise
    finally:
        result["passed_checks"] = sum(row["status"] == "PASS" for row in checks)
        base.write_json(args.output, result)
    print(json.dumps({"status": result["status"], "device": args.device, "passed_checks": result["passed_checks"]}), flush=True)


if __name__ == "__main__":
    main()
