"""Independent M4 engineering controls, without optimization or model selection."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from unittest.mock import patch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.unified_matr import methodology_operators as operators
from models.unified_matr.methodology_full import MethodologyFullModel
from models.unified_matr.model import state_hash
from models.unified_matr.selfcheck import make_batch, tensor_clone
from scripts.unified_matr import train_methodology_full as runner

CONTRACT_SHA = "94ac2388411306e162243ced735188c373c03a6daabbe1cf49e9ae94c7136523"
BASELINE = "c81aba21d71b337e9fa5f4d16fd37c6eb483eb12"
SOURCES = tuple(runner.DEPENDENCIES) + ("scripts/verify_m4_method_contract.py",)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(records, name, condition, **detail):
    records.append({"check": name, "status": "PASS" if bool(condition) else "FAIL", **detail})
    if not condition:
        raise AssertionError(name)


def maximum_error(first, second):
    return float((first.detach() - second.detach()).abs().max().cpu())


def close(records, name, first, second, tolerance):
    error = maximum_error(first, second)
    check(records, name, error <= tolerance, maximum_absolute_error=error, tolerance=tolerance)


def scalar_ema_oracle(value, rho, valid, cycles):
    """Explicit recurrence independent of cumulative-log dense expansion."""
    all_internal, all_visible, changes = [], [], []
    transitions = 0
    for batch in range(len(value)):
        previous_state = torch.zeros_like(value[batch, 0])
        previous_cycle = None
        internal, visible = [], []
        for step in range(value.shape[1]):
            if bool(valid[batch, step]):
                if previous_cycle is None:
                    current = value[batch, step]
                else:
                    gap = cycles[batch, step] - previous_cycle
                    retention = rho[batch, step].pow(gap)
                    current = retention * previous_state + (1 - retention) * value[batch, step]
                    changes.append(((current - previous_state) / gap).square().sum())
                    transitions += 1
                previous_state = current
                previous_cycle = cycles[batch, step]
                visible.append(current)
            else:
                visible.append(torch.zeros_like(previous_state))
            internal.append(previous_state)
        all_internal.append(torch.stack(internal))
        all_visible.append(torch.stack(visible))
    penalty = sum(changes, value.new_zeros(())) / max(1, transitions * value.shape[-1])
    return torch.stack(all_visible), torch.stack(all_internal), penalty


def verify_dynamic_scan(records, device):
    generator = torch.Generator().manual_seed(4201)
    valid = torch.tensor([[False, True, False, True, True, False, True],
                          [False, False, False, False, False, False, False]], device=device)
    cycles = torch.tensor([[0, 2, 0, 7, 11, 0, 20], [0] * 7], dtype=torch.float64, device=device)
    value = torch.randn(2, 7, 64, generator=generator, dtype=torch.float64).to(device).requires_grad_()
    rho = (.2 + .6 * torch.rand(2, 7, 64, generator=generator, dtype=torch.float64)).to(device).requires_grad_()
    state, penalty, diagnostic = operators.dynamic_ema(value, rho, valid, cycles)
    expected, internal, expected_penalty = scalar_ema_oracle(value, rho, valid, cycles)
    close(records, "dense_dynamic_scan_matches_scalar_recurrence_FP64", state, expected, 2e-12)
    close(records, "invalid_internal_state_holds_previous_valid", diagnostic["internal_state"], internal, 2e-12)
    close(records, "persistence_penalty_has_same_per_cycle_squared_units", penalty, expected_penalty, 2e-12)
    weight = torch.linspace(.1, 1, state.numel(), device=device).reshape_as(state)
    actual_grad = torch.autograd.grad((state * weight).sum() + penalty, (value, rho), retain_graph=True)
    oracle_grad = torch.autograd.grad((expected * weight).sum() + expected_penalty, (value, rho))
    for name, actual, oracle in zip(("phi", "rho"), actual_grad, oracle_grad):
        close(records, "scalar_gradient_oracle_" + name, actual, oracle, 4e-11)
        check(records, "zero_invalid_gradient_" + name, bool((actual[~valid] == 0).all()))
    close(records, "first_valid_exact_initialization", state[0, 1], value[0, 1], 0)
    check(records, "all_invalid_cell_visible_state_zero", bool((state[1] == 0).all()))
    close(records, "actual_cycle_gap_uses_previous_valid", diagnostic["gap"][0, [3, 4, 6]],
          cycles.new_tensor([5, 4, 9]), 0)


def verify_rho_context(records, device):
    torch.manual_seed(4202)
    module = operators.DynamicPersistence(15, 20, 26).to(device)
    batch, length = 1, 7
    value = torch.randn(batch, length, 64, device=device)
    valid = torch.tensor([[False, True, False, True, True, False, True]], device=device)
    cycles = torch.tensor([[0, 2, 0, 7, 11, 0, 20]], device=device)
    global_values = torch.randn(batch, length, 20, device=device)
    global_mask = torch.ones_like(global_values, dtype=torch.bool)
    local_mask = torch.ones(batch, length, 64, device=device, dtype=torch.bool)
    local_mask[:, 3, :16] = False
    local = torch.randn(batch, length, 8, 26, device=device)
    local_feature_mask = torch.ones_like(local, dtype=torch.bool)
    local_feature_mask[:, 4, 0, 6] = False
    policy = torch.randn(batch, 15, device=device)
    policy_mask = torch.ones_like(policy, dtype=torch.bool)
    captured = []
    handle = module.rho_net[0].register_forward_pre_hook(lambda _, args: captured.append(args[0]))
    args = (value, valid, cycles, global_values, global_mask, local_mask,
            local, local_feature_mask, policy, policy_mask)
    state, _, quality, diagnostic = module(*args, return_diagnostics=True)
    handle.remove()
    context = captured[0]
    check(records, "rho_context_registered_172_coordinates", context.shape == (1, 7, 172))
    close(records, "rho_previous_valid_global", context[0, 3, 64:84], global_values[0, 1], 0)
    close(records, "rho_robust_causal_median_drift_columns2_11", context[0, 3, -4:-2],
          (global_values[0, 3, [2, 11]] - global_values[0, 1, [2, 11]]) / 5, 0)
    check(records, "rho_first_valid_has_no_previous_global_or_drift", bool((context[0, 1, 64:104] == 0).all()
          and (context[0, 1, -4:] == 0).all()))
    close(records, "quality_residual_support_per_bin", quality[0, 3, :8],
          value.new_tensor([0, 0, 1, 1, 1, 1, 1, 1]), 0)
    check(records, "quality_noise_mask_zeroes_unsupported_proxy", quality[0, 4, 16] == 0 and quality[0, 4, 24] == 0)
    close(records, "rho_context_known_policy_values", context[0, 3, 136:151], policy[0], 0)
    check(records, "rho_clipped_bounds", bool((diagnostic["rho"] >= .01).all() and (diagnostic["rho"] <= .99).all()))
    for name, index in (("current_physics", 0), ("previous_valid_global", 64),
                        ("quality", 104), ("policy", 136), ("actual_gap", 166), ("robust_drift", 168)):
        changed = context.clone()
        changed[0, 3, index] += .5
        changed_rho = module.rho_net(changed).sigmoid().clamp(.01, .99)
        effect = maximum_error(changed_rho[0, 3], diagnostic["rho"][0, 3])
        check(records, "rho_responds_to_" + name, effect > 0, maximum_absolute_change=effect)
    changed_args = [item.clone() for item in args]
    for index in (0, 3, 6):
        changed_args[index][:, 4:] = 900
    changed_args[5][:, 4:] = False
    changed_state, _, _, changed_diagnostic = module(*changed_args, return_diagnostics=True)
    close(records, "future_prefix_response_cannot_change_earlier_rho", changed_diagnostic["rho"][:, :4], diagnostic["rho"][:, :4], 0)
    close(records, "future_prefix_response_cannot_change_earlier_persistence", changed_state[:, :4], state[:, :4], 0)


def verify_ssd_oracle(records, device):
    torch.manual_seed(4203)
    x = torch.randn(2, 7, 3, 4, dtype=torch.float64, device=device).requires_grad_()
    delta = (torch.rand(2, 7, 3, dtype=torch.float64, device=device) * .1).requires_grad_()
    a = (-torch.rand(3, dtype=torch.float64, device=device)).requires_grad_()
    b = torch.randn(2, 7, 5, dtype=torch.float64, device=device).requires_grad_()
    c = torch.randn(2, 7, 5, dtype=torch.float64, device=device).requires_grad_()
    skip = torch.randn(3, dtype=torch.float64, device=device).requires_grad_()
    variables = (x, delta, a, b, c, skip)
    state = x.new_zeros(2, 3, 4, 5)
    outputs = []
    for step in range(7):
        retention = torch.exp(delta[:, step] * a)
        write = delta[:, step, :, None, None] * x[:, step, :, :, None] * b[:, step, None, None, :]
        state = retention[..., None, None] * state + write
        outputs.append((state * c[:, step, None, None, :]).sum(-1) + x[:, step] * skip[None, :, None])
    serial = torch.stack(outputs, dim=1)
    dense = operators.dense_ssd(*variables)
    close(records, "M4_actual_SSD_matches_independent_serial_FP64", dense, serial, 2e-12)
    actual_gradients = torch.autograd.grad(dense.square().sum(), variables, retain_graph=True)
    serial_gradients = torch.autograd.grad(serial.square().sum(), variables)
    for name, actual, expected in zip(("x", "delta", "A", "B", "C", "skip"), actual_gradients, serial_gradients):
        close(records, "M4_actual_SSD_gradient_oracle_" + name, actual, expected, 5e-11)


def small_batch(device):
    batch = make_batch(device=device)
    batch["query_cycles"] = torch.tensor([100, 101, 200, 500, 1000, 5000], dtype=torch.float32, device=device)
    return batch


def verify_model(records, device):
    torch.manual_seed(42)
    model = MethodologyFullModel(15, 20, 26).to(device).eval()
    batch = small_batch(device)
    output = model(**batch, return_context=True, return_diagnostics=True)
    check(records, "normal_entry_output_shapes_finite", all(output[key].shape == (2, 6) and bool(torch.isfinite(output[key]).all())
          for key in ("trajectory", "trend", "raw_trend", "integrated_loss", "rate_per_cycle")))
    check(records, "both_history_WB_exact_zero_without_bias", all(bool((block.mixer.write_projection.weight == 0).all())
          and block.mixer.write_projection.bias is None for block in model.history))
    check(records, "initial_b_modulation_exact_zero", all(bool((item["Bprime"] == item["B"]).all())
          for item in output["diagnostics"]["history"]))
    loss = output["trajectory"].sum() + output["persistence_penalty"] * .001
    loss.backward()
    wb_norms = [float(block.mixer.write_projection.weight.grad.norm()) for block in model.history]
    check(records, "first_backward_reaches_both_WB", all(value > 0 for value in wb_norms), gradient_l2=wb_norms)
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        for block in model.history:
            values = torch.linspace(-.03, .03, block.mixer.write_projection.weight.numel(), device=device)
            block.mixer.write_projection.weight.copy_(values.reshape_as(block.mixer.write_projection.weight))
    captured_convolutions, captured_ssd_b = [], []
    handles = [block.mixer.conv1d.register_forward_hook(lambda _, args, result: captured_convolutions.append(result))
               for block in model.history]
    original_ssd = operators.dense_ssd

    def capture_ssd(x, delta, a, b, c, skip):
        if x.shape[1] == batch["raw_viq"].shape[1]:
            captured_ssd_b.append(b)
        return original_ssd(x, delta, a, b, c, skip)

    with patch.object(operators, "dense_ssd", capture_ssd):
        perturbed = model(**batch, return_diagnostics=True)
    for handle in handles:
        handle.remove()
    for layer, (block, history) in enumerate(zip(model.history, perturbed["diagnostics"]["history"])):
        postconv = F.silu(captured_convolutions[layer][:, :, :100].transpose(1, 2))
        expected_b = postconv[..., block.mixer.inner:block.mixer.inner + 16]
        expected_b = torch.where(batch["point_mask"].any(-1)[..., None], expected_b, 0)
        close(records, f"layer{layer}_B_is_actual_postconvolution_split", history["B"], expected_b, 0)
        close(records, f"layer{layer}_Bprime_actually_passed_to_dense_ssd", captured_ssd_b[layer], history["Bprime"], 0)
        state = perturbed["diagnostics"]["persistence"]["state"]
        expected_change = history["gate"] * block.mixer.write_projection(state).tanh()
        expected_change = torch.where(history["modulation_valid"][..., None], expected_change, 0)
        close(records, f"layer{layer}_Bprime_matches_registered_formula", history["Bprime"], history["B"] + expected_change, 0)
        check(records, f"layer{layer}_controlled_WB_produces_nonzero_B_change", bool((history["write_modulation"].abs() > 0).any()))
        check(records, f"layer{layer}_invalid_cycles_never_modulate_B", bool((history["write_modulation"][~history["modulation_valid"]] == 0).all()))
    effect = maximum_error(perturbed["trajectory"], output["trajectory"])
    check(records, "controlled_WB_perturbation_changes_normal_prediction", effect > 0, maximum_absolute_change=effect)
    perturbed["trajectory"].sum().backward()
    rho_gradient = float(model.persistence.rho_net[-1].weight.grad.norm())
    check(records, "perturbed_write_graph_reaches_dynamic_rho", rho_gradient > 0, gradient_l2=rho_gradient)
    model.zero_grad(set_to_none=True)
    changed = tensor_clone(batch)
    record_valid = batch["point_mask"].any(-1)
    for key in ("raw_viq", "reference_viq", "global_features", "local_residual", "local_features"):
        changed[key][~record_valid] = 1e20
    with torch.no_grad():
        clean = model(**batch)
        dirty = model(**changed)
    close(records, "arbitrary_invalid_payload_has_no_prediction_effect", clean["trajectory"], dirty["trajectory"], 0)
    gradient_batch = tensor_clone(batch)
    for key in ("raw_viq", "reference_viq", "global_features", "local_residual", "local_features"):
        gradient_batch[key].requires_grad_()
    grad_output = model(**gradient_batch)
    grad_output["trajectory"].sum().backward()
    for key in ("raw_viq", "reference_viq", "global_features", "local_residual", "local_features"):
        gradient = gradient_batch[key].grad
        check(records, "invalid_record_input_gradient_zero_" + key, bool((gradient[~record_valid] == 0).all()))
    for bad_id in (101, 0, 1.5):
        invalid = tensor_clone(batch)
        invalid["cycle_ids"] = invalid["cycle_ids"].float()
        invalid["cycle_ids"][:, -1] = bad_id
        try:
            model(**invalid)
        except ValueError:
            rejected = True
        else:
            rejected = False
        check(records, "reject_invalid_supported_prefix_cycle_" + str(bad_id), rejected)
    close(records, "trend_anchor100_is_prefix_anchor", clean["raw_trend"][:, 0], batch["anchor_soh100"], 0)
    check(records, "raw_trend_monotone_and_nonnegative_rate", bool((clean["raw_trend"].diff(dim=1) <= 0).all()
          and (clean["rate_per_cycle"] > 0).all()))
    check(records, "observation_residual_bounded_003_and_output_bounded", bool((clean["observation_residual"].abs() <= .03).all()
          and (clean["trajectory"] >= 0).all() and (clean["trajectory"] <= 1.2).all()))
    close(records, "physical_rate_per_cycle_equals_per1000_div1000", clean["rate_per_cycle"], clean["rate_per_1000_cycles"] / 1000, 0)
    queries = [200.5, 5000, 100, 200, 1000, 101, 500]
    with torch.no_grad():
        unordered = model(**{**batch, "query_cycles": batch["raw_viq"].new_tensor(queries)})
        individual = model(**{**batch, "query_cycles": batch["raw_viq"].new_tensor([200.5])})
    for key in ("raw_trend", "integrated_loss", "rate_per_1000_cycles", "trajectory"):
        close(records, "isolated_vs_joint_query_" + key, individual[key][:, 0], unordered[key][:, 0], 3e-7)
        indices = [queries.index(float(cycle)) for cycle in batch["query_cycles"]]
        close(records, "query_reordering_invariant_" + key, clean[key], unordered[key][:, indices], 3e-7)
    return model


def verify_integral(records, device):
    head = MethodologyFullModel(15, 20, 26).flow_head.to(device).double()
    with torch.no_grad():
        for parameter in head.net.parameters():
            parameter.zero_()
    state = torch.zeros(2, 16, dtype=torch.float64, device=device)
    query = torch.tensor([[100, 101, 200.5, 5000], [100, 101, 200.5, 5000]], dtype=torch.float64, device=device)
    integral, rate = head(state, state, query)
    expected_rate = F.softplus(head.rate_offset)
    expected_integral = (query - 100) * expected_rate / 1000
    close(records, "constant_rate_integral_physical_1000_denominator", integral, expected_integral, 1e-12)
    close(records, "inherited_pre_fixed_initial_rate_scale_01_per1000", rate, torch.full_like(rate, .1), 2e-8)


def verify_loss(records, device):
    settings = json.loads((ROOT / "configs/unified_matr_seed42.json").read_text(encoding="utf-8"))["training"]
    prediction = torch.linspace(.99, .8, 2 * 1000, device=device).reshape(2, 1000)
    target = np.full((2, 1000), .9, dtype=np.float32)
    mask = np.ones((2, 1000), dtype=bool)
    mask[0, 10:] = False
    rate_targets = np.array([[.0001, .0002], [.0003, .0004]], dtype=np.float32)
    rate_mask = np.array([[True, False], [True, True]])
    output = {"trajectory": prediction, "persistence_penalty": prediction.new_tensor(.25)}
    loss = runner.training_loss(output, np.array([0, 1]), target, mask, rate_targets, rate_mask, settings, device)
    values = prediction.detach().cpu().numpy()
    trajectory_loss = np.mean([np.abs(values[i, mask[i]] - target[i, mask[i]]).mean() for i in range(2)])
    rates = np.stack(((values[:, 200 - 101] - values[:, 500 - 101]) / 300,
                      (values[:, 500 - 101] - values[:, 1000 - 101]) / 500), axis=1)
    auxiliary = np.mean([np.abs(rates[i, rate_mask[i]] - rate_targets[i, rate_mask[i]]).mean() for i in range(2)]) * 1000
    expected = trajectory_loss + .1 * auxiliary + .001 * .25
    check(records, "registered_cellmean_loss_and_scaled_auxiliary_arithmetic", abs(float(loss.detach()) - float(expected)) < 2e-7,
          actual=float(loss.detach()), independent=float(expected))


def verify_static_and_data(records, data_root, device):
    methodology = ROOT / "research/M4_FULL_METHODOLOGY_RUN_ONLY_ZH.md"
    check(records, "exact_full_only_remote_contract_sha256", digest(methodology) == CONTRACT_SHA, sha256=digest(methodology))
    config = json.loads((ROOT / "configs/unified_matr_seed42.json").read_text(encoding="utf-8"))
    settings = config["training"]
    check(records, "fixed_seed42_150epochs_batch8_optimizer_and_loss", config["training_seeds"] == [42]
          and settings["epochs"] == 150 and settings["batch_size"] == 8 and settings["optimizer"] == "AdamW"
          and settings["learning_rate"] == .0003 and settings["weight_decay"] == .01
          and settings["auxiliary_rate_scale"] == 1000 and settings["auxiliary_rate_loss_weight"] == .1
          and settings["persistence_soft_penalty_weight"] == .001 and settings["early_stopping"] is False)
    check(records, "only_M4_six_existing_splits", runner.ARM == "M4_FULL_METHOD"
          and runner.SPLITS == tuple(f"cell_fold_{i}" for i in range(5)) + ("unseen_protocol_sensitivity",))
    signatures = set(inspect.signature(MethodologyFullModel.forward).parameters)
    check(records, "forward_only_frozen_prefix_and_known_policy_inputs", signatures == set(runner.baseline.INPUT_KEYS)
          | {"self", "query_cycles", "return_context", "return_diagnostics"})
    old_files = ("models/unified_matr/model.py", "models/unified_matr/legacy_operators.py",
                 "scripts/unified_matr/train_seed42.py", "configs/unified_matr_seed42.json",
                 "configs/unified_matr_seed42_splits.json")
    for name in old_files:
        frozen = subprocess.run(["git", "show", f"{BASELINE}:{name}"], cwd=ROOT, capture_output=True, check=True).stdout
        current = (ROOT / name).read_bytes()
        normalized = lambda value: value.replace(b"\r\n", b"\n")
        check(records, "baseline_source_preserved_" + name, normalized(frozen) == normalized(current), current_sha256=digest(ROOT / name))
    if data_root is None:
        records.append({"check": "actual_frozen_data_loader_and_first_batch", "status": "NOT_RUN", "reason": "--data omitted"})
        return
    data = runner.load_frozen_data(data_root)
    registration = runner.model_contract(data, data_root, ROOT / "configs/unified_matr_seed42.json", methodology)
    check(records, "frozen169_inputs_labels_splits_and_source_fingerprints", len(data[1]) == 169 and bool(data[2].all()),
          data_freeze_sha256=registration["data_freeze_sha256"], split_contract_sha256=registration["split_contract_sha256"],
          dataset_fingerprint_sha256=registration["data_fingerprint_sha256"])
    check(records, "exact_six_unit_optimizer_budget_12450", registration["planned_optimizer_updates"] == 12450,
          updates_by_split=registration["expected_updates_by_split"])
    identifiers = {uid: index for index, uid in enumerate(data[1])}
    training_ids = data[-2]["splits"][runner.SPLITS[0]]["train"]
    training_rows = np.array([identifiers[uid] for uid in training_ids])
    scaled, _ = runner.fold_scaler(data[0], training_rows)
    rows = training_rows[:2]
    batch = runner.to_batch(scaled, rows, device)
    torch.manual_seed(42)
    model = runner.make_model(data[0], settings).to(device).eval()
    check(records, "real_model_initial_state_matches_contract", state_hash(model) == registration["initialization"]["models"][runner.ARM]["initial_state_sha256"])
    result = model(**batch, query_cycles=torch.tensor([100, 101, 200, 500, 1000, 5000], device=device), return_diagnostics=True)
    loss = result["trajectory"].sum() + .001 * result["persistence_penalty"]
    loss.backward()
    check(records, "actual_train_prefix_normal_entry_forward_backward_finite", bool(torch.isfinite(loss))
          and all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()),
          optimizer_updates=0, test_observations_consumed=0, training_batch_size=len(rows))
    del result, loss, model
    verify_first_batch_exact(records, data, scaled, training_rows, settings, device)


def verify_first_batch_exact(records, data, scaled, training_rows, settings, device):
    """Match the registered real smoke's train/dropout/full-loss execution path."""
    old_threads = torch.get_num_threads()
    torch.set_num_threads(4)
    rows = np.random.default_rng(42).permutation(training_rows)[:8]
    repeats = []
    repeat_count = 12 if device == "cuda" else 2
    for repeat in range(repeat_count):
        runner.seed_everything()
        model = runner.make_model(data[0], settings).to(device).train()
        batch = runner.to_batch(scaled, rows, device)
        output = model(**batch, query_cycles=torch.as_tensor(data[4], device=device))
        loss = runner.training_loss(output, rows, data[5], data[6], data[7], data[8], settings, device)
        loss.backward()
        repeats.append({
            "prediction_sha256": hashlib.sha256(output["trajectory"].detach().cpu().numpy().tobytes()).hexdigest(),
            "loss": float(loss.detach()),
            "gradient_sha256": runner.gradient_receipt(model)["sha256"],
        })
        del output, loss, model
    torch.set_num_threads(old_threads)
    check(records, "actual_seed42_trainmode_full_loss_fourthread_first_batch_bitwise_repeat",
          all(receipt == repeats[0] for receipt in repeats[1:]), repeats=repeats,
          fresh_repeats=repeat_count, optimizer_updates=0, test_observations_consumed=0, batch_size=len(rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    source_hashes = {name: digest(ROOT / name) for name in SOURCES}
    records = []
    failure = None
    try:
        verify_dynamic_scan(records, args.device)
        verify_rho_context(records, args.device)
        verify_ssd_oracle(records, args.device)
        verify_model(records, args.device)
        verify_integral(records, args.device)
        verify_loss(records, args.device)
        verify_static_and_data(records, args.data, args.device)
        check(records, "tested_sources_unchanged_through_verification", source_hashes == {name: digest(ROOT / name) for name in SOURCES})
    except Exception:
        failure = traceback.format_exc()
    result = {
        "status": "PASS" if failure is None else "FAIL", "device": args.device,
        "checks": records, "passed_checks": sum(row["status"] == "PASS" for row in records),
        "source_sha256": source_hashes, "contract_sha256": CONTRACT_SHA,
        "environment": {"torch": torch.__version__, "numpy": np.__version__, "cuda": torch.version.cuda,
                        "device_name": torch.cuda.get_device_name(0) if args.device == "cuda" else "CPU",
                        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
        "scope": "Engineering controls and actual training-prefix forward/backward only. No optimizer updates or test scoring.",
        "rate_scale": {"physical_units": "SOH/cycle = rate_per_1000_cycles/1000",
                       "initialization": "0.1/1000 SOH/cycle is the pre-fixed inherited rate-head scale. No test-informed tuning."},
        "advantages_zh": "独立标量递推及梯度oracle、真实卷积与SSD输入捕获、冻结数据正常入口验证。",
        "limitations_zh": "工程检查验证合同实现与数值行为，正式六单元预测能力仍须实际训练和冻结评分。未重复整套训练。",
        "confidence_zh": "高，限已执行工程控制与逐文件SHA绑定范围。",
        "failure": failure,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": result["status"], "passed_checks": result["passed_checks"], "output": str(args.output)}, ensure_ascii=False))
    if failure:
        print(failure)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
