"""Independent single-change, input-removal and reproducibility controls for A1-A4."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from unittest.mock import patch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.unified_matr import methodology_operators as operators
from models.unified_matr.legacy_operators import LegacyFlowHead
from models.unified_matr.methodology_ablations import ABLATION_ARMS, MethodologyAblationModel
from models.unified_matr.methodology_full import MethodologyFullModel
from models.unified_matr.model import initialize
from models.unified_matr.selfcheck import tensor_clone
from scripts.unified_matr import train_methodology_ablations as runner
import verify_m4_method_contract as full_qa

check = full_qa.check
close = full_qa.close
error = full_qa.maximum_error
digest = full_qa.digest


def model_for(arm, device):
    return MethodologyAblationModel(arm, 15, 20, 26).to(device).eval()


def verify_shared_initialization(records, arm, model, baseline):
    original = baseline.state_dict()
    current = model.state_dict()
    allowed = {
        ABLATION_ARMS[0]: ("flow_head.",),
        ABLATION_ARMS[1]: ("history.0.mixer.write_", "history.1.mixer.write_"),
        ABLATION_ARMS[2]: ("persistence.rho_net.",),
        ABLATION_ARMS[3]: ("physics.", "persistence.rho_net.0.",
                           "history.0.mixer.write_gate.", "history.1.mixer.write_gate."),
    }[arm]
    changed = []
    preserved = 0
    for name in sorted(set(original) | set(current)):
        same = name in original and name in current and torch.equal(original[name], current[name])
        if same:
            preserved += original[name].numel()
        else:
            changed.append(name)
            check(records, arm + "/only_registered_component_changes/" + name, name.startswith(allowed))
    check(records, arm + "/has_expected_structural_change", bool(changed),
          changed_state_keys=changed, shared_identical_state_elements=preserved)
    if arm == ABLATION_ARMS[3]:
        close(records, arm + "/global_combine_exact_M4_global_columns", model.physics.combine[0].weight,
              baseline.physics.combine[0].weight[:, :64], 0)
        columns = list(range(104)) + list(range(136, 172))
        close(records, arm + "/rho_exact_retained_M4_columns140", model.persistence.rho_net[0].weight,
              baseline.persistence.rho_net[0].weight[:, columns], 0)
        for layer in range(2):
            close(records, arm + f"/gate{layer}_exact_M4_state_columns64", model.history[layer].mixer.write_gate.weight,
                  baseline.history[layer].mixer.write_gate.weight[:, :64], 0)


def verify_a1(records, model, batch, baseline_output):
    output = model(**batch, return_context=True, return_diagnostics=True)
    for name in ("state", "condition"):
        close(records, "A1/unchanged_encoder_" + name, output["context"][name], baseline_output["context"][name], 0)
    legacy = initialize(42, "legacy_head", LegacyFlowHead).to(batch["raw_viq"].device)
    for name, value in legacy.state_dict().items():
        close(records, "A1/original_legacy_initialization/" + name, model.flow_head.state_dict()[name], value, 0)
    queries = batch["query_cycles"][None].expand(2, -1)
    tau = (queries - 100) / 1000
    correction = legacy(output["context"]["state"], output["context"]["condition"], tau)
    close(records, "A1/exact_original_Flow_prediction", output["raw_trend"], batch["anchor_soh100"][:, None] + correction, 0)
    close(records, "A1/compatibility_signed_loss_is_minus_correction", output["integrated_loss"], -correction, 0)
    head = copy.deepcopy(model.flow_head).double()
    state = output["context"]["state"].detach().double()
    condition = output["context"]["condition"].detach().double()
    query = torch.tensor([[101., 200.5, 500., 1000.]], device=state.device, dtype=torch.float64).expand(2, -1).clone().requires_grad_()
    loss, signed_rate = head(state, condition, query)
    query_derivative = torch.autograd.grad(loss.sum(), query)[0]
    close(records, "A1/signed_rate_matches_analytic_Flow_derivative_after100", signed_rate / 1000, query_derivative, 3e-13)
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
        head.decoder[0].weight[0, 0] = 1
        head.decoder[2].weight[0, 0] = -1
    state = torch.full((1, 16), .5, device=state.device, dtype=torch.float64)
    condition = torch.zeros_like(state)
    query = state.new_tensor([[100, 101, 200, 500]])
    loss, signed_rate = head(state, condition, query)
    check(records, "A1/control_demonstrates_allowed_Flow_recovery", bool((-loss).diff(dim=1).gt(0).all()
          and (signed_rate < 0).all()), maximum_recovery=float((-loss).max()))


def verify_a2(records, model, baseline, batch, original):
    output = model(**batch, return_context=True, return_diagnostics=True)
    close(records, "A2/initial_prediction_exact_zero_WB_M4", output["trajectory"], original["trajectory"], 0)
    for name in ("state", "rho", "quality"):
        close(records, "A2/persistence_retained_" + name, output["diagnostics"]["persistence"][name],
              original["diagnostics"]["persistence"][name], 0)
    altered_full = copy.deepcopy(baseline)
    with torch.no_grad():
        for block in altered_full.history:
            block.mixer.write_projection.weight.fill_(.02)
    altered = altered_full(**batch, return_diagnostics=True)
    check(records, "A2/M4_controlled_WB_reaches_B", all(bool((item["write_modulation"].abs() > 0).any())
          for item in altered["diagnostics"]["history"]))
    check(records, "A2/actual_removed_B_parameters", all("write_" not in name for name, _ in model.named_parameters()))
    check(records, "A2/retained_dynamic_rho_varies", float(output["diagnostics"]["persistence"]["rho"].std()) > 0)
    handle = model.selector_condition.register_forward_hook(lambda _, args, result: torch.zeros_like(result))
    selector_removed = model(**batch)
    handle.remove()
    change = error(selector_removed["trajectory"], output["trajectory"])
    check(records, "A2/selector_condition_still_affects_normal_prediction", change > 0, maximum_absolute_change=change)


def verify_a3(records, model, batch):
    output = model(**batch, return_diagnostics=True)
    rho = output["diagnostics"]["persistence"]["rho"]
    close(records, "A3/static_channel_rho_initialized09", rho, torch.full_like(rho, .9), 1e-7)
    check(records, "A3/exactly64_learnable_rho_parameters", sum(p.numel() for p in model.persistence.rho_net.parameters()) == 64)
    changed = tensor_clone(batch)
    changed["global_features"] += 100
    changed["local_features"] *= -5
    changed["policy"] += 20
    changed["local_mask"] = ~changed["local_mask"]
    changed_output = model(**changed, return_diagnostics=True)
    close(records, "A3/rho_invariant_to_response_quality_policy", changed_output["diagnostics"]["persistence"]["rho"], rho, 0)
    valid = torch.tensor([[False, True, False, True]], device=rho.device)
    values = torch.randn(1, 4, 64, device=rho.device)
    cycles = torch.tensor([[0, 2, 0, 9]], device=rho.device)
    fixed_rho = model.persistence.rho_net.logits.sigmoid().clamp(.01, .99).expand_as(values)
    state, _, diagnostic = operators.dynamic_ema(values, fixed_rho, valid, cycles)
    expected = .9 ** 7 * values[0, 1] + (1 - .9 ** 7) * values[0, 3]
    close(records, "A3/actual_gap_remains_in_retention_exponent", state[0, 3], expected, 4e-7)
    close(records, "A3/first_valid_initialization", state[0, 1], values[0, 1], 0)
    output["trajectory"].sum().backward()
    gradient = model.persistence.rho_net.logits.grad
    check(records, "A3/static_logits_receive_gradient", gradient is not None and float(gradient.norm()) > 0,
          gradient_l2=None if gradient is None else float(gradient.norm()))
    model.zero_grad(set_to_none=True)


def verify_a4(records, model, baseline, batch):
    valid = batch["point_mask"].any(-1)
    global_mask = batch["global_mask"] & valid[..., None]
    values = torch.where(global_mask, batch["global_features"], 0)
    global_token = baseline.physics.global_encoder(torch.cat((values, global_mask.to(values.dtype)), -1))
    expected = baseline.physics.combine(torch.cat((global_token, torch.zeros_like(global_token)), -1))
    expected = torch.where((valid & global_mask.any(-1))[..., None], expected, 0)
    actual, _ = model.physics(batch["global_features"], batch["global_mask"], batch["local_residual"],
                              batch["local_mask"], batch["local_features"], batch["local_feature_mask"], valid)
    check(records, "A4/FP32_reduced_GEMM_rounding_difference_recorded", bool(torch.isfinite(actual).all()
          and torch.isfinite(expected).all()), maximum_absolute_error=error(actual, expected),
          explanation="The128-input and64-input CUDA GEMMs may round differently. Mathematical equivalence is checked in FP64 below.")
    with torch.no_grad():
        original_physics = copy.deepcopy(baseline.physics).double()
        reduced_physics = copy.deepcopy(model.physics).double()
        double_values = torch.where(global_mask, batch["global_features"].double(), 0)
        double_token = original_physics.global_encoder(torch.cat((double_values, global_mask.double()), -1))
        double_expected = original_physics.combine(torch.cat((double_token, torch.zeros_like(double_token)), -1))
        double_expected = torch.where((valid & global_mask.any(-1))[..., None], double_expected, 0)
        double_actual, _ = reduced_physics(batch["global_features"].double(), batch["global_mask"],
                                           batch["local_residual"], batch["local_mask"], batch["local_features"],
                                           batch["local_feature_mask"], valid)
    close(records, "A4/physics_equivalent_M4_zero_local_token_FP64_oracle", double_actual, double_expected, 2e-12)
    output = model(**batch, return_diagnostics=True)
    check(records, "A4/local_quality_all_zero", bool((output["diagnostics"]["persistence"]["quality"] == 0).all()))
    changed = tensor_clone(batch)
    changed["local_residual"].fill_(float("nan"))
    changed["local_features"].fill_(float("nan"))
    changed["local_mask"] = ~changed["local_mask"]
    changed["local_feature_mask"] = ~changed["local_feature_mask"]
    other = model(**changed, return_diagnostics=True)
    for name in ("trajectory", "raw_trend", "persistence_penalty"):
        close(records, "A4/local_values_masks_noise_completely_removed_" + name, other[name], output[name], 0)
    for name in ("raw_viq", "reference_viq", "global_features", "policy"):
        changed = tensor_clone(batch)
        changed[name] = changed[name] + .5
        other = model(**changed)
        change = error(other["trajectory"], output["trajectory"])
        check(records, "A4/retained_input_really_affects_prediction_" + name, change > 0, maximum_absolute_change=change)
    gradient_batch = tensor_clone(batch)
    for name in ("local_residual", "local_features"):
        gradient_batch[name].requires_grad_()
    model(**gradient_batch)["trajectory"].sum().backward()
    for name in ("local_residual", "local_features"):
        gradient = gradient_batch[name].grad
        check(records, "A4/removed_input_zero_gradient_" + name, gradient is None or bool((gradient == 0).all()),
              gradient_is_none=gradient is None)
    model.zero_grad(set_to_none=True)


def verify_common(records, arm, model, batch):
    captured = []
    original_ssd = operators.dense_ssd

    def capture_ssd(x, delta, a, b, c, skip):
        if x.shape[1] == 100:
            captured.append(b)
        return original_ssd(x, delta, a, b, c, skip)

    with torch.no_grad():
        for block in model.history:
            if hasattr(block.mixer, "write_projection"):
                block.mixer.write_projection.weight.fill_(.02)
    with patch.object(operators, "dense_ssd", capture_ssd):
        output = model(**batch, return_context=True, return_diagnostics=True)
    for layer, diagnostic in enumerate(output["diagnostics"]["history"]):
        close(records, arm + f"/actual_history{layer}_SSD_receives_Bprime", captured[layer], diagnostic["Bprime"], 0)
        if arm == ABLATION_ARMS[1]:
            close(records, arm + f"/Bprime_equals_B_layer{layer}", diagnostic["Bprime"], diagnostic["B"], 0)
        else:
            check(records, arm + f"/B_modulation_retained_layer{layer}", bool((diagnostic["write_modulation"].abs() > 0).any()))
    changed = tensor_clone(batch)
    for name in ("raw_viq", "reference_viq", "global_features", "local_residual", "local_features"):
        changed[name][:, 60:] += 10
    other = model(**changed, return_diagnostics=True)
    for name in ("state", "rho"):
        close(records, arm + "/later_prefix_cannot_change_earlier_persistence_" + name,
              other["diagnostics"]["persistence"][name][:, :60], output["diagnostics"]["persistence"][name][:, :60], 0)
    for layer in range(2):
        close(records, arm + f"/later_prefix_cannot_change_earlier_B_layer{layer}",
              other["diagnostics"]["history"][layer]["Bprime"][:, :60],
              output["diagnostics"]["history"][layer]["Bprime"][:, :60], 0)
    leading_missing = tensor_clone(batch)
    leading_missing["point_mask"][:, :3] = False
    initialized = model(**leading_missing, return_diagnostics=True)["diagnostics"]["persistence"]
    close(records, arm + "/first_valid_after_leading_missing_initializes_current_response",
          initialized["state"][:, 3], initialized["current_response"][:, 3], 0)
    close(records, arm + "/internal_state_holds_across_invalid_records",
          initialized["internal_state"][0, 14:18], initialized["internal_state"][0, 13:14].expand(4, -1), 2e-6)
    valid = batch["point_mask"].any(-1)
    gradient_batch = tensor_clone(batch)
    names = ("raw_viq", "reference_viq", "global_features", "local_residual", "local_features")
    for name in names:
        gradient_batch[name].requires_grad_()
    result = model(**gradient_batch)
    loss = result["trajectory"].sum() + .001 * result["persistence_penalty"]
    loss.backward()
    for name in names:
        gradient = gradient_batch[name].grad
        check(records, arm + "/invalid_record_zero_gradient_" + name,
              gradient is None or bool((gradient[~valid] == 0).all()))
    parameter_gradients = {name: parameter.grad for name, parameter in model.named_parameters()}
    check(records, arm + "/all_reached_parameter_gradients_finite",
          all(value is None or bool(torch.isfinite(value).all()) for value in parameter_gradients.values()),
          total_parameters=sum(parameter.numel() for parameter in model.parameters()),
          trainable_parameters=sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
          reached_parameters=sum(parameter.numel() for name, parameter in model.named_parameters() if parameter_gradients[name] is not None),
          nonzero_gradient_parameters=sum(parameter.numel() for name, parameter in model.named_parameters()
                                          if parameter_gradients[name] is not None and bool(parameter_gradients[name].abs().any())))
    model.zero_grad(set_to_none=True)
    close(records, arm + "/anchor100", output["raw_trend"][:, 0], batch["anchor_soh100"], 0)
    check(records, arm + "/observation_bound_003", bool((output["observation_residual"].abs() <= .03).all()))
    queries = [5000, 200, 100, 200.5, 1000, 500, 101]
    other = model(**{**batch, "query_cycles": batch["query_cycles"].new_tensor(queries)})
    single = model(**{**batch, "query_cycles": batch["query_cycles"].new_tensor([200.5])})
    for name in ("trajectory", "raw_trend", "rate_per_cycle"):
        indices = [queries.index(float(value)) for value in batch["query_cycles"]]
        close(records, arm + "/query_reorder_" + name, output[name], other[name][:, indices], 4e-7)
        close(records, arm + "/query_single_joint_" + name, single[name][:, 0], other[name][:, 3], 4e-7)
    if arm != ABLATION_ARMS[0]:
        check(records, arm + "/retained_nonnegative_rate_and_monotonic_trend",
              bool((output["rate_per_cycle"] > 0).all() and (output["raw_trend"].diff(dim=1) <= 0).all()))
    invalid = tensor_clone(batch)
    invalid["cycle_ids"] = invalid["cycle_ids"].clone()
    invalid["cycle_ids"][:, -1] = 101
    try:
        model(**invalid)
    except ValueError:
        rejected = True
    else:
        rejected = False
    check(records, arm + "/future_cycle_rejected_as_prefix", rejected)


def verify_frozen_sources(records):
    m4 = json.loads((ROOT / "results/unified_matr_methodology_full_seed42/MODEL_CONTRACT.json").read_text(encoding="utf-8"))
    old = json.loads((ROOT / "results/unified_matr_seed42/runs/RUN_REGISTRATION.json").read_text(encoding="utf-8"))
    for label, contract in (("M4_eight_sources", m4), ("old43_sources", old)):
        mismatches = [item["path"] for item in contract["sources"] if digest(ROOT / item["path"]) != item["sha256"]]
        check(records, label + "/historical_sha_preserved", not mismatches, protected_count=len(contract["sources"]), mismatches=mismatches)
    return m4


def verify_data_and_reproducibility(records, device, data_root, m4_contract):
    data = runner.load_frozen_data(data_root)
    old_config = runner.read_json(ROOT / "configs/unified_matr_seed42.json")
    config = runner.read_json(ROOT / "configs/unified_matr_ablations_seed42.json")
    check(records, "training_and_reporting_exact_M4", config["training"] == old_config["training"]
          and config["reporting"] == old_config["reporting"] and config["training_seeds"] == [42])
    check(records, "frozen_data_identity_matches_M4", digest(data_root / "DATA_FREEZE.json") == m4_contract["data_freeze_sha256"]
          and digest(data_root / "dataset_fingerprint.sha256") == m4_contract["data_fingerprint_sha256"]
          and digest(data_root / "split_contract.json") == m4_contract["split_contract_sha256"])
    updates = {name: 150 * int(np.ceil(len(roles["train"]) / 8)) for name, roles in data[-2]["splits"].items()}
    check(records, "24_main_fits_49800_plus8400repeat_plus56smoke", sum(updates.values()) * 4 == 49800
          and config["formal_optimizer_updates"] == 49800
          and config["full_training_reproduction"]["optimizer_updates_total"] == 8400
          and config["engineering_optimizer_updates_total"] == 56
          and config["total_executed_optimizer_updates"] == 58256, updates_per_arm=updates)
    check(records, "same_endpoint_auxiliary_loss_and_fold_local_scaler", runner.training_loss is runner.full.training_loss
          and runner.fold_scaler is runner.full.fold_scaler)
    identifiers = {uid: number for number, uid in enumerate(data[1])}
    training = np.array([identifiers[uid] for uid in data[-2]["splits"][runner.SPLITS[0]]["train"]])
    scaled, _ = runner.fold_scaler(data[0], training)
    rows = np.random.default_rng(42).permutation(training)[:8]
    torch.set_num_threads(4)
    repeat_count = 12 if device == "cuda" else 2
    for arm in ABLATION_ARMS:
        repeats = []
        for _ in range(repeat_count):
            runner.seed_everything()
            model = runner.make_model(arm, data[0], config["training"]).to(device).train()
            output = model(**runner.to_batch(scaled, rows, device), query_cycles=torch.as_tensor(data[4], device=device))
            loss = runner.training_loss(output, rows, data[5], data[6], data[7], data[8], config["training"], device)
            loss.backward()
            receipt = runner.full.gradient_receipt(model)
            check(records, arm + "/real_train_batch_gradients_finite_repeat" + str(len(repeats)),
                  all(item["finite"] for item in receipt["parameters"].values()))
            repeats.append({"prediction_sha256": hashlib.sha256(output["trajectory"].detach().cpu().numpy().tobytes()).hexdigest(),
                            "loss": float(loss.detach()), "gradient_sha256": receipt["sha256"]})
            del output, loss, model
        check(records, arm + "/fresh_real_train_first_batch_bitwise_reproducible",
              all(item == repeats[0] for item in repeats[1:]), fresh_repeats=repeat_count, repeats=repeats,
              optimizer_updates=0, test_observations_consumed=0)
    torch.set_num_threads(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--data", type=Path, default=ROOT / "data/unified_matr_seed42")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    sources = tuple(runner.DEPENDENCIES) + (
        "scripts/verify_m4_ablation_contract.py", "scripts/verify_m4_method_contract.py",
        "configs/unified_matr_ablations_seed42.json", "research/M4_ABLATIONS_SEED42_CONTRACT_ZH.md",
    )
    source_hashes = {name: digest(ROOT / name) for name in sources}
    records = []
    failure = None
    try:
        m4_contract = verify_frozen_sources(records)
        full_qa.verify_dynamic_scan(records, args.device)
        full_qa.verify_ssd_oracle(records, args.device)
        batch = full_qa.small_batch(args.device)
        baseline = MethodologyFullModel(15, 20, 26).to(args.device).eval()
        baseline_output = baseline(**batch, return_context=True, return_diagnostics=True)
        for arm in ABLATION_ARMS:
            model = model_for(arm, args.device)
            verify_shared_initialization(records, arm, model, baseline)
            if arm == ABLATION_ARMS[0]:
                verify_a1(records, model, batch, baseline_output)
            elif arm == ABLATION_ARMS[1]:
                verify_a2(records, model, baseline, batch, baseline_output)
            elif arm == ABLATION_ARMS[2]:
                verify_a3(records, model, batch)
            else:
                verify_a4(records, model, baseline, batch)
            verify_common(records, arm, model, batch)
            del model
        del baseline_output, baseline
        verify_data_and_reproducibility(records, args.device, args.data, m4_contract)
        check(records, "tested_source_hashes_unchanged", source_hashes == {name: digest(ROOT / name) for name in sources})
    except Exception:
        failure = traceback.format_exc()
    result = {
        "status": "PASS" if failure is None else "FAIL", "device": args.device,
        "passed_checks": sum(row["status"] == "PASS" for row in records), "checks": records,
        "source_sha256": source_hashes,
        "contract_sha256": digest(ROOT / "research/M4_ABLATIONS_SEED42_CONTRACT_ZH.md"),
        "optimizer_updates": 0, "test_observations_consumed": 0,
        "advantages_zh": "逐arm初始化差异核查、真实SSD输入捕获、移除信息零作用/梯度控制、保留路径扰动、真实训练首批复现。",
        "limitations_zh": "工程合成控制与真实训练首批支持实现范围，正式性能及完整训练复现需要实际完成。参数量差异按信息移除连接直接记录。",
        "confidence_zh": "高，限已执行工程控制与SHA绑定范围。",
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
