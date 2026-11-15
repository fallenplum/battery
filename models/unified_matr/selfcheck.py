"""Deterministic synthetic engineering checks, separate from formal experiments."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from .legacy_operators import LegacyFlowHead, MambaBlock, analytic_flow, dense_ssd
from .model import ARMS, PERSISTENCE_ARMS, RATE_ARMS, UnifiedMATRModel, model_registry


def make_batch(policy_dim=15, global_dim=20, local_feature_dim=26, device="cpu"):
    generator = torch.Generator().manual_seed(422026)
    random = lambda *shape: torch.randn(*shape, generator=generator).to(device)
    mask = torch.ones(2, 100, 64, dtype=torch.bool, device=device)
    mask[0, 14:18] = False
    mask[1, 28, :16] = False
    raw = random(2, 100, 64, 3) * 0.1
    raw[..., 0] += torch.linspace(2.8, 3.3, 64, device=device)
    raw[..., 1] -= 4.0
    raw[..., 2] += torch.linspace(0.0, 0.5, 64, device=device)
    raw[~mask] = float("nan")
    global_mask = torch.ones(2, 100, global_dim, dtype=torch.bool, device=device)
    global_mask[1, 11, -1] = False
    local_mask = mask.clone()
    feature_mask = torch.ones(2, 100, 8, local_feature_dim, dtype=torch.bool, device=device)
    reference = raw * 0.8
    reference[..., 0] = raw[..., 0]
    return dict(raw_viq=raw, point_mask=mask,
                cycle_ids=torch.arange(1, 101, device=device)[None].expand(2, -1),
                reference_viq=reference, policy=random(2, policy_dim),
                policy_mask=torch.ones(2, policy_dim, dtype=torch.bool, device=device),
                global_features=random(2, 100, global_dim), global_mask=global_mask,
                local_residual=random(2, 100, 64), local_mask=local_mask,
                local_features=random(2, 100, 8, local_feature_dim), local_feature_mask=feature_mask,
                anchor_soh100=torch.tensor([0.97, 1.01], device=device),
                query_cycles=torch.cat((torch.arange(100, 201, device=device, dtype=torch.float32),
                                         torch.tensor([200.5], device=device),
                                         torch.arange(201, 5001, device=device, dtype=torch.float32))))


def tensor_clone(batch):
    return {name: value.clone() for name, value in batch.items()}


def check(condition, label, records, **details):
    if not condition:
        raise AssertionError(label)
    records.append(dict(check=label, status="PASS", **details))


def output_equal(first, second):
    return all(torch.equal(first[key], second[key]) for key in first if key != "context")


def check_operators(records, legacy_root=None):
    generator = torch.Generator().manual_seed(42)
    x = torch.randn(2, 7, 3, 4, generator=generator, dtype=torch.float64)
    delta = torch.rand(2, 7, 3, generator=generator, dtype=torch.float64) * 0.1
    a = -torch.rand(3, generator=generator, dtype=torch.float64)
    b = torch.randn(2, 7, 5, generator=generator, dtype=torch.float64)
    c = torch.randn(2, 7, 5, generator=generator, dtype=torch.float64)
    skip = torch.randn(3, generator=generator, dtype=torch.float64)
    state = x.new_zeros(2, 3, 4, 5)
    outputs = []
    for index in range(7):
        decay = torch.exp(delta[:, index] * a)
        write = delta[:, index, :, None, None] * x[:, index, :, :, None] * b[:, index, None, None]
        state = decay[..., None, None] * state + write
        outputs.append((state * c[:, index, None, None]).sum(-1) + x[:, index] * skip[None, :, None])
    dense = dense_ssd(x, delta, a, b, c, skip)
    error = float((dense - torch.stack(outputs, 1)).abs().max())
    check(error < 1e-12, "dense_ssd_matches_serial_FP64", records, maximum_absolute_error=error)
    if legacy_root is not None:
        source = Path(legacy_root) / "src/models/battery_mamba_prefix_future/model.py"
        specification = importlib.util.spec_from_file_location("unified_matr_legacy_core_check", source)
        module = importlib.util.module_from_spec(specification)
        previous_bytecode_setting = sys.dont_write_bytecode
        try:
            sys.dont_write_bytecode = True
            specification.loader.exec_module(module)
        finally:
            sys.dont_write_bytecode = previous_bytecode_setting
        reference = module.dense_ssd(x, delta, a, b, c, skip)
        check(torch.equal(dense, reference), "dense_ssd_old_source_bitwise_equal", records)
        signal_path = Path(legacy_root) / "scripts/signal_only_4lodo_epo/model.py"
        signal_tree = ast.parse(signal_path.read_text(encoding="utf-8"))
        masked_class = next(node for node in signal_tree.body if isinstance(node, ast.ClassDef) and node.name == "MaskedMamba2")
        namespace = dict(torch=torch, F=torch.nn.functional, DenseMamba2=module.DenseMamba2, dense_ssd=module.dense_ssd)
        exec(compile(ast.Module(body=[masked_class], type_ignores=[]), str(signal_path), "exec"), namespace)
        old_mixer = namespace["MaskedMamba2"](d_model=64, d_state=16, expand=2, headdim=16).double().eval()
        history = MambaBlock().double().eval()
        old_mixer.load_state_dict(history.mixer.state_dict())
        raw = torch.randn(2, 13, 64, generator=generator, dtype=torch.float64)
        valid = torch.ones(2, 13, dtype=torch.bool)
        valid[0, 4:7] = False
        normalized = torch.where(valid[..., None], history.norm(raw), 0.0)
        old_output = old_mixer(normalized, valid)
        new_output = history.mixer(normalized, valid)
        check(torch.equal(old_output, new_output), "history_masked_mamba_old_source_bitwise_equal", records)
        self_reference_path = Path(legacy_root) / "scripts/self_reference_selection/model.py"
        tree = ast.parse(self_reference_path.read_text(encoding="utf-8"))
        old_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SelfReferenceFlow")
        local_function = next(node for node in old_class.body if isinstance(node, ast.FunctionDef) and node.name == "local_block")
        namespace = dict(torch=torch, F=torch.nn.functional, dense_ssd=module.dense_ssd)
        exec(compile(ast.Module(body=[local_function], type_ignores=[]), str(self_reference_path), "exec"), namespace)
        local = MambaBlock(local=True).double().eval()
        reference_tokens = torch.randn(2, 13, 64, generator=generator, dtype=torch.float64)
        old_owner = SimpleNamespace(local=local, sr_arm="SR-111", capture_selection=False)
        old_output = namespace["local_block"](old_owner, raw, reference_tokens, valid)
        new_output = local(raw, valid, reference_tokens)
        check(torch.equal(old_output, new_output), "SR111_Delta_B_C_routing_old_source_bitwise_equal", records)
    state = torch.randn(2, 16, generator=generator, dtype=torch.float64)
    rate = torch.rand(2, 16, generator=generator, dtype=torch.float64)
    mean = torch.randn(2, 16, generator=generator, dtype=torch.float64)
    first, second = 0.23, 0.77
    composed = analytic_flow(analytic_flow(state, first, rate, mean), second, rate, mean)
    direct = analytic_flow(state, first + second, rate, mean)
    check(torch.allclose(composed, direct, atol=1e-14, rtol=1e-14), "old_analytic_flow_semigroup", records)
    head = LegacyFlowHead().double()
    condition = torch.randn(2, 16, generator=generator, dtype=torch.float64)
    query = torch.tensor([[0.0, 0.1, 0.9], [0.0, 0.2, 0.8]], dtype=torch.float64)
    output = head(state, condition, query)
    check(bool((output[:, 0].abs() < 1e-15).all()), "legacy_flow_exact_boundary_tolerance_FP64", records)
    if legacy_root is not None:
        flow_path = Path(legacy_root) / "scripts/state_dynamics_shared_flow/model.py"
        tree = ast.parse(flow_path.read_text(encoding="utf-8"))
        selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                    and node.name in ("flow", "rate_bias", "Generator")]
        namespace = dict(torch=torch, nn=torch.nn, F=torch.nn.functional, math=math)
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(flow_path), "exec"), namespace)
        old_generator = namespace["Generator"](16).double()
        old_generator.net.load_state_dict(head.generator.state_dict())
        old_rate, old_mean = old_generator(condition)
        future = namespace["flow"](state[:, None], query[..., None], (old_rate[:, None], old_mean[:, None]))
        correction = head.decoder(future).squeeze(-1) - head.decoder(state).squeeze(-1)[:, None]
        reference = torch.where(query == 0.0, 0.0, 0.2 * correction)
        check(torch.equal(output, reference), "analytic_flow_head_old_source_bitwise_equal_native_conversion", records)


def run_checks(device="cpu", legacy_root=None):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    records = []
    check_operators(records, legacy_root)
    batch = make_batch(device=device)
    models = {arm: UnifiedMATRModel(arm, 15, 20, 26).to(device).eval() for arm in ARMS}
    registries = [model_registry(model) for model in models.values()]
    for left_arm, right_arm in itertools.combinations(ARMS, 2):
        left, right = models[left_arm].state_dict(), models[right_arm].state_dict()
        shared = [key for key in left.keys() & right.keys() if left[key].shape == right[key].shape]
        check(all(torch.equal(left[key], right[key]) for key in shared),
              f"common_initial_state_values_{left_arm}_{right_arm}", records, common_tensors=len(shared))
    check(len({entry["parameters"] for entry in registries if entry["arm"] in ("M1", "M2", "M3", "M3-NP", "M3-NR")}) == 1,
          "five_full_information_arms_exact_parameter_match", records)
    check(all(entry["flow_head_parameters"] == 2752 for entry in registries), "analytic_rate_flow_heads_2752_active_parameters", records)
    full = next(entry["parameters"] for entry in registries if entry["arm"] == "M3")
    global_only = next(entry["parameters"] for entry in registries if entry["arm"] == "M3-NL")
    check(abs(full - global_only) <= 80, "NL_active_width_approximate_budget_match", records,
          full_parameters=full, global_only_parameters=global_only, delta=global_only - full)
    outputs = {}
    for arm, model in models.items():
        output = model(**batch, return_context=True)
        context = output.pop("context")
        outputs[arm] = output
        check(all(bool(torch.isfinite(value).all()) for value in output.values()), f"{arm}_finite_forward", records)
        check(output["trajectory"].shape == (2, len(batch["query_cycles"])), f"{arm}_full_query_shape", records)
        check(bool((output["trend"][:, 0] - batch["anchor_soh100"]).abs().max() < 1e-6), f"{arm}_trend_shared_observed_anchor", records)
        check(bool(output["observation_residual"].abs().max() <= 0.03), f"{arm}_observation_bound", records)
        state, condition = context["state"], context["condition"]
        tau = (context["queries"] - 100) / 1000.0
        observation_input = torch.cat((state[:, None].expand(-1, tau.shape[1], -1),
                                       condition[:, None].expand(-1, tau.shape[1], -1), tau[..., None]), -1)
        expected_residual = 0.03 * model.observation_head(observation_input).squeeze(-1).tanh()
        check(torch.equal(output["observation_residual"], expected_residual), f"{arm}_common_bounded_observation_formula_without_time_ramp", records)
        if arm in RATE_ARMS:
            check(bool((output["trend"][:, 1:] <= output["trend"][:, :-1]).all()), f"{arm}_monotone_trend", records)
            check(bool((output["rate_per_1000_cycles"] >= 0).all()), f"{arm}_nonnegative_rate", records)
        if arm not in PERSISTENCE_ARMS:
            check(float(output["persistence_penalty"]) == 0.0, f"{arm}_no_persistence_penalty", records)
        if torch.device(device).type == "cuda":
            cpu_model = UnifiedMATRModel(arm, 15, 20, 26).eval()
            cpu_batch = {name: value.cpu() for name, value in batch.items()}
            with torch.no_grad():
                cpu_output = cpu_model(**cpu_batch)
            error = max(float((output[name].detach().cpu() - cpu_output[name]).abs().max()) for name in output)
            check(error <= 1e-5, f"{arm}_CPU_CUDA_prediction_parity", records, maximum_absolute_error=error)
        loss = output["trajectory"].mean() + 1e-3 * output["persistence_penalty"]
        loss.backward()
        inactive = [name for name, parameter in model.named_parameters()
                    if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()) or float(parameter.grad.abs().sum()) == 0.0]
        check(not inactive, f"{arm}_all_registered_parameters_receive_nonzero_finite_gradient", records, inactive=inactive)
    check(output_equal(outputs["M2"], outputs["M3-NR"]), "M2_NR_duplicate_graph_initial_outputs", records)
    altered = tensor_clone(batch)
    for key in ("raw_viq", "reference_viq", "local_residual", "local_features"):
        altered[key].fill_(float("nan"))
    altered["local_mask"] = ~altered["local_mask"]
    altered["local_feature_mask"] = ~altered["local_feature_mask"]
    changed_nl = models["M3-NL"](**altered)
    check(output_equal(outputs["M3-NL"], changed_nl), "NL_raw_reference_local_values_and_local_masks_bitwise_invariant", records)
    altered = tensor_clone(batch)
    for key in ("global_features", "local_residual", "local_features"):
        altered[key].fill_(float("nan"))
    altered["global_mask"] = ~altered["global_mask"]
    changed_m0 = models["M0"](**altered)
    check(output_equal(outputs["M0"], changed_m0), "M0_explicit_physical_inputs_bitwise_invariant", records)
    altered = tensor_clone(batch)
    altered["raw_viq"][~altered["point_mask"]] = 1e20
    altered["reference_viq"][~altered["point_mask"]] = -1e20
    altered["global_features"][~altered["global_mask"]] = float("nan")
    altered["local_residual"][~altered["local_mask"]] = float("nan")
    changed = models["M3"](**altered)
    check(output_equal(outputs["M3"], changed), "invalid_values_do_not_enter_supported_predictions", records)
    empty = tensor_clone(batch)
    empty["point_mask"].fill_(False)
    empty["raw_viq"].fill_(float("nan"))
    empty["reference_viq"].fill_(float("nan"))
    for arm, model in models.items():
        empty_output = model(**empty)
        check(all(bool(torch.isfinite(value).all()) for value in empty_output.values()), f"{arm}_all_invalid_prefix_finite", records)
    invalid = tensor_clone(batch)
    invalid["cycle_ids"][0, -1] = 101
    try:
        models["M3"](**invalid)
    except ValueError:
        rejected = True
    else:
        rejected = False
    check(rejected, "future_cycle_as_inference_input_rejected", records)
    invalid = tensor_clone(batch)
    invalid["cycle_ids"][0, -1] = 99
    try:
        models["M3"](**invalid)
    except ValueError:
        rejected = True
    else:
        rejected = False
    check(rejected, "duplicate_or_unsorted_supported_cycle_rejected", records)
    return dict(status="PASS", scope="synthetic_engineering_selfcheck_only", formal_training=False,
                optimizer_updates=0, seed=42, device=device, checks=len(records), records=records,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                query_fixture="dense integer100..5000 plus off-grid200.5; forward and backward allseven arms",
                dimension_fixture=dict(policy_dim=15, global_dim=20, local_feature_dim=26), models=registries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--legacy-root")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_checks(args.device, args.legacy_root)
    content = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(content, encoding="utf-8")
    print(json.dumps(dict(status=result["status"], checks=result["checks"], scope=result["scope"], device=result["device"])))


if __name__ == "__main__":
    main()
