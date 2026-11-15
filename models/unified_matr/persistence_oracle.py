"""Independent literal loop oracle for the parameter-preserving EMA vectorization."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from .model import SoftPersistence


def loop_reference(value, valid, cycle_ids, retention_logits, persistent=True):
    """Preserve the original loop, including gapclamp1, first support and penalty."""
    retain = retention_logits.sigmoid()
    if not persistent:
        return value * retain, value.new_zeros(())
    state = value.new_zeros(len(value), 64)
    last_cycle = cycle_ids.new_zeros(len(value))
    seen = torch.zeros(len(value), device=value.device, dtype=torch.bool)
    outputs, penalties = [], []
    for index in range(value.shape[1]):
        gap = (cycle_ids[:, index] - last_cycle).clamp_min(1).to(value.dtype)
        weight = retain[None].pow(gap[:, None])
        proposal = torch.where(seen[:, None], weight * state + (1 - weight) * value[:, index], value[:, index])
        change = (proposal - state) / gap[:, None]
        penalties.append(torch.where((valid[:, index] & seen)[:, None], change.square(), 0.0))
        state = torch.where(valid[:, index, None], proposal, state)
        outputs.append(torch.where(valid[:, index, None], state, 0.0))
        last_cycle = torch.where(valid[:, index], cycle_ids[:, index], last_cycle)
        seen = seen | valid[:, index]
    transitions = (valid.sum(1) - valid.any(1).to(torch.long)).sum()
    denominator = transitions.clamp_min(1) * value.shape[-1]
    return torch.stack(outputs, 1), torch.stack(penalties, 1).sum() / denominator


def compare_case(name, values, valid, cycles, logits, dtype, device, persistent=True):
    model = SoftPersistence().to(device=device, dtype=dtype)
    with torch.no_grad():
        model.retention_logits.copy_(logits.to(device=device, dtype=dtype))
    left_value = values.to(device=device, dtype=dtype).detach().requires_grad_(True)
    right_value = left_value.detach().clone().requires_grad_(True)
    left_cycles = cycles.to(device=device, dtype=dtype).detach().requires_grad_(True)
    right_cycles = left_cycles.detach().clone().requires_grad_(True)
    left_logits = model.retention_logits.detach().clone().requires_grad_(True)
    valid = valid.to(device)
    expected_state, expected_penalty = loop_reference(left_value, valid, left_cycles, left_logits, persistent)
    actual_state, actual_penalty = model(right_value, valid, right_cycles, persistent)
    generator = torch.Generator().manual_seed(42)
    objective = torch.randn(values.shape, generator=generator, dtype=torch.float64).to(device=device, dtype=dtype)
    left_loss = (expected_state * objective).mean() + 0.37 * expected_penalty
    right_loss = (actual_state * objective).mean() + 0.37 * actual_penalty
    left_grad = torch.autograd.grad(left_loss, (left_value, left_cycles, left_logits), allow_unused=True)
    right_grad = torch.autograd.grad(right_loss, (right_value, right_cycles, model.retention_logits), allow_unused=True)
    tolerance = dict(atol=1e-10, rtol=1e-9) if dtype == torch.float64 else dict(atol=2e-6, rtol=2e-5)
    comparisons = {"states": (expected_state, actual_state), "penalty": (expected_penalty, actual_penalty)}
    for label, expected, actual, prototype in zip(("value_gradient", "cycle_gradient", "retain_logits_gradient"),
                                                 left_grad, right_grad, (left_value, left_cycles, left_logits)):
        comparisons[label] = (torch.zeros_like(prototype) if expected is None else expected,
                              torch.zeros_like(prototype) if actual is None else actual)
    details = {}
    for label, (expected, actual) in comparisons.items():
        finite = bool(torch.isfinite(expected).all() and torch.isfinite(actual).all())
        error = float((expected - actual).abs().max())
        passed = finite and torch.allclose(expected, actual, **tolerance)
        details[label] = {"status": "PASS" if passed else "FAIL", "maximum_absolute_error": error, "finite": finite}
        if not passed:
            raise AssertionError(f"{name}/{dtype}/{device}/{label}: {details[label]}")
    return {"case": name, "dtype": str(dtype), "device": device, "persistent": persistent,
            "status": "PASS", "tolerance": tolerance, "comparisons": details}


def run_checks():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    generator = torch.Generator().manual_seed(42)
    values = torch.randn(3, 100, 64, generator=generator, dtype=torch.float64)
    regular = torch.arange(1, 101, dtype=torch.float64)[None].repeat(3, 1)
    full = torch.ones(3, 100, dtype=torch.bool)
    irregular = torch.randint(1, 5, (3, 100), generator=generator).cumsum(1).double()
    mask = torch.rand(3, 100, generator=generator) > 0.3
    mask[:, :7], mask[:, -9:] = False, False
    mixed = mask.clone()
    mixed[0] = False
    mixed[1] = False
    mixed[1, 27] = True
    nonmonotone = regular.clone()
    nonmonotone[:, 12:19] -= 15
    nonmonotone[:, 42:44] = nonmonotone[:, 41:42]
    logits = torch.full((64,), torch.log(torch.tensor(9.0, dtype=torch.float64)))
    extreme = logits.clone()
    extreme[:8] = torch.tensor([-1000., -100., -30., -2., 0., 15., 50., 1000.])
    cases = [("all_valid_regular", full, regular, logits),
             ("missing_first_last_and_internal_nonuniform", mask, irregular, logits),
             ("all_missing", torch.zeros_like(full), irregular, logits),
             ("mixed_all_missing_single_support_and_irregular", mixed, irregular, logits),
             ("repeated_reversed_and_negative_coordinates_gapclamp1", mask, nonmonotone, logits),
             ("saturated_zero_one_and_extreme_retention", mask, irregular, extreme)]
    records = []
    devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
    for device in devices:
        for dtype in (torch.float64, torch.float32):
            for name, valid, cycles, retain in cases:
                records.append(compare_case(name, values, valid, cycles, retain, dtype, device))
            records.append(compare_case("ordinary_framewise_control", values, mask, irregular, logits, dtype, device, persistent=False))
    maxima = {label: max(row["comparisons"][label]["maximum_absolute_error"] for row in records)
              for label in records[0]["comparisons"]}
    return {"status": "PASS", "scope": "synthetic_loop_oracle_equivalence_only", "optimizer_updates": 0,
            "scientific_results_read": False, "parameter_count": 64, "cases": len(records), "records": records,
            "maximum_absolute_errors_all_cases": maxima,
            "input_contract": "finite physical-response encoder tokens and finite cycle coordinates; missing support explicit in bool mask",
            "dtype_rounding": "Real-valued functions unchanged; FP32/F64 arithmetic checked at predeclared tolerances",
            "strict_deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_checks()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "cases", "maximum_absolute_errors_all_cases", "optimizer_updates")}))
