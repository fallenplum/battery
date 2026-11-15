"""Independent literal-loop oracle for M4 dynamic persistence, without fitting."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
from models.unified_matr.methodology_operators import dynamic_ema


def loop_oracle(value, rho, valid, cycles):
    state = torch.zeros_like(value[:, 0])
    previous_cycle = torch.zeros_like(cycles[:, 0])
    seen = torch.zeros(value.shape[0], dtype=torch.bool, device=value.device)
    outputs, internals, penalties = [], [], []
    count = valid.sum().new_zeros(())
    for position in range(value.shape[1]):
        observed = valid[:, position]
        gap = (cycles[:, position] - previous_cycle).clamp_min(1).to(value.dtype)
        current = torch.where(observed[:, None], value[:, position], 0.0)
        retained = rho[:, position].pow(gap[:, None])
        proposal = torch.where(seen[:, None], retained * state + (1 - retained) * current, current)
        transition = observed & seen
        change = torch.where(transition[:, None], (proposal - state) / gap[:, None], 0.0)
        penalties.append(change.square().sum())
        count = count + transition.sum()
        state = torch.where(observed[:, None], proposal, state)
        previous_cycle = torch.where(observed, cycles[:, position], previous_cycle)
        seen = seen | observed
        outputs.append(torch.where(observed[:, None], state, 0.0))
        internals.append(state)
    return torch.stack(outputs, 1), torch.stack(penalties).sum() / (count.clamp_min(1) * value.shape[-1]), torch.stack(internals, 1)


def compare(name, valid, cycles, dtype, device):
    generator = torch.Generator().manual_seed(42)
    value = torch.randn(*valid.shape, 64, generator=generator, dtype=dtype).to(device)
    rho = (torch.rand(*value.shape, generator=generator, dtype=dtype) * .98 + .01).to(device)
    if name == "retention_bounds":
        rho[..., :32] = .01
        rho[..., 32:] = .99
    valid, cycles = valid.to(device), cycles.to(device=device, dtype=dtype)
    inputs = [tensor.detach().clone().requires_grad_(True) for tensor in (value, rho, cycles)]
    reference_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in (value, rho, cycles)]
    state, penalty, diagnostic = dynamic_ema(inputs[0], inputs[1], valid, inputs[2])
    expected, expected_penalty, internal = loop_oracle(reference_inputs[0], reference_inputs[1], valid, reference_inputs[2])
    gradients = torch.autograd.grad(state.square().mean() + penalty * .1, inputs)
    expected_gradients = torch.autograd.grad(expected.square().mean() + expected_penalty * .1, reference_inputs)
    atol, rtol = (1e-10, 1e-9) if dtype == torch.float64 else (2e-6, 2e-5)
    pairs = [(state, expected), (penalty, expected_penalty), (diagnostic["internal_state"], internal)]
    pairs.extend(zip(gradients, expected_gradients))
    if not all(torch.allclose(first, last, atol=atol, rtol=rtol) for first, last in pairs):
        raise AssertionError(f"Dynamic EMA loop mismatch for {name}/{device}/{dtype}")
    altered_value, altered_rho = value.clone(), rho.clone()
    altered_value[:, 7:] += 100
    altered_rho[:, 7:] = .5
    altered, _, _ = dynamic_ema(altered_value, altered_rho, valid, cycles)
    if not torch.equal(state.detach()[:, :7], altered[:, :7]):
        raise AssertionError("Future-token changes altered earlier persistence states")
    return {"case": name, "device": device, "dtype": str(dtype), "status": "PASS",
            "state_max_abs_error": float((state - expected).abs().max()),
            "penalty_abs_error": float((penalty - expected_penalty).abs()),
            "input_gradient_max_abs_errors": [float((first - last).abs().max()) for first, last in zip(gradients, expected_gradients)],
            "future_token_prefix_invariance_exact": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPOSITORY / "results/unified_matr_methodology_full_seed42/EMA_ORACLE_QA.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    valid = torch.ones(3, 12, dtype=torch.bool)
    missing = valid.clone()
    missing[0, :3] = False
    missing[0, 7:9] = False
    missing[1, -3:] = False
    missing[2] = False
    single = torch.zeros_like(valid)
    single[0, 6] = True
    single[1, 0] = True
    single[2, -1] = True
    regular = torch.arange(1, 13)[None].expand(3, -1)
    uneven = torch.tensor([1, 2, 5, 9, 10, 17, 21, 23, 29, 31, 35, 40])[None].expand(3, -1)
    clamped = torch.tensor([1, 2, 2, 1, 8, 12, 12, 18, 19, 25, 31, 32])[None].expand(3, -1)
    cases = (("regular", valid, regular), ("missing_and_all_missing", missing, uneven),
             ("single_observation", single, uneven), ("retention_bounds", missing, uneven),
             ("nonincreasing_gap_clamp", valid, clamped))
    devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
    rows = [compare(name, mask, cycles, dtype, device) for device in devices
            for dtype in (torch.float64, torch.float32) for name, mask, cycles in cases]
    source = Path(__file__).read_bytes()
    result = {"status": "PASS", "scope": "independent_dynamic_ema_loop_oracle_forward_backward_and_causality",
              "optimizer_updates": 0, "formal_fits": 0, "checks": rows,
              "source_sha256": hashlib.sha256(source).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "cases": len(rows), "optimizer_updates": 0,
                      "state_max_abs_error": max(row["state_max_abs_error"] for row in rows),
                      "gradient_max_abs_error": max(max(row["input_gradient_max_abs_errors"]) for row in rows)}))


if __name__ == "__main__":
    main()
