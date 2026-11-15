"""Verify the M4-local scan adjoint against unchanged baseline operators."""

import copy
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.unified_matr import legacy_operators as legacy
from models.unified_matr.methodology_full import MethodologyFullModel, model_registry
from models.unified_matr.methodology_operators import StableMaskedMamba2, StableRateFlowHead, dense_ssd, deterministic_cumsum
from models.unified_matr.model import RateFlowHead, UnifiedMATRModel, initialize
from models.unified_matr.selfcheck import make_batch


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def check_pair(label, first, last, first_inputs, last_inputs, dtype):
    atol, rtol = (1e-10, 1e-9) if dtype == torch.float64 else (2e-6, 2e-5)
    forward_equal = torch.equal(first, last)
    if not forward_equal:
        raise AssertionError("Forward arithmetic changed at " + label)
    first_grad = torch.autograd.grad(first.square().mean(), first_inputs)
    last_grad = torch.autograd.grad(last.square().mean(), last_inputs)
    if not all(torch.allclose(a, b, atol=atol, rtol=rtol) for a, b in zip(first_grad, last_grad)):
        raise AssertionError("Operator adjoint differs beyond tolerance at " + label)
    return {"check": label, "status": "PASS", "forward_bitwise_equal": forward_equal,
            "maximum_gradient_difference": max(float((a - b).abs().max()) for a, b in zip(first_grad, last_grad))}


def operator_cases(device, dtype):
    generator = torch.Generator().manual_seed(42)
    random = lambda *shape: torch.randn(*shape, generator=generator, dtype=dtype).to(device)
    value = random(2, 9, 4).requires_grad_()
    other = value.detach().clone().requires_grad_()
    rows = [check_pair(f"cumsum/{device}/{dtype}", legacy.deterministic_cumsum(value, 1), deterministic_cumsum(other, 1), [value], [other], dtype)]
    values = [random(2, 9, 2, 4), random(2, 9, 2).sigmoid() * .1, -random(2).abs(),
              random(2, 9, 3), random(2, 9, 3), random(2)]
    first = [value.clone().requires_grad_() for value in values]
    last = [value.clone().requires_grad_() for value in values]
    rows.append(check_pair(f"SSD/{device}/{dtype}", legacy.dense_ssd(*first), dense_ssd(*last), first, last, dtype))
    original = initialize(42, "local", lambda: legacy.MaskedMamba2(state=4, expand=1)).to(device=device, dtype=dtype)
    repaired = StableMaskedMamba2(copy.deepcopy(original))
    mask = torch.ones(2, 8, dtype=torch.bool, device=device)
    mask[0, 2] = False
    raw, reference = random(2, 8, 64).requires_grad_(), random(2, 8, 64).requires_grad_()
    raw_other, ref_other = raw.detach().clone().requires_grad_(), reference.detach().clone().requires_grad_()
    rows.append(check_pair(f"local_reference_mixer/{device}/{dtype}", original(raw, mask, reference), repaired(raw_other, mask, ref_other),
                           [raw, reference, *original.parameters()], [raw_other, ref_other, *repaired.parameters()], dtype))
    original_head = initialize(42, "rate_head", RateFlowHead).to(device=device, dtype=dtype)
    repaired_head = initialize(42, "rate_head", StableRateFlowHead).to(device=device, dtype=dtype)
    state, condition = random(2, 16).requires_grad_(), random(2, 16).requires_grad_()
    state_other, cond_other = state.detach().clone().requires_grad_(), condition.detach().clone().requires_grad_()
    queries = torch.tensor([[100, 101, 200, 200.5, 500, 1000, 5000]] * 2, dtype=dtype, device=device)
    first_output = torch.cat(original_head(state, condition, queries), -1)
    last_output = torch.cat(repaired_head(state_other, cond_other, queries), -1)
    rows.append(check_pair(f"rate_integral/{device}/{dtype}", first_output, last_output,
                           [state, condition, *original_head.parameters()], [state_other, cond_other, *repaired_head.parameters()], dtype))
    return rows


def main():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    checks = [row for device in ("cpu", "cuda") for dtype in (torch.float64, torch.float32) for row in operator_cases(device, dtype)]
    models = []
    for device in ("cpu", "cuda"):
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        model = MethodologyFullModel(15, 20, 26).to(device)
        batch = make_batch(device=device)
        output = model(**batch, return_diagnostics=True)
        (output["trajectory"].mean() + .001 * output["persistence_penalty"]).backward()
        gradients = [parameter.grad for parameter in model.parameters()]
        if not all(gradient is not None and bool(torch.isfinite(gradient).all()) for gradient in gradients):
            raise AssertionError("A model gradient is missing or nonfinite")
        if not all(bool(model.history[index].mixer.write_projection.weight.grad.abs().max() > 0) for index in range(2)):
            raise AssertionError("Zero-initialized history write projection has no gradient")
        models.append({"device": device, "registry": model_registry(model), "finite_gradient_tensors": len(gradients),
                       "WB_gradient_max_abs": [float(model.history[index].mixer.write_projection.weight.grad.abs().max()) for index in range(2)],
                       "trajectory_shape": list(output["trajectory"].shape), "queries": {"count": 4902, "minimum": 100, "maximum": 5000, "additional_off_grid": 200.5}})
    original = UnifiedMATRModel("M3", 15, 20, 26)
    repaired = MethodologyFullModel(15, 20, 26)
    baseline_state, m4_state = original.state_dict(), repaired.state_dict()
    common = [name for name in baseline_state.keys() & m4_state.keys() if baseline_state[name].shape == m4_state[name].shape]
    if not all(torch.equal(baseline_state[name], m4_state[name]) for name in common):
        raise AssertionError("Shared initial baseline parameter values changed")
    result_root = ROOT / "results/unified_matr_methodology_full_seed42"
    before = json.loads((result_root / "engineering_diagnostics/before_scan_fix_repeat6.json").read_text())
    after = json.loads((result_root / "engineering_diagnostics/after_all_scan_fix_repeat20.json").read_text())
    if before["records"][0]["prediction_sha256"] != after["records"][0]["prediction_sha256"] or before["records"][0]["loss"] != after["records"][0]["loss"]:
        raise AssertionError("Real first-batch forward arithmetic changed after the scan repair")
    if not after["all_forward_loss_and_gradients_exact"]:
        raise AssertionError("Repaired real first-batch repeatability failed")
    result = {"status": "PASS", "scope": "mathematically equivalent M4-local scan adjoints and normal model engineering",
              "optimizer_updates": 0, "formal_updates": 0, "checks": checks, "models": models,
              "common_baseline_initial_tensors_exact": len(common), "real_first_batch_forward_before_after_exact": True,
              "real_fresh_first_batch_repetitions_exact": len(after["records"]),
              "repair_scope": "Local/reference SSD, history SSD, dynamic EMA and rate integral use a CPU forward/reverse adjoint inside a single CUDA autograd node. Baseline source files remain unchanged.",
              "model_sources": [{"path": name, "sha256": digest(ROOT / name)} for name in
                                ("models/unified_matr/methodology_full.py", "models/unified_matr/methodology_operators.py")],
              "source_sha256": digest(Path(__file__))}
    target = result_root / "MODEL_ENGINEERING_QA_AFTER_SCAN_FIX.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "operator_cases": len(checks), "model_devices": len(models),
                      "common_initial_tensors": len(common), "real_repetitions_exact": len(after["records"]), "optimizer_updates": 0}))


if __name__ == "__main__":
    main()
