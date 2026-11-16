import copy
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pytest
import torch

from models.drs_peft.battery_baseline import (
    B3InputScaler, B3M1SharedHead, M1Encoder, reference_scaled_viq,
)
from models.unified_matr.model import UnifiedMATRModel
from scripts.drs_peft.data import load_dataset, SourceOnlyScaler
from scripts.drs_peft.episodes import load_registration


@pytest.fixture(scope="module")
def material():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    data = load_dataset("MATR")
    reg = load_registration()
    rows = data.rows(reg["source_cell_roles"]["source_train"])
    arrays = SourceOnlyScaler().fit(data, rows).transform(data)
    scaler = B3InputScaler().fit(arrays, rows, [data.ids[i] for i in rows])
    tensor_keys = ("x", "point_mask", "cycle_ids", "policy", "policy_mask", "anchor", "nominal_capacity_Ah",
                   "voltage_grid_V", "global_features", "global_mask", "local_residual", "local_mask",
                   "local_features", "local_feature_mask")
    early = {key: torch.from_numpy(arrays[key][rows[:2]].copy()) for key in tensor_keys}
    torch.manual_seed(42)
    model = B3M1SharedHead(28, scaler).eval()
    return data, rows, arrays, scaler, early, model


def test_source_only_scaler_and_physical_floors(material):
    data, rows, arrays, scaler, early, model = material
    mutated = dict(arrays)
    outside = np.asarray(sorted(set(range(len(data.ids)))-set(rows)))
    for key in ("x", "global_features", "local_residual", "local_features"):
        mutated[key] = arrays[key].copy()
        mutated[key][outside] = 1e6
    actual = B3InputScaler().fit(mutated, rows, scaler.training_ids)
    assert actual.to_dict() == scaler.to_dict()
    for record in scaler.statistics.values():
        assert np.isfinite(record["std"]).all()
        assert (np.asarray(record["std"]) >= np.asarray(record["physical_std_floor"])).all()


def test_reference_is_same_curve_scale_and_not_median_curve(material):
    data, rows, arrays, scaler, early, model = material
    raw, reference = reference_scaled_viq(early["x"], early["point_mask"], early["cycle_ids"])
    npmask = early["point_mask"].numpy()
    q = early["x"][..., 2].numpy()
    values = np.where(npmask[:, :10], q[:, :10], np.nan)
    span = np.nanquantile(values, .95, axis=-1)-np.nanquantile(values, .05, axis=-1)
    scale = np.nanmedian(span, axis=1)
    expected = raw.numpy().copy()
    expected[..., 1:] /= scale[:, None, None, None]
    np.testing.assert_allclose(reference.numpy(), expected, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(reference[..., 0], raw[..., 0], rtol=0, atol=0)


def test_encoder_summary_matches_original_m1(material):
    data, rows, arrays, scaler, early, model = material
    legacy = UnifiedMATRModel("M1", policy_dim=28, global_dim=20, local_feature_dim=26).eval()
    prepared = model.input_scaler(early)
    captured = []
    hook = legacy.condition_pool.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    try:
        with torch.no_grad():
            legacy(**prepared, anchor_soh100=early["anchor"], query_cycles=torch.tensor([100., 101., 500.]))
    finally:
        hook.remove()
    with torch.no_grad():
        actual = model.encoder(**prepared)
    torch.testing.assert_close(actual, captured[0], rtol=0, atol=0)


def test_shared_head_anchor_query_consistency_and_backward(material):
    data, rows, arrays, scaler, early, model = material
    query = torch.tensor([100., 101., 500., 1000., 5000.])
    model.zero_grad(set_to_none=True)
    out = model(early, query)
    torch.testing.assert_close(out[:, 0], early["anchor"], rtol=0, atol=0)
    assert torch.isfinite(out).all()
    with torch.no_grad():
        partial = model(early, query[[2, 4]])
    torch.testing.assert_close(partial, out[:, [2, 4]], rtol=2e-6, atol=2e-6)
    out[:, 1:].mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    assert not hasattr(model.encoder, "flow_head")
    assert not hasattr(model.encoder, "observation_head")


def test_masked_nonfinite_values_do_not_change_prediction(material):
    data, rows, arrays, scaler, early, model = material
    corrupted = {key: value.clone() for key, value in early.items()}
    corrupted["x"][~corrupted["point_mask"]] = float("nan")
    corrupted["policy"][~corrupted["policy_mask"]] = float("nan")
    for key, mask in (("global_features", "global_mask"), ("local_residual", "local_mask"),
                       ("local_features", "local_feature_mask")):
        corrupted[key][~corrupted[mask]] = float("nan")
    with torch.no_grad():
        expected = model(early, torch.tensor([101., 500., 1000.]))
        actual = model(corrupted, torch.tensor([101., 500., 1000.]))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_training_dropout_has_exact_rng_replay(material):
    data, rows, arrays, scaler, early, model = material
    training = copy.deepcopy(model).train()
    query = torch.tensor([101., 500., 1000.])
    state = torch.get_rng_state()
    first = training(early, query)
    torch.set_rng_state(state)
    replay = training(early, query)
    torch.testing.assert_close(first, replay, rtol=0, atol=0)
    first.mean().backward()
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in training.parameters())


def test_invalid_cycle_ids_are_rejected(material):
    data, rows, arrays, scaler, early, model = material
    for invalid in (1.5, 1.0, 101.0):
        bad = {key: value.clone() for key, value in early.items()}
        bad["cycle_ids"] = bad["cycle_ids"].float()
        bad["cycle_ids"][:, 1] = invalid
        with pytest.raises(ValueError, match="cycle"):
            model(bad, torch.tensor([101.]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cpu_cuda_shared_head_forward_and_gradients(material):
    data, rows, arrays, scaler, early, model = material
    cuda = copy.deepcopy(model).cuda().eval()
    cuda_early = {key: value.cuda() for key, value in early.items()}
    query = torch.tensor([100., 101., 500., 1000.])
    with torch.no_grad():
        cpu = model(early, query)
    gpu = cuda(cuda_early, query.cuda())
    torch.testing.assert_close(gpu.cpu(), cpu, rtol=2e-4, atol=2e-4)
    gpu[:, 1:].mean().backward()
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in cuda.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_dropout_has_exact_rng_replay(material):
    data, rows, arrays, scaler, early, model = material
    cuda = copy.deepcopy(model).cuda().train()
    cuda_early = {key: value.cuda() for key, value in early.items()}
    query = torch.tensor([101., 500., 1000.], device="cuda")
    cpu_state, cuda_state = torch.get_rng_state(), torch.cuda.get_rng_state()
    first = cuda(cuda_early, query)
    torch.set_rng_state(cpu_state)
    torch.cuda.set_rng_state(cuda_state)
    replay = cuda(cuda_early, query)
    torch.testing.assert_close(first, replay, rtol=0, atol=0)
    first.mean().backward()
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in cuda.parameters())
