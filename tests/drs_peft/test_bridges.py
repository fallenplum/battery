"""Engineering tests using the pinned official pretrained checkpoint."""
import copy
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pytest
import torch

from models.drs_peft.backbone import MoiraiBackbone, MODEL_SHA256, sha256_file
from models.drs_peft.bridges import (
    NativeBridge, CycleEmbeddingBridge, cycle_statistics, validate_prefix,
)


ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("DRS_PEFT_BACKBONE_DIR", str(ROOT.parents[1]/".cache_drs_peft/moirai_small")))


@pytest.fixture(scope="module", autouse=True)
def deterministic_execution():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


@pytest.fixture(scope="module")
def bridges():
    if not (CACHE/"model.safetensors").is_file():
        pytest.fail("Pinned official checkpoint is required for bridge engineering tests")
    torch.manual_seed(42)
    return {"native": NativeBridge(CACHE, policy_dim=3).eval(),
            "embedding": CycleEmbeddingBridge(CACHE, policy_dim=3).eval()}


def example(batch_size=2, device="cpu"):
    generator = torch.Generator().manual_seed(91)
    x = torch.randn(batch_size, 100, 64, 4, generator=generator)*.05
    x[..., 0] = torch.linspace(2.8, 3.3, 64)
    x[..., 1] -= 4.0
    x[..., 2] += .9
    mask = torch.ones(batch_size, 100, 64, dtype=torch.bool)
    mask[:, 20:23] = False
    mask[:, 30, :16] = False
    cycle = torch.arange(1, 101).float()[None].expand(batch_size, -1).clone()
    policy = torch.tensor([1., 4., .5])[None].expand(batch_size, -1).clone()
    policy_mask = torch.ones_like(policy, dtype=torch.bool)
    policy_mask[:, 1] = False
    anchor = torch.full((batch_size,), .97)
    return {key: value.to(device) for key, value in dict(
        x=x, point_mask=mask, cycle_ids=cycle, policy=policy,
        policy_mask=policy_mask, anchor=anchor).items()}


def test_authentic_weights_and_target_shapes(bridges):
    assert sha256_file(CACHE/"model.safetensors") == MODEL_SHA256
    model = bridges["native"].backbone
    assert model.pretrained
    assert model.original_parameter_count == 13_827_528
    assert len(model.linear_targets("qv")) == 12
    assert len(model.linear_targets("qkvo_ffn")) == 42
    assert all(tuple(layer.weight.shape) == (384, 384) for layer in model.linear_targets().values())
    assert not any(parameter.requires_grad for parameter in model.parameters())


@pytest.mark.parametrize("kind", ["native", "embedding"])
@pytest.mark.parametrize("batch_size", [1, 8])
def test_cpu_shapes(bridges, kind, batch_size):
    with torch.no_grad():
        state = bridges[kind](**example(batch_size))
    assert state.shape == (batch_size, 64)
    assert torch.isfinite(state).all()


@pytest.mark.parametrize("kind", ["native", "embedding"])
def test_invalid_values_policy_masks_and_batch_identity(bridges, kind):
    model = bridges[kind]
    original = example(3)
    corrupted = {key: value.clone() for key, value in original.items()}
    corrupted["x"][~corrupted["point_mask"]] = float("nan")
    corrupted["policy"][~corrupted["policy_mask"]] = float("nan")
    corrupted["cycle_ids"][~corrupted["point_mask"].any(-1)] = float("nan")
    with torch.no_grad():
        expected = model(**original)
        actual = model(**corrupted)
        permutation = torch.tensor([2, 0, 1])
        permuted = model(**{key: value[permutation] for key, value in original.items()})
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(permuted, expected[permutation], rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("kind", ["native", "embedding"])
def test_frozen_backbone_keeps_bridge_gradient_path(bridges, kind):
    model = bridges[kind]
    model.zero_grad(set_to_none=True)
    inputs = example(2)
    inputs["x"].requires_grad_(True)
    state = model(**inputs)
    loss = (state*torch.linspace(-1, 1, 64)).mean()
    loss.backward()
    assert torch.isfinite(inputs["x"].grad).all()
    assert inputs["x"].grad.abs().sum() > 0
    assert model.fusion[0].weight.grad is not None
    assert torch.isfinite(model.fusion[0].weight.grad).all()
    assert not any(parameter.grad is not None for parameter in model.backbone.parameters())
    if kind == "embedding":
        assert model.cycle_projection[0].weight.grad.abs().sum() > 0


def test_native_path_matches_official_full_forward(bridges):
    bridge = bridges["native"]
    inputs = example(2)
    safe, valid, cycles = validate_prefix(**inputs, policy_dim=3)
    features, feature_mask = cycle_statistics(safe, inputs["point_mask"])
    packed, token_valid = bridge.token_inputs(features, feature_mask, cycles)
    captured = []
    hook = bridge.backbone.module.encoder.register_forward_hook(lambda module, args, output: captured.append(output))
    try:
        with torch.no_grad():
            bridge.backbone.module(**packed, prediction_mask=torch.zeros_like(packed["sample_id"], dtype=torch.bool))
    finally:
        hook.remove()
    with torch.no_grad():
        actual = bridge.backbone.encode_native(**packed)
    torch.testing.assert_close(actual, captured[0], rtol=0, atol=0)
    assert packed["target"].shape == (2, 156, 128)
    assert token_valid.shape == (2, 12, 13)


@pytest.mark.parametrize("kind", ["native", "embedding"])
def test_all_missing_prefix_is_finite_and_future_not_an_input(bridges, kind):
    inputs = example(1)
    inputs["point_mask"].zero_()
    inputs["x"].fill_(float("nan"))
    with torch.no_grad():
        assert torch.isfinite(bridges[kind](**inputs)).all()
    with pytest.raises(TypeError):
        bridges[kind](**inputs, query_future_target=torch.tensor([1000.]))


def test_random_control_is_explicit_and_different(bridges):
    torch.manual_seed(42)
    random_bridge = NativeBridge(CACHE, policy_dim=3, pretrained=False)
    random = random_bridge.backbone
    assert not random.pretrained
    assert random.original_parameter_count == bridges["native"].backbone.original_parameter_count
    assert not torch.equal(random.module.encoder.layers[0].self_attn.q_proj.weight,
                           bridges["native"].backbone.module.encoder.layers[0].self_attn.q_proj.weight)
    for key, value in random_bridge.fusion.state_dict().items():
        torch.testing.assert_close(value, bridges["native"].fusion.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["native", "embedding"])
def test_real_matr_prefix_batch_one_and_eight(bridges, kind):
    from n012.data import load_bundle
    bundle = load_bundle(ROOT, "MATR")
    x = bundle.x[:8].copy()
    mask = bundle.mask[:8]
    nominal = bundle.cell_metadata.loc[bundle.ids[:8], "nominal_capacity_Ah"].to_numpy(np.float32)
    x[..., 1] /= nominal[:, None, None]
    x[..., 2] /= nominal[:, None, None]
    for i in range(8):
        reference = np.nanmedian(np.where(mask[i, :10], x[i, :10, :, 2], np.nan), axis=0)
        x[i, ..., 3] = np.where(mask[i], x[i, ..., 2]-reference[None], 0.0)
    inputs = dict(x=torch.from_numpy(x), point_mask=torch.from_numpy(mask),
                  cycle_ids=torch.from_numpy(bundle.cycle[:8]),policy=torch.zeros(8, 3),
                  policy_mask=torch.zeros(8, 3, dtype=torch.bool), anchor=torch.from_numpy(bundle.anchor[:8]))
    with torch.no_grad():
        result = bridges[kind](**inputs)
        single = bridges[kind](**{key: value[:1] for key, value in inputs.items()})
    assert torch.isfinite(result).all()
    torch.testing.assert_close(single, result[:1], rtol=2e-5, atol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("kind", ["native", "embedding"])
@pytest.mark.parametrize("batch_size", [1, 8])
def test_cuda_cpu_agreement_and_gradient(bridges, kind, batch_size):
    cpu = bridges[kind]
    cuda = copy.deepcopy(cpu).cuda().eval()
    with torch.no_grad():
        reference = cpu(**example(batch_size))
        actual = cuda(**example(batch_size, "cuda")).cpu()
    torch.testing.assert_close(actual, reference, rtol=3e-4, atol=3e-4)
    cuda.zero_grad(set_to_none=True)
    inputs = example(batch_size, "cuda")
    loss = (cuda(**inputs)*torch.linspace(-1, 1, 64, device="cuda")).mean()
    loss.backward()
    gradient = cuda.fusion[0].weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum()>0


@pytest.mark.parametrize("kind", ["native", "embedding"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_q_projection_gradient_for_peft_and_p1(bridges, kind, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model = copy.deepcopy(bridges[kind]).to(device)
    projection = model.backbone.module.encoder.layers[0].self_attn.q_proj.weight
    original = projection.detach().clone()
    projection.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    output = model(**example(2, device))
    (output*torch.linspace(-1, 1, 64, device=device)).mean().backward()
    assert projection.grad is not None and torch.isfinite(projection.grad).all()
    assert projection.grad.abs().sum()>0
    torch.testing.assert_close(projection.detach(), original, rtol=0, atol=0)
