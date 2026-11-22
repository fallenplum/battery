"""Fixed input projection PEFT and paired, prefix-only Moirai moments."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from models.drs_peft.abba_adapters import LoRALinear
from models.drs_peft.onesided import replace_layer
from models.drs_peft.peft import _local_seed, random_orthogonal_bases

RANDOM_FIXED = "random_fixed"
TOTAL_ACTIVATION = "total_activation_covariance"
SELF_REFERENCE = "self_reference_generalized_eigen"
MOMENT_FIELDS = (
    "factual_first",
    "reference_first",
    "factual_second",
    "reference_second",
    "difference_second",
)


def _linear_targets(model):
    """Select exactly the original 12 Moirai-small Q/V sites."""
    expected = [
        f"encoder.layers.{index}.self_attn.{name}_proj"
        for index in range(6)
        for name in ("q", "v")
    ]
    targets = model.bridge.backbone.linear_targets("qv")
    if set(targets) != set(expected):
        raise ValueError("Expected twelve unwrapped Moirai-small Q/V linear targets")
    layers = {name: targets[name] for name in expected}
    if any(layer.weight.shape != (384, 384) for layer in layers.values()):
        raise ValueError(
            "Registered Moirai-small Q/V projections must all be 384 × 384"
        )
    return layers


class FixedInputLinear(nn.Module):
    """Apply the trainable update B Rᵀ with frozen orthonormal input basis R."""

    def __init__(self, base: nn.Linear, basis: torch.Tensor):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("Fixed input projection requires nn.Linear")
        coordinates = torch.as_tensor(basis).detach().to(base.weight).clone()
        if (
            coordinates.ndim != 2
            or coordinates.shape[0] != base.in_features
            or not 0 < coordinates.shape[1] <= base.in_features
            or not bool(torch.isfinite(coordinates).all())
        ):
            raise ValueError("Expected a finite input-width by rank basis")
        rank = coordinates.shape[1]
        gram = coordinates.double().T @ coordinates.double()
        identity = torch.eye(rank, dtype=gram.dtype, device=gram.device)
        if not torch.allclose(gram, identity, rtol=0, atol=1e-5):
            raise ValueError("Fixed input basis must be Euclidean orthonormal")
        self.base = base.requires_grad_(False)
        self.rank = rank
        self.register_buffer("basis", coordinates)
        self.B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))

    def delta_weight(self):
        return self.B @ self.basis.T

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.basis.T), self.B)


def install_projection_adapter(
    model, method, basis=None, seed=42, lora_rank=4, projection_rank=8
):
    """Install one budget-matched adapter family around the frozen model."""
    targets = _linear_targets(model)
    if not isinstance(lora_rank, int) or not isinstance(projection_rank, int):
        raise TypeError("Adapter ranks must be integers")
    if (
        not 0 < lora_rank <= 384
        or projection_rank != 2 * lora_rank
        or projection_rank > 384
    ):
        raise ValueError("For square Q/V sites, use projection_rank=2*lora_rank")
    if method not in ("lora", "fixed"):
        raise ValueError("Use standard LoRA or a frozen input projection")
    if method == "lora" and basis is not None:
        raise ValueError("Standard LoRA does not use a frozen input basis")
    if method == "fixed" and (basis is None or set(basis) != set(targets)):
        raise ValueError("Provide one fixed basis per registered projection")
    if method == "fixed" and any(
        torch.as_tensor(value).shape != (384, projection_rank)
        for value in basis.values()
    ):
        raise ValueError("A fixed basis does not match the registered rank")

    adapters = {}
    for name, layer in targets.items():
        adapters[name] = (
            LoRALinear(
                layer,
                lora_rank,
                alpha=lora_rank,
                dropout=0.0,
                seed=_local_seed(seed, name),
            )
            if method == "lora"
            else FixedInputLinear(layer, basis[name])
        )
    model.requires_grad_(False)
    # The original projections are frozen even when their adapters are trained.
    report = {}
    for name, adapter in adapters.items():
        for parameter_name, parameter in adapter.named_parameters():
            parameter.requires_grad_(not parameter_name.startswith("base."))
        replace_layer(model.backbone, name, adapter)
        count = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
        report[name] = {
            "trainable_parameters": count,
            "update_rank_upper_bound": lora_rank
            if method == "lora"
            else projection_rank,
            "fixed_basis_bytes": 0
            if method == "lora"
            else adapter.basis.numel() * adapter.basis.element_size(),
        }
    expected = len(targets) * 384 * projection_rank
    actual = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if actual != expected:
        raise RuntimeError(
            "The trainable parameter budget differs from the registered comparison"
        )
    return report


@dataclass
class LayerMoments:
    """Sums of equal-cell, valid-token first and second moments in CPU FP64."""

    factual_first: torch.Tensor
    reference_first: torch.Tensor
    factual_second: torch.Tensor
    reference_second: torch.Tensor
    difference_second: torch.Tensor


@dataclass
class ActivationMoments:
    """Streaming statistics over disjoint batches of physical cells."""

    cells: int
    valid_tokens: int
    minimum_valid_tokens: int
    maximum_valid_tokens: int
    layers: dict[str, LayerMoments]

    def merge(self, incoming):
        """Add a disjoint batch in place, retaining only fixed-size moments."""
        if self is incoming or set(self.layers) != set(incoming.layers):
            raise ValueError(
                "Cannot merge the same batch or different projection rosters"
            )
        for name, layer in self.layers.items():
            other = incoming.layers[name]
            if any(
                getattr(layer, field).shape != getattr(other, field).shape
                for field in MOMENT_FIELDS
            ):
                raise ValueError("Incompatible activation moment shapes")
        for name, layer in self.layers.items():
            other = incoming.layers[name]
            for field in MOMENT_FIELDS:
                getattr(layer, field).add_(getattr(other, field))
        self.cells += incoming.cells
        self.valid_tokens += incoming.valid_tokens
        self.minimum_valid_tokens = min(
            self.minimum_valid_tokens, incoming.minimum_valid_tokens
        )
        self.maximum_valid_tokens = max(
            self.maximum_valid_tokens, incoming.maximum_valid_tokens
        )
        return self


def _masked_inputs(value, valid):
    if value.ndim != 3 or value.shape[:2] != valid.shape:
        raise ValueError(
            "Projection input does not follow the registered native-token layout"
        )
    safe = (
        value.detach()
        .to(device="cpu", dtype=torch.float64)
        .masked_fill(~valid[..., None], 0)
    )
    if not bool(torch.isfinite(safe).all()):
        raise ValueError("A valid native token has a nonfinite projection input")
    return safe


def _paired_moments(factual, reference, weights):
    root_weight = weights.sqrt()[..., None]

    def second(values):
        rows = (values * root_weight).flatten(0, 1)
        return rows.T @ rows

    return LayerMoments(
        (factual * weights[..., None]).sum((0, 1)),
        (reference * weights[..., None]).sum((0, 1)),
        second(factual),
        second(reference),
        second(factual - reference),
    )


@torch.no_grad()
def collect_paired_activations(
    model, features, reference_features, feature_mask, cycles
):
    """Run the frozen factual/reference paths and accumulate real Q/V input moments.

    The model's responses method must execute the two native encoder passes in
    factual-first order. Padding is identified by the same token packer used
    for inference. Each physical cell contributes unit weight across all valid
    tokens, including cells with partial final patches.
    """
    if model.training:
        raise ValueError("Activation probes require the frozen model in eval mode")
    if (
        features.ndim != 3
        or features.shape != reference_features.shape
        or features.shape != feature_mask.shape
        or feature_mask.dtype != torch.bool
        or cycles.shape != features.shape[:2]
        or len(features) == 0
    ):
        raise ValueError(
            "Factual and reference inputs need one unchanged prefix and mask layout"
        )
    targets = _linear_targets(model)
    _, native_mask = model.bridge.token_inputs(features, feature_mask, cycles)
    valid = native_mask.flatten(1).cpu()
    counts = valid.sum(-1)
    if not bool(counts.gt(0).all()):
        raise ValueError("Each probe cell needs at least one valid native token")
    weights = valid.double() / counts[:, None]

    factual, moments = {}, {}

    def collect(name):
        def hook(module, args):
            if (
                name in moments
                or len(args) != 1
                or args[0].shape[-1] != module.in_features
            ):
                raise RuntimeError("Unexpected native Q/V call during paired probing")
            value = _masked_inputs(args[0], valid)
            if name not in factual:
                factual[name] = value
            else:
                moments[name] = _paired_moments(factual.pop(name), value, weights)

        return hook

    handles = [
        layer.register_forward_pre_hook(collect(name))
        for name, layer in targets.items()
    ]
    try:
        model.responses(
            features, feature_mask, cycles, reference_features=reference_features
        )
    finally:
        for handle in handles:
            handle.remove()
    if factual or set(moments) != set(targets):
        raise RuntimeError(
            "Paired response did not visit each native projection exactly twice"
        )
    return ActivationMoments(
        cells=len(features),
        valid_tokens=int(counts.sum()),
        minimum_valid_tokens=int(counts.min()),
        maximum_valid_tokens=int(counts.max()),
        layers=moments,
    )


def _symmetric(value):
    return (value + value.T) * 0.5


def _psd_eigenvalues(matrix, raw_energy, name):
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError(f"{name} has nonfinite moments")
    values, vectors = torch.linalg.eigh(matrix)
    # Use the scale of the raw second moment to account for cancellation when
    # subtracting the mean outer product from a nearly constant signal.
    scale = max(float(values.abs().max()), raw_energy, torch.finfo(matrix.dtype).tiny)
    tolerance = 64 * matrix.shape[0] * torch.finfo(matrix.dtype).eps * scale
    if float(values[0]) < -tolerance:
        raise ValueError(f"{name} is not positive semidefinite within FP64 precision")
    positive = int(values.gt(tolerance).sum())
    return (
        values,
        vectors,
        {
            "min_eigenvalue": float(values[0]),
            "max_eigenvalue": float(values[-1]),
            "psd_absolute_tolerance": tolerance,
            "positive_directions": positive,
            "trace": float(matrix.trace()),
        },
    )


def _check_moment_layer(layer, cells):
    dimension = layer.factual_first.numel()
    for field in MOMENT_FIELDS:
        value = getattr(layer, field)
        shape = (dimension,) if field.endswith("_first") else (dimension, dimension)
        if (
            value.shape != shape
            or value.dtype != torch.float64
            or value.device.type != "cpu"
        ):
            raise ValueError("Require CPU FP64 streaming sufficient statistics")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("A sufficient statistic is nonfinite")
    factual_mean, reference_mean = (
        layer.factual_first / cells,
        layer.reference_first / cells,
    )
    mf, mr, md = (
        layer.factual_second / cells,
        layer.reference_second / cells,
        layer.difference_second / cells,
    )
    factual_cov = _symmetric(mf - factual_mean.outer(factual_mean))
    reference_cov = _symmetric(mr - reference_mean.outer(reference_mean))
    return (
        factual_cov,
        reference_cov,
        _symmetric(md),
        (float(mf.trace()), float(mr.trace())),
    )


def _relative_residual(left, right):
    numerator = torch.linalg.matrix_norm(left - right)
    denominator = max(
        float(torch.linalg.matrix_norm(left)),
        float(torch.linalg.matrix_norm(right)),
        torch.finfo(left.dtype).tiny,
    )
    return float(numerator) / denominator


def _orthogonality_error(gram):
    identity = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    return float((gram - identity).abs().max())


def _energy_facts(basis, factual_cov, reference_cov, difference_cov):
    # Each trace is the energy of the Euclidean orthogonal projection into R.
    result = {}
    for name, covariance in (
        ("factual_centered", factual_cov),
        ("reference_centered", reference_cov),
        ("paired_difference_uncentered", difference_cov),
    ):
        trace = float(covariance.trace())
        energy = float((basis.T @ covariance @ basis).trace())
        result[f"{name}_projected_energy"] = energy
        result[f"{name}_projected_fraction"] = energy / trace if trace > 0 else None
    result["euclidean_orthogonality_max_abs"] = _orthogonality_error(basis.T @ basis)
    return result


def _fit_layer(layer, cells, k, ridge, seed):
    factual_cov, reference_cov, difference_cov, raw_energy = _check_moment_layer(
        layer, cells
    )
    dimension = factual_cov.shape[0]
    if not 1 <= k <= dimension:
        raise ValueError("Projection rank exceeds the recorded activation dimension")
    factual_values, factual_vectors, factual_audit = _psd_eigenvalues(
        factual_cov, raw_energy[0], "Total factual activation covariance"
    )
    _, _, reference_audit = _psd_eigenvalues(
        reference_cov, raw_energy[1], "Self-reference activation covariance"
    )
    _, _, difference_audit = _psd_eigenvalues(
        difference_cov, float(difference_cov.trace()), "Paired activation second moment"
    )
    if (
        min(
            factual_audit["positive_directions"],
            difference_audit["positive_directions"],
        )
        < k
    ):
        raise ValueError(
            "Activation moments have fewer resolved positive directions than the fixed rank"
        )
    reference_scale = reference_audit["trace"] / dimension
    if reference_scale <= 0:
        raise ValueError(
            "Trace-scaled ridge cannot regularize a zero reference covariance"
        )
    denominator = reference_cov + (ridge * reference_scale) * torch.eye(
        dimension, dtype=torch.float64
    )
    lower, information = torch.linalg.cholesky_ex(denominator)
    if int(information) != 0:
        raise ValueError(
            "Regularized self-reference covariance is not positive definite"
        )
    triangular = torch.linalg.solve_triangular(lower, difference_cov, upper=False)
    whitened = _symmetric(
        torch.linalg.solve_triangular(lower, triangular.T, upper=False).T
    )
    generalized_values, whitened_vectors, generalized_audit = _psd_eigenvalues(
        whitened, float(whitened.trace()), "Whitened paired activation moment"
    )
    if generalized_audit["positive_directions"] < k:
        raise ValueError(
            "Generalized eigenproblem has fewer resolved positive directions than the fixed rank"
        )
    indices = torch.arange(dimension - 1, dimension - k - 1, -1)
    unwhitened = torch.linalg.solve_triangular(
        lower.T, whitened_vectors[:, indices], upper=True
    )
    selected_values = generalized_values[indices]
    generalized_residual = _relative_residual(
        difference_cov @ unwhitened, (denominator @ unwhitened) * selected_values
    )
    metric_orthogonality = _orthogonality_error(unwhitened.T @ denominator @ unwhitened)
    self_reference, triangular_qr = torch.linalg.qr(unwhitened, mode="reduced")
    signs = torch.where(triangular_qr.diag() < 0, -1, 1)
    self_reference = self_reference * signs
    total = factual_vectors[:, indices]
    total_residual = _relative_residual(
        factual_cov @ total, total * factual_values[indices]
    )
    _, random = random_orthogonal_bases(k, dimension, k, seed=seed, dtype=torch.float64)
    bases = {
        RANDOM_FIXED: random,
        TOTAL_ACTIVATION: total,
        SELF_REFERENCE: self_reference,
    }
    energies = {
        method: _energy_facts(basis, factual_cov, reference_cov, difference_cov)
        for method, basis in bases.items()
    }
    errors = [
        generalized_residual,
        total_residual,
        metric_orthogonality,
        *(item["euclidean_orthogonality_max_abs"] for item in energies.values()),
    ]
    if any(not math.isfinite(value) or value > 1e-7 for value in errors):
        raise ValueError("The projection eigensystem failed its FP64 numerical audit")
    factual_rms = math.sqrt(max(raw_energy[0], 0) / dimension)
    paired_rms = math.sqrt(max(difference_audit["trace"], 0) / dimension)
    audit = {
        "factual_centered_covariance": factual_audit,
        "reference_centered_covariance": reference_audit,
        "paired_difference_uncentered_moment": difference_audit,
        "generalized_spectrum": generalized_audit,
        "regularized_reference_min_eigenvalue": reference_audit["min_eigenvalue"]
        + ridge * reference_scale,
        "ridge_trace_scale": reference_scale,
        "ridge_absolute": ridge * reference_scale,
        "selected_total_eigenvalues": factual_values[indices].tolist(),
        "selected_generalized_eigenvalues": selected_values.tolist(),
        "generalized_equation_relative_residual": generalized_residual,
        "total_covariance_equation_relative_residual": total_residual,
        "pre_qr_reference_metric_orthogonality_max_abs": metric_orthogonality,
        "factual_token_rms": factual_rms,
        "paired_change_token_rms": paired_rms,
        "paired_change_relative_rms": paired_rms / factual_rms
        if factual_rms > 0
        else None,
        "projection_energies": energies,
    }
    return bases, audit


def fit_projection_bases(moments, k=8, ridge=0.1, seed=42):
    """Construct the three fixed input spaces from label-free valid-token moments."""
    if not isinstance(k, int) or k < 1 or not math.isfinite(ridge) or ridge <= 0:
        raise ValueError(
            "A positive integer rank and positive finite ridge are required"
        )
    if moments.cells < 1 or not moments.layers or moments.minimum_valid_tokens < 1:
        raise ValueError("Cannot fit from empty activation moments")
    result = {method: {} for method in (RANDOM_FIXED, TOTAL_ACTIVATION, SELF_REFERENCE)}
    audit = {
        "cells": moments.cells,
        "valid_tokens": moments.valid_tokens,
        "minimum_valid_tokens_per_cell": moments.minimum_valid_tokens,
        "maximum_valid_tokens_per_cell": moments.maximum_valid_tokens,
        "moment_weighting": "one unit per physical cell, divided over its valid native tokens",
        "center_factual_and_reference_at_weighted_corpus_mean": True,
        "center_paired_difference": False,
        "rank": k,
        "ridge_multiplier": ridge,
        "basis_seed": seed,
        "target_values_used": 0,
        "layers": {},
    }
    for name, moments_layer in moments.layers.items():
        try:
            bases, facts = _fit_layer(
                moments_layer, moments.cells, k, ridge, _local_seed(seed, name)
            )
        except ValueError as exc:
            raise ValueError(f"{name}: {exc}") from exc
        for method, matrix in bases.items():
            result[method][name] = matrix
        audit["layers"][name] = facts
    return result, audit
