"""Spectrum-preserving per-cell rotations and differentiable Support adaptation."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from models.drs_peft.onesided import replace_layer, source_layers
from models.drs_peft.peft import svd_bases
from scripts.drs_peft.abba_engine import merged_reference, reference_modules

RANK = 4
ROTATION_MODES = ("constant", "conditioned", "shuffled", "fixed_half")
CONDITIONED_MODES = ("constant", "conditioned", "shuffled", "drs", "gain")
MODES = ("static", "old_static", "joint", "drs", "gain", *ROTATION_MODES)
_ROTATION_CONTEXT = ContextVar("spectral_rotation_context", default=None)


def _gram_error(basis):
    values = basis.detach().cpu().double()
    identity = torch.eye(values.shape[1], dtype=torch.float64)
    return float((values.T @ values - identity).abs().max())


@dataclass(frozen=True)
class RotationBasis:
    """One frozen left-plane frame and one frozen right frame."""

    u0: torch.Tensor
    u_perp: torch.Tensor
    v: torch.Tensor
    singular_values: torch.Tensor
    scale: float

    def audit(self):
        left = torch.cat((self.u0, self.u_perp), dim=1)
        result = {
            "left_joint_gram_max_abs": _gram_error(left),
            "right_gram_max_abs": _gram_error(self.v),
            "left_cross_gram_max_abs": float(
                (
                    self.u0.detach().cpu().double().T
                    @ self.u_perp.detach().cpu().double()
                )
                .abs()
                .max()
            ),
        }
        if max(result.values()) > 2e-5 or not all(
            math.isfinite(v) for v in result.values()
        ):
            raise ValueError(
                "Each left rotation plane and right frame must be orthonormal"
            )
        return result


def _fit_layer_basis(layer):
    if layer.rank != RANK:
        raise ValueError("The frozen gradient basis must use registered rank four")
    u0 = layer.U.detach().clone()
    v = layer.V.detach().clone()
    if u0.shape[0] < 2 * RANK:
        raise ValueError("The left complement needs four independent directions")
    effective = layer.base.weight + layer.scale * (u0 @ layer.effective_core @ v.T)
    weight = effective.detach().cpu().double()
    left = torch.linalg.qr(u0.detach().cpu().double(), mode="reduced").Q
    normal, _, singular_values = svd_bases(weight - left @ (left.T @ weight), RANK)
    if singular_values[-1] <= 1e-12 * torch.linalg.vector_norm(weight):
        raise ValueError(
            "The projected source weight lacks four identifiable normal directions"
        )
    pivot = normal.abs().argmax(dim=0, keepdim=True)
    normal = normal * normal.gather(0, pivot).sign()
    basis = RotationBasis(
        u0,
        normal.to(u0),
        v,
        singular_values,
        float(layer.scale),
    )
    basis.audit()
    return basis


@torch.no_grad()
def build_rotation_bases(reference):
    """Keep source-fit gradient frames and complete U0 from projected frozen weights."""
    old = source_layers(reference)
    if len(old) != 12 or any(
        not name.endswith(("q_proj", "v_proj")) for name, _ in old
    ):
        raise ValueError("The reference must have 12 source Q/V subspace layers")
    return {name: _fit_layer_basis(layer) for name, layer in old}


def _angle_shape(values, angle):
    if values.ndim < 2 or angle.shape != (len(values), values.shape[-1]):
        raise ValueError(
            "Angles must match the batch of physical cells and output rank"
        )
    return angle.reshape(
        (len(values),) + (1,) * (values.ndim - 2) + (values.shape[-1],)
    )


class SpectralRotationLinear(nn.Module):
    """Evaluate small rank factors without per-cell dense weight matrices."""

    def __init__(self, base, basis, index, mode):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.index = index
        self.mode = mode
        self.scale = float(basis.scale)
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("Source adapter scale must be finite and positive")
        self.basis_audit = basis.audit()
        expected_left = (base.out_features, RANK)
        if (
            basis.u0.shape != expected_left
            or basis.u_perp.shape != expected_left
            or basis.v.shape != (base.in_features, RANK)
        ):
            raise ValueError("Rotation basis dimensions differ from their linear layer")
        left, right = basis.u0, basis.v
        if mode == "joint":
            left = torch.cat((basis.u0, basis.u_perp), dim=1)
            right = basis.v[:, :2]
        self.register_buffer("left", left.detach().to(base.weight).clone())
        self.register_buffer("right", right.detach().to(base.weight).clone())
        self.register_buffer(
            "normal",
            basis.u_perp.detach().to(base.weight).clone()
            if mode in ROTATION_MODES
            else None,
        )

    @property
    def core_shape(self):
        return self.left.shape[1], self.right.shape[1]

    def delta_weight(self, core, modulation=None):
        """Materialize only on demand for spectral audits, in weight convention."""
        if core.shape != self.core_shape:
            raise ValueError("Core has the wrong shape for this adapter")
        left = self.left
        if self.mode in ROTATION_MODES:
            left = left * modulation.cos().unsqueeze(
                -2
            ) + self.normal * modulation.sin().unsqueeze(-2)
        return self.scale * (left @ self._gated_core(core, modulation) @ self.right.T)

    def _gated_core(self, core, modulation):
        if self.mode == "drs":
            output_gate, input_gate = modulation.split(RANK, dim=-1)
            return output_gate.unsqueeze(-1) * core * input_gate.unsqueeze(-2)
        if self.mode == "gain":
            return modulation.unsqueeze(-1) * core
        return core

    def frobenius(self, core, modulation=None):
        """Small-Gram norm including the float32 basis orthogonality residual."""
        left_gram = self.left.T @ self.left
        if self.mode in ROTATION_MODES:
            cosine = modulation.cos()
            sine = modulation.sin()
            cross = self.left.T @ self.normal
            normal_gram = self.normal.T @ self.normal
            left_gram = (
                cosine.unsqueeze(-1) * left_gram * cosine.unsqueeze(-2)
                + cosine.unsqueeze(-1) * cross * sine.unsqueeze(-2)
                + sine.unsqueeze(-1) * cross.T * cosine.unsqueeze(-2)
                + sine.unsqueeze(-1) * normal_gram * sine.unsqueeze(-2)
            )
        effective = self._gated_core(core, modulation)
        right_gram = self.right.T @ self.right
        squared = (effective * ((left_gram @ effective) @ right_gram)).sum(dim=(-2, -1))
        return squared.clamp_min(0).sqrt() * self.scale

    def forward(self, inputs):
        result = self.base(inputs)
        context = _ROTATION_CONTEXT.get()
        if context is None or self not in context:
            return result
        core, modulation = context[self]
        rank_inputs = inputs @ self.right
        if self.mode == "drs":
            output_gate, input_gate = modulation.split(RANK, dim=-1)
            rank_inputs = rank_inputs * _angle_shape(rank_inputs, input_gate)
            modulation = output_gate
        rank_values = self.scale * F.linear(rank_inputs, core)
        if self.mode in ROTATION_MODES:
            theta = _angle_shape(rank_values, modulation)
            return (
                result
                + (rank_values * theta.cos()) @ self.left.T
                + (rank_values * theta.sin()) @ self.normal.T
            )
        if self.mode in ("drs", "gain"):
            rank_values = rank_values * _angle_shape(rank_values, modulation)
        return result + rank_values @ self.left.T


def masked_cell_mse(prediction, target, observed):
    """Average first within each Support cell and then across physical cells."""
    if (
        prediction.ndim != 2
        or prediction.shape != target.shape
        or target.shape != observed.shape
    ):
        raise ValueError("Trajectory outcomes must be [physical_cell,time] tensors")
    if (
        observed.dtype != torch.bool
        or not bool(observed.any(dim=1).all())
        or not len(prediction)
    ):
        raise ValueError("Each Support cell needs at least one observed outcome")
    if not bool(
        torch.isfinite(prediction).all() and torch.isfinite(target[observed]).all()
    ):
        raise FloatingPointError("Nonfinite predictions or observed Support labels")
    residual = prediction - torch.where(observed, target, prediction)
    return (residual.square().sum(1) / observed.sum(1)).mean()


class SpectralRotationModel(nn.Module):
    """One adapter core per Q/V layer, shared across cells within an episode."""

    def __init__(
        self, reference, bases, mode, descriptor_dim, *, angle_max=math.pi / 4
    ):
        super().__init__()
        if (
            mode not in MODES
            or not isinstance(descriptor_dim, int)
            or descriptor_dim <= 0
        ):
            raise ValueError(
                "Register a rotation control and a positive descriptor dimension"
            )
        if not math.isfinite(angle_max) or not 0 < angle_max <= math.pi / 2:
            raise ValueError("Maximum rotation angle must be in (0, pi/2]")
        self.mode = mode
        self.descriptor_dim = descriptor_dim
        self.angle_max = float(angle_max)
        self.reference = (
            copy.deepcopy(reference).eval().requires_grad_(False)
            if hasattr(reference, "abba_module_names")
            else merged_reference(reference)
        )
        source = reference_modules(self.reference)
        self.layer_names = tuple(name for name, _ in source)
        if set(bases) != set(self.layer_names):
            raise ValueError(
                "The registered frames must cover exactly the reference Q/V layers"
            )
        self.layers = {}
        for index, (name, base) in enumerate(source):
            layer = SpectralRotationLinear(base, bases[name], index, mode)
            replace_layer(self.reference.backbone, name, layer)
            self.layers[name] = layer
        weight = source[0][1].weight
        self.modulation_dim = 2 * RANK if mode == "drs" else RANK
        if mode in CONDITIONED_MODES:
            with torch.random.fork_rng(devices=[]):
                self.controller = nn.Linear(
                    descriptor_dim,
                    len(source) * self.modulation_dim,
                    dtype=weight.dtype,
                )
            self.controller.to(weight.device)
            nn.init.zeros_(self.controller.weight)
            nn.init.zeros_(self.controller.bias)
        else:
            self.controller = None

    @property
    def backbone(self):
        return self.reference.backbone

    @property
    def input_keys(self):
        return self.reference.input_keys

    def train(self, mode=True):
        super().train(mode)
        self.reference.eval()
        return self

    def initial_cores(self, *, requires_grad=True):
        return {
            name: layer.left.new_zeros(layer.core_shape, requires_grad=requires_grad)
            for name, layer in self.layers.items()
        }

    def budget(self):
        return {
            "episode_parameters": sum(
                math.prod(layer.core_shape) for layer in self.layers.values()
            ),
            "controller_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
            "constant_identifiable_parameters": len(self.layers) * RANK
            if self.mode == "constant"
            else None,
            "adapted_modules": len(self.layers),
            "maximum_matrix_rank": min(next(iter(self.layers.values())).core_shape),
            "source_scales": sorted({layer.scale for layer in self.layers.values()}),
        }

    @staticmethod
    def core_norms(cores):
        return {name: torch.linalg.vector_norm(core) for name, core in cores.items()}

    @staticmethod
    def scale_cores(cores, scalar):
        return {name: core * scalar for name, core in cores.items()}

    @torch.no_grad()
    def update_frobenius(
        self, cores, descriptor=None, *, batch_size=None, per_layer=False
    ):
        if batch_size is None:
            if descriptor is None:
                raise ValueError(
                    "Provide the physical batch size if no descriptors are passed"
                )
            batch_size = len(descriptor)
        modulation = self.modulation(batch_size, descriptor)
        norms = []
        for name, layer in self.layers.items():
            condition = None if modulation is None else modulation[:, layer.index]
            norms.append(layer.frobenius(cores[name], condition).expand(batch_size))
        values = torch.stack(norms, dim=-1)
        return values if per_layer else torch.linalg.vector_norm(values, dim=-1)

    def modulation(self, batch_size, descriptor=None):
        layer = next(iter(self.layers.values()))
        if self.mode == "fixed_half":
            return layer.left.new_full(
                (batch_size, len(self.layers), RANK), math.pi / 4
            )
        if self.controller is None:
            return None
        if self.mode == "constant":
            # Express each constant angle through one bias. Repeating a unit
            # input would multiply its effective Adam step by descriptor_dim + 1.
            descriptor = layer.left.new_zeros((batch_size, self.descriptor_dim))
        if descriptor is None or descriptor.shape != (batch_size, self.descriptor_dim):
            raise ValueError(
                "Each conditioned cell needs one fixed-dimension early descriptor"
            )
        if not bool(torch.isfinite(descriptor).all()):
            raise FloatingPointError("Conditioning features must be finite")
        raw = self.controller(descriptor.to(layer.left)).reshape(
            batch_size, len(self.layers), self.modulation_dim
        )
        if self.mode in ("drs", "gain"):
            return 1 + 0.5 * raw.tanh()
        return self.angle_max * raw.tanh()

    def forward(self, early, query, descriptor=None, cores=None):
        if cores is None:
            return self.reference(early, query)
        if not isinstance(cores, Mapping) or set(cores) != set(self.layer_names):
            raise ValueError("An episode needs exactly one core for each Q/V layer")
        modulation = self.modulation(len(early["anchor"]), descriptor)
        context = {
            layer: (
                cores[name],
                None if modulation is None else modulation[:, layer.index],
            )
            for name, layer in self.layers.items()
        }
        if any(item[0].shape != layer.core_shape for layer, item in context.items()):
            raise ValueError("An episode core has the wrong dimensions")
        token = _ROTATION_CONTEXT.set(context)
        try:
            return self.reference(early, query)
        finally:
            _ROTATION_CONTEXT.reset(token)

    def adapt_episode(
        self,
        support_early,
        support_query,
        target,
        observed,
        support_descriptor=None,
        *,
        steps,
        lr,
        clip_norm=1.0,
        differentiable=True,
        cores=None,
    ):
        """Return an ephemeral SGD-adapted core mapping from actual Support loss."""
        if not isinstance(steps, int) or steps < 0 or not math.isfinite(lr) or lr <= 0:
            raise ValueError(
                "Use a nonnegative number of inner steps and positive learning rate"
            )
        if clip_norm is not None and (not math.isfinite(clip_norm) or clip_norm <= 0):
            raise ValueError("Gradient clipping must use a positive finite norm")
        values = self.initial_cores() if cores is None else cores
        with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
            for _ in range(steps):
                prediction = self(
                    support_early, support_query, support_descriptor, values
                )
                loss = masked_cell_mse(prediction, target, observed)
                gradients = torch.autograd.grad(
                    loss, tuple(values.values()), create_graph=differentiable
                )
                if clip_norm is not None:
                    squared_norm = torch.stack(
                        [grad.square().sum() for grad in gradients]
                    ).sum()
                    norm = (squared_norm + 1e-24).sqrt()
                    multiplier = (clip_norm / (norm + 1e-6)).clamp_max(1.0)
                    gradients = tuple(grad * multiplier for grad in gradients)
                values = {
                    name: current - lr * grad
                    for (name, current), grad in zip(
                        values.items(), gradients, strict=True
                    )
                }
                if not differentiable:
                    values = {
                        name: core.detach().requires_grad_()
                        for name, core in values.items()
                    }
        return values
