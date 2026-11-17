"""Source-preserving LoRA-XS with a zero-initialized singular-value-scaled core."""

from __future__ import annotations

import copy
import hashlib
import json

import torch
from torch import nn
from torch.nn import functional as F

from .onesided import NoPolicyBatteryTSFM, replace_layer, source_layers


def xs_payload_fingerprint(payload):
    digest = hashlib.sha256()
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in ("layers", "payload_sha256")
    }
    digest.update(json.dumps(metadata, sort_keys=True, allow_nan=False).encode())
    for name, factors in sorted(payload["layers"].items()):
        digest.update(name.encode())
        for key, tensor in sorted(factors.items()):
            value = tensor.detach().cpu().contiguous()
            digest.update(key.encode())
            digest.update(str(value.dtype).encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class XSLinear(nn.Module):
    """Keep f0 and append (alpha/r) U delta_R Sigma V.T without dropping Sigma."""

    def __init__(self, source, factors, rank=4, alpha=4):
        super().__init__()
        self.source = source
        self.source.requires_grad_(False)
        self.rank, self.alpha = int(rank), float(alpha)
        if self.rank != 4 or self.alpha != 4:
            raise ValueError("The specialization contrast fixes LoRA-XS rank=alpha=4")
        m, n = source.base.weight.shape
        shapes = {"U": (m, self.rank), "singular": (self.rank,), "V": (n, self.rank)}
        for key, shape in shapes.items():
            value = factors[key]
            if tuple(value.shape) != shape or not bool(torch.isfinite(value).all()):
                raise ValueError("A LoRA-XS SVD factor has an invalid shape or value")
            self.register_buffer(
                "core_xs_" + key, value.detach().to(source.core).clone()
            )
        if bool((self.core_xs_singular < 0).any()):
            raise ValueError("Singular values must be nonnegative")
        self.register_buffer("core_adaptation_scale", source.core.new_tensor(1.0))
        self.core_xs_delta = nn.Parameter(source.core.new_zeros(self.rank, self.rank))

    def increment_matrix(self):
        core = self.core_xs_delta * self.core_xs_singular[None]
        return (
            self.core_adaptation_scale
            * (self.alpha / self.rank)
            * (self.core_xs_U @ core @ self.core_xs_V.T)
        )

    def forward(self, x):
        projected = (x @ self.core_xs_V) * self.core_xs_singular
        change = F.linear(projected, self.core_xs_delta) @ self.core_xs_U.T
        return (
            self.source(x)
            + self.core_adaptation_scale * (self.alpha / self.rank) * change
        )


class XSCalibrationModel(NoPolicyBatteryTSFM):
    """Freeze the common source, training only 12 rank-four delta_R matrices."""

    method = "LoRA-XS"

    def __init__(self, source, payload):
        if any(isinstance(layer, XSLinear) for layer in source.backbone.modules()):
            raise ValueError("The source already contains a LoRA-XS update")
        if payload.get("payload_sha256") != xs_payload_fingerprint(payload):
            raise RuntimeError("The frozen LoRA-XS factors changed")
        super().__init__(source)
        self.requires_grad_(False)
        layers = source_layers(self)
        if set(payload["layers"]) != {name for name, _ in layers}:
            raise ValueError("The LoRA-XS factors and source target matrices differ")
        for name, layer in layers:
            replace_layer(
                self.backbone,
                name,
                XSLinear(
                    layer, payload["layers"][name], payload["rank"], payload["alpha"]
                ),
            )
        self.xs_payload_sha256 = payload["payload_sha256"]

    @torch.no_grad()
    def set_adaptation_scale(self, scale):
        if not 0 <= float(scale) <= 1:
            raise ValueError("Post-fit shrinkage must lie within [0,1]")
        for layer in self.backbone.modules():
            if isinstance(layer, XSLinear):
                layer.core_adaptation_scale.fill_(float(scale))
        return self

    def parameter_report(self):
        report = super().parameter_report()
        layers = [
            layer for layer in self.backbone.modules() if isinstance(layer, XSLinear)
        ]
        report.update(
            adaptation_method="LoRA-XS zero-increment variant",
            adaptation_parameters=sum(layer.core_xs_delta.numel() for layer in layers),
            rank=4,
            alpha=4,
            scale=1.0,
            dropout=0.0,
            singular_values_retained=True,
            SVD_source="official_W_plus_frozen_source_core_increment",
            SVD_precision="CPU_float64",
            frozen_xs_factor_bytes=sum(
                value.numel() * value.element_size()
                for name, value in self.named_buffers()
                if ".core_xs_" in name
            ),
            xs_payload_sha256=self.xs_payload_sha256,
        )
        return report

    @torch.no_grad()
    def merged_copy(self):
        result = copy.deepcopy(self)
        for name, layer in list(result.backbone.named_modules()):
            if isinstance(layer, XSLinear):
                layer.source.base.weight.add_(layer.increment_matrix())
                replace_layer(result.backbone, name, layer.source)
        merged = NoPolicyBatteryTSFM(result)
        merged.requires_grad_(False)
        merged.official_backbone_identity = dict(
            merged.official_backbone_identity, merged_adaptation=self.method
        )
        return merged
