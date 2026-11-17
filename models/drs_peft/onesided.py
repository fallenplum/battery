"""No-policy reference functions and appended few-shot update branches."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from contextvars import ContextVar

import torch
from torch import nn
from torch.nn import functional as F

from .peft import SubspaceLinear
from .trajectory import BatteryTSFM

INPUT_KEYS = ("x", "point_mask", "cycle_ids", "policy", "policy_mask", "anchor")
GEOMETRIC_ARMS = ("CORE", "OUT", "IN", "NORMAL")
_COORDINATES = ContextVar("onesided_coordinates", default=None)


def no_policy_inputs(early, policy_dim=0):
    """Keep the established shape while erasing both metadata channels."""
    result = {
        name: early[name]
        for name in INPUT_KEYS
        if name not in ("policy", "policy_mask")
    }
    shape = (len(early["x"]), int(policy_dim))
    result["policy"] = early["x"].new_zeros(shape)
    result["policy_mask"] = torch.zeros(
        shape, device=early["x"].device, dtype=torch.bool
    )
    return result


@contextmanager
def coordinate_context(coordinates):
    token = _COORDINATES.set(coordinates)
    try:
        yield
    finally:
        _COORDINATES.reset(token)


class NoPolicyBatteryTSFM(BatteryTSFM):
    """Adopt a model before source training and enforce its input contract."""

    input_keys = INPUT_KEYS
    input_scope = "VIQ+earlySOH / no-policy"

    def __init__(
        self, model, policy_dim=0, bridge="native_multivariate", pretrained=True
    ):
        if isinstance(model, nn.Module):
            nn.Module.__init__(self)
            self.bridge = model.bridge
            self.head = model.head
            self.bridge_kind = model.bridge_kind
            self.official_backbone_identity = model.official_backbone_identity
        else:
            if policy_dim != 0 or bridge != "native_multivariate":
                raise ValueError(
                    "The source contract requires native_multivariate with policy_dim=0"
                )
            super().__init__(model, 0, bridge, pretrained=pretrained)
            self.official_backbone_identity = self.bridge.backbone.audit()

    def encode(self, early):
        return self.bridge.encode(**no_policy_inputs(early, self.bridge.policy_dim))

    def parameter_report(self):
        return dict(
            super().parameter_report(),
            input_scope=self.input_scope,
            policy_values_forced_zero=True,
            policy_masks_forced_zero=True,
            early_SOH_anchor_retained=True,
            native_interface="twelve_statistics_per_cycle",
            policy_dimension=self.bridge.policy_dim,
        )


class DictionaryLinear(nn.Module):
    """Apply a frozen matrix dictionary using one model-wide coordinate vector."""

    def __init__(self, source, specification):
        super().__init__()
        if not isinstance(source, SubspaceLinear) or bool(source.target_active):
            raise ValueError("Dictionary updates require a static source core")
        self.source = source
        self.source.requires_grad_(False)
        self.kind = specification["kind"]
        if self.kind not in GEOMETRIC_ARMS:
            raise ValueError("Unknown geometric dictionary block")
        directions = specification["directions"].detach().to(source.core).clone()
        if directions.ndim != 3 or not bool(torch.isfinite(directions).all()):
            raise ValueError("Dictionary directions must be finite [q,rows,cols]")
        self.dimension = directions.shape[0]
        if self.dimension < 1:
            raise ValueError("The actual dictionary rank must be positive")
        self.register_buffer("core_dictionary_directions", directions)
        for name in ("left", "right"):
            value = specification.get(name)
            self.register_buffer(
                "core_dictionary_" + name,
                None if value is None else value.detach().to(source.core).clone(),
            )
        m, n = source.base.weight.shape
        r = source.rank
        shapes = {"CORE": (r, r), "OUT": (m, r), "IN": (r, n), "NORMAL": (m, n)}
        if tuple(directions.shape[1:]) != shapes[self.kind]:
            raise ValueError("Dictionary factors differ from the source weight")
        for side, required, expected in (
            ("left", self.kind in ("CORE", "IN"), (m, r)),
            ("right", self.kind in ("CORE", "OUT"), (n, r)),
        ):
            value = getattr(self, "core_dictionary_" + side)
            if required and (value is None or tuple(value.shape) != expected):
                raise ValueError("A required dictionary projection factor is missing")

    def coefficients(self, coordinates=None):
        coordinates = _COORDINATES.get() if coordinates is None else coordinates
        if coordinates is None or coordinates.shape != (self.dimension,):
            raise ValueError("The shared adaptation vector has the wrong dimension")
        return torch.einsum(
            "q,qij->ij",
            coordinates.to(self.core_dictionary_directions),
            self.core_dictionary_directions,
        )

    def increment_matrix(self, coordinates):
        result = self.coefficients(coordinates)
        if self.kind in ("CORE", "IN"):
            result = self.core_dictionary_left @ result
        if self.kind in ("CORE", "OUT"):
            result = result @ self.core_dictionary_right.T
        return result

    def forward(self, x):
        coefficients = self.coefficients()
        if self.kind == "CORE":
            change = (
                F.linear(x @ self.core_dictionary_right, coefficients)
                @ self.core_dictionary_left.T
            )
        elif self.kind == "OUT":
            change = F.linear(x @ self.core_dictionary_right, coefficients)
        elif self.kind == "IN":
            change = F.linear(x, coefficients) @ self.core_dictionary_left.T
        else:
            change = F.linear(x, coefficients)
        return self.source(x) + change


def source_layers(model):
    if any(
        isinstance(layer, (DictionaryLinear, AppendedLowRankLinear))
        for layer in model.backbone.modules()
    ):
        raise ValueError(
            "Reconstruct the common static source before adding another adaptation"
        )
    layers = [
        (name, layer)
        for name, layer in model.backbone.named_modules()
        if isinstance(layer, SubspaceLinear)
    ]
    if not layers or any(bool(layer.target_active) for _, layer in layers):
        raise ValueError("A fitted static source model is required")
    return layers


def replace_layer(backbone, name, layer):
    parent_name, _, attribute = name.rpartition(".")
    parent = backbone.get_submodule(parent_name) if parent_name else backbone
    setattr(parent, attribute, layer)


class DictionaryBatteryTSFM(NoPolicyBatteryTSFM):
    """Freeze f0 and train only the globally shared dictionary coordinates."""

    def __init__(self, model, payload):
        super().__init__(model)
        self.requires_grad_(False)
        layers = source_layers(self)
        specifications = payload["layers"]
        if set(specifications) != {name for name, _ in layers}:
            raise ValueError("Dictionary and source target matrices must agree")
        dimensions = {value["directions"].shape[0] for value in specifications.values()}
        kinds = {value["kind"] for value in specifications.values()}
        if len(dimensions) != 1 or len(kinds) != 1:
            raise ValueError("All layers share one block and one coordinate vector")
        self.dimension = dimensions.pop()
        self.arm = kinds.pop()
        if payload.get("dimension", self.dimension) != self.dimension:
            raise ValueError("The dictionary metadata overstates its effective rank")
        if hasattr(self.backbone, "core_coordinates"):
            raise ValueError("The source already contains an adaptation branch")
        for name, layer in layers:
            replace_layer(
                self.backbone, name, DictionaryLinear(layer, specifications[name])
            )
        self.backbone.register_parameter(
            "core_coordinates",
            nn.Parameter(layers[0][1].core.new_zeros(self.dimension)),
        )

    @property
    def coordinates(self):
        return self.backbone.core_coordinates

    def encode_with_coordinates(self, early, coordinates):
        with coordinate_context(coordinates):
            return super().encode(early)

    def encode(self, early):
        return self.encode_with_coordinates(early, self.coordinates)

    def forward(self, early, query, coordinates=None):
        coordinates = self.coordinates if coordinates is None else coordinates
        state = self.encode_with_coordinates(early, coordinates)
        return self.head(state, early["anchor"], query)

    def parameter_report(self):
        report = super().parameter_report()
        dictionary_bytes = sum(
            value.numel() * value.element_size()
            for name, value in self.named_buffers()
            if ".core_dictionary_" in name
        )
        report.update(
            adaptation_method=self.arm,
            adaptation_parameters=self.dimension,
            frozen_dictionary_bytes=dictionary_bytes,
        )
        return report

    @torch.no_grad()
    def merged_copy(self):
        result = copy.deepcopy(self)
        for name, layer in list(result.backbone.named_modules()):
            if isinstance(layer, DictionaryLinear):
                layer.source.base.weight.add_(
                    layer.increment_matrix(result.coordinates)
                )
                replace_layer(result.backbone, name, layer.source)
        del result.backbone.core_coordinates
        merged = NoPolicyBatteryTSFM(result)
        merged.requires_grad_(False)
        merged.official_backbone_identity = dict(
            merged.official_backbone_identity, merged_adaptation=self.arm
        )
        return merged


class AppendedLowRankLinear(nn.Module):
    """Keep the entire source function and append ordinary BA or fixed-A BA."""

    def __init__(self, source, method, seed):
        super().__init__()
        method = "LoRA-FA" if method == "LoRA_FA" else method
        if method not in ("LoRA", "LoRA-FA"):
            raise ValueError("Unknown appended low-rank baseline")
        self.source = source
        self.source.requires_grad_(False)
        self.method = method
        self.rank = 1 if method == "LoRA" else 2
        self.register_buffer("core_adaptation_scale", source.core.new_tensor(1.0))
        m, n = source.base.weight.shape
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed))
            self.lora_a = nn.Linear(n, self.rank, bias=False)
            self.lora_b = nn.Linear(self.rank, m, bias=False)
            if method == "LoRA-FA":
                q, r = torch.linalg.qr(
                    torch.randn(n, self.rank, dtype=torch.float64), mode="reduced"
                )
                signs = torch.where(r.diag() < 0, -1.0, 1.0)
                self.lora_a.weight.data.copy_((q * signs).T.float())
            nn.init.zeros_(self.lora_b.weight)
        self.lora_a.to(source.core)
        self.lora_b.to(source.core)
        self.lora_a.requires_grad_(method == "LoRA")

    def forward(self, x):
        return self.source(x) + self.core_adaptation_scale * self.lora_b(self.lora_a(x))

    def increment_matrix(self):
        return self.core_adaptation_scale * (self.lora_b.weight @ self.lora_a.weight)


class LowRankBatteryTSFM(NoPolicyBatteryTSFM):
    def __init__(self, model, method, seed=42):
        super().__init__(model)
        self.requires_grad_(False)
        self.method = "LoRA-FA" if method == "LoRA_FA" else method
        layers = source_layers(self)
        for index, (name, layer) in enumerate(layers):
            replace_layer(
                self.backbone,
                name,
                AppendedLowRankLinear(layer, self.method, seed + index * 1009),
            )

    @torch.no_grad()
    def set_adaptation_scale(self, scale):
        if not 0 <= float(scale) <= 1:
            raise ValueError("The registered post-fit shrinkage must be within [0,1]")
        for layer in self.backbone.modules():
            if isinstance(layer, AppendedLowRankLinear):
                layer.core_adaptation_scale.fill_(float(scale))
        return self

    def parameter_report(self):
        report = super().parameter_report()
        layers = [
            layer
            for layer in self.backbone.modules()
            if isinstance(layer, AppendedLowRankLinear)
        ]
        report.update(
            adaptation_method=self.method,
            adaptation_parameters=sum(
                value.numel() for value in self.parameters() if value.requires_grad
            ),
            rank=layers[0].rank,
            alpha=layers[0].rank,
            scale=1.0,
            dropout=0.0,
            frozen_A_bytes=sum(
                layer.lora_a.weight.numel() * layer.lora_a.weight.element_size()
                for layer in layers
            )
            if self.method == "LoRA-FA"
            else 0,
        )
        return report

    @torch.no_grad()
    def merged_copy(self):
        result = copy.deepcopy(self)
        for name, layer in list(result.backbone.named_modules()):
            if isinstance(layer, AppendedLowRankLinear):
                layer.source.base.weight.add_(layer.increment_matrix())
                replace_layer(result.backbone, name, layer.source)
        merged = NoPolicyBatteryTSFM(result)
        merged.requires_grad_(False)
        merged.official_backbone_identity = dict(
            merged.official_backbone_identity, merged_adaptation=self.method
        )
        return merged


class CalibrationBatteryTSFM(NoPolicyBatteryTSFM):
    """Fit delta-gamma and beta while preserving the source at zero coordinates."""

    def __init__(self, model):
        super().__init__(model)
        source_layers(self)
        self.requires_grad_(False)
        self.core_calibration = nn.Parameter(next(self.head.parameters()).new_zeros(2))

    @property
    def coordinates(self):
        return self.core_calibration

    def forward(self, early, query, coordinates=None):
        base = super().forward(early, query)
        coordinates = self.coordinates if coordinates is None else coordinates
        if coordinates.shape != (2,):
            raise ValueError("CAL2 has delta-gamma and beta coordinates")
        coordinates = coordinates.to(base)
        time = (query.to(base) - 100) / 1000
        return (
            base
            + coordinates[0] * (base - early["anchor"][:, None])
            + coordinates[1] * time
        )

    def parameter_report(self):
        return dict(
            super().parameter_report(),
            adaptation_method="CAL2",
            adaptation_parameters=2,
            gamma_initial=1.0,
            beta_initial=0.0,
        )


def dictionary_jacobian(model, early, query, backend="autograd"):
    """Cache a single physical cell's f0 and d f / d a at zero coordinates.

    The encoder is differentiated q times. The small query head then applies
    the chain rule independently of the number of observed future samples.
    """
    if len(early["x"]) != 1 or not isinstance(model, DictionaryBatteryTSFM):
        raise ValueError("Dictionary Jacobian caching expects one physical cell")
    if backend not in ("autograd", "func"):
        raise ValueError("Choose the explicit autograd or func JVP backend")
    model.eval()
    zero = torch.zeros_like(model.coordinates)
    identity = torch.eye(model.dimension, device=zero.device, dtype=zero.dtype)
    encoder = lambda value: model.encode_with_coordinates(early, value)
    jvp = torch.autograd.functional.jvp if backend == "autograd" else torch.func.jvp
    columns = []
    source_state = None
    for tangent in identity:
        state, state_derivative = jvp(encoder, (zero,), (tangent,))
        source_state = state.detach()
        head = lambda value: model.head(value, early["anchor"], query)
        _, derivative = jvp(head, (source_state,), (state_derivative.detach(),))
        columns.append(derivative[0].detach())
    with torch.no_grad():
        prediction = model.head(source_state, early["anchor"], query)[0]
    return prediction.detach(), torch.stack(columns, dim=1)


def calibration_jacobian(model, early, query):
    """Return the exact two-column CAL2 design at its zero update."""
    if len(early["x"]) != 1 or not isinstance(model, CalibrationBatteryTSFM):
        raise ValueError("CAL2 caching expects one physical cell")
    model.eval()
    with torch.no_grad():
        prediction = model(
            early, query, coordinates=torch.zeros_like(model.coordinates)
        )[0]
    time = ((query.to(prediction) - 100) / 1000).reshape(-1)
    design = torch.stack((prediction - early["anchor"][0], time), dim=1)
    return prediction.detach(), design.detach()
