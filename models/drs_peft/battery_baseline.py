"""Fresh M1 encoder baseline with source-only preprocessing and the shared head."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from models.unified_matr.model import (
    SingleDischargeFrontend, PhysicalResponseEncoder, SoftPersistence,
    MambaBlock, initialize, clean_observed, sinusoidal_position,
)
from scripts.unified_matr.physics_features import GLOBAL_COLUMNS, LOCAL_COLUMNS
from .trajectory import CycleQueryHead


PHYSICS_PAIRS = {
    "global_features": "global_mask", "local_residual": "local_mask",
    "local_features": "local_feature_mask",
}


def reference_scaled_viq(x, point_mask, cycle_ids):
    """Same-curve I/Q divided by a legal per-cell early Q-span.

    Canonical I and Q have both been divided by nominal capacity. Dividing by
    the canonical early Q-span therefore reproduces the legacy reference units.
    """
    raw = torch.where(point_mask[..., None], x[..., :3], 0.0)
    reference_mask = point_mask & ((cycle_ids >= 1) & (cycle_ids <= 10))[..., None]
    q = torch.where(reference_mask, raw[..., 2], float("nan"))
    quantiles = torch.nanquantile(q, raw.new_tensor([.05, .95]), dim=-1)
    spans = quantiles[1]-quantiles[0]
    scale = torch.nanquantile(spans, .5, dim=1)
    scale = torch.where(torch.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    divisor = torch.stack((torch.ones_like(scale), scale, scale), -1)[:, None, None]
    return raw, raw/divisor


def _feature_floor(name):
    if name.endswith("support_count"):
        return 1.0
    if name.endswith("signed_direction"):
        return 1.0
    if name.endswith(("availability_fraction", "direction_consistency")):
        return .1
    if "slope_change_per_cycle_squared" in name:
        return 1e-5
    if "slope_per_cycle" in name:
        return 1e-4
    if name.startswith("b_V") or name.endswith(("centroid_V", "peak_V")):
        return .02
    if "Ah2" in name:
        return 1e-6
    return 1e-3


def _canonical_physics(early, key):
    value = early[key]
    nominal = early["nominal_capacity_Ah"]
    grid = early["voltage_grid_V"]
    if grid.ndim == 1:
        grid = grid[None].expand(len(nominal), -1)
    span = grid[:, -1]-grid[:, 0]
    if key == "local_residual":
        return value/nominal[:, None, None]
    names = GLOBAL_COLUMNS if key == "global_features" else LOCAL_COLUMNS
    columns = []
    dimensions = value.ndim-2
    capacity = nominal.reshape(-1, *([1]*dimensions))
    window = span.reshape(-1, *([1]*dimensions))
    left = grid[:, 0].reshape(-1, *([1]*dimensions))
    for j, name in enumerate(names):
        column = value[..., j]
        if name.endswith(("signed_direction", "direction_consistency", "support_count", "availability_fraction")):
            columns.append(column)
        elif "Ah2" in name:
            columns.append(column/capacity.square())
        elif "Ah" in name:
            columns.append(column/capacity)
        elif name.endswith(("centroid_V", "peak_V")):
            columns.append((column-left)/window)
        elif name.startswith("b_V"):
            columns.append(column/window)
        else:
            columns.append(column)
    return torch.stack(columns, -1)


def _components(early):
    raw, reference = reference_scaled_viq(early["x"], early["point_mask"], early["cycle_ids"])
    result = dict(raw_viq=raw, reference_viq=reference)
    result.update({key: _canonical_physics(early, key) for key in PHYSICS_PAIRS})
    return result


class B3InputScaler:
    """Source-train population moments with fixed physical standard-deviation floors."""
    def fit(self, early, training_rows, training_ids=None):
        rows = np.asarray(training_rows, dtype=int)
        if len(rows) == 0 or len(np.unique(rows)) != len(rows):
            raise ValueError("B3 scaling requires unique nonempty source-train rows")
        keys = ("x", "point_mask", "cycle_ids", "nominal_capacity_Ah", "voltage_grid_V",
                *PHYSICS_PAIRS, *PHYSICS_PAIRS.values())
        tensors = {key: torch.as_tensor(early[key][rows].copy()) for key in keys}
        components = _components(tensors)
        masks = dict(raw_viq=tensors["point_mask"][..., None].expand_as(components["raw_viq"]),
                     reference_viq=tensors["point_mask"][..., None].expand_as(components["reference_viq"]))
        masks.update({key: tensors[mask] for key, mask in PHYSICS_PAIRS.items()})
        self.statistics = {}
        for key, tensor in components.items():
            values = tensor.numpy().astype(float)
            mask = masks[key].numpy()
            if not np.isfinite(values[mask]).all():
                raise ValueError("A supported B3 source value is nonfinite")
            if key in ("raw_viq", "reference_viq"):
                floor = np.array([.1, 1.0, .05])
                axes = (0, 1, 2)
            elif key == "local_residual":
                floor, axes = 1e-3, (0, 1, 2)
            else:
                names = GLOBAL_COLUMNS if key == "global_features" else LOCAL_COLUMNS
                floor = np.array([_feature_floor(name) for name in names])
                axes = tuple(range(values.ndim-1))
            count = mask.sum(axis=axes)
            mean = np.divide(np.where(mask, values, 0).sum(axis=axes), count,
                             out=np.zeros_like(count, dtype=float), where=count>0)
            variance = np.divide(np.where(mask, (values-mean)**2, 0).sum(axis=axes), count,
                                 out=np.zeros_like(count, dtype=float), where=count>0)
            self.statistics[key] = dict(mean=np.asarray(mean).tolist(),
                std=np.maximum(np.sqrt(variance), floor).tolist(), supported=np.asarray(count).tolist(),
                physical_std_floor=np.asarray(floor).tolist())
        self.training_rows = rows.tolist()
        self.training_ids = list(training_ids) if training_ids is not None else []
        return self

    def to_dict(self):
        return dict(statistics=self.statistics, training_rows=self.training_rows, training_ids=self.training_ids,
                    policy_scaler="the shared SourceOnlyScaler output is used unchanged",
                    reference_definition="same-curve I/Q divided by median 95%-5% Q-span of observed cycles1..10",
                    zero_variance_rule="fixed physical unit floors; never target-fitted or arbitrarily clipped")


class B3ScalerModule(nn.Module):
    def __init__(self, metadata):
        super().__init__()
        self.metadata = metadata
        for key, record in metadata["statistics"].items():
            for name in ("mean", "std", "supported"):
                self.register_buffer(key+"_"+name, torch.as_tensor(record[name], dtype=torch.float32))

    def forward(self, early):
        components = _components(early)
        masks = dict(raw_viq=early["point_mask"][..., None].expand_as(components["raw_viq"]),
                     reference_viq=early["point_mask"][..., None].expand_as(components["reference_viq"]))
        masks.update({key: early[mask] for key, mask in PHYSICS_PAIRS.items()})
        result = dict(point_mask=early["point_mask"], cycle_ids=early["cycle_ids"],
                      policy=early["policy"].to(self.raw_viq_mean.dtype), policy_mask=early["policy_mask"])
        for key, values in components.items():
            values = values.to(getattr(self, key+"_mean").dtype)
            mask = masks[key] & (getattr(self, key+"_supported") > 0)
            normalized = (torch.where(mask, values, 0)-getattr(self, key+"_mean"))/getattr(self, key+"_std")
            result[key] = torch.where(mask, normalized, 0)
            if key in PHYSICS_PAIRS:
                result[PHYSICS_PAIRS[key]] = mask
        return result


class M1Encoder(nn.Module):
    """Legacy M1 operations through its exact mean/last 128-dimensional summary."""
    def __init__(self, policy_dim, global_dim=20, local_feature_dim=26, seed=42, dropout=.1):
        super().__init__()
        self.policy_dim = int(policy_dim)
        self.frontend = initialize(seed, "frontend", SingleDischargeFrontend)
        with torch.no_grad():
            self.frontend.phase_embedding.copy_(initialize(seed, "frontend_phase", lambda: torch.randn(64)*.02))
        self.local = initialize(seed, "local", lambda: MambaBlock(local=True, dropout=dropout))
        self.cycle_projection = initialize(seed, "cycle_projection", lambda: nn.Linear(66, 64))
        self.policy_encoder = initialize(seed, "policy", lambda: nn.Sequential(nn.Linear(2*policy_dim, 64), nn.GELU(), nn.Linear(64, 64)))
        self.history = initialize(seed, "history", lambda: nn.ModuleList([MambaBlock(dropout=dropout), MambaBlock(dropout=dropout)]))
        self.history_norm = initialize(seed, "history_norm", lambda: nn.LayerNorm(64))
        self.physics = initialize(seed, "physics", lambda: PhysicalResponseEncoder(global_dim, local_feature_dim))
        self.persistence = initialize(seed, "persistence", SoftPersistence)
        self.content_condition = initialize(seed, "content_condition", lambda: nn.Linear(64, 64))
        self.selector_condition = initialize(seed, "selector_condition", lambda: nn.Linear(64, 64))

    def forward(self, raw_viq, point_mask, cycle_ids, reference_viq, policy, policy_mask,
                global_features, global_mask, local_residual, local_mask, local_features, local_feature_mask):
        if raw_viq.ndim != 4 or raw_viq.shape[1:] != (100, 64, 3) or point_mask.shape != raw_viq.shape[:-1]:
            raise ValueError("B3 requires Bx100x64x3 VIQ and matching masks")
        if point_mask.dtype != torch.bool or cycle_ids.shape != raw_viq.shape[:2] or reference_viq.shape != raw_viq.shape:
            raise ValueError("B3 VIQ/reference/original-cycle axes do not match")
        valid = point_mask.any(-1)
        safe_ids = torch.where(valid, cycle_ids, 0)
        supported = safe_ids[valid]
        if not bool(torch.isfinite(supported).all() and (supported >= 1).all() and (supported <= 100).all()):
            raise ValueError("B3 accepts only declared observed prefix cycles1..100")
        if not bool((supported == supported.round()).all()):
            raise ValueError("B3 cycle identifiers must be integer-valued")
        previous = torch.where(valid, safe_ids, -torch.inf).cummax(1).values
        if bool((valid[:, 1:] & (safe_ids[:, 1:] <= previous[:, :-1])).any()):
            raise ValueError("B3 supported cycle identifiers must increase strictly")
        if policy.shape != (len(raw_viq), self.policy_dim):
            raise ValueError("B3 known-policy dimension differs")
        physical, physical_valid = self.physics(global_features, global_mask, local_residual, local_mask,
                                                 local_features, local_feature_mask, valid)
        condition, penalty = self.persistence(physical, physical_valid, safe_ids, persistent=False)
        raw_tokens, token_valid = self.frontend(raw_viq, point_mask)
        reference_tokens, _ = self.frontend(reference_viq, point_mask)
        raw_tokens = raw_tokens+self.content_condition(physical)[:, :, None]
        reference_tokens = reference_tokens+self.selector_condition(condition)[:, :, None]
        shape = raw_tokens.shape
        mixed = self.local(raw_tokens.reshape(-1, 8, 64), token_valid.reshape(-1, 8),
                           reference_tokens.reshape(-1, 8, 64)).reshape(shape)
        pooled = (mixed*token_valid[..., None]).sum(-2)/token_valid.sum(-1).clamp_min(1)[..., None]
        safe_raw = clean_observed(raw_viq, point_mask[..., None].expand_as(raw_viq), "VIQ")
        edges = point_mask[..., 1:] & point_mask[..., :-1]
        length = torch.where(edges, (safe_raw[..., 1:, :]-safe_raw[..., :-1, :]).abs().sum(-1), 0).sum(-1)
        encoded = self.cycle_projection(torch.cat((pooled, valid[..., None].to(raw_viq.dtype), length[..., None]), -1))
        policy_value = clean_observed(policy, policy_mask, "policy")
        policy_token = self.policy_encoder(torch.cat((policy_value, policy_mask.to(policy.dtype)), -1))
        encoded = torch.where(valid[..., None], encoded+sinusoidal_position(safe_ids, 64).to(encoded.dtype)+policy_token[:, None], 0)
        for block in self.history:
            encoded = block(encoded, valid)
        encoded = torch.where(valid[..., None], self.history_norm(encoded), 0)
        mean = encoded.sum(1)/valid.sum(1).clamp_min(1)[:, None]
        index = torch.where(valid, torch.arange(100, device=raw_viq.device)[None], -1).amax(-1)
        last = encoded[torch.arange(len(encoded), device=raw_viq.device), index.clamp_min(0)]
        last = torch.where((index >= 0)[:, None], last, 0)
        return torch.cat((mean, last), -1)


class B3M1SharedHead(nn.Module):
    method = "B3_M1_encoder_shared_head"
    input_keys = (
        "x", "point_mask", "cycle_ids", "policy", "policy_mask", "anchor",
        "nominal_capacity_Ah", "voltage_grid_V", "global_features", "global_mask",
        "local_residual", "local_mask", "local_features", "local_feature_mask",
    )

    def __init__(self, policy_dim, input_scaler, seed=42, dropout=.1):
        super().__init__()
        metadata = input_scaler.to_dict() if isinstance(input_scaler, B3InputScaler) else input_scaler
        self.input_scaler = B3ScalerModule(metadata)
        self.encoder = M1Encoder(policy_dim, seed=seed, dropout=dropout)
        self.state_fusion = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.LayerNorm(64))
        self.head = CycleQueryHead()

    def encode(self, early):
        return self.state_fusion(self.encoder(**self.input_scaler(early)))

    def forward(self, early, query):
        return self.head(self.encode(early), early["anchor"], query)

    def parameter_report(self):
        encoder = sum(value.numel() for value in self.encoder.parameters())
        fusion = sum(value.numel() for value in self.state_fusion.parameters())
        head = sum(value.numel() for value in self.head.parameters())
        bridge_trainable=sum(value.numel() for module in (self.encoder, self.state_fusion)
                             for value in module.parameters() if value.requires_grad)
        return dict(method=self.method, bridge=encoder+fusion, bridge_trainable=bridge_trainable,
                    head=head, head_trainable=sum(v.numel() for v in self.head.parameters() if v.requires_grad),
                    backbone_total=0, backbone_current_trainable=0, peft=0,
                    total=sum(v.numel() for v in self.parameters()), trainable=sum(v.numel() for v in self.parameters() if v.requires_grad),
                    battery_encoder=encoder, state_fusion=fusion, same_continuous_query_head=True,
                    legacy_flow_and_observation_heads_used=False, legacy_output_clamp_used=False,
                    input_budget="canonical VIQ plus prefix-scaled reference, explicit G20/local64/local8x26 and masks, known policy")
