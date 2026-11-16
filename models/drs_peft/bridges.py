"""Prefix-only physical-statistics and cycle-embedding bridges to official Moirai."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .backbone import MoiraiBackbone


FEATURE_NAMES = (
    "mean_voltage_V", "mean_signed_current_C_rate", "mean_capacity_SOH", "mean_early_drift_SOH",
    "std_voltage_V", "std_signed_current_C_rate", "std_capacity_SOH", "std_early_drift_SOH",
    "capacity_span_SOH", "early_drift_span_SOH", "voltage_capacity_covariance_V_SOH",
    "voltage_early_drift_covariance_V_SOH",
)
PHYSICAL_SCALES = (5.0, 5.0, 1.0, 1.0)


def masked_mean(value, mask, dim):
    safe = torch.where(mask, value, 0.0)
    return safe.sum(dim) / mask.sum(dim).clamp_min(1)


def validate_prefix(x, point_mask, cycle_ids, policy, policy_mask, anchor, policy_dim):
    if x.ndim != 4 or x.shape[1:] != (100, 64, 4):
        raise ValueError("Bridge input must be [B,100,64,4] in V, signed C-rate, Q/SOH and early-drift/SOH units")
    batch = len(x)
    if point_mask.dtype != torch.bool or point_mask.shape != x.shape[:-1]:
        raise ValueError("point_mask must be a matching boolean array")
    if cycle_ids.shape != (batch, 100):
        raise ValueError("cycle_ids must be [B,100]")
    if policy.shape != (batch, policy_dim) or policy_mask.shape != policy.shape or policy_mask.dtype != torch.bool:
        raise ValueError("Policy values and boolean masks differ from the registered dimension")
    if anchor.shape != (batch,) or not bool(torch.isfinite(anchor).all()):
        raise ValueError("Each cell needs a finite K100 SOH anchor")
    curve_valid = point_mask.any(-1)
    cycles = torch.where(curve_valid, cycle_ids, 0)
    valid_cycles = cycles[curve_valid]
    if not bool(torch.isfinite(valid_cycles).all() and (valid_cycles >= 1).all() and
                (valid_cycles <= 100).all() and (valid_cycles == valid_cycles.floor()).all()):
        raise ValueError("Supported prefix coordinates must be registered integer cycles 1 through 100")
    safe = torch.where(point_mask[..., None], x, 0.0)
    if not bool(torch.isfinite(safe).all() and torch.isfinite(torch.where(policy_mask, policy, 0.0)).all()):
        raise ValueError("A supported electrical or policy value is nonfinite")
    return safe, curve_valid, cycles


def cycle_statistics(safe, point_mask):
    """Twelve fixed electrical descriptors, with per-descriptor support masks.

    Each voltage coordinate comes from measured V. No voltage-bin identity is
    transferred across data domains. All descriptors are dimensionless after
    division by the declared physical unit scales.
    """
    scales = safe.new_tensor(PHYSICAL_SCALES)
    values = safe / scales
    observed = point_mask[..., None].expand_as(values)
    mean = masked_mean(values, observed, -2)
    variance = masked_mean((values-mean[..., None, :]).square(), observed, -2)
    deviation = torch.sqrt(variance + 1e-10)
    count = point_mask.sum(-1)
    available = count > 0
    variation_available = count > 1
    q = values[..., 2:4]
    qmask = point_mask[..., None].expand_as(q)
    maximum = torch.where(qmask, q, float("-inf")).amax(-2)
    minimum = torch.where(qmask, q, float("inf")).amin(-2)
    span = torch.where(available[..., None], maximum-minimum, 0.0)
    centered_v = values[..., 0]-mean[..., 0, None]
    centered_q = q-mean[..., None, 2:4]
    covariance = masked_mean(centered_v[..., None]*centered_q, qmask, -2)
    features = torch.cat((mean, deviation, span, covariance), -1)
    masks = torch.cat((available[..., None].expand(*available.shape, 4),
                       variation_available[..., None].expand(*available.shape, 8)), -1)
    return torch.where(masks, features, 0.0), masks


def prefix_descriptors(features, masks, cycles):
    mean = masked_mean(features, masks, 1)
    variance = masked_mean((features-mean[:, None]).square(), masks, 1)
    deviation = torch.sqrt(variance+1e-10)
    available = masks.any(1)
    deviation = torch.where(available, deviation, 0.0)
    early = masks & ((cycles > 0) & (cycles <= 20))[..., None]
    late = masks & ((cycles >= 81) & (cycles <= 100))[..., None]
    difference_mask = early.any(1) & late.any(1)
    difference = masked_mean(features, late, 1)-masked_mean(features, early, 1)
    difference = torch.where(difference_mask, difference, 0.0)
    return torch.cat((mean, deviation, difference, masks.float().mean(1), difference_mask.float()), -1)


def temporal_pool(tokens, valid):
    average = masked_mean(tokens, valid[..., None].expand_as(tokens), -2)
    positions = torch.arange(valid.shape[-1], device=valid.device)
    index = torch.where(valid, positions, 0).amax(-1)
    gathered = tokens.gather(-2, index[..., None, None].expand(*index.shape, 1, tokens.shape[-1])).squeeze(-2)
    last = torch.where(valid.any(-1)[..., None], gathered, 0.0)
    return average, last


class PrefixBridge(nn.Module):
    """Shared absolute-prefix, policy, anchor context and 64-dimensional output."""
    kind = "base"

    def __init__(self, checkpoint_dir, policy_dim, pretrained=True, policy_scale=None):
        super().__init__()
        self.policy_dim = int(policy_dim)
        # Backbone construction and HF loading may consume different RNG paths.
        # Preserve the caller's CPU stream for matched bridge/head initial values.
        with torch.random.fork_rng(devices=[]):
            self.backbone = MoiraiBackbone(checkpoint_dir, pretrained=pretrained)
        scale = torch.ones(self.policy_dim) if policy_scale is None else torch.as_tensor(policy_scale, dtype=torch.float32)
        if scale.shape != (self.policy_dim,) or not bool(torch.isfinite(scale).all() and (scale>0).all()):
            raise ValueError("Policy scales must be fixed finite positive values, one per registered field")
        self.register_buffer("policy_scale", scale)
        self.fusion = nn.Sequential(nn.Linear(829+2*self.policy_dim, 64), nn.GELU(), nn.LayerNorm(64))

    def encode(self, x, point_mask, cycle_ids, policy, policy_mask, anchor):
        safe, valid, cycles = validate_prefix(x, point_mask, cycle_ids, policy, policy_mask, anchor, self.policy_dim)
        features, feature_mask = cycle_statistics(safe, point_mask)
        descriptors = prefix_descriptors(features, feature_mask, cycles)
        mean, last = self.encode_response(safe, point_mask, cycles, features, feature_mask)
        known_policy = torch.where(policy_mask, policy/self.policy_scale, 0.0)
        context = torch.cat((mean, last, descriptors, known_policy, policy_mask.float(), anchor[:, None]), -1)
        return self.fusion(context)

    def forward(self, x, point_mask, cycle_ids, policy, policy_mask, anchor):
        return self.encode(x, point_mask, cycle_ids, policy, policy_mask, anchor)

    def parameter_budget(self):
        backbone_ids = {id(value) for value in self.backbone.parameters()}
        bridge_parameters = [value for value in self.parameters() if id(value) not in backbone_ids]
        return dict(
            bridge=sum(value.numel() for value in bridge_parameters),
            bridge_trainable=sum(value.numel() for value in bridge_parameters if value.requires_grad),
            backbone_total=self.backbone.original_parameter_count,
            backbone_current_trainable=sum(value.numel() for value in self.backbone.parameters() if value.requires_grad),
            total_current_trainable=sum(value.numel() for value in self.parameters() if value.requires_grad),
        )


class NativeBridge(PrefixBridge):
    kind = "native_multivariate"
    patch_size = 8

    def token_inputs(self, features, feature_mask, cycles):
        batch, length, channels = features.shape
        pad = (-length) % self.patch_size
        values = F.pad(features, (0, 0, 0, pad))
        observed = F.pad(feature_mask, (0, 0, 0, pad), value=False)
        padded_cycles = F.pad(cycles, (0, pad))
        steps = values.shape[1]//self.patch_size
        target = values.reshape(batch, steps, self.patch_size, channels).permute(0, 3, 1, 2)
        mask = observed.reshape(batch, steps, self.patch_size, channels).permute(0, 3, 1, 2)
        target = F.pad(target.reshape(batch, channels*steps, self.patch_size), (0, 128-self.patch_size))
        mask = F.pad(mask.reshape(batch, channels*steps, self.patch_size), (0, 128-self.patch_size), value=False)
        valid = mask.any(-1)
        sample_id = valid.long()
        maximum_cycle = padded_cycles.reshape(batch, steps, self.patch_size).amax(-1)
        time_id = ((maximum_cycle-1)/self.patch_size).floor().clamp_min(0).long()
        time_id = time_id[:, None].expand(batch, channels, steps).reshape(batch, channels*steps)
        variate_id = torch.arange(channels, device=features.device)[None, :, None].expand(batch, channels, steps).reshape(batch, channels*steps)
        patches = torch.full_like(sample_id, self.patch_size)
        return dict(target=target, observed_mask=mask, sample_id=sample_id, time_id=time_id,
                    variate_id=variate_id, patch_size=patches), valid.reshape(batch, channels, steps)

    def encode_response(self, safe, point_mask, cycles, features, feature_mask):
        inputs, valid = self.token_inputs(features, feature_mask, cycles)
        encoded = self.backbone.encode_native(**inputs)
        tokens = encoded.reshape(*valid.shape, self.backbone.hidden_size)
        variable_mean, variable_last = temporal_pool(tokens, valid)
        variable_valid = valid.any(-1)
        mask = variable_valid[..., None].expand_as(variable_mean)
        return masked_mean(variable_mean, mask, 1), masked_mean(variable_last, mask, 1)


class CycleEmbeddingBridge(PrefixBridge):
    kind = "cycle_embedding"

    def __init__(self, checkpoint_dir, policy_dim, pretrained=True, policy_scale=None):
        super().__init__(checkpoint_dir, policy_dim, pretrained, policy_scale)
        self.cycle_encoder = nn.Sequential(
            nn.Conv1d(5, 32, kernel_size=5, padding=2), nn.GELU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2), nn.GELU(),
        )
        self.cycle_projection = nn.Sequential(nn.Linear(32, 384), nn.LayerNorm(384))

    def encode_response(self, safe, point_mask, cycles, features, feature_mask):
        batch, length = safe.shape[:2]
        scaled = safe/safe.new_tensor(PHYSICAL_SCALES)
        values = torch.cat((scaled, point_mask[..., None].float()), -1)
        values = values.reshape(batch*length, 64, 5).transpose(1, 2)
        encoded = self.cycle_encoder(values).transpose(1, 2)
        mask = point_mask.reshape(batch*length, 64, 1).expand_as(encoded)
        embedded = self.cycle_projection(masked_mean(encoded, mask, 1)).reshape(batch, length, 384)
        valid = point_mask.any(-1)
        embedded = torch.where(valid[..., None], embedded, 0.0)
        sample_id = valid.long()
        time_id = (cycles-1).clamp_min(0).long()
        variate_id = torch.zeros_like(sample_id)
        result = self.backbone.encode_hidden(embedded, sample_id, time_id, variate_id)
        return temporal_pool(result, valid)
