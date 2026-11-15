"""Response-conditioned persistence and actual post-convolution SSM B routing."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .legacy_operators import dense_ssd
from .model import clean_observed, initialize


class _DeterministicCudaCumsum(torch.autograd.Function):
    """Keep a CPU scan and its adjoint within one CUDA autograd node."""

    @staticmethod
    def forward(context, value, dim):
        context.dim = dim
        return value.to("cpu").cumsum(dim).to(value.device)

    @staticmethod
    def backward(context, gradient):
        dim = context.dim
        cpu_gradient = gradient.to("cpu")
        reverse = cpu_gradient.flip((dim,)).cumsum(dim).flip((dim,))
        return reverse.to(gradient.device), None


def deterministic_cumsum(value, dim):
    if value.device.type == "cuda" and torch.are_deterministic_algorithms_enabled():
        return _DeterministicCudaCumsum.apply(value, dim)
    return value.cumsum(dim)


def previous_valid(valid, cycle_ids, dtype):
    positions = torch.arange(valid.shape[1], device=valid.device)
    last = torch.where(valid, positions[None], -1).cummax(1).values
    previous = F.pad(last[:, :-1], (1, 0), value=-1)
    seen = previous >= 0
    before = cycle_ids.gather(1, previous.clamp_min(0))
    before = torch.where(seen, before, 0.0)
    gap = (cycle_ids - before).clamp_min(1).to(dtype)
    return previous, seen, gap


def dynamic_ema(value, rho, valid, cycle_ids):
    """Dense causal expansion of channel-wise, time-varying retention."""
    previous, seen, gap = previous_valid(valid, cycle_ids, value.dtype)
    log_retention = rho.log() * gap[..., None]
    log_steps = torch.where((valid & seen)[..., None], log_retention, 0.0)
    accumulated = deterministic_cumsum(log_steps, 1).transpose(1, 2)
    log_decay = accumulated[..., :, None] - accumulated[..., None, :]
    positions = torch.arange(value.shape[1], device=value.device)
    causal = (positions[:, None] >= positions[None, :])[None] & valid[:, None, :]
    decay = log_decay.masked_fill(~causal[:, None], -torch.inf).exp()
    write = torch.where((valid & ~seen)[..., None], 1.0, -torch.expm1(log_retention))
    write = torch.where(valid[..., None], write, 0.0)
    safe_value = torch.where(valid[..., None], value, 0.0)
    internal = (decay @ (write * safe_value).transpose(1, 2)[..., None]).squeeze(-1).transpose(1, 2)
    before = internal.gather(1, previous.clamp_min(0)[..., None].expand_as(internal))
    before = torch.where(seen[..., None], before, 0.0)
    transition = valid & seen
    drift = torch.where(transition[..., None], (internal - before) / gap[..., None], 0.0)
    denominator = transition.sum().clamp_min(1) * value.shape[-1]
    penalty = drift.square().sum() / denominator
    state = torch.where(valid[..., None], internal, 0.0)
    effective = torch.where(valid[..., None], log_retention.exp(), 1.0)
    return state, penalty, {"rho": rho, "effective_retention": effective, "write_weight": write,
                            "gap": gap, "valid": valid, "previous_valid_index": previous, "seen": seen,
                            "state": state, "internal_state": internal, "drift_per_cycle": drift}


def drift_consistency(drift, mask):
    """Causal sign consistency over the last ten available robust drifts."""
    rank = deterministic_cumsum(mask.to(torch.long), 1)
    distance = rank[:, :, None, :] - rank[:, None, :, :]
    positions = torch.arange(drift.shape[1], device=drift.device)
    selected = ((distance >= 0) & (distance < 10) & mask[:, None]
                & (positions[:, None] >= positions[None, :])[None, :, :, None])
    count = selected.sum(2)
    total = (selected.to(drift.dtype) * drift.sign()[:, None]).sum(2)
    return total.abs() / count.clamp_min(1)


class DynamicPersistence(nn.Module):
    def __init__(self, policy_dim, global_dim, local_feature_dim):
        super().__init__()
        if global_dim < 12 or local_feature_dim < 7:
            raise ValueError("The frozen robust a/b and local noise feature columns are required")
        self.context_dim = 64 + 2 * global_dim + 32 + 2 * policy_dim + 2 + 4
        self.rho_net = nn.Sequential(nn.Linear(self.context_dim, 64), nn.GELU(), nn.Linear(64, 64))
        nn.init.normal_(self.rho_net[-1].weight, std=1e-3)
        nn.init.constant_(self.rho_net[-1].bias, math.log(.9 / .1))

    def forward(self, value, valid, cycle_ids, global_features, global_mask, local_mask,
                local_features, local_feature_mask, policy, policy_mask, return_diagnostics=False):
        previous, seen, gap = previous_valid(valid, cycle_ids, value.dtype)
        supported_global = global_mask & valid[..., None]
        global_value = clean_observed(global_features, supported_global, "dynamic persistence global context")
        gathered = previous.clamp_min(0)[..., None].expand_as(global_value)
        previous_global = global_value.gather(1, gathered)
        previous_mask = supported_global.gather(1, gathered) & seen[..., None]
        previous_global = torch.where(previous_mask, previous_global, 0.0)
        coefficient_indices = torch.as_tensor([2, 11], device=value.device)
        robust = global_value.index_select(-1, coefficient_indices)
        prior_robust = previous_global.index_select(-1, coefficient_indices)
        robust_mask = supported_global.index_select(-1, coefficient_indices) & previous_mask.index_select(-1, coefficient_indices)
        robust_drift = torch.where(robust_mask, (robust - prior_robust) / gap[..., None], 0.0)
        supported_local = local_mask & valid[..., None]
        feature_mask = local_feature_mask & valid[..., None, None]
        noise_mask = feature_mask[..., 6]
        noise = clean_observed(local_features[..., 6], noise_mask, "dynamic persistence noise context")
        residual_fraction = supported_local.reshape(*value.shape[:2], 8, 8).to(value.dtype).mean(-1)
        feature_fraction = feature_mask.to(value.dtype).mean(-1)
        quality = torch.cat((residual_fraction, feature_fraction, noise, noise_mask.to(value.dtype)), -1)
        policy_value = clean_observed(policy, policy_mask, "dynamic persistence policy context")
        policy_context = torch.cat((policy_value, policy_mask.to(value.dtype)), -1)
        policy_context = policy_context[:, None].expand(-1, value.shape[1], -1)
        context = torch.cat((value, previous_global, previous_mask.to(value.dtype), quality, policy_context,
                             gap[..., None], seen[..., None].to(value.dtype), robust_drift, robust_mask.to(value.dtype)), -1)
        rho = self.rho_net(context).sigmoid().clamp(.01, .99)
        state, penalty, diagnostic = dynamic_ema(value, rho, valid, cycle_ids)
        if return_diagnostics:
            diagnostic.update(quality=quality, current_response=value, robust_drift_per_cycle=robust_drift,
                              robust_drift_mask=robust_mask,
                              drift_direction_consistency=drift_consistency(robust_drift, robust_mask))
        return state, penalty, quality, diagnostic if return_diagnostics else None


class WriteModulatedMixer(nn.Module):
    def __init__(self, original, seed, layer):
        super().__init__()
        self.inner, self.state, self.head_dim, self.heads = original.inner, original.state, original.head_dim, original.heads
        for name in ("in_proj", "conv1d", "norm", "out_proj"):
            setattr(self, name, getattr(original, name))
        for name in ("dt_bias", "A_log", "D"):
            self.register_parameter(name, getattr(original, name))
        self.write_projection = initialize(seed, f"methodology/history{layer}/write_projection", lambda: nn.Linear(64, self.state, bias=False))
        nn.init.zeros_(self.write_projection.weight)
        self.write_gate = initialize(seed, f"methodology/history{layer}/write_gate", lambda: nn.Linear(96, self.state))

    def forward(self, raw, valid, response, quality, response_valid, return_diagnostics=False):
        raw = torch.where(valid[..., None], raw, 0.0)
        projected = self.in_proj(raw)
        z, xbc, delta = torch.split(projected, (self.inner, self.inner + 2 * self.state, self.heads), -1)
        x, b, c = torch.split(xbc, (self.inner, self.state, self.state), -1)
        xbc = torch.cat((x, b, c), -1)
        xbc = F.silu(self.conv1d(xbc.transpose(1, 2))[:, :, :raw.shape[1]].transpose(1, 2))
        x, b, c = torch.split(torch.where(valid[..., None], xbc, 0.0), (self.inner, self.state, self.state), -1)
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        delta = torch.where(valid[..., None], F.softplus(delta.to(dtype) + self.dt_bias.to(dtype)), 0.0)
        gate = self.write_gate(torch.cat((response, quality), -1)).sigmoid()
        modulation_valid = valid & response_valid
        modification = torch.where(modulation_valid[..., None], gate * self.write_projection(response).tanh(), 0.0)
        modified_b = torch.where(valid[..., None], b + modification, 0.0)
        output = dense_ssd(x.reshape(*x.shape[:2], self.heads, self.head_dim), delta,
                           -self.A_log.to(dtype).exp(), modified_b, c, self.D)
        output = output.reshape(*x.shape[:2], self.inner).to(x.dtype)
        result = torch.where(valid[..., None], self.out_proj(self.norm(output, z)), 0.0)
        diagnostics = None
        if return_diagnostics:
            diagnostics = {"B": b, "Bprime": modified_b, "write_modulation": modification,
                           "gate": gate, "modulation_valid": modulation_valid,
                           "B_l2": b.norm(dim=-1), "Bprime_l2": modified_b.norm(dim=-1),
                           "delta_l2": modification.norm(dim=-1)}
        return result, diagnostics


class WriteModulatedHistoryBlock(nn.Module):
    def __init__(self, original, seed, layer):
        super().__init__()
        self.norm, self.dropout = original.norm, original.dropout
        self.mixer = WriteModulatedMixer(original.mixer, seed, layer)

    def forward(self, raw, valid, response, quality, response_valid, return_diagnostics=False):
        raw = torch.where(valid[..., None], raw, 0.0)
        normalized = torch.where(valid[..., None], self.norm(raw), 0.0)
        mixed, diagnostic = self.mixer(normalized, valid, response, quality, response_valid, return_diagnostics)
        result = raw + self.dropout(mixed)
        return torch.where(valid[..., None], result, 0.0), diagnostic
