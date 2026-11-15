"""Fixed seven-arm engineering for the unified MATR full-trajectory task."""

from __future__ import annotations

import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from .legacy_operators import LegacyFlowHead, MambaBlock, derived_seed, deterministic_cumsum, isolated_cpu_seed, sinusoidal_position


ARMS = ("M0", "M1", "M2", "M3", "M3-NL", "M3-NP", "M3-NR")
PERSISTENCE_ARMS = ("M2", "M3", "M3-NL", "M3-NR")
RATE_ARMS = ("M3", "M3-NL", "M3-NP")


def clean_observed(value, mask, name):
    if mask.dtype != torch.bool or value.shape != mask.shape:
        raise ValueError(f"{name} requires an equal-shaped boolean mask")
    safe = torch.where(mask, value, 0.0)
    if not bool(torch.isfinite(safe).all()):
        raise ValueError(f"{name} has nonfinite supported values")
    return safe


def initialize(seed, name, constructor):
    with isolated_cpu_seed(derived_seed(seed, "unified_matr/" + name)):
        return constructor()


class SingleDischargeFrontend(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.LayerNorm(64))
        self.phase_embedding = nn.Parameter(torch.empty(64).normal_(std=0.02))
        self.register_buffer("patch_position", sinusoidal_position(torch.arange(8), 64))

    def forward(self, viq, mask):
        batch = len(viq)
        safe = clean_observed(viq, mask[..., None].expand_as(viq), "VIQ")
        patches = safe.reshape(batch, 100, 8, 8, 3).transpose(-1, -2).flatten(-2)
        masks = mask.reshape(batch, 100, 8, 8)
        valid = masks.any(-1)
        tokens = self.embedding(torch.cat((patches, masks.to(viq.dtype)), -1))
        tokens = tokens + self.phase_embedding + self.patch_position[None, None]
        return torch.where(valid[..., None], tokens, 0.0), valid


class GlobalOnlyFrontend(nn.Module):
    """An active global-feature bottleneck for the complete local-input ablation."""
    def __init__(self):
        super().__init__()
        self.global_embedding = nn.Sequential(nn.Linear(64, 16), nn.GELU(), nn.Linear(16, 64), nn.LayerNorm(64))
        self.phase_embedding = nn.Parameter(torch.empty(64).normal_(std=0.02))
        self.register_buffer("patch_position", sinusoidal_position(torch.arange(8), 64))

    def forward(self, global_token, point_mask):
        valid = point_mask.reshape(len(global_token), 100, 8, 8).any(-1)
        tokens = self.global_embedding(global_token)[:, :, None] + self.phase_embedding + self.patch_position[None, None]
        return torch.where(valid[..., None], tokens, 0.0), valid


class PhysicalResponseEncoder(nn.Module):
    def __init__(self, global_dim, local_feature_dim, remove_local=False):
        super().__init__()
        self.remove_local = remove_local
        self.local_feature_dim = local_feature_dim
        self.global_encoder = nn.Sequential(nn.Linear(2 * global_dim, 32), nn.GELU(), nn.Linear(32, 64))
        if remove_local:
            # Every additional bottleneck parameter acts on available global inputs.
            # Integer hidden width gives a close, explicitly reported budget match.
            removed_budget = 14640 + 64 * local_feature_dim if local_feature_dim else 13552
            hidden = max(1, round((removed_budget - 64) / 129))
            self.global_only_expansion = nn.Sequential(nn.Linear(64, hidden), nn.GELU(), nn.Linear(hidden, 64))
            self.global_combine = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.LayerNorm(64))
            self.match_hidden = hidden
        else:
            self.residual_encoder = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 16))
            if local_feature_dim:
                self.local_encoder = nn.Sequential(nn.Linear(2 * local_feature_dim, 32), nn.GELU(), nn.Linear(32, 16))
                self.bin_combine = nn.Linear(32, 16)
            self.bin_position = nn.Parameter(torch.empty(8, 16).normal_(std=0.02))
            self.local_projection = nn.Linear(128, 64)
            self.combine = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.LayerNorm(64))
            self.match_hidden = None

    def forward(self, global_features, global_mask, local_residual, local_mask, local_features, local_feature_mask, record_valid):
        supported_global = global_mask & record_valid[..., None]
        global_value = clean_observed(global_features, supported_global, "global features")
        global_token = self.global_encoder(torch.cat((global_value, supported_global.to(global_value.dtype)), -1))
        if self.remove_local:
            # No local value, local mask, or local-derived feature is inspected here.
            value = self.global_combine(global_token + self.global_only_expansion(global_token))
        else:
            supported_local = local_mask & record_valid[..., None]
            residual = clean_observed(local_residual, supported_local, "local residual")
            residual = residual.reshape(*residual.shape[:2], 8, 8)
            residual_mask = supported_local.reshape(*supported_local.shape[:2], 8, 8)
            bins = self.residual_encoder(torch.cat((residual, residual_mask.to(residual.dtype)), -1))
            bin_valid = residual_mask.any(-1)
            if self.local_feature_dim:
                feature_mask = local_feature_mask & record_valid[..., None, None]
                feature_value = clean_observed(local_features, feature_mask, "local features")
                feature_bins = self.local_encoder(torch.cat((feature_value, feature_mask.to(feature_value.dtype)), -1))
                bins = self.bin_combine(torch.cat((bins, feature_bins), -1))
                bin_valid = bin_valid | feature_mask.any(-1)
            bins = torch.where(bin_valid[..., None], bins + self.bin_position, 0.0)
            local_token = self.local_projection(bins.flatten(-2))
            value = self.combine(torch.cat((global_token, local_token), -1))
        available = record_valid & supported_global.any(-1)
        if not self.remove_local:
            available = record_valid & (supported_global.any(-1) | bin_valid.any(-1))
        return torch.where(available[..., None], value, 0.0), available


class SoftPersistence(nn.Module):
    def __init__(self):
        super().__init__()
        self.retention_logits = nn.Parameter(torch.full((64,), math.log(0.9 / 0.1)))

    def forward(self, value, valid, cycle_ids, persistent):
        retain = self.retention_logits.sigmoid()
        if not persistent:
            return value * retain, value.new_zeros(())
        steps = torch.arange(value.shape[1], device=value.device)
        last_valid = torch.where(valid, steps[None], -1).cummax(1).values
        previous_index = F.pad(last_valid[:, :-1], (1, 0), value=-1)
        seen = previous_index >= 0
        previous_cycle = cycle_ids.gather(1, previous_index.clamp_min(0))
        previous_cycle = torch.where(seen, previous_cycle, 0.0)
        gap = (cycle_ids - previous_cycle).clamp_min(1).to(value.dtype)
        elapsed = deterministic_cumsum(torch.where(valid & seen, gap, 0.0), 1)
        distance = elapsed[:, :, None] - elapsed[:, None, :]
        causal = (steps[:, None] >= steps[None, :])[None] & valid[:, None, :]
        # The zero-retention limit is explicit, avoiding log(0) and preserving
        # the original saturated-sigmoid state and gradient behavior.
        positive = retain > 0
        safe_retain = torch.where(positive, retain, 1.0)
        log_decay = safe_retain.log()[None, :, None, None] * distance[:, None]
        decay = log_decay.masked_fill(~causal[:, None], -torch.inf).exp()
        zero_decay = ((distance == 0) & causal).to(value.dtype)[:, None]
        decay = torch.where(positive[None, :, None, None], decay, zero_decay)
        write = torch.where((valid & ~seen)[..., None], 1.0, 1 - retain[None, None].pow(gap[..., None]))
        write = torch.where(valid[..., None], write, 0.0)
        safe_value = torch.where(valid[..., None], value, 0.0)
        weighted = (write * safe_value).transpose(1, 2)
        state = (decay @ weighted[..., None]).squeeze(-1).transpose(1, 2)
        previous_state = state.gather(1, previous_index.clamp_min(0)[..., None].expand_as(state))
        previous_state = torch.where(seen[..., None], previous_state, 0.0)
        change = (state - previous_state) / gap[..., None]
        penalties = torch.where((valid & seen)[..., None], change.square(), 0.0)
        transitions = (valid.sum(1) - valid.any(1).to(torch.long)).sum()
        denominator = transitions.clamp_min(1) * value.shape[-1]
        return torch.where(valid[..., None], state, 0.0), penalties.sum() / denominator


class RateFlowHead(nn.Module):
    def __init__(self):
        super().__init__()
        # 2,752 active parameters, exactly matching LegacyFlowHead.
        self.net = nn.Sequential(nn.Linear(33, 45), nn.GELU(), nn.Linear(45, 26), nn.GELU(), nn.Linear(26, 1, bias=False))
        nn.init.normal_(self.net[-1].weight, std=1e-3)
        self.register_buffer("rate_offset", torch.tensor(math.log(math.expm1(0.1))))

    def forward(self, state, condition, query_cycles):
        end = int(torch.ceil(query_cycles.max()).detach().item())
        grid = torch.arange(100, max(101, end) + 1, device=state.device, dtype=state.dtype)
        tau = (grid - 100) / 1000.0
        context = torch.cat((state, condition), -1)[:, None].expand(-1, len(grid), -1)
        features = torch.cat((context, tau[None, :, None].expand(len(state), -1, -1)), -1)
        rate = F.softplus(self.net(features).squeeze(-1) + self.rate_offset)
        increments = (rate[:, 1:] + rate[:, :-1]) * 0.0005
        cumulative = torch.cat((rate.new_zeros(len(state), 1), deterministic_cumsum(increments, 1)), -1)
        location = (query_cycles - 100).clamp_min(0)
        left = location.floor().long().clamp_max(len(grid) - 2)
        fraction = location - left.to(location.dtype)
        loss = cumulative.gather(1, left) + fraction * (cumulative.gather(1, left + 1) - cumulative.gather(1, left))
        instantaneous = rate.gather(1, left) + fraction * (rate.gather(1, left + 1) - rate.gather(1, left))
        return loss, instantaneous


class UnifiedMATRModel(nn.Module):
    def __init__(self, arm, policy_dim, global_dim, local_feature_dim, seed=42, dropout=0.1, observation_bound=0.03, soh_bounds=(0.0, 1.2)):
        super().__init__()
        if arm not in ARMS or seed != 42:
            raise ValueError("The registered matrix requires one of seven arms and seed42")
        if policy_dim < 1 or global_dim < 2 or local_feature_dim < 0:
            raise ValueError("Known policy and explicit global/local feature dimensions are required")
        self.arm, self.seed = arm, seed
        self.policy_dim, self.global_dim, self.local_feature_dim = policy_dim, global_dim, local_feature_dim
        self.observation_bound, self.soh_bounds = float(observation_bound), tuple(soh_bounds)
        self.frontend = initialize(seed, "frontend", GlobalOnlyFrontend if arm == "M3-NL" else SingleDischargeFrontend)
        with torch.no_grad():
            self.frontend.phase_embedding.copy_(initialize(seed, "frontend_phase", lambda: torch.randn(64) * 0.02))
        self.local = initialize(seed, "local", lambda: MambaBlock(local=True, dropout=dropout))
        self.cycle_projection = initialize(seed, "cycle_projection", lambda: nn.Linear(66, 64))
        self.policy_encoder = initialize(seed, "policy", lambda: nn.Sequential(nn.Linear(2 * policy_dim, 64), nn.GELU(), nn.Linear(64, 64)))
        self.history = initialize(seed, "history", lambda: nn.ModuleList([MambaBlock(dropout=dropout), MambaBlock(dropout=dropout)]))
        self.history_norm = initialize(seed, "history_norm", lambda: nn.LayerNorm(64))
        self.condition_pool = initialize(seed, "condition_pool", lambda: nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 16)))
        self.state_projection = initialize(seed, "state_projection", lambda: nn.Sequential(nn.Linear(128, 40), nn.GELU(), nn.Linear(40, 16)))
        self.observation_head = initialize(seed, "observation", lambda: nn.Sequential(nn.Linear(33, 32), nn.GELU(), nn.Linear(32, 1)))
        if arm != "M0":
            self.physics = initialize(seed, "physics", lambda: PhysicalResponseEncoder(global_dim, local_feature_dim, remove_local=arm == "M3-NL"))
            self.persistence = initialize(seed, "persistence", SoftPersistence)
            self.content_condition = initialize(seed, "content_condition", lambda: nn.Linear(64, 64))
            self.selector_condition = initialize(seed, "selector_condition", lambda: nn.Linear(64, 64))
        self.flow_head = initialize(seed, "rate_head" if arm in RATE_ARMS else "legacy_head", RateFlowHead if arm in RATE_ARMS else LegacyFlowHead)

    def forward(self, raw_viq, point_mask, cycle_ids, reference_viq, policy, policy_mask,
                global_features, global_mask, local_residual, local_mask, local_features,
                local_feature_mask, anchor_soh100, query_cycles, return_context=False):
        if raw_viq.ndim != 4 or raw_viq.shape[1:] != (100, 64, 3) or point_mask.shape != raw_viq.shape[:-1]:
            raise ValueError("The data contract requires Bx100x64x3 discharge VIQ and matching masks")
        if point_mask.dtype != torch.bool or cycle_ids.shape != raw_viq.shape[:2] or reference_viq.shape != raw_viq.shape:
            raise ValueError("VIQ/reference/original-cycle axes do not match")
        record_valid = point_mask.any(-1)
        supported_ids = cycle_ids[record_valid]
        if not bool(torch.isfinite(supported_ids).all() and (supported_ids >= 1).all() and (supported_ids <= 100).all() and (supported_ids == supported_ids.floor()).all()):
            raise ValueError("Only original cycles1..100 may enter the inference prefix")
        safe_ids = torch.where(record_valid, cycle_ids, 0)
        prior_ids = safe_ids.cummax(1).values[:, :-1]
        if bool(((safe_ids[:, 1:] <= prior_ids) & record_valid[:, 1:] & (prior_ids > 0)).any()):
            raise ValueError("Supported original cycles must be strictly increasing")
        if policy.shape != (len(raw_viq), self.policy_dim):
            raise ValueError("Known-policy dimensionality differs from the frozen model contract")
        policy_value = clean_observed(policy, policy_mask, "policy")
        policy_token = self.policy_encoder(torch.cat((policy_value, policy_mask.to(policy.dtype)), -1))
        persistence_penalty = raw_viq.new_zeros(())
        if self.arm != "M0":
            if global_features.shape != (*raw_viq.shape[:2], self.global_dim):
                raise ValueError("Global feature dimensions differ from the frozen model contract")
            physical, physical_valid = self.physics(global_features, global_mask, local_residual, local_mask,
                                                     local_features, local_feature_mask, record_valid)
            condition, persistence_penalty = self.persistence(physical, physical_valid, safe_ids, self.arm in PERSISTENCE_ARMS)
        if self.arm == "M3-NL":
            # Raw/reference values and every local-derived input are excluded.
            raw_tokens, token_valid = self.frontend(physical, point_mask)
            reference_tokens = raw_tokens
        else:
            raw_tokens, token_valid = self.frontend(raw_viq, point_mask)
            reference_tokens, _ = self.frontend(reference_viq, point_mask)
        if self.arm != "M0":
            raw_tokens = raw_tokens + self.content_condition(physical)[:, :, None]
            reference_tokens = reference_tokens + self.selector_condition(condition)[:, :, None]
        shape = raw_tokens.shape
        mixed = self.local(raw_tokens.reshape(-1, 8, 64), token_valid.reshape(-1, 8), reference_tokens.reshape(-1, 8, 64)).reshape(shape)
        pooled = (mixed * token_valid[..., None]).sum(-2) / token_valid.sum(-1).clamp_min(1)[..., None]
        if self.arm == "M3-NL":
            length = point_mask.to(pooled.dtype).mean(-1)
        else:
            safe_raw = clean_observed(raw_viq, point_mask[..., None].expand_as(raw_viq), "VIQ")
            edges = point_mask[..., 1:] & point_mask[..., :-1]
            length = torch.where(edges, (safe_raw[..., 1:, :] - safe_raw[..., :-1, :]).abs().sum(-1), 0.0).sum(-1)
        encoded = self.cycle_projection(torch.cat((pooled, record_valid[..., None].to(raw_viq.dtype), length[..., None]), -1))
        encoded = torch.where(record_valid[..., None], encoded + sinusoidal_position(safe_ids, 64).to(encoded.dtype) + policy_token[:, None], 0.0)
        for block in self.history:
            encoded = block(encoded, record_valid)
        encoded = torch.where(record_valid[..., None], self.history_norm(encoded), 0.0)
        mean = encoded.sum(1) / record_valid.sum(1).clamp_min(1)[:, None]
        indices = torch.where(record_valid, torch.arange(100, device=raw_viq.device)[None], -1).max(-1).values
        last = encoded[torch.arange(len(encoded), device=raw_viq.device), indices.clamp_min(0)]
        last = torch.where((indices >= 0)[:, None], last, 0.0)
        summary = torch.cat((mean, last), -1)
        condition, state = self.condition_pool(summary), self.state_projection(summary).tanh()
        query = torch.as_tensor(query_cycles, device=raw_viq.device, dtype=raw_viq.dtype)
        if query.ndim == 1:
            query = query[None].expand(len(raw_viq), -1)
        if query.ndim != 2 or query.shape[0] != len(raw_viq) or query.shape[1] == 0 or not bool(torch.isfinite(query).all() and (query >= 100).all()):
            raise ValueError("Queries must be finite known coordinates at or after original cycle100")
        if anchor_soh100.shape != (len(raw_viq),) or not bool(torch.isfinite(anchor_soh100).all()):
            raise ValueError("A finite observed-prefix SOH100 anchor is required for every arm")
        tau = (query - 100) / 1000.0
        if self.arm in RATE_ARMS:
            integrated_loss, rate = self.flow_head(state, condition, query)
            trend = anchor_soh100[:, None] - integrated_loss
        else:
            trend = anchor_soh100[:, None] + self.flow_head(state, condition, tau)
            rate = None
        observation_input = torch.cat((state[:, None].expand(-1, query.shape[1], -1), condition[:, None].expand(-1, query.shape[1], -1), tau[..., None]), -1)
        residual = self.observation_bound * self.observation_head(observation_input).squeeze(-1).tanh()
        result = {"trajectory": (trend + residual).clamp(*self.soh_bounds), "trend": trend.clamp(*self.soh_bounds),
                  "observation_residual": residual, "persistence_penalty": persistence_penalty}
        if rate is not None:
            result["rate_per_1000_cycles"] = rate
        if return_context:
            result["context"] = {"state": state, "condition": condition, "record_valid": record_valid,
                                 "raw_trend": trend, "queries": query}
        return result


def state_hash(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def model_registry(model):
    return {"arm": model.arm, "seed": model.seed, "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "initial_state_sha256": state_hash(model), "policy_dim": model.policy_dim,
            "global_dim": model.global_dim, "local_feature_dim": model.local_feature_dim,
            "flow_head_parameters": sum(p.numel() for p in model.flow_head.parameters()),
            "single_discharge_patch_geometry": [8, 8], "local_width_heads_state": [64, 4, 4],
            "history_layers_width_state": [2, 64, 16], "latent_state_dim": 16,
            "persistent": model.arm in PERSISTENCE_ARMS, "rate_flow": model.arm in RATE_ARMS,
            "local_removed": model.arm == "M3-NL", "nonlocal_matching_hidden": getattr(getattr(model, "physics", None), "match_hidden", None),
            "input_ablation": "global_features_policy_anchor_and_support_only" if model.arm == "M3-NL" else "raw_reference_VIQ_with_arm_specific_physics",
            "observation_bound": model.observation_bound, "soh_bounds": list(model.soh_bounds),
            "time_scale_cycles": 1000, "initialization": "from_scratch_per_module_seed42_common_state_keys_identical"}
