"""Complete M4 methodology with dynamic persistence and true history B writing."""

from __future__ import annotations

import torch
from torch import nn

from .legacy_operators import MambaBlock, sinusoidal_position
from .methodology_operators import DynamicPersistence, StableMaskedMamba2, StableRateFlowHead, WriteModulatedHistoryBlock
from .model import PhysicalResponseEncoder, RateFlowHead, SingleDischargeFrontend, clean_observed, initialize, state_hash


class MethodologyFullModel(nn.Module):
    def __init__(self, policy_dim, global_dim, local_feature_dim, seed=42, dropout=.1,
                 observation_bound=.03, soh_bounds=(0.0, 1.2)):
        super().__init__()
        if seed != 42 or policy_dim < 1:
            raise ValueError("The complete methodology uses seed42 and known policy inputs")
        self.arm, self.seed = "M4_FULL_METHOD", seed
        self.policy_dim, self.global_dim, self.local_feature_dim = policy_dim, global_dim, local_feature_dim
        self.observation_bound, self.soh_bounds = float(observation_bound), tuple(soh_bounds)
        self.frontend = initialize(seed, "frontend", SingleDischargeFrontend)
        with torch.no_grad():
            self.frontend.phase_embedding.copy_(initialize(seed, "frontend_phase", lambda: torch.randn(64) * .02))
        self.local = initialize(seed, "local", lambda: MambaBlock(local=True, dropout=dropout))
        self.local.mixer = StableMaskedMamba2(self.local.mixer)
        self.cycle_projection = initialize(seed, "cycle_projection", lambda: nn.Linear(66, 64))
        self.policy_encoder = initialize(seed, "policy", lambda: nn.Sequential(nn.Linear(2 * policy_dim, 64), nn.GELU(), nn.Linear(64, 64)))
        original_history = initialize(seed, "history", lambda: nn.ModuleList([MambaBlock(dropout=dropout), MambaBlock(dropout=dropout)]))
        self.history = nn.ModuleList([WriteModulatedHistoryBlock(block, seed, index) for index, block in enumerate(original_history)])
        self.history_norm = initialize(seed, "history_norm", lambda: nn.LayerNorm(64))
        self.condition_pool = initialize(seed, "condition_pool", lambda: nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 16)))
        self.state_projection = initialize(seed, "state_projection", lambda: nn.Sequential(nn.Linear(128, 40), nn.GELU(), nn.Linear(40, 16)))
        self.observation_head = initialize(seed, "observation", lambda: nn.Sequential(nn.Linear(33, 32), nn.GELU(), nn.Linear(32, 1)))
        self.physics = initialize(seed, "physics", lambda: PhysicalResponseEncoder(global_dim, local_feature_dim))
        self.persistence = initialize(seed, "methodology/dynamic_persistence", lambda: DynamicPersistence(policy_dim, global_dim, local_feature_dim))
        self.content_condition = initialize(seed, "content_condition", lambda: nn.Linear(64, 64))
        self.selector_condition = initialize(seed, "selector_condition", lambda: nn.Linear(64, 64))
        self.flow_head = initialize(seed, "rate_head", StableRateFlowHead)

    def forward(self, raw_viq, point_mask, cycle_ids, reference_viq, policy, policy_mask,
                global_features, global_mask, local_residual, local_mask, local_features,
                local_feature_mask, anchor_soh100, query_cycles, return_context=False,
                return_diagnostics=False):
        if raw_viq.ndim != 4 or raw_viq.shape[1:] != (100, 64, 3) or point_mask.shape != raw_viq.shape[:-1]:
            raise ValueError("M4 requires Bx100x64x3 prefix VIQ and corresponding point masks")
        if point_mask.dtype != torch.bool or cycle_ids.shape != raw_viq.shape[:2] or reference_viq.shape != raw_viq.shape:
            raise ValueError("Prefix masks, reference views and actual-cycle coordinates differ")
        record_valid = point_mask.any(-1)
        supported_ids = cycle_ids[record_valid]
        if not bool(torch.isfinite(supported_ids).all() and (supported_ids >= 1).all() and (supported_ids <= 100).all() and (supported_ids == supported_ids.floor()).all()):
            raise ValueError("Only original integer cycles1..100 may enter the prefix")
        safe_ids = torch.where(record_valid, cycle_ids, 0.0)
        previous = safe_ids.cummax(1).values[:, :-1]
        if bool(((safe_ids[:, 1:] <= previous) & record_valid[:, 1:] & (previous > 0)).any()):
            raise ValueError("Valid physical cycle IDs must be strictly increasing")
        if policy.shape != (len(raw_viq), self.policy_dim) or global_features.shape != (*raw_viq.shape[:2], self.global_dim):
            raise ValueError("Frozen policy or global feature dimensions differ")
        policy_value = clean_observed(policy, policy_mask, "policy")
        policy_token = self.policy_encoder(torch.cat((policy_value, policy_mask.to(policy.dtype)), -1))
        physical, physical_valid = self.physics(global_features, global_mask, local_residual, local_mask,
                                                 local_features, local_feature_mask, record_valid)
        response, penalty, quality, persistence_diagnostic = self.persistence(
            physical, physical_valid, safe_ids, global_features, global_mask, local_mask,
            local_features, local_feature_mask, policy, policy_mask, return_diagnostics)
        raw_tokens, token_valid = self.frontend(raw_viq, point_mask)
        reference_tokens, _ = self.frontend(reference_viq, point_mask)
        raw_tokens = raw_tokens + self.content_condition(physical)[:, :, None]
        reference_tokens = reference_tokens + self.selector_condition(response)[:, :, None]
        shape = raw_tokens.shape
        mixed = self.local(raw_tokens.reshape(-1, 8, 64), token_valid.reshape(-1, 8), reference_tokens.reshape(-1, 8, 64)).reshape(shape)
        pooled = (mixed * token_valid[..., None]).sum(-2) / token_valid.sum(-1).clamp_min(1)[..., None]
        safe_raw = clean_observed(raw_viq, point_mask[..., None].expand_as(raw_viq), "VIQ")
        edges = point_mask[..., 1:] & point_mask[..., :-1]
        length = torch.where(edges, (safe_raw[..., 1:, :] - safe_raw[..., :-1, :]).abs().sum(-1), 0.0).sum(-1)
        encoded = self.cycle_projection(torch.cat((pooled, record_valid[..., None].to(raw_viq.dtype), length[..., None]), -1))
        encoded = torch.where(record_valid[..., None], encoded + sinusoidal_position(safe_ids, 64).to(encoded.dtype) + policy_token[:, None], 0.0)
        history_diagnostics = []
        for block in self.history:
            encoded, diagnostic = block(encoded, record_valid, response, quality, physical_valid, return_diagnostics)
            if return_diagnostics:
                history_diagnostics.append(diagnostic)
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
            raise ValueError("Queries must be finite physical coordinates at or after cycle100")
        if anchor_soh100.shape != (len(raw_viq),) or not bool(torch.isfinite(anchor_soh100).all()):
            raise ValueError("A finite prefix SOH100 anchor is required")
        integrated, rate = self.flow_head(state, condition, query)
        raw_trend = anchor_soh100[:, None] - integrated
        tau = (query - 100) / 1000.0
        observation_input = torch.cat((state[:, None].expand(-1, query.shape[1], -1),
                                       condition[:, None].expand(-1, query.shape[1], -1), tau[..., None]), -1)
        residual = self.observation_bound * self.observation_head(observation_input).squeeze(-1).tanh()
        result = {"trajectory": (raw_trend + residual).clamp(*self.soh_bounds),
                  "trend": raw_trend.clamp(*self.soh_bounds), "raw_trend": raw_trend,
                  "integrated_loss": integrated, "observation_residual": residual,
                  "persistence_penalty": penalty, "rate_per_1000_cycles": rate, "rate_per_cycle": rate / 1000}
        if return_context:
            result["context"] = {"state": state, "condition": condition, "record_valid": record_valid,
                                 "raw_trend": raw_trend, "queries": query}
        if return_diagnostics:
            result["diagnostics"] = {"persistence": persistence_diagnostic, "history": history_diagnostics,
                                     "flow": {"queries": query, "raw_trend": raw_trend, "integrated_loss": integrated,
                                              "rate_per_1000_cycles": rate, "rate_per_cycle": rate / 1000,
                                              "anchor_soh100": anchor_soh100},
                                     "state": state, "condition": condition, "record_valid": record_valid}
        return result


def model_registry(model):
    return {"arm": model.arm, "seed": model.seed,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            "initial_state_sha256": state_hash(model), "policy_dim": model.policy_dim,
            "global_dim": model.global_dim, "local_feature_dim": model.local_feature_dim,
            "rho_context_dim": model.persistence.context_dim, "rho_hidden_width": 64, "rho_bounds": [.01, .99],
            "rho_explicit_robust_drift_columns": [2, 11], "local_quality_dim": 32,
            "rho_explicit_drift_semantics": "actual-gap differences of frozen causal a/b median states, not raw a/b coefficients",
            "history_layers_width_state": [2, 64, 16], "history_write_gate_input_dim": 96,
            "write_projection_zero_initialized": True, "write_projection_bias": False,
            "dynamic_retention": True, "actual_postconvolution_history_B_modulation": True,
            "persistence_penalty": "mean squared state change per actual-cycle gap over valid transitions and64channels",
            "rate_head_unit": "SOH per1000cycles", "rate_per_cycle_unit": "SOH percycle",
            "rate_integration": "fixed unit-cycle trapezoid with1000cycle time normalization",
            "observation_bound": model.observation_bound, "soh_bounds": list(model.soh_bounds),
            "initialization": "seed42 isolated CPU module initialization; common baseline state keys preserved"}
