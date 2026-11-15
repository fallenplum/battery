"""Four independent, single-change ablations of the frozen complete M4 model."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .legacy_operators import LegacyFlowHead, analytic_flow
from .methodology_full import MethodologyFullModel, model_registry as full_registry
from .methodology_operators import DynamicPersistence, StableMaskedMamba2
from .model import clean_observed, initialize, state_hash


ABLATION_ARMS = (
    "A1_M4_LEGACY_FLOW",
    "A2_M4_NO_B_WRITE",
    "A3_M4_STATIC_RHO",
    "A4_M4_NO_LOCAL_PHYSICS",
)


class LegacyFlowWithSignedRate(LegacyFlowHead):
    """Original decoder difference plus its detached, signed analytic derivative.

    The first return value is an interface-compatible signed capacity loss.
    It is computed directly from the analytic Flow and is never integrated.
    The derivative is diagnostic and contributes no additional training loss.
    """

    def forward(self, state, condition, query_cycles):
        tau = (query_cycles - 100) / 1000.0
        rate, mean = self.generator(condition).chunk(2, -1)
        rate, mean = F.softplus(rate) + 1e-4, mean.tanh()
        future = analytic_flow(state[:, None], tau[..., None], rate[:, None], mean[:, None])
        correction = self.decoder(future).squeeze(-1) - self.decoder(state).squeeze(-1)[:, None]
        change = torch.where(tau == 0, 0.0, .2 * correction)
        with torch.no_grad():
            tangent = -rate[:, None] * torch.exp(-rate[:, None] * tau[..., None]) * (state[:, None] - mean[:, None])
            preactivation = self.decoder[0](future)
            tangent = F.linear(tangent, self.decoder[0].weight)
            gelu_derivative = (.5 * (1 + torch.erf(preactivation / math.sqrt(2)))
                               + preactivation * torch.exp(-.5 * preactivation.square()) / math.sqrt(2 * math.pi))
            signed_rate = -.2 * F.linear(gelu_derivative * tangent, self.decoder[2].weight).squeeze(-1)
        return -change, signed_rate


class NoWriteMixer(StableMaskedMamba2):
    """Keep the history operator while removing both B-write parameter branches."""

    def forward(self, raw, valid, response, quality, response_valid, return_diagnostics=False):
        z, x, b, c, delta = self.components(raw, valid)
        output = self.finish(x, b, c, delta, z, valid)
        diagnostic = None
        if return_diagnostics:
            zero = torch.zeros_like(b)
            diagnostic = {"B": b, "Bprime": b, "write_modulation": zero,
                          "gate": zero, "modulation_valid": valid & response_valid,
                          "B_l2": b.norm(dim=-1), "Bprime_l2": b.norm(dim=-1),
                          "delta_l2": zero.norm(dim=-1)}
        return output, diagnostic


class StaticRetentionLogits(nn.Module):
    """One learned retention logit per channel, shared across cells and cycles."""

    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.full((64,), math.log(.9 / .1)))

    def forward(self, context):
        return self.logits.expand(*context.shape[:-1], -1)


class ActiveColumnsLinear(nn.Module):
    """Copy only coefficients multiplying retained input columns."""

    def __init__(self, original, columns):
        super().__init__()
        columns = torch.as_tensor(columns, dtype=torch.long)
        self.register_buffer("input_columns", columns)
        self.weight = nn.Parameter(original.weight.detach().index_select(1, columns).clone())
        self.bias = nn.Parameter(original.bias.detach().clone()) if original.bias is not None else None
        self.in_features, self.out_features = len(columns), original.out_features

    def forward(self, value):
        return F.linear(value.index_select(-1, self.input_columns), self.weight, self.bias)


class GlobalPhysicsEncoder(nn.Module):
    """The M4 global path with its local token fixed to zero and then removed."""

    def __init__(self, original):
        super().__init__()
        self.global_encoder = original.global_encoder
        self.combine = original.combine
        self.combine[0] = ActiveColumnsLinear(self.combine[0], range(64))

    def forward(self, global_features, global_mask, local_residual, local_mask,
                local_features, local_feature_mask, record_valid):
        supported = global_mask & record_valid[..., None]
        values = clean_observed(global_features, supported, "global features")
        token = self.global_encoder(torch.cat((values, supported.to(values.dtype)), -1))
        # The copied active-column layer accepts the original 128-column layout.
        combined = self.combine(torch.cat((token, torch.zeros_like(token)), -1))
        valid = record_valid & supported.any(-1)
        return torch.where(valid[..., None], combined, 0.0), valid


class GlobalQualityFreePersistence(DynamicPersistence):
    """Reuse M4 recurrence/context while excluding every local-derived input."""

    def __init__(self, original, policy_dim, global_dim, local_feature_dim):
        nn.Module.__init__(self)
        self.local_feature_dim = local_feature_dim
        self.context_dim = original.context_dim - 32
        quality_start = 64 + 2 * global_dim
        columns = list(range(quality_start)) + list(range(quality_start + 32, original.context_dim))
        self.rho_net = original.rho_net
        self.rho_net[0] = ActiveColumnsLinear(self.rho_net[0], columns)

    def forward(self, value, valid, cycle_ids, global_features, global_mask, local_mask,
                local_features, local_feature_mask, policy, policy_mask, return_diagnostics=False):
        shape = value.shape[:2]
        absent_residual = torch.zeros(*shape, 64, dtype=torch.bool, device=value.device)
        absent_features = value.new_zeros(*shape, 8, self.local_feature_dim)
        absent_feature_mask = torch.zeros_like(absent_features, dtype=torch.bool)
        return super().forward(value, valid, cycle_ids, global_features, global_mask, absent_residual,
                               absent_features, absent_feature_mask, policy, policy_mask, return_diagnostics)


class MethodologyAblationModel(MethodologyFullModel):
    """Construct each registered arm directly from the same complete M4 model."""

    def __init__(self, arm, policy_dim, global_dim, local_feature_dim, seed=42, dropout=.1,
                 observation_bound=.03, soh_bounds=(0.0, 1.2)):
        if arm not in ABLATION_ARMS:
            raise ValueError("Unknown independent M4 ablation arm")
        super().__init__(policy_dim, global_dim, local_feature_dim, seed, dropout, observation_bound, soh_bounds)
        self.arm = arm
        if arm == "A1_M4_LEGACY_FLOW":
            self.flow_head = initialize(seed, "legacy_head", LegacyFlowWithSignedRate)
        elif arm == "A2_M4_NO_B_WRITE":
            for block in self.history:
                block.mixer = NoWriteMixer(block.mixer)
        elif arm == "A3_M4_STATIC_RHO":
            self.persistence.rho_net = StaticRetentionLogits()
        elif arm == "A4_M4_NO_LOCAL_PHYSICS":
            self.physics = GlobalPhysicsEncoder(self.physics)
            self.persistence = GlobalQualityFreePersistence(self.persistence, policy_dim, global_dim, local_feature_dim)
            for block in self.history:
                block.mixer.write_gate = ActiveColumnsLinear(block.mixer.write_gate, range(64))

    def forward(self, raw_viq, point_mask, cycle_ids, reference_viq, policy, policy_mask,
                global_features, global_mask, local_residual, local_mask, local_features,
                local_feature_mask, anchor_soh100, query_cycles, return_context=False,
                return_diagnostics=False):
        output = super().forward(raw_viq, point_mask, cycle_ids, reference_viq, policy, policy_mask,
                                 global_features, global_mask, local_residual, local_mask, local_features,
                                 local_feature_mask, anchor_soh100, query_cycles, return_context, return_diagnostics)
        if return_diagnostics:
            output["diagnostics"]["flow"]["structure"] = (
                "analytic_decoder_difference_with_detached_signed_derivative"
                if self.arm == "A1_M4_LEGACY_FLOW" else "nonnegative_rate_trapezoid_integral")
        return output


def model_registry(model):
    registry = full_registry(model)
    legacy = model.arm == "A1_M4_LEGACY_FLOW"
    no_write = model.arm == "A2_M4_NO_B_WRITE"
    static = model.arm == "A3_M4_STATIC_RHO"
    no_local = model.arm == "A4_M4_NO_LOCAL_PHYSICS"
    registry.update(
        initial_state_sha256=state_hash(model), independent_parent="M4_FULL_METHOD",
        active_parameters=sum(parameter.numel() for parameter in model.parameters()),
        parameter_matching="actual active architecture counts; no artificial budget matching",
        flow_head_parameters=sum(parameter.numel() for parameter in model.flow_head.parameters()),
        dynamic_retention=not static, rho_hidden_width=0 if static else 64,
        rho_context_dim=0 if static else model.persistence.context_dim,
        rho_compatibility_input_layout_dim=64 + 2 * model.global_dim + 32 + 2 * model.policy_dim + 6,
        static_rho_channels=64 if static else 0, static_rho_initial=.9 if static else None,
        actual_postconvolution_history_B_modulation=not no_write,
        history_write_gate_input_dim=0 if no_write else (64 if no_local else 96),
        history_gate_compatibility_input_layout_dim=0 if no_write else 96,
        write_projection_zero_initialized=not no_write,
        local_physics_removed=no_local, local_quality_dim=0 if no_local else 32,
        diagnostic_quality_shape=32,
        input_ablation="remove_local_physics_and_its_quality_only" if no_local else "complete_M4_inputs",
        raw_and_reference_VIQ_retained=True, local_self_reference_Mamba_retained=True,
        rate_head_unit="signed SOH per1000cycles" if legacy else "SOH per1000cycles",
        rate_per_cycle_unit="signed SOH percycle" if legacy else "SOH percycle",
        rate_integration="none; signed decoder difference in integrated_loss compatibility field" if legacy else registry["rate_integration"],
        instantaneous_rate_diagnostic_only=legacy, nonnegative_rate_structure=not legacy,
        signed_rate_derivative="d(anchor-raw_trend)/d((cycle-100)/1000)" if legacy else None,
        anchor_rate_semantics="analytic right derivative; exact tau==0 output branch has zero autograd derivative" if legacy else None,
        rate_auxiliary="unchanged observed-trajectory endpoint difference",
        removed_quality_columns="rho[104:136], gate[64:96] for P15/G20" if no_local else None,
    )
    return registry
