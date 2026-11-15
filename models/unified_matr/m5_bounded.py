"""M5: bounded, nonincreasing latent SOH with an optional soft transition.

One registered change from M4 is its trajectory generator. No certified knee
or chemical failure-mode labels are assumed. Must use M4 frozen prefix inputs.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from .methodology_full import MethodologyFullModel, model_registry as m4_registry
from .methodology_operators import deterministic_cumsum
from .model import initialize, state_hash


class BoundedRelativeHazardHead(nn.Module):
    """Low-dimensional positive bounded hazard, per 1000 physical cycles.

    H(t)=integral_0^((t-100)/1000) lambda(s) ds. The caller maps this to
    SOH_trend(t)=anchor_soh100*exp(-H(t)). H is NOT absolute capacity loss.
    The soft center is a descriptive transition parameter, not an observed knee.
    """
    def __init__(self, lambda_max=2.0, init_hazard=0.1):
        super().__init__()
        if not 0 < init_hazard < lambda_max:
            raise ValueError("Fixed hazard bounds must be positive and ordered")
        self.net = nn.Sequential(nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 6))
        nn.init.normal_(self.net[-1].weight, mean=0, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("baseline_logit", torch.tensor(math.log(init_hazard / (lambda_max - init_hazard))))
        self.lambda_max = float(lambda_max)

    def forward(self, state, condition, query_cycles):
        if state.ndim != 2 or state.shape[-1] != 16 or condition.shape != state.shape:
            raise ValueError("Expected state and condition [B,16]")
        if query_cycles.ndim != 2 or query_cycles.shape[0] != len(state):
            raise ValueError("Expected queries [B,Q]")
        if not bool(torch.isfinite(query_cycles).all()) or bool((query_cycles < 100).any()):
            raise ValueError("Queries must be finite physical cycles >=100")
        end = max(101, int(torch.ceil(query_cycles.max()).detach().item()))
        grid = torch.arange(100, end + 1, device=state.device, dtype=state.dtype)
        tau = (grid - 100) / 1000.0
        params = self.net(torch.cat((state, condition), -1))
        a, slope, bend, transition, kraw, wraw = params.unbind(-1)
        kappa = 0.1 + 3.5 * torch.sigmoid(kraw)
        width = 0.05 + 0.8 * torch.sigmoid(wraw)
        tau2 = tau[None, :]
        t2 = torch.tanh(tau2 / 2.0)
        local_gate = torch.sigmoid((tau2 - kappa[:, None]) / width[:, None]) \
                     - torch.sigmoid((-kappa[:, None]) / width[:, None])
        logits = (self.baseline_logit + 2.0*torch.tanh(a)[:,None]
                  + 1.5*torch.tanh(slope)[:,None]*t2
                  + torch.tanh(bend)[:,None]*t2.square()
                  + 2.0*torch.tanh(transition)[:,None]*local_gate)
        intensity = self.lambda_max * torch.sigmoid(logits)
        steps = (intensity[:, 1:] + intensity[:, :-1]) * 0.0005
        hazard_grid = torch.cat((intensity.new_zeros(len(state),1), deterministic_cumsum(steps,1)),dim=-1)
        x = (query_cycles - 100).clamp_min(0)
        left = x.floor().long().clamp_max(len(grid)-2)
        frac = x - left.to(x.dtype)
        H = hazard_grid.gather(1,left) + frac*(hazard_grid.gather(1,left+1)-hazard_grid.gather(1,left))
        lam = intensity.gather(1,left) + frac*(intensity.gather(1,left+1)-intensity.gather(1,left))
        return H,lam


    def soft_transition_parameters(self, state, condition):
        """Only statistical model coordinates; not physically verified knee events."""
        p = self.net(torch.cat((state, condition), -1))
        center = 100.0 + 1000.0 * (0.1 + 3.5 * torch.sigmoid(p[:, 4]))
        width = 1000.0 * (0.05 + 0.8 * torch.sigmoid(p[:, 5]))
        amplitude = 2.0 * torch.tanh(p[:, 3])
        return {"soft_transition_center_cycle": center,
                "soft_transition_width_cycles": width,
                "soft_transition_signed_amplitude": amplitude}


class M5BoundedModel(MethodologyFullModel):
    """Reuse all M4 early-cycle routes, transform the latent hazard at output."""
    def __init__(self, policy_dim, global_dim, local_feature_dim, seed=42, dropout=.1,
                 observation_bound=.03, soh_bounds=(0., 1.2)):
        super().__init__(policy_dim, global_dim, local_feature_dim, seed=seed,
                         dropout=dropout, observation_bound=observation_bound, soh_bounds=soh_bounds)
        self.arm = "M5_BOUNDED_TRANSITION"
        self.flow_head = initialize(seed, "m5_bounded_relative_hazard", BoundedRelativeHazardHead)

    def forward(self, raw_viq, point_mask, cycle_ids, reference_viq, policy, policy_mask,
                global_features, global_mask, local_residual, local_mask, local_features,
                local_feature_mask, anchor_soh100, query_cycles, return_context=False,
                return_diagnostics=False):
        if not bool(torch.isfinite(anchor_soh100).all() and (anchor_soh100 >= 0).all()):
            raise ValueError("The bounded nonnegative trend requires nonnegative finite prefix SOH")
        result = super().forward(raw_viq, point_mask, cycle_ids, reference_viq, policy, policy_mask,
                                 global_features, global_mask, local_residual, local_mask,
                                 local_features, local_feature_mask, anchor_soh100, query_cycles,
                                 return_context=return_context, return_diagnostics=return_diagnostics)
        # super.forward provisionally treats H as absolute loss. Replace all
        # affected returned tensors BEFORE loss evaluation.
        H = result["integrated_loss"]
        rel_hazard = result["rate_per_1000_cycles"]
        trend = anchor_soh100[:,None] * torch.exp(-H)
        loss_absolute = anchor_soh100[:,None] - trend
        absolute_rate_per_1000 = trend * rel_hazard
        result.update(trajectory=(trend+result["observation_residual"]).clamp(*self.soh_bounds),
                      trend=trend.clamp(*self.soh_bounds), raw_trend=trend,
                      integrated_loss=loss_absolute,
                      rate_per_1000_cycles=absolute_rate_per_1000,
                      rate_per_cycle=absolute_rate_per_1000/1000.0)
        if return_context:
            result["context"]["raw_trend"] = trend
            result["context"]["relative_hazard_integral"] = H
        if return_diagnostics:
            result["diagnostics"]["flow"].update(
                raw_trend=trend, integrated_loss=loss_absolute,
                rate_per_1000_cycles=absolute_rate_per_1000,
                rate_per_cycle=absolute_rate_per_1000/1000.0,
                relative_hazard_per_1000=rel_hazard,
                relative_hazard_integral=H,
                structure="positive_bounded_relative_hazard_exponential_trend",
                transition_parameter_semantics="unvalidated_soft_transition_not_knee_label")
            result["diagnostics"]["flow"].update(self.flow_head.soft_transition_parameters(
                result["diagnostics"]["state"], result["diagnostics"]["condition"]))
        return result


def model_registry(model):
    d=m4_registry(model)
    d.update(arm="M5_BOUNDED_TRANSITION",
             parameters=sum(p.numel() for p in model.parameters()),
             trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
             initial_state_sha256=state_hash(model),
             rate_integration="fixed_physical_cycle_trapezoid_for_bounded_relative_hazard_then_exponential",
             hazard_bound_per_1000_cycles=[0.0,model.flow_head.lambda_max],
             trend_formula="SOH100*exp(-integrated_relative_hazard)",
             transition_label="not_a_certified_knee_event")
    return d
