"""Fixed-start, weight-space-bounded conditioning for source-group E1--E3."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .diagnostic_dualview import INPUT_KEYS, CoreGates
from .peft import _GATES, SubspaceLinear, peft_context
from .trajectory import BatteryTSFM

MODES = ("constant", "absolute", "relative", "dual")
EVALUATION_MODES = ("trained_ball", "matched_norm")


class BoundedConditionedLinear(SubspaceLinear):
    """Keep a frozen source core and bound its full gated weight increment."""

    def __init__(
        self,
        adapter,
        radius_multiplier,
        source_norm_reference=None,
        reference_weight_fraction=0.01,
    ):
        if not isinstance(adapter, SubspaceLinear):
            raise TypeError("A fitted static SubspaceLinear adapter is required")
        if bool(adapter.target_active) or adapter.delta_core is not None:
            raise ValueError("The E1--E3 source start must be a static adapter")
        multiplier = float(radius_multiplier)
        fraction = float(reference_weight_fraction)
        if not math.isfinite(multiplier) or multiplier < 0:
            raise ValueError("The radius multiplier must be finite and nonnegative")
        if not math.isfinite(fraction) or fraction <= 0:
            raise ValueError(
                "The reference weight fraction must be finite and positive"
            )
        super().__init__(
            adapter.base,
            adapter.U,
            adapter.V,
            core=adapter.core,
            alpha=adapter.alpha,
            dropout=0.0,
            method=adapter.method,
            require_orthogonal=adapter.orthogonal_bases,
        )
        self.core.requires_grad_(False)
        self.source_core = self.core.detach().clone()
        self.delta_core = nn.Parameter(torch.zeros_like(self.core))
        left = torch.linalg.qr(adapter.U.detach().cpu().double(), mode="reduced").R
        right = torch.linalg.qr(adapter.V.detach().cpu().double(), mode="reduced").R
        self.register_buffer("core_left_factor", left.to(device=self.core.device))
        self.register_buffer("core_right_factor", right.to(device=self.core.device))
        reference = (
            fraction * float(adapter.base.weight.detach().double().norm().cpu())
            if source_norm_reference is None
            else float(source_norm_reference)
        )
        if not math.isfinite(reference) or reference <= 0:
            raise ValueError(
                "The common physical-weight norm reference must be positive"
            )
        self.register_buffer(
            "core_norm_reference",
            torch.tensor(reference, dtype=torch.float64, device=self.core.device),
        )
        self.register_buffer(
            "core_radius_multiplier",
            torch.tensor(multiplier, dtype=torch.float64, device=self.core.device),
        )
        if hasattr(adapter, "singular_values"):
            self.register_buffer(
                "singular_values", adapter.singular_values.detach().clone()
            )
        self.evaluation_mode = "trained_ball"

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # All source-continuation state exists at construction. Bypass the
        # inherited target-start hook, which can replace optimizer Parameters.
        nn.Module._load_from_state_dict(self, state_dict, prefix, *args, **kwargs)

    def _physical_norm(self, core):
        compact = (
            self.core_left_factor.double()
            @ core.double()
            @ self.core_right_factor.double().T
        )
        return self.scale * torch.linalg.matrix_norm(compact, ord="fro", dim=(-2, -1))

    def conditioned_increment(
        self,
        batch_size,
        absolute_gate=None,
        relative_gate=None,
        collect_diagnostics=True,
    ):
        proposal = (self.source_core + self.delta_core)[None].expand(batch_size, -1, -1)
        for gate in (absolute_gate, relative_gate):
            if gate is not None and gate.shape != (batch_size, self.rank):
                raise ValueError(
                    "Each conditional gate must be [physical_cell_batch,rank]"
                )
        if absolute_gate is not None:
            proposal = proposal * absolute_gate[:, None, :]
        if relative_gate is not None:
            proposal = proposal * relative_gate[:, :, None]
        raw_increment = proposal.double() - self.source_core.double()[None]
        raw_norm = self._physical_norm(raw_increment)
        radius = self.core_radius_multiplier * self.core_norm_reference
        nonzero = raw_norm > 0
        if self.evaluation_mode == "matched_norm":
            if self.training:
                raise RuntimeError("Matched-norm evaluation requires model.eval()")
            denominator = torch.where(nonzero, raw_norm, torch.ones_like(raw_norm))
            factor = torch.where(nonzero, radius / denominator, 0.0)
        elif self.evaluation_mode == "trained_ball":
            outside = raw_norm > radius
            denominator = torch.where(outside, raw_norm, torch.ones_like(raw_norm))
            factor = torch.where(outside, radius / denominator, 1.0)
            factor = torch.where(radius > 0, factor, 0.0)
        else:
            raise ValueError("Unknown physical update evaluation mode")
        increment = (raw_increment * factor[:, None, None]).to(self.core)
        if not collect_diagnostics:
            return increment, {}
        actual_norm = self._physical_norm(increment)
        diagnostics = {
            "raw_increment_norm": raw_norm,
            "increment_norm": actual_norm,
            "radius": radius.expand_as(raw_norm),
            "radius_multiplier": self.core_radius_multiplier.expand_as(raw_norm),
            "norm_reference": self.core_norm_reference.expand_as(raw_norm),
            "baseline_update_norm": self._physical_norm(self.source_core).expand_as(
                raw_norm
            ),
            "total_update_norm": self._physical_norm(
                self.source_core.double()[None] + increment
            ),
            "cap_active": raw_norm > radius,
            "zero_increment": actual_norm == 0,
            "zero_proposed_increment": ~nonzero,
            "matched_norm_unattainable": ~nonzero & (radius > 0),
        }
        return increment, diagnostics

    def forward(self, x, abs_gate=None, drift_gate=None):
        if x.ndim < 2:
            raise ValueError("The leading dimension must identify physical cells")
        contextual = _GATES.get().get(id(self), (None, None))
        absolute = contextual[0] if abs_gate is None else abs_gate
        relative = contextual[1] if drift_gate is None else drift_gate
        increment, _ = self.conditioned_increment(
            len(x), absolute, relative, collect_diagnostics=False
        )
        rank_input = x @ self.V
        baseline = F.linear(rank_input, self.source_core)
        flattened = rank_input.reshape(len(x), -1, self.rank)
        change = torch.bmm(flattened, increment.transpose(1, 2)).reshape_as(rank_input)
        return self.base(x) + self.scale * ((baseline + change) @ self.U.T)


class ConditionedBatteryTSFM(BatteryTSFM):
    """Freeze Bridge/Head and source cores, then fit matched conditional deltas."""

    input_keys = INPUT_KEYS

    def __init__(
        self,
        base_model,
        mode,
        radius_multiplier,
        source_norm_reference=None,
        reference_weight_fraction=0.01,
    ):
        nn.Module.__init__(self)
        if mode not in MODES:
            raise ValueError("Unknown E1--E3 conditioning mode")
        self.bridge = base_model.bridge
        self.head = base_model.head
        self.bridge_kind = base_model.bridge_kind
        self.official_backbone_identity = base_model.official_backbone_identity
        self.mode = mode
        self.evaluation_mode = "trained_ball"
        self.requires_grad_(False)
        selected = [
            (name, layer)
            for name, layer in self.backbone.named_modules()
            if isinstance(layer, SubspaceLinear)
        ]
        if len(selected) != 12 or any(layer.rank != 4 for _, layer in selected):
            raise ValueError(
                "The source experiment requires twelve rank-four Q/V adapters"
            )
        if hasattr(self.backbone, "core_gates"):
            raise ValueError(
                "The static source model already contains conditional gates"
            )
        for name, layer in selected:
            parent_name, _, attribute = name.rpartition(".")
            parent = self.backbone.get_submodule(parent_name)
            reference = (
                None if source_norm_reference is None else source_norm_reference[name]
            )
            setattr(
                parent,
                attribute,
                BoundedConditionedLinear(
                    layer, radius_multiplier, reference, reference_weight_fraction
                ),
            )
        self.backbone.add_module(
            "core_gates", CoreGates([name for name, _ in selected])
        )
        if (
            sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            )
            != 1440
        ):
            raise RuntimeError("The conditional delta and gate budget must equal 1440")

    def _gates(self, early):
        absolute, relative = early["diagnostic_absolute"], early["diagnostic_relative"]
        if self.mode in ("constant", "relative"):
            absolute = torch.ones_like(absolute)
        if self.mode in ("constant", "absolute"):
            relative = torch.ones_like(relative)
        return self.backbone.core_gates(absolute, relative)

    def set_evaluation_mode(self, name):
        if name not in EVALUATION_MODES:
            raise ValueError("Unknown E1--E3 evaluation mode")
        self.evaluation_mode = name
        for layer in self.backbone.modules():
            if isinstance(layer, BoundedConditionedLinear):
                layer.evaluation_mode = name
        return self

    def forward(self, early, query):
        absolute, relative = self._gates(early)
        with peft_context(self.backbone, absolute, relative):
            return super().forward(early, query)

    @torch.no_grad()
    def update_diagnostics(self, early):
        absolute, relative = self._gates(early)
        return {
            name: layer.conditioned_increment(
                len(early["diagnostic_absolute"]), absolute[name], relative[name]
            )[1]
            for name, layer in self.backbone.named_modules()
            if isinstance(layer, BoundedConditionedLinear)
        }

    def parameter_report(self):
        report = super().parameter_report()
        report.update(
            source_continuation_trainable_parameters=1440,
            conditional_delta_parameters=192,
            conditional_gate_parameters=1248,
            conditioning_mode=self.mode,
            update_control="per-cell per-layer physical-weight Frobenius increment ball around fixed C0",
            evaluation_mode=self.evaluation_mode,
        )
        return report
