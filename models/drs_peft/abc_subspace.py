"""Fixed-start subspace probes with explicit physical and effective budgets."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .peft import SubspaceLinear
from .trajectory import BatteryTSFM

ARMS = (
    "fixed_rank4",
    "rank4_plus_small",
    "matched_fixed",
    "matched_residual",
    "matched_lora",
)
MATCHED_RANKS = (25, 28, 28, 28, 28, 29)
MATCHED_PERP_SHAPES = ((21, 29), (24, 32), (24, 32), (24, 32), (24, 32), (25, 33))
EXPECTED_PARAMETERS = {
    "fixed_rank4": 192,
    "rank4_plus_small": 300,
    "matched_fixed": 9204,
    "matched_residual": 9204,
    "matched_lora": 9204,
}


def _orthonormal(matrix):
    gram = matrix.detach().double().T @ matrix.detach().double()
    return torch.allclose(
        gram,
        torch.eye(len(gram), dtype=gram.dtype, device=gram.device),
        atol=2e-5,
        rtol=2e-5,
    )


class SubspaceProbeLinear(SubspaceLinear):
    """Retain a frozen rank-four source function and fit a bounded increment."""

    def __init__(self, adapter, arm, bases, reference_fraction=0.01, radius=1.0):
        if not isinstance(adapter, SubspaceLinear) or adapter.rank != 4:
            raise TypeError("C probes require a rank-four static source adapter")
        if bool(adapter.target_active) or adapter.delta_core is not None:
            raise ValueError("The source adapter must be frozen before continuation")
        if arm not in ARMS or not adapter.orthogonal_bases:
            raise ValueError("Unknown arm or nonorthogonal source bases")
        if not math.isfinite(reference_fraction) or reference_fraction <= 0:
            raise ValueError("The W0 norm reference fraction must be positive")
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("The increment radius multiplier must be positive")
        super().__init__(
            adapter.base,
            adapter.U,
            adapter.V,
            adapter.core,
            alpha=adapter.alpha,
            dropout=0.0,
        )
        self.core.requires_grad_(False)
        self.source_core = self.core.detach().clone()
        self.arm = arm
        self.evaluation_mode = "trained_ball"
        self.register_buffer(
            "core_radius",
            torch.tensor(
                float(
                    radius
                    * reference_fraction
                    * adapter.base.weight.detach().double().norm()
                ),
                dtype=torch.float64,
                device=self.core.device,
            ),
        )
        self.register_buffer(
            "core_norm_reference", self.core_radius.detach().clone() / radius
        )
        self.register_buffer(
            "core_radius_multiplier",
            torch.tensor(float(radius), dtype=torch.float64, device=self.core.device),
        )
        self.register_parameter("core_perp", None)
        self.register_parameter("core_a_free", None)
        self.register_buffer("core_anchor", None)

        if arm == "matched_lora":
            initial = bases["lora_a"].to(self.core).flatten()
            if initial.shape != (self.base.in_features,) or not bool(
                torch.isfinite(initial).all()
            ):
                raise ValueError(
                    "LoRA initialization must be one finite fit-only direction"
                )
            anchor = int(initial.abs().argmax())
            if initial[anchor].abs() == 0:
                raise ValueError("A nonzero fit-only direction is required")
            initial = initial / initial[anchor]
            indices = torch.tensor(
                [index for index in range(len(initial)) if index != anchor],
                device=self.core.device,
            )
            self.register_buffer("core_a_indices", indices)
            self.core_anchor = torch.tensor(anchor, device=self.core.device)
            self.core_a_free = nn.Parameter(initial[indices].detach().clone())
            self.delta_core = nn.Parameter(
                torch.zeros(
                    self.base.out_features,
                    dtype=self.core.dtype,
                    device=self.core.device,
                )
            )
            self.register_buffer("core_increment_u", None)
            self.register_buffer("core_increment_v", None)
            self.register_buffer("core_left_factor", None)
            self.register_buffer("core_right_factor", None)
            return

        if arm == "matched_fixed":
            left, right = bases["fixed_u"].to(self.core), bases["fixed_v"].to(self.core)
            self.delta_core = nn.Parameter(
                torch.zeros(
                    left.shape[1],
                    right.shape[1],
                    dtype=self.core.dtype,
                    device=self.core.device,
                )
            )
        else:
            self.delta_core = nn.Parameter(torch.zeros_like(self.core))
            left, right = self.U, self.V
            if arm in ("rank4_plus_small", "matched_residual"):
                perp_left, perp_right = (
                    bases["perp_u"].to(self.core),
                    bases["perp_v"].to(self.core),
                )
                if arm == "rank4_plus_small":
                    perp_left, perp_right = perp_left[:, :3], perp_right[:, :3]
                left, right = (
                    torch.cat((left, perp_left), dim=1),
                    torch.cat((right, perp_right), dim=1),
                )
                self.core_perp = nn.Parameter(
                    torch.zeros(
                        perp_left.shape[1],
                        perp_right.shape[1],
                        dtype=self.core.dtype,
                        device=self.core.device,
                    )
                )
        if (
            left.shape[0] != self.base.out_features
            or right.shape[0] != self.base.in_features
        ):
            raise ValueError(
                "The increment bases differ from the physical weight shape"
            )
        if not _orthonormal(left) or not _orthonormal(right):
            raise ValueError("Increment coordinates must be column-orthonormal")
        self.register_buffer("core_increment_u", left.detach().clone())
        self.register_buffer("core_increment_v", right.detach().clone())
        left_factor = torch.linalg.qr(left.detach().cpu().double(), mode="reduced").R
        right_factor = torch.linalg.qr(right.detach().cpu().double(), mode="reduced").R
        self.register_buffer("core_left_factor", left_factor.to(self.core.device))
        self.register_buffer("core_right_factor", right_factor.to(self.core.device))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        nn.Module._load_from_state_dict(self, state_dict, prefix, *args, **kwargs)

    def lora_a(self):
        result = torch.ones(
            self.base.in_features, dtype=self.core.dtype, device=self.core.device
        )
        return result.scatter(0, self.core_a_indices, self.core_a_free)

    def coefficient_matrix(self):
        if self.core_perp is None:
            return self.delta_core
        return torch.block_diag(self.delta_core, self.core_perp)

    def _raw_norm(self):
        if self.arm == "matched_lora":
            return (
                self.scale
                * self.delta_core.double().norm()
                * self.lora_a().double().norm()
            )
        coefficients = self.coefficient_matrix().double()
        return (
            self.scale
            * (self.core_left_factor @ coefficients @ self.core_right_factor.T).norm()
        )

    def _scale_factor(self, raw_norm):
        if self.evaluation_mode == "matched_norm":
            if self.training:
                raise RuntimeError("Matched-norm evaluation requires model.eval()")
            nonzero = raw_norm > 0
            denominator = torch.where(nonzero, raw_norm, torch.ones_like(raw_norm))
            return torch.where(
                nonzero, self.core_radius / denominator, torch.zeros_like(raw_norm)
            )
        if self.evaluation_mode != "trained_ball":
            raise ValueError("Unknown increment evaluation mode")
        outside = raw_norm > self.core_radius
        denominator = torch.where(outside, raw_norm, torch.ones_like(raw_norm))
        return torch.where(
            outside, self.core_radius / denominator, torch.ones_like(raw_norm)
        )

    def increment_matrix(self):
        if self.arm == "matched_lora":
            raw = self.delta_core.double()[:, None] * self.lora_a().double()[None, :]
        else:
            raw = (
                self.core_increment_u.double()
                @ self.coefficient_matrix().double()
                @ self.core_increment_v.double().T
            )
        return self.scale * raw * self._scale_factor(self._raw_norm())

    def forward(self, x, abs_gate=None, drift_gate=None):
        if abs_gate is not None or drift_gate is not None:
            raise ValueError("C probes do not use conditional gates")
        baseline = F.linear(x @ self.V, self.source_core) @ self.U.T
        factor = self._scale_factor(self._raw_norm()).to(self.core)
        if self.arm == "matched_lora":
            change = F.linear(x, self.lora_a()[None]) * self.delta_core * factor
        else:
            change = (
                F.linear(x @ self.core_increment_v, self.coefficient_matrix() * factor)
                @ self.core_increment_u.T
            )
        return self.base(x) + self.scale * (baseline + change)

    @torch.no_grad()
    def update_diagnostics(self, batch_size):
        raw_norm = self._raw_norm()
        factor = self._scale_factor(raw_norm)
        change = self.increment_matrix()
        baseline = (
            self.scale * self.U.double() @ self.source_core.double() @ self.V.double().T
        )
        values = {
            "raw_increment_norm": raw_norm,
            "increment_norm": change.norm(),
            "radius": self.core_radius,
            "radius_multiplier": self.core_radius_multiplier,
            "norm_reference": self.core_norm_reference,
            "baseline_update_norm": baseline.norm(),
            "total_update_norm": (baseline + change).norm(),
            "cap_active": raw_norm > self.core_radius,
            "zero_increment": factor * raw_norm == 0,
            "matched_norm_unattainable": raw_norm == 0,
        }
        return {name: value.expand(batch_size) for name, value in values.items()}


class SubspaceProbeTSFM(BatteryTSFM):
    """Freeze the entire shared source function before fitting one C increment."""

    input_keys = ("x", "point_mask", "cycle_ids", "policy", "policy_mask", "anchor")

    def __init__(self, static_model, arm, bases, reference_fraction=0.01, radius=1.0):
        nn.Module.__init__(self)
        if arm not in ARMS:
            raise ValueError("Unknown subspace probe")
        self.bridge, self.head = static_model.bridge, static_model.head
        self.bridge_kind = static_model.bridge_kind
        self.official_backbone_identity = static_model.official_backbone_identity
        self.arm = arm
        self.requires_grad_(False)
        layers = [
            (name, layer)
            for name, layer in self.backbone.named_modules()
            if isinstance(layer, SubspaceLinear)
        ]
        if len(layers) != 12 or set(bases) != {name for name, _ in layers}:
            raise ValueError("All twelve Q/V matrices need fit-only increment bases")
        for name, layer in layers:
            parent_name, _, attribute = name.rpartition(".")
            setattr(
                self.backbone.get_submodule(parent_name),
                attribute,
                SubspaceProbeLinear(
                    layer, arm, bases[name], reference_fraction, radius
                ),
            )
        actual = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        if actual != EXPECTED_PARAMETERS[arm]:
            raise RuntimeError(f"C effective parameter budget differs: {actual}")

    def set_evaluation_mode(self, mode):
        if mode not in ("trained_ball", "matched_norm"):
            raise ValueError("Unknown increment evaluation mode")
        for layer in self.backbone.modules():
            if isinstance(layer, SubspaceProbeLinear):
                layer.evaluation_mode = mode
        return self

    @torch.no_grad()
    def update_diagnostics(self, early):
        return {
            name: layer.update_diagnostics(len(early["x"]))
            for name, layer in self.backbone.named_modules()
            if isinstance(layer, SubspaceProbeLinear)
        }

    def parameter_report(self):
        report = super().parameter_report()
        report.update(
            subspace_probe=self.arm,
            source_continuation_trainable_parameters=EXPECTED_PARAMETERS[self.arm],
            fixed_source_rank=4,
            update_control="physical_weight_Frobenius_increment_ball_around_frozen_rank4_C0",
            gauge_fixed_rank1=self.arm == "matched_lora",
        )
        if self.arm == "matched_lora":
            report.update(
                generic_rank1_weight_manifold_dimension=9204,
                initial_zero_B_weight_jacobian_rank=4608,
                initial_A_weight_jacobian_rank=0,
                gauge_chart="largest_fit_v1_coordinate_fixed_to_one",
                gauge_chart_scope="rank1_factors_with_nonzero_anchor_coordinate",
            )
        return report
