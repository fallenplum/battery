"""Source-development diagnostic of input-conditioned, fixed-basis cores."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .peft import SubspaceLinear, peft_context
from .trajectory import BatteryTSFM

MODES = ("constant", "absolute", "relative", "dual", "shuffled_dual")
FEATURE_DIM = 12
INPUT_KEYS = (
    "x",
    "point_mask",
    "cycle_ids",
    "policy",
    "policy_mask",
    "anchor",
    "diagnostic_absolute",
    "diagnostic_relative",
)
ABSOLUTE_NAMES = tuple(
    f"{statistic}_{channel}"
    for statistic in ("mean", "std", "min", "max")
    for channel in ("voltage_V", "signed_current_C", "capacity_SOH")
)
RELATIVE_NAMES = tuple(
    f"{statistic}_{channel}"
    for statistic in (
        "mean_delta",
        "std_delta",
        "late_minus_early",
        "slope_per_100_cycles",
    )
    for channel in ("voltage_V", "signed_current_C", "capacity_SOH")
)


def prefix_view_descriptors(x, point_mask, cycle_ids):
    """Produce two measured-VIQ views using only registered cycles 1 through 100.

    The relative reference is the pointwise median of cycles 1 through 10,
    requiring five observations at each bin. The fourth input channel is ignored.
    """
    values = np.asarray(x, dtype=np.float64)[..., :3]
    mask = np.asarray(point_mask, dtype=bool)
    cycles = np.asarray(cycle_ids, dtype=np.float64)
    if values.ndim != 4 or values.shape[1:] != (100, 64, 3):
        raise ValueError("Diagnostic inputs must have shape [B,100,64,>=3]")
    if mask.shape != values.shape[:-1] or cycles.shape != values.shape[:2]:
        raise ValueError("Diagnostic prefix masks or coordinates have wrong shapes")
    supported_cycles = cycles[mask.any(-1)]
    if not np.isfinite(values[mask]).all() or not np.isfinite(supported_cycles).all():
        raise ValueError("A supported prefix value is nonfinite")
    if ((supported_cycles < 1) | (supported_cycles > 100)).any():
        raise ValueError("Gate descriptors admit only prefix cycle coordinates 1..100")

    count = mask.sum((1, 2))
    safe = np.where(mask[..., None], values, 0.0)
    mean = safe.sum((1, 2)) / np.maximum(count[:, None], 1)
    variance = np.where(mask[..., None], (values - mean[:, None, None]) ** 2, 0).sum(
        (1, 2)
    ) / np.maximum(count[:, None], 1)
    minimum = np.where(mask[..., None], values, np.inf).min((1, 2))
    maximum = np.where(mask[..., None], values, -np.inf).max((1, 2))
    absolute = np.concatenate((mean, np.sqrt(variance), minimum, maximum), 1)
    absolute_mask = np.repeat((count > 0)[:, None], FEATURE_DIM, axis=1)
    absolute = np.where(absolute_mask, absolute, 0)

    reference_mask = mask & ((cycles >= 1) & (cycles <= 10))[..., None]
    reference = np.zeros((len(values), 64, 3), dtype=np.float64)
    reference_count = reference_mask.sum(1)
    for row in range(len(values)):
        for point in np.flatnonzero(reference_count[row] >= 5):
            reference[row, point] = np.median(
                values[row, reference_mask[row, :, point], point], axis=0
            )
    relative_mask = mask & (reference_count[:, None] >= 5)
    relative = values - reference[:, None]
    relative_count = relative_mask.sum((1, 2))
    relative_mean = np.where(relative_mask[..., None], relative, 0).sum(
        (1, 2)
    ) / np.maximum(relative_count[:, None], 1)
    relative_variance = np.where(
        relative_mask[..., None], (relative - relative_mean[:, None, None]) ** 2, 0
    ).sum((1, 2)) / np.maximum(relative_count[:, None], 1)

    def period_mean(lower, upper):
        selected = relative_mask & ((cycles >= lower) & (cycles <= upper))[..., None]
        selected_count = selected.sum((1, 2))
        value = np.where(selected[..., None], relative, 0).sum((1, 2)) / np.maximum(
            selected_count[:, None], 1
        )
        return value, selected_count > 0

    early_mean, early_available = period_mean(1, 20)
    late_mean, late_available = period_mean(81, 100)
    cycle_count = relative_mask.sum(2)
    cycle_available = cycle_count > 0
    cycle_mean = np.where(relative_mask[..., None], relative, 0).sum(2) / np.maximum(
        cycle_count[..., None], 1
    )
    normalized_cycles = cycles / 100.0
    cycle_average = np.where(cycle_available, normalized_cycles, 0).sum(1) / np.maximum(
        cycle_available.sum(1), 1
    )
    centered_cycles = np.where(
        cycle_available, normalized_cycles - cycle_average[:, None], 0
    )
    denominator = (centered_cycles**2).sum(1)
    slope = (cycle_mean * centered_cycles[..., None]).sum(1) / np.maximum(
        denominator[:, None], 1e-12
    )
    relative_features = np.concatenate(
        (relative_mean, np.sqrt(relative_variance), late_mean - early_mean, slope), 1
    )
    relative_feature_mask = np.concatenate(
        (
            np.repeat((relative_count > 0)[:, None], 6, axis=1),
            np.repeat((early_available & late_available)[:, None], 3, axis=1),
            np.repeat((denominator > 0)[:, None], 3, axis=1),
        ),
        axis=1,
    )
    relative_features = np.where(relative_feature_mask, relative_features, 0)
    return {
        "absolute": absolute.astype(np.float32),
        "absolute_mask": absolute_mask,
        "relative": relative_features.astype(np.float32),
        "relative_mask": relative_feature_mask,
    }


class SourceViewScaler:
    """Fit gate standardization from source-training physical cells only."""

    def fit(self, descriptors, rows, ids):
        rows = np.asarray(rows, dtype=np.int64)
        if len(rows) != len(set(rows.tolist())) or len(rows) != len(ids):
            raise ValueError("Scaler source identities must be unique physical cells")
        self.source_ids = list(ids)
        self.statistics = {}
        for view in ("absolute", "relative"):
            values = descriptors[view][rows].astype(np.float64)
            masks = descriptors[view + "_mask"][rows]
            counts = masks.sum(0)
            mean = np.where(masks, values, 0).sum(0) / np.maximum(counts, 1)
            variance = np.where(masks, (values - mean) ** 2, 0).sum(0) / np.maximum(
                counts, 1
            )
            floor = 1e-3 if view == "absolute" else 1e-4
            self.statistics[view] = {
                "mean": mean.tolist(),
                "scale": np.maximum(np.sqrt(variance), floor).tolist(),
                "empirical_std": np.sqrt(variance).tolist(),
                "supported_cells": counts.tolist(),
                "physical_std_floor": floor,
            }
        return self

    def transform(self, descriptors):
        result = {}
        for view, statistics in self.statistics.items():
            standardized = (
                descriptors[view] - np.asarray(statistics["mean"])
            ) / np.asarray(statistics["scale"])
            result[view] = np.where(
                descriptors[view + "_mask"], standardized, 0
            ).astype(np.float32)
        return result

    def to_dict(self):
        return {
            "source_ids": self.source_ids,
            "feature_names": {
                "absolute": list(ABSOLUTE_NAMES),
                "relative": list(RELATIVE_NAMES),
            },
            "feature_units": {
                view: [unit for _ in range(4) for unit in ("V", "C-rate", "Q/Qnom")]
                for view in ("absolute", "relative")
            },
            "statistics": self.statistics,
        }


class CoreGates(nn.Module):
    """Two matched linear views per adapter, initialized to exact unit gates."""

    def __init__(self, names, rank=4):
        super().__init__()
        self.names = tuple(names)
        self.rank = int(rank)
        self.absolute = nn.ModuleList(nn.Linear(FEATURE_DIM, rank) for _ in self.names)
        self.relative = nn.ModuleList(nn.Linear(FEATURE_DIM, rank) for _ in self.names)
        for parameter in self.parameters():
            nn.init.zeros_(parameter)

    def forward(self, absolute, relative):
        abs_gates = {
            name: 1 + 0.5 * torch.tanh(layer(absolute))
            for name, layer in zip(self.names, self.absolute)
        }
        rel_gates = {
            name: 1 + 0.5 * torch.tanh(layer(relative))
            for name, layer in zip(self.names, self.relative)
        }
        return abs_gates, rel_gates


class DualViewDiagnosticTSFM(BatteryTSFM):
    """Keep bridge/head keys and apply gates within the same frozen U/V span."""

    input_keys = INPUT_KEYS

    def __init__(self, base_model, mode):
        nn.Module.__init__(self)
        if mode not in MODES:
            raise ValueError("Unknown diagnostic modulation mode")
        self.bridge = base_model.bridge
        self.head = base_model.head
        self.bridge_kind = base_model.bridge_kind
        self.official_backbone_identity = base_model.official_backbone_identity
        self.mode = mode
        selected = [
            (name, module)
            for name, module in self.backbone.named_modules()
            if isinstance(module, SubspaceLinear)
        ]
        if len(selected) != 12 or any(module.rank != 4 for _, module in selected):
            raise ValueError(
                "The diagnostic requires the frozen twelve rank-four Q/V adapters"
            )
        if hasattr(self.backbone, "core_gates"):
            raise ValueError("Core gates were already installed")
        # This exact prefix also preserves gate state in the frozen trainer's
        # adapter allowlist and assigns the registered adapter learning rate.
        self.backbone.add_module(
            "core_gates", CoreGates([name for name, _ in selected])
        )

    def forward(self, early, query):
        absolute = early["diagnostic_absolute"]
        relative = early["diagnostic_relative"]
        if self.mode in ("constant", "relative"):
            absolute = torch.ones_like(absolute)
        if self.mode in ("constant", "absolute"):
            relative = torch.ones_like(relative)
        abs_gates, rel_gates = self.backbone.core_gates(absolute, relative)
        with peft_context(self.backbone, abs_gates, rel_gates):
            return super().forward(early, query)

    def parameter_report(self):
        report = super().parameter_report()
        report["diagnostic_gate_parameters"] = sum(
            parameter.numel() for parameter in self.backbone.core_gates.parameters()
        )
        report["diagnostic_mode"] = self.mode
        report["fixed_basis_semantics"] = (
            "input-conditioned core scaling within the original U/V linear update space"
        )
        return report
