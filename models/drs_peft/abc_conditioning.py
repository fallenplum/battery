"""Matched frozen and jointly optimized Bridge/Head conditioning arms."""

from __future__ import annotations

from .e123_conditioning import ConditionedBatteryTSFM


class FactorialConditionedBatteryTSFM(ConditionedBatteryTSFM):
    """Change Bridge/Head trainability while preserving the E1--E3 operator."""

    def __init__(
        self,
        base_model,
        mode,
        radius_multiplier,
        train_bridge_head=False,
        reference_weight_fraction=0.01,
    ):
        requested_mode = mode
        if mode == "shuffled_dual":
            mode = "dual"
        super().__init__(
            base_model,
            mode,
            radius_multiplier,
            reference_weight_fraction=reference_weight_fraction,
        )
        self.requested_mode = requested_mode
        self.train_bridge_head = bool(train_bridge_head)
        for name, parameter in self.named_parameters():
            if not name.startswith("bridge.backbone.module."):
                parameter.requires_grad_(self.train_bridge_head)
        adapter_count = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if name.startswith("bridge.backbone.module.") and parameter.requires_grad
        )
        if adapter_count != 1440:
            raise RuntimeError("The factorial adapter/gate budget must equal 1440")

    def parameter_report(self):
        report = super().parameter_report()
        common = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if not name.startswith("bridge.backbone.module.")
            and parameter.requires_grad
        )
        report.update(
            bridge_head_regime="joint" if self.train_bridge_head else "frozen",
            bridge_head_trainable_parameters=common,
            source_continuation_trainable_parameters=1440 + common,
            conditional_delta_parameters=192,
            conditional_gate_parameters=1248,
            conditioning_mode=self.requested_mode,
        )
        return report
