"""Common native Moirai bridge for prefix-only supervised SOH prediction."""

from __future__ import annotations

import torch
from torch import nn

from .bridges import NativeBridge, masked_mean, prefix_descriptors, temporal_pool
from .trajectory import CycleQueryHead


class SharedBatteryModel(nn.Module):
    """Use the same weight tensors for the 12- and 30-channel inputs."""

    def __init__(self, backbone_dir):
        super().__init__()
        self.bridge = NativeBridge(backbone_dir, policy_dim=0, pretrained=True)
        hidden = self.bridge.backbone.hidden_size
        self.fusion12 = nn.Sequential(
            nn.Linear(2 * hidden + 5 * 12 + 1, 64), nn.GELU(), nn.LayerNorm(64)
        )
        self.fusion30 = nn.Sequential(
            nn.Linear(2 * hidden + 5 * 30 + 1, 64), nn.GELU(), nn.LayerNorm(64)
        )
        self.head = CycleQueryHead()

    @property
    def backbone(self):
        return self.bridge.backbone.module

    def responses(self, features, feature_mask, cycles, reference_features=None):
        """Evaluate both views with the real view's packed normalization."""
        if (
            features.ndim != 3
            or features.shape[1] != 100
            or features.shape[2] not in (12, 30)
            or feature_mask.shape != features.shape
            or feature_mask.dtype != torch.bool
            or cycles.shape != features.shape[:2]
        ):
            raise ValueError("Expected paired 100-by-30 prefix channels")
        if (
            reference_features is not None
            and reference_features.shape != features.shape
        ):
            raise ValueError("Both prefix views must align")
        safe = torch.where(feature_mask, features, 0.0)
        real, valid = self.bridge.token_inputs(safe, feature_mask, cycles)
        backbone = self.bridge.backbone
        module = backbone.module
        with torch.no_grad():
            location, scale = module.scaler(
                real["target"],
                real["observed_mask"],
                real["sample_id"],
                real["variate_id"],
            )

        def encode(target):
            token = module.in_proj((target - location) / scale, real["patch_size"])
            return backbone.encode_hidden(
                token, real["sample_id"], real["time_id"], real["variate_id"]
            )

        factual = encode(real["target"])
        if reference_features is None:
            return factual, valid
        reference, ref_valid = self.bridge.token_inputs(
            torch.where(feature_mask, reference_features, 0.0), feature_mask, cycles
        )
        if not torch.equal(ref_valid, valid):
            raise RuntimeError("Paired response token coordinates changed")
        return factual, encode(reference["target"]), valid

    def context(self, features, feature_mask, cycles, anchor, channels=30):
        if channels not in (12, 30):
            raise ValueError("Only the registered channel ablation is supported")
        if channels == 12:
            features = features[..., :12]
            feature_mask = feature_mask[..., :12]
        cycles = cycles.masked_fill(~feature_mask.any(-1), 0)
        values = torch.where(feature_mask, features, 0.0)
        descriptors = prefix_descriptors(values, feature_mask, cycles)
        encoded, valid = self.responses(values, feature_mask, cycles)
        response = encoded.reshape(*valid.shape, -1)
        variable_mean, variable_last = temporal_pool(response, valid)
        observed_variates = valid.any(-1)[..., None].expand_as(variable_mean)
        mean = masked_mean(variable_mean, observed_variates, 1)
        last = masked_mean(variable_last, observed_variates, 1)
        return torch.cat((mean, last, descriptors, anchor[:, None]), -1)

    def predict_context(self, context, anchor, future_cycles):
        fusion = self.fusion12 if context.shape[-1] == 829 else self.fusion30
        return self.head(fusion(context), anchor, future_cycles)

    def forward(
        self, features, feature_mask, cycles, anchor, future_cycles, channels=30
    ):
        context = self.context(features, feature_mask, cycles, anchor, channels)
        return self.predict_context(context, anchor, future_cycles)
