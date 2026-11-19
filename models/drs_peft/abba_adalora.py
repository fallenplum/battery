"""Original AdaLoRA training operator and rank allocation for frozen linear layers.

Reference: QingruZhang/AdaLoRA at d10f5ebee16c478fa2f41a44a237b38e8c9b0338,
loralib/loralib/adalora.py (MIT, copyright Microsoft Corporation).
The wrapper preserves the given affine map and uses the upstream unmerged path.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from models.drs_peft.peft import _rank_check


class AdaLoRALinear(nn.Module):
    """Frozen affine map with a trainable SVD-style update B diag(E) A."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float | None = None,
        dropout: float = 0.0,
        seed: int | None = None,
    ):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("AdaLoRA requires nn.Linear")
        _rank_check(base.out_features, base.in_features, rank)
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.method = "adalora"
        self.rank = rank
        self.alpha = float(rank if alpha is None else alpha)
        if not math.isfinite(self.alpha):
            raise ValueError("AdaLoRA alpha must be finite")
        self.scaling = self.alpha if self.alpha > 0 else float(rank)
        self.base = base.requires_grad_(False)
        self.dropout = nn.Dropout(dropout)

        initial_a = torch.empty(rank, base.in_features, dtype=base.weight.dtype)
        initial_b = torch.empty(base.out_features, rank, dtype=base.weight.dtype)
        generator = None if seed is None else torch.Generator().manual_seed(int(seed))
        nn.init.normal_(initial_a, std=0.02, generator=generator)
        nn.init.normal_(initial_b, std=0.02, generator=generator)
        self.lora_A = nn.Parameter(initial_a.to(base.weight))
        self.lora_E = nn.Parameter(base.weight.new_zeros((rank, 1)))
        self.lora_B = nn.Parameter(initial_b.to(base.weight))
        # The original denominator is the initial rank, even after masking E.
        self.ranknum = nn.Parameter(base.weight.new_full((1,), rank), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.dropout(x) @ (self.lora_A * self.lora_E).T @ self.lora_B.T
        return self.base(x) + update * self.scaling / (self.ranknum + 1.0e-5)

    def delta_weight(self) -> torch.Tensor:
        """Return the effective affine weight update, excluding input dropout."""
        update = self.lora_B @ (self.lora_A * self.lora_E)
        return update * self.scaling / (self.ranknum + 1.0e-5)


class AdaLoRARankAllocator:
    """Global cubic allocator with the authors' post-optimizer EMA and masking.

    Call once per optimizer update with a zero-based step and gradients still
    present. During final fine-tuning the scores are frozen and E is re-masked
    after each update, while all three factor tensors stay in the optimizer.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        target_rank: int,
        total_steps: int,
        initial_warmup: int,
        final_warmup: int,
        mask_interval: int,
        beta1: float = 0.85,
        beta2: float = 0.85,
    ):
        self.layers = {
            name: layer
            for name, layer in model.named_modules()
            if isinstance(layer, AdaLoRALinear)
        }
        self.total_rank = sum(layer.rank for layer in self.layers.values())
        self.target_rank = target_rank * len(self.layers)
        if not 0 < self.target_rank < self.total_rank:
            raise ValueError("AdaLoRA target rank budget must be below the initial budget")
        if not (0 <= initial_warmup and 0 <= final_warmup
                and initial_warmup + final_warmup < total_steps and mask_interval > 0):
            raise ValueError("Invalid AdaLoRA schedule")
        if not (0.0 < beta1 < 1.0 and 0.0 < beta2 < 1.0):
            raise ValueError("AdaLoRA EMA factors must be in (0, 1)")

        self.total_steps = total_steps
        self.initial_warmup = initial_warmup
        self.final_start = total_steps - final_warmup
        self.mask_interval = mask_interval
        self.beta1 = beta1
        self.beta2 = beta2
        self.rank_pattern: dict[str, int] = {}
        self._mean_sensitivity: dict[nn.Parameter, torch.Tensor] = {}
        self._mean_deviation: dict[nn.Parameter, torch.Tensor] = {}

    def schedule_threshold(self, step: int) -> tuple[int, bool]:
        """Return the current global rank budget and whether this step masks E."""
        if not 0 <= step < self.total_steps:
            raise ValueError("AdaLoRA expects a zero-based optimizer step within the run")
        if step <= self.initial_warmup:
            return self.total_rank, False
        if step > self.final_start:
            return self.target_rank, True
        fraction = 1 - (step - self.initial_warmup) / (self.final_start - self.initial_warmup)
        budget = int(self.target_rank + (self.total_rank - self.target_rank) * fraction**3)
        return budget, step % self.mask_interval == 0

    def _update_scores(self) -> None:
        for name, layer in self.layers.items():
            for factor in (layer.lora_A, layer.lora_E, layer.lora_B):
                if factor.grad is None:
                    raise RuntimeError(f"AdaLoRA needs gradients before clearing them at {name}")
                if factor not in self._mean_sensitivity:
                    self._mean_sensitivity[factor] = torch.zeros_like(factor)
                    self._mean_deviation[factor] = torch.zeros_like(factor)
                sensitivity = (factor * factor.grad).abs()
                mean = self.beta1 * self._mean_sensitivity[factor] + (1 - self.beta1) * sensitivity
                self._mean_sensitivity[factor] = mean
                self._mean_deviation[factor] = (
                    self.beta2 * self._mean_deviation[factor]
                    + (1 - self.beta2) * (sensitivity - mean).abs()
                )

    def _score(self, factor: nn.Parameter) -> torch.Tensor:
        return self._mean_sensitivity[factor] * self._mean_deviation[factor]

    def _mask(self, budget: int) -> float:
        triplet_scores = []
        for layer in self.layers.values():
            score_a = self._score(layer.lora_A).mean(dim=1)
            score_b = self._score(layer.lora_B).mean(dim=0)
            score_e = self._score(layer.lora_E).flatten()
            triplet_scores.append(score_e + (score_a + score_b))

        # CUDA kthvalue is unavailable with deterministic algorithms enabled.
        # Only the rank scores move to CPU and masking still uses the same cutoff.
        threshold = torch.kthvalue(torch.cat(triplet_scores).cpu(), self.total_rank - budget).values.item()
        remaining_ranks = []
        for layer, scores in zip(self.layers.values(), triplet_scores):
            # Upstream removes every tie at the cutoff, so realized rank can be lower.
            remove = scores <= threshold
            layer.lora_E.masked_fill_(remove[:, None], 0.0)
            remaining_ranks.append((~remove).sum())
        self.rank_pattern = dict(zip(self.layers, torch.stack(remaining_ranks).tolist()))
        return threshold

    @torch.no_grad()
    def update_and_mask(self, step: int) -> tuple[int, float | None]:
        budget, apply_mask = self.schedule_threshold(step)
        if step < self.final_start:
            self._update_scores()
        threshold = self._mask(budget) if apply_mask else None
        return budget, threshold


def adalora_orthogonal_regularization(model: nn.Module, weight: float = 0.1) -> torch.Tensor:
    """Average the original unsquared Frobenius penalties across A and B."""
    penalties = []
    for layer in model.modules():
        if isinstance(layer, AdaLoRALinear):
            identity = torch.eye(layer.rank, dtype=layer.lora_A.dtype, device=layer.lora_A.device)
            penalties.append(torch.linalg.matrix_norm(layer.lora_A @ layer.lora_A.T - identity))
            penalties.append(torch.linalg.matrix_norm(layer.lora_B.T @ layer.lora_B - identity))
    if not penalties:
        raise ValueError("AdaLoRA regularization requires at least one adapted layer")
    return weight * sum(penalties) / len(penalties)
