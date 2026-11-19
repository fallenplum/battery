"""Original ABBA, HiRA and LoRA linear updates for the Moirai comparison."""

from __future__ import annotations

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from .peft import LoRALinear as _ExistingLoRALinear
from .peft import _rank_check


UPSTREAM_ADAPTERS = {
    "abba": {
        "paper": "https://openreview.net/forum?id=NvSRYp0oaX",
        "revision": "6a3c9fcae791101043136a5e31136131291004ec",
        "source": "https://github.com/CERT-Lab/abba/blob/6a3c9fcae791101043136a5e31136131291004ec/abba.py",
        "blob_sha": "2c0da07693e2625db44294574afd7df83acb470b",
    },
    "hira": {
        "paper": "https://openreview.net/forum?id=TwJrTz9cRS",
        "revision": "f75f0cfe73e6729f9e1aa5402c5ca5ae36a8c0e8",
        "source": "https://github.com/hqsiswiliam/hira/blob/f75f0cfe73e6729f9e1aa5402c5ca5ae36a8c0e8/hira/tuners/lora.py",
        "blob_sha": "662ff27f788daface421e6f7a0f63994d1d00992",
    },
    "lora": {
        "paper": "https://openreview.net/forum?id=nZeVKeeFYf9",
        "revision": "c4593f060e6a368d7bb5af5273b8e42810cdef90",
        "source": "https://github.com/microsoft/LoRA/blob/c4593f060e6a368d7bb5af5273b8e42810cdef90/loralib/layers.py",
        "blob_sha": "637865600fc9afbf76831e99e3d71eaca6871bda",
    },
    "adalora": {
        "paper": "https://openreview.net/forum?id=lq62uWRJjiY",
        "revision": "d10f5ebee16c478fa2f41a44a237b38e8c9b0338",
        "source": "https://github.com/QingruZhang/AdaLoRA/blob/d10f5ebee16c478fa2f41a44a237b38e8c9b0338/loralib/loralib/adalora.py",
        "blob_sha": "a8a2b557eae63a2f0ce6247dcec5689ca96b856d",
    },
}


class _LinearUpdate:
    """Read the same frozen baseline and materialize updates for analysis."""

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def effective_weight(self):
        return self.weight + self.delta_weight()

    @torch.no_grad()
    def merged_linear(self):
        merged = copy.deepcopy(self.base)
        merged.weight.copy_(self.effective_weight())
        return merged


class LoRALinear(_LinearUpdate, _ExistingLoRALinear):
    """Standard Microsoft LoRA with alpha/r scaling and zero initial update."""

    def delta_weight(self):
        return (self.lora_b.weight @ self.lora_a.weight) * self.scale


class HiRALinear(LoRALinear):
    """HiRA with the frozen original weight as its Hadamard modulator."""

    def __init__(self, base, rank, *, seed=42):
        super().__init__(base, rank, alpha=rank, dropout=0.0, seed=seed)
        self.method = "hira"

    def modulation(self):
        return (self.lora_a.weight.T @ self.lora_b.weight.T).T

    def delta_weight(self):
        return self.weight * self.modulation()

    def effective_weight(self):
        return self.weight * (1 + self.modulation())

    def forward(self, x):
        result = F.linear(x.to(self.lora_a.weight.dtype), self.effective_weight(), self.bias)
        return result.to(x.dtype)


class ABBALinear(_LinearUpdate, nn.Module):
    """ABBA's SVD/zero initialization and Khatri–Rao training path."""

    method = "abba"

    def __init__(self, base, rank, *, alpha=None, seed=42):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("ABBA requires a materialized nn.Linear")
        if isinstance(rank, int):
            if rank % 2:
                raise ValueError("The ABBA total budget rank must be even")
            rank = (rank // 2, rank // 2)
        if not isinstance(rank, (list, tuple)) or len(rank) != 2:
            raise ValueError("ABBA rank must specify a pair or an even total budget")
        self.r1, self.r2 = rank
        _rank_check(base.out_features, base.in_features, self.r1)
        _rank_check(base.out_features, base.in_features, self.r2)
        if self.r1 != self.r2:
            raise ValueError("The original ABBA initializer requires equal factor ranks")
        self.rank = self.r1 + self.r2

        # The release uses sqrt(alpha_i). The paper uses alpha/sqrt(r_i).
        if isinstance(alpha, (tuple, list)):
            self.alpha1, self.alpha2 = map(float, alpha)
            self.paper_alpha = None
        else:
            self.paper_alpha = 2.0 if alpha is None else float(alpha)
            if not math.isfinite(self.paper_alpha) or self.paper_alpha <= 0:
                raise ValueError("ABBA paper alpha must be finite and positive")
            self.alpha1 = self.paper_alpha**2 / self.r1
            self.alpha2 = self.paper_alpha**2 / self.r2
        if not all(math.isfinite(a) and a > 0 for a in (self.alpha1, self.alpha2)):
            raise ValueError("ABBA release scales require positive finite alpha_i")
        self.scaling1 = float(torch.tensor(self.alpha1).sqrt())
        self.scaling2 = float(torch.tensor(self.alpha2).sqrt())
        self.scale = self.scaling1 * self.scaling2
        self.base = base.requires_grad_(False)

        device = self.weight.device
        cuda_devices = [device.index] if device.type == "cuda" else []
        with torch.no_grad(), torch.random.fork_rng(devices=cuda_devices):
            torch.random.default_generator.manual_seed(int(seed))
            if cuda_devices:
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(int(seed))
            u, s, v = torch.svd_lowrank(self.weight.float(), q=self.r1, niter=10)
            sqrt_s = s.sqrt().unsqueeze(1)
            a1 = (v.T * sqrt_s).contiguous()
            b1 = (u * sqrt_s.T).contiguous()
            a2 = nn.init.kaiming_uniform_(torch.randn_like(a1))
            b2 = torch.zeros_like(b1)
        self.lora_a1 = nn.Parameter(a1.to(self.weight))
        self.lora_b1 = nn.Parameter(b1.to(self.weight))
        self.lora_a2 = nn.Parameter(a2.to(self.weight))
        self.lora_b2 = nn.Parameter(b2.to(self.weight))

    def delta_weight(self):
        first = (self.lora_b1 @ self.lora_a1) * self.scaling1
        second = (self.lora_b2 @ self.lora_a2) * self.scaling2
        return first * second

    def forward(self, x):
        b1 = self.lora_b1 * self.scaling1
        b2 = self.lora_b2 * self.scaling2
        factor_rank = self.r1 * self.r2
        b_star = (b1.unsqueeze(2) * b2.unsqueeze(1)).reshape(self.out_features, factor_rank)
        a_star = (self.lora_a1.unsqueeze(1) * self.lora_a2.unsqueeze(0)).reshape(
            factor_rank, self.in_features
        )
        return self.base(x) + F.linear(F.linear(x, a_star), b_star)


def make_adapter(base, method, rank, *, alpha=None, seed=42):
    """Construct one original adapter around a common frozen linear layer."""
    if method == "lora":
        return LoRALinear(base, rank, alpha=alpha, seed=seed)
    if method == "hira":
        if alpha not in (None, 1.0):
            raise ValueError("Original HiRA uses unit modulation scale")
        return HiRALinear(base, rank, seed=seed)
    if method == "abba":
        return ABBALinear(base, rank, alpha=alpha, seed=seed)
    raise ValueError(f"Unsupported linear adapter: {method}")
