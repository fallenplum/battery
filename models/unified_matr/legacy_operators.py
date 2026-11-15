"""Restricted SRSR/Mamba and analytic Flow operators reused from the project.

The equations and tensor layout follow the original self-reference selection,
signal-only Mamba, and state-dynamics Flow implementations. This module carries
the CPU/CUDA dense SSD reference operator. Source identities are recorded in
SOURCE_LINEAGE.json.
"""

from contextlib import contextmanager
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F


def derived_seed(seed, name):
    return int.from_bytes(hashlib.sha256(f"{seed}/{name}".encode()).digest()[:8], "big") % (2**31)


@contextmanager
def isolated_cpu_seed(seed):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(seed))
        yield


def sinusoidal_position(cycle_ids, width):
    frequency = torch.exp(torch.arange(0, width, 2, device=cycle_ids.device, dtype=torch.float32) * (-math.log(10000.0) / width))
    phase = cycle_ids.float()[..., None] * frequency
    return torch.stack((torch.sin(phase), torch.cos(phase)), -1).flatten(-2)


def deterministic_cumsum(value, dim):
    """Keep strict CUDA reproducibility through an autograd-preserving CPU scan."""
    if value.device.type == "cuda" and torch.are_deterministic_algorithms_enabled():
        return value.to("cpu").cumsum(dim).to(value.device)
    return value.cumsum(dim)


def dense_ssd(x, delta, a, b, c, skip):
    """Causal h_t=exp(delta*A)h_(t-1)+delta*x_t*B_t expansion."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        x, delta, a, b, c, skip = [value.to(dtype) for value in (x, delta, a, b, c, skip)]
        cumulative = deterministic_cumsum((delta * a).permute(0, 2, 1), -1)
        log_decay = cumulative[..., :, None] - cumulative[..., None, :]
        causal = torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool).tril()
        decay = torch.exp(log_decay.masked_fill(~causal, -torch.inf))
        weights = decay * (c @ b.transpose(1, 2))[:, None]
        output = (weights @ (x * delta[..., None]).permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        return output + x * skip[None, None, :, None]


class GatedRMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x, gate):
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        value = x.to(dtype) * F.silu(gate.to(dtype))
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-5)
        return (value * self.weight.to(dtype)).to(x.dtype)


class MaskedMamba2(nn.Module):
    def __init__(self, width=64, state=16, expand=2):
        super().__init__()
        self.inner, self.state, self.head_dim = width * expand, state, 16
        self.heads = self.inner // self.head_dim
        self.in_proj = nn.Linear(width, 2 * self.inner + 2 * state + self.heads, bias=False)
        channels = self.inner + 2 * state
        self.conv1d = nn.Conv1d(channels, channels, 4, padding=3, groups=channels, bias=True)
        delta = torch.exp(torch.rand(self.heads) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp_min(1e-4)
        self.dt_bias = nn.Parameter(delta + torch.log(-torch.expm1(-delta)))
        self.A_log = nn.Parameter(torch.empty(self.heads).uniform_(1, 16).log())
        self.D = nn.Parameter(torch.ones(self.heads))
        self.norm = GatedRMSNorm(self.inner)
        self.out_proj = nn.Linear(self.inner, width, bias=False)

    def forward(self, raw, valid, reference=None):
        raw = torch.where(valid[..., None], raw, 0.0)
        projected = self.in_proj(raw)
        z, xbc, delta = torch.split(projected, (self.inner, self.inner + 2 * self.state, self.heads), -1)
        x, b, c = torch.split(xbc, (self.inner, self.state, self.state), -1)
        if reference is not None:
            reference = torch.where(valid[..., None], reference, 0.0)
            other = self.in_proj(reference)
            _, ref_xbc, delta = torch.split(other, (self.inner, self.inner + 2 * self.state, self.heads), -1)
            _, b, c = torch.split(ref_xbc, (self.inner, self.state, self.state), -1)
        xbc = torch.cat((x, b, c), -1)
        xbc = F.silu(self.conv1d(xbc.transpose(1, 2))[:, :, :raw.shape[1]].transpose(1, 2))
        x, b, c = torch.split(torch.where(valid[..., None], xbc, 0.0), (self.inner, self.state, self.state), -1)
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        delta = torch.where(valid[..., None], F.softplus(delta.to(dtype) + self.dt_bias.to(dtype)), 0.0)
        output = dense_ssd(x.reshape(*x.shape[:2], self.heads, self.head_dim), delta,
                           -self.A_log.to(dtype).exp(), b, c, self.D)
        output = output.reshape(*x.shape[:2], self.inner).to(x.dtype)
        return torch.where(valid[..., None], self.out_proj(self.norm(output, z)), 0.0)


class MambaBlock(nn.Module):
    def __init__(self, local=False, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(64)
        self.mixer = MaskedMamba2(state=4 if local else 16, expand=1 if local else 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, raw, valid, reference=None):
        raw = torch.where(valid[..., None], raw, 0.0)
        normalized = torch.where(valid[..., None], self.norm(raw), 0.0)
        normalized_reference = None
        if reference is not None:
            normalized_reference = torch.where(valid[..., None], self.norm(reference), 0.0)
        result = raw + self.dropout(self.mixer(normalized, valid, normalized_reference))
        return torch.where(valid[..., None], result, 0.0)


def analytic_flow(state, delta, rate, mean):
    decay = torch.exp(-rate * delta)
    return decay * state + (-torch.expm1(-rate * delta)) * mean


class LegacyFlowHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 32))
        self.decoder = nn.Sequential(nn.Linear(16, 64), nn.GELU(), nn.Linear(64, 1, bias=False))
        with torch.no_grad():
            rate = torch.logspace(math.log10(0.3), math.log10(30.0), 16) - 1e-4
            self.generator[-1].weight[:16].zero_()
            self.generator[-1].bias[:16].copy_(torch.log(torch.expm1(rate)))
            self.generator[-1].weight[16:].normal_(std=1e-3)
            self.generator[-1].bias[16:].zero_()
            self.decoder[-1].weight.normal_(std=0.01)

    def forward(self, state, condition, tau):
        rate, mean = self.generator(condition).chunk(2, -1)
        rate, mean = F.softplus(rate) + 1e-4, mean.tanh()
        future = analytic_flow(state[:, None], tau[..., None], rate[:, None], mean[:, None])
        correction = self.decoder(future).squeeze(-1) - self.decoder(state).squeeze(-1)[:, None]
        return torch.where(tau == 0, 0.0, 0.2 * correction)
