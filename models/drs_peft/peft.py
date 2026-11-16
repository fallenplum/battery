"""Auditable low-rank adapters with explicit source/target parameter scopes.

Linear weights follow PyTorch's [out_features, in_features] convention.
All constructions preserve the original base tensor. LoRA-GA keeps its fixed
initial compensation as a buffer, which is algebraically equivalent to the
authors' compensated frozen base. Per-cell gates act on the rank coordinates.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F


_GATES = ContextVar("drs_peft_gates", default={})
METHODS = ("lora", "random_core", "weight_svd_core", "grad_core", "lora_xs", "lora_ga")


def _rank_check(out_dim, in_dim, rank):
    if not isinstance(rank, int) or not 1 <= rank <= min(out_dim, in_dim):
        raise ValueError(
            "rank must be a positive integer within both matrix dimensions"
        )


def _local_seed(seed, name):
    payload = f"{int(seed)}/drs_peft/{name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def random_orthogonal_bases(
    out_dim, in_dim, rank, seed=42, dtype=torch.float32, device="cpu"
):
    """Draw independent, column-orthonormal bases without touching global RNGs."""
    _rank_check(out_dim, in_dim, rank)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    bases = []
    for dimension in (out_dim, in_dim):
        matrix = torch.randn(dimension, rank, dtype=torch.float64, generator=generator)
        q, r = torch.linalg.qr(matrix, mode="reduced")
        sign = torch.where(r.diag() < 0, -torch.ones(rank), torch.ones(rank))
        bases.append((q * sign).to(device=device, dtype=dtype))
    return tuple(bases)


def svd_bases(matrix, rank):
    """Exact CPU float64 SVD, returned in the input tensor's dtype/device."""
    if matrix.ndim != 2 or not bool(torch.isfinite(matrix).all()):
        raise ValueError("A finite rank-two matrix is required")
    _rank_check(*matrix.shape, rank)
    u, singular, vh = torch.linalg.svd(
        matrix.detach().cpu().double(), full_matrices=False
    )
    return (u[:, :rank].to(matrix), vh[:rank].T.to(matrix), singular[:rank].to(matrix))


class LoRALinear(nn.Module):
    """Standard LoRA with frozen base, Kaiming A, zero B and alpha/r scaling."""

    def __init__(self, base, rank, alpha=None, dropout=0.0, seed=42):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("The adapter requires nn.Linear")
        _rank_check(base.out_features, base.in_features, rank)
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        self.base = base
        self.rank = rank
        self.alpha = float(rank if alpha is None else alpha)
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        self.scale = self.alpha / rank
        self.method = "lora"
        self.dropout = nn.Dropout(dropout)
        # Only the CPU generator changes inside this context.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed))
            self.lora_a = nn.Linear(
                base.in_features,
                rank,
                bias=False,
                dtype=base.weight.dtype,
                device="cpu",
            )
            self.lora_b = nn.Linear(
                rank,
                base.out_features,
                bias=False,
                dtype=base.weight.dtype,
                device="cpu",
            )
            nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b.weight)
        self.lora_a.to(base.weight.device)
        self.lora_b.to(base.weight.device)
        self.base.requires_grad_(False)

    def forward(self, x):
        return self.base(x) + self.scale * self.lora_b(self.lora_a(self.dropout(x)))


class LoRAGALinear(LoRALinear):
    """Gradient-aligned initialization with a fixed original-function offset.

    The exact SVD replaces the official randomized low-rank SVD. Direction and
    scaling definitions follow Outsider565/LoRA-GA. Dropout=0 or eval mode gives
    the original base function at initialization, up to floating arithmetic.
    """

    def __init__(
        self,
        base,
        rank,
        gradient,
        alpha=None,
        dropout=0.0,
        seed=42,
        direction="ArB2r",
        scaling="stable",
        gamma=16.0,
    ):
        if gradient.shape != base.weight.shape or not bool(
            torch.isfinite(gradient).all()
        ):
            raise ValueError(
                "LoRA-GA needs the finite full gradient of this exact weight"
            )
        if 2 * rank > min(base.weight.shape):
            raise ValueError(
                "LoRA-GA direction splitting requires 2r <= min(weight.shape)"
            )
        if direction not in ("ArB2r", "A2rBr", "ArBr") or scaling not in (
            "stable",
            "unit",
            "gd",
        ):
            raise ValueError("Unsupported LoRA-GA direction/scaling")
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        super().__init__(base, rank, alpha, dropout, seed)
        self.method = "lora_ga"
        self.ga_direction, self.ga_scaling, self.ga_gamma = (
            direction,
            scaling,
            float(gamma),
        )
        u, v, _ = svd_bases(gradient, 2 * rank)
        if direction == "ArB2r":
            b, a = u[:, rank : 2 * rank], v[:, :rank].T
        elif direction == "A2rBr":
            b, a = u[:, :rank], v[:, rank : 2 * rank].T
        else:
            b, a = u[:, 0 : 2 * rank : 2], v[:, 1 : 2 * rank : 2].T
        factor = (
            base.out_features**0.25 / math.sqrt(gamma) if scaling == "stable" else 1.0
        )
        if scaling == "gd":
            if self.scale == 0:
                raise ValueError("gd scaling needs a nonzero alpha/r")
            factor = 1.0 / self.scale
        with torch.no_grad():
            self.lora_a.weight.copy_(a * factor)
            self.lora_b.weight.copy_(b * factor)
        self.register_buffer("initial_a", self.lora_a.weight.detach().clone())
        self.register_buffer("initial_b", self.lora_b.weight.detach().clone())

    def forward(self, x):
        current = self.lora_b(self.lora_a(self.dropout(x)))
        initial = F.linear(F.linear(x, self.initial_a), self.initial_b)
        return self.base(x) + self.scale * (current - initial)


class SubspaceLinear(nn.Module):
    """W0 + scale U D_drift C_eff D_abs V.T with per-cell diagonal gates."""

    def __init__(
        self,
        base,
        u,
        v,
        core=None,
        alpha=None,
        dropout=0.0,
        method="grad_core",
        require_orthogonal=True,
    ):
        super().__init__()
        if not isinstance(base, nn.Linear) or u.ndim != 2 or v.ndim != 2:
            raise TypeError("A Linear layer and two rank-two bases are required")
        rank = u.shape[1]
        _rank_check(base.out_features, base.in_features, rank)
        if u.shape != (base.out_features, rank) or v.shape != (base.in_features, rank):
            raise ValueError("Basis shape differs from the base weight")
        if not bool(torch.isfinite(u).all() and torch.isfinite(v).all()):
            raise ValueError("Basis contains nonfinite values")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        if require_orthogonal:
            tolerance = 5e-3 if u.dtype in (torch.float16, torch.bfloat16) else 2e-5
            for basis in (u, v):
                gram = basis.detach().double().T @ basis.detach().double()
                if not torch.allclose(
                    gram,
                    torch.eye(rank, device=gram.device, dtype=gram.dtype),
                    atol=tolerance,
                    rtol=tolerance,
                ):
                    raise ValueError(
                        "The requested projection bases are not column-orthonormal"
                    )
        if core is None:
            core = torch.zeros(
                rank, rank, device=base.weight.device, dtype=base.weight.dtype
            )
        if core.shape != (rank, rank) or not bool(torch.isfinite(core).all()):
            raise ValueError("Core must be finite with shape [r,r]")
        self.base = base
        self.rank = rank
        self.alpha = float(rank if alpha is None else alpha)
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        self.scale = self.alpha / rank
        self.method = method
        self.orthogonal_bases = bool(require_orthogonal)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("U", u.detach().to(base.weight).clone())
        self.register_buffer("V", v.detach().to(base.weight).clone())
        self.core = nn.Parameter(core.detach().to(base.weight).clone())
        self.register_buffer("source_core", None)
        self.register_parameter("delta_core", None)
        self.register_buffer(
            "target_active", torch.tensor(False, device=base.weight.device)
        )
        self.base.requires_grad_(False)

    @property
    def effective_core(self):
        return (
            self.source_core + self.delta_core
            if bool(self.target_active)
            else self.core
        )

    def start_target_adaptation(self):
        if not bool(self.target_active):
            self.source_core = self.core.detach().clone()
            self.delta_core = nn.Parameter(torch.zeros_like(self.core))
            self.target_active.fill_(True)
        self.core.requires_grad_(False)
        self.delta_core.requires_grad_(True)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if prefix + "delta_core" in state_dict:
            self.start_target_adaptation()
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x, abs_gate=None, drift_gate=None):
        contextual = _GATES.get().get(id(self), (None, None))
        abs_gate = contextual[0] if abs_gate is None else abs_gate
        drift_gate = contextual[1] if drift_gate is None else drift_gate
        rank_input = self.dropout(x) @ self.V
        if abs_gate is not None:
            rank_input = rank_input * _expand_gate(abs_gate, rank_input, self.rank)
        rank_output = F.linear(rank_input, self.effective_core)
        if drift_gate is not None:
            rank_output = rank_output * _expand_gate(drift_gate, rank_output, self.rank)
        return self.base(x) + self.scale * (rank_output @ self.U.T)


def _expand_gate(gate, value, rank):
    if gate.ndim != 2 or gate.shape != (value.shape[0], rank):
        raise ValueError("Each gate must have shape [physical_cell_batch,r]")
    if not bool(torch.isfinite(gate).all()):
        raise ValueError("Gate contains nonfinite values")
    return gate.to(value).reshape((len(gate),) + (1,) * (value.ndim - 2) + (rank,))


class LoRAXSLinear(SubspaceLinear):
    """Author-style U R Sigma V.T with Gaussian std1e-5 trainable R.

    The source implementation decomposes W.T. This gives B=U and A=Sigma V.T
    in PyTorch's weight convention. Sigma is absorbed into this module's V.
    The exact float64 SVD replaces the author's randomized TruncatedSVD.
    """

    def __init__(self, base, rank, alpha=None, dropout=0.0, seed=42):
        u, v, singular = svd_bases(base.weight, rank)
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        core = (
            torch.randn(rank, rank, dtype=torch.float64, generator=generator) * 1e-5
        ).to(base.weight)
        super().__init__(
            base,
            u,
            v * singular[None],
            core,
            alpha,
            dropout,
            method="lora_xs",
            require_orthogonal=False,
        )
        self.register_buffer("singular_values", singular.detach().clone())


@contextmanager
def peft_context(module, abs_gate=None, drift_gate=None):
    """Set rank gates for this context. Include backward when checkpointing."""
    context = dict(_GATES.get())
    layers = [
        (name, layer)
        for name, layer in module.named_modules()
        if isinstance(layer, SubspaceLinear)
    ]
    if not layers and (abs_gate is not None or drift_gate is not None):
        raise ValueError("This model has no subspace adapter for the supplied gates")
    for name, layer in layers:
        a = abs_gate[name] if isinstance(abs_gate, Mapping) else abs_gate
        d = drift_gate[name] if isinstance(drift_gate, Mapping) else drift_gate
        for gate in (a, d):
            if gate is not None and (gate.ndim != 2 or gate.shape[1] != layer.rank):
                raise ValueError("Gate rank does not match the selected layer")
        context[id(layer)] = (a, d)
    token = _GATES.set(context)
    try:
        yield
    finally:
        _GATES.reset(token)


def install_peft(
    module,
    method,
    rank,
    alpha=None,
    dropout=0.0,
    target_names=("q_proj", "v_proj"),
    bases=None,
    gradients=None,
    seed=42,
    ga_direction="ArB2r",
    ga_scale="stable",
    ga_gamma=16.0,
):
    """Install adapters atomically and return records with actual parameter counts.

    `bases` for grad_core maps exact qualified module names to (U,V). The caller
    is responsible for source-only gradient provenance. Existing adapters fail
    closed instead of being wrapped a second time.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method {method}")
    if any(
        isinstance(layer, (LoRALinear, SubspaceLinear)) for layer in module.modules()
    ):
        raise ValueError("A clean backbone is required before installing adapters")
    targets = [
        (name, layer)
        for name, layer in module.named_modules()
        if isinstance(layer, nn.Linear) and name.rsplit(".", 1)[-1] in set(target_names)
    ]
    if not targets:
        raise ValueError("No requested Linear target was found")
    flags = [(parameter, parameter.requires_grad) for parameter in module.parameters()]
    replacements = []
    try:
        for name, base in targets:
            local_seed = _local_seed(seed, name)
            if method == "lora":
                adapter = LoRALinear(base, rank, alpha, dropout, local_seed)
            elif method == "lora_ga":
                if gradients is None or name not in gradients:
                    raise ValueError(f"Missing source gradient for {name}")
                adapter = LoRAGALinear(
                    base,
                    rank,
                    gradients[name],
                    alpha,
                    dropout,
                    local_seed,
                    ga_direction,
                    ga_scale,
                    ga_gamma,
                )
            elif method == "lora_xs":
                adapter = LoRAXSLinear(base, rank, alpha, dropout, local_seed)
            else:
                if method == "random_core":
                    u, v = random_orthogonal_bases(
                        base.out_features,
                        base.in_features,
                        rank,
                        local_seed,
                        base.weight.dtype,
                        base.weight.device,
                    )
                elif method == "weight_svd_core":
                    u, v, _ = svd_bases(base.weight, rank)
                else:
                    if bases is None or name not in bases:
                        raise ValueError(f"Missing source-only basis for {name}")
                    u, v = bases[name][:2]
                adapter = SubspaceLinear(
                    base, u, v, alpha=alpha, dropout=dropout, method=method
                )
            replacements.append((name, adapter))
    except Exception:
        for parameter, flag in flags:
            parameter.requires_grad_(flag)
        raise
    module.requires_grad_(False)
    for name, adapter in replacements:
        parent_name, _, attribute = name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        setattr(parent, attribute, adapter)
        if isinstance(adapter, SubspaceLinear):
            adapter.core.requires_grad_(True)
        else:
            adapter.lora_a.requires_grad_(True)
            adapter.lora_b.requires_grad_(True)
    return trainable_parameter_report(module)["layers"]


def begin_target_adaptation(module, mode="core"):
    """Freeze the entire passed model, then expose only the requested target scope."""
    if mode not in ("core", "lora"):
        raise ValueError("mode must be core or lora")
    cls = SubspaceLinear if mode == "core" else LoRALinear
    selected = [layer for layer in module.modules() if isinstance(layer, cls)]
    if not selected:
        raise ValueError(f"No adapters are available for target mode {mode}")
    module.requires_grad_(False)
    for layer in selected:
        if mode == "core":
            layer.start_target_adaptation()
        else:
            layer.lora_a.requires_grad_(True)
            layer.lora_b.requires_grad_(True)
    return trainable_parameter_report(module)


def adaptation_parameters(module):
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


def trainable_parameter_report(module):
    records = []
    basis_numel, compensation_numel = 0, 0
    for name, layer in module.named_modules():
        if not isinstance(layer, (LoRALinear, SubspaceLinear)):
            continue
        local_trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
        basis = (
            layer.U.numel() + layer.V.numel()
            if isinstance(layer, SubspaceLinear)
            else 0
        )
        compensation = (
            layer.initial_a.numel() + layer.initial_b.numel()
            if isinstance(layer, LoRAGALinear)
            else 0
        )
        basis_numel += basis
        compensation_numel += compensation
        records.append(
            dict(
                name=name,
                method=layer.method,
                rank=layer.rank,
                weight_shape=list(layer.base.weight.shape),
                scale=layer.scale,
                trainable=local_trainable,
                base_parameters=sum(p.numel() for p in layer.base.parameters()),
                frozen_basis_elements=basis,
                fixed_compensation_elements=compensation,
                target_active=bool(layer.target_active)
                if isinstance(layer, SubspaceLinear)
                else False,
            )
        )
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return dict(
        total_parameters=total,
        trainable_parameters=trainable,
        frozen_parameters=total - trainable,
        adapter_trainable_parameters=sum(r["trainable"] for r in records),
        frozen_basis_elements=basis_numel,
        fixed_compensation_elements=compensation_numel,
        layers=records,
    )
