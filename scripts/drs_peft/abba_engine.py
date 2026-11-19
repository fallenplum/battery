"""Frozen Moirai adapters and equal-cell few-shot optimization."""

from __future__ import annotations

import copy
import hashlib
import json
import time

import numpy as np
import torch
from torch import nn

from models.drs_peft.abba_adalora import (
    AdaLoRALinear,
    AdaLoRARankAllocator,
    adalora_orthogonal_regularization,
)
from models.drs_peft.abba_adapters import make_adapter
from models.drs_peft.onesided import replace_layer, source_layers
from scripts.drs_peft.onesided_calibration import observed_positions
from scripts.drs_peft.train import early_batch
from scripts.drs_peft.transfer_scope_data import load_future_rows


def merged_reference(model):
    """Materialize the effective specialized linear weights at the frozen point."""
    frozen = copy.deepcopy(model).eval().requires_grad_(False)
    layers = source_layers(frozen)
    frozen.abba_module_names = tuple(name for name, _ in layers)
    for name, source in layers:
        base = copy.deepcopy(source.base)
        with torch.no_grad():
            base.weight.add_(
                source.scale * (source.U @ source.effective_core @ source.V.T)
            )
        replace_layer(frozen.backbone, name, base)
    return frozen


def snapshot(model):
    return {
        name: value.detach().cpu().clone()
        for name, value in model.named_parameters()
        if value.requires_grad
    }


def restore(model, state):
    parameters = {
        name: value for name, value in model.named_parameters() if value.requires_grad
    }
    if set(parameters) != set(state):
        raise RuntimeError("Adapter-only state does not match the frozen model")
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(state[name])


def synchronized(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


class AdapterExperiment:
    def __init__(self, reference, method, seed, cfg, device):
        self.options = cfg["methods"][method]
        self.opt = cfg["optimization"]
        self.device = device
        self.model = copy.deepcopy(reference).eval().requires_grad_(False)
        self.layers = {}
        kind = self.options["type"]
        self.is_ada = kind == "AdaLoRA"
        for name, base in reference_modules(self.model):
            module_seed = int.from_bytes(
                hashlib.sha256(f"{seed}|abba_moirai|{name}".encode()).digest()[:8],
                "little",
            ) % (2**63 - 1)
            if self.is_ada:
                adapter = AdaLoRALinear(
                    base, self.options["rank"], self.options["alpha"], seed=module_seed
                )
            else:
                adapter = make_adapter(
                    base,
                    kind.lower(),
                    self.options["rank"],
                    alpha=self.options["alpha"],
                    seed=module_seed,
                )
            replace_layer(self.model.backbone, name, adapter)
            self.layers[name] = adapter
        self.model.to(device)
        self.initial = snapshot(self.model)
        self.parameter_count = sum(p.numel() for p in self.initial.values())

    def fit(self, support, rate):
        restore(self.model, self.initial)
        self.model.train()
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            parameters, lr=rate, weight_decay=self.opt["weight_decay"]
        )
        allocator = None
        steps = self.opt["steps"]
        if self.is_ada:
            c = self.options
            allocator = AdaLoRARankAllocator(
                self.model.backbone,
                target_rank=c["target_rank"],
                total_steps=steps,
                initial_warmup=c["initial_warmup"],
                final_warmup=c["final_warmup"],
                mask_interval=c["mask_interval"],
                beta1=c["beta1"],
                beta2=c["beta2"],
            )
        with torch.no_grad():
            before = float(support_mse(self.model, support))
        synchronized(self.device)
        started = time.perf_counter()
        history = []
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = support_mse(self.model, support)
            if self.is_ada:
                loss = loss + adalora_orthogonal_regularization(
                    self.model, weight=self.options["orthogonality_weight"]
                )
            loss.backward()
            norm = nn.utils.clip_grad_norm_(
                parameters, self.opt["clip_norm"], error_if_nonfinite=True
            )
            optimizer.step()
            if allocator is not None:
                budget, _ = allocator.update_and_mask(step)
                history.append(int(budget))
        self.model.eval()
        with torch.no_grad():
            after = float(support_mse(self.model, support))
        synchronized(self.device)
        if not np.isfinite([before, after]).all():
            raise FloatingPointError("Support objective became nonfinite")
        allocated = allocator.rank_pattern if allocator is not None else {}
        if self.is_ada and (
            set(allocated) != set(self.layers)
            or sum(allocated.values()) > len(self.layers) * self.options["target_rank"]
        ):
            raise RuntimeError(
                "AdaLoRA terminal budget differs from the registered allocator"
            )
        return {
            "support_mse_before": before,
            "support_mse_after": after,
            "trainable_parameters": self.parameter_count,
            "optimizer_steps": steps,
            "actual_optimizer_steps": steps,
            "fit_seconds": time.perf_counter() - started,
            "allocated_rank_by_module": allocated,
            "adalora_budget_history": history,
            "final_gradient_norm_before_clip": float(norm),
        }

    @torch.no_grad()
    def spectrum(self, detail):
        rows, active = [], 0
        for name, layer in self.layers.items():
            delta = layer.delta_weight().detach().cpu().double().numpy()
            weight = layer.base.weight.detach().cpu().double().numpy()
            values = np.linalg.svd(delta, full_matrices=False, compute_uv=False)
            norm = float(np.linalg.norm(values))
            maximum = float(values[0])
            probability = values[values > 0] / values.sum() if maximum else np.empty(0)
            energy = (
                0
                if not maximum
                else int(
                    np.searchsorted(np.cumsum(values**2), 0.99 * np.sum(values**2)) + 1
                )
            )
            kind = self.options["type"]
            rank = detail["allocated_rank_by_module"].get(name, self.options["rank"])
            bound = (
                min(delta.shape)
                if kind == "HiRA"
                else (rank // 2) ** 2
                if kind == "ABBA"
                else rank
            )
            actual_rank = (
                int(np.count_nonzero(values > 1e-5 * maximum)) if maximum else 0
            )
            if actual_rank > bound:
                raise RuntimeError(
                    f"Numerical rank exceeds the operator bound at {name}"
                )
            nonzero_svd_triplets = (
                int(layer.lora_E.count_nonzero()) if self.is_ada else None
            )
            active += (
                rank * (sum(delta.shape) + 1)
                if self.is_ada
                else sum(p.numel() for p in layer.parameters() if p.requires_grad)
            )
            rows.append(
                {
                    "module": name,
                    "rows": delta.shape[0],
                    "cols": delta.shape[1],
                    "frobenius": norm,
                    "relative_frobenius": norm / float(np.linalg.norm(weight)),
                    "spectral_norm": maximum,
                    "numerical_rank_rtol_1e_5": actual_rank,
                    "stable_rank": norm**2 / maximum**2 if maximum else 0.0,
                    "entropy_rank": float(
                        np.exp(-np.sum(probability * np.log(probability)))
                    )
                    if maximum
                    else 0.0,
                    "energy_rank_99": energy,
                    "singular_values_json": json.dumps(
                        values.tolist(), separators=(",", ":")
                    ),
                    "allocated_rank": rank,
                    "rank_bound": min(min(delta.shape), bound),
                    "nonzero_svd_triplets": nonzero_svd_triplets,
                }
            )
        return rows, active


def reference_modules(model):
    """Match the original Moirai target linear module names."""
    layers = [
        (name, model.backbone.get_submodule(name)) for name in model.abba_module_names
    ]
    if len(layers) != 12 or any(
        not isinstance(layer, nn.Linear) for _, layer in layers
    ):
        raise RuntimeError(f"Expected 12 source adaptation modules, got {len(layers)}")
    return layers


def support_batch(data, data_root, uids, device, maximum):
    if len(uids) != len(set(uids)) or not uids:
        raise ValueError("Few-shot Support uses distinct physical cells")
    targets, observed = load_future_rows(data.domain, data_root, uids)
    indexes = [observed_positions(mask, maximum) for mask in observed]
    n = max(map(len, indexes))
    cycles = np.full((len(uids), n), 100.0, dtype=np.float32)
    sampled = np.zeros_like(cycles)
    mask = np.zeros_like(cycles, dtype=bool)
    for i, pos in enumerate(indexes):
        cycles[i, : len(pos)] = data.query[pos]
        sampled[i, : len(pos)] = targets[i, pos]
        mask[i, : len(pos)] = True
    return (
        early_batch(data.inputs, data.rows(uids), device),
        torch.as_tensor(cycles, device=device),
        torch.as_tensor(sampled, device=device),
        torch.as_tensor(mask, device=device),
    )


def support_mse(model, batch):
    early, cycles, target, mask = batch
    err = (model(early, cycles) - target).square()
    return (err.masked_fill(~mask, 0).sum(dim=1) / mask.sum(dim=1)).mean()
