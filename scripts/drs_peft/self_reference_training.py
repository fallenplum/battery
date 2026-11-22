"""Full-observation, equal-cell supervised fits on the common Moirai state."""

from __future__ import annotations

import copy
import itertools
import time
from dataclasses import dataclass

import numpy as np
import torch

from models.drs_peft.self_reference_adapters import install_projection_adapter
from scripts.drs_peft.abba_engine import synchronized
from scripts.drs_peft.spectral_objective import masked_cell_loss

GRID = np.arange(101, 5001, dtype=np.float32)
METHOD_BASIS = {
    "RANDOM_30": "random_fixed",
    "VARIANCE_30": "total_activation_covariance",
    "SELF_REFERENCE_30": "self_reference_generalized_eigen",
}


@dataclass(frozen=True)
class Role:
    ids: list
    features: torch.Tensor
    reference: torch.Tensor
    mask: torch.Tensor
    cycles: torch.Tensor
    anchor: torch.Tensor
    targets: np.ndarray
    observed: np.ndarray

    def arrays(self, indices):
        return (
            self.features[indices],
            self.mask[indices],
            self.cycles[indices],
            self.anchor[indices],
        )


def role_bundle(bundle, ids, labels, device):
    lookup = {uid: i for i, uid in enumerate(bundle["ids"].astype(str).tolist())}
    rows = [lookup[uid] for uid in ids]

    def take(field):
        return torch.as_tensor(bundle[field][rows], device=device)

    target, mask = labels.arrays(ids)
    return Role(
        ids,
        take("real_features"),
        take("reference_features"),
        take("feature_mask"),
        take("cycles"),
        take("real_anchor"),
        target,
        mask,
    )


def training_targets(role, indices, device):
    """Pack every actually observed future point, padding only for batching."""
    masks = role.observed[indices]
    counts = masks.sum(axis=1)
    if np.any(counts <= 0):
        raise ValueError(
            "Every supervised training cell must have a valid future point"
        )
    width = int(counts.max())
    cycles = np.full((len(indices), width), 100, dtype=np.float32)
    target = np.zeros_like(cycles)
    observed = np.zeros_like(cycles, dtype=bool)
    for i, row in enumerate(indices):
        valid = masks[i]
        n = counts[i]
        cycles[i, :n] = GRID[valid]
        target[i, :n] = role.targets[row, valid]
        observed[i, :n] = True
    return tuple(
        torch.as_tensor(value, device=device) for value in (cycles, target, observed)
    )


def batches(size, batch_size, *, seed=None):
    order = (
        np.arange(size)
        if seed is None
        else np.random.default_rng(seed).permutation(size)
    )
    for start in range(0, size, batch_size):
        yield order[start : start + batch_size]


def observed_mae(prediction, role):
    if prediction.shape != role.targets.shape or not np.isfinite(prediction).all():
        raise ValueError("Actual predictions must cover the fixed finite forecast grid")
    differences = np.where(
        role.observed, prediction.astype(np.float64) - role.targets, 0
    )
    return np.abs(differences).sum(axis=1) / role.observed.sum(axis=1)


def full_prediction(model, role, channels, batch_size, device):
    model.eval()
    cycles = torch.as_tensor(GRID, device=device)
    synchronized(device)
    started = time.perf_counter()
    with torch.no_grad():
        result = [
            model(*role.arrays(index), cycles, channels=channels).cpu().numpy()
            for index in batches(len(role.ids), batch_size)
        ]
    synchronized(device)
    return np.concatenate(result), time.perf_counter() - started


def split_common_state(model):
    return {
        "fusion12": {
            name: value.cpu().clone()
            for name, value in model.fusion12.state_dict().items()
        },
        "fusion30": {
            name: value.cpu().clone()
            for name, value in model.fusion30.state_dict().items()
        },
        "head": {
            name: value.cpu().clone() for name, value in model.head.state_dict().items()
        },
    }


def load_common_state(model, stored):
    model.fusion12.load_state_dict(stored["fusion12"])
    model.fusion30.load_state_dict(stored["fusion30"])
    model.head.load_state_dict(stored["head"])


def warmup(model, train, cfg, fold, device):
    """Train one shared Bridge and Head with the two input losses equally weighted."""
    model.eval()
    model.requires_grad_(False)
    model.fusion12.requires_grad_(True)
    model.fusion30.requires_grad_(True)
    model.head.requires_grad_(True)
    with torch.no_grad():
        contexts = {
            channels: torch.cat(
                [
                    model.context(*train.arrays(indices), channels=channels)
                    for indices in batches(len(train.ids), cfg["train_batch_size"])
                ]
            )
            for channels in (12, 30)
        }
    parameters = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=cfg["warmup_lr"], weight_decay=0)
    rows = []
    for epoch in range(1, cfg["warmup_epochs"] + 1):
        risks = {12: 0.0, 30: 0.0}
        updates = 0
        started = time.perf_counter()
        for index in batches(
            len(train.ids),
            cfg["train_batch_size"],
            seed=cfg["seed"] + 1009 * epoch + fold,
        ):
            grid, y, mask = training_targets(train, index, device)
            optimizer.zero_grad(set_to_none=True)
            losses = {
                channels: masked_cell_loss(
                    model.predict_context(context[index], train.anchor[index], grid),
                    y,
                    mask,
                    kind="mae",
                )
                for channels, context in contexts.items()
            }
            loss = sum(
                cfg["warmup_view_weights"][str(channels)] * value
                for channels, value in losses.items()
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            for channels, value in losses.items():
                risks[channels] += float(value.detach()) * len(index)
            updates += 1
        synchronized(device)
        rows.append(
            {
                "fold": fold,
                "epoch": epoch,
                "optimizer_updates": updates,
                "training_cells": len(train.ids),
                "training_points": int(train.observed.sum()),
                "mae_12": risks[12] / len(train.ids),
                "mae_30": risks[30] / len(train.ids),
                "seconds": time.perf_counter() - started,
            }
        )
    model.eval().requires_grad_(False)
    return split_common_state(model), rows


@dataclass(frozen=True)
class Trial:
    method: str
    rate: float
    ridge: float

    @property
    def channels(self):
        return 12 if self.method == "LORA_12" else 30

    @property
    def is_fixed(self):
        return self.method in METHOD_BASIS

    def relative(self):
        return f"{self.method}/lr{self.rate:.9g}_ridge{self.ridge:.9g}"


def grid(cfg):
    for method, rate in itertools.product(cfg["methods"], cfg["adapter_rates"]):
        ridges = (
            cfg["self_reference_ridge_multipliers"]
            if method == "SELF_REFERENCE_30"
            else [cfg["self_reference_ridge_multipliers"][1]]
        )
        for ridge in ridges:
            yield Trial(method, rate, ridge)


def build_trial(common, trial, bases, cfg, device):
    model = copy.deepcopy(common)
    basis = bases[trial.ridge][METHOD_BASIS[trial.method]] if trial.is_fixed else None
    install_projection_adapter(
        model,
        "fixed" if trial.is_fixed else "lora",
        basis=basis,
        seed=cfg["seed"],
        lora_rank=cfg["lora_rank"],
        projection_rank=cfg["fixed_input_rank"],
    )
    parameters = {
        name: param for name, param in model.named_parameters() if param.requires_grad
    }
    if (
        sum(v.numel() for v in parameters.values())
        != cfg["expected_adapter_parameters"]
    ):
        raise RuntimeError("Trainable PEFT parameter budget differs")
    return model.to(device), parameters


def fit_trial(model, parameters, trial, train, tune, cfg, fold, device):
    """One uninterrupted source trajectory, saving the three tuning checkpoints."""
    values = list(parameters.values())
    optimizer = torch.optim.AdamW(values, lr=trial.rate, weight_decay=0)
    records, trace, snapshots, predictions = [], [], {}, {}
    base_update = 0
    for epoch in range(1, cfg["adapter_epoch_checkpoints"][-1] + 1):
        model.train()
        started = time.perf_counter()
        if str(device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
        total, steps = 0.0, 0
        for index in batches(
            len(train.ids),
            cfg["train_batch_size"],
            seed=cfg["seed"] + 1009 * epoch + fold,
        ):
            cycles, y, mask = training_targets(train, index, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(*train.arrays(index), cycles, channels=trial.channels)
            loss = masked_cell_loss(prediction, y, mask, kind="mae")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(values, max_norm=1, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(index)
            steps += 1
        synchronized(device)
        base_update += steps
        facts = {
            "fold": fold,
            "method": trial.method,
            "channels": trial.channels,
            "lr": trial.rate,
            "ridge": trial.ridge,
            "epoch": epoch,
            "optimizer_updates": steps,
            "cumulative_updates": base_update,
            "training_mae": total / len(train.ids),
            "training_future_points": int(train.observed.sum()),
            "training_seconds": time.perf_counter() - started,
            "allocated_peak_bytes": torch.cuda.max_memory_allocated(device)
            if str(device).startswith("cuda")
            else 0,
            "trainable_parameters": sum(value.numel() for value in values),
        }
        trace.append(facts)
        if epoch in cfg["adapter_epoch_checkpoints"]:
            prediction, elapsed = full_prediction(
                model, tune, trial.channels, cfg["train_batch_size"], device
            )
            records.append(
                facts
                | {
                    "tune_mae": float(observed_mae(prediction, tune).mean()),
                    "tune_inference_seconds": elapsed,
                }
            )
            snapshots[epoch] = {
                name: value.detach().cpu().clone() for name, value in parameters.items()
            }
            predictions[epoch] = prediction
    return records, trace, snapshots, predictions


def attach_snapshot(parameters, path, epoch):
    saved = torch.load(path, map_location="cpu", weights_only=True)[epoch]
    if set(parameters) != set(saved):
        raise RuntimeError("Selected adapter tensor identities differ")
    with torch.no_grad():
        for name, value in parameters.items():
            value.copy_(saved[name])
