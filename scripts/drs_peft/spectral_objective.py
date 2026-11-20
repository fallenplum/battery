"""Matched-rate MSE Support unrolls with exact MSE or MAE Query objectives."""

from __future__ import annotations

import math
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from scripts.drs_peft.abba_engine import synchronized
from scripts.drs_peft.spectral_rotation import masked_cell_mse
from scripts.drs_peft.spectral_rotation_engine import (
    peak_allocated,
    reset_peak,
    sampled,
)
from scripts.drs_peft.spectral_rotation_protocol import CORE_MODES

OBJECTIVES = {"MM": "mse", "MA": "mae"}
_MAPPER_MODES = {"ROTATION_TRUE", "ROTATION_CONST", "ROTATION_SHUFFLED", "GAIN"}


def _masked_residual(prediction, target, observed):
    if (
        prediction.ndim != 2
        or prediction.shape != target.shape
        or target.shape != observed.shape
    ):
        raise ValueError("Trajectory outcomes must be [physical_cell,time] tensors")
    if (
        observed.dtype != torch.bool
        or not len(prediction)
        or not bool(observed.any(dim=1).all())
    ):
        raise ValueError("Each physical cell needs at least one observed outcome")
    if not bool(
        torch.isfinite(prediction).all() and torch.isfinite(target[observed]).all()
    ):
        raise FloatingPointError("Nonfinite predictions or observed trajectory labels")
    return prediction - torch.where(observed, target, prediction), observed.sum(dim=1)


def masked_cell_loss(prediction, target, observed, kind="mse"):
    """Use exact |e| or e² and give each physical cell the same weight."""
    if kind not in OBJECTIVES.values():
        raise ValueError(f"Unregistered Query objective: {kind}")
    residual, counts = _masked_residual(prediction, target, observed)
    pointwise = residual.abs() if kind == "mae" else residual.square()
    return (pointwise.sum(dim=1) / counts).mean()


def unroll_support(
    model,
    early,
    query,
    target,
    observed,
    support_descriptor=None,
    *,
    lr,
    steps=3,
    clip_norm=1.0,
    differentiable=True,
):
    """Replay the v1 MSE update with diagnostics from the same gradients."""
    if not isinstance(steps, int) or steps <= 0 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("The Support unroll needs positive steps and learning rate")
    if not math.isfinite(clip_norm) or clip_norm <= 0:
        raise ValueError("Use a finite positive Support gradient clip")
    cores = model.initial_cores()
    trace = []
    with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
        for _ in range(steps):
            prediction = model(early, query, support_descriptor, cores)
            loss = masked_cell_mse(prediction, target, observed)
            gradients = torch.autograd.grad(
                loss, tuple(cores.values()), create_graph=differentiable
            )
            squared_norm = torch.stack(
                [grad.square().sum() for grad in gradients]
            ).sum()
            # Match the original clipping stabilization and differentiate through it.
            norm = (squared_norm + 1e-24).sqrt()
            multiplier = (clip_norm / (norm + 1e-6)).clamp_max(1.0)
            gradients = tuple(grad * multiplier for grad in gradients)
            cores = {
                name: core - lr * grad
                for (name, core), grad in zip(cores.items(), gradients, strict=True)
            }
            if not differentiable:
                cores = {
                    name: core.detach().requires_grad_() for name, core in cores.items()
                }
            with torch.no_grad():
                raw_norm = squared_norm.detach().sqrt()
                trace.append(
                    torch.stack(
                        (
                            loss.detach(),
                            raw_norm,
                            raw_norm * multiplier.detach(),
                            multiplier.detach(),
                        )
                    )
                )
    return cores, trace


def _objective_code(objective):
    if objective in OBJECTIVES:
        return objective, OBJECTIVES[objective]
    for code, kind in OBJECTIVES.items():
        if objective == kind:
            return code, kind
    raise ValueError(f"Unregistered objective: {objective}")


def _meta_roles(item):
    support, query = item["support"], item["query"]
    if (
        len(support) != 3
        or len(set(support)) != 3
        or len(query) != 3
        or len(set(query)) != 3
        or set(support) & set(query)
    ):
        raise ValueError(
            "A training episode needs disjoint three-cell Support and Query roles"
        )
    donors = item["donors"]
    for role, uids in (("support", support), ("query", query)):
        assigned = donors[role]
        if (
            len(assigned) != len(uids)
            or set(assigned) != set(uids)
            or any(a == b for a, b in zip(assigned, uids, strict=True))
        ):
            raise ValueError(
                f"Mapper {role} descriptors must use the registered within-role derangement"
            )
    return support, query, donors


def _step_telemetry(values):
    row = {}
    for step, (mse, before, after, coefficient) in enumerate(values, start=1):
        prefix = f"support_step_{step}"
        row.update(
            {
                f"{prefix}_mse": mse,
                f"{prefix}_grad_norm": before,
                f"{prefix}_grad_norm_after_clip": after,
                f"{prefix}_clip_coefficient": coefficient,
                f"{prefix}_clipped": coefficient < 1,
            }
        )
    return row


def fit_mapper(
    model,
    method,
    plans,
    data,
    labels,
    descriptors,
    cfg,
    fold,
    device,
    *,
    objective,
    core_lr,
    training_seed,
):
    """Train one new controller and a fresh Adam at the registered inner rate."""
    objective, kind = _objective_code(objective)
    if (
        method not in _MAPPER_MODES
        or cfg["core_steps"] != 3
        or len(plans) != cfg["train_episodes"]
    ):
        raise ValueError("Expected one registered three-step mapper fit")
    if len({item["draw"] for item in plans}) != len(plans):
        raise ValueError("Training episode keys must be unique")
    if not math.isfinite(core_lr) or core_lr <= 0 or core_lr not in cfg["core_rates"]:
        raise ValueError(
            "Mapper Support rate must be an exact registered positive alpha"
        )
    if model.controller is None or model.mode != CORE_MODES[method]:
        raise ValueError("Mapper fit requires a conditioned model")
    model.train()
    parameters = tuple(model.controller.parameters())
    if any(
        not p.requires_grad or bool(torch.count_nonzero(p.detach())) for p in parameters
    ):
        raise RuntimeError(
            "Every mapper training arm must own a fresh all-zero controller"
        )
    if model.reference.training or any(
        p.requires_grad for p in model.reference.parameters()
    ):
        raise RuntimeError("The reference backbone, bridge and head must remain frozen")
    optimizer = torch.optim.Adam(parameters, lr=cfg["meta_lr"])
    shuffled = method == "ROTATION_SHUFFLED"
    trace = []
    reset_peak(device)
    synchronized(device)
    started = time.perf_counter()
    for item in plans:
        episode_started = time.perf_counter()
        support, query, donors = _meta_roles(item)
        sb = sampled(data, labels, support, cfg["support_max_observed"], device)
        qb = sampled(data, labels, query, cfg["meta_query_max_observed"], device)
        phi_s = descriptors.tensor(donors["support"] if shuffled else support, device)
        phi_q = descriptors.tensor(donors["query"] if shuffled else query, device)
        optimizer.zero_grad(set_to_none=True)
        cores, inner_trace = unroll_support(
            model,
            *sb.batch(),
            phi_s,
            steps=cfg["core_steps"],
            lr=core_lr,
            clip_norm=1.0,
            differentiable=True,
        )
        prediction = model(qb.early, qb.cycles, descriptor=phi_q, cores=cores)
        residual, counts = _masked_residual(prediction, qb.targets, qb.observed)
        query_mae = (residual.abs().sum(dim=1) / counts).mean()
        query_mse = (residual.square().sum(dim=1) / counts).mean()
        objective_value = query_mae if kind == "mae" else query_mse
        objective_value.backward()
        mapper_norm = torch.nn.utils.clip_grad_norm_(
            parameters, 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        coefficient = (1.0 / (mapper_norm + 1e-6)).clamp_max(1.0)
        q_mae, q_mse, norm, coefficient = (
            torch.stack(
                (
                    query_mae.detach(),
                    query_mse.detach(),
                    mapper_norm.detach(),
                    coefficient,
                )
            )
            .cpu()
            .tolist()
        )
        if not all(math.isfinite(v) for v in (q_mae, q_mse, norm, coefficient)):
            raise FloatingPointError(
                "Nonfinite Query loss or mapper gradient in exact unroll"
            )
        telemetry = _step_telemetry(torch.stack(inner_trace).cpu().tolist())
        if not all(math.isfinite(v) for v in telemetry.values()):
            raise FloatingPointError(
                "Nonfinite Support loss or gradient in exact unroll"
            )
        episode_uid = item.get(
            "episode_uid", f"fold{fold}/seed{training_seed}/draw{item['draw']}"
        )
        trace.append(
            {
                "fold": fold,
                "method": method,
                "objective": objective,
                "query_objective": kind,
                "alpha": float(core_lr),
                "training_seed": training_seed,
                "meta_episode": item["draw"],
                "episode_uid": episode_uid,
                "train_query_mae": q_mae,
                "train_query_mse": q_mse,
                "mapper_grad_norm": norm,
                "mapper_grad_norm_after_clip": norm * coefficient,
                "mapper_clip_coefficient": coefficient,
                "mapper_clipped": coefficient < 1,
                "support_ids": "|".join(support),
                "query_ids": "|".join(query),
                "support_observations": int(sb.used.sum()),
                "meta_query_observations": int(qb.used.sum()),
                "support_descriptor_ids": "|".join(
                    donors["support"] if shuffled else support
                )
                if method != "ROTATION_CONST"
                else "",
                "query_descriptor_ids": "|".join(donors["query"] if shuffled else query)
                if method != "ROTATION_CONST"
                else "",
                "meta_episode_seconds": time.perf_counter() - episode_started,
                **telemetry,
            }
        )
        del cores, prediction, residual, objective_value, query_mae, query_mse
    synchronized(device)
    cost = {
        "source_mapping_updates": len(trace),
        "source_inner_updates": len(trace) * cfg["core_steps"],
        "source_fit_seconds": time.perf_counter() - started,
        "source_peak_allocated_bytes": peak_allocated(device),
        "source_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
        if torch.device(device).type == "cuda"
        else 0,
        "source_nonzero_meta_gradient_steps": sum(
            row["mapper_grad_norm"] > 0 for row in trace
        ),
    }
    if not cost["source_nonzero_meta_gradient_steps"]:
        raise RuntimeError("The exact Support-to-Query chain never updated the mapper")
    return trace, cost
