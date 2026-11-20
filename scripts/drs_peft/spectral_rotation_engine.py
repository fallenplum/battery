"""Execute actual Support updates and meta-gradients on frozen Moirai."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from scripts.drs_peft.abba_engine import AdapterExperiment, synchronized
from scripts.drs_peft.onesided_calibration import observed_positions, ridge_solve
from scripts.drs_peft.run_correction_scale import (
    function_radius,
    optimization_snapshots,
)
from scripts.drs_peft.spectral_rotation import SpectralRotationModel, masked_cell_mse
from scripts.drs_peft.spectral_rotation_protocol import (
    CORE_MODES,
    different_descriptors,
)
from scripts.drs_peft.train import early_batch
from scripts.drs_peft.transfer_scope_data import load_future_rows
from scripts.drs_peft.transfer_scope_geometry import calibration_design


@dataclass
class Labels:
    values: dict

    @classmethod
    def read(cls, data_root, ids, allowed, sealed):
        ids = sorted(set(ids))
        if not ids or set(ids) - allowed or set(ids) & sealed:
            raise RuntimeError(
                "Attempt to decode labels outside the registered source role"
            )
        target, observed = load_future_rows("MATR", data_root, ids)
        if not observed.any(axis=1).all():
            raise ValueError("Expected supervision for every source cell")
        return cls(dict(zip(ids, zip(target, observed, strict=True), strict=True)))

    def arrays(self, uids):
        target, observed = zip(*(self.values[uid] for uid in uids), strict=True)
        return np.stack(target), np.stack(observed)


class Descriptors:
    """Standardize a prefix roster using this fold's fit cells only."""

    def __init__(self, data, fit):
        uids = data["ids"].astype(str).tolist()
        if len(set(uids)) != len(uids):
            raise ValueError("Repeated feature UID")
        self.index = {uid: i for i, uid in enumerate(uids)}
        values = data["values"].astype(np.float64)
        valid = data["mask"].astype(bool)
        if values.shape != valid.shape or values.shape != (
            len(uids),
            len(data["columns"]),
        ):
            raise ValueError("Invalid feature roster shape")
        if not np.isfinite(values[valid]).all():
            raise ValueError("Invalid observed early descriptor")
        rows = np.array([self.index[uid] for uid in fit])
        train = np.where(valid[rows], values[rows], np.nan)
        active = np.isfinite(train).sum(axis=0) >= 2
        mean = np.zeros(values.shape[1])
        std = np.ones_like(mean)
        mean[active] = np.nanmean(train[:, active], axis=0)
        scale = np.nanstd(train[:, active], axis=0)
        std[active] = np.maximum(scale, 1e-8)
        self.values = np.where(
            valid & active, np.clip((values - mean) / std, -3, 3), 0
        ).astype(np.float32)
        self.audit = {
            "fit_ids": fit,
            "features": data["columns"].astype(str).tolist(),
            "train_valid_counts": valid[rows].sum(axis=0).tolist(),
            "active_columns": int(active.sum()),
            "train_mean": mean.tolist(),
            "train_scale": std.tolist(),
            "available_values": int(valid.sum()),
            "total_values": int(valid.size),
            "standard_deviation_floor": 1e-8,
            "clip_standardized": 3,
        }

    def tensor(self, uids, device):
        return torch.as_tensor(
            self.values[[self.index[uid] for uid in uids]], device=device
        )


@dataclass
class Sample:
    early: dict
    cycles: torch.Tensor
    targets: torch.Tensor
    observed: torch.Tensor
    used: np.ndarray

    def batch(self):
        return self.early, self.cycles, self.targets, self.observed


def sampled(data, labels, uids, limit, device):
    target, observed = labels.arrays(uids)
    indexes = [observed_positions(mask, limit) for mask in observed]
    width = max(map(len, indexes))
    coords = np.full((len(uids), width), 100.0, dtype=np.float32)
    values = np.zeros_like(coords)
    mask = np.zeros_like(coords, dtype=bool)
    used = np.zeros_like(observed)
    for i, points in enumerate(indexes):
        coords[i, : len(points)] = data.query[points]
        values[i, : len(points)] = target[i, points]
        mask[i, : len(points)] = True
        used[i, points] = True
    return Sample(
        early_batch(data.inputs, data.rows(uids), device),
        *(torch.as_tensor(x, device=device) for x in (coords, values, mask)),
        used,
    )


def data_descriptor(descriptors, uids, device, salt=None):
    donor_ids = different_descriptors(uids, salt) if salt is not None else uids
    return descriptors.tensor(donor_ids, device)


def peak_allocated(device):
    return (
        torch.cuda.max_memory_allocated(device)
        if torch.device(device).type == "cuda"
        else 0
    )


def reset_peak(device):
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def predict_arrays(model, data, uids, device, descriptor=None, cores=None):
    synchronized(device)
    started = time.perf_counter()
    cycles = torch.as_tensor(data.query, device=device)
    result = []
    with torch.no_grad():
        for offset in range(0, len(uids), 8):
            end = offset + 8
            early = early_batch(data.inputs, data.rows(uids[offset:end]), device)
            if isinstance(model, SpectralRotationModel):
                output = model(
                    early, cycles, descriptor=descriptor[offset:end], cores=cores
                )
            else:
                output = model(early, cycles)
            result.append(output.cpu().numpy())
    synchronized(device)
    out = np.concatenate(result)
    if out.shape != (len(uids), len(data.query)) or not np.isfinite(out).all():
        raise FloatingPointError("Nonfinite actual rollout")
    return out, time.perf_counter() - started


def error(prediction, actual, observed, power):
    difference = np.where(observed, prediction.astype(np.float64) - actual, 0.0)
    if not observed.any(axis=1).all():
        raise ValueError("Expected observed source labels for every scored cell")
    return float(
        np.mean((np.abs(difference) ** power).sum(axis=1) / observed.sum(axis=1))
    )


@dataclass
class Episode:
    support: list
    query: list
    anchors: list
    draw: int
    support_batch: Sample
    support_y: np.ndarray
    support_observed: np.ndarray
    query_y: np.ndarray
    query_observed: np.ndarray
    baseline: np.ndarray
    descriptors: dict
    base_predict_seconds: float

    @property
    def ids(self):
        return self.support + self.query + self.anchors

    @property
    def query_start(self):
        return len(self.support)

    @property
    def anchor_start(self):
        return len(self.support) + len(self.query)

    def scores(self, prediction):
        k, a = self.query_start, self.anchor_start
        return {
            "support_sampled_mse": error(
                prediction[:k], self.support_y, self.support_batch.used, 2
            ),
            "support_full_mse": error(
                prediction[:k], self.support_y, self.support_observed, 2
            ),
            "query_mae": error(prediction[k:a], self.query_y, self.query_observed, 1),
            "query_mse": error(prediction[k:a], self.query_y, self.query_observed, 2),
            "anchor_rms": function_radius(prediction[a:], self.baseline[a:]),
        }


def prepare_episode(
    data, labels, episode, anchors, reference, features, cfg, device, salt
):
    support, query = episode["support"], episode["query"]
    ids = support + query + anchors
    descriptors = {
        "true": data_descriptor(features, ids, device),
        "shuffled": torch.cat(
            [
                data_descriptor(features, group, device, salt=(salt, role))
                for role, group in (
                    ("support", support),
                    ("query", query),
                    ("anchor", anchors),
                )
            ]
        ),
    }
    base, time_base = predict_arrays(reference, data, ids, device)
    inputs = sampled(data, labels, support, cfg["support_max_observed"], device)
    sy, sm = labels.arrays(support)
    qy, qm = labels.arrays(query)
    return Episode(
        support,
        query,
        anchors,
        episode["draw"],
        inputs,
        sy,
        sm,
        qy,
        qm,
        base,
        descriptors,
        time_base,
    )


def fit_mapper(model, method, plans, data, labels, descriptors, cfg, fold, device):
    model.train()
    optimize = torch.optim.Adam(model.controller.parameters(), lr=cfg["meta_lr"])
    trace = []
    reset_peak(device)
    synchronized(device)
    started = time.perf_counter()
    for item in plans:
        support, query = item["support"], item["query"]
        sb = sampled(data, labels, support, cfg["support_max_observed"], device)
        qb = sampled(data, labels, query, cfg["meta_query_max_observed"], device)
        salt = (cfg["seed"], fold, "meta", item["draw"])
        shuffled = method == "ROTATION_SHUFFLED"
        phi_s = data_descriptor(
            descriptors, support, device, (salt, "support") if shuffled else None
        )
        phi_q = data_descriptor(
            descriptors, query, device, (salt, "query") if shuffled else None
        )
        optimize.zero_grad(set_to_none=True)
        cores = model.adapt_episode(
            *sb.batch(),
            phi_s,
            steps=cfg["core_steps"],
            lr=cfg["meta_core_lr"],
            clip_norm=1.0,
            differentiable=True,
        )
        prediction = model(qb.early, qb.cycles, descriptor=phi_q, cores=cores)
        loss = masked_cell_mse(prediction, qb.targets, qb.observed)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(
            model.controller.parameters(), 1.0, error_if_nonfinite=True
        )
        optimize.step()
        trace.append(
            {
                "fold": fold,
                "method": method,
                "meta_episode": item["draw"],
                "train_query_mse": float(loss.detach()),
                "mapper_grad_norm": float(grad),
                "support_ids": "|".join(support),
                "query_ids": "|".join(query),
            }
        )
        del cores, prediction, loss
    synchronized(device)
    stats = {
        "source_mapping_updates": len(trace),
        "source_inner_updates": len(trace) * cfg["core_steps"],
        "source_fit_seconds": time.perf_counter() - started,
        "source_peak_allocated_bytes": peak_allocated(device),
        "source_nonzero_meta_gradient_steps": sum(
            row["mapper_grad_norm"] > 0 for row in trace
        ),
    }
    if not stats["source_nonzero_meta_gradient_steps"]:
        raise RuntimeError(
            "The genuine Support/Query unroll did not update the controller"
        )
    return trace, stats


def core_model(reference, bases, method, features, cfg, device):
    return SpectralRotationModel(
        reference,
        bases,
        CORE_MODES[method],
        features.values.shape[1],
        angle_max=cfg["angle_max_radians"],
    ).to(device)


def frozen_mapper(model):
    model.eval().requires_grad_(False)
    if model.controller is not None and model.controller.training:
        raise RuntimeError("Condition map must stay frozen during source scoring")


def fit_core(model, ep, data, method, cfg, rate, device):
    phi = ep.descriptors["shuffled" if method == "ROTATION_SHUFFLED" else "true"]
    synchronized(device)
    reset_peak(device)
    started = time.perf_counter()
    state = model.adapt_episode(
        *ep.support_batch.batch(),
        phi[: ep.query_start],
        steps=cfg["core_steps"],
        lr=rate,
        clip_norm=1.0,
        differentiable=False,
    )
    synchronized(device)
    runtime = {
        "adapt_seconds": time.perf_counter() - started,
        "adapt_peak_allocated_bytes": peak_allocated(device),
        "adapt_updates": cfg["core_steps"],
    }
    prediction, extra = predict_core(model, ep, data, phi, state, device)
    return state, prediction, runtime | extra


def predict_core(model, ep, data, phi, state, device):
    prediction, inferred = predict_arrays(model, data, ep.ids, device, phi, state)
    with torch.no_grad():
        anchor = phi[ep.anchor_start :]
        norm_by_uid = model.update_frobenius(state, anchor).double()
        norm = float(norm_by_uid.square().mean().sqrt())
        modulation = model.modulation(len(anchor), anchor)
        stats = {}
        if model.mode in {"constant", "conditioned", "shuffled"}:
            stats = {
                "anchor_mean_abs_angle_rad": float(modulation.abs().mean()),
                "anchor_mean_angle_sd_rad": float(
                    modulation.std(dim=0, unbiased=False).mean()
                ),
                "anchor_max_abs_angle_rad": float(modulation.abs().max()),
            }
    return prediction, {
        "inference_seconds": inferred,
        "weight_frobenius_rms": norm,
    } | stats


def fit_cal2(ep, data, rate, eta):
    started = time.perf_counter()
    design = calibration_design(ep.baseline, data.anchor[data.rows(ep.ids)], data.query)
    used = ep.support_batch.used
    residual = ep.support_y - ep.baseline[: ep.query_start]
    solve = ridge_solve(
        [cell[mask] for cell, mask in zip(design[: ep.query_start], used, strict=True)],
        [cell[mask] for cell, mask in zip(residual, used, strict=True)],
        rate,
        eta=eta,
    )
    adapted = time.perf_counter() - started
    started = time.perf_counter()
    prediction = ep.baseline + design @ solve["a"]
    return prediction, {
        "adapt_seconds": adapted,
        "adapt_peak_allocated_bytes": 0,
        "adapt_updates": 0,
        "inference_seconds": ep.base_predict_seconds + time.perf_counter() - started,
        "target_parameters": 2,
        "source_mapper_parameters": 0,
        "basis_storage_bytes": 0,
    }


def lora_operator(base, parent, cfg, device):
    return AdapterExperiment(base, "LORA_R8", cfg["seed"], parent, device)


def lora_snapshots(operator, ep, data, device, rate, steps):
    for step, telemetry in optimization_snapshots(
        operator, ep.support_batch.batch(), rate, steps
    ):
        prediction, seconds = predict_arrays(operator.model, data, ep.ids, device)
        with torch.no_grad():
            frobenius = (
                sum(
                    float(layer.delta_weight().norm()) ** 2
                    for layer in operator.layers.values()
                )
                ** 0.5
            )
        yield (
            step,
            prediction,
            {
                "adapt_seconds": telemetry["training_seconds"],
                "adapt_peak_allocated_bytes": telemetry["fit_peak_allocated_bytes"]
                or 0,
                "adapt_updates": step,
                "inference_seconds": seconds,
                "target_parameters": operator.parameter_count,
                "source_mapper_parameters": 0,
                "basis_storage_bytes": 0,
                "weight_frobenius_rms": frobenius,
            },
        )
