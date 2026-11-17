"""Warmup-only versus specialized source functions and cached LoRA-XS adaptation."""

from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock

from models.drs_peft.onesided import source_layers
from models.drs_peft.specialization import XSCalibrationModel, xs_payload_fingerprint
from scripts.drs_peft.data import ROOT, file_hash
from scripts.drs_peft.gradient_audit import state_fingerprint
from scripts.drs_peft.onesided_engine import FewShotEngine, _fingerprint, _plain
from scripts.drs_peft.run_onesided_reference import (
    checked_checkpoint,
    reference_folder,
    static_factory,
    warmup_factory,
)
from scripts.drs_peft.train import read_json, save_torch, selected_state


def load_pretrained_reference(
    reference_root, backbone_dir, fold_id, seed, device="cpu"
):
    folder = reference_folder(reference_root, fold_id, seed)
    complete = read_json(folder / "REFERENCE_COMPLETE.json")
    warmup, checkpoint = checked_checkpoint(folder / "warmup")
    warmup_sha256 = file_hash(checkpoint)
    if warmup_sha256 != complete["warmup_checkpoint_sha256"]:
        raise RuntimeError("The shared Bridge/Head warmup checkpoint changed")
    bases = torch.load(
        folder / "bases/SOURCE_BASES.pt", map_location="cpu", weights_only=False
    )
    model = static_factory(backbone_dir, seed, warmup, bases["bases"])()
    with torch.no_grad():
        for _, layer in source_layers(model):
            layer.core.zero_()
    model.requires_grad_(False)
    model.specialization_identity = {
        "source_condition": "pretrained_official_backbone_with_shared_warmup_Bridge_Head",
        "source_checkpoint_sha256": warmup_sha256,
        "warmup_checkpoint_sha256": warmup_sha256,
        "source_core_increment": "identically_zero",
        "source_U_V_role": "zero_contribution_structural_placeholders",
        "source_state_sha256": state_fingerprint(model),
    }
    return model.to(device).eval()


def audit_warmup_replay(
    model, reference_root, backbone_dir, fold_id, seed, early, query
):
    """Compare the zero-core wrapper with the bare shared warmup on supplied prefixes."""
    folder = reference_folder(reference_root, fold_id, seed)
    warmup, _ = checked_checkpoint(folder / "warmup")
    bare = warmup_factory(backbone_dir, seed)()
    from scripts.drs_peft.train import load_selected

    load_selected(bare, warmup["model"])
    device = next(model.parameters()).device
    bare = bare.to(device).eval()
    model.eval()
    with torch.no_grad():
        expected = bare(early, query)
        actual = model(early, query)
    difference = float((actual - expected).abs().max())
    if not torch.allclose(actual, expected, rtol=0, atol=1e-7):
        raise RuntimeError("The zero-core reference failed to replay bare warmup")
    return {
        "bit_exact": bool(torch.equal(actual, expected)),
        "maximum_absolute_difference": difference,
        "absolute_tolerance": 1e-7,
        "future_labels_read": 0,
        "source_core_nonzero_entries": sum(
            int(torch.count_nonzero(layer.core)) for _, layer in source_layers(model)
        ),
    }


def build_xs_payload(model, rank=4):
    if int(rank) != 4:
        raise ValueError("The registered LoRA-XS comparison fixes rank=4")
    started = time.monotonic()
    factors = {}
    for name, layer in source_layers(model):
        weight = layer.base.weight.detach().cpu().double()
        increment = float(layer.scale) * (
            layer.U.detach().cpu().double()
            @ layer.effective_core.detach().cpu().double()
            @ layer.V.detach().cpu().double().T
        )
        effective = weight + increment
        if not bool(torch.isfinite(effective).all()) or rank > min(effective.shape):
            raise ValueError(
                "A finite effective source weight supporting rank four is required"
            )
        left, singular, vh = torch.linalg.svd(effective, full_matrices=False)
        factors[name] = {
            "U": left[:, :rank].contiguous(),
            "singular": singular[:rank].contiguous(),
            "V": vh[:rank].T.contiguous(),
        }
    payload = {
        "method": "LoRA-XS zero-increment variant",
        "rank": int(rank),
        "alpha": int(rank),
        "source_state_sha256": state_fingerprint(model),
        "factorization": "CPU_float64_SVD_of_W_base_plus_source_scale_U_C0_VT",
        "singular_values_retained": True,
        "layers": factors,
    }
    payload["payload_sha256"] = xs_payload_fingerprint(payload)
    return payload, {
        "SVD_seconds": time.monotonic() - started,
        "SVD_matrices": len(factors),
        "new_SVD_evaluations": len(factors),
        "factor_payload_sha256": payload["payload_sha256"],
    }


def cache_xs_payload(model, path, rank=4):
    """Construct once per reference, validating source identity on every cache read."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    source_sha256 = state_fingerprint(model)
    with FileLock(str(path.with_suffix(".lock"))):
        if path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=False)
            payload = saved["payload"]
            if (
                payload["source_state_sha256"] != source_sha256
                or payload["rank"] != rank
                or payload["payload_sha256"] != xs_payload_fingerprint(payload)
            ):
                raise RuntimeError(
                    "Cached LoRA-XS factors belong to another source or changed"
                )
            return payload, dict(
                saved["construction"], new_SVD_evaluations=0, cache_reused=True
            )
        payload, construction = build_xs_payload(model, rank)
        save_torch(path, {"payload": payload, "construction": construction})
        return payload, dict(construction, cache_reused=False)


def train_xs_support(model, support, learning_rate, save_steps=(10, 30)):
    """Use the same cell-mean MSE objective and optimizer as the original LoRA arm."""
    if not isinstance(model, XSCalibrationModel) or not support:
        raise ValueError("LoRA-XS Support fitting needs its source-preserving model")
    steps = tuple(sorted({int(step) for step in save_steps}))
    if (
        not steps
        or steps[0] < 1
        or not np.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("Positive finite optimizer settings are required")
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not names or any(not name.endswith(".core_xs_delta") for name in names):
        raise ValueError("Only LoRA-XS delta_R entries can be trained")
    model.set_adaptation_scale(1.0)
    optimizer = torch.optim.AdamW(parameters, lr=float(learning_rate), weight_decay=0.0)
    history, snapshots = [], {}
    started = time.monotonic()
    for step in range(1, steps[-1] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for cell in support:
            if len(cell["early"]["x"]) != 1:
                raise ValueError("Each Support entry must be one physical cell")
            prediction = model(cell["early"], cell["query"])[0]
            target = torch.as_tensor(
                cell["target"], device=prediction.device, dtype=prediction.dtype
            ).reshape(-1)
            if prediction.shape != target.shape or not bool(
                torch.isfinite(target).all()
            ):
                raise ValueError(
                    "Support targets differ from sampled physical coordinates"
                )
            losses.append((prediction - target).square().mean())
        loss = torch.stack(losses).mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("The LoRA-XS Support loss is nonfinite")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        history.append(
            {
                "step": step,
                "cell_mean_MSE_before_update": float(loss.detach()),
                "gradient_norm_before_clip": float(norm),
            }
        )
        if step in steps:
            model.eval()
            with torch.no_grad():
                predictions = [
                    model(cell["early"], cell["query"])[0].cpu().numpy()
                    for cell in support
                ]
            snapshots[step] = {
                "model": selected_state(model),
                "support_predictions": predictions,
                "optimizer_updates": step,
                "elapsed_seconds": time.monotonic() - started,
                "learning_rate": float(learning_rate),
                "weight_decay": 0.0,
                "gradient_clip": 1.0,
                "physical_support_cells": len(support),
                "observations_per_cell": [len(value) for value in predictions],
            }
    return {
        "snapshots": snapshots,
        "history": history,
        "optimizer_updates": steps[-1],
        "physical_support_cells": len(support),
    }


class SpecializationEngine(FewShotEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.implementation_sha256 = _fingerprint(
            {
                "parent_implementation_sha256": self.implementation_sha256,
                "specialization_model_sha256": file_hash(
                    ROOT / "models/drs_peft/specialization.py"
                ),
                "specialization_engine_sha256": file_hash(__file__),
            }
        )
        self._source_state_sha256 = None

    def register_method(self, name, payload=None):
        if name != "LoRA-XS":
            return super().register_method(name, payload)
        if payload is None or payload["payload_sha256"] != xs_payload_fingerprint(
            payload
        ):
            raise ValueError("LoRA-XS requires authenticated frozen SVD factors")
        if self._source_state_sha256 is None:
            self._source_state_sha256 = state_fingerprint(self.f0)
        if payload["source_state_sha256"] != self._source_state_sha256:
            raise RuntimeError("LoRA-XS factors belong to another source function")
        if (
            name in self.registrations
            and self.registrations[name]["payload_sha256"] != payload["payload_sha256"]
        ):
            raise RuntimeError("The registered LoRA-XS SVD factors changed")
        self.registrations[name] = {
            "kind": "xs",
            "payload": payload,
            "payload_sha256": payload["payload_sha256"],
        }

    def _construct(self, method):
        if method == "LoRA-XS":
            model = XSCalibrationModel(
                copy.deepcopy(self.f0), self.registrations[method]["payload"]
            )
            return model.to(self.device).eval()
        return super()._construct(method)

    def nonlinear_trials(self, method, support, lr, steps=(10, 30)):
        if method != "LoRA-XS":
            return super().nonlinear_trials(method, support, lr, steps)
        cells, identities = self._support(support)
        identity = dict(
            self._method_identity(method),
            support=identities,
            lr=float(lr),
            steps=sorted({int(value) for value in steps}),
            kind="nonlinear_support_xs",
        )
        key = _fingerprint(identity)
        path, receipt_path = self._trial_path(key)
        with self._lock(key):
            if receipt_path.exists():
                self.counters["nonlinear_cache_hits"] += 1
                return self._load_trial(path, receipt_path, identity)
            model = self._construct(method)
            result = train_xs_support(model, cells, lr, steps)
            snapshots = {}
            for step, record in result["snapshots"].items():
                if not all(
                    np.isfinite(value).all() for value in record["support_predictions"]
                ):
                    raise FloatingPointError(
                        "The LoRA-XS Support prediction is nonfinite"
                    )
                state = dict(
                    self._method_identity(method),
                    kind="lowrank",
                    selected_state=record["model"],
                    adaptation_scale=1.0,
                )
                details = {
                    name: value for name, value in record.items() if name != "model"
                }
                details.update(
                    support_ids=[cell["uid"] for cell in cells],
                    history=result["history"][:step],
                    parameter_report=model.parameter_report(),
                )
                snapshots[step] = {"state": state, "details": _plain(details)}
            self._save_trial(path, receipt_path, identity, snapshots)
            self.counters["nonlinear_optimizer_updates"] += result["optimizer_updates"]
            return snapshots
