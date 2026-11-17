"""Reusable label-separated support fitting and fixed-grid nonlinear inference."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock

from models.drs_peft.onesided import (
    INPUT_KEYS,
    CalibrationBatteryTSFM,
    DictionaryBatteryTSFM,
    LowRankBatteryTSFM,
    NoPolicyBatteryTSFM,
)
from scripts.drs_peft.data import file_hash, write_json
from scripts.drs_peft.gradient_audit import state_fingerprint
from scripts.drs_peft.onesided_calibration import (
    actual_local_predictions,
    cache_local_design,
    ridge_solve,
    train_lowrank_support,
)
from scripts.drs_peft.train import load_selected


def _array(value):
    return (
        value.detach().cpu().numpy()
        if isinstance(value, torch.Tensor)
        else np.asarray(value)
    )


def _fingerprint(value):
    """Hash exact tensor values and small metadata without pickling them."""
    digest = hashlib.sha256()

    def update(item):
        if isinstance(item, dict):
            for key in sorted(item):
                digest.update(str(key).encode("utf-8"))
                update(item[key])
        elif isinstance(item, (list, tuple)):
            for entry in item:
                update(entry)
        elif isinstance(item, (torch.Tensor, np.ndarray)):
            values = np.ascontiguousarray(_array(item))
            digest.update(str(values.dtype).encode())
            digest.update(str(values.shape).encode())
            digest.update(values.tobytes())
        else:
            digest.update(json.dumps(item, sort_keys=True, allow_nan=False).encode())

    update(value)
    return digest.hexdigest()


def _save_torch(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    torch.save(value, pending)
    os.replace(pending, path)


def _plain(value):
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (torch.Tensor, np.ndarray)):
        return _array(value).tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class FewShotEngine:
    """Keep source tensors separate from Support targets supplied by the caller.

    ``inputs`` contains six early-input arrays and an ``ids`` vector. No future
    response array or future observation mask is retained by this engine.
    """

    def __init__(
        self,
        f0,
        inputs,
        data_query,
        backbone_identity,
        private_dir,
        device="cuda",
        seed=42,
    ):
        if not isinstance(f0, NoPolicyBatteryTSFM) or f0.bridge.policy_dim != 0:
            raise ValueError("The engine requires the zero-policy source architecture")
        self.f0 = copy.deepcopy(f0).cpu().eval()
        self.f0.requires_grad_(False)
        self.ids = [str(value) for value in inputs["ids"]]
        if len(self.ids) != len(set(self.ids)):
            raise ValueError("Physical input identities must be unique")
        self.inputs = {name: inputs[name] for name in INPUT_KEYS}
        if any(len(value) != len(self.ids) for value in self.inputs.values()):
            raise ValueError("All early-input arrays must share the physical UID axis")
        if tuple(self.inputs["policy"].shape) != (len(self.ids), 0) or tuple(
            self.inputs["policy_mask"].shape
        ) != (len(self.ids), 0):
            raise ValueError("Both policy arrays must have zero columns")
        self.query = np.asarray(data_query, dtype=np.float32)
        if (
            self.query.ndim != 1
            or not len(self.query)
            or not np.isfinite(self.query).all()
            or np.any(np.diff(self.query) <= 0)
            or np.any(self.query < 100)
        ):
            raise ValueError("A fixed increasing physical cycle grid is required")
        self.lookup = {uid: index for index, uid in enumerate(self.ids)}
        self.device = str(device)
        self.seed = int(seed)
        self.private_dir = Path(private_dir)
        self.private_dir.mkdir(parents=True, exist_ok=True)
        self.source_sha256 = (
            backbone_identity.get("source_checkpoint_sha256")
            or backbone_identity.get("reference_checkpoint_sha256")
            or backbone_identity.get("checkpoint_sha256")
            or state_fingerprint(self.f0)
        )
        self.implementation_sha256 = _fingerprint(
            {
                "engine": file_hash(Path(__file__)),
                "model": file_hash(
                    Path(__file__).parents[2] / "models/drs_peft/onesided.py"
                ),
                "calibration": file_hash(
                    Path(__file__).with_name("onesided_calibration.py")
                ),
            }
        )
        self.source_identity = dict(backbone_identity)
        self.registrations = {}
        self.active_method = None
        self.active_model = None
        self._prefix_hashes = {}
        self.last_prediction_details = {}
        self.counters = {
            "jacobian_builds": 0,
            "jacobian_cache_hits": 0,
            "geometry_solves": 0,
            "geometry_cache_hits": 0,
            "nonlinear_optimizer_updates": 0,
            "nonlinear_cache_hits": 0,
            "baseline_cell_forwards": 0,
            "adapted_cell_forwards": 0,
        }

    def register_method(self, name, payload=None):
        if name == "ZERO":
            raise ValueError(
                "ZERO is the shared baseline function and needs no registration"
            )
        kind = (
            "geometry"
            if payload is not None
            else ("calibration" if name == "CAL2" else "lowrank")
        )
        if kind == "lowrank" and name not in ("LoRA", "LoRA_FA", "LoRA-FA"):
            raise ValueError("A custom method needs an explicit geometric dictionary")
        registration = {
            "kind": kind,
            "payload": payload,
            "payload_sha256": _fingerprint(payload) if payload is not None else None,
        }
        if (
            name in self.registrations
            and self.registrations[name]["payload_sha256"]
            != registration["payload_sha256"]
        ):
            raise RuntimeError("An existing method dictionary changed")
        self.registrations[name] = registration

    def _construct(self, method):
        source = copy.deepcopy(self.f0)
        if method == "ZERO":
            model = source
        else:
            specification = self.registrations[method]
            if specification["kind"] == "geometry":
                model = DictionaryBatteryTSFM(source, specification["payload"])
            elif specification["kind"] == "calibration":
                model = CalibrationBatteryTSFM(source)
            else:
                model = LowRankBatteryTSFM(source, method, seed=self.seed)
        return model.to(self.device).eval()

    def _active(self, method):
        if self.active_method != method:
            self.active_model = None
            self.active_method = method
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
            self.active_model = self._construct(method)
        return self.active_model

    def _early(self, uids):
        rows = [self.lookup[str(uid)] for uid in uids]
        return {
            name: torch.as_tensor(_array(values)[rows], device=self.device)
            for name, values in self.inputs.items()
        }

    def _prefix_sha(self, uid):
        if uid not in self._prefix_hashes:
            index = self.lookup[uid]
            self._prefix_hashes[uid] = _fingerprint(
                {
                    name: _array(values)[index : index + 1]
                    for name, values in self.inputs.items()
                }
            )
        return self._prefix_hashes[uid]

    def _method_identity(self, method):
        registration = self.registrations[method]
        return {
            "source_checkpoint_sha256": self.source_sha256,
            "method": method,
            "dictionary_sha256": registration["payload_sha256"],
            "implementation_sha256": self.implementation_sha256,
            "seed": self.seed,
        }

    def _lock(self, key):
        folder = self.private_dir / "_locks"
        folder.mkdir(parents=True, exist_ok=True)
        return FileLock(str(folder / (key + ".lock")))

    def _support(self, support):
        if not support or len({str(cell["uid"]) for cell in support}) != len(support):
            raise ValueError("Support must contain distinct physical cells")
        prepared, identities = [], []
        for cell in sorted(support, key=lambda value: str(value["uid"])):
            uid = str(cell["uid"])
            if uid not in self.lookup:
                raise ValueError("Support identity is outside this engine's input pool")
            early = {
                name: torch.as_tensor(cell["early"][name], device=self.device)
                for name in INPUT_KEYS
            }
            if _fingerprint(early) != self._prefix_sha(uid):
                raise RuntimeError(
                    "The supplied Support prefix differs from its registered UID"
                )
            query = torch.as_tensor(
                cell["query"], device=self.device, dtype=torch.float32
            ).reshape(-1)
            target = np.asarray(_array(cell["target"]), dtype=np.float32).reshape(-1)
            if (
                not 0 < len(query) <= 512
                or len(query) != len(target)
                or not np.isfinite(target).all()
            ):
                raise ValueError(
                    "Support needs one to 512 actual observed target points"
                )
            if "positions" in cell:
                positions = np.asarray(cell["positions"], dtype=np.int64)
                if (
                    positions.ndim != 1
                    or len(positions) != len(query)
                    or np.any(positions < 0)
                    or np.any(positions >= len(self.query))
                    or np.any(np.diff(positions) <= 0)
                    or not np.array_equal(self.query[positions], _array(query))
                ):
                    raise ValueError(
                        "Support coordinates and fixed-grid positions differ"
                    )
            if (
                not bool(torch.isfinite(query).all())
                or not np.isin(_array(query), self.query).all()
            ):
                raise ValueError("Support queries must belong to the fixed cycle grid")
            prepared.append(
                {"uid": uid, "early": early, "query": query, "target": target}
            )
            identities.append(
                {
                    "uid": uid,
                    "prefix_sha256": self._prefix_sha(uid),
                    "query_sha256": _fingerprint(query),
                    "target_sha256": _fingerprint(target),
                }
            )
        return prepared, identities

    def _jacobian(self, method, uid, query):
        uid = str(uid)
        identity = dict(
            self._method_identity(method),
            uid=uid,
            prefix_sha256=self._prefix_sha(uid),
            query_sha256=_fingerprint(query),
        )
        key = _fingerprint(identity)
        folder = self.private_dir / "jacobians"
        folder.mkdir(parents=True, exist_ok=True)
        array_path, receipt_path = folder / (key + ".npz"), folder / (key + ".json")
        with self._lock(key):
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                if receipt["identity"] != identity or receipt[
                    "array_sha256"
                ] != file_hash(array_path):
                    raise RuntimeError("A cached label-free Jacobian identity changed")
                with np.load(array_path, allow_pickle=False) as archive:
                    result = {
                        "f0": archive["f0"].copy(),
                        "design": archive["design"].copy(),
                    }
                result.update(
                    {
                        name: receipt[name]
                        for name in ("backend", "seconds", "dimension", "observations")
                    }
                )
                self.counters["jacobian_cache_hits"] += 1
                return result
            model = self._active(method)
            early = self._early([uid])
            result = cache_local_design(model, early, query, backend="autograd")
            pending = array_path.with_suffix(".pending.npz")
            np.savez_compressed(pending, f0=result["f0"], design=result["design"])
            os.replace(pending, array_path)
            write_json(
                receipt_path,
                dict(
                    identity=identity,
                    array_sha256=file_hash(array_path),
                    **{
                        name: result[name]
                        for name in ("backend", "seconds", "dimension", "observations")
                    },
                ),
            )
            self.counters["jacobian_builds"] += 1
            return result

    def _trial_path(self, key):
        folder = self.private_dir / "adaptation_states"
        folder.mkdir(parents=True, exist_ok=True)
        return folder / (key + ".pt"), folder / (key + ".json")

    def _load_trial(self, path, receipt_path, identity):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt["identity"] != identity or receipt["state_sha256"] != file_hash(
            path
        ):
            raise RuntimeError("A cached Support adaptation changed")
        return torch.load(path, map_location="cpu", weights_only=False)

    def _save_trial(self, path, receipt_path, identity, result):
        _save_torch(path, result)
        write_json(
            receipt_path, {"identity": identity, "state_sha256": file_hash(path)}
        )

    def geometry_trial(self, method, support, hyper):
        if self.registrations[method]["kind"] not in ("geometry", "calibration"):
            raise ValueError("A local ridge trial requires a geometric method or CAL2")
        cells, identities = self._support(support)
        identity = dict(
            self._method_identity(method),
            support=identities,
            hyper=dict(hyper),
            kind="local_ridge",
        )
        key = _fingerprint(identity)
        path, receipt_path = self._trial_path(key)
        with self._lock(key):
            if receipt_path.exists():
                state, details = self._load_trial(path, receipt_path, identity)
                self.counters["geometry_cache_hits"] += 1
                return state, details
            started = time.monotonic()
            caches = [
                self._jacobian(method, cell["uid"], cell["query"]) for cell in cells
            ]
            solved = ridge_solve(
                [value["design"] for value in caches],
                [
                    cell["target"] - value["f0"]
                    for cell, value in zip(cells, caches, strict=True)
                ],
                hyper["lambda_multiplier"],
                eta=hyper.get("eta", 1.0),
                epsilon=hyper.get("epsilon", 1e-12),
            )
            state = dict(
                self._method_identity(method),
                kind="coordinates",
                a=solved["a"].copy(),
                local_system={key: solved[key] for key in ("gram", "rhs", "undamped")},
            )
            model = self._active(method)
            predictions = [
                actual_local_predictions(
                    model, cell["early"], cell["query"], solved["a"], cache
                )
                for cell, cache in zip(cells, caches, strict=True)
            ]
            details = dict(
                {
                    key: value
                    for key, value in solved.items()
                    if key not in ("a", "undamped", "gram", "rhs")
                },
                adapted_coordinate_norm=float(np.linalg.norm(solved["a"])),
                elapsed_seconds=time.monotonic() - started,
                support_ids=[cell["uid"] for cell in cells],
                support_actual_predictions=[value["actual"] for value in predictions],
                support_linear_predictions=[value["linear"] for value in predictions],
                support_linearization_gap=[
                    value["linearization_gap"] for value in predictions
                ],
                support_gap_MAE=float(
                    np.mean([value["gap_MAE"] for value in predictions])
                ),
                parameter_report=model.parameter_report(),
                new_optimizer_updates=0,
            )
            details = _plain(details)
            self._save_trial(path, receipt_path, identity, (state, details))
            self.counters["geometry_solves"] += 1
            return state, details

    def nonlinear_trials(self, method, support, lr, steps=(10, 30)):
        if self.registrations[method]["kind"] != "lowrank":
            raise ValueError(
                "Nonlinear Support optimization is reserved for LoRA baselines"
            )
        cells, identities = self._support(support)
        identity = dict(
            self._method_identity(method),
            support=identities,
            lr=float(lr),
            steps=sorted({int(value) for value in steps}),
            kind="nonlinear_support",
        )
        key = _fingerprint(identity)
        path, receipt_path = self._trial_path(key)
        with self._lock(key):
            if receipt_path.exists():
                self.counters["nonlinear_cache_hits"] += 1
                return self._load_trial(path, receipt_path, identity)
            model = self._construct(method)
            result = train_lowrank_support(model, cells, lr, save_steps=steps)
            snapshots = {}
            for step, record in result["snapshots"].items():
                if not all(
                    np.isfinite(value).all() for value in record["support_predictions"]
                ):
                    raise FloatingPointError(
                        "The fitted Support prediction is nonfinite"
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
            del model
            return snapshots

    def _install_state(self, method, state):
        if any(
            state.get(key) != value
            for key, value in self._method_identity(method).items()
        ):
            raise RuntimeError(
                "An adaptation state belongs to another source function or method"
            )
        model = self._active(method)
        if state["kind"] == "coordinates":
            with torch.no_grad():
                model.coordinates.copy_(
                    torch.as_tensor(
                        state["a"],
                        device=model.coordinates.device,
                        dtype=model.coordinates.dtype,
                    )
                )
        elif state["kind"] == "lowrank":
            load_selected(model, state["selected_state"])
            model.set_adaptation_scale(state.get("adaptation_scale", 1.0))
        else:
            raise ValueError("Unknown adaptation state representation")
        return model.eval()

    def _predict(self, model, query_ids):
        query_ids = [str(value) for value in query_ids]
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("Query identities must be distinct physical cells")
        if not query_ids:
            return np.empty((0, len(self.query)), dtype=np.float32)
        query = torch.as_tensor(self.query, device=self.device)
        output = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(query_ids), 8):
                output.append(
                    model(self._early(query_ids[start : start + 8]), query)
                    .detach()
                    .cpu()
                    .numpy()
                )
        prediction = np.concatenate(output)
        if not np.isfinite(prediction).all():
            raise FloatingPointError(
                "The real fixed-grid Query prediction is nonfinite"
            )
        return prediction

    def baseline_prediction(self, query_ids):
        prediction = self._predict(self._active("ZERO"), query_ids)
        self.counters["baseline_cell_forwards"] += len(query_ids)
        return prediction

    def predict_state(self, method, state, query_ids):
        model = self._install_state(method, state)
        started = time.monotonic()
        prediction = self._predict(model, query_ids)
        self.counters["adapted_cell_forwards"] += len(query_ids)
        self.last_prediction_details = {
            "seconds": time.monotonic() - started,
            "physical_query_cells": len(query_ids),
            "observations_per_query": len(self.query),
        }
        return prediction

    def support_diagnostics(self, method, state, support):
        """Evaluate the selected, fully damped state on its legal Support targets."""
        cells, _ = self._support(support)
        adapted = self._install_state(method, state)
        baseline = self._construct("ZERO")
        before, after = [], []
        with torch.no_grad():
            for cell in cells:
                before.append(baseline(cell["early"], cell["query"])[0].cpu().numpy())
                after.append(adapted(cell["early"], cell["query"])[0].cpu().numpy())
        if not all(np.isfinite(value).all() for value in before + after):
            raise FloatingPointError("Selected Support diagnostics are nonfinite")
        targets = [cell["target"] for cell in cells]
        before_error = [
            value.astype(np.float64) - target.astype(np.float64)
            for value, target in zip(before, targets, strict=True)
        ]
        after_error = [
            value.astype(np.float64) - target.astype(np.float64)
            for value, target in zip(after, targets, strict=True)
        ]
        return _plain(
            {
                "support_ids": [cell["uid"] for cell in cells],
                "support_observations_per_cell": [len(value) for value in targets],
                "support_mae_before": float(
                    np.mean([np.abs(error).mean() for error in before_error])
                ),
                "support_mae_after": float(
                    np.mean([np.abs(error).mean() for error in after_error])
                ),
                "support_mse_before": float(
                    np.mean([np.square(error).mean() for error in before_error])
                ),
                "support_mse_after": float(
                    np.mean([np.square(error).mean() for error in after_error])
                ),
                "support_before_predictions": before,
                "support_after_predictions": after,
                "adaptation_scale": state.get("adaptation_scale", 1.0),
                "evaluation": "actual_nonlinear_cell_mean",
            }
        )

    def query_diagnostics(self, method, state, query_ids, prediction=None):
        """Compare actual and local fixed-grid predictions without Query labels."""
        if state["kind"] != "coordinates":
            raise ValueError("A local linearization needs geometric coordinates")
        query_ids = [str(uid) for uid in query_ids]
        if not query_ids or len(query_ids) != len(set(query_ids)):
            raise ValueError("A local diagnostic needs distinct physical Query cells")
        actual = (
            self.predict_state(method, state, query_ids)
            if prediction is None
            else np.asarray(prediction)
        )
        if actual.shape != (len(query_ids), len(self.query)):
            raise ValueError("The actual predictions use another Query axis")
        query = torch.as_tensor(self.query, device=self.device)
        caches = [self._jacobian(method, uid, query) for uid in query_ids]
        linear = np.stack(
            [
                cache["f0"].astype(np.float64)
                + cache["design"].astype(np.float64) @ np.asarray(state["a"])
                for cache in caches
            ]
        )
        gap = actual.astype(np.float64) - linear
        return _plain(
            {
                "query_ids": query_ids,
                "observations_per_query": len(self.query),
                "query_actual_predictions": actual,
                "query_linear_predictions": linear,
                "query_linearization_gap": gap,
                "query_gap_MAE": float(np.abs(gap).mean()),
                "query_gap_maximum_absolute": float(np.abs(gap).max()),
                "query_labels_read": 0,
            }
        )

    def replay_and_merge(self, method, state, query_ids):
        if not query_ids:
            raise ValueError("A replay audit needs at least one Query prefix")
        uid = str(query_ids[0])
        model = self._install_state(method, state)
        expected = self._predict(model, [uid])
        serialized = io.BytesIO()
        torch.save(state, serialized)
        state_bytes = serialized.tell()
        serialized.seek(0)
        replay_state = torch.load(serialized, map_location="cpu", weights_only=False)
        replay = self._construct(method)
        if replay_state["kind"] == "coordinates":
            with torch.no_grad():
                replay.coordinates.copy_(
                    torch.as_tensor(
                        replay_state["a"],
                        device=self.device,
                        dtype=replay.coordinates.dtype,
                    )
                )
        else:
            load_selected(replay, replay_state["selected_state"])
            replay.set_adaptation_scale(replay_state.get("adaptation_scale", 1.0))
        repeated = self._predict(replay, [uid])
        if not np.array_equal(expected, repeated):
            raise RuntimeError("Saved adaptation state did not exactly replay")
        merge_difference = None
        if hasattr(model, "merged_copy"):
            merged = model.merged_copy().eval()
            merged_prediction = self._predict(merged, [uid])
            merge_difference = float(
                np.abs(merged_prediction.astype(float) - expected.astype(float)).max()
            )
            if not np.allclose(merged_prediction, expected, atol=1e-6, rtol=1e-6):
                raise RuntimeError(
                    "Merged static weights changed the prediction beyond FP32 tolerance"
                )
        return {
            "uid": uid,
            "serialized_state_replay_maximum_difference": 0.0,
            "merged_prediction_maximum_difference": merge_difference,
            "merge_absolute_tolerance": 1e-6,
            "source_checkpoint_sha256": self.source_sha256,
            "serialized_adaptation_state_bytes": state_bytes,
        }
