"""Authenticated reference reuse and prefix-only V2 inference."""
from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np
import torch

from scripts.drs_peft.data import CACHE, ROOT, file_hash, write_json
from scripts.drs_peft.onesided_calibration import observed_positions
from scripts.drs_peft.onesided_engine import FewShotEngine, _fingerprint
from scripts.drs_peft.run_onesided_reference import load_reference_model, reference_folder
from scripts.drs_peft.run_specialization import load_prediction
from scripts.drs_peft.train import read_json
from scripts.drs_peft.transfer_scope_data import load_future_rows
from scripts.drs_peft.transfer_scope_geometry import calibration_design, fit_prefix_geometry, fit_scales


PARENT = ROOT / "runtime/drs_peft/onesided_fewshot_v1"
REFERENCES = PARENT / "references"
POOLED = ROOT / "results/drs_peft/specialization_comparison_v1/target_total_K"
BACKBONE = ROOT.parents[1] / ".cache_drs_peft/moirai_small"


class Reference:
    def __init__(self, args, fold, seed):
        self.args, self.fold, self.seed = args, fold, seed
        folder = reference_folder(args.reference_root, fold, seed)
        self.identity = read_json(folder / "REFERENCE_COMPLETE.json")
        self.checkpoint_sha = self.identity["reference_checkpoint_sha256"]
        dictionary = folder / "geometry/DICTIONARIES.pt"
        receipt = read_json(dictionary.with_name("DICTIONARIES_COMPLETE.json"))
        if (
            receipt["identity"]["reference_checkpoint_sha256"] != self.checkpoint_sha
            or receipt["identity"]["reference_state_sha256"] != self.identity["reference_state_sha256"]
            or receipt["identity"]["fit_ids"] != self.identity["fit_ids"]
            or next(x["sha256"] for x in receipt["files"] if x["path"] == dictionary.name) != file_hash(dictionary)
        ):
            raise RuntimeError("OUT dictionary and source reference lineage differ")
        self.dictionary_sha = file_hash(dictionary)
        self.payload = torch.load(dictionary, map_location="cpu", weights_only=False)["arms"]["OUT"]
        if self.payload["dimension"] != 16:
            raise RuntimeError("The registered OUT dictionary must have 16 dimensions")
        self.model = load_reference_model(args.reference_root, args.backbone_dir, fold, seed, "cpu")
        self.engine = None
        self.cost = dict(
            source_reference_fits=0, optimizer_updates=0, jacobian_builds=0,
            parent_jacobian_hits=0, new_cache_jacobian_hits=0, embedding_cell_forwards=0,
            baseline_cell_forwards=0, nonlinear_query_cell_forwards=0,
            nonlinear_support_cell_forwards=0, ridge_solves=0,
        )
        self.reused = {}

    def set_data(self, prefixes):
        self.prefixes = prefixes
        if self.engine is not None:
            del self.engine
        self.engine = FewShotEngine(
            self.model, prefixes.inputs, prefixes.query, self.identity,
            self.args.private_root / "cache" / str(self.fold) / f"s{self.seed}" / prefixes.domain,
            device=self.args.device, seed=self.seed,
        )
        self.engine.register_method("OUT", self.payload)

    def embedding_and_base(self, ids, *, predict=True):
        model = self.engine._active("ZERO")
        z, f0 = [], []
        query = torch.as_tensor(self.prefixes.query, device=self.args.device)
        with torch.no_grad():
            for start in range(0, len(ids), 8):
                batch = self.engine._early(ids[start:start + 8])
                state = model.encode(batch)
                z.append(state.cpu().numpy())
                if predict:
                    f0.append(model.head(state, batch["anchor"], query).cpu().numpy())
        self.cost["embedding_cell_forwards"] += len(ids)
        self.cost["baseline_cell_forwards"] += len(ids) if predict else 0
        return np.concatenate(z), np.concatenate(f0) if predict else None

    def parent_designs(self):
        if self.fold == "final":
            suffix = f"D4_{self.prefixes.domain}/foldfinal/seed{self.seed}"
        else:
            suffix = f"D2/fold{self.fold}/seed{self.seed}"
        return PARENT / "engine" / suffix / "jacobians"

    def designs(self, ids, positions):
        """Read authenticated legacy entries, and batch exact JVPs for misses."""
        grid = self.prefixes.query
        folder = self.engine.private_dir / "design"
        folder.mkdir(parents=True, exist_ok=True)
        found, missing = {}, []
        for uid, pos in zip(ids, positions, strict=True):
            cycles = grid[pos]
            identity = dict(
                self.engine._method_identity("OUT"), uid=uid,
                prefix_sha256=self.engine._prefix_sha(uid),
                query_sha256=_fingerprint(cycles),
            )
            key = _fingerprint(identity)
            path = folder / f"{key}.npz"
            candidates = [(path, identity, "new_cache_jacobian_hits", None)]
            legacy = self.parent_designs()
            legacy_path = legacy / f"{key}.npz"
            candidates.append((legacy_path, identity, "parent_jacobian_hits", None))
            if len(cycles) != len(grid):
                full_identity = dict(identity, query_sha256=_fingerprint(grid))
                candidates.append((legacy / f"{_fingerprint(full_identity)}.npz", full_identity, "parent_jacobian_hits", pos))
            for candidate, expected, counter, subset in candidates:
                if not candidate.with_suffix(".json").exists():
                    continue
                receipt = read_json(candidate.with_suffix(".json"))
                if receipt["identity"] != expected or receipt["array_sha256"] != file_hash(candidate):
                    raise RuntimeError(f"Jacobian provenance mismatch at {candidate}")
                with np.load(candidate, allow_pickle=False) as archive:
                    f0, jac = archive["f0"], archive["design"]
                    found[uid] = (f0, jac) if subset is None else (f0[subset], jac[subset])
                if found[uid][0].shape != (len(pos),) or found[uid][1].shape != (len(pos), 16):
                    raise RuntimeError("Jacobian grid or dimension mismatch")
                self.cost[counter] += 1
                if counter == "parent_jacobian_hits":
                    self.reused[candidate.relative_to(ROOT).as_posix()] = receipt["array_sha256"]
                break
            else:
                missing.append((uid, pos, path, identity))

        for start in range(0, len(missing), self.args.jvp_batch):
            group = missing[start:start + self.args.jvp_batch]
            for (uid, _, path, identity), (f0, jac) in zip(
                group, self.build_designs(group), strict=True
            ):
                save_npz(path, f0=f0, design=jac)
                write_json(path.with_suffix(".json"), {
                    "identity": identity, "array_sha256": file_hash(path),
                    "backend": "batched_exact_autograd_JVP",
                })
                found[uid] = (f0, jac)
                self.cost["jacobian_builds"] += 1
        return found

    def build_designs(self, group):
        model = self.engine._active("OUT")
        early = self.engine._early([item[0] for item in group])
        zero = torch.zeros_like(model.coordinates)
        tangents = torch.eye(16, dtype=zero.dtype, device=zero.device)
        encoder = lambda a: model.encode_with_coordinates(early, a)
        derivative = torch.autograd.functional.jvp

        # Each series has its own fixed coordinates. Pad only the small head grid
        # to share its pointwise JVP, then retain each physical cell's exact axis.
        grid = self.prefixes.query
        count = [len(item[1]) for item in group]
        padded = np.full((len(group), max(count)), 100, dtype=np.float32)
        for i, (_, positions, _, _) in enumerate(group):
            padded[i, :count[i]] = grid[positions]
        query = torch.as_tensor(padded, device=zero.device)
        head = lambda state: model.head(state, early["anchor"], query)
        columns = []
        for tangent in tangents:
            state, direction = derivative(encoder, zero, tangent)
            _, column = derivative(head, state.detach(), direction.detach())
            columns.append(column.detach().cpu().numpy())
        with torch.no_grad():
            base = head(state.detach()).cpu().numpy()
        jacobian = np.stack(columns, axis=-1)
        return [(base[i, :n], jacobian[i, :n]) for i, n in enumerate(count)]

    def freeze_fit(self, prefixes):
        ids = self.identity["fit_ids"]
        path = self.args.private_root / "fit" / str(self.fold) / f"s{self.seed}" / "FIT.npz"
        identity = {
            "checkpoint_sha256": self.checkpoint_sha, "dictionary_sha256": self.dictionary_sha,
            "fit_ids": ids, "data_manifest_sha256": prefixes.cache_manifest_sha256,
            "max_points": 512, "code_sha256": file_hash(Path(__file__)),
        }
        if path.with_suffix(".json").exists():
            receipt = read_json(path.with_suffix(".json"))
            if receipt["identity"] != identity or receipt["array_sha256"] != file_hash(path):
                raise RuntimeError("Source-only fit scale cache changed")
            with np.load(path, allow_pickle=False) as archive:
                features, grams = archive["features"], archive["grams"]
        else:
            self.set_data(prefixes)
            features, _ = self.embedding_and_base(ids, predict=False)
            _, masks = load_future_rows("MATR", self.args.data_root, ids, load_targets=False)
            positions = [observed_positions(row) for row in masks]
            designs = self.designs(ids, positions)
            anchors = prefixes.anchor[prefixes.rows(ids)]
            grams = []
            for uid, anchor, pos in zip(ids, anchors, positions, strict=True):
                base, jac = designs[uid]
                b = calibration_design(base, anchor, prefixes.query[pos])
                a = np.column_stack((b, jac))
                grams.append(a.T @ a / len(pos))
            grams = np.stack(grams)
            save_npz(path, features=features, grams=grams)
            write_json(path.with_suffix(".json"), {"identity": identity, "array_sha256": file_hash(path)})
        if features.shape != (len(ids), 64):
            raise RuntimeError("The frozen representation must have 64 coordinates")
        self.geometry = fit_prefix_geometry(features)
        self.scales = fit_scales(grams)
        self.fit_record = dict(
            identity=identity, array_sha256=file_hash(path), ridge_scales=list(self.scales),
            positive_distance_median=self.geometry.d0, active_dimensions=int(self.geometry.mask.sum()),
            source_mean=self.geometry.source_mean.tolist(), source_std=self.geometry.source_std.tolist(),
        )
        return self.fit_record

    def actual(self, ids, baseline, b, coefficients, *, positions=None):
        """Evaluate nonlinear OUT, adding CAL2 on the unchanged f0 trajectory."""
        coeff = np.asarray(coefficients, dtype=np.float64)
        n = len(ids)
        c = np.broadcast_to(coeff[:, :2], (n, 2))
        correction = np.einsum("ntd,nd->nt", b, c)
        if coeff.shape[1] == 2 or not coeff[:, 2:].any():
            return baseline.astype(np.float64) + correction
        model = self.engine._active("OUT")
        grid = self.prefixes.query
        output = []
        # Shared coordinates allow regular inference batching. Local states are
        # applied independently, with each Query's coefficients and own prefix.
        indices = [list(range(n))] if len(coeff) == 1 and positions is None else [[i] for i in range(n)]
        started = time.perf_counter()
        with torch.no_grad():
            for group in indices:
                for j in range(0, len(group), 8):
                    block = group[j:j + 8]
                    index = block[0]
                    if len(coeff) == 1:
                        a = coeff[0, 2:]
                    else:
                        a = coeff[index, 2:]
                    early = self.engine._early([ids[i] for i in block])
                    cycles = grid if positions is None else grid[positions[index]]
                    query = torch.as_tensor(cycles, device=self.args.device)
                    coordinates = torch.as_tensor(a, dtype=model.coordinates.dtype, device=query.device)
                    output.extend(model(early, query, coordinates=coordinates).cpu().numpy())
        self.cost["nonlinear_query_cell_forwards" if positions is None else "nonlinear_support_cell_forwards"] += n
        self.last_forward_seconds = time.perf_counter() - started
        return np.asarray(output, dtype=np.float64) + correction


def save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".pending.npz")
    np.savez_compressed(pending, **arrays)
    pending.replace(path)


def old_zero(task, seed, family="SPECIALIZED"):
    path = POOLED / "units" / task["task_id"] / f"seed{seed}" / f"{family}_ZERO/K0_draw0/RECEIPT.json"
    receipt = read_json(path)
    if receipt["query_ids"] != task["query_ids"] or receipt["support_ids"] or receipt["K"] != 0:
        raise RuntimeError("The parent ZERO identity differs from the active task")
    return load_prediction(receipt), receipt
