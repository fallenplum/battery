"""Same-reference gradient dictionaries and source-only transfer spectra."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from models.drs_peft.peft import SubspaceLinear
from scripts.drs_peft.data import file_hash, write_json
from scripts.drs_peft.gradient_audit import (
    _collect,
    _replay_gradients,
    state_fingerprint,
)

ARMS = ("CORE", "OUT", "IN", "NORMAL")
DEFAULT_RELATIVE_RANK_TOLERANCE = 1e-10
DEFAULT_ABSOLUTE_ENERGY_FRACTION = 1e-24


def as_float64(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("Gradient geometry requires finite arrays")
    return result


def orthonormal_span(value):
    """Choose an exact float64 coordinate basis for an existing column span."""
    matrix = as_float64(value)
    if matrix.ndim != 2 or not 0 < matrix.shape[1] <= matrix.shape[0]:
        raise ValueError("A nonempty tall source basis is required")
    left, triangular = np.linalg.qr(matrix, mode="reduced")
    if np.min(np.abs(np.diag(triangular))) <= 1e-12:
        raise ValueError("The source projection basis is rank deficient")
    signs = np.where(np.diag(triangular) < 0, -1.0, 1.0)
    return left * signs


def projected_coordinates(gradients, u, v):
    """Return compact coordinates of the four fixed-rank geometric blocks."""
    values, u, v = (as_float64(item) for item in (gradients, u, v))
    if values.ndim not in (2, 3) or values.shape[-2:] != (u.shape[0], v.shape[0]):
        raise ValueError("Gradient matrices and source projection dimensions differ")
    if u.shape[1] != v.shape[1]:
        raise ValueError("The frozen source adapter must have equal left/right rank")
    gv = values @ v
    utg = u.T @ values
    core = u.T @ gv
    out = gv - u @ core
    incoming = utg - core @ v.T
    normal = values - u @ utg - gv @ v.T + u @ core @ v.T
    return {"CORE": core, "OUT": out, "IN": incoming, "NORMAL": normal}


def materialize_block(coordinates, arm, u, v):
    values = as_float64(coordinates)
    if arm == "CORE":
        return u @ values @ v.T
    if arm == "OUT":
        return values @ v.T
    if arm == "IN":
        return u @ values
    if arm == "NORMAL":
        return values
    raise ValueError("Unknown one-sided geometry arm")


def physical_cell_weights(protocols, weighting="protocol_equal"):
    protocols = np.asarray(protocols, dtype=str)
    if protocols.ndim != 1 or not len(protocols):
        raise ValueError("Nonempty physical-cell protocol identities are required")
    if weighting == "cell_equal":
        return np.full(len(protocols), 1.0 / len(protocols))
    if weighting != "protocol_equal":
        raise ValueError("Unknown preregistered physical-cell weighting")
    groups, inverse, counts = np.unique(
        protocols, return_inverse=True, return_counts=True
    )
    return 1.0 / (len(groups) * counts[inverse])


def source_layer_bases(model):
    result = {
        name: (layer.U.detach().cpu(), layer.V.detach().cpu())
        for name, layer in model.backbone.named_modules()
        if isinstance(layer, SubspaceLinear)
    }
    if len(result) != 12 or any(
        u.shape != (384, 4) or v.shape != (384, 4) for u, v in result.values()
    ):
        raise ValueError("The reference must have twelve rank-four Q/V adapters")
    return result


def collect_same_point_gradients(
    model,
    data,
    inputs,
    fit_ids,
    reference_checkpoint,
    cache_root,
    seed,
    checkpoint_stage="final_static_f0",
):
    """Collect each fit cell at the actual frozen f0, with no-policy inputs."""
    fit_ids = list(fit_ids)
    if not fit_ids or len(fit_ids) != len(set(fit_ids)):
        raise ValueError("Fit identities must be unique physical cells")
    rows = data.rows(fit_ids)
    if np.any(np.asarray(inputs["policy"])[rows] != 0) or np.any(
        np.asarray(inputs["policy_mask"])[rows]
    ):
        raise ValueError(
            "Policy values and masks must be zero before gradient collection"
        )
    reference_checkpoint = Path(reference_checkpoint)
    checkpoint_sha = file_hash(reference_checkpoint)
    if checkpoint_stage not in ("no_policy_warmup", "final_static_f0"):
        raise ValueError("Unknown reference-gradient checkpoint stage")
    if checkpoint_stage == "final_static_f0":
        source_layer_bases(model)
    layers = {}
    for name, layer in model.backbone.named_modules():
        if name.rsplit(".", 1)[-1] not in ("q_proj", "v_proj"):
            continue
        base = layer.base if isinstance(layer, SubspaceLinear) else layer
        if not isinstance(base, torch.nn.Linear) or tuple(base.weight.shape) != (
            384,
            384,
        ):
            raise ValueError("The reference Q/V gradient target family changed")
        layers[name] = base
    if len(layers) != 12:
        raise ValueError("The reference must expose twelve official Q/V weights")
    role = (
        "ONESIDED_F0_FIT"
        if checkpoint_stage == "final_static_f0"
        else "ONESIDED_WARMUP_FIT"
    )
    before = state_fingerprint(model)
    started = time.perf_counter()
    gradients, receipt = _collect(
        model,
        data,
        inputs,
        rows,
        rows,
        layers,
        Path(cache_root) / "raw",
        seed,
        role,
        "nominal",
        next(model.parameters()).device,
        before,
    )
    replay = _replay_gradients(
        model,
        data,
        inputs,
        [(role, "nominal", rows, rows)],
        {(role, "nominal"): gradients},
        layers,
        next(model.parameters()).device,
    )
    if state_fingerprint(model) != before:
        raise RuntimeError("Fit gradient extraction changed the frozen reference")
    if (
        receipt["identity"]["ids"] != fit_ids
        or receipt["identity"]["model_sha256"] != before
    ):
        raise RuntimeError(
            "Gradient collection differs from the declared f0 and fit cells"
        )
    identity = {
        "reference_checkpoint_sha256": checkpoint_sha,
        "reference_state_sha256": before,
        "fit_ids": fit_ids,
        "seed": int(seed),
        "policy_values_and_masks_zero": True,
        "gradient_checkpoint": checkpoint_stage,
        "gradient_loss": "physical_cell_observed_full_future_MAE",
        "gradient_evaluations": len(fit_ids),
        "endpoint_replay_evaluations": 2 if len(fit_ids) > 1 else 1,
        "optimizer_updates": 0,
        "elapsed_seconds": time.perf_counter() - started,
        "raw_gradient_receipt": receipt,
        "replay": replay,
    }
    marker = Path(cache_root) / "SAME_POINT_GRADIENT_COLLECTION.json"
    if marker.exists():
        previous = json.loads(marker.read_text(encoding="utf-8"))
        stable_keys = (
            "reference_checkpoint_sha256",
            "reference_state_sha256",
            "fit_ids",
            "seed",
            "policy_values_and_masks_zero",
            "gradient_checkpoint",
            "gradient_loss",
            "raw_gradient_receipt",
        )
        if any(previous[key] != identity[key] for key in stable_keys):
            raise RuntimeError("The cached reference gradient collection changed")
        identity = previous
    else:
        write_json(marker, identity)
    return gradients, identity


def _projector_bases(bases):
    result, precision = {}, []
    for name, (raw_u, raw_v) in bases.items():
        raw_u, raw_v = as_float64(raw_u), as_float64(raw_v)
        u, v = orthonormal_span(raw_u), orthonormal_span(raw_v)
        result[name] = (u, v)
        precision.append(
            {
                "layer": name,
                "source_left_gram_error": float(
                    np.max(np.abs(raw_u.T @ raw_u - np.eye(raw_u.shape[1])))
                ),
                "source_right_gram_error": float(
                    np.max(np.abs(raw_v.T @ raw_v - np.eye(raw_v.shape[1])))
                ),
                "left_projector_change": float(
                    np.max(np.abs(raw_u @ raw_u.T - u @ u.T))
                ),
                "right_projector_change": float(
                    np.max(np.abs(raw_v @ raw_v.T - v @ v.T))
                ),
            }
        )
    return result, precision


def _block_rows(values, coordinates, u, v, ids, protocols, name):
    blocks = {arm: materialize_block(coordinates[arm], arm, u, v) for arm in ARMS}
    flat = {arm: value.reshape(len(ids), -1) for arm, value in blocks.items()}
    total = np.square(values).sum((1, 2))
    energy = {arm: np.square(value).sum((1, 2)) for arm, value in blocks.items()}
    reconstructed = sum(blocks.values())
    records = []
    for index, uid in enumerate(ids):
        denominator = max(float(total[index]), np.finfo(np.float64).tiny)
        orthogonality = (
            max(
                abs(float(np.dot(flat[first][index], flat[second][index])))
                for position, first in enumerate(ARMS)
                for second in ARMS[position + 1 :]
            )
            / denominator
        )
        records.append(
            {
                "cell_uid": uid,
                "protocol_uid": str(protocols[index]),
                "layer": name,
                "gradient_energy": float(total[index]),
                "gradient_norm": float(np.sqrt(total[index])),
                "zero_gradient": bool(total[index] == 0),
                "relative_reconstruction_error": float(
                    np.linalg.norm(values[index] - reconstructed[index])
                    / np.sqrt(denominator)
                ),
                "relative_energy_closure_error": float(
                    abs(sum(item[index] for item in energy.values()) - total[index])
                    / denominator
                ),
                "maximum_relative_cross_block_inner_product": orthogonality,
                **{arm + "_energy": float(energy[arm][index]) for arm in ARMS},
                **{
                    arm + "_energy_ratio": float(energy[arm][index] / denominator)
                    if total[index] > 0
                    else None
                    for arm in ARMS
                },
            }
        )
    return records


def _eigen_dictionary(gram, requested_dimension, relative_tolerance, energy_floor):
    gram = (gram + gram.T) * 0.5
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    scale = max(float(values[0]), 0.0)
    threshold = max(scale * relative_tolerance, energy_floor)
    if float(values[-1]) < -max(scale * 1e-8, energy_floor):
        raise FloatingPointError("The gradient sample Gram is numerically indefinite")
    rank = int(np.count_nonzero(values > threshold))
    dimension = min(int(requested_dimension), rank)
    selected_values, selected_vectors = values[:dimension], vectors[:, :dimension]
    for column in range(dimension):
        pivot = int(np.argmax(np.abs(selected_vectors[:, column])))
        if selected_vectors[pivot, column] < 0:
            selected_vectors[:, column] *= -1
    return values, selected_vectors, selected_values, rank, threshold


def _layer_payload(arm, directions, u, v):
    result = {
        "kind": arm,
        "directions": torch.from_numpy(directions.astype(np.float32)),
    }
    if arm in ("CORE", "IN"):
        result["left"] = torch.from_numpy(u.astype(np.float32))
    if arm in ("CORE", "OUT"):
        result["right"] = torch.from_numpy(v.astype(np.float32))
    return result


def dictionary_buffer_bytes(payload):
    return sum(
        value.numel() * value.element_size()
        for layer in payload["layers"].values()
        for value in layer.values()
        if torch.is_tensor(value)
    )


def layer_dictionary_gram(layer):
    arm = layer["kind"]
    directions = as_float64(layer["directions"])
    dimension = len(directions)
    if dimension == 0:
        return np.zeros((0, 0), dtype=np.float64)
    transformed = directions
    if arm in ("CORE", "IN"):
        left = as_float64(layer["left"])
        transformed = (left.T @ left) @ transformed
    if arm in ("CORE", "OUT"):
        right = as_float64(layer["right"])
        transformed = transformed @ (right.T @ right)
    return transformed.reshape(dimension, -1) @ directions.reshape(dimension, -1).T


def dictionary_gram(payload):
    dimension = payload["dimension"]
    result = np.zeros((dimension, dimension), dtype=np.float64)
    for layer in payload["layers"].values():
        result += layer_dictionary_gram(layer)
    return result


def _fit_coordinate_contribution(values, layer):
    projected = values
    if layer["kind"] in ("CORE", "IN"):
        projected = as_float64(layer["left"]).T @ projected
    if layer["kind"] in ("CORE", "OUT"):
        projected = projected @ as_float64(layer["right"])
    directions = as_float64(layer["directions"])
    return (
        projected.reshape(len(values), -1) @ directions.reshape(len(directions), -1).T
    )


def build_dictionary_bundle(
    gradients,
    bases,
    ids,
    protocols,
    requested_dimension=16,
    weighting="protocol_equal",
    relative_rank_tolerance=DEFAULT_RELATIVE_RANK_TOLERANCE,
    absolute_energy_fraction=DEFAULT_ABSOLUTE_ENERGY_FRACTION,
    identity=None,
):
    """Fit four global Frobenius dictionaries from unaveraged fit gradients."""
    started = time.perf_counter()
    if (
        identity is not None
        and identity.get("gradient_checkpoint", "final_static_f0") != "final_static_f0"
    ):
        raise ValueError("D1 dictionaries require the final f0 gradients")
    ids, protocols = list(ids), np.asarray(protocols, dtype=str)
    if len(ids) != len(set(ids)) or len(ids) != len(protocols) or not ids:
        raise ValueError("Dictionary fitting requires unique physical fit cells")
    if set(gradients) != set(bases) or not gradients:
        raise ValueError("Gradient and source-basis target layers differ")
    if not isinstance(requested_dimension, int) or requested_dimension < 1:
        raise ValueError("A positive requested dictionary dimension is required")
    if not 0 < relative_rank_tolerance < 1 or not 0 < absolute_energy_fraction < 1:
        raise ValueError("Rank tolerances must be finite positive fractions")
    weights = physical_cell_weights(protocols, weighting)
    root_weights = np.sqrt(weights)
    projections, precision = _projector_bases(bases)
    grams = {arm: np.zeros((len(ids), len(ids)), dtype=np.float64) for arm in ARMS}
    block_records, total_energy = [], 0.0
    for name, raw in gradients.items():
        values = as_float64(raw)
        if values.ndim != 3 or len(values) != len(ids):
            raise ValueError("Every matrix gradient cache needs one row per fit cell")
        u, v = projections[name]
        coordinates = projected_coordinates(values, u, v)
        block_records.extend(
            _block_rows(values, coordinates, u, v, ids, protocols, name)
        )
        total_energy += float(weights @ np.square(values).sum((1, 2)))
        for arm in ARMS:
            flat = coordinates[arm].reshape(len(ids), -1)
            scaled = flat * root_weights[:, None]
            grams[arm] += scaled @ scaled.T
    block_table = pd.DataFrame(block_records)
    for key in (
        "relative_reconstruction_error",
        "relative_energy_closure_error",
        "maximum_relative_cross_block_inner_product",
    ):
        if block_table[key].max() > 1e-10:
            raise FloatingPointError(
                "The same-point gradient decomposition failed " + key
            )
    arms, mixing, spectra = {}, {}, []
    floor = max(total_energy * absolute_energy_fraction, np.finfo(np.float64).tiny)
    for arm in ARMS:
        eigenvalues, eigenvectors, selected, rank, threshold = _eigen_dictionary(
            grams[arm],
            requested_dimension,
            relative_rank_tolerance,
            floor,
        )
        dimension = len(selected)
        mixing[arm] = root_weights[:, None] * eigenvectors / np.sqrt(selected)[None]
        arms[arm] = {
            "arm": arm,
            "dimension": dimension,
            "requested_dimension": requested_dimension,
            "numerical_rank": rank,
            "rank_threshold": threshold,
            "status": "ESTIMABLE" if dimension > 0 else "NO_NUMERICAL_DIRECTION",
            "layers": {},
        }
        spectra.extend(
            {
                "arm": arm,
                "index": index + 1,
                "eigenvalue": float(value),
                "selected": index < dimension,
                "rank_threshold": threshold,
            }
            for index, value in enumerate(eigenvalues)
        )
    fit_coordinates = {
        arm: np.zeros((len(ids), arms[arm]["dimension"]), dtype=np.float64)
        for arm in ARMS
    }
    for name, raw in gradients.items():
        values = as_float64(raw)
        u, v = projections[name]
        coordinates = projected_coordinates(values, u, v)
        for arm in ARMS:
            directions = np.tensordot(mixing[arm].T, coordinates[arm], axes=1)
            layer = _layer_payload(arm, directions, u, v)
            arms[arm]["layers"][name] = layer
            if arms[arm]["dimension"]:
                fit_coordinates[arm] += _fit_coordinate_contribution(values, layer)
    for arm, payload in arms.items():
        actual_gram = dictionary_gram(payload)
        dimension = payload["dimension"]
        error = (
            float(np.max(np.abs(actual_gram - np.eye(dimension))))
            if dimension
            else None
        )
        if error is not None and error > 1e-5:
            raise FloatingPointError(
                "The saved float32 dictionary lost global orthogonality"
            )
        payload["frozen_buffer_bytes"] = dictionary_buffer_bytes(payload)
        payload["maximum_frobenius_gram_error"] = error
    cell_weights = dict(zip(ids, weights, strict=True))
    global_energy_ratios = {}
    for arm in ARMS:
        projected_energy = sum(
            cell_weights[row["cell_uid"]] * row[arm + "_energy"]
            for row in block_records
        )
        global_energy_ratios[arm] = (
            float(projected_energy / total_energy) if total_energy > 0 else None
        )
    bundle = {
        "identity": identity if identity is not None else {"scope": "engineering_only"},
        "fit_ids": ids,
        "fit_protocols": protocols.tolist(),
        "physical_cells": len(ids),
        "protocol_groups": len(np.unique(protocols)),
        "weighting": weighting,
        "weights": weights.tolist(),
        "requested_dimension": requested_dimension,
        "common_dimension": min(payload["dimension"] for payload in arms.values()),
        "relative_rank_tolerance": relative_rank_tolerance,
        "absolute_energy_fraction": absolute_energy_fraction,
        "total_weighted_gradient_energy": total_energy,
        "global_energy_ratios": global_energy_ratios,
        "global_energy_ratio_scope": "sum_over_layers_and_protocol_equal_weighted_cells_before_dividing_by_total_energy",
        "projection_basis_precision": precision,
        "arms": arms,
        "fit_coordinates": {
            arm: torch.from_numpy(value) for arm, value in fit_coordinates.items()
        },
        "construction_seconds": time.perf_counter() - started,
        "optimizer_updates": 0,
    }
    return bundle, block_table, pd.DataFrame(spectra)


def rotate_dictionary(payload, rotation, name=None):
    """Choose orthonormal coordinates inside an already fitted dictionary."""
    rotation = as_float64(rotation)
    old_dimension = payload["dimension"]
    if (
        rotation.ndim != 2
        or rotation.shape[0] != old_dimension
        or not 0 < rotation.shape[1] <= old_dimension
    ):
        raise ValueError(
            "A nonempty coordinate rotation with compatible rank is required"
        )
    if not np.allclose(
        rotation.T @ rotation, np.eye(rotation.shape[1]), atol=1e-10, rtol=1e-10
    ):
        raise ValueError("Coordinate selection must have orthonormal columns")
    result = {
        key: value
        for key, value in payload.items()
        if key
        not in (
            "layers",
            "dimension",
            "frozen_buffer_bytes",
            "maximum_frobenius_gram_error",
        )
    }
    result.update(arm=name or payload["arm"], dimension=rotation.shape[1], layers={})
    for layer_name, layer in payload["layers"].items():
        selected = {key: value for key, value in layer.items() if key != "directions"}
        selected["directions"] = torch.from_numpy(
            np.tensordot(rotation.T, as_float64(layer["directions"]), axes=1).astype(
                np.float32
            )
        )
        result["layers"][layer_name] = selected
    result["frozen_buffer_bytes"] = dictionary_buffer_bytes(result)
    result["maximum_frobenius_gram_error"] = float(
        np.max(np.abs(dictionary_gram(result) - np.eye(result["dimension"])))
    )
    if result["maximum_frobenius_gram_error"] > 2e-5:
        raise FloatingPointError("The selected dictionary lost its Frobenius metric")
    return result


def truncate_dictionary(payload, dimension):
    if not isinstance(dimension, int) or not 0 < dimension <= payload["dimension"]:
        raise ValueError(
            "Strict dimension matching requires supported nonzero coordinates"
        )
    return rotate_dictionary(payload, np.eye(payload["dimension"], dimension))


def transfer_spectrum(
    bundle, seed=42, arm="OUT", maximum_dimension=8, relative_eigenvalue_tolerance=1e-10
):
    """Compare AUTO/CROSS/RANDOM on the same pairable fit information pool."""
    coordinates = as_float64(bundle["fit_coordinates"][arm])
    protocols = np.asarray(bundle["fit_protocols"], dtype=str)
    payload = bundle["arms"][arm]
    if coordinates.shape != (len(protocols), payload["dimension"]):
        raise ValueError("Fit gradient coordinates and candidate dictionary differ")
    groups = [
        np.flatnonzero(protocols == protocol) for protocol in sorted(set(protocols))
    ]
    pairable = [rows for rows in groups if len(rows) >= 2]
    metadata = {
        "candidate_arm": arm,
        "candidate_dimension": payload["dimension"],
        "requested_dimension": maximum_dimension,
        "pairable_protocol_groups": len(pairable),
        "pairable_physical_cells": sum(len(rows) for rows in pairable),
        "ordered_distinct_cell_pairs": sum(
            len(rows) * (len(rows) - 1) for rows in pairable
        ),
        "pairable_ids": [
            bundle["fit_ids"][index] for rows in pairable for index in rows
        ],
        "excluded_singleton_ids": [
            bundle["fit_ids"][rows[0]] for rows in groups if len(rows) == 1
        ],
        "task_weighting": "pairable_protocol_equal; all_ordered_distinct_single_cell_support_query_pairs",
        "fit_information_pool_matched_for_AUTO_and_CROSS": True,
        "source_checkpoint_identity": bundle["identity"],
        "source_only": True,
        "optimizer_updates": 0,
    }
    if not pairable or not payload["dimension"]:
        metadata.update(
            status="NOT_ESTIMABLE_NO_PAIRABLE_DIRECTION",
            dimension=0,
            dictionaries={},
            rotations={},
        )
        return metadata
    dimension = payload["dimension"]
    auto, cross = np.zeros((dimension, dimension)), np.zeros((dimension, dimension))
    for rows in pairable:
        values = coordinates[rows]
        self_products = values.T @ values
        summed = values.sum(0)
        auto += self_products / len(rows)
        cross += (np.outer(summed, summed) - self_products) / (
            len(rows) * (len(rows) - 1)
        )
    auto, cross = auto / len(pairable), cross / len(pairable)
    auto_values, auto_vectors = np.linalg.eigh((auto + auto.T) * 0.5)
    cross_values, cross_vectors = np.linalg.eigh((cross + cross.T) * 0.5)
    auto_order, cross_order = (
        np.argsort(auto_values)[::-1],
        np.argsort(cross_values)[::-1],
    )
    auto_values, auto_vectors = auto_values[auto_order], auto_vectors[:, auto_order]
    cross_values, cross_vectors = (
        cross_values[cross_order],
        cross_vectors[:, cross_order],
    )
    threshold = max(
        float(np.max(np.abs(cross_values))) * relative_eigenvalue_tolerance,
        np.finfo(np.float64).tiny,
    )
    positive_rank = int(np.count_nonzero(cross_values > threshold))
    effective = min(maximum_dimension, positive_rank, dimension)
    metadata.update(
        positive_cross_rank=positive_rank,
        cross_positive_threshold=threshold,
        cross_eigenvalues=cross_values.tolist(),
        auto_eigenvalues=auto_values.tolist(),
        cross_matrix=cross.tolist(),
        auto_matrix=auto.tolist(),
        dimension=effective,
    )
    if not effective:
        metadata.update(
            status="NOT_ESTIMABLE_NO_POSITIVE_CROSS_DIRECTION",
            dictionaries={},
            rotations={},
        )
        return metadata
    role_seed = int(
        hashlib.sha256(f"{seed}|onesided_transfer_random|{arm}".encode()).hexdigest()[
            :16
        ],
        16,
    )
    random_matrix = np.random.default_rng(role_seed).standard_normal(
        (dimension, effective)
    )
    random_rotation, triangular = np.linalg.qr(random_matrix, mode="reduced")
    random_rotation *= np.where(np.diag(triangular) < 0, -1.0, 1.0)
    rotations = {
        "AUTO": auto_vectors[:, :effective],
        "CROSS": cross_vectors[:, :effective],
        "RANDOM": random_rotation,
    }
    metadata.update(
        status="ESTIMABLE",
        rotations={name: torch.from_numpy(value) for name, value in rotations.items()},
        dictionaries={
            name: rotate_dictionary(payload, value, name)
            for name, value in rotations.items()
        },
    )
    return metadata


def write_dictionary_artifacts(directory, bundle, block_table, spectra):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "DICTIONARIES_COMPLETE.json"
    if marker.exists():
        receipt = json.loads(marker.read_text(encoding="utf-8"))
        if (
            receipt["identity"] != bundle["identity"]
            or receipt["fit_ids"] != bundle["fit_ids"]
            or receipt["requested_dimension"] != bundle["requested_dimension"]
            or receipt["common_dimension"] != bundle["common_dimension"]
            or receipt["weighting"] != bundle["weighting"]
            or receipt["relative_rank_tolerance"] != bundle["relative_rank_tolerance"]
            or receipt["absolute_energy_fraction"] != bundle["absolute_energy_fraction"]
            or receipt["implementation_sha256"] != file_hash(__file__)
        ):
            raise RuntimeError(
                "A completed dictionary belongs to another frozen construction"
            )
        for record in receipt["files"]:
            if file_hash(directory / record["path"]) != record["sha256"]:
                raise RuntimeError("A completed dictionary artifact changed")
        return receipt
    temporary = directory / "DICTIONARIES.pending.pt"
    torch.save(bundle, temporary)
    temporary.replace(directory / "DICTIONARIES.pt")
    block_table.to_csv(directory / "SAME_POINT_GRADIENT_BLOCKS.csv", index=False)
    spectra.to_csv(directory / "DICTIONARY_SPECTRA.csv", index=False)
    groups = block_table.groupby(["layer", "protocol_uid"], as_index=False).mean(
        numeric_only=True
    )
    counts = block_table.groupby(["layer", "protocol_uid"], as_index=False).agg(
        physical_cells=("cell_uid", "nunique"),
        zero_gradient_cells=("zero_gradient", "sum"),
    )
    groups = groups.merge(counts, on=["layer", "protocol_uid"], validate="one_to_one")
    groups.to_csv(directory / "SAME_POINT_PROTOCOL_BLOCKS.csv", index=False)
    budgets = [
        {
            "arm": arm,
            "requested_dimension": payload["requested_dimension"],
            "actual_dimension": payload["dimension"],
            "numerical_rank": payload["numerical_rank"],
            "frozen_dictionary_bytes": payload["frozen_buffer_bytes"],
            "maximum_global_frobenius_gram_error": payload[
                "maximum_frobenius_gram_error"
            ],
        }
        for arm, payload in bundle["arms"].items()
    ]
    pd.DataFrame(budgets).to_csv(
        directory / "ADAPTER_DICTIONARY_BUDGETS.csv", index=False
    )
    receipt = {
        "status": "COMPLETE_SAME_POINT_FIT_DICTIONARIES",
        "identity": bundle["identity"],
        "fit_ids": bundle["fit_ids"],
        "requested_dimension": bundle["requested_dimension"],
        "common_dimension": bundle["common_dimension"],
        "weighting": bundle["weighting"],
        "relative_rank_tolerance": bundle["relative_rank_tolerance"],
        "absolute_energy_fraction": bundle["absolute_energy_fraction"],
        "implementation_sha256": file_hash(__file__),
        "construction_seconds": bundle["construction_seconds"],
        "optimizer_updates": 0,
        "files": [
            {"path": name, "sha256": file_hash(directory / name)}
            for name in (
                "DICTIONARIES.pt",
                "SAME_POINT_GRADIENT_BLOCKS.csv",
                "SAME_POINT_PROTOCOL_BLOCKS.csv",
                "DICTIONARY_SPECTRA.csv",
                "ADAPTER_DICTIONARY_BUDGETS.csv",
            )
        ],
    }
    write_json(marker, receipt)
    return receipt


def _self_test():
    generator = np.random.default_rng(7)
    ids = [f"cell_{index}" for index in range(12)]
    gradients = {"layer": generator.standard_normal((12, 9, 8))}
    bases = {
        "layer": (
            orthonormal_span(generator.standard_normal((9, 2))),
            orthonormal_span(generator.standard_normal((8, 2))),
        )
    }
    bundle, blocks, _ = build_dictionary_bundle(
        gradients,
        bases,
        ids,
        [str(index // 3) for index in range(12)],
        requested_dimension=4,
    )
    assert bundle["common_dimension"] == 4
    assert blocks.relative_reconstruction_error.max() < 1e-12
    assert transfer_spectrum(bundle, maximum_dimension=2)["dimension"] <= 2
    print(
        json.dumps(
            {"status": "PASSED_SYNTHETIC_GEOMETRY_SELF_TEST", "optimizer_updates": 0}
        )
    )


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        _self_test()
