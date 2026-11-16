"""P1 source-only gradient diagnostics with reproducible covariance subspaces.

This module performs zero optimizer updates. Full per-cell gradient arrays live
in runtime caches. Published outputs contain statistics, compact bases and cache
checksums. Final target Query labels are never indexed by the audit.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits
import torch
from torch import nn

from models.drs_peft.peft import random_orthogonal_bases
from .data import ROOT, file_hash, write_json


EARLY_KEYS = ("x", "point_mask", "cycle_ids", "policy", "policy_mask", "anchor")
DEFAULTS = dict(
    ranks=[2, 4, 8, 16, 40, 56, 80],
    primary_rank=4,
    random_replicates=20,
    bootstrap_draws=2000,
    basis_bootstrap_draws=20,
    analysis_threads=2,
    absolute_control=True,
    shuffle_control=True,
    target_names=["q_proj", "v_proj"],
    expected_target_count=12,
    minimum_capture_gap=0.02,
    minimum_stability=0.5,
    candidate_family_diagnostic=False,
)


def _json_hash(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        ).encode()
    ).hexdigest()


def _seed(seed, role):
    return int.from_bytes(
        hashlib.sha256(f"{seed}/P1/{role}".encode()).digest()[:8], "big"
    ) % (2**31 - 1)


def state_fingerprint(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if not torch.is_tensor(value):
            raise TypeError(
                "P1 state identity requires tensor-valued state_dict entries"
            )
        value = value.detach().cpu().contiguous()
        for payload in (
            name.encode(),
            str(value.dtype).encode(),
            str(tuple(value.shape)).encode(),
            value.reshape(-1).view(torch.uint8).numpy().tobytes(),
        ):
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
    return digest.hexdigest()


def _implementation_fingerprints(model):
    records = {"P1_gradient_audit": file_hash(__file__)}
    for cls in {type(module) for module in model.modules()}:
        try:
            filename = inspect.getsourcefile(cls)
        except TypeError:
            filename = None
        if filename is not None and Path(filename).is_file():
            records[f"{cls.__module__}.{cls.__qualname__}"] = file_hash(filename)
    return dict(sorted(records.items()))


def _rows(data, value):
    values = np.asarray(value)
    rows = (
        data.rows(values.tolist())
        if values.dtype.kind in ("U", "S", "O")
        else values.astype(np.int64)
    )
    if (
        len(rows) == 0
        or len(set(rows.tolist())) != len(rows)
        or (rows < 0).any()
        or (rows >= len(data.ids)).any()
    ):
        raise ValueError("P1 roles require nonempty unique valid physical-cell rows")
    return rows


def _role_fingerprint(data, early, rows, donors):
    digest = hashlib.sha256()
    arrays = {key: np.asarray(early[key])[rows] for key in EARLY_KEYS}
    arrays.update(
        target=data.targets[donors], observed=data.observed[donors], query=data.query
    )
    for key, value in arrays.items():
        value = np.ascontiguousarray(value)
        for payload in (
            key.encode(),
            str(value.dtype).encode(),
            str(value.shape).encode(),
            value.tobytes(),
        ):
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
    digest.update(
        _json_hash(
            dict(ids=[data.ids[i] for i in rows], donors=[data.ids[i] for i in donors])
        ).encode()
    )
    return digest.hexdigest()


@contextmanager
def _gradient_scope(model, parameters):
    flags = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    training = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        model.requires_grad_(False)
        for parameter in parameters:
            parameter.requires_grad_(True)
        yield
    finally:
        for parameter, flag in flags:
            parameter.requires_grad_(flag)
        for module, flag in training:
            module.training = flag


def _batch(early, row, device, dtype, nominal, variant):
    result = {}
    for key in EARLY_KEYS:
        array = np.asarray(early[key])[row : row + 1]
        value = torch.as_tensor(array, device=device)
        result[key] = value.to(dtype=dtype) if value.is_floating_point() else value
    if variant == "absolute_units":
        result["x"] = result["x"].clone()
        result["x"][..., 1:] *= float(nominal)
    return result


def _collect(
    model,
    data,
    early,
    rows,
    donors,
    layers,
    cache,
    seed,
    role,
    variant,
    device,
    state_sha,
):
    dtype = next(iter(layers.values())).weight.dtype
    if dtype not in (torch.float32, torch.float64):
        raise ValueError(
            "P1 requires an explicitly registered float32/float64 model copy"
        )
    device_name = (
        torch.cuda.get_device_name(torch.device(device))
        if str(device).startswith("cuda")
        else "cpu"
    )
    identity = dict(
        seed=int(seed),
        role=role,
        variant=variant,
        model_sha256=state_sha,
        source_data_sha256=_role_fingerprint(data, early, rows, donors),
        ids=[data.ids[i] for i in rows],
        outcome_donors=[data.ids[i] for i in donors],
        layers={name: list(layer.weight.shape) for name, layer in layers.items()},
        torch_version=torch.__version__,
        device=device_name,
        parameter_dtype=str(dtype),
        implementation_sha256=_implementation_fingerprints(model),
    )
    folder = cache / (role + "_" + variant + "_" + _json_hash(identity)[:16])
    marker = folder / "COMPLETE.json"
    if marker.exists():
        old = json.loads(marker.read_text(encoding="utf-8"))
        if old["identity"] != identity:
            raise RuntimeError("P1 gradient cache identity changed")
        for record in old["files"]:
            if file_hash(folder / record["path"]) != record["sha256"]:
                raise RuntimeError("P1 cached gradient bytes changed")
        return {
            name: np.load(folder / record["path"], mmap_mode="r")
            for name, record in old["layer_files"].items()
        }, old
    folder.mkdir(parents=True, exist_ok=True)
    storage_dtype = "float64" if dtype == torch.float64 else "float32"
    filenames = {name: f"layer_{i:02d}.npy" for i, name in enumerate(layers)}
    arrays = {
        name: np.lib.format.open_memmap(
            folder / filenames[name],
            mode="w+",
            dtype=storage_dtype,
            shape=(len(rows), *layer.weight.shape),
        )
        for name, layer in layers.items()
    }
    values = []
    parameters = [layer.weight for layer in layers.values()]
    query = torch.as_tensor(data.query, device=device, dtype=dtype)
    with _gradient_scope(model, parameters):
        for position, (row, donor) in enumerate(zip(rows, donors)):
            mask = torch.as_tensor(data.observed[donor : donor + 1], device=device)
            if not bool(mask.any()):
                raise ValueError(
                    "A selected source/meta-validation cell has no future support"
                )
            target = torch.as_tensor(
                data.targets[donor : donor + 1], device=device, dtype=dtype
            )
            inputs = _batch(
                early, int(row), device, dtype, data.nominal_capacity_Ah[row], variant
            )
            with torch.autocast(device_type=torch.device(device).type, enabled=False):
                prediction = model(inputs, query)
                if prediction.shape != target.shape:
                    raise ValueError(
                        "P1 model output and fixed qualified query grid differ"
                    )
                loss = (prediction[mask] - target[mask]).abs().mean()
                gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("P1 loss is nonfinite")
            for (name, layer), gradient in zip(layers.items(), gradients):
                supported = gradient is not None
                gradient = (
                    torch.zeros_like(layer.weight) if gradient is None else gradient
                )
                if not bool(torch.isfinite(gradient).all()):
                    raise FloatingPointError("P1 gradient is nonfinite")
                arrays[name][position] = gradient.detach().cpu().numpy()
                values.append(
                    dict(
                        phase="P1",
                        source_seed=seed,
                        role=role,
                        variant=variant,
                        layer=name,
                        cell_uid=data.ids[row],
                        outcome_donor_uid=data.ids[donor],
                        protocol_uid=str(data.protocol_ids[row]),
                        observed_points=int(mask.sum().cpu()),
                        loss=float(loss.detach().cpu()),
                        gradient_norm=float(gradient.detach().norm().cpu()),
                        connected=supported,
                        zero_gradient=bool((gradient == 0).all()),
                    )
                )
            if (position + 1) % 10 == 0 or position + 1 == len(rows):
                print(
                    json.dumps(
                        dict(
                            event="P1_CELL_GRADIENTS",
                            role=role,
                            variant=variant,
                            seed=seed,
                            completed=position + 1,
                            total=len(rows),
                        )
                    ),
                    flush=True,
                )
    for array in arrays.values():
        array.flush()
    pd.DataFrame(values).to_csv(folder / "cell_gradient_checks.csv", index=False)
    records = [
        dict(
            path=name,
            sha256=file_hash(folder / name),
            bytes=(folder / name).stat().st_size,
        )
        for name in [*filenames.values(), "cell_gradient_checks.csv"]
    ]
    receipt = dict(
        status="COMPLETE_SOURCE_GRADIENT_DIAGNOSTIC",
        identity=identity,
        optimizer_updates=0,
        layer_files={
            name: next(record for record in records if record["path"] == filename)
            for name, filename in filenames.items()
        },
        files=records,
        folder=str(folder.resolve()),
    )
    write_json(marker, receipt)
    return arrays, receipt


def _replay_gradients(model, data, early, definitions, gradients, layers, device):
    """Recompute two endpoint cells per registered role after the main pass."""
    dtype = next(iter(layers.values())).weight.dtype
    parameters = [layer.weight for layer in layers.values()]
    query = torch.as_tensor(data.query, device=device, dtype=dtype)
    records = []
    with _gradient_scope(model, parameters):
        for role, variant, rows, donors in definitions:
            for position in sorted({0, len(rows) - 1}):
                row, donor = int(rows[position]), int(donors[position])
                mask = torch.as_tensor(data.observed[donor : donor + 1], device=device)
                target = torch.as_tensor(
                    data.targets[donor : donor + 1], device=device, dtype=dtype
                )
                inputs = _batch(
                    early, row, device, dtype, data.nominal_capacity_Ah[row], variant
                )
                with torch.autocast(
                    device_type=torch.device(device).type, enabled=False
                ):
                    prediction = model(inputs, query)
                    loss = (prediction[mask] - target[mask]).abs().mean()
                    replay = torch.autograd.grad(loss, parameters, allow_unused=True)
                for (name, layer), gradient in zip(layers.items(), replay):
                    current = (
                        torch.zeros_like(layer.weight) if gradient is None else gradient
                    )
                    current = current.detach().cpu().numpy()
                    saved = np.asarray(gradients[(role, variant)][name][position])
                    absolute = np.abs(current - saved)
                    denominator = np.maximum(
                        np.maximum(np.abs(current), np.abs(saved)), 1e-12
                    )
                    passed = bool(np.allclose(current, saved, rtol=1e-5, atol=1e-8))
                    records.append(
                        dict(
                            role=role,
                            variant=variant,
                            cell_uid=data.ids[row],
                            layer=name,
                            max_absolute_error=float(absolute.max()),
                            max_relative_error=float((absolute / denominator).max()),
                            passed=passed,
                        )
                    )
                    if not passed:
                        raise RuntimeError(
                            f"P1 endpoint gradient replay failed for {role}/{variant}/{name}"
                        )
    return dict(
        status="PASSED",
        scope="first and last physical cell of every collected role",
        rtol=1e-5,
        atol=1e-8,
        compared_layer_gradients=len(records),
        replay_physical_cells=sum(
            len({0, len(rows) - 1}) for _, _, rows, _ in definitions
        ),
        records=records,
    )


def _group_indices(protocols):
    return {
        str(group): np.flatnonzero(protocols == group) for group in np.unique(protocols)
    }


def _group_balanced_weights(groups, n):
    weight = np.zeros(n, float)
    for positions in groups.values():
        weight[positions] = 1 / (len(groups) * len(positions))
    return weight


def _candidate_family_diagnostic(
    model, data, early, source_rows, meta_rows, output, seed, device
):
    """Stream the registered source12/meta8 secondary QKVO/FFN diagnostic."""
    source = sorted(source_rows, key=lambda row: data.ids[row])[:12]
    meta = sorted(meta_rows, key=lambda row: data.ids[row])[:8]
    if len(source) != 12 or len(meta) != 8:
        raise ValueError(
            "The registered candidate-family diagnostic requires source12/meta8"
        )
    layers = model.bridge.backbone.linear_targets("qkvo_ffn")
    parameters = [layer.weight for layer in layers.values()]
    dtype = parameters[0].dtype
    query = torch.as_tensor(data.query, device=device, dtype=dtype)
    means = {
        name: np.zeros(tuple(layer.weight.shape), np.float64)
        for name, layer in layers.items()
    }
    checks = []
    source_state = state_fingerprint(model)

    def gradients(row, role):
        mask = torch.as_tensor(data.observed[row : row + 1], device=device)
        if not bool(mask.any()):
            raise ValueError(
                "Candidate diagnostic cell has no source/meta future support"
            )
        target = torch.as_tensor(
            data.targets[row : row + 1], device=device, dtype=dtype
        )
        inputs = _batch(
            early, int(row), device, dtype, data.nominal_capacity_Ah[row], "nominal"
        )
        with torch.autocast(device_type=torch.device(device).type, enabled=False):
            prediction = model(inputs, query)
            loss = (prediction[mask] - target[mask]).abs().mean()
            values = torch.autograd.grad(loss, parameters, allow_unused=True)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Candidate diagnostic loss is nonfinite")
        result = {}
        for (name, layer), value in zip(layers.items(), values):
            connected = value is not None
            value = torch.zeros_like(layer.weight) if value is None else value
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError("Candidate diagnostic gradient is nonfinite")
            array = value.detach().cpu().double().numpy()
            result[name] = array
            checks.append(
                dict(
                    seed=seed,
                    role=role,
                    cell_uid=data.ids[row],
                    protocol_uid=str(data.protocol_ids[row]),
                    layer=name,
                    connected=connected,
                    gradient_norm=float(np.linalg.norm(array)),
                    observed_points=int(mask.sum().cpu()),
                )
            )
        return result

    with _gradient_scope(model, parameters), threadpool_limits(limits=2):
        for row in source:
            for name, value in gradients(row, "source12").items():
                means[name] += value / len(source)
        bases, spectra = {}, []
        for name, mean in means.items():
            ranks = [rank for rank in (2, 4, 8, 16) if rank <= min(mean.shape)]
            u, v, singular = _svd(mean, max(ranks))
            bases[name] = (u, v, ranks)
            denominator = float(np.square(singular).sum())
            cumulative = np.cumsum(np.square(singular))
            for index, value in enumerate(singular):
                spectra.append(
                    dict(
                        seed=seed,
                        layer=name,
                        out_features=mean.shape[0],
                        in_features=mean.shape[1],
                        index=index + 1,
                        singular_value=float(value),
                        cumulative_energy=float(cumulative[index] / denominator)
                        if denominator > 0
                        else None,
                        mean_gradient_norm=float(np.linalg.norm(mean)),
                    )
                )
        captures = []
        for row in meta:
            for name, value in gradients(row, "meta8").items():
                u, v, ranks = bases[name]
                for rank, capture in _capture(value, u, v, ranks).items():
                    captures.append(
                        dict(
                            seed=seed,
                            cell_uid=data.ids[row],
                            protocol_uid=str(data.protocol_ids[row]),
                            layer=name,
                            rank=rank,
                            capture=capture,
                            gradient_norm=float(np.linalg.norm(value)),
                            statistic="secondary_individual_heldout_capture",
                            basis="source12_mean_SVD",
                        )
                    )
    if state_fingerprint(model) != source_state:
        raise RuntimeError("Candidate diagnostic changed the source model state")
    pd.DataFrame(spectra).to_csv(output / "P1_CANDIDATE_SPECTRA.csv", index=False)
    pd.DataFrame(captures).to_csv(output / "P1_CANDIDATE_CAPTURE.csv", index=False)
    pd.DataFrame(checks).to_csv(output / "P1_CANDIDATE_CELL_CHECKS.csv", index=False)
    torch.save(
        dict(
            seed=seed,
            source_ids=[data.ids[row] for row in source],
            source_checkpoint_sha256=source_state,
            means={
                name: torch.from_numpy(value.astype(np.float32))
                for name, value in means.items()
            },
            bases={
                name: (
                    torch.from_numpy(u.astype(np.float32)),
                    torch.from_numpy(v.astype(np.float32)),
                )
                for name, (u, v, _) in bases.items()
            },
        ),
        output / "P1_CANDIDATE_MEANS.pt",
    )
    record = dict(
        status="COMPLETE_SECONDARY_DESCRIPTIVE_DIAGNOSTIC",
        family="qkvo_ffn",
        seed=seed,
        source_ids=[data.ids[row] for row in source],
        meta_validation_ids=[data.ids[row] for row in meta],
        target_matrices=len(layers),
        extra_gradient_forward_backward_evaluations=len(source) + len(meta),
        source_state_sha256=source_state,
        optimizer_updates=0,
        target_query_labels_used=0,
        inferential_scope="secondary mean-SVD spectrum and individual heldout capture; no covariance/bootstrap/family selection",
        main_QV_target_set_changed=False,
        rank_or_target_selection_permitted=False,
        source_mean_weighting="uniform physical cells among the twelve lexicographically first source IDs",
        tensors=dict(
            path="P1_CANDIDATE_MEANS.pt",
            sha256=file_hash(output / "P1_CANDIDATE_MEANS.pt"),
        ),
    )
    write_json(output / "P1_CANDIDATE_MANIFEST.json", record)
    return record


def _covariances(gradients, weights):
    left = np.matmul(gradients, gradients.transpose(0, 2, 1))
    right = np.matmul(gradients.transpose(0, 2, 1), gradients)
    return (
        np.tensordot(weights, left, axes=1),
        np.tensordot(weights, right, axes=1),
        left,
        right,
    )


def _top(covariance, rank):
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = eigh(
        covariance,
        subset_by_index=[len(covariance) - rank, len(covariance) - 1],
        check_finite=False,
    )
    return eigenvectors[:, ::-1].copy(), np.maximum(eigenvalues[::-1], 0)


def _svd(matrix, rank):
    u, s, vh = np.linalg.svd(matrix, full_matrices=False)
    return u[:, :rank], vh[:rank].T, s


def _overlap(u, v):
    singular = np.linalg.svd(u.T @ v, compute_uv=False)
    angles = np.arccos(np.clip(singular, 0, 1)) * 180 / np.pi
    return float(np.square(singular).mean()), float(angles.mean()), float(angles.max())


def _hierarchical_weights(groups, n, rng):
    keys = list(groups)
    weights = np.zeros(n, float)
    for index in rng.integers(0, len(keys), len(keys)):
        positions = groups[keys[index]]
        draw = rng.choice(positions, len(positions), replace=True)
        np.add.at(weights, draw, 1 / (len(groups) * len(positions)))
    return weights


def _capture(gradient, u, v, ranks):
    denominator = float(np.square(gradient).sum())
    if denominator == 0:
        return {rank: None for rank in ranks}
    projection = np.square(u.T @ gradient @ v)
    return {rank: float(projection[:rank, :rank].sum() / denominator) for rank in ranks}


def _interval(values, rng, draws):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None, None
    replicates = values[rng.integers(0, len(values), (draws, len(values)))].mean(1)
    return float(np.quantile(replicates, 0.025)), float(np.quantile(replicates, 0.975))


def _plot(output, spectra, capture, stability, seed):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), layout="constrained")
    for name, rows in spectra[spectra.kind == "gradient_mean_singular"].groupby(
        "layer"
    ):
        values = rows.sort_values("index").value.to_numpy()
        total = np.square(values).sum()
        if total > 0:
            axes[0].plot(
                np.arange(1, len(values) + 1),
                np.cumsum(np.square(values)) / total,
                lw=0.7,
                alpha=0.65,
            )
    axes[0].set(
        xlabel="Rank",
        ylabel="Cumulative mean-gradient energy",
        xlim=(1, 80),
        ylim=(0, 1.02),
        title=f"Seed {seed}, {capture.layer.nunique()} Q/V matrices",
    )
    held = capture[capture.statistic == "group_mean"].copy()
    held["family"] = held.method.str.replace(r"random_\d+", "random", regex=True)
    for family in ("grad_cov", "weight_svd", "random", "shuffled_cov"):
        part = held[held.family == family].groupby("rank").capture.mean()
        if len(part):
            axes[1].plot(part.index, part.values, "o-", ms=3, label=family)
    axes[1].set(
        xlabel="Rank",
        ylabel="Held-out source capture",
        ylim=(0, 1.02),
        title=f"{held.protocol_uid.nunique()} meta-validation groups",
    )
    axes[1].legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2)
    grouped = stability.groupby("side").overlap
    for index, (side, values) in enumerate(grouped):
        axes[2].scatter(
            np.full(len(values), index), values, s=5, alpha=0.35, label=side
        )
    axes[2].set(
        xticks=[0, 1],
        xticklabels=["Left", "Right"],
        ylabel="Bootstrap projection overlap",
        ylim=(0, 1.02),
        title="Source hierarchical refits",
    )
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"P1_GRADIENT_SUBSPACE.{suffix}", dpi=300)
    plt.close(fig)


def run_audit(
    model,
    data,
    early_inputs_numpy,
    source_rows,
    validation_rows,
    output_root,
    seed=42,
    device="cuda",
    config=None,
):
    """Compute source/meta diagnostics, keep original weights/flags, return a record.

    The caller loads the P0 B0 checkpoint and the source-only scaler. No target
    Query labels, target adaptation or optimizer steps occur in this function.
    """
    cfg = dict(DEFAULTS)
    cfg.update(config or {})
    if "cache_root" in cfg:
        cfg["cache_root"] = str(cfg["cache_root"])
    for key in (
        "random_replicates",
        "bootstrap_draws",
        "basis_bootstrap_draws",
        "analysis_threads",
    ):
        if int(cfg[key]) < 1:
            raise ValueError(f"P1 {key} must be positive")
    source_rows, validation_rows = (
        _rows(data, source_rows),
        _rows(data, validation_rows),
    )
    if set(source_rows) & set(validation_rows):
        raise ValueError("Source and meta-validation physical cells overlap")
    source_protocol = np.asarray(data.protocol_ids)[source_rows].astype(str)
    meta_protocol = np.asarray(data.protocol_ids)[validation_rows].astype(str)
    if set(source_protocol) & set(meta_protocol):
        raise ValueError(
            "P1 requires exact source/meta-validation protocol disjointness"
        )
    if cfg.get("source_ids") is not None and cfg["source_ids"] != [
        data.ids[i] for i in source_rows
    ]:
        raise ValueError("Source role differs from registered IDs")
    if cfg.get("validation_ids") is not None and cfg["validation_ids"] != [
        data.ids[i] for i in validation_rows
    ]:
        raise ValueError("Meta-validation role differs from registered IDs")
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    names = set(cfg["target_names"])
    layers = {
        name: layer
        for name, layer in model.backbone.named_modules()
        if isinstance(layer, nn.Linear) and name.rsplit(".", 1)[-1] in names
    }
    if len(layers) != cfg["expected_target_count"]:
        raise ValueError(
            f"Expected {cfg['expected_target_count']} clean Q/V matrices, found {len(layers)}"
        )
    ranks = sorted(set(int(rank) for rank in cfg["ranks"]))
    if not ranks or ranks[0] < 1:
        raise ValueError("P1 ranks must be nonempty positive integers")
    max_rank = max(ranks)
    if max_rank > min(min(layer.weight.shape) for layer in layers.values()):
        raise ValueError("A registered rank exceeds an audited target matrix")
    if cfg["primary_rank"] not in ranks:
        raise ValueError("The primary rank must be in the registered list")
    state_before = state_fingerprint(model)
    identity = dict(
        seed=int(seed),
        config=cfg,
        model_state_sha256=state_before,
        source_ids=[data.ids[i] for i in source_rows],
        meta_validation_ids=[data.ids[i] for i in validation_rows],
        source_protocols=sorted(set(source_protocol)),
        meta_validation_protocols=sorted(set(meta_protocol)),
        source_data_sha256=_role_fingerprint(
            data, early_inputs_numpy, source_rows, source_rows
        ),
        meta_data_sha256=_role_fingerprint(
            data, early_inputs_numpy, validation_rows, validation_rows
        ),
        query_grid=data.query.astype(int).tolist(),
        target_Query_future_labels_accessed=False,
        implementation_sha256=_implementation_fingerprints(model),
    )
    registration = output / "P1_REGISTRATION.json"
    if (
        registration.exists()
        and json.loads(registration.read_text(encoding="utf-8")) != identity
    ):
        raise RuntimeError("P1 output identity changed; use a new experiment version")
    write_json(registration, identity)
    marker = output / "COMPLETE.json"
    if marker.exists():
        receipt = json.loads(marker.read_text(encoding="utf-8"))
        for row in receipt["files"]:
            if file_hash(output / row["path"]) != row["sha256"]:
                raise RuntimeError("Completed P1 output changed")
        manifests = json.loads(
            (output / "P1_CACHE_MANIFEST.json").read_text(encoding="utf-8")
        )
        for cache_receipt in manifests.values():
            for row in cache_receipt["files"]:
                if (
                    file_hash(Path(cache_receipt["folder"]) / row["path"])
                    != row["sha256"]
                ):
                    raise RuntimeError("Completed P1 runtime gradient cache changed")
        return receipt
    cache = (
        Path(cfg.get("cache_root", ROOT / "runtime/drs_peft/p1_gradients"))
        / f"seed{seed}"
    )
    cache.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    all_gradients, receipts = {}, {}
    definitions = [
        ("source", "nominal", source_rows, source_rows),
        ("meta_validation", "nominal", validation_rows, validation_rows),
    ]
    if cfg["shuffle_control"]:
        if len(source_rows) < 2:
            raise ValueError("Outcome permutation requires at least two source cells")
        order = np.random.default_rng(
            _seed(seed, "source_outcome_permutation")
        ).permutation(source_rows)
        mapping = dict(zip(order.tolist(), np.roll(order, 1).tolist()))
        donors = np.asarray([mapping[int(row)] for row in source_rows])
        definitions.append(("source_shuffled", "nominal", source_rows, donors))
    if cfg["absolute_control"]:
        definitions.extend(
            [
                ("source", "absolute_units", source_rows, source_rows),
                ("meta_validation", "absolute_units", validation_rows, validation_rows),
            ]
        )
    for role, variant, rows, donors in definitions:
        gradients, receipt = _collect(
            model,
            data,
            early_inputs_numpy,
            rows,
            donors,
            layers,
            cache,
            seed,
            role,
            variant,
            device,
            state_before,
        )
        all_gradients[(role, variant)], receipts[role + "/" + variant] = (
            gradients,
            receipt,
        )
    replay = _replay_gradients(
        model, data, early_inputs_numpy, definitions, all_gradients, layers, device
    )
    write_json(output / "P1_GRADIENT_REPLAY.json", replay)
    if state_fingerprint(model) != state_before:
        raise RuntimeError(
            "Source model parameters or persistent buffers changed during gradient audit"
        )
    source_groups = _group_indices(source_protocol)
    meta_groups = _group_indices(meta_protocol)
    weights = _group_balanced_weights(source_groups, len(source_rows))
    capture_rows, spectrum_rows, group_rows, angle_rows, stability_rows = (
        [],
        [],
        [],
        [],
        [],
    )
    bundle = dict(
        bases={},
        mean_gradients={},
        group_mean_gradients={},
        mean_gradient_weighting="mean_gradients is uniform physical-cell full-future MAE for LoRA-GA; group_mean_gradients and mean-SVD use equal protocols",
        weight_svd_bases={},
        mean_svd_bases={},
        demeaned_bases={},
        label_shuffle_bases={},
        absolute_control_bases={},
        rank=max_rank,
        seed=seed,
        source_ids=identity["source_ids"],
        source_checkpoint_sha256=state_before,
        basis_origin="equal-protocol, equal-cell second-moment row/column gradient energy covariance",
    )
    with threadpool_limits(limits=int(cfg["analysis_threads"])):
        for layer_index, (name, layer) in enumerate(layers.items()):
            source = np.asarray(
                all_gradients[("source", "nominal")][name], dtype=np.float64
            )
            meta = np.asarray(
                all_gradients[("meta_validation", "nominal")][name], dtype=np.float64
            )
            lc, rc, left_cells, right_cells = _covariances(source, weights)
            u, lvalues = _top(lc, max_rank)
            v, rvalues = _top(rc, max_rank)
            group_means = {
                group: source[positions].mean(0)
                for group, positions in source_groups.items()
            }
            global_mean = np.mean(np.stack(list(group_means.values())), axis=0)
            bundle["mean_gradients"][name] = torch.from_numpy(
                source.mean(0).astype(np.float32)
            )
            bundle["group_mean_gradients"][name] = torch.from_numpy(
                global_mean.astype(np.float32)
            )
            um, vm, mean_singular = _svd(global_mean, max_rank)
            demeaned = source.copy()
            for group, positions in source_groups.items():
                demeaned[positions] -= group_means[group]
            ldc, rdc, _, _ = _covariances(demeaned, weights)
            ud, _ = _top(ldc, max_rank)
            vd, _ = _top(rdc, max_rank)
            uw, vw, _ = _svd(layer.weight.detach().cpu().double().numpy(), max_rank)
            bases = {
                "grad_cov": (u, v),
                "weight_svd": (uw, vw),
                "gradient_mean_svd": (um, vm),
                "demeaned_cov": (ud, vd),
            }
            if cfg["shuffle_control"]:
                shuffle = np.asarray(
                    all_gradients[("source_shuffled", "nominal")][name],
                    dtype=np.float64,
                )
                sl, sr, _, _ = _covariances(shuffle, weights)
                us, _ = _top(sl, max_rank)
                vs, _ = _top(sr, max_rank)
                bases["shuffled_cov"] = (us, vs)
                del shuffle, sl, sr
            if cfg["absolute_control"]:
                absolute = np.asarray(
                    all_gradients[("source", "absolute_units")][name], dtype=np.float64
                )
                al, ar, _, _ = _covariances(absolute, weights)
                ua, _ = _top(al, max_rank)
                va, _ = _top(ar, max_rank)
                bases["absolute_cov"] = (ua, va)
                del absolute, al, ar
            for replicate in range(int(cfg["random_replicates"])):
                ur, vr = random_orthogonal_bases(
                    *layer.weight.shape,
                    max_rank,
                    _seed(seed, f"{name}/random/{replicate}"),
                    dtype=torch.float64,
                )
                bases[f"random_{replicate:02d}"] = (ur.numpy(), vr.numpy())
            for key, basis_name in [
                ("bases", "grad_cov"),
                ("weight_svd_bases", "weight_svd"),
                ("mean_svd_bases", "gradient_mean_svd"),
                ("demeaned_bases", "demeaned_cov"),
                ("label_shuffle_bases", "shuffled_cov"),
                ("absolute_control_bases", "absolute_cov"),
            ]:
                if basis_name in bases:
                    bundle[key][name] = tuple(
                        torch.from_numpy(value.astype(np.float32))
                        for value in bases[basis_name]
                    )
            for kind, values in [
                ("gradient_mean_singular", mean_singular),
                (
                    "row_covariance_eigenvalue",
                    np.maximum(eigh(lc, eigvals_only=True)[::-1], 0),
                ),
                (
                    "column_covariance_eigenvalue",
                    np.maximum(eigh(rc, eigvals_only=True)[::-1], 0),
                ),
            ]:
                for position, value in enumerate(values):
                    spectrum_rows.append(
                        dict(
                            phase="P1",
                            seed=seed,
                            layer=name,
                            kind=kind,
                            index=position + 1,
                            value=float(value),
                        )
                    )
            group_bases = {}
            for group, positions in source_groups.items():
                mean = group_means[group]
                if np.linalg.norm(mean) > 0:
                    gu, gv, _ = _svd(mean, int(cfg["primary_rank"]))
                    group_bases[group] = (gu, gv)
                cosines = []
                for position in positions:
                    if len(positions) > 1:
                        other = (mean * len(positions) - source[position]) / (
                            len(positions) - 1
                        )
                        denominator = np.linalg.norm(source[position]) * np.linalg.norm(
                            other
                        )
                        if denominator > 0:
                            cosines.append(
                                float(np.sum(source[position] * other) / denominator)
                            )
                group_rows.append(
                    dict(
                        phase="P1",
                        seed=seed,
                        layer=name,
                        protocol_uid=group,
                        n_cells=len(positions),
                        group_mean_gradient_norm=float(np.linalg.norm(mean)),
                        mean_cell_gradient_norm=float(
                            np.mean(np.linalg.norm(source[positions], axis=(1, 2)))
                        ),
                        leave_one_cell_out_cosine=float(np.mean(cosines))
                        if cosines
                        else None,
                        within_group_variation_supported=len(positions) > 1,
                        global_cancellation_ratio=float(
                            np.linalg.norm(global_mean)
                            / (
                                np.mean(
                                    [np.linalg.norm(g) for g in group_means.values()]
                                )
                                + 1e-300
                            )
                        ),
                    )
                )
            group_names = list(group_bases)
            for i, a in enumerate(group_names):
                for b in group_names[i + 1 :]:
                    for side, position in (("left", 0), ("right", 1)):
                        overlap, angle, maximum = _overlap(
                            group_bases[a][position], group_bases[b][position]
                        )
                        angle_rows.append(
                            dict(
                                phase="P1",
                                seed=seed,
                                layer=name,
                                group_a=a,
                                group_b=b,
                                side=side,
                                rank=cfg["primary_rank"],
                                projection_overlap=overlap,
                                mean_angle_degrees=angle,
                                max_angle_degrees=maximum,
                            )
                        )
            for group, positions in meta_groups.items():
                cases = [("group_mean", None, meta[positions].mean(0))]
                cases.extend(
                    (
                        "individual_cell",
                        data.ids[validation_rows[position]],
                        meta[position],
                    )
                    for position in positions
                )
                for statistic, uid, g in cases:
                    for method, (bu, bv) in bases.items():
                        for rank, value in _capture(g, bu, bv, ranks).items():
                            capture_rows.append(
                                dict(
                                    phase="P1",
                                    seed=seed,
                                    layer=name,
                                    protocol_uid=group,
                                    statistic=statistic,
                                    cell_uid=uid,
                                    group_cells=len(positions),
                                    method=method,
                                    rank=rank,
                                    capture=value,
                                    gradient_norm=float(np.linalg.norm(g)),
                                    zero_gradient=value is None,
                                    evaluated_units="nominal_capacity_normalized",
                                )
                            )
            if cfg["absolute_control"]:
                absmeta = np.asarray(
                    all_gradients[("meta_validation", "absolute_units")][name],
                    dtype=np.float64,
                )
                for group, positions in meta_groups.items():
                    g = absmeta[positions].mean(0)
                    for method in ("grad_cov", "absolute_cov", "weight_svd"):
                        for rank, value in _capture(g, *bases[method], ranks).items():
                            capture_rows.append(
                                dict(
                                    phase="P1",
                                    seed=seed,
                                    layer=name,
                                    protocol_uid=group,
                                    statistic="absolute_unit_group_mean",
                                    cell_uid=None,
                                    group_cells=len(positions),
                                    method=method,
                                    rank=rank,
                                    capture=value,
                                    gradient_norm=float(np.linalg.norm(g)),
                                    zero_gradient=value is None,
                                    evaluated_units="absolute_A_and_Ah_same_checkpoint_counterfactual",
                                )
                            )
                del absmeta
            rng = np.random.default_rng(_seed(seed, name + "/basis_bootstrap"))
            rank = int(cfg["primary_rank"])
            for replicate in range(int(cfg["basis_bootstrap_draws"])):
                w = _hierarchical_weights(source_groups, len(source_rows), rng)
                ub, _ = _top(np.tensordot(w, left_cells, axes=1), rank)
                vb, _ = _top(np.tensordot(w, right_cells, axes=1), rank)
                for side, current, reference in (
                    ("left", ub, u[:, :rank]),
                    ("right", vb, v[:, :rank]),
                ):
                    overlap, angle, maximum = _overlap(current, reference)
                    if np.linalg.norm(lc) == 0 or np.linalg.norm(rc) == 0:
                        overlap = angle = maximum = None
                    stability_rows.append(
                        dict(
                            phase="P1",
                            seed=seed,
                            layer=name,
                            rank=rank,
                            replicate=replicate,
                            side=side,
                            overlap=overlap,
                            mean_angle_degrees=angle,
                            max_angle_degrees=maximum,
                            sampling_unit="protocol_then_independent_cell",
                        )
                    )
            print(
                json.dumps(
                    dict(
                        event="P1_LAYER_ANALYSIS",
                        layer=name,
                        completed=layer_index + 1,
                        total=len(layers),
                    )
                ),
                flush=True,
            )
            del (
                source,
                meta,
                left_cells,
                right_cells,
                demeaned,
                bases,
                group_means,
                group_bases,
            )
    capture = pd.DataFrame(capture_rows)
    spectra = pd.DataFrame(spectrum_rows)
    stability = pd.DataFrame(stability_rows)
    groups = pd.DataFrame(group_rows)
    capture.to_csv(output / "P1_CAPTURE.csv", index=False)
    spectra.to_csv(output / "P1_SPECTRA.csv", index=False)
    groups.to_csv(output / "P1_SOURCE_GROUPS.csv", index=False)
    pd.DataFrame(
        angle_rows,
        columns=[
            "phase",
            "seed",
            "layer",
            "group_a",
            "group_b",
            "side",
            "rank",
            "projection_overlap",
            "mean_angle_degrees",
            "max_angle_degrees",
        ],
    ).to_csv(output / "P1_PRINCIPAL_ANGLES.csv", index=False)
    stability.to_csv(output / "P1_BASIS_STABILITY.csv", index=False)
    torch.save(bundle, output / "P1_BASES.pt")
    checks = []
    for key, receipt in receipts.items():
        cache_checks = pd.read_csv(Path(receipt["folder"]) / "cell_gradient_checks.csv")
        checks.append(cache_checks)
    pd.concat(checks).to_csv(output / "P1_CELL_GRADIENTS.csv", index=False)
    held = capture[
        (capture.statistic == "group_mean") & (capture["rank"] == cfg["primary_rank"])
    ]
    pivot = held.pivot_table(
        index=["protocol_uid", "layer"], columns="method", values="capture"
    )
    random_columns = [name for name in pivot.columns if name.startswith("random_")]
    pivot["random_mean"] = pivot[random_columns].mean(1)
    comparisons = []
    for reference in ("random_mean", "weight_svd", "shuffled_cov"):
        if reference not in pivot:
            continue
        if "grad_cov" not in pivot:
            continue
        differences = (
            (pivot["grad_cov"] - pivot[reference])
            .groupby("protocol_uid")
            .mean()
            .dropna()
        )
        low, high = _interval(
            differences.to_numpy(),
            np.random.default_rng(_seed(seed, "CI/" + reference)),
            int(cfg["bootstrap_draws"]),
        )
        comparisons.append(
            dict(
                phase="P1",
                seed=seed,
                reference=reference,
                candidate="grad_cov",
                rank=cfg["primary_rank"],
                valid_protocol_groups=int(differences.notna().sum()),
                mean_capture_gap=float(differences.mean())
                if len(differences)
                else None,
                paired_protocol_CI_low=low,
                paired_protocol_CI_high=high,
            )
        )
    pd.DataFrame(comparisons).to_csv(output / "P1_CAPTURE_COMPARISONS.csv", index=False)
    valid_stability = stability.overlap.dropna()
    stability_mean = float(valid_stability.mean()) if len(valid_stability) else None
    primary = [
        row for row in comparisons if row["reference"] in ("random_mean", "weight_svd")
    ]
    capture_ok = (
        len(primary) == 2
        and all(
            row["valid_protocol_groups"] >= 2
            and row["mean_capture_gap"] is not None
            and row["mean_capture_gap"] >= cfg["minimum_capture_gap"]
            and row["paired_protocol_CI_low"] is not None
            and row["paired_protocol_CI_low"] > 0
            for row in primary
        )
        and stability_mean is not None
        and stability_mean >= cfg["minimum_stability"]
    )
    nominal_unique = np.unique(
        np.asarray(data.nominal_capacity_Ah)[source_rows]
    ).tolist()
    candidate = (
        _candidate_family_diagnostic(
            model,
            data,
            early_inputs_numpy,
            source_rows,
            validation_rows,
            output,
            seed,
            device,
        )
        if cfg["candidate_family_diagnostic"]
        else None
    )
    result = dict(
        status="COMPLETE_SOURCE_GRADIENT_DIAGNOSTIC",
        seed=int(seed),
        primary_rank=int(cfg["primary_rank"]),
        source_cells=len(source_rows),
        source_groups=len(source_groups),
        meta_validation_cells=len(validation_rows),
        meta_validation_groups=len(meta_groups),
        within_group_variation_supported_source_groups=sum(
            len(rows) > 1 for rows in source_groups.values()
        ),
        meta_validation_groups_with_replicate_cells=sum(
            len(rows) > 1 for rows in meta_groups.values()
        ),
        source_layers_with_nonzero_gradient=int(
            (groups.groupby("layer").mean_cell_gradient_norm.max() > 0).sum()
        ),
        target_matrices=len(layers),
        optimizer_updates=0,
        physical_cell_gradient_evaluations=sum(
            len(receipt["identity"]["ids"]) for receipt in receipts.values()
        ),
        independent_endpoint_replay_layer_gradients=replay["compared_layer_gradients"],
        independent_endpoint_replay_physical_cells=replay["replay_physical_cells"],
        total_gradient_forward_backward_evaluations=sum(
            len(receipt["identity"]["ids"]) for receipt in receipts.values()
        )
        + replay["replay_physical_cells"],
        secondary_candidate_diagnostic=candidate,
        source_nominal_capacities_Ah=nominal_unique,
        capacity_variation_supported=len(nominal_unique) > 1,
        absolute_unit_control_interpretation="same-checkpoint source-only unit perturbation; constant source capacity limits invariance claims",
        source_model_before_sha256=state_before,
        source_model_after_sha256=state_fingerprint(model),
        target_Query_future_labels_accessed=False,
        target_query_labels_used=0,
        information_budget_semantics="cohort files may be loaded in memory; final target future labels never enter gradients, selection, or scores",
        comparisons=comparisons,
        mean_basis_bootstrap_overlap=stability_mean,
        single_seed_capture_criterion_passed=bool(capture_ok),
        H1_status="PARTIAL_SINGLE_SEED_REQUIRES_THREE_SEED_AGGREGATION",
        basis_file="P1_BASES.pt",
        gradient_cache_manifests={
            key: dict(
                folder=receipt["folder"],
                identity=receipt["identity"],
                files=receipt["files"],
            )
            for key, receipt in receipts.items()
        },
        elapsed_seconds=time.monotonic() - started,
    )
    if candidate is not None:
        result["total_gradient_forward_backward_evaluations"] += candidate[
            "extra_gradient_forward_backward_evaluations"
        ]
    if result["source_model_after_sha256"] != state_before:
        raise RuntimeError("Source state changed while analyzing gradients")
    write_json(output / "P1_RESULT.json", result)
    write_json(output / "P1_CACHE_MANIFEST.json", result["gradient_cache_manifests"])
    text = f"""# P1 源退化梯度诊断

本次为seed{seed}的真实源数据诊断，源训练{len(source_rows)}颗、{len(source_groups)}个exact protocol，元验证{len(validation_rows)}颗、{len(meta_groups)}组。全部基只由源训练构造，优化器更新为0。最终目标Query标签未参与梯度、选模或评分计算。

主基由每组等权、组内每cell等权的梯度能量左右协方差构建，另有mean-SVD、组内去均值、W0-SVD、随机正交及源标签打乱控制。完整单cell梯度在gitignored runtime缓存，校验和与身份在P1_CACHE_MANIFEST.json，小型基在P1_BASES.pt。

主rank为{cfg["primary_rank"]}。单seed捕获条件满足状态为{capture_ok}，bootstrap基重合均值为{stability_mean}。H1当前标PARTIAL，正式门槛仍需三source seed汇总。单cell梯度秩受token预算约束，当前谱形与投影不能单独认证电池特异退化机制。

每个收集role的首尾cell在缓存写完后独立重新前后向，逐层比较梯度并保存P1_GRADIENT_REPLAY.json。此次复算覆盖{replay["compared_layer_gradients"]}个层梯度，不将有限复算描述为全量重放。

source checkpoint、head与bridge的before/after指纹相同，临时requires_grad与train/eval标志均已恢复。source名义容量取值为{nominal_unique}。绝对单位控制是在同一checkpoint上的输入单位扰动，容量恒定时缺少跨容量可比性证据。

各层和各group的投影、主角度、逐cell梯度支持及bootstrap在对应CSV中保留。打乱控制若同样捕获方向，应作为个体退化特异性不足的证据。优势为真实梯度、独立源工况留出和多控制，风险为有限组内重复、单seed、固定head与桥接诱发的共同方向。工程身份保持的置信度高，H1的科学支持仍需完整阶段结果。
"""
    (output / "P1_GRADIENT_AUDIT_ZH.md").write_text(text, encoding="utf-8")
    _plot(output, spectra, capture, stability, seed)
    published = [
        path
        for path in output.iterdir()
        if path.is_file()
        and path.name != "COMPLETE.json"
        and not path.name.endswith(".pending")
    ]
    receipt = {
        key: value for key, value in result.items() if key != "gradient_cache_manifests"
    }
    receipt["files"] = [
        dict(path=path.name, sha256=file_hash(path), bytes=path.stat().st_size)
        for path in sorted(published)
    ]
    write_json(marker, receipt)
    return receipt


def aggregate_audits(paths, output_root, config=None):
    """Evaluate the registered three-seed P1 capture gate from actual records."""
    options = dict(
        required_seeds=[42, 123, 2026],
        primary_rank=4,
        bootstrap_draws=2000,
        minimum_capture_gap=0.02,
        minimum_stability=0.5,
        minimum_groups=2,
    )
    options.update(config or {})
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    registrations, results, captures, stability = [], [], [], []
    fingerprints = []
    for path in paths:
        folder = Path(path)
        if folder.is_file():
            folder = folder.parent
        closed = folder / "P1_PUBLIC_COMPLETE.json"
        manifest_key = "public_files"
        if not closed.exists():
            closed = folder / "COMPLETE.json"
            manifest_key = "files"
        manifest = json.loads(closed.read_text(encoding="utf-8"))
        for file in manifest[manifest_key]:
            if file_hash(folder / file["path"]) != file["sha256"]:
                raise RuntimeError(
                    "P1 aggregation input bytes changed after completion"
                )
        registration = json.loads(
            (folder / "P1_REGISTRATION.json").read_text(encoding="utf-8")
        )
        result = json.loads((folder / "P1_RESULT.json").read_text(encoding="utf-8"))
        if result["status"] != "COMPLETE_SOURCE_GRADIENT_DIAGNOSTIC":
            raise ValueError("P1 aggregation requires completed diagnostic records")
        if result["source_model_before_sha256"] != result["source_model_after_sha256"]:
            raise ValueError("P1 source model changed during a seed diagnostic")
        if (
            result["optimizer_updates"] != 0
            or result["target_Query_future_labels_accessed"]
        ):
            raise ValueError(
                "P1 seed violates the zero-update source-only information budget"
            )
        registrations.append(registration)
        results.append(result)
        captures.append(pd.read_csv(folder / "P1_CAPTURE.csv"))
        stability.append(pd.read_csv(folder / "P1_BASIS_STABILITY.csv"))
        fingerprints.append(
            dict(
                seed=result["seed"],
                files=[
                    dict(path=name, sha256=file_hash(folder / name))
                    for name in (
                        "P1_RESULT.json",
                        "P1_REGISTRATION.json",
                        "P1_CAPTURE.csv",
                        "P1_BASIS_STABILITY.csv",
                    )
                ],
            )
        )
    if not results:
        raise ValueError("P1 aggregation requires at least one actual seed diagnostic")
    seeds = [int(record["seed"]) for record in results]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Repeated source seeds cannot count as independent P1 runs")
    if not set(seeds).issubset(options["required_seeds"]):
        raise ValueError("Unregistered source seed supplied to P1 aggregation")
    role_keys = (
        "source_ids",
        "meta_validation_ids",
        "source_protocols",
        "meta_validation_protocols",
        "source_data_sha256",
        "meta_data_sha256",
        "query_grid",
    )
    reference = registrations[0]
    for registration in registrations:
        if any(registration[key] != reference[key] for key in role_keys):
            raise ValueError(
                "P1 seeds differ in physical roles, protocol split, data or query grid"
            )
        if int(registration["config"]["primary_rank"]) != options["primary_rank"]:
            raise ValueError(
                "P1 primary rank differs from the registered aggregate decision"
            )
        if registration["implementation_sha256"] != reference["implementation_sha256"]:
            raise ValueError(
                "P1 seeds were computed with different method implementations"
            )
    combined = pd.concat(captures, ignore_index=True)
    held = combined[
        (combined.statistic == "group_mean")
        & (combined["rank"] == options["primary_rank"])
    ]
    pivot = held.pivot_table(
        index=["seed", "protocol_uid", "layer"], columns="method", values="capture"
    )
    random_columns = [name for name in pivot.columns if name.startswith("random_")]
    if random_columns:
        pivot["random_mean"] = pivot[random_columns].mean(1)
    paired, summaries = [], []
    for control in ("random_mean", "weight_svd", "shuffled_cov"):
        if control not in pivot or "grad_cov" not in pivot:
            continue
        differences = (
            (pivot["grad_cov"] - pivot[control])
            .groupby(["seed", "protocol_uid"])
            .mean()
            .dropna()
        )
        for (seed, group), value in differences.items():
            paired.append(
                dict(
                    seed=int(seed),
                    protocol_uid=group,
                    reference=control,
                    capture_gap=float(value),
                )
            )
        by_seed = differences.groupby("seed").mean()
        by_group = differences.groupby("protocol_uid").mean()
        low, high = _interval(
            by_group.to_numpy(),
            np.random.default_rng(_seed(42, "P1_AGGREGATE/" + control)),
            options["bootstrap_draws"],
        )
        summaries.append(
            dict(
                reference=control,
                rank=options["primary_rank"],
                mean_capture_gap=float(by_group.mean()) if len(by_group) else None,
                paired_protocol_CI_low=low,
                paired_protocol_CI_high=high,
                valid_protocol_groups=len(by_group),
                seed_gaps={
                    str(int(seed)): float(value) for seed, value in by_seed.items()
                },
                all_registered_seed_directions_positive=set(by_seed.index)
                == set(options["required_seeds"])
                and bool((by_seed > 0).all()),
                between_seed_gap_std=float(by_seed.std(ddof=1))
                if len(by_seed) > 1
                else None,
                bootstrap_unit="same exact-protocol group paired across seeds; condition on the fitted source seeds",
            )
        )
    stability_table = pd.concat(stability, ignore_index=True)
    stability_by_seed = stability_table.groupby("seed").overlap.mean()
    finite_stability = stability_by_seed.dropna()
    stability_mean = float(finite_stability.mean()) if len(finite_stability) else None
    primary = [
        row for row in summaries if row["reference"] in ("random_mean", "weight_svd")
    ]
    all_seeds = set(seeds) == set(options["required_seeds"])
    passed = (
        all_seeds
        and len(primary) == 2
        and all(
            row["valid_protocol_groups"] >= options["minimum_groups"]
            and row["mean_capture_gap"] >= options["minimum_capture_gap"]
            and row["paired_protocol_CI_low"] is not None
            and row["paired_protocol_CI_low"] > 0
            and row["all_registered_seed_directions_positive"]
            for row in primary
        )
        and stability_mean is not None
        and stability_mean >= options["minimum_stability"]
    )
    status = "PASSED" if passed else ("NEGATIVE" if all_seeds else "PARTIAL")
    permutation = next(
        (row for row in summaries if row["reference"] == "shuffled_cov"), None
    )
    permutation_weakened = bool(
        permutation is not None
        and permutation["mean_capture_gap"] is not None
        and permutation["mean_capture_gap"] >= options["minimum_capture_gap"]
        and permutation["paired_protocol_CI_low"] is not None
        and permutation["paired_protocol_CI_low"] > 0
        and permutation["all_registered_seed_directions_positive"]
    )
    decision = dict(
        stage="P1",
        status=status,
        H1_status="SHARED_SOURCE_LOW_DIMENSIONAL_CAPTURE_SUPPORTED"
        if passed
        else "H1_NOT_SUPPORTED_BY_COMPLETED_CAPTURE_BRANCH"
        if all_seeds
        else "PARTIAL",
        decision_config=options,
        seeds=sorted(seeds),
        required_seeds=options["required_seeds"],
        comparisons=summaries,
        basis_stability_mean=stability_mean,
        basis_stability_by_seed={
            str(int(seed)): float(value) if np.isfinite(value) else None
            for seed, value in stability_by_seed.items()
        },
        permutation_capture_weakening_observed=permutation_weakened,
        battery_specificity_claim="UNESTABLISHED; label-shuffle weakening alone cannot identify a battery-specific mechanism",
        permutation_interpretation="source labels induce distinguishable directions under this fixed checkpoint"
        if permutation_weakened
        else "the label-shuffle control does not establish outcome-specific source directions",
        heldout_update_branch="NOT_EVALUATED_BY_ZERO_UPDATE_P1",
        P2_minimal_controls_allowed=True,
        P2_parameter_matched_controls_allowed=True,
        P2_additional_rank_curves_allowed=bool(passed),
        P2_large_rank_curve_allowed=bool(passed),
        P3_P4_large_expansion_allowed=bool(passed),
        optimizer_updates=0,
        total_gradient_forward_backward_evaluations=sum(
            record["total_gradient_forward_backward_evaluations"] for record in results
        ),
        target_Query_future_labels_accessed=False,
        inputs=fingerprints,
    )
    write_json(
        output / "P1_AGGREGATE_REGISTRATION.json",
        dict(config=options, inputs=fingerprints),
    )
    pd.DataFrame(
        paired, columns=["seed", "protocol_uid", "reference", "capture_gap"]
    ).to_csv(output / "P1_PAIRED_GROUP_GAPS.csv", index=False)
    write_json(output / "P1_GATE.json", decision)
    text = f"""# P1 三种子门槛审计

源seed为{sorted(seeds)}，门槛状态为**{status}**。主rank{options["primary_rank"]}的捕获差按同一exact protocol跨seed配对，再以protocol为单位进行{options["bootstrap_draws"]}次bootstrap。seed方向、seed间波动和每seed基稳定性另行保留。该区间以本轮已拟合的source seed为条件。

平均bootstrap基重合为{stability_mean}。label-shuffle控制弱化观测为{permutation_weakened}。当前证据限于固定source checkpoint的共享梯度方向，电池特异机制仍未建立。单cell低秩与固定head/bridge共同方向会产生相近谱形。

P2同rank主比较及近实际参数匹配对照均允许继续。额外rank2/8曲线及P3/P4扩展放行状态为{passed}。零更新P1尚未评价真实留出更新收益分支，阴性记录保留该证据边界。最终目标Query标签参与梯度与选模计算为0。
"""
    (output / "P1_AGGREGATE_AUDIT_ZH.md").write_text(text, encoding="utf-8")
    return decision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backbone-dir", type=Path, required=True)
    parser.add_argument(
        "--data-root", type=Path, default=ROOT / "runtime/drs_peft/data_v1"
    )
    parser.add_argument(
        "--episodes", type=Path, default=ROOT / "configs/drs_peft/episodes_v1.json"
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--bridge", choices=["native_multivariate", "cycle_embedding"], required=True
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Completed diagnostic units are verified and reused",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    from .data import load_dataset, SourceOnlyScaler
    from .episodes import load_registration
    from models.drs_peft.trajectory import BatteryTSFM
    from .train import load_selected

    registration = load_registration(args.episodes)
    data = load_dataset("MATR", cache_root=args.data_root, build_if_missing=False)
    source_ids = registration["source_cell_roles"]["source_train"]
    meta_ids = registration["source_cell_roles"]["source_meta_validation"]
    source, meta = data.rows(source_ids), data.rows(meta_ids)
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    config.update(source_ids=source_ids, validation_ids=meta_ids)
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    source_cells=len(source),
                    meta_validation_cells=len(meta),
                    query_points=len(data.query),
                    config=config,
                )
            ),
            flush=True,
        )
        return
    scaler = SourceOnlyScaler().fit(data, source)
    model = BatteryTSFM(
        args.backbone_dir, data.policy.shape[1], bridge=args.bridge, pretrained=True
    ).to(args.device)
    checkpoint = torch.load(
        args.checkpoint, map_location=args.device, weights_only=False
    )
    if "model" not in checkpoint:
        raise ValueError(
            "The CLI requires the registered P0 checkpoint model-state key"
        )
    load_selected(model, checkpoint["model"])
    result = run_audit(
        model,
        data,
        scaler.transform(data),
        source,
        meta,
        args.output_root,
        args.seed,
        args.device,
        config,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "seed",
                    "source_cells",
                    "meta_validation_cells",
                    "single_seed_capture_criterion_passed",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
