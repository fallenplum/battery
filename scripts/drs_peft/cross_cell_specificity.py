"""Posthoc D1 cross-cell geometry from immutable gradient arrays, CPU only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .data import ROOT, file_hash, write_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def unit_gradients(path):
    values = np.array(np.load(path, mmap_mode="r"), dtype=np.float64, copy=True)
    values = values.reshape(len(values), -1)
    norms = np.linalg.norm(values, axis=1)
    nonzero = norms > 0
    values[nonzero] /= norms[nonzero, None]
    return values, nonzero


def gram_statistics(values, nonzero, protocols=None):
    supported = values[nonzero]
    gram = supported @ supported.T
    eigenvalues = np.maximum(np.linalg.eigvalsh((gram + gram.T) * 0.5), 0)
    mass = eigenvalues / eigenvalues.sum()
    positive = mass > 0
    offdiag = np.abs(gram[~np.eye(len(gram), dtype=bool)])
    result = {
        "physical_cells": len(values),
        "nonzero_gradient_cells": int(nonzero.sum()),
        "mean_cross_cell_absolute_cosine": float(offdiag.mean()),
        "median_cross_cell_absolute_cosine": float(np.median(offdiag)),
        "p05_cross_cell_absolute_cosine": float(np.quantile(offdiag, 0.05)),
        "p95_cross_cell_absolute_cosine": float(np.quantile(offdiag, 0.95)),
        "participation_effective_rank": float(
            eigenvalues.sum() ** 2 / np.square(eigenvalues).sum()
        ),
        "entropy_effective_rank": float(
            np.exp(-np.sum(mass[positive] * np.log(mass[positive])))
        ),
        "top1_normalized_vector_covariance_energy": float(
            eigenvalues[-1] / eigenvalues.sum()
        ),
    }
    if protocols is not None:
        protocols = np.asarray(protocols)[nonzero]
        upper = np.triu(np.ones_like(gram, dtype=bool), k=1)
        same = protocols[:, None] == protocols[None]
        for label, mask in (
            ("within_protocol", upper & same),
            ("cross_protocol", upper & ~same),
        ):
            pairs = np.abs(gram[mask])
            result[label + "_unordered_pairs"] = len(pairs)
            result[label + "_absolute_cosine_mean"] = (
                float(pairs.mean()) if len(pairs) else None
            )
            result[label + "_absolute_cosine_min"] = (
                float(pairs.min()) if len(pairs) else None
            )
            result[label + "_absolute_cosine_max"] = (
                float(pairs.max()) if len(pairs) else None
            )
    return gram, result


def load_sources(root, seed):
    p1 = read_json(ROOT / f"runtime/drs_peft/P1/seed{seed}/P1_CACHE_MANIFEST.json")
    old = {
        variant: read_json(Path(p1[role]["folder"]) / "COMPLETE.json")
        for variant, role in (
            ("true_full", "source/nominal"),
            ("shuffled_full", "source_shuffled/nominal"),
        )
    }
    new = read_json(root / f"seed{seed}/D1_NEW_CACHE_MANIFEST.json")
    return {**old, **new}


def run(root):
    output = root / "D1_CROSS_CELL_GEOMETRY.csv"
    records, inputs = [], []
    metadata_path = ROOT / "runtime/drs_peft/data_v1/MATR/cell_metadata.csv"
    metadata = pd.read_csv(
        metadata_path, usecols=["cell_uid", "protocol_uid"]
    ).set_index("cell_uid")
    with threadpool_limits(limits=2):
        for seed in (42, 123, 2026):
            receipts = load_sources(root, seed)
            ids = receipts["true_full"]["identity"]["ids"]
            protocols = metadata.loc[ids, "protocol_uid"].to_numpy(str)
            for layer in receipts["true_full"]["layer_files"]:
                paths = {}
                for variant, receipt in receipts.items():
                    record = receipt["layer_files"][layer]
                    path = Path(receipt["folder"]) / record["path"]
                    if file_hash(path) != record["sha256"]:
                        raise RuntimeError(
                            "D1 cross-cell analysis encountered changed gradient bytes"
                        )
                    paths[variant] = path
                    inputs.append(
                        {
                            "seed": seed,
                            "variant": variant,
                            "layer": layer,
                            "path": str(path),
                            "sha256": record["sha256"],
                        }
                    )
                truth, truth_nonzero = unit_gradients(paths["true_full"])
                truth_gram, _ = gram_statistics(truth, truth_nonzero)
                for variant, path in paths.items():
                    values, nonzero = (
                        (truth, truth_nonzero)
                        if variant == "true_full"
                        else unit_gradients(path)
                    )
                    gram, stats = gram_statistics(values, nonzero, protocols)
                    cross = truth @ values.T
                    alignment = float(
                        np.square(cross).sum()
                        / np.sqrt(np.square(truth_gram).sum() * np.square(gram).sum())
                    )
                    records.append(
                        {
                            "seed": seed,
                            "layer": layer,
                            "supervision": variant,
                            **stats,
                            "normalized_vector_covariance_alignment_to_true": alignment,
                        }
                    )
                print(
                    json.dumps(
                        {"event": "D1_CROSS_CELL_LAYER", "seed": seed, "layer": layer}
                    ),
                    flush=True,
                )
    frame = pd.DataFrame(records)
    frame.to_csv(output, index=False)
    summary = frame.groupby(["seed", "supervision"], as_index=False).mean(
        numeric_only=True
    )
    summary_path = root / "D1_CROSS_CELL_GEOMETRY_SUMMARY.csv"
    summary.to_csv(summary_path, index=False)
    write_json(
        root / "D1_CROSS_CELL_MANIFEST.json",
        {
            "status": "COMPLETE_POSTHOC_CACHED_SOURCE_DIAGNOSTIC",
            "implementation_sha256": file_hash(__file__),
            "protocol_metadata": {
                "path": str(metadata_path),
                "sha256": file_hash(metadata_path),
            },
            "inputs": inputs,
            "outputs": [
                {"path": str(path.resolve()), "sha256": file_hash(path)}
                for path in (output, summary_path)
            ],
            "new_model_forwards": 0,
            "new_model_autograd_evaluations": 0,
            "optimizer_updates": 0,
            "final_target_Query_future_labels_accessed": False,
            "description": "Per-layer cosine Gram across 85 unit-norm physical-cell gradients. Participation rank is tr(K)^2/tr(K^2); entropy rank is exp(-sum(p log p)). Covariance alignment uses squared Frobenius cross-Gram and is invariant to each cell's global gradient sign.",
            "exploratory_trigger": "Added after the completed same-cell label-replacement diagnostic showed near-collinear gradients; this is a posthoc mechanism check, independent of the formal D1 collection manifest",
            "pair_dependence": "Within/cross-protocol pairs share physical cells; pair counts are descriptive and no pair-level confidence interval is computed",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "results/drs_peft/diagnostics_v1/D1_specificity",
    )
    args = parser.parse_args()
    run(args.root)


if __name__ == "__main__":
    main()
