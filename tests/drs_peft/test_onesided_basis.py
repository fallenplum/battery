"""Regression checks for global fit-only geometry and transfer spectra."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.drs_peft.onesided_basis import (
    ARMS,
    build_dictionary_bundle,
    dictionary_buffer_bytes,
    dictionary_gram,
    layer_dictionary_gram,
    materialize_block,
    orthonormal_span,
    physical_cell_weights,
    projected_coordinates,
    rotate_dictionary,
    transfer_spectrum,
    truncate_dictionary,
    write_dictionary_artifacts,
)


def example(seed=17, cells=18, layers=2, requested=6):
    rng = np.random.default_rng(seed)
    gradients, bases = {}, {}
    for index in range(layers):
        name = f"layer_{index}"
        gradients[name] = rng.standard_normal((cells, 10, 9))
        bases[name] = (
            orthonormal_span(rng.standard_normal((10, 3))),
            orthonormal_span(rng.standard_normal((9, 3))),
        )
    ids = [f"cell_{index}" for index in range(cells)]
    protocols = [f"protocol_{index // 3}" for index in range(cells)]
    bundle, blocks, spectra = build_dictionary_bundle(
        gradients,
        bases,
        ids,
        protocols,
        requested_dimension=requested,
        identity={"reference_checkpoint_sha256": "same_reference"},
    )
    return bundle, blocks, spectra, gradients, bases


def physical_dictionary(layer):
    directions = layer["directions"].double().numpy()
    left = layer.get("left")
    right = layer.get("right")
    if left is not None:
        directions = left.double().numpy() @ directions
    if right is not None:
        directions = directions @ right.double().numpy().T
    return directions


def test_projection_names_and_all_four_orthogonal_blocks():
    rng = np.random.default_rng(5)
    g = rng.standard_normal((11, 9))
    u = orthonormal_span(rng.standard_normal((11, 3)))
    v = orthonormal_span(rng.standard_normal((9, 3)))
    coordinates = projected_coordinates(g, u, v)
    matrices = {arm: materialize_block(coordinates[arm], arm, u, v) for arm in ARMS}
    pu, pv = u @ u.T, v @ v.T
    np.testing.assert_allclose(matrices["CORE"], pu @ g @ pv, atol=2e-15)
    np.testing.assert_allclose(matrices["OUT"], (np.eye(11) - pu) @ g @ pv, atol=2e-15)
    np.testing.assert_allclose(matrices["IN"], pu @ g @ (np.eye(9) - pv), atol=2e-15)
    np.testing.assert_allclose(
        matrices["NORMAL"], (np.eye(11) - pu) @ g @ (np.eye(9) - pv), atol=2e-15
    )
    np.testing.assert_allclose(sum(matrices.values()), g, atol=2e-15)
    for position, first in enumerate(ARMS):
        for second in ARMS[position + 1 :]:
            assert abs(np.sum(matrices[first] * matrices[second])) < 2e-13


def test_protocol_equal_weighting_preserves_single_physical_cell_units():
    weights = physical_cell_weights(["a", "a", "a", "b", "c", "c"])
    np.testing.assert_allclose(weights, [1 / 9, 1 / 9, 1 / 9, 1 / 3, 1 / 6, 1 / 6])
    np.testing.assert_allclose(
        physical_cell_weights(["a", "a", "b"], "cell_equal"), [1 / 3] * 3
    )
    with pytest.raises(ValueError):
        physical_cell_weights(["a"], "future_life_clusters")


@pytest.mark.parametrize("arm", ARMS)
def test_global_frobenius_dictionary_and_true_coordinate_inner_products(arm):
    bundle, blocks, _, gradients, _ = example()
    payload = bundle["arms"][arm]
    expected_gram = np.zeros((6, 6))
    expected_coordinates = np.zeros((18, 6))
    for name, layer in payload["layers"].items():
        dense = physical_dictionary(layer)
        flat = dense.reshape(6, -1)
        expected_gram += flat @ flat.T
        np.testing.assert_allclose(
            layer_dictionary_gram(layer), flat @ flat.T, atol=1e-14
        )
        expected_coordinates += gradients[name].reshape(18, -1) @ flat.T
    np.testing.assert_allclose(expected_gram, np.eye(6), atol=2e-7)
    np.testing.assert_allclose(dictionary_gram(payload), expected_gram, atol=2e-15)
    np.testing.assert_allclose(
        bundle["fit_coordinates"][arm].numpy(), expected_coordinates, atol=1e-13
    )
    assert blocks.relative_reconstruction_error.max() < 2e-15
    assert blocks.maximum_relative_cross_block_inner_product.max() < 2e-15
    assert payload["frozen_buffer_bytes"] == dictionary_buffer_bytes(payload)


def test_zero_block_has_actual_zero_rank_without_padded_parameters():
    rng = np.random.default_rng(12)
    u = orthonormal_span(rng.standard_normal((9, 2)))
    v = orthonormal_span(rng.standard_normal((8, 2)))
    coordinates = rng.standard_normal((12, 2, 2))
    gradients = {"layer": u @ coordinates @ v.T}
    bundle, _, _ = build_dictionary_bundle(
        gradients,
        {"layer": (u, v)},
        [str(i) for i in range(12)],
        [str(i // 3) for i in range(12)],
        requested_dimension=4,
    )
    assert bundle["arms"]["CORE"]["dimension"] == 4
    assert bundle["common_dimension"] == 0
    for arm in ("OUT", "IN", "NORMAL"):
        assert bundle["arms"][arm]["dimension"] == 0
        assert bundle["arms"][arm]["status"] == "NO_NUMERICAL_DIRECTION"
        assert bundle["fit_coordinates"][arm].shape == (12, 0)


def test_orthogonal_span_repair_preserves_existing_source_column_spaces():
    rng = np.random.default_rng(20)
    basis = orthonormal_span(rng.standard_normal((12, 3)))
    perturbed = basis.astype(np.float32).astype(float)
    repaired = orthonormal_span(perturbed)
    np.testing.assert_allclose(repaired.T @ repaired, np.eye(3), atol=1e-15)
    np.testing.assert_allclose(repaired @ repaired.T @ perturbed, perturbed, atol=1e-15)


def test_global_energy_ratio_uses_layer_energy_before_division():
    bundle, blocks, _, _, _ = example()
    weights = dict(zip(bundle["fit_ids"], bundle["weights"], strict=True))
    total = sum(
        row.gradient_energy * weights[row.cell_uid] for row in blocks.itertuples()
    )
    for arm in ARMS:
        energy = sum(
            getattr(row, arm + "_energy") * weights[row.cell_uid]
            for row in blocks.itertuples()
        )
        assert bundle["global_energy_ratios"][arm] == pytest.approx(energy / total)
    assert sum(bundle["global_energy_ratios"].values()) == pytest.approx(1)


def test_transfer_cross_matrix_is_all_distinct_cell_pairs_and_auto_pool_matches():
    bundle, _, _, _, _ = example(cells=9, requested=4)
    bundle["fit_protocols"] = [
        "a",
        "a",
        "b",
        "b",
        "b",
        "singleton1",
        "singleton2",
        "singleton3",
        "singleton4",
    ]
    coordinates = np.array(
        [
            [1, 0, 0, 0],
            [2, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 2, 0, 0],
            [0, 3, 0, 0],
            [1e4] * 4,
            [2e4] * 4,
            [3e4] * 4,
            [4e4] * 4,
        ],
        dtype=float,
    )
    bundle["fit_coordinates"]["OUT"] = torch.from_numpy(coordinates)
    result = transfer_spectrum(bundle, maximum_dimension=3)
    expected_cross = np.zeros((4, 4))
    expected_auto = np.zeros((4, 4))
    for selected in ([0, 1], [2, 3, 4]):
        expected_auto += (
            sum(np.outer(coordinates[i], coordinates[i]) for i in selected)
            / len(selected)
            / 2
        )
        pairs = [
            np.outer(coordinates[i], coordinates[j])
            for i in selected
            for j in selected
            if i != j
        ]
        expected_cross += sum(pairs) / len(pairs) / 2
    np.testing.assert_allclose(result["auto_matrix"], expected_auto)
    np.testing.assert_allclose(result["cross_matrix"], expected_cross)
    assert result["pairable_protocol_groups"] == 2
    assert result["pairable_physical_cells"] == 5
    assert result["ordered_distinct_cell_pairs"] == 8
    assert result["dimension"] == 2
    assert all(payload["dimension"] == 2 for payload in result["dictionaries"].values())


def test_transfer_does_not_take_absolute_negative_cross_eigenvalues():
    bundle, _, _, _, _ = example(cells=6, requested=3)
    bundle["fit_protocols"] = ["a", "a", "b", "b", "c", "c"]
    bundle["fit_coordinates"]["OUT"] = torch.tensor(
        [
            [1.0, 0, 0],
            [-1.0, 0, 0],
            [0, 1.0, 0],
            [0, -1.0, 0],
            [0, 0, 1.0],
            [0, 0, -1.0],
        ],
        dtype=torch.float64,
    )
    result = transfer_spectrum(bundle)
    assert result["positive_cross_rank"] == 0
    assert result["status"] == "NOT_ESTIMABLE_NO_POSITIVE_CROSS_DIRECTION"
    assert result["dimension"] == 0
    assert result["dictionaries"] == {}


def test_singleton_protocols_leave_cross_unestimable():
    bundle, _, _, _, _ = example(cells=6, requested=3)
    bundle["fit_protocols"] = [str(i) for i in range(6)]
    result = transfer_spectrum(bundle)
    assert result["status"] == "NOT_ESTIMABLE_NO_PAIRABLE_DIRECTION"
    assert result["pairable_physical_cells"] == 0


def test_random_coordinate_control_is_reproducible_and_orthonormal():
    bundle, _, _, _, _ = example()
    first = transfer_spectrum(bundle, seed=42, maximum_dimension=3)
    second = transfer_spectrum(bundle, seed=42, maximum_dimension=3)
    torch.testing.assert_close(
        first["rotations"]["RANDOM"], second["rotations"]["RANDOM"], rtol=0, atol=0
    )
    rotation = first["rotations"]["RANDOM"].numpy()
    np.testing.assert_allclose(
        rotation.T @ rotation, np.eye(first["dimension"]), atol=1e-14
    )


def test_strict_common_dimension_and_rotation_reject_invalid_coordinates():
    bundle, _, _, _, _ = example()
    payload = truncate_dictionary(bundle["arms"]["OUT"], 2)
    assert payload["dimension"] == 2
    np.testing.assert_allclose(dictionary_gram(payload), np.eye(2), atol=1e-7)
    with pytest.raises(ValueError):
        truncate_dictionary(payload, 0)
    with pytest.raises(ValueError):
        rotate_dictionary(payload, np.ones((2, 2)))


def test_artifacts_keep_per_cell_and_per_protocol_support_counts(tmp_path):
    bundle, blocks, spectra, _, _ = example()
    receipt = write_dictionary_artifacts(tmp_path, bundle, blocks, spectra)
    assert receipt["status"] == "COMPLETE_SAME_POINT_FIT_DICTIONARIES"
    saved = torch.load(tmp_path / "DICTIONARIES.pt", weights_only=False)
    assert saved["identity"] == {"reference_checkpoint_sha256": "same_reference"}
    record = json.loads(
        (tmp_path / "DICTIONARIES_COMPLETE.json").read_text(encoding="utf-8")
    )
    assert len(record["files"]) == 5
    table = pd.read_csv(tmp_path / "SAME_POINT_PROTOCOL_BLOCKS.csv")
    assert table.physical_cells.eq(3).all()
    before = {
        record["path"]: (tmp_path / record["path"]).read_bytes()
        for record in receipt["files"]
    }
    replay = write_dictionary_artifacts(tmp_path, bundle, blocks, spectra)
    assert replay == receipt
    assert all(
        (tmp_path / name).read_bytes() == content for name, content in before.items()
    )
    bundle["identity"] = {"reference_checkpoint_sha256": "different_reference"}
    with pytest.raises(RuntimeError):
        write_dictionary_artifacts(tmp_path, bundle, blocks, spectra)


def test_invalid_fit_identity_or_target_family_is_rejected():
    bundle, _, _, gradients, bases = example()
    with pytest.raises(ValueError):
        build_dictionary_bundle(
            gradients, bases, ["same"] * 18, bundle["fit_protocols"]
        )
    with pytest.raises(ValueError):
        build_dictionary_bundle(
            gradients, {}, bundle["fit_ids"], bundle["fit_protocols"]
        )
