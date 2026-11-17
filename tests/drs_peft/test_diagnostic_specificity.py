from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.drs_peft.data import file_hash, write_json
from scripts.drs_peft.diagnostic_specificity import (
    checked_p1_arrays,
    coefficients,
    cosine,
    fit_prefix_slope,
    output_diagnostics,
    supervision_arrays,
)


def fixture_data():
    return SimpleNamespace(
        targets=np.array([[0.9, 0.8, 0.7], [0.95, 0.0, 0.85]], np.float32),
        observed=np.array([[True, True, True], [True, False, True]]),
        anchor=np.array([1.0, 1.1], np.float32),
        query=np.array([101.0, 102.0, 103.0]),
        ids=["source_a", "source_b"],
        protocol_ids=np.array(["group_a", "group_b"]),
    )


def test_common_mask_permutation_changes_labels_and_preserves_mask():
    data = fixture_data()
    original = data.targets.copy()
    result = supervision_arrays(data, np.array([0, 1]), np.array([1, 0]), [-0.01, 0.02])
    true, mask = result["true_common"]
    donor, donor_mask = result["shuffled_common"]
    assert np.array_equal(mask, donor_mask)
    assert np.array_equal(mask, [[True, False, True], [True, False, True]])
    assert np.array_equal(donor, data.targets[[1, 0]])
    assert np.array_equal(true, data.targets)
    assert np.array_equal(data.targets, original)
    assert np.array_equal(result["flat_full"][1], data.observed)
    assert np.array_equal(result["linear_full"][1], data.observed)
    assert result["linear_full"][0][1, -1] > data.anchor[1]


def test_output_coefficients_equal_autograd_for_cell_mae():
    prediction = torch.tensor(
        [[0.9, 0.6, 0.9]], dtype=torch.float64, requires_grad=True
    )
    target = torch.tensor([[0.8, 0.0, 0.9]], dtype=torch.float64)
    mask = torch.tensor([[True, False, True]])
    loss = (prediction[mask] - target[mask]).abs().mean()
    actual = torch.autograd.grad(loss, prediction)[0].numpy()
    expected = coefficients(prediction.detach().numpy(), target.numpy(), mask.numpy())
    np.testing.assert_array_equal(actual, expected)


def test_prefix_ols_preserves_positive_slope_and_rejects_future():
    cycles = np.arange(1, 101)
    assert fit_prefix_slope(cycles, 0.9 + cycles * 0.001) == pytest.approx(0.001)
    with pytest.raises(ValueError, match="1..100"):
        fit_prefix_slope(cycles + 1, 0.9 + cycles * 0.001)


def test_common_coefficients_have_zero_exclusive_mask_energy():
    data = fixture_data()
    rows, donors = np.array([0, 1]), np.array([1, 0])
    supervision = supervision_arrays(data, rows, donors, [-0.01, 0.02])
    prediction = np.array([[0.92, 0.91, 0.9], [1.0, 1.01, 1.02]])
    frame = output_diagnostics(data, rows, donors, prediction, supervision, 42)
    common = frame.loc[frame.left.eq("true_common")]
    assert common.exact_mask_match.all()
    assert common.coefficient_delta_exclusive_mask_energy_fraction.eq(0).all()


def test_signed_cosine_retains_sign_reversal():
    value = np.array([[1.0, 0.0], [2.0, 3.0]])
    assert cosine(value, -value) == pytest.approx(-1.0)
    assert cosine(value, np.zeros_like(value)) is None


def test_compact_p1_cache_manifest_uses_checked_complete_layer_mapping(tmp_path):
    manifest = {}
    for role, ids in (
        ("source/nominal", ["a", "b"]),
        ("source_shuffled/nominal", ["a", "b"]),
        ("meta_validation/nominal", ["c"]),
    ):
        folder = tmp_path / role.replace("/", "_")
        folder.mkdir()
        array_path = folder / "layer_00.npy"
        np.save(array_path, np.ones((len(ids), 2, 2)))
        records = [
            {
                "path": array_path.name,
                "sha256": file_hash(array_path),
                "bytes": array_path.stat().st_size,
            }
        ]
        compact = {"identity": {"ids": ids}, "files": records, "folder": str(folder)}
        write_json(
            folder / "COMPLETE.json",
            dict(compact, layer_files={"q_proj": records[0]}, optimizer_updates=0),
        )
        manifest[role] = compact
    write_json(tmp_path / "seed42/P1_CACHE_MANIFEST.json", manifest)
    arrays, _, _ = checked_p1_arrays(tmp_path, 42, ["a", "b"], ["c"])
    assert arrays["true_full"]["q_proj"].shape == (2, 2, 2)
