"""Source-scaled prefix distances and cell-weighted V2 calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


CALIBRATION_DIM = 2
JOINT_DIM = 18


def _finite_array(values, name):
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _positive(value, name):
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


@dataclass(frozen=True)
class PrefixGeometry:
    source_mean: np.ndarray
    source_std: np.ndarray
    mask: np.ndarray
    d0: float
    std_floor: float

    @property
    def distinguishable(self):
        return self.d0 > 0.0

    def squared_distances(self, z_support, z_query):
        """Return one independent [K] distance row for each Query prefix."""
        support = _finite_array(z_support, "Support prefixes")
        query = np.atleast_2d(_finite_array(z_query, "Query prefixes"))
        dimension = len(self.mask)
        if support.ndim != 2 or query.ndim != 2:
            raise ValueError("Prefix arrays must have shape [cell, feature]")
        if support.shape[1] != dimension or query.shape[1] != dimension:
            raise ValueError("Prefix features must match the source-fit dimension")
        if not self.mask.any():
            return np.zeros((len(query), len(support)), dtype=np.float64)
        scale = np.maximum(self.source_std[self.mask], self.std_floor)
        differences = query[:, None, self.mask] - support[None, :, self.mask]
        distances = np.square(differences / scale).mean(axis=-1)
        return _finite_array(distances, "Squared standardized prefix distances")

    def weights(self, z_support, z_query, tau_ratio):
        """Use the absolute Gaussian kernel and source-only bandwidth reference."""
        ratio = _positive(tau_ratio, "Bandwidth ratio")
        distances = self.squared_distances(z_support, z_query)
        if not self.distinguishable:
            return np.ones(distances.shape, dtype=np.float64)
        bandwidth = _positive(self.d0 * ratio, "Bandwidth")
        # A remote Query may underflow to exactly zero, the ridge's no-update limit.
        with np.errstate(over="ignore", under="ignore"):
            return np.exp(-0.5 * np.square(np.sqrt(distances) / bandwidth))


def fit_prefix_geometry(z_fit, std_floor=1e-8):
    """Fit a population standard deviation and positive-pair distance median."""
    features = _finite_array(z_fit, "Source-fit prefixes")
    if features.ndim != 2 or not features.size:
        raise ValueError("Source-fit prefixes must be a nonempty cell-by-feature matrix")
    floor = _positive(std_floor, "Standard deviation floor")
    mean = _finite_array(features.mean(axis=0), "Source prefix mean")
    std = _finite_array(features.std(axis=0), "Source prefix standard deviation")
    # Use the observed range to exclude exact constant columns even with rounding.
    mask = features.max(axis=0) > features.min(axis=0)
    preliminary = PrefixGeometry(mean, std, mask, 0.0, floor)
    squared = preliminary.squared_distances(features, features)
    pairs = squared[np.triu_indices(len(features), k=1)]
    positive = np.sqrt(pairs[pairs > 0.0])
    d0 = float(np.median(positive)) if len(positive) else 0.0
    return PrefixGeometry(mean, std, mask, d0, floor)


def calibration_design(f0, anchor, cycles):
    """Build the CAL2 columns from the unchanged reference prediction."""
    reference = _finite_array(f0, "Reference prediction")
    anchor = _finite_array(anchor, "Anchor")
    cycles = _finite_array(cycles, "Registered cycles")
    if not reference.ndim or not cycles.ndim or reference.shape[-1] != cycles.shape[-1]:
        raise ValueError("Reference predictions must align with cycle coordinates")
    # A scalar anchor works for one cell or a broadcast batch. Batch anchors are [Q].
    level = reference - anchor[..., None]
    rate = np.broadcast_to((cycles - 100.0) / 1000.0, reference.shape)
    if level.shape != reference.shape:
        raise ValueError("Anchor must provide one value per reference trajectory")
    return np.stack([level, rate], axis=-1)


def cell_normal_equations(design, residual):
    """Divide by each cell's observed point count before averaging cells."""
    matrix = _finite_array(design, "Cell design")
    error = _finite_array(residual, "Cell residual")
    if matrix.ndim != 2 or matrix.shape[1] not in (CALIBRATION_DIM, JOINT_DIM):
        raise ValueError("Cell design must contain the 2 CAL2 or 18 joint columns")
    if error.ndim != 1 or len(error) != len(matrix) or not len(error):
        raise ValueError("Cell residual must match the nonempty observed design")
    gram = _finite_array(matrix.T @ matrix / len(matrix), "Cell Gram")
    rhs = _finite_array(matrix.T @ error / len(matrix), "Cell right-hand side")
    return gram, rhs


def fit_scales(grams, scale_floor=1e-12):
    """Freeze block means; use a positive floor when a source block is zero."""
    matrices = _finite_array(grams, "Source-fit cell Grams")
    if matrices.ndim != 3 or not len(matrices):
        raise ValueError("Source-fit Gram shape must be [cell, dimension, dimension]")
    dimension = matrices.shape[1]
    if matrices.shape[2] != dimension or dimension not in (CALIBRATION_DIM, JOINT_DIM):
        raise ValueError("Source-fit Gram dimensions must match CAL2 or JOINT18")
    floor = _positive(scale_floor, "Ridge scale floor")
    diagonal = np.diagonal(matrices, axis1=-2, axis2=-1)
    if (diagonal < 0.0).any():
        raise ValueError("A source-fit Gram diagonal cannot be negative")
    calibration = float(diagonal[:, :CALIBRATION_DIM].mean()) or floor
    internal = None
    if dimension == JOINT_DIM:
        internal = float(diagonal[:, CALIBRATION_DIM:].mean()) or floor
    return _positive(calibration, "CAL2 Gram scale"), (
        _positive(internal, "OUT Gram scale") if internal is not None else None
    )


def neighbor_statistics(weights):
    """Return absolute kernel coverage and effective neighbor count per Query."""
    array = _checked_weights(weights)
    if not array.shape[1]:
        zeros = np.zeros(len(array), dtype=np.float64)
        return {
            "mean_weight": zeros.copy(),
            "max_weight": zeros.copy(),
            "effective_neighbors": zeros,
        }
    maximum = array.max(axis=1)
    # Scale locally for the diagnostic's ratio only to avoid squaring tiny weights.
    scaled = np.divide(
        array, maximum[:, None], out=np.zeros_like(array), where=maximum[:, None] > 0.0
    )
    denominator = np.square(scaled).sum(axis=1)
    effective = np.divide(
        np.square(scaled.sum(axis=1)),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0.0,
    )
    return {
        "mean_weight": array.mean(axis=1),
        "max_weight": maximum,
        "effective_neighbors": effective,
    }


def _checked_weights(weights):
    array = np.atleast_2d(_finite_array(weights, "Absolute Support weights"))
    if array.ndim != 2 or ((array < 0.0) | (array > 1.0)).any():
        raise ValueError("Support weights must form a [Query, Support] matrix in [0, 1]")
    return array


def _cholesky_solve(system, right):
    factor = np.linalg.cholesky(system)
    factor_t = factor.swapaxes(-2, -1)
    return np.linalg.solve(factor_t, np.linalg.solve(factor, right))


def _schur_solve(system, right):
    calibration = system[:, :CALIBRATION_DIM, :CALIBRATION_DIM]
    cross = system[:, :CALIBRATION_DIM, CALIBRATION_DIM:]
    cross_t = cross.swapaxes(-2, -1)
    internal = system[:, CALIBRATION_DIM:, CALIBRATION_DIM:]
    rhs_c = right[:, :CALIBRATION_DIM]
    rhs_a = right[:, CALIBRATION_DIM:]
    projected = _cholesky_solve(calibration, np.concatenate([cross, rhs_c], axis=-1))
    schur = internal - cross_t @ projected[..., :-1]
    schur = (schur + schur.swapaxes(-2, -1)) / 2.0
    a = _cholesky_solve(schur, rhs_a - cross_t @ projected[..., -1:])
    c = _cholesky_solve(calibration, rhs_c - cross @ a)
    return np.concatenate([c, a], axis=1)


def solve_calibration(
    grams, rhs, weights, *, lambda_multiplier, scales, eta, support_ids=None,
    solver="cholesky",
):
    """Solve one cell-weighted ridge objective per Query row in CPU float64."""
    matrices = _finite_array(grams, "Support cell Grams")
    vector = _finite_array(rhs, "Support right-hand sides")
    weight = _checked_weights(weights)
    if vector.ndim != 2:
        raise ValueError("Support right-hand sides must be [Support, dimension]")
    cells, dimension = vector.shape
    if dimension not in (CALIBRATION_DIM, JOINT_DIM):
        raise ValueError("Calibration must use 2 or 18 coordinates")
    if matrices.shape != (cells, dimension, dimension) or weight.shape[1] != cells:
        raise ValueError("Every physical Support cell needs one Gram, RHS and weight")
    if (np.diagonal(matrices, axis1=-2, axis2=-1) < 0.0).any():
        raise ValueError("A Support Gram diagonal cannot be negative")
    if support_ids is not None:
        ids = tuple(support_ids)
        if len(ids) != cells or len(set(ids)) != cells:
            raise ValueError("Support IDs must name distinct physical cells")
        order = np.argsort(ids, kind="stable")
        matrices, vector, weight = matrices[order], vector[order], weight[:, order]

    multiplier = _positive(lambda_multiplier, "Ridge multiplier")
    scale_c, scale_a = scales
    diagonal = np.full(dimension, _positive(scale_c, "CAL2 Gram scale"))
    if dimension == JOINT_DIM:
        diagonal[CALIBRATION_DIM:] = _positive(scale_a, "OUT Gram scale")
    ridge = multiplier * diagonal
    if not np.isfinite(ridge).all() or (ridge <= 0.0).any():
        raise ValueError("Scaled ridge must remain finite and strictly positive")
    damping = float(eta)
    if not 0.0 <= damping <= 1.0:
        raise ValueError("Eta must be in [0, 1]")
    if solver not in {"cholesky", "schur"}:
        raise ValueError("Unknown calibration solver")
    if not cells or not damping:
        return np.zeros((len(weight), dimension), dtype=np.float64)

    unregularized = np.einsum("qk,kij->qij", weight, matrices) / cells
    system = (unregularized + unregularized.swapaxes(-2, -1)) / 2.0
    system += np.diag(ridge)
    right = (np.einsum("qk,ki->qi", weight, vector) / cells)[..., None]
    system = _finite_array(system, "Regularized normal system")
    right = _finite_array(right, "Weighted right-hand side")
    try:
        if solver == "schur" and dimension == JOINT_DIM:
            solution = _schur_solve(system, right)
        else:
            solution = _cholesky_solve(system, right)
    except np.linalg.LinAlgError as error:
        raise FloatingPointError("Calibration ridge system is not positive definite") from error
    return _finite_array(damping * solution[..., 0], "Calibration solution")
