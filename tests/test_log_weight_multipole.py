"""Numerical validation and operation bounds for signed-log residue weights."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from log_weight_multipole import (absolute_matvec, direct_log_weights,
                                 multipole_positive_scaling,
                                 multipole_potential, multipole_weights)
from coherent_overlap_encoding import a_matrix_positive_scaling, overlap_encoding
from fourier.amatrix import _a_matrix_from_contents_log


@pytest.mark.parametrize("k", [1, 2, 4, 8, 32, 64])
@pytest.mark.parametrize("family", ["staircase", "random", "wide"])
def test_all_weights_and_reconstructed_matrix(k, family):
    rng = np.random.default_rng(1500 + k)
    gaps = (np.ones(2*k-2, dtype=int) if family == "staircase" else
            rng.integers(1, 20, size=2*k-2) if family == "random" else
            2 ** rng.integers(0, 16, size=2*k-2))
    merged = np.r_[0, np.cumsum(gaps)][::-1]
    ac, rc = merged[::2], merged[1::2]
    result = multipole_weights(ac, rc, log_tolerance=1e-10)
    la, lb = direct_log_weights(ac, rc)
    np.testing.assert_allclose(result.log_alpha_squared, la, atol=2e-11, rtol=0)
    np.testing.assert_allclose(result.log_beta_squared, lb, atol=2e-11, rtol=0)
    assert result.log_error_bound <= 1e-10
    matrix = np.column_stack((result.alpha,
        result.alpha[:, None] * result.beta / (ac[:, None] - rc)))
    assert np.linalg.norm(matrix - _a_matrix_from_contents_log(ac, rc), 2) < 2e-10
    n, p, levels = 2*k-1, result.order, result.levels
    assert result.moment_terms == n * (p+1) * levels
    assert result.evaluation_terms <= 3 * n * (p+1) * levels
    assert result.occupied_boxes <= n * levels


@pytest.mark.parametrize("order", [1, 2, 4, 8, 12])
def test_series_error_bound_at_low_orders(order):
    rng = np.random.default_rng(71)
    merged = np.r_[0, np.cumsum(rng.integers(1, 5, 62))][::-1]
    ac, rc = merged[::2], merged[1::2]
    result = multipole_weights(ac, rc, order=order)
    la, lb = direct_log_weights(ac, rc)
    error = max(np.max(np.abs(result.log_alpha_squared-la)),
                np.max(np.abs(result.log_beta_squared-lb)))
    assert error < result.log_error_bound + 1e-12


def test_large_common_translation_removed_before_float_arithmetic():
    ac, rc = [7, 3, -2], [6, 0]
    shift = 10**16
    ordinary = multipole_weights(ac, rc)
    translated = multipole_weights([x+shift for x in ac], [y+shift for y in rc])
    np.testing.assert_array_equal(ordinary.alpha, translated.alpha)
    np.testing.assert_array_equal(ordinary.beta, translated.beta)


@pytest.mark.parametrize("ac,rc", [([], []), ([2, 0], [3]), ([2.5, 0], [1])])
def test_invalid_contents(ac, rc):
    with pytest.raises(ValueError):
        multipole_weights(ac, rc)


@pytest.mark.parametrize("order", [1, 2, 4, 8, 16, 28])
@pytest.mark.parametrize("kernel", ["log", "inverse"])
def test_general_potential_series_bound(order, kernel):
    rng = np.random.default_rng(170)
    positions = np.r_[0, np.cumsum(rng.integers(1, 7, 78))]
    charges = rng.normal(size=len(positions))
    differences = np.abs(positions[:, None] - positions)
    np.fill_diagonal(differences, 1)
    matrix = np.log(differences) if kernel == "log" else 1.0 / differences
    np.fill_diagonal(matrix, 0)
    result = multipole_potential(positions, charges, order=order, kernel=kernel)
    assert np.max(np.abs(result.values - matrix @ charges)) < result.series_error_bound + 2e-13


@pytest.mark.parametrize("k", [1, 2, 8, 32, 64])
@pytest.mark.parametrize("transpose", [False, True])
def test_absolute_matvec_against_independent_dense_matrix(k, transpose):
    rng = np.random.default_rng(64+k)
    merged = np.r_[0, np.cumsum(rng.integers(1, 200, 2*k-2))][::-1]
    ac, rc = merged[::2], merged[1::2]
    weights = multipole_weights(ac, rc, log_tolerance=1e-12)
    vector = rng.normal(size=k)
    result, _ = absolute_matvec(ac, rc, weights.alpha, weights.beta, vector,
                                transpose=transpose, tolerance=1e-11)
    matrix = np.abs(_a_matrix_from_contents_log(ac, rc))
    np.testing.assert_allclose(result, (matrix.T if transpose else matrix) @ vector,
                               rtol=0, atol=3e-11)


@pytest.mark.parametrize("k", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("family", ["staircase", "random", "wide"])
def test_complete_multipole_scaling_and_coherent_amplification(k, family):
    rng = np.random.default_rng(700+k)
    gaps = (np.ones(2*k-2, dtype=int) if family == "staircase" else
            rng.integers(1, 20, size=2*k-2) if family == "random" else
            2 ** rng.integers(0, 20, size=2*k-2))
    merged = np.r_[0, np.cumsum(gaps)][::-1]
    ac, rc = merged[::2], merged[1::2]
    result = multipole_positive_scaling(ac, rc, tolerance=1e-12)
    exact = _a_matrix_from_contents_log(ac, rc)
    dense = a_matrix_positive_scaling(exact)
    np.testing.assert_allclose(result.left, dense.left, rtol=0, atol=2e-12)
    np.testing.assert_allclose(result.right, dense.right, rtol=0, atol=2e-12)
    assert np.all(np.abs(exact) @ result.right <= 2*result.absolute_norm_bound*result.left)
    assert np.all(np.abs(exact).T @ result.left <= 2*result.absolute_norm_bound*result.right)
    weights = result.weights
    approximate = np.column_stack((weights.alpha,
        weights.alpha[:, None] * weights.beta / (ac[:, None] - rc)))
    encoding = overlap_encoding(approximate, result.normalization,
                                 left_scaling=result.left, right_scaling=result.right)
    output, failure = encoding.amplify(result.amplification_steps)
    assert np.linalg.norm(output - exact, 2) < 1e-10
    assert np.linalg.norm(failure, 2) < 1e-10
    assert result.normalization >= 4*result.absolute_norm_bound


def test_potential_power_of_two_span_and_leaf_neighbors():
    positions = [0, 1, 2, 4, 8, 16]
    charges = np.ones(len(positions))
    result = multipole_potential(positions, charges, order=30, kernel="inverse")
    distances = np.abs(np.subtract.outer(positions, positions)).astype(float)
    np.fill_diagonal(distances, np.inf)
    np.testing.assert_allclose(result.values, (1/distances).sum(axis=1), atol=1e-13)
