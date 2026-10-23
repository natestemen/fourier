"""Validate row/column overlap and clean, correctly phased amplification."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from coherent_overlap_encoding import (  # noqa: E402
    a_matrix_positive_scaling,
    amplification_parameters,
    overlap_encoding,
)
from fourier.amatrix import _a_matrix_from_contents_log


@pytest.mark.parametrize("k", [1, 2, 3, 4, 8, 17, 32])
@pytest.mark.parametrize("family", ["staircase", "integer", "wide_integer"])
def test_overlap_and_clean_oaa(k, family):
    rng = np.random.default_rng(6280 + k)
    gaps = (np.ones(2 * k - 2) if family == "staircase" else
            rng.integers(1, 12, 2 * k - 2) if family == "integer" else
            rng.integers(1, 10000, 2 * k - 2))
    merged = np.r_[0, np.cumsum(gaps)][::-1]
    matrix = _a_matrix_from_contents_log(merged[::2], merged[1::2])
    steps, normalization = amplification_parameters(k)
    assert normalization >= 2 * np.sqrt(k)
    assert steps <= 1 + np.pi * np.sqrt(k) / 2
    encoded = overlap_encoding(matrix, normalization)
    p, q = encoded.row_isometry, encoded.column_isometry
    np.testing.assert_allclose(p.T @ p, np.eye(k), atol=2e-14, rtol=0)
    np.testing.assert_allclose(q.T @ q, np.eye(k), atol=2e-14, rtol=0)
    np.testing.assert_allclose(normalization * encoded.block, matrix,
                               atol=2e-14, rtol=0)
    output, failure = encoded.amplify(steps)
    # This also checks the phase of the answer; comparing probabilities alone
    # would miss an erroneous (-1)**steps factor on controlled branches.
    assert np.linalg.norm(output - matrix, 2) < 2e-11
    assert np.linalg.norm(failure, 2) < 2e-11


def test_arbitrary_superposition_and_identity_padded_sector():
    rng = np.random.default_rng(687)
    small, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    padded = np.eye(8)
    padded[:5, :5] = small
    steps, normalization = amplification_parameters(8)
    output, failure = overlap_encoding(padded, normalization).amplify(steps)
    state = rng.normal(size=8) + 1j * rng.normal(size=8)
    state /= np.linalg.norm(state)
    np.testing.assert_allclose(output @ state, padded @ state, atol=2e-14, rtol=0)
    np.testing.assert_allclose(failure @ state, 0, atol=2e-14, rtol=0)


@pytest.mark.parametrize("relative_error", [1e-3, 1e-5, 1e-7])
def test_entry_perturbations_give_clean_error_not_postselection(relative_error):
    rng = np.random.default_rng(241)
    matrix, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    perturbed = matrix * (1 + rng.uniform(-relative_error, relative_error, matrix.shape))
    steps, normalization = amplification_parameters(8)
    output, failure = overlap_encoding(perturbed, normalization).amplify(steps)
    # A numerical robustness check, not a substitute for an analytic bound.
    # In particular we count leakage, not merely the normalized good block.
    assert np.linalg.norm(output - matrix, 2) < 8 * relative_error
    assert np.linalg.norm(failure, 2) < 8 * relative_error


def test_private_padding_sectors_and_invalid_normalization():
    encoded = overlap_encoding(np.eye(3))
    p, q = encoded.row_isometry, encoded.column_isometry
    assert np.count_nonzero(p[9:12]) == 3
    assert np.count_nonzero(q[12:15]) == 3
    assert not np.any(p[12:15])
    assert not np.any(q[9:12])
    with pytest.raises(ValueError):
        overlap_encoding(np.ones((3, 3)), normalization=2)
    with pytest.raises(ValueError):
        amplification_parameters(0)
    with pytest.raises(ValueError):
        encoded.amplify(-1)


@pytest.mark.parametrize("k", [1, 2, 4, 9, 16, 32])
@pytest.mark.parametrize("family", ["staircase", "integer"])
def test_dyadic_norm_bound_positive_scaling_and_logarithmic_oaa(k, family):
    rng = np.random.default_rng(1001 + k)
    gaps = (np.ones(2 * k - 2) if family == "staircase" else
            rng.integers(1, 100, 2 * k - 2))
    merged = np.r_[0, np.cumsum(gaps)][::-1]
    matrix = _a_matrix_from_contents_log(merged[::2], merged[1::2])
    scaling = a_matrix_positive_scaling(matrix)
    b = np.abs(matrix)
    assert np.linalg.norm(b, 2) <= scaling.absolute_norm_bound
    assert np.min(scaling.left) >= 1
    assert np.min(scaling.right) >= 1
    assert np.max(scaling.left) <= 2 * np.sqrt(2 * k)
    assert np.max(scaling.right) <= 2 * np.sqrt(2 * k)
    lam = 2 * scaling.absolute_norm_bound
    assert np.max(b @ scaling.right / scaling.left) <= lam
    assert np.max(b.T @ scaling.left / scaling.right) <= lam
    assert scaling.normalization >= 2 * lam
    encoded = overlap_encoding(matrix, scaling.normalization,
                               left_scaling=scaling.left,
                               right_scaling=scaling.right)
    np.testing.assert_allclose(encoded.block * scaling.normalization, matrix,
                               atol=2e-14, rtol=0)
    output, failure = encoded.amplify(scaling.amplification_steps)
    assert np.linalg.norm(output - matrix, 2) < 2e-11
    assert np.linalg.norm(failure, 2) < 2e-11


def test_scaling_rejects_inapplicable_sign_pattern():
    with pytest.raises(ValueError):
        a_matrix_positive_scaling(np.array([[0., -1.], [1., 0.]]))


def test_logarithmic_scaling_of_identity_padded_a_matrix():
    matrix = np.eye(8)
    matrix[:5, :5] = _a_matrix_from_contents_log(
        np.arange(4, -5, -2), np.arange(3, -4, -2))
    scaling = a_matrix_positive_scaling(matrix)
    encoded = overlap_encoding(matrix, scaling.normalization,
                               left_scaling=scaling.left,
                               right_scaling=scaling.right)
    output, failure = encoded.amplify(scaling.amplification_steps)
    assert np.linalg.norm(output - matrix, 2) < 1e-12
    assert np.linalg.norm(failure, 2) < 1e-12


def test_finite_scalings_cancel_exactly_and_keep_padding_slack():
    k = 16
    matrix = _a_matrix_from_contents_log(
        np.arange(k - 1, -k, -2), np.arange(k - 2, 1 - k, -2))
    scaling = a_matrix_positive_scaling(matrix)
    rng = np.random.default_rng(281)
    left = scaling.left + rng.uniform(-.01 / np.sqrt(k), .01 / np.sqrt(k), k)
    right = scaling.right + rng.uniform(-.01 / np.sqrt(k), .01 / np.sqrt(k), k)
    encoded = overlap_encoding(matrix, scaling.normalization,
                               left_scaling=left, right_scaling=right)
    np.testing.assert_allclose(encoded.block * scaling.normalization, matrix,
                               atol=2e-14, rtol=0)
    output, failure = encoded.amplify(scaling.amplification_steps)
    assert np.linalg.norm(output - matrix, 2) < 1e-12
    assert np.linalg.norm(failure, 2) < 1e-12


@pytest.mark.parametrize("relative_error", [1e-3, 1e-5, 1e-7])
def test_balanced_amplification_entry_errors_include_failure(relative_error):
    k = 9
    matrix = _a_matrix_from_contents_log(
        np.arange(k - 1, -k, -2), np.arange(k - 2, 1 - k, -2))
    scaling = a_matrix_positive_scaling(matrix)
    rng = np.random.default_rng(719)
    perturbed = matrix * (1 + rng.uniform(-relative_error, relative_error, (k, k)))
    encoded = overlap_encoding(perturbed, scaling.normalization,
                               left_scaling=scaling.left,
                               right_scaling=scaling.right)
    output, failure = encoded.amplify(scaling.amplification_steps)
    assert np.linalg.norm(output - matrix, 2) < 8 * relative_error
    assert np.linalg.norm(failure, 2) < 8 * relative_error
