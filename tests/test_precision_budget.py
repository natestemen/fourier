"""Check the explicit mathematical budgets, not a compiled quantum circuit."""

from fractions import Fraction
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from precision_budget import (horner_scaling, precision_budget,
                              relative_mass_angle_bound)
from coherent_overlap_encoding import a_matrix_positive_scaling, overlap_encoding
from fourier.amatrix import _a_matrix_from_contents_log


@pytest.mark.parametrize("k", [1, 2, 3, 16, 127, 1024, 10**9])
def test_public_budgets_have_slack(k):
    budget = precision_budget(k, max(1, 2*k), Fraction(1, 10**9))
    assert 2 ** (2*budget.neumann_steps) >= 8*k
    assert budget.normalization >= 2*budget.neumann_lambda
    assert budget.normalization <= budget.normalization_upper_bound
    assert budget.tree_leaves >= k+1
    assert budget.tree_leaves <= 2*(k+1)
    assert budget.circuit_error_bound <= budget.epsilon / 4
    assert budget.scaling_vector_error_bound < .01 / np.sqrt(k)
    assert budget.padded_probability_lower_bound > .49
    assert Fraction(1, 2**budget.probability_fraction_bits) <= (
        budget.probability_arithmetic_absolute_error)


def test_fraction_bits_grow_logarithmically():
    small = precision_budget(10**3, 10**6, Fraction(1, 10**6))
    large = precision_budget(10**30, 10**60, Fraction(1, 10**60))
    assert large.probability_fraction_bits < 1000
    assert large.probability_fraction_bits < 12*small.probability_fraction_bits


def test_common_public_budget_for_varying_identity_padded_sectors():
    budget = precision_budget(8, 100)
    rng = np.random.default_rng(1927)
    sector_outputs, targets, failures = [], [], []
    for actual_k in (1, 2, 5, 8):
        matrix = np.eye(8)
        matrix[:actual_k, :actual_k] = _a_matrix_from_contents_log(
            np.arange(actual_k-1, -actual_k, -2),
            np.arange(actual_k-2, 1-actual_k, -2))
        left, right = horner_scaling(matrix, budget)
        encoded = overlap_encoding(matrix, budget.normalization,
                                   left_scaling=left, right_scaling=right)
        output, failure = encoded.amplify(budget.amplification_steps)
        vector = rng.normal(size=8) + 1j*rng.normal(size=8)
        sector_outputs.append(output @ vector)
        targets.append(matrix @ vector)
        failures.append(failure @ vector)
    # Concatenating sectors models a coherent direct sum. No per-sector
    # renormalization or phase alignment is permitted in this comparison.
    norm = np.linalg.norm(np.concatenate(targets))
    assert np.linalg.norm(np.concatenate(sector_outputs)-np.concatenate(targets)) / norm < 1e-12
    assert np.linalg.norm(np.concatenate(failures)) / norm < 1e-12


@pytest.mark.parametrize("k", [1, 2, 4, 9, 16])
def test_horner_stability_and_exact_stored_scaling_cancellation(k):
    rng = np.random.default_rng(9100+k)
    gaps = rng.integers(1, 10, 2*k-2)
    merged = np.r_[0, np.cumsum(gaps)][::-1]
    matrix = _a_matrix_from_contents_log(merged[::2], merged[1::2])
    budget = precision_budget(k, max(1, int(merged[0]-merged[-1])))
    left, right = horner_scaling(matrix, budget)
    direct = a_matrix_positive_scaling(matrix)
    np.testing.assert_allclose(left, direct.left, atol=1e-14, rtol=0)
    np.testing.assert_allclose(right, direct.right, atol=1e-14, rtol=0)
    delta = float(budget.balancing_entry_relative_error)
    approximate = matrix*(1+rng.uniform(-delta, delta, matrix.shape))
    tau = float(budget.balancing_matvec_absolute_error)
    errors = rng.uniform(-tau, tau, (budget.neumann_steps, 2*k))
    stored_left, stored_right = horner_scaling(
        matrix, budget, approximate_matrix=approximate, component_errors=errors)
    observed_error = np.linalg.norm(np.r_[stored_left-left, stored_right-right])
    assert observed_error <= budget.scaling_vector_error_bound
    encoding = overlap_encoding(matrix, budget.normalization,
                                left_scaling=stored_left, right_scaling=stored_right)
    assert np.min(encoding.row_isometry[k*k:k*k+k].diagonal()**2) > .49
    np.testing.assert_allclose(encoding.block*budget.normalization, matrix,
                               atol=2e-14, rtol=0)
    output, failure = encoding.amplify(budget.amplification_steps)
    assert np.linalg.norm(output-matrix, 2) < 1e-11
    assert np.linalg.norm(failure, 2) < 1e-11


@pytest.mark.parametrize("eta", [1e-2, 1e-5, 1e-8])
def test_relative_mass_angle_bound_at_extreme_imbalances(eta):
    rng = np.random.default_rng(412)
    masses = 10.0**rng.uniform(-250, 0, size=(2000, 2))
    perturbed = masses*(1+rng.uniform(-eta, eta, size=masses.shape))
    angles = 2*np.arctan2(np.sqrt(masses[:, 1]), np.sqrt(masses[:, 0]))
    new_angles = 2*np.arctan2(np.sqrt(perturbed[:, 1]), np.sqrt(perturbed[:, 0]))
    assert np.max(np.abs(new_angles-angles)) <= relative_mass_angle_bound(eta)+1e-15
    assert relative_mass_angle_bound(eta) < 4*eta


@pytest.mark.parametrize("k,length,eps", [(0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, 2)])
def test_invalid_public_parameters(k, length, eps):
    with pytest.raises(ValueError):
        precision_budget(k, length, eps)
