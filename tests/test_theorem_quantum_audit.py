"""Independent finite-dimensional audits of the clean overlap interface.

These check the projector argument with arbitrary unitary extensions, not
only the projected good block. They are algebra regression tests, not a
compiled quantum implementation or a proof of its asymptotic gate count.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from coherent_overlap_encoding import overlap_encoding  # noqa: E402


def _unitary_extension(isometry, seed):
    """Keep prescribed columns, with an independently random complement."""
    dimension, rank = isometry.shape
    _, _, vh = np.linalg.svd(isometry.T, full_matrices=True)
    complement = vh[rank:].T
    rng = np.random.default_rng(seed)
    rotation, _ = np.linalg.qr(rng.normal(size=(dimension - rank,) * 2))
    extension = np.column_stack((isometry, complement @ rotation))
    np.testing.assert_allclose(extension.T @ extension, np.eye(dimension), atol=2e-14)
    return extension


def _small_plane_rotation(dimension, angle, first, second):
    result = np.eye(dimension)
    c, s = np.cos(angle), np.sin(angle)
    result[np.ix_([first, second], [first, second])] = [[c, -s], [s, c]]
    return result


@pytest.mark.parametrize("steps", [1, 2, 5])
@pytest.mark.parametrize("perturbation", [0.0, 1e-7, 1e-4])
def test_clean_oaa_error_only_needs_isometry_closeness(steps, perturbation):
    angle = 0.31
    matrix = np.array([[np.cos(angle), -np.sin(angle)],
                       [np.sin(angle), np.cos(angle)]])
    normalization = 1 / np.sin(np.pi / (4 * steps + 2))
    encoded = overlap_encoding(matrix, normalization)
    p, q = encoded.row_isometry, encoded.column_isometry
    dimension, rank = p.shape
    p_tilde = _small_plane_rotation(dimension, perturbation, 0, 6) @ p
    q_tilde = _small_plane_rotation(dimension, -perturbation, 2, 7) @ q
    # The extensions need not be close, even when perturbation is zero.
    u_p = _unitary_extension(p, 1)
    u_p_tilde = _unitary_extension(p_tilde, 71)
    u_q_tilde = _unitary_extension(q_tilde, 182)
    assert np.linalg.norm(u_p - u_p_tilde, 2) > 1
    clean = np.eye(dimension)[:, :rank]
    clean_reflection = 2 * clean @ clean.T - np.eye(dimension)
    r_p = u_p_tilde @ clean_reflection @ u_p_tilde.T
    r_q = u_q_tilde @ clean_reflection @ u_q_tilde.T
    actual = (u_p_tilde.T @ np.linalg.matrix_power(-r_q @ r_p, steps)
              @ u_q_tilde @ clean)
    xi = max(np.linalg.norm(p - p_tilde, 2), np.linalg.norm(q - q_tilde, 2))
    assert np.linalg.norm(actual - clean @ matrix, 2) <= (8 * steps + 2) * xi + 2e-13
    # This includes all auxiliary output, not only the leading good block.
    assert np.linalg.norm(actual[rank:], 2) <= (8 * steps + 2) * xi + 2e-13


def test_omitting_scratch_from_reflection_invalidates_isometry_argument():
    clean = np.eye(4)[:, :1]
    identity = np.eye(4)
    different_extension = identity[:, [0, 2, 1, 3]]
    np.testing.assert_array_equal(identity @ clean, different_extension @ clean)
    full_clean_reflection = 2 * clean @ clean.T - identity
    np.testing.assert_array_equal(
        identity @ full_clean_reflection @ identity.T,
        different_extension @ full_clean_reflection @ different_extension.T,
    )
    # Forgetting a scratch bit accepts an extra input vector, e_1. The
    # extensions agree on all clean inputs but disagree maximally here.
    incomplete_reflection = np.diag([1, 1, -1, -1])
    difference = (identity @ incomplete_reflection @ identity.T
                  - different_extension @ incomplete_reflection @ different_extension.T)
    assert np.linalg.norm(difference, 2) == 2


@pytest.mark.parametrize("steps", [1, 3, 5])
def test_odd_amplification_sign_is_a_controlled_relative_phase(steps):
    matrix = np.array([[0., 1.], [1., 0.]])
    mu = 1 / np.sin(np.pi / (4 * steps + 2))
    encoded = overlap_encoding(matrix, mu)
    p, q = encoded.row_isometry, encoded.column_isometry
    dimension = p.shape[0]
    r_p = 2 * p @ p.T - np.eye(dimension)
    r_q = 2 * q @ q.T - np.eye(dimension)
    desired = p @ matrix
    correct = np.linalg.matrix_power(-r_q @ r_p, steps) @ q
    wrong = np.linalg.matrix_power(r_q @ r_p, steps) @ q
    np.testing.assert_allclose(correct, desired, atol=2e-14)
    np.testing.assert_allclose(wrong, -desired, atol=2e-14)
    # Coherent active/inactive control forbids independently phase-aligning
    # the active branch. On an equal mixture, the output error is sqrt(2).
    input_state = np.array([1., 0.])
    assert np.linalg.norm((wrong - desired) @ input_state) / np.sqrt(2) == pytest.approx(np.sqrt(2))
