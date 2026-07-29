"""Tests for the diagonal-compression and Jacobi descriptions of A-matrices."""

import numpy as np
import pytest
import sympy as sp
from yungdiagram import YoungDiagram

from fourier.amatrix import (
    a_matrix,
    a_matrix_from_contents,
    a_matrix_symbolic,
    addable_contents,
    removable_contents,
)
from fourier.diagrams import staircase
from fourier.structured import (
    arrowhead_from_contents,
    cayley_factorization,
    jacobi_factorization_from_contents,
    projective_cauchy_from_contents,
    staircase_jacobi_coefficients,
    symmetric_content_factorization,
)


SAMPLE = [
    YoungDiagram(p)
    for p in [[1], [2, 1], [4, 2], [5, 3, 1], [6, 4, 2, 1]]
]


@pytest.mark.parametrize("yd", SAMPLE, ids=lambda d: str(d.partition))
def test_arrowhead_intertwining_identity(yd):
    ac = np.asarray(addable_contents(yd), dtype=float)
    rc = np.asarray(removable_contents(yd), dtype=float)
    A = a_matrix(yd)
    H = arrowhead_from_contents(ac, rc)

    assert np.allclose(np.diag(ac) @ A, A @ H, atol=2e-12)
    assert np.allclose(A.T @ np.diag(ac) @ A, H, atol=2e-12)


@pytest.mark.parametrize("yd", SAMPLE, ids=lambda d: str(d.partition))
def test_full_projective_cauchy_form(yd):
    ac = np.asarray(addable_contents(yd), dtype=float)
    rc = np.asarray(removable_contents(yd), dtype=float)
    form = projective_cauchy_from_contents(ac, rc)

    assert np.allclose(form.matrix(), a_matrix(yd), atol=2e-12)
    assert np.allclose(
        form.displacement,
        form.row_scale[:, None] * form.column_scale[None, :],
        atol=2e-12,
    )
    assert np.linalg.matrix_rank(form.displacement, tol=1e-11) == 1


@pytest.mark.parametrize("yd", SAMPLE, ids=lambda d: str(d.partition))
def test_jacobi_factorization_reconstructs_a_matrix(yd):
    ac = np.asarray(addable_contents(yd), dtype=float)
    rc = np.asarray(removable_contents(yd), dtype=float)
    factorization = jacobi_factorization_from_contents(ac, rc)

    assert np.allclose(factorization.matrix(), a_matrix(yd), atol=3e-12)
    assert np.allclose(
        factorization.polynomial.T @ factorization.polynomial,
        np.eye(len(ac)),
        atol=2e-12,
    )
    assert np.allclose(
        factorization.associated.T @ factorization.associated,
        np.eye(len(ac)),
        atol=2e-12,
    )
    assert np.allclose(factorization.tail_eigenvalues, rc, atol=2e-12)


def test_jacobi_factorization_is_affine_invariant():
    gaps = np.array([1.2, 0.7, 3.1, 0.4, 2.3, 1.8])
    seq = np.r_[0.0, np.cumsum(gaps)][::-1]
    ac, rc = seq[0::2], seq[1::2]

    base = jacobi_factorization_from_contents(ac, rc).matrix()
    shifted = jacobi_factorization_from_contents(4.5 * ac - 17.0, 4.5 * rc - 17.0)
    assert np.allclose(base, shifted.matrix(), atol=2e-12)
    assert np.allclose(base, a_matrix_from_contents(ac, rc), atol=2e-12)


@pytest.mark.parametrize("yd", SAMPLE, ids=lambda d: str(d.partition))
def test_fixed_gauge_cayley_factorization(yd):
    A = a_matrix(yd)
    factorization = cayley_factorization(A)

    assert np.allclose(
        factorization.generator + factorization.generator.T,
        np.zeros_like(A),
        atol=1e-14,
    )
    assert np.allclose(factorization.matrix(), A, atol=3e-12)
    assert np.isfinite(factorization.gauge_condition)


@pytest.mark.parametrize("k", [2, 3, 4, 8, 16, 32])
def test_staircase_closed_jacobi_recurrence(k):
    n = k - 1
    ac = np.array([n - 2.0 * i for i in range(k)])
    rc = np.array([n - 1.0 - 2.0 * j for j in range(k - 1)])
    factorization = jacobi_factorization_from_contents(ac, rc)
    diagonal, off_diagonal = staircase_jacobi_coefficients(k)

    assert np.allclose(np.diag(factorization.jacobi), diagonal, atol=2e-12)
    assert np.allclose(
        np.diag(factorization.jacobi, 1), off_diagonal, atol=3e-12
    )


@pytest.mark.parametrize("k", [2, 3, 4, 5, 8, 16, 32])
def test_staircase_parity_factorization(k):
    n = k - 1
    ac = np.array([n - 2.0 * i for i in range(k)])
    rc = np.array([n - 1.0 - 2.0 * j for j in range(k - 1)])
    factorization = symmetric_content_factorization(ac, rc)

    assert np.allclose(
        factorization.matrix(), a_matrix_from_contents(ac, rc), atol=3e-12
    )
    assert np.allclose(
        factorization.plus_block.T @ factorization.plus_block,
        np.eye(factorization.plus_block.shape[0]),
        atol=3e-12,
    )
    assert np.allclose(
        factorization.minus_block.T @ factorization.minus_block,
        np.eye(factorization.minus_block.shape[0]),
        atol=3e-12,
    )
    if k >= 4:
        assert factorization.rotation_count < k * (k - 1) // 2


def test_nonstaircase_self_conjugate_parity_factorization():
    yd = YoungDiagram([4, 2, 1, 1])
    ac = np.asarray(addable_contents(yd), dtype=float)
    rc = np.asarray(removable_contents(yd), dtype=float)
    factorization = symmetric_content_factorization(ac, rc)
    assert np.allclose(factorization.matrix(), a_matrix(yd), atol=3e-12)


def test_parity_factorization_rejects_asymmetric_contents():
    ac = np.array([5.0, 2.0, -1.0, -4.0])
    rc = np.array([4.0, 0.0, -2.0])
    with pytest.raises(ValueError, match="centrally symmetric|same center"):
        symmetric_content_factorization(ac, rc)


def test_staircase_k4_has_exact_five_rotation_circuit():
    """An explicit counterexample to the old six-rotation heuristic claim."""
    root = sp.sqrt
    rotations = [
        ((1, 3), root(2) / 2, root(2) / 2),
        ((2, 3), 3 * root(6) / 8, root(10) / 8),
        ((0, 1), root(10) / 4, root(6) / 4),
        ((0, 3), -root(2) / 2, -root(2) / 2),
        ((1, 2), -root(2) / 2, -root(2) / 2),
    ]
    product = sp.eye(4)
    for (i, j), c, s in rotations:
        rotation = sp.eye(4)
        rotation[i, i] = rotation[j, j] = c
        rotation[i, j], rotation[j, i] = -s, s
        product = rotation @ product

    difference = product @ sp.diag(-1, 1, 1, 1) - a_matrix_symbolic(staircase(3))
    assert difference.applyfunc(sp.simplify) == sp.zeros(4)
