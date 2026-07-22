"""Structural facts about A-matrices that the whole project relies on."""

import numpy as np
import pytest
from yungdiagram import YoungDiagram

from fourier.amatrix import (
    a_matrix,
    a_matrix_from_contents,
    a_matrix_generic4,
    a_matrix_symbolic,
    addable_contents,
    cauchy_form,
    random_content_a_matrix,
    removable_contents,
    staircase_a_matrix,
)
from fourier.diagrams import diagrams_with_addable_cells, staircase

SAMPLE = [
    YoungDiagram(p)
    for p in [[1], [2, 1], [3, 2, 1], [4, 2], [5, 3, 1], [4, 3, 2, 1], [6, 4, 2, 1]]
]


@pytest.mark.parametrize("yd", SAMPLE, ids=lambda d: str(d.partition))
def test_orthogonal(yd):
    A = a_matrix(yd)
    k = A.shape[0]
    assert A.shape == (k, k)
    assert np.allclose(A.T @ A, np.eye(k), atol=1e-12)


def test_four_addable_det_is_minus_one():
    for yd in diagrams_with_addable_cells(4, 12):
        assert np.isclose(np.linalg.det(a_matrix(yd)), -1.0, atol=1e-10)


@pytest.mark.parametrize("yd", SAMPLE[:5], ids=lambda d: str(d.partition))
def test_symbolic_matches_numeric(yd):
    A_num = a_matrix(yd)
    A_sym = np.array(a_matrix_symbolic(yd).evalf(30).tolist(), dtype=float)
    assert np.allclose(A_num, A_sym, atol=1e-12)


@pytest.mark.parametrize("yd", SAMPLE, ids=lambda d: str(d.partition))
def test_contents_strictly_interlace(yd):
    ac = addable_contents(yd)
    rc = removable_contents(yd)
    assert len(ac) == len(rc) + 1
    merged = [x for pair in zip(ac, rc) for x in pair] + [ac[-1]]
    assert all(merged[i] > merged[i + 1] for i in range(len(merged) - 1))


@pytest.mark.parametrize("yd", SAMPLE[1:], ids=lambda d: str(d.partition))
def test_cauchy_factorization(yd):
    form = cauchy_form(yd)
    assert np.allclose(form.matrix(), a_matrix(yd), atol=1e-12)
    # displacement rank 1: diag(ac)·C − C·diag(rc) = 1·1ᵀ
    assert np.allclose(form.displacement, np.ones_like(form.displacement), atol=1e-12)


@pytest.mark.parametrize("yd", SAMPLE[1:], ids=lambda d: str(d.partition))
def test_from_contents_reproduces_diagram_matrix(yd):
    A = a_matrix(yd)
    A_fit = a_matrix_from_contents(
        np.array(addable_contents(yd), dtype=float),
        np.array(removable_contents(yd), dtype=float),
    )
    assert np.allclose(A, A_fit, atol=1e-12)


@pytest.mark.parametrize("yd", SAMPLE[1:], ids=lambda d: str(d.partition))
def test_matvecs_agree(yd):
    form = cauchy_form(yd)
    A = a_matrix(yd)
    rng = np.random.default_rng(42)
    v = rng.standard_normal(A.shape[0])
    assert np.allclose(form.matvec(v), A @ v, atol=1e-10)
    assert np.allclose(form.matvec_fast(v), A @ v, atol=1e-8)


@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_staircase_toeplitz_matvec(k):
    yd = staircase(k)
    form = cauchy_form(yd)
    # the staircase Cauchy core is Toeplitz …
    C = form.core
    for i in range(C.shape[0] - 1):
        for j in range(C.shape[1] - 1):
            assert np.isclose(C[i, j], C[i + 1, j + 1])
    # … so the FFT mat-vec applies
    rng = np.random.default_rng(0)
    v = rng.standard_normal(k + 1)
    assert np.allclose(form.matvec_toeplitz(v), a_matrix(yd) @ v, atol=1e-9)


def test_staircase_a_matrix_matches_hook_route():
    A_fast = staircase_a_matrix(16)
    A_hook = a_matrix(staircase(15))
    assert np.allclose(A_fast, A_hook, atol=1e-12)


def test_large_content_constructions_are_orthogonal():
    for A in (staircase_a_matrix(300), random_content_a_matrix(300, np.random.default_rng(0))):
        assert np.allclose(A.T @ A, np.eye(300), atol=1e-10)


def test_content_gap_sequences_biject_with_diagrams():
    """Any interlaced integer content sequence is realized by a genuine
    diagram: consecutive gaps are the block heights and widths.  So the
    content-based constructors sample actual Young diagrams."""
    rng = np.random.default_rng(42)
    for _ in range(5):
        k = int(rng.integers(4, 9))
        gaps = rng.integers(1, 5, size=2 * k - 2)
        seq = np.concatenate([[0.0], np.cumsum(gaps)])[::-1]
        ac, rc = seq[0::2], seq[1::2]

        heights = [int(ac[j] - rc[j]) for j in range(k - 1)]
        widths = [int(rc[j] - ac[j + 1]) for j in range(k - 1)]
        rows = []
        for j in range(k - 1):
            rows += [sum(widths[j:])] * heights[j]
        yd = YoungDiagram(rows)

        assert len(yd.addable_cells()) == k
        A_content = a_matrix_from_contents(ac - ac[-1], rc - ac[-1])
        assert np.allclose(a_matrix(yd), A_content, atol=1e-12)


def test_generic4_matches_concrete():
    """The generic symbolic family evaluated at (w,h) equals the concrete matrix.

    Blocks (w1,h1)=(2,2), (w2,h2)=(2,2), (w3,h3)=(2,2) give the partition
    (6,6,4,4,2,2), which is 4-addable with all blocks ≥ 2 (the generic stratum).
    """
    A_sym, syms = a_matrix_generic4()
    subs = dict(zip(syms, [2, 2, 2, 2, 2, 2]))
    A_gen = np.array(A_sym.subs(subs).evalf(30).tolist(), dtype=float)
    A_con = a_matrix(YoungDiagram([6, 6, 4, 4, 2, 2]))
    assert np.allclose(A_gen, A_con, atol=1e-12)
