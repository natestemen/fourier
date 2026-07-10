import numpy as np
import pytest
from scipy.stats import ortho_group
from yungdiagram import YoungDiagram

from fourier.amatrix import a_matrix
from fourier.decompositions import (
    cs_butterfly,
    cs_factor,
    givens_factor,
    givens_reconstruct,
    parallel_depth,
    wei_di_fit,
)


@pytest.mark.parametrize("p", [[2, 1], [3, 2, 1], [4, 3, 2, 1], [6, 4, 2, 1]])
def test_givens_roundtrip_amatrix(p):
    A = a_matrix(YoungDiagram(p))
    gates, signs = givens_factor(A)
    k = A.shape[0]
    assert len(gates) <= k * (k - 1) // 2
    assert np.allclose(givens_reconstruct(k, gates, signs), A, atol=1e-10)


@pytest.mark.parametrize("k", [3, 5, 8])
def test_givens_roundtrip_random_orthogonal(k):
    A = ortho_group.rvs(k, random_state=np.random.default_rng(k))
    gates, signs = givens_factor(A)
    assert np.allclose(givens_reconstruct(k, gates, signs), A, atol=1e-10)


@pytest.mark.parametrize("p", [[3, 2, 1], [4, 3, 2, 1], [5, 4, 3, 2, 1]])
def test_cs_roundtrip(p):
    A = a_matrix(YoungDiagram(p))
    cs = cs_factor(A)
    assert np.allclose(cs.matrix(), A, atol=1e-12)
    assert cs.p + cs.q == A.shape[0]


@pytest.mark.parametrize("k", [2, 3, 4, 5, 6, 8, 16])
def test_cs_butterfly_roundtrip(k):
    A = ortho_group.rvs(k, random_state=np.random.default_rng(k + 100))
    gates, signs = cs_butterfly(A)
    assert np.allclose(givens_reconstruct(k, gates, signs), A, atol=1e-9)


def test_cs_butterfly_depth_beats_givens():
    """The butterfly's disjoint CS layers roughly halve the dependency depth
    (≈ k vs ≈ 2k for column-major Givens)."""
    k = 16
    A = ortho_group.rvs(k, random_state=np.random.default_rng(0))
    g_givens, _ = givens_factor(A)
    g_cs, _ = cs_butterfly(A)
    assert parallel_depth(g_cs) <= k
    assert parallel_depth(g_givens) >= 2 * k - 4


def test_wei_di_reproduces_amatrix():
    """Finding 5: 3 CNOT + 6 Ry suffices for a 4-addable A-matrix (det = −1)."""
    A = a_matrix(YoungDiagram([3, 2, 1]))
    wd = wei_di_fit(A, restarts=10, seed=1)
    assert wd.det_fix  # det = −1
    assert wd.n_cnots == 3
    assert wd.residual < 1e-12
    assert np.allclose(wd.matrix(), A, atol=1e-6)


def test_wei_di_so4_needs_two_cnots():
    X = ortho_group.rvs(4, random_state=np.random.default_rng(7))
    if np.linalg.det(X) < 0:
        X[0] *= -1
    wd = wei_di_fit(X, restarts=10, seed=2)
    assert not wd.det_fix
    assert wd.n_cnots == 2
    assert wd.residual < 1e-12
