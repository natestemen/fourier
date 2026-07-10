import numpy as np
import pytest

from fourier.amatrix import a_matrix
from fourier.diagrams import diagrams_with_addable_cells
from fourier.weyl import block_rotation_form, leakiness_rank, weyl_coordinates

CZ = np.diag([1.0, 1.0, 1.0, -1.0]).astype(complex)
SWAP = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
)


def test_weyl_a_is_pi_over_4_for_all_small_4addable():
    """Finding 1: every 4-addable A-matrix sits on the a = π/4 Weyl face."""
    count = 0
    for yd in diagrams_with_addable_cells(4, 14):
        a, _, _ = weyl_coordinates(a_matrix(yd))
        assert np.isclose(a, np.pi / 4, atol=1e-8), f"{yd.partition}: a={a}"
        count += 1
    assert count > 20  # make sure the loop actually exercised the family


def test_leakiness_reference_gates():
    assert leakiness_rank(np.eye(4, dtype=complex))[0] == 0
    assert leakiness_rank(CZ)[0] == 2
    assert leakiness_rank(SWAP)[0] == 0


def test_amatrices_are_not_leaky():
    """Finding 1 (continued): despite a = π/4, A-matrices are non-leaky."""
    for yd in diagrams_with_addable_cells(4, 12):
        U = a_matrix(yd).astype(complex)
        assert leakiness_rank(U, "left")[0] == 3
        assert leakiness_rank(U, "right")[0] == 3


@pytest.mark.parametrize("max_size", [12])
def test_block_rotation_form(max_size):
    """Every 4-addable A-matrix conjugates to diag(1, −1) ⊕ R(θ)."""
    for yd in diagrams_with_addable_cells(4, max_size):
        A = a_matrix(yd)
        Q, theta, err = block_rotation_form(A)
        assert np.allclose(Q.T @ Q, np.eye(4), atol=1e-8)
        assert err < 1e-8, f"{yd.partition}: err={err}"
