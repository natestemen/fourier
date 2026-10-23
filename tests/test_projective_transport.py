"""Projective covariance, including all pole choices and a genuine diagram."""

import numpy as np
import pytest
from yungdiagram import YoungDiagram

from fourier.amatrix import (
    _a_matrix_from_contents_log,
    a_matrix,
    addable_contents,
    removable_contents,
    staircase_a_matrix,
)
from fourier.projective_transport import transport_at_removable


@pytest.mark.parametrize("k", [2, 3, 4, 8, 16, 32])
@pytest.mark.parametrize("family", ["staircase", "random"])
def test_projective_transport_for_every_removable_pole(k, family):
    rng = np.random.default_rng(20261022 + k)
    gaps = np.ones(2 * k - 2) if family == "staircase" else rng.uniform(0.3, 3, 2 * k - 2)
    merged = np.r_[0.0, np.cumsum(gaps)][::-1]
    ac, rc = merged[::2], merged[1::2]
    original = _a_matrix_from_contents_log(ac, rc)
    for q in range(k - 1):
        transport = transport_at_removable(ac, rc, q)
        target = _a_matrix_from_contents_log(transport.addable, transport.removable)
        assert np.linalg.norm(transport.matrix(original) - target, ord=2) < 5e-12
        assert np.array_equal(np.sort(transport.row_order), np.arange(k))
        assert np.array_equal(np.sort(transport.column_order), np.arange(k))
        assert np.all(transport.matrix(original)[:, 0] > 0)


def test_nonuniform_non_self_conjugate_young_diagram_has_staircase_circuit():
    # Invert at the top removable content 1, scale by 6 and translate by -1.
    transport = transport_at_removable([2, 0, -2], [1, -1], 0)
    diagram = YoungDiagram([5, 5, 5, 5, 5, 5, 3])
    assert np.allclose(6 * transport.addable - 1, addable_contents(diagram))
    assert np.allclose(6 * transport.removable - 1, removable_contents(diagram))
    assert np.allclose(transport.matrix(staircase_a_matrix(3)), a_matrix(diagram))


@pytest.mark.parametrize("pole", [-1, 2, 0.5])
def test_invalid_pole_rejected(pole):
    with pytest.raises(ValueError, match="pole_index"):
        transport_at_removable([2, 0, -2], [1, -1], pole)


def test_invalid_contents_rejected():
    with pytest.raises(ValueError, match="interlace"):
        transport_at_removable([2, 0, -2], [3, -1], 0)
    with pytest.raises(ValueError, match="finite"):
        transport_at_removable([np.inf, 0, -2], [1, -1], 0)
    with pytest.raises(ValueError, match="pole_index"):
        transport_at_removable([0], [], 0)
