"""Finite witnesses for the explicitly restricted amplification lower bound."""

from math import asin, ceil, pi, sin, sqrt
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from amplification_lower_bound_probe import (  # noqa: E402
    compulsory_row_slots, modular_gate_lower_bound, one_row_induction,
    physical_w_gate_lower_bound, scalar_query_lower_bound,
)
from alternative_qft_probe import (  # noqa: E402
    direct_induction, induction_bases, induction_reference,
)


@pytest.mark.parametrize("m", range(2, 11))
def test_closed_form_matches_independently_constructed_young_induction(m):
    closed = one_row_induction(m)
    np.testing.assert_allclose(closed, induction_reference((m - 1,)), atol=2e-14)
    np.testing.assert_allclose(closed.T @ closed, np.eye(m), atol=2e-14)
    branching = np.zeros(m)
    branching[0] = 1 / sqrt(m)
    branching[-1] = sqrt((m - 1) / m)
    expected = np.zeros(m)
    expected[-1] = 1
    np.testing.assert_allclose(closed.T @ branching, expected, atol=2e-14)


@pytest.mark.parametrize("m", range(3, 6))
@pytest.mark.parametrize("extra_steps", [0, 2])
def test_full_w_tableau_marginals_force_touching_every_interior_row_slot(m, extra_steps):
    probe = direct_induction((m - 1,), steps=ceil(sqrt(m)) + extra_steps)
    actual = probe.unitary @ probe.source
    _, outputs = induction_bases((m - 1,))
    path_indices = np.arange(probe.ambient_dimension) % probe.path_dimension
    # The first path basis state is the single source tableau/star word.
    for j in range(2, m):
        changed = [1 + k for k, (_, word) in enumerate(outputs) if word[j - 1] == 2]
        rows = np.isin(path_indices, changed)
        probability = np.linalg.norm(actual[rows, j - 1]) ** 2
        assert probability == pytest.approx((j - 1) / j)
        assert probability >= 0.5 - 1e-14
        # A circuit not touching the slot's distinguishing bit has zero
        # projection onto these coordinates, so this is a norm-error floor.
        assert sqrt(probability) >= 1 / sqrt(2) - 1e-14


@pytest.mark.parametrize("m", [3, 4, 17, 64, 1000])
@pytest.mark.parametrize("locality", [1, 2, 3])
def test_support_gate_bound_is_linear_for_constant_error(m, locality):
    assert compulsory_row_slots(m, 0.1) == tuple(range(2, m))
    assert physical_w_gate_lower_bound(m, locality, 0.1) == ceil((m - 2) / locality)


def test_large_error_can_remove_some_support_witnesses():
    assert compulsory_row_slots(9, sqrt(0.75)) == (5, 6, 7, 8)
    assert physical_w_gate_lower_bound(2) == 0
    np.testing.assert_array_equal(one_row_induction(1), [[1]])


@pytest.mark.parametrize("m", [2, 4, 16, 256, 4096])
def test_scalar_lower_bound_has_sqrt_scaling_and_obeys_normalization_obstruction(m):
    eps = 0.1
    best = scalar_query_lower_bound(sqrt(m), eps)
    assert best >= (2 / pi) * (1 - eps) * sqrt(m)
    assert scalar_query_lower_bound(2 * sqrt(m), eps) >= best


@pytest.mark.parametrize("steps", [0, 1, 2, 5, 20])
def test_exact_grover_polynomial_obeys_bernstein_query_bound(steps):
    q = 2 * steps + 1
    theta = pi / (2 * q)
    normalization = 1 / sin(theta)
    assert sin(q * asin(1 / normalization)) == pytest.approx(1)
    assert scalar_query_lower_bound(normalization, 0) <= q + 1e-12
    # p(x)=sin(q asin(x)) is an odd degree-q polynomial, bounded on [-1,1].
    xs = np.linspace(-1, 1, 1001)
    ys = np.sin(q * np.arcsin(xs))
    np.testing.assert_allclose(ys, -ys[::-1], atol=2e-14)
    assert np.max(abs(ys)) <= 1 + 1e-14


@pytest.mark.parametrize("n", [16, 64, 256, 1024])
def test_modular_lower_bound_sum_has_five_halves_scaling(n):
    bound = modular_gate_lower_bound(n)
    assert 0.1 * n ** 2.5 <= bound <= 0.4 * n ** 2.5


@pytest.mark.parametrize("bad", [0, -1, 1.5])
def test_invalid_sizes_rejected(bad):
    with pytest.raises(ValueError):
        one_row_induction(bad)
    with pytest.raises(ValueError):
        modular_gate_lower_bound(bad)


@pytest.mark.parametrize("bad", [-0.1, 1, float("nan")])
def test_invalid_errors_rejected(bad):
    with pytest.raises(ValueError):
        scalar_query_lower_bound(2, bad)
    with pytest.raises(ValueError):
        compulsory_row_slots(8, bad)


@pytest.mark.parametrize("bad", [0, 0.9, float("inf"), float("nan")])
def test_invalid_normalizations_rejected(bad):
    with pytest.raises(ValueError):
        scalar_query_lower_bound(bad)
