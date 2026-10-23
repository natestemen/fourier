"""Exact checks of the explicit precision ledger; not an asymptotic proof."""

from fractions import Fraction
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from direct_induction_precision import certified_tree_angle  # noqa: E402
from direct_qft_precision_schedule import (  # noqa: E402
    ceil_log2, column_precision_schedule, full_qft_column_error,
    guarded_ratio_angle, guard_precision,
)


@pytest.mark.parametrize("bits", [1, 4, 12, 32, 80])
def test_integer_horner_angle_is_inside_independent_rational_error_enclosure(bits):
    # Ratios straddling 1/4 exercise both range-reduction branches. Very
    # unbalanced and enormous masses exclude hidden binary64 assumptions.
    pairs = [(0, 0), (1, 0), (0, 1), (1, 1), (1, 3), (1, 4), (1, 5),
             (2**180 + 7, 1), (1, 2**180 + 7), (2**150 + 7, 2**149 - 3)]
    for left, right in pairs:
        result = guarded_ratio_angle(left, right, bits)
        certificate = certified_tree_angle(left, right, result.fractional_bits + 8)
        assert result.value - result.error_bound <= certificate.low
        assert certificate.high <= result.value + result.error_bound
        assert result.error_bound < Fraction(1, 1 << bits)
        if left and right:
            assert result.root_trials == result.fractional_bits + 1
        assert result.horner_terms == result.fractional_bits + 2


@pytest.mark.parametrize("degree", [2, 3, 16, 1000, 10**9])
@pytest.mark.parametrize("epsilon", [Fraction(1, 2), Fraction(1, 10**6), Fraction(1, 2**200)])
def test_shape_independent_precision_covers_entire_error_budget(degree, epsilon):
    eta = full_qft_column_error(degree, epsilon)
    schedule = column_precision_schedule(degree, eta)
    k = schedule.corner_capacity
    assert k * k <= 32 * degree
    assert Fraction(k - 1, 1 << schedule.weight_bits) <= Fraction(1, 2 * (2 * degree) ** 2)
    assert schedule.weight_state_error_bound_squared <= (eta / 8) ** 2
    assert schedule.angle_error_bound < eta / 64
    assert schedule.synthesis_error_bound == eta / 16
    assert eta / 8 + schedule.angle_error_bound + schedule.synthesis_error_bound < eta
    # A loose linear bound on bit length, including guard bits.
    logsize = ceil_log2(Fraction(2 * (degree + 1)) / eta)
    assert max(schedule.weight_bits, schedule.guarded_angle_bits) <= 8 * logsize + 32


def test_guard_budget_and_exact_log_ceiling_boundaries():
    for c in range(1, 1000):
        q = guard_precision(c)
        assert Fraction(64 * (q + 1), 1 << q) < Fraction(1, 1 << c)
    for k in range(1, 100):
        assert ceil_log2(1 << k) == k
        assert ceil_log2((1 << k) + 1) == k + 1
        assert ceil_log2(Fraction((1 << k) * 7 - 1, 7)) == k


@pytest.mark.parametrize("left,right,bits", [(-1, 2, 4), (1, 2.5, 4), (True, 2, 4), (1, 2, 0)])
def test_invalid_ratio_parameters_are_rejected(left, right, bits):
    with pytest.raises(ValueError):
        guarded_ratio_angle(left, right, bits)
