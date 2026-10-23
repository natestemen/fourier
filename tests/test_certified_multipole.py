"""Exact containment tests; no floating agreement is called a certificate."""

from fractions import Fraction
from pathlib import Path
import sys

import mpmath as mp
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from certified_multipole import (BinaryInterval, certified_potential, certified_weights, certified_balancing,
                                exp_interval, log_integer, sqrt_interval, tree_angle_interval)


def contains(interval, exact):
    scale = 1 << interval.bits
    return Fraction(interval.lo, scale) <= exact <= Fraction(interval.hi, scale)


def test_directed_interval_arithmetic_contains_exact_rationals():
    for a in [Fraction(-10, 3), Fraction(-1, 7), Fraction(0), Fraction(8, 5)]:
        for b in [Fraction(-2, 3), Fraction(3, 11), Fraction(4)]:
            x, y = (BinaryInterval.ratio(v.numerator, v.denominator, 20) for v in (a, b))
            assert contains(x + y, a + b)
            assert contains(x - y, a - b)
            assert contains(x * y, a * b)
            assert contains(x / y, a / b)


@pytest.mark.parametrize("value", [1, 2, 3, 7, 255, 2**40 + 3])
def test_logarithm_encloses_high_precision_reference(value):
    # Mathematical guarantee comes from the positive series remainder and
    # directed integer arithmetic; mpmath is an independent regression oracle.
    with mp.workdps(100):
        result = log_integer(value, bits=96, exponent_bound=42)
        exact = mp.log(value)
        assert mp.mpf(result.lo) / 2**96 <= exact <= mp.mpf(result.hi) / 2**96
        assert result.radius < 1e-24


@pytest.mark.parametrize("positions,charges", [
    ([0], [1]), ([0, 1, 2, 3], [-1, 1, -1, 1]),
    ([0, 2, 5, 10, 17], [3, -2, 1, 0, 4]),
    ([1, 16, 31, 63], [1, 2, -3, 1]),
])
def test_inverse_kernel_contains_exact_rational_answer(positions, charges):
    result = certified_potential(positions, charges, capacity=8, span_bound=63,
                                 charge_bound=4, order=20, bits=80, kernel="inverse")
    for i, t in enumerate(positions):
        exact = sum((Fraction(q, abs(t-s)) for j, (s, q) in enumerate(zip(positions, charges))
                     if i != j), Fraction(0))
        assert contains(result.values[i], exact)
        assert result.values[i].radius < 1e-7
    assert all(x.lo == x.hi == 0 for x in result.values[len(positions):])
    assert result.joins_clean


@pytest.mark.parametrize("positions", [[0, 1, 2], [0, 2, 9, 10, 31], [0, 15, 16, 17, 30]])
def test_signed_log_potential_includes_series_and_rounding_error(positions):
    charges = [-1 if i % 2 == 0 else 1 for i in range(len(positions))]
    result = certified_potential(positions, charges, capacity=6, span_bound=31,
                                 charge_bound=1, order=20, bits=80)
    with mp.workdps(100):
        for i, t in enumerate(positions):
            exact = sum((q * mp.log(abs(t-s)) for j, (s, q) in enumerate(zip(positions, charges))
                         if i != j), mp.mpf(0))
            interval = result.values[i]
            assert mp.mpf(interval.lo) / 2**80 <= exact <= mp.mpf(interval.hi) / 2**80
            assert interval.radius < 1e-8


def test_schedule_independent_of_shape_and_inactive_slots():
    a = certified_potential([0, 1], [1, -1], capacity=6, span_bound=31,
                            charge_bound=2, order=12, bits=48)
    b = certified_potential([0, 4, 5, 13, 24, 31], [1, 2, 0, -1, 1, -2], capacity=6,
                            span_bound=31, charge_bound=2, order=12, bits=48)
    assert (a.moment_terms, a.evaluation_terms, a.join_comparators, a.join_scan_steps) == (
        b.moment_terms, b.evaluation_terms, b.join_comparators, b.join_scan_steps)
    assert a.joins_clean and b.joins_clean


def test_more_precision_reduces_rounding_enclosure():
    args = dict(capacity=4, span_bound=31, charge_bound=1, order=24)
    coarse = certified_potential([0, 3, 7, 31], [-1, 1, -1, 1], bits=24, **args)
    fine = certified_potential([0, 3, 7, 31], [-1, 1, -1, 1], bits=64, **args)
    assert max(x.radius for x in fine.values) < max(x.radius for x in coarse.values) / 100


def test_public_parameter_promises_are_enforced():
    with pytest.raises(ValueError):
        certified_potential([0, 0], [1, 1], capacity=2, span_bound=3,
                            charge_bound=1, order=10)
    with pytest.raises(ValueError):
        certified_potential([0, 4], [1, 1], capacity=2, span_bound=3,
                            charge_bound=1, order=10)
    with pytest.raises(ValueError):
        certified_potential([0], [2], capacity=2, span_bound=3,
                            charge_bound=1, order=10)


@pytest.mark.parametrize("argument", [Fraction(-7), Fraction(-1, 3), Fraction(0),
                                    Fraction(1, 7), Fraction(8)])
def test_certified_exponential(argument):
    x = BinaryInterval.ratio(argument.numerator, argument.denominator, 96)
    interval = exp_interval(x, argument_bound=8)
    with mp.workdps(100):
        exact = mp.exp(mp.mpf(argument.numerator) / argument.denominator)
        assert mp.mpf(interval.lo) / 2**96 <= exact <= mp.mpf(interval.hi) / 2**96
        assert interval.radius < 1e-19


@pytest.mark.parametrize("ac,rc", [([0], []), ([1, -1], [0]),
                                   ([8, 3, -2, -10], [7, 1, -3])])
def test_certified_weights_contain_exact_residue_products(ac, rc):
    weights = certified_weights(ac, rc, capacity=4, span_bound=31, order=24, bits=96)
    for i, x in enumerate(ac):
        alpha2 = Fraction(1)
        for y in rc:
            alpha2 *= abs(x-y)
        for j, other in enumerate(ac):
            if i != j:
                alpha2 /= abs(x-other)
        interval = weights.alpha[i]
        assert contains(interval * interval, alpha2)
    for j, y in enumerate(rc):
        beta2 = Fraction(1)
        for x in ac:
            beta2 *= abs(x-y)
        for h, other in enumerate(rc):
            if j != h:
                beta2 /= abs(y-other)
        interval = weights.beta[j]
        assert contains(interval * interval, beta2)
    assert all(x.lo == x.hi == 0 for x in weights.alpha[len(ac):])


@pytest.mark.parametrize("left,right", [(0, 0), (0, 1), (1, 0), (1, 1), (2, 3),
                                       (1, 2**40), (2**40, 1)])
def test_tree_angles_certify_balanced_unbalanced_and_zero_branches(left, right):
    result = tree_angle_interval(left, right, bits=96)
    with mp.workdps(100):
        exact = 2 * mp.atan2(mp.sqrt(right), mp.sqrt(left)) if left or right else mp.mpf(0)
        assert mp.mpf(result.lo) / 2**96 <= exact <= mp.mpf(result.hi) / 2**96
        assert result.radius < 1e-19


def test_sqrt_interval_is_outward_even_for_inexact_endpoints():
    value = BinaryInterval.ratio(2, 3, 48)
    root = sqrt_interval(value)
    assert root.lo * root.lo <= value.lo << 48
    assert root.hi * root.hi >= value.hi << 48


@pytest.mark.parametrize("ac,rc", [([0], []), ([1, -1], [0]),
                                   ([8, 3, -2, -10], [7, 1, -3])])
def test_full_balancing_has_rigorous_positive_padding_certificate(ac, rc):
    import numpy as np
    from fourier.amatrix import a_matrix_from_contents

    result = certified_balancing(ac, rc, capacity=4, span_bound=31, order=32, bits=96)
    assert result.padding_lower.lo / 2**96 > .49
    assert result.joins_clean
    assert result.vector_error.hi / 2**96 < 1e-10
    matrix = np.eye(4)
    matrix[:len(ac), :len(ac)] = a_matrix_from_contents(np.array(ac), np.array(rc))
    left = np.array([x.midpoint for x in result.left])
    right = np.array([x.midpoint for x in result.right])
    lam = 2 * ((4 - 1).bit_length() + 1)
    assert max(np.max(abs(matrix) @ right / left), np.max(abs(matrix).T @ left / right)) < 1.01*lam
