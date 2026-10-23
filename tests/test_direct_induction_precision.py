"""Exact-rational and endpoint audits for direct-induction precision."""

from fractions import Fraction
from math import atan2, sqrt
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from alternative_qft_probe import add_row, addable_rows, dimension, partitions  # noqa: E402
from direct_induction_precision import (  # noqa: E402
    bounded_weight_factors,
    branch_approximation,
    branch_precision_budget,
    certified_tree_angle,
    exact_branch_weights,
    fixed_tree_rotations,
    generator_angle_bits,
    tree_prepared_state,
    young_reflection_block,
)


def _corner_contents(shape):
    addable = tuple(
        (shape[row - 1] if row <= len(shape) else 0) + 1 - row
        for row in addable_rows(shape)
    )
    removable = tuple(
        length - row
        for row, length in enumerate(shape, 1)
        if length > (shape[row] if row < len(shape) else 0)
    )
    return addable, removable


@pytest.mark.parametrize("m", range(1, 11))
def test_bounded_products_equal_exact_branching_dimension_ratios(m):
    """This ties the arithmetic lemma to the actual induction coefficient."""
    for shape in partitions(m - 1):
        addable, removable = _corner_contents(shape)
        weights = exact_branch_weights(addable, removable)
        expected = tuple(
            Fraction(dimension(add_row(shape, row)), m * dimension(shape))
            for row in addable_rows(shape)
        )
        assert weights == expected
        assert sum(weights, Fraction()) == 1
        assert len(addable) * (len(addable) - 1) // 2 <= m - 1
        span = max(1, addable[0] - addable[-1])
        assert span <= 2 * m
        for index, weight in enumerate(weights):
            factors = bounded_weight_factors(addable, removable, index)
            assert len(factors) == len(addable) - 1
            assert all(0 < numerator < denominator <= span
                       for numerator, denominator in factors)
            assert weight >= Fraction(1, span * span)


@pytest.mark.parametrize(
    "merged",
    [
        (0,),
        (5, 4, 1, 0, -3),
        (10, 7, 3, -2, -5, -9, -12),
        (100, 99, 7, 3, -20),
    ],
)
@pytest.mark.parametrize("bits", [4, 9, 17, 31])
def test_directed_rounding_has_linear_exact_error(merged, bits):
    addable, removable = merged[::2], merged[1::2]
    exact = exact_branch_weights(addable, removable)
    try:
        audit = branch_approximation(addable, removable, bits)
    except ArithmeticError:
        # Very low precision is permitted to erase every leaf; the theorem's
        # public positivity budget explicitly excludes this case.
        assert sum(int(weight * (1 << bits)) for weight in exact) == 0
        return

    scale = 1 << bits
    for weight, numerator in zip(exact, audit.dyadic_numerators):
        rounded = Fraction(numerator, scale)
        assert 0 <= weight - rounded <= audit.per_weight_error_bound
    assert audit.missing_mass <= len(exact) * audit.per_weight_error_bound
    assert audit.l1_error <= 2 * audit.missing_mass
    assert audit.l1_error <= audit.state_error_bound_squared
    assert sum(audit.normalized, Fraction()) == 1


@pytest.mark.parametrize("m", [2, 3, 8, 31, 128, 1024, 10**6])
@pytest.mark.parametrize("eta", [Fraction(1, 10), Fraction(1, 10**6), Fraction(1, 10**15)])
def test_public_budget_is_logarithmic_positive_and_within_error(m, eta):
    # Largest integer k satisfying binom(k,2)<=M-1.
    k = 1
    while (k + 1) * k // 2 <= m - 1:
        k += 1
    length = 2 * m
    budget = branch_precision_budget(k, length, eta)
    assert budget.padded_capacity >= k
    assert budget.padded_capacity < 2 * k
    assert Fraction(k - 1, 1 << budget.weight_fraction_bits) < Fraction(1, length * length)
    assert float(budget.weight_state_error_bound_squared) ** 0.5 < float(eta) / 2
    assert budget.tree_angle_error_bound < eta / 8
    assert budget.padded_capacity * budget.per_rotation_synthesis_error <= eta / 4
    # An explicit coarse affine envelope establishes O(log M+log(1/eta)).
    public_log_size = m.bit_length() + eta.denominator.bit_length()
    assert budget.word_bits <= 8 * public_log_size + 32
    # The fixed masked factor schedule is O(K^2)=O(M), including padding.
    assert budget.padded_capacity**2 <= 32 * m


@pytest.mark.parametrize(
    "left,right",
    [
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
        (1, 10**12),
        (10**12, 1),
        (17, 31),
        (31, 17),
    ],
)
@pytest.mark.parametrize("bits", [8, 16, 24])
def test_rational_angle_certificate_contains_endpoint_safe_reference(left, right, bits):
    interval = certified_tree_angle(left, right, bits)
    expected = 0.0 if left + right == 0 else atan2(sqrt(right), sqrt(left))
    assert float(interval.low) <= expected + 2e-15
    assert expected <= float(interval.high) + 2e-15
    assert interval.width < Fraction(1, 1 << bits)
    if right == 0:
        assert interval.low == interval.high == 0


@pytest.mark.parametrize("k", [1, 2, 3, 5, 8, 13])
def test_fixed_tree_includes_zero_padding_and_prepares_normalized_dyadics(k):
    numerators = tuple((index + 1) ** 2 for index in range(k))
    capacity = 1 << (k - 1).bit_length()
    rotations = fixed_tree_rotations(numerators, 24, capacity=capacity)
    assert len(rotations) == capacity - 1
    # Every padded-only subtree is explicitly an unreachable identity, while
    # every mixed active/padded node has the exact zero-child endpoint.
    assert all(node.angle.low == node.angle.high == 0
               for node in rotations if node.right_mass == 0)

    exact = np.sqrt(np.array(numerators + (0,) * (capacity - k), dtype=float))
    exact /= np.linalg.norm(exact)
    ideal = tree_prepared_state(numerators, capacity=capacity)
    approximate = tree_prepared_state(numerators, angle_bits=24, capacity=capacity)
    np.testing.assert_allclose(ideal, exact, atol=2e-15, rtol=0)
    assert np.linalg.norm(approximate - exact) <= max(0, capacity - 1) / 2**24 + 2e-15
    assert np.count_nonzero(approximate[k:]) == 0


@pytest.mark.parametrize("axial", [-97, -5, -2, -1, 1, 2, 5, 97])
@pytest.mark.parametrize("bits", [8, 16, 24])
def test_young_generator_angle_rounding_has_uniform_operator_bound(axial, bits):
    exact = young_reflection_block(axial)
    rounded = young_reflection_block(axial, angle_bits=bits)
    if abs(axial) == 1:
        np.testing.assert_array_equal(rounded, exact)
    else:
        assert np.linalg.norm(rounded - exact, 2) <= 2 ** (-bits - 1) + 2e-15
    np.testing.assert_allclose(rounded.T @ rounded, np.eye(2), atol=2e-15, rtol=0)


@pytest.mark.parametrize("m", [1, 2, 7, 64, 1024, 10**9])
def test_generator_precision_sums_over_the_fixed_threshold_schedule(m):
    eta = Fraction(1, 10**9)
    bits = generator_angle_bits(m, eta)
    assert m * Fraction(1, 1 << (bits + 1)) <= eta / 16
    assert bits <= 4 * (m.bit_length() + eta.denominator.bit_length()) + 8


@pytest.mark.parametrize(
    "addable,removable",
    [
        ((2, 0), (3,)),
        ((2, 0), (1, 0)),
        ((2.0, 0), (1,)),
        ((2, 1), (1,)),
    ],
)
def test_invalid_content_records_are_rejected(addable, removable):
    with pytest.raises((TypeError, ValueError)):
        exact_branch_weights(addable, removable)
