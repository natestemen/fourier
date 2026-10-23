"""All-size exact witnesses for a deliberately restricted modular theorem."""

from fractions import Fraction
from itertools import product
from math import isqrt, sqrt
from pathlib import Path
import random
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from modular_lower_bound_probe import (  # noqa: E402
    base_size,
    constant_error_module_gate_lower_bound,
    constant_error_streaming_gate_lower_bound,
    family_parameters,
    module_gate_lower_bound,
    padded_partition,
    padded_residue_ratio,
    selected_corner,
    streaming_gate_lower_bound,
    verify_size,
)
from sublinear_core_probe import (  # noqa: E402
    contents_from_partition,
    exact_alpha_squared,
    frobenius_from_partition,
    hard_contents,
    partition_from_contents,
)


@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 8, 12, 31, 64, 1000])
def test_every_size_parameter_interval(m):
    for padding in range(64 * m + 35):
        size = base_size(m) + padding
        parameters = family_parameters(size)
        assert parameters["m"] == m
        assert parameters["padding"] == padding
        assert parameters["bits"] == max(0, m // 2 - 1)
        assert parameters["span"] == 16 * m + padding
        assert (9 * parameters["span"]) ** 2 < 200 ** 2 * size
    assert family_parameters(base_size(m + 1))["m"] == m + 1


@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 8, 12])
@pytest.mark.parametrize("position", ["none", "one", "middle", "last"])
def test_exhaustive_family_with_boundary_and_interior_padding(m, position):
    padding = {"none": 0, "one": 1, "middle": 32 * m, "last": 64 * m + 34}[position]
    size = base_size(m) + padding
    result = verify_size(size)
    bits = result["bits"]
    assert result["boxes"] == size
    assert result["checked_flips"] == (bits * 2 ** (bits - 1) if bits else 0)
    assert result["strict_error_threshold"] == Fraction(1, 9 * result["span"])


def test_first_nontrivial_family_at_every_intermediate_size():
    for size in range(base_size(4), base_size(5)):
        result = verify_size(size)
        assert result["bits"] == result["checked_flips"] == 1


@pytest.mark.parametrize("m", [8, 31, 64])
def test_random_large_frobenius_support_and_residue_ratios(m):
    rng = random.Random(62347 + m)
    for _ in range(12):
        size = base_size(m) + rng.randrange(64 * m + 35)
        parameters = family_parameters(size)
        bit = rng.randrange(parameters["bits"])
        before = tuple(0 if i == bit else rng.randrange(2) for i in range(parameters["bits"]))
        after = before[:bit] + (1,) + before[bit + 1:]
        old_shape = padded_partition(size, before)
        new_shape = padded_partition(size, after)
        d, arms, legs = frobenius_from_partition(old_shape)
        new_d, new_arms, new_legs = frobenius_from_partition(new_shape)
        assert sum(old_shape) == sum(new_shape) == size
        assert (d, legs) == (new_d, new_legs)
        assert arms[0] == new_arms[0] == 8 * m - 1 + parameters["padding"]
        changed = {i for i in range(d) if arms[i] != new_arms[i]}
        assert changed
        assert changed <= set(range(8 * (bit + 1), 8 * (bit + 2)))
        x, y = contents_from_partition(old_shape)
        ordinal, row = selected_corner(size, bit)
        assert row == 8 * (bit + 1) + 1
        assert x[ordinal] == 8 * m - 16 * (bit + 1)
        assert padded_residue_ratio(size, before, bit) == Fraction(49, 81)
        assert exact_alpha_squared(x, y, ordinal) >= Fraction(1, parameters["span"] ** 2)


@pytest.mark.parametrize("padding", [1, 17, 546])
def test_padding_introduces_only_fixed_top_corners(padding):
    m = 8
    bits = (1, 0, 1)
    full_bits = (0,) + bits + (0,) * (m - len(bits) - 1)
    x, y = hard_contents(full_bits)
    shape = padded_partition(base_size(m) + padding, bits)
    new_x, new_y = contents_from_partition(shape)
    assert new_x == (8 * m + padding, 8 * m - 1) + x[1:]
    assert new_y == (8 * m + padding - 1,) + y
    old_shape = partition_from_contents(x, y)
    old_d, old_arms, old_legs = frobenius_from_partition(old_shape)
    d, arms, legs = frobenius_from_partition(shape)
    assert d == old_d
    assert legs == old_legs
    assert arms == (old_arms[0] + padding,) + old_arms[1:]


@pytest.mark.parametrize("size", [524, 525, 814, 815, 2072, 8000, 100000])
def test_public_outer_register_capacity(size):
    parameters = family_parameters(size)
    for outer in [size + 1, size + 50, 2 * size]:
        capacity = isqrt(outer)
        assert parameters["durfee"] <= capacity
        assert parameters["k"] <= 2 * capacity + 2


@pytest.mark.parametrize("arity", [1, 2, 3, 7])
@pytest.mark.parametrize("n", [1, 35, 524, 525, 815, 2000, 10000])
def test_grouped_sum_matches_each_fresh_module_count(n, arity):
    expected = sum((n - size) * module_gate_lower_bound(size, arity)
                   for size in range(n))
    assert streaming_gate_lower_bound(n, arity) == expected


@pytest.mark.parametrize("arity", [1, 2, 3])
def test_restricted_count_has_predicted_asymptotic_coefficient(arity):
    n = 10 ** 10
    expected = 1 / (30 * sqrt(2) * arity)
    actual = streaming_gate_lower_bound(n, arity) / n ** 2.5
    assert actual == pytest.approx(expected, rel=0.001)


def test_existing_global_budget_suffices_for_local_contract():
    # epsilon <= 1 and N <= n-1: compare squared positive quantities.
    for n in range(14, 1000):
        assert 200 ** 2 * (n - 1) <= (4 * n * (n + 1)) ** 2


@pytest.mark.parametrize("m", [4, 5, 8, 12])
@pytest.mark.parametrize("padding", [0, 1, 99])
def test_exact_conditional_variance_sum_for_all_missing_bits(m, padding):
    size = base_size(m) + padding
    bit_count = family_parameters(size)["bits"]
    variance_sum = Fraction(0)
    for bit in range(bit_count):
        conditional_sum = Fraction(0)
        for others in product((0, 1), repeat=bit_count - 1):
            bits = others[:bit] + (0,) + others[bit:]
            x, y = contents_from_partition(padded_partition(size, bits))
            ordinal, _ = selected_corner(size, bit)
            h_squared = exact_alpha_squared(x, y, ordinal)
            assert padded_residue_ratio(size, bits, bit) == Fraction(49, 81)
            # Var(h,7h/9) is exactly h^2/81, with no square root needed.
            variance = h_squared / 81
            assert variance >= Fraction(1, 2592 * m)
            conditional_sum += variance
        variance_sum += conditional_sum / 2 ** (bit_count - 1)
    assert variance_sum >= Fraction(bit_count, 2592 * m)


@pytest.mark.parametrize("arity", [1, 2, 3, 7])
def test_constant_error_per_module_missing_block_allowance(arity):
    for m in range(1, 1000):
        size = base_size(m)
        touched = max(0, m // 2 - 1 - m // 8)
        assert constant_error_module_gate_lower_bound(size, arity) == (
            touched + arity - 1) // arity
        assert constant_error_module_gate_lower_bound(size, arity, Fraction(0)) == (
            module_gate_lower_bound(size, arity))
        assert constant_error_module_gate_lower_bound(size, arity, Fraction(1)) == 0


@pytest.mark.parametrize("arity", [1, 2, 3])
@pytest.mark.parametrize("n", [1, 35, 525, 2000, 10000])
def test_constant_error_grouped_sum_matches_individual_modules(n, arity):
    expected = sum((n - size) * constant_error_module_gate_lower_bound(size, arity)
                   for size in range(n))
    assert constant_error_streaming_gate_lower_bound(n, arity) == expected


@pytest.mark.parametrize("arity", [1, 2, 3])
def test_constant_error_count_asymptotic_coefficient(arity):
    n = 10 ** 10
    expected = 1 / (40 * sqrt(2) * arity)
    actual = constant_error_streaming_gate_lower_bound(n, arity) / n ** 2.5
    assert actual == pytest.approx(expected, rel=0.001)


def test_exact_variance_accuracy_threshold_changes_missing_allowance():
    m = 16
    size = base_size(m)
    at_threshold = Fraction(1, 2592 * m)
    before_threshold = at_threshold - Fraction(1, 10 ** 10)
    bits = family_parameters(size)["bits"]
    assert constant_error_module_gate_lower_bound(size, 1, before_threshold) == bits
    assert constant_error_module_gate_lower_bound(size, 1, at_threshold) == bits - 1
    with pytest.raises(ValueError):
        constant_error_module_gate_lower_bound(size, error_squared=Fraction(-1))


def test_existing_global_budget_satisfies_constant_error_contract():
    for n in range(6, 1000):
        assert Fraction(1, 4 * n * (n + 1)) <= Fraction(1, 144)


@pytest.mark.parametrize("value", [-1, 0, 34, 35.5])
def test_invalid_family_sizes(value):
    with pytest.raises(ValueError):
        family_parameters(value)


@pytest.mark.parametrize("bits", [(), (0, 1), (2,), (-1,)])
def test_invalid_family_bit_records(bits):
    with pytest.raises(ValueError):
        padded_partition(base_size(4), bits)


@pytest.mark.parametrize("size,arity", [(-1, 2), (0, 0), (0, -2), (1, 1.5)])
def test_invalid_gate_count_parameters(size, arity):
    with pytest.raises(ValueError):
        module_gate_lower_bound(size, arity)


def test_invalid_flip_and_empty_streaming_size():
    with pytest.raises(ValueError):
        padded_residue_ratio(base_size(4), (1,), 0)
    with pytest.raises(ValueError):
        selected_corner(base_size(4), 1)
    with pytest.raises(ValueError):
        streaming_gate_lower_bound(0)
