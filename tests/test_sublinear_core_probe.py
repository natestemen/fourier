"""Exact same-size Young witnesses for the fresh-input gate-access bound."""

from fractions import Fraction
from pathlib import Path
import random
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from sublinear_core_probe import (  # noqa: E402
    changed_residue_ratio,
    contents_from_partition,
    exact_alpha_squared,
    frobenius_from_partition,
    hard_contents,
    partition_from_contents,
    verify_family,
)


@pytest.mark.parametrize("m", range(1, 7))
def test_exhaustive_same_size_family(m):
    summary = verify_family(m)
    assert summary["boxes"] == 32 * m * m + 3 * m
    assert summary["checked_flips"] == m * 2 ** (m - 1)


def test_smallest_witness():
    first = partition_from_contents(*hard_contents((0,)))
    second = partition_from_contents(*hard_contents((1,)))
    assert first == (8, 8, 6, 6, 4, 1, 1, 1)
    assert second == (8, 5, 5, 5, 4, 4, 2, 2)
    assert sum(first) == sum(second) == 35
    assert changed_residue_ratio((0,), 0) == Fraction(49, 81)


def test_independence_of_distant_bits():
    bits = (1, 0, 1, 1, 0, 1, 0, 0)
    for r, bit in enumerate(bits):
        if bit == 0:
            assert changed_residue_ratio(bits, r) == Fraction(49, 81)


@pytest.mark.parametrize("m", [8, 31, 64])
def test_random_larger_same_size_records(m):
    rng = random.Random(20261022 + m)
    for _ in range(20):
        block = rng.randrange(m)
        bits = tuple(0 if r == block else rng.randrange(2) for r in range(m))
        x, y = hard_contents(bits)
        partition = partition_from_contents(x, y)
        assert frobenius_from_partition(partition)[0] == 4 * m
        assert sum(partition) == 32 * m * m + 3 * m
        assert contents_from_partition(partition) == (x, y)
        assert changed_residue_ratio(bits, block) == Fraction(49, 81)
        assert exact_alpha_squared(x, y, 4 * block) >= Fraction(1, (16 * m) ** 2)


def test_nonnegative_word_translation_preserves_witness():
    bits = (1, 0, 1, 0, 0)
    m = len(bits)
    x, y = hard_contents(bits)
    shifted_x = tuple(z + 8 * m for z in x)
    shifted_y = tuple(z + 8 * m for z in y)
    assert min(shifted_x) == 0
    assert max(shifted_x) == 16 * m
    assert all(0 < z < 16 * m for z in shifted_y)
    for i in range(len(x)):
        assert exact_alpha_squared(x, y, i) == exact_alpha_squared(shifted_x, shifted_y, i)


@pytest.mark.parametrize("m", [2, 3, 4, 5, 8, 31, 64])
def test_disjoint_frobenius_support_and_fixed_addable_row(m):
    rng = random.Random(35 + m)
    bits = tuple(rng.randrange(2) if r < m // 2 else 0 for r in range(m))
    for block in range(m // 2):
        before = bits[:block] + (0,) + bits[block + 1:]
        after = bits[:block] + (1,) + bits[block + 1:]
        x, y = hard_contents(before)
        new_x, new_y = hard_contents(after)
        old_shape = partition_from_contents(x, y)
        new_shape = partition_from_contents(new_x, new_y)
        d, arms, legs = frobenius_from_partition(old_shape)
        new_d, new_arms, new_legs = frobenius_from_partition(new_shape)
        assert d == new_d == 4 * m
        assert legs == new_legs
        changed = {i for i in range(d) if arms[i] != new_arms[i]}
        assert changed
        assert changed <= set(range(8 * block, 8 * block + 8))
        assert sum(x[j] - y[j] for j in range(4 * block)) + 1 == 8 * block + 1
        assert sum(new_x[j] - new_y[j] for j in range(4 * block)) + 1 == 8 * block + 1


@pytest.mark.parametrize("bits", [(), (2,), (0, -1)])
def test_invalid_bit_records(bits):
    with pytest.raises(ValueError):
        hard_contents(bits)


def test_reject_noncentered_or_noninterlaced_records():
    with pytest.raises(ValueError):
        partition_from_contents((4, 0), (1,))
    with pytest.raises(ValueError):
        partition_from_contents((1, -1), (1,))
    with pytest.raises(ValueError):
        contents_from_partition((1, 2))
