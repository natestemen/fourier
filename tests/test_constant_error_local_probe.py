"""Exact checks for the constant-error fresh-local lower-bound proof."""

from fractions import Fraction
from itertools import product
from pathlib import Path
import random
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from constant_error_local_probe import (  # noqa: E402
    block_ratio,
    forced_touched_blocks,
    midpoint_residue,
    padded_partition,
    padding_residue_ratio,
    residue_ratio_to_midpoints,
    size_parameters,
    uniform_ratio_bound,
)
from sublinear_core_probe import (  # noqa: E402
    contents_from_partition,
    exact_alpha_squared,
    frobenius_from_partition,
    hard_contents,
)


@pytest.mark.parametrize("m", range(1, 7))
def test_exhaustive_residue_identity_and_uniform_lower_bound(m):
    for bits in product((0, 1), repeat=m):
        x, y = hard_contents(bits)
        for r in range(m):
            residue = exact_alpha_squared(x, y, 4 * r)
            ratio = residue_ratio_to_midpoints(bits, r)
            assert residue == midpoint_residue(m, 4 * r) * ratio
            assert ratio >= uniform_ratio_bound() > Fraction(1, 4)
            assert residue >= Fraction(1, 32 * m)


@pytest.mark.parametrize("m", [1, 2, 7, 16, 64])
def test_all_midpoint_residues_including_endpoints(m):
    x = tuple(8 * m - 4 * i for i in range(4 * m + 1))
    y = tuple(x[j] - 2 for j in range(4 * m))
    for i in range(4 * m + 1):
        assert midpoint_residue(m, i) == exact_alpha_squared(x, y, i)
        assert midpoint_residue(m, i) >= Fraction(1, 8 * m)


def test_exceptional_and_tail_factor_bounds():
    for t in [-1, 0]:
        for bit in [0, 1]:
            assert Fraction(7, 12) <= block_ratio(t, bit) <= 1
    for j in range(1, 200):
        for t in [j, -j - 1]:
            for bit in [0, 1]:
                assert 0 < 1 - block_ratio(t, bit) <= Fraction(5, 256 * j * j)
    assert uniform_ratio_bound() == Fraction(2891, 9216)


@pytest.mark.parametrize("m", [8, 9, 16, 31])
def test_padding_preserves_frobenius_support_and_residue_ratio(m):
    rng = random.Random(3000 + m)
    bits = tuple(rng.randrange(2) if 0 < r < m // 2 else 0 for r in range(m))
    for padding in [0, 1, 64 * m + 34]:
        for r in range(1, m // 2):
            before = bits[:r] + (0,) + bits[r + 1:]
            after = bits[:r] + (1,) + bits[r + 1:]
            before_shape = padded_partition(before, padding)
            after_shape = padded_partition(after, padding)
            x, y = hard_contents(before)
            padded_x, padded_y = contents_from_partition(before_shape)
            new_x, new_y = contents_from_partition(after_shape)
            content = x[4 * r]
            old_residue = exact_alpha_squared(x, y, 4 * r)
            residue = exact_alpha_squared(padded_x, padded_y, padded_x.index(content))
            new_residue = exact_alpha_squared(new_x, new_y, new_x.index(content))
            assert residue / old_residue == padding_residue_ratio(r, padding) >= 1
            assert new_residue / residue == Fraction(49, 81)
            assert residue >= Fraction(1, 32 * m)
            d, arms, legs = frobenius_from_partition(before_shape)
            new_d, new_arms, new_legs = frobenius_from_partition(after_shape)
            assert d == new_d == 4 * m
            assert legs == new_legs
            changed = {i for i in range(d) if arms[i] != new_arms[i]}
            assert changed and changed <= set(range(8 * r, 8 * r + 8))
            assert before_shape[8 * r] - 8 * r == content
            assert after_shape[8 * r] - 8 * r == content


@pytest.mark.parametrize("m", [1, 2, 16, 64, 127])
def test_every_box_count_between_consecutive_hard_sizes(m):
    initial = 32 * m * m + 3 * m
    for padding in range(64 * m + 35):
        assert size_parameters(initial + padding) == (m, padding)
    assert size_parameters(initial + 64 * m + 35) == (m + 1, 0)


def test_explicit_constant_error_and_gate_count_constants():
    assert Fraction(2592, 144 ** 2) == Fraction(1, 8)
    for m in range(16, 1000):
        assert forced_touched_blocks(m) >= Fraction(m, 4)
    # Var(h,7h/9) = h^2/81, with h^2 >= 1/(32m).
    assert Fraction(1, 4) * (1 - Fraction(7, 9)) ** 2 == Fraction(1, 81)
    assert 32 * 81 == 2592
