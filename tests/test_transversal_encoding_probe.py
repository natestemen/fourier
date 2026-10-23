"""Exact fixed-network reversible prefix-rank conversion checks."""

from itertools import permutations
from pathlib import Path
import random
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from transversal_encoding_probe import (  # noqa: E402
    bitonic_comparators,
    canonical_cycle_product,
    permutation_to_transversal,
    transversal_to_permutation,
)


@pytest.mark.parametrize("n", range(8))
def test_exhaustive_permutations_through_seven(n):
    forward_counts = inverse_counts = None
    for permutation in permutations(range(1, n + 1)):
        expected = tuple(sum(value <= permutation[i] for value in permutation[:i + 1])
                         for i in range(n))
        forward = permutation_to_transversal(permutation)
        assert forward.outputs == expected
        assert forward.clean
        inverse = transversal_to_permutation(forward.outputs)
        assert inverse.outputs == permutation
        assert inverse.clean
        assert canonical_cycle_product(forward.outputs) == permutation
        current_forward = (forward.comparator_applications, forward.scan_steps,
                           forward.erased_decisions, forward.public_interval_sizes)
        current_inverse = (inverse.comparator_applications, inverse.scan_steps,
                           inverse.erased_decisions, inverse.public_interval_sizes)
        if forward_counts is None:
            forward_counts, inverse_counts = current_forward, current_inverse
        assert current_forward == forward_counts
        assert current_inverse == inverse_counts
        assert forward.comparator_applications == 2 * forward.erased_decisions
        assert inverse.comparator_applications == 2 * inverse.erased_decisions


@pytest.mark.parametrize("n", [8, 9, 15, 16, 17, 31, 32, 33, 64, 127])
def test_random_uneven_intervals_and_xor_cleanup(n):
    rng = random.Random(20261022 + n)
    for _ in range(5):
        permutation = tuple(rng.sample(range(1, n + 1), n))
        code = permutation_to_transversal(permutation).outputs
        before = tuple(rng.randrange(2 ** n.bit_length()) for _ in range(n))
        forward = permutation_to_transversal(permutation, before)
        assert forward.outputs == tuple(a ^ b for a, b in zip(before, code))
        assert permutation_to_transversal(permutation, forward.outputs).outputs == before
        inverse = transversal_to_permutation(code, before)
        assert inverse.outputs == tuple(a ^ b for a, b in zip(before, permutation))
        assert transversal_to_permutation(code, inverse.outputs).outputs == before
        assert forward.clean and inverse.clean


def test_in_place_bijection_composition_is_clean():
    permutation = (4, 1, 7, 2, 6, 3, 5)
    code = permutation_to_transversal(permutation).outputs
    # U_f; SWAP; U_(f^-1): the second XOR target initially contains g.
    erased = transversal_to_permutation(code, permutation)
    assert erased.outputs == (0,) * len(permutation)
    assert erased.clean


@pytest.mark.parametrize("size", [1, 2, 4, 8, 16, 32])
def test_network_size_is_public(size):
    levels = size.bit_length() - 1
    assert len(bitonic_comparators(size)) == size * levels * (levels + 1) // 4


def test_invalid_promises_and_widths_are_rejected_by_harness():
    with pytest.raises(ValueError):
        permutation_to_transversal((1, 1))
    with pytest.raises(ValueError):
        transversal_to_permutation((1, 3))
    with pytest.raises(ValueError):
        permutation_to_transversal((1,), (0, 0))
    with pytest.raises(ValueError):
        bitonic_comparators(3)
