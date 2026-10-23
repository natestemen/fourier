"""Independent audit of intermediate wrapper invariants and input convention.

These are small-instance mathematical witnesses, not a compiled gate-count
certificate.  The production stream already has final-matrix tests; this file
additionally checks every intermediate level against the representation formula.
"""

from functools import lru_cache
from itertools import permutations
from math import factorial, isqrt
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from streaming_induction import (  # noqa: E402
    STAR, add_row, direct_induction_matrix, evolve, induction_bases,
    initial_state, partitions, shape_of_word, tableaux,
)
from fourier.compact_shapes import FrobeniusShape  # noqa: E402


@lru_cache(maxsize=None)
def _reference_columns(shape, max_index):
    inputs, outputs = induction_bases(shape, max_index)
    matrix = direct_induction_matrix(shape, max_index)
    return inputs, outputs, matrix


@pytest.mark.parametrize("size", range(6))
def test_every_intermediate_level_is_the_extended_induction_map(size):
    outer = size + 1
    max_index = outer + 2
    for partition in partitions(size):
        for original in tableaux(tuple(partition)):
            for index in range(1, max_index + 1):
                state = {initial_state(index, original): 1.0}
                for level in range(1, outer + 1):
                    state = evolve(state, level)
                    prefix_word = original[:level - 1]
                    source_shape = shape_of_word(prefix_word)
                    inputs, outputs, matrix = _reference_columns(source_shape, max_index)
                    column = inputs.index((index, prefix_word))
                    expected = {
                        (output.endpoint, output.word, output.flag): matrix[row, column]
                        for row, output in enumerate(outputs)
                        if abs(matrix[row, column]) > 1e-14
                    }
                    actual = {}
                    for basis, coefficient in state.items():
                        # Capacity is public outer size, including on the
                        # empty/invalid branches and every intermediate level.
                        assert basis.prefix.max_boxes == outer
                        assert basis.prefix.capacity == isqrt(outer)
                        assert basis.prefix.to_partition() == source_shape
                        assert basis.word[level:] == original[level - 1:]
                        if index > level:
                            assert basis.flag == index - level + 1
                            assert basis.token == STAR
                            assert basis.word[:level] == prefix_word + (STAR,)
                            endpoint = source_shape
                        else:
                            assert basis.flag == 1
                            assert basis.token > 0
                            endpoint = add_row(source_shape, basis.token)
                            assert shape_of_word(basis.word[:level]) == endpoint
                        actual[(endpoint, basis.word[:level], basis.flag)] = coefficient
                    assert actual.keys() == expected.keys()
                    np.testing.assert_allclose(
                        [actual[key] for key in expected], list(expected.values()),
                        atol=5e-14, rtol=5e-14,
                    )


@pytest.mark.parametrize("size", range(10))
def test_outer_boundary_erases_old_shape_using_the_retained_right_row(size):
    """The appended RIGHT row is the token, not generally the last left row."""
    for partition in partitions(size):
        source = FrobeniusShape.from_partition(tuple(partition), max_boxes=size + 1)
        for corner in source.corners("add").valid_records:
            right_row = corner.row
            target = source.add_box(right_row)
            token = right_row
            token ^= target.row_checksum - source.row_checksum
            assert token == 0
            recovered_source = target.remove_box(right_row)
            assert not any(a ^ b for a, b in zip(source.to_words(), recovered_source.to_words()))


def _one_line_to_transversals(permutation):
    """Fixed scans from the proposed proof, with all inactive words erased."""
    work = list(permutation)
    indices = [0] * len(work)
    for level in range(len(work), 0, -1):
        index = work[level - 1]
        indices[level - 1] = index
        for slot in range(level):
            value = work[slot]
            work[slot] = level if value == index else value - int(value > index)
        assert work[level - 1] == level
        work[level - 1] ^= level
    assert not any(work)
    return tuple(indices)


def _reconstruct_transversal_product(indices):
    """Compute t_n ... t_1 as composition on one-line images."""
    work = list(range(1, len(indices) + 1))
    for level, index in enumerate(indices, 1):
        for slot in range(len(work)):
            value = work[slot]
            if index <= value <= level:
                work[slot] = index if value == level else value + 1
    return tuple(work)


@pytest.mark.parametrize("n", range(1, 8))
def test_clean_one_line_conversion_has_the_same_product_convention(n):
    seen = set()
    for permutation in permutations(range(1, n + 1)):
        indices = _one_line_to_transversals(permutation)
        assert all(1 <= index <= level for level, index in enumerate(indices, 1))
        assert _reconstruct_transversal_product(indices) == permutation
        seen.add(indices)
    assert len(seen) == factorial(n)
