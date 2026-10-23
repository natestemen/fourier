"""Reversible contracts for the standalone direct-induction theorem.

These finite checks do not compile a gate-efficient circuit.  In particular,
the dictionaries model basis states, never an assumed coherent lookup oracle.
The cached-generator model explicitly retains the canonicalization bit until
after its rotation and checks cleanup on both output branches.
"""

from collections import defaultdict
from itertools import product
from math import ceil, factorial, pi, sin, sqrt
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from alternative_qft_probe import (  # noqa: E402
    add_row, addable_rows, arbitrary_completion, dimension, direct_induction,
    outer_boundary, partitions, representation, tableaux, transversal,
    young_generator,
)


def _records(word):
    """Within-row ranks; finite model of a fixed reversible sort-and-scan."""
    counts = defaultdict(int)
    result = []
    for row in word:
        counts[row] += 1
        result.append((row, counts[row]))
    return tuple(result)


def _axial(records, j):
    row_a, col_a = records[j - 1]
    row_b, col_b = records[j]
    return col_b - row_b - col_a + row_a


def _swap(records, j):
    return records[:j - 1] + (records[j], records[j - 1]) + records[j + 1:]


def _cached_generator(word, j):
    """Return ((output records, erased sign, erased cache), amplitude)."""
    original = _records(word)
    d = _axial(original, j)
    assert d != 0
    if abs(d) == 1:
        # Swapping equal-row/equal-column consecutive labels is invalid.
        return {(original, 0, (0,) * len(word)): float(d)}
    flag = int(d < 0)
    canonical = _swap(original, j) if flag else original
    positive_d = _axial(canonical, j)
    assert positive_d == abs(d) > 1
    diagonal = 1 / positive_d
    off_diagonal = sqrt(1 - diagonal * diagonal)
    reflection = np.array([[diagonal, off_diagonal],
                           [off_diagonal, -diagonal]])
    output = {}
    for new_flag in (0, 1):
        # Arithmetic depends only on the canonical records, untouched by A_D.
        erased_distance = positive_d ^ _axial(canonical, j)
        assert erased_distance == 0
        final_records = _swap(canonical, j) if new_flag else canonical
        erased_sign = new_flag ^ int(_axial(final_records, j) < 0)
        final_word = tuple(row for row, _ in final_records)
        rebuilt = _records(final_word)
        erased_cache = tuple(col ^ rebuilt_col
                             for (_, col), (_, rebuilt_col)
                             in zip(final_records, rebuilt))
        output[(final_records, erased_sign, erased_cache)] = reflection[new_flag, flag]
    return output


@pytest.mark.parametrize("m", range(2, 8))
def test_canonical_pair_generator_preserves_cache_and_erases_all_work(m):
    for shape in partitions(m):
        words = tableaux(shape)
        lookup = {word: index for index, word in enumerate(words)}
        for j in range(1, m):
            actual = np.zeros((len(words), len(words)))
            for column, word in enumerate(words):
                for (records, sign, cache), amplitude in _cached_generator(word, j).items():
                    assert sign == 0 and not any(cache)
                    output_word = tuple(row for row, _ in records)
                    assert output_word in lookup
                    actual[lookup[output_word], column] += amplitude
            np.testing.assert_allclose(actual, young_generator(shape, j), atol=2e-15)
            # The whole clean map is its own inverse, not just its success block.
            np.testing.assert_allclose(actual @ actual, np.eye(len(words)), atol=2e-15)


@pytest.mark.parametrize("m", range(2, 7))
def test_descending_public_threshold_schedule_has_correct_product_convention(m):
    schedule = tuple(range(m - 1, 0, -1))
    assert len(schedule) == m - 1
    for shape in partitions(m):
        for i in range(1, m + 1):
            actual = np.eye(dimension(shape))
            for j in schedule:
                if i <= j:
                    actual = young_generator(shape, j) @ actual
            np.testing.assert_allclose(actual, representation(shape, transversal(m, i)), atol=3e-15)
            inverse = np.eye(dimension(shape))
            for j in reversed(schedule):
                if i <= j:
                    inverse = young_generator(shape, j) @ inverse
            np.testing.assert_allclose(inverse @ actual, np.eye(len(actual)), atol=4e-15)


def _sorted_neighbor_endpoint(word):
    """Membership criterion independently checked against ballot-prefix words."""
    m = len(word)
    if any(not 1 <= row <= m for row in word):
        return None
    row_records = sorted((row, label) for label, row in enumerate(word, 1))
    counts = defaultdict(int)
    by_column = []
    for row, label in row_records:
        counts[row] += 1
        by_column.append((counts[row], row, label))
    previous_column = previous_row = previous_label = 0
    for column, row, label in sorted(by_column):
        if column != previous_column:
            if row != 1:
                return None
        elif row != previous_row + 1 or label <= previous_label:
            return None
        previous_column, previous_row, previous_label = column, row, label
    return tuple(counts[row] for row in range(1, len(counts) + 1))


@pytest.mark.parametrize("m", range(1, 6))
def test_fixed_sort_membership_rejects_padding_and_matches_all_small_tableaux(m):
    valid = {word: shape for shape in partitions(m) for word in tableaux(shape)}
    # Include code zero, and the out-of-range padded code m+1.
    for word in product(range(m + 2), repeat=m):
        assert _sorted_neighbor_endpoint(word) == valid.get(word)


@pytest.mark.parametrize("m", range(1, 6))
def test_padded_global_source_and_target_promises_have_matching_rank_per_control(m):
    total_source = total_target = 0
    for shape in partitions(m - 1):
        padded_shape = shape + (0,) * (m - len(shape))
        source = {(padded_shape, i, p + (0,), 0, 0)
                  for i in range(1, m + 1) for p in tableaux(shape)}
        target = {(padded_shape, 0, q, 0, 0)
                  for row in addable_rows(shape)
                  for q in tableaux(add_row(shape, row))}
        assert len(source) == len(target) == m * dimension(shape)
        assert not source & target  # Distinct star versus erased-index promises.
        assert all(_sorted_neighbor_endpoint(item[2][:-1]) == shape for item in source)
        assert all(_sorted_neighbor_endpoint(item[2]) in
                   {add_row(shape, row) for row in addable_rows(shape)} for item in target)
        total_source += len(source)
        total_target += len(target)
    assert total_source == total_target


def _phase_on_coordinate(dimension_, coordinate, angle):
    result = np.eye(dimension_, dtype=complex)
    result[coordinate, coordinate] = np.exp(1j * angle)
    return result


@pytest.mark.parametrize("seed", [18, 53, 77])
def test_isometry_and_approximate_reflection_errors_obey_explicit_ledger(seed):
    probe = direct_induction((1,))
    w = arbitrary_completion(probe, seed)
    dimension_ = probe.ambient_dimension
    source_coordinate = int(np.flatnonzero(probe.source[:, 0])[0])
    rotation = np.eye(dimension_, dtype=complex)
    c, s = np.cos(1e-4), np.sin(1e-4)
    rotation[np.ix_([source_coordinate, dimension_ - 1],
                    [source_coordinate, dimension_ - 1])] = [[c, -s], [s, c]]
    approximate_w = w @ rotation
    eta = np.linalg.norm((approximate_w - probe.unitary) @ probe.source, 2)
    r_i = 2 * probe.source @ probe.source.conj().T - np.eye(dimension_)
    r_j = 2 * probe.target @ probe.target.conj().T - np.eye(dimension_)
    approximate_i = _phase_on_coordinate(dimension_, source_coordinate, 2e-4) @ r_i
    approximate_j = _phase_on_coordinate(dimension_, dimension_ - 1, -3e-4) @ r_j
    rho_i = np.linalg.norm(approximate_i - r_i, 2)
    rho_j = np.linalg.norm(approximate_j - r_j, 2)
    iterate = -approximate_w @ approximate_i @ approximate_w.conj().T @ approximate_j
    full = np.linalg.matrix_power(iterate, probe.steps) @ approximate_w
    expected = probe.target @ probe.reference
    error = np.linalg.norm(full @ probe.source - expected, 2)
    assert error <= (4 * probe.steps + 1) * eta + probe.steps * (rho_i + rho_j)
    # The same bound controls the complete inverse action on ideal targets.
    inverse_error = np.linalg.norm(full.conj().T @ probe.target -
                                   probe.source @ probe.reference.conj().T, 2)
    assert inverse_error == pytest.approx(error, abs=3e-15)


def test_public_attenuation_is_valid_and_all_shapes_share_the_same_rotation_angle():
    for m in range(1, 2001):
        steps = ceil(sqrt(m))
        theta = pi / (4 * steps + 2)
        normalization = 1 / sin(theta)
        assert normalization >= sqrt(m)
        assert sqrt(m) / normalization <= 1
        assert (2 * steps + 1) * theta == pytest.approx(pi / 2)


@pytest.mark.parametrize("m", range(2, 6))
def test_outer_transfer_is_injective_and_old_control_recoverable_from_right_prefix(m):
    images = set()
    count = 0
    for shape in partitions(m - 1):
        for row in addable_rows(shape):
            endpoint = add_row(shape, row)
            for q in tableaux(endpoint):
                for right in tableaux(shape):
                    witness = outer_boundary(shape, q, right)
                    image = (endpoint, q, witness.new_right_path)
                    assert image not in images
                    images.add(image)
                    count += 1
                    assert _sorted_neighbor_endpoint(witness.new_right_path[:-1]) == shape
                    assert witness.added_row == row
                    assert not any(witness.erased_old_shape)
    assert count == factorial(m)
