"""Independent small-system checks for the compact streaming QFT schedule."""

from math import factorial
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from streaming_induction import (  # noqa: E402
    STAR, StreamState, addable_rows, add_row, clean_output, direct_fourier_matrix,
    direct_induction_matrix, evolve, fourier_basis, induction_bases, partitions,
    initial_state, compact_a, reference_a,
    streaming_resource_skeleton,
    inverse_streaming_column,
    shape_of_word, streaming_fourier_matrix, streaming_induction_matrix,
    streaming_state, tableaux, transversal_words, young_generator,
)


@pytest.mark.parametrize("size", range(0, 7))
def test_all_induction_blocks_against_direct_representation(size):
    for partition in partitions(size):
        shape = tuple(partition)
        actual = streaming_induction_matrix(shape)
        expected = direct_induction_matrix(shape)
        np.testing.assert_allclose(actual, expected, atol=5e-14, rtol=5e-14)
        np.testing.assert_allclose(actual.conj().T @ actual, np.eye(len(actual)), atol=5e-14)


@pytest.mark.parametrize("shape", [(), (1,), (2, 1), (2, 2), (3, 1, 1)])
def test_invalid_transversals_and_coherent_valid_invalid_superpositions(shape):
    max_index = sum(shape) + 4
    actual = streaming_induction_matrix(shape, max_index)
    expected = direct_induction_matrix(shape, max_index)
    np.testing.assert_allclose(actual, expected, atol=5e-14)
    np.testing.assert_allclose(actual.conj().T @ actual, np.eye(len(actual)), atol=5e-14)
    rng = np.random.default_rng(238)
    vector = rng.normal(size=len(actual)) + 1j * rng.normal(size=len(actual))
    vector /= np.linalg.norm(vector)
    np.testing.assert_allclose(actual @ vector, expected @ vector, atol=5e-14)
    np.testing.assert_allclose(actual.conj().T @ (actual @ vector), vector, atol=5e-14)


@pytest.mark.parametrize("shape", [(), (1,), (2, 1), (3, 2), (3, 2, 1)])
def test_local_inverse_recovers_all_workspace_and_relative_phases(shape):
    n = sum(shape) + 1
    rng = np.random.default_rng(531)
    inputs, _ = induction_bases(shape, n + 2)
    vector = rng.normal(size=len(inputs)) + 1j * rng.normal(size=len(inputs))
    vector /= np.linalg.norm(vector)
    initial = {initial_state(index, word): amplitude
               for (index, word), amplitude in zip(inputs, vector)}
    state = initial
    for level in range(1, n + 1):
        state = evolve(state, level)
    for basis in state:
        _, witness = clean_output(basis, shape)
        assert witness.clean
    for level in range(n, 0, -1):
        state = evolve(state, level, inverse=True)
    assert state.keys() == initial.keys()
    np.testing.assert_allclose(list(state.values()), [initial[key] for key in state], atol=5e-14)
    assert all(basis.prefix.size == 0 and basis.token == STAR for basis in state)


def test_both_diagonal_and_detour_branches_are_exercised():
    shape = (3, 2, 1)
    found = set()
    for word in tableaux(shape):
        for index in range(1, sum(shape) + 2):
            state = {initial_state(index, word): 1.0}
            for level in range(1, len(word) + 2):
                for basis in state:
                    if level > 1 and basis.token:
                        found.add("diagonal" if basis.token == basis.word[level - 1] else "detour")
                    if not basis.token:
                        found.add("inactive")
                state = evolve(state, level)
    assert found == {"diagonal", "detour", "inactive"}


@pytest.mark.parametrize("n", range(1, 7))
def test_whole_fourier_transform_against_direct_young_orthogonal_formula(n):
    actual = streaming_fourier_matrix(n)
    expected = direct_fourier_matrix(n)
    assert actual.shape == (factorial(n), factorial(n))
    assert len(fourier_basis(n)) == len(transversal_words(n)) == factorial(n)
    np.testing.assert_allclose(actual, expected, atol=5e-14, rtol=5e-14)
    np.testing.assert_allclose(actual.conj().T @ actual, np.eye(len(actual)), atol=5e-14)
    rng = np.random.default_rng(789)
    vector = rng.normal(size=len(actual)) + 1j * rng.normal(size=len(actual))
    vector /= np.linalg.norm(vector)
    np.testing.assert_allclose(actual @ vector, expected @ vector, atol=5e-14)
    np.testing.assert_allclose(actual.conj().T @ (actual @ vector), vector, atol=5e-14)


def test_signs_not_just_absolute_amplitudes():
    actual = streaming_induction_matrix((1,))
    np.testing.assert_allclose(actual, np.array([[1, 1], [-1, 1]]) / np.sqrt(2), atol=1e-14)
    assert np.any(streaming_fourier_matrix(4).real < -0.1)


@pytest.mark.parametrize("shape", [(2, 1), (3, 2), (3, 2, 1)])
def test_independent_young_generators_obey_coxeter_relations(shape):
    n = sum(shape)
    generators = [young_generator(shape, i) for i in range(1, n)]
    identity = np.eye(len(tableaux(shape)))
    for generator in generators:
        np.testing.assert_allclose(generator @ generator, identity, atol=1e-14)
    for left, right in zip(generators, generators[1:]):
        np.testing.assert_allclose(left @ right @ left, right @ left @ right, atol=1e-14)
    for i, left in enumerate(generators):
        for right in generators[i + 2:]:
            np.testing.assert_allclose(left @ right, right @ left, atol=1e-14)


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        streaming_state(0, (1,))
    with pytest.raises(ValueError):
        shape_of_word((2,))
    with pytest.raises(ValueError):
        induction_bases((2,), 2)


def test_compact_interface_matches_independent_corner_formula():
    from fourier.compact_shapes import FrobeniusShape
    for size in range(9):
        for partition in partitions(size):
            shape = tuple(partition)
            compact = FrobeniusShape.from_partition(shape, max_boxes=10)
            adds, removes, matrix = compact_a(compact)
            expected_adds, expected_removes, expected = reference_a(shape)
            assert adds == expected_adds
            assert removes == expected_removes
            np.testing.assert_array_equal(matrix, expected)


def test_no_expanded_shape_conversion_inside_inner_stream(monkeypatch):
    from fourier.compact_shapes import FrobeniusShape
    word = (1, 2, 1, 3, 2, 1)
    state = {initial_state(3, word): 1.0}
    def forbidden(*args, **kwargs):
        raise AssertionError("expanded shape conversion inside streaming recursion")
    monkeypatch.setattr(FrobeniusShape, "to_partition", forbidden)
    monkeypatch.setattr(FrobeniusShape, "from_partition", forbidden)
    for level in range(1, len(word) + 2):
        state = evolve(state, level)
    for level in range(len(word) + 1, 0, -1):
        state = evolve(state, level, inverse=True)
    assert len(state) == 1


@pytest.mark.parametrize("n", [1, 4, 16, 64, 256, 1024])
def test_fixed_capacity_resource_skeleton_is_subcubic(n):
    resources = streaming_resource_skeleton(n)
    assert resources.embedding_calls == n * (n + 1) // 2
    assert resources.padded_inner_slot_work <= 3 * n ** 2.5
    assert resources.naive_boundary_slot_work <= 3 * n ** 2.5
    assert resources.max_shape_slots <= 2 * np.sqrt(n)


@pytest.mark.parametrize("shape", [(), (1,), (2, 1), (3, 2)])
def test_inverse_rebuilds_workspace_from_only_clean_external_output(shape):
    n = sum(shape) + 1
    inputs, outputs = induction_bases(shape, n + 2)
    input_index = {basis: i for i, basis in enumerate(inputs)}
    inverse = np.zeros((len(inputs), len(outputs)), dtype=complex)
    for column, output in enumerate(outputs):
        for basis, amplitude in inverse_streaming_column(output, shape).items():
            inverse[input_index[basis], column] = amplitude
    np.testing.assert_allclose(inverse, direct_induction_matrix(shape, n + 2).conj().T, atol=5e-14)
