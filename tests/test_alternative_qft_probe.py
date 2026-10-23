"""Direct induction OAA audits, independent of the streaming recursion."""

from math import factorial, sqrt
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from alternative_qft_probe import (  # noqa: E402
    amplification_resource_skeleton, arbitrary_completion, dimension,
    direct_induction, fourier_by_amplified_induction, fourier_reference,
    induction_bases, induction_reference, outer_boundary, partitions,
    shape_from_word, tableaux, tight_right_adjacent_walk, young_generator,
)


@pytest.mark.parametrize("m", range(1, 5))
def test_every_small_induction_has_same_dimensional_promises_and_clean_output(m):
    for shape in partitions(m - 1):
        probe = direct_induction(shape)
        n = m * dimension(shape)
        assert probe.source.shape == probe.target.shape == (probe.ambient_dimension, n)
        np.testing.assert_allclose(probe.unitary.conj().T @ probe.unitary,
                                   np.eye(probe.ambient_dimension), atol=2e-14)
        np.testing.assert_allclose(probe.reference.conj().T @ probe.reference, np.eye(n), atol=2e-14)
        np.testing.assert_allclose(probe.target.conj().T @ probe.unitary @ probe.source,
                                   probe.reference / probe.normalization, atol=2e-14)
        # This assertion includes every flag/scratch/index/path coordinate.
        np.testing.assert_allclose(probe.amplified_columns(), probe.target @ probe.reference,
                                   atol=5e-14)
        np.testing.assert_allclose(probe.inverse_columns(), probe.source @ probe.reference.conj().T,
                                   atol=5e-14)


@pytest.mark.parametrize("seed", [8, 21])
def test_unrelated_unitary_extensions_and_branch_completions_have_identical_clean_map(seed):
    probe = direct_induction((1,), completion_seed=seed)
    changed = arbitrary_completion(probe, seed + 101)
    assert np.linalg.norm(changed - probe.unitary, 2) > 1
    np.testing.assert_allclose(changed @ probe.source, probe.unitary @ probe.source, atol=2e-14)
    np.testing.assert_allclose(probe.amplified_columns(changed), probe.target @ probe.reference, atol=5e-14)
    np.testing.assert_allclose(probe.inverse_columns(changed), probe.source @ probe.reference.conj().T, atol=5e-14)


@pytest.mark.parametrize("omitted", ["scratch", "flag"])
def test_omitting_work_from_source_reflection_breaks_clean_amplification(omitted):
    probe = direct_induction((1,))
    changed = arbitrary_completion(probe, 561)
    offset = (sum(probe.shape) + 1) * probe.path_dimension
    if omitted == "flag":
        offset *= probe.scratch_dimension
    wrong_source = np.column_stack((probe.source, np.roll(probe.source, offset, axis=0)))
    actual = probe.amplified_columns(changed, wrong_source)
    assert np.linalg.norm(actual - probe.target @ probe.reference, 2) > 0.1


@pytest.mark.parametrize("eta", [1e-7, 1e-4])
def test_error_bound_uses_clean_isometry_not_full_unitary_distance(eta):
    probe = direct_induction((1,))
    changed = arbitrary_completion(probe, 75)
    source_index = int(np.flatnonzero(probe.source[:, 0])[0])
    other = probe.ambient_dimension - 1
    rotation = np.eye(probe.ambient_dimension, dtype=complex)
    c, s = np.cos(eta), np.sin(eta)
    rotation[np.ix_([source_index, other], [source_index, other])] = [[c, -s], [s, c]]
    approximate = changed @ rotation
    xi = np.linalg.norm((approximate - probe.unitary) @ probe.source, 2)
    assert np.linalg.norm(approximate - probe.unitary, 2) > 1
    assert np.linalg.norm(probe.amplified_columns(approximate) - probe.target @ probe.reference, 2) <= (
        (4 * probe.steps + 1) * xi + 2e-13)


@pytest.mark.parametrize("n", range(1, 6))
def test_complete_qft_against_direct_young_representation(n):
    actual = fourier_by_amplified_induction(n)
    expected = fourier_reference(n)
    assert actual.shape == (factorial(n), factorial(n))
    np.testing.assert_allclose(actual, expected, atol=8e-14)
    np.testing.assert_allclose(actual.conj().T @ actual, np.eye(factorial(n)), atol=8e-14)
    rng = np.random.default_rng(107)
    state = rng.normal(size=len(actual)) + 1j * rng.normal(size=len(actual))
    state /= np.linalg.norm(state)
    np.testing.assert_allclose(actual.conj().T @ (actual @ state), state, atol=8e-14)


def test_larger_nontrivial_induction_block_keeps_entire_workspace_clean():
    probe = direct_induction((3, 2))
    assert probe.ambient_dimension == 840
    np.testing.assert_allclose(probe.amplified_columns(), probe.target @ probe.reference, atol=5e-14)
    np.testing.assert_allclose(probe.inverse_columns(), probe.source @ probe.reference.conj().T, atol=5e-14)


def test_relative_phases_and_complex_reference_entanglement():
    probe = direct_induction((1,), steps=3, completion_seed=4)
    np.testing.assert_allclose(induction_reference((1,)), [[1 / sqrt(2), 1 / sqrt(2)],
                                                        [-1 / sqrt(2), 1 / sqrt(2)]], atol=1e-14)
    # Columns can be entangled with an arbitrary reference; compare the full map.
    rng = np.random.default_rng(89)
    joint = rng.normal(size=(2, 3)) + 1j * rng.normal(size=(2, 3))
    joint /= np.linalg.norm(joint)
    actual = probe.amplified_columns() @ joint
    expected = probe.target @ probe.reference @ joint
    np.testing.assert_allclose(actual, expected, atol=5e-14)
    # Removing each iterate's minus sign produces a relative sign for odd r.
    assert np.linalg.norm((-1) ** probe.steps * actual - expected) == pytest.approx(2)


def test_external_shape_superposition_uses_public_size_and_preserves_relative_phases():
    probes = [direct_induction(shape, completion_seed=seed)
              for seed, shape in enumerate(partitions(3), 40)]
    capacity = max(probe.ambient_dimension for probe in probes)
    rank = sum(probe.source.shape[1] for probe in probes)
    w = np.eye(len(probes) * capacity, dtype=complex)
    source = np.zeros((len(w), rank), dtype=complex)
    target = np.zeros_like(source)
    reference = np.zeros((rank, rank))
    offset = 0
    for control, probe in enumerate(probes):
        rows = slice(control * capacity, control * capacity + probe.ambient_dimension)
        cols = slice(offset, offset + probe.source.shape[1])
        w[rows, rows] = probe.unitary
        source[rows, cols] = probe.source
        target[rows, cols] = probe.target
        reference[cols, cols] = probe.reference
        offset += probe.source.shape[1]
    assert len({probe.steps for probe in probes}) == 1
    assert len({probe.normalization for probe in probes}) == 1
    reflected = w @ source
    state = reflected.copy()
    for _ in range(probes[0].steps):
        state = 2 * target @ (target.conj().T @ state) - state
        state = -(2 * reflected @ (reflected.conj().T @ state) - state)
    np.testing.assert_allclose(state, target @ reference, atol=5e-14)


@pytest.mark.parametrize("m", range(1, 6))
def test_outer_endpoint_cleanup_uses_endpoint_difference_not_last_left_symbol(m):
    found_distinct_last_symbol = False
    for shape in partitions(m - 1):
        _, outputs = induction_bases(shape)
        for mu, q in outputs:
            for right in tableaux(shape):
                witness = outer_boundary(shape, q, right)
                assert witness.endpoint == mu
                assert witness.recovered_old_shape == shape
                assert not any(witness.erased_old_shape)
                assert shape_from_word(witness.new_right_path) == mu
                found_distinct_last_symbol |= q[-1] != witness.added_row
    if m >= 3:
        assert found_distinct_last_symbol


@pytest.mark.parametrize("word", [(2,), (1, 2, 2), (1, 3), (0,), (1, -1)])
def test_local_neighbor_validity_check_rejects_non_tableaux(word):
    with pytest.raises(ValueError):
        shape_from_word(word)


def test_fixed_weighted_tag_erasure_cannot_improve_sqrt_normalization():
    rng = np.random.default_rng(991)
    for m in range(2, 12):
        tag = rng.uniform(0.1, 1.0, size=m)
        tag /= np.linalg.norm(tag)
        mu = 1 / min(tag)
        attenuation = 1 / (mu * tag)
        assert mu >= sqrt(m)
        assert max(attenuation) <= 1 + 1e-14
        np.testing.assert_allclose(tag * attenuation, np.ones(m) / mu, atol=1e-14)


@pytest.mark.parametrize("shape", [(2, 1), (3, 1), (2, 2)])
def test_independent_generators_obey_coxeter_relations(shape):
    assert len(tableaux(shape)) == dimension(shape)
    generators = [young_generator(shape, j) for j in range(1, sum(shape))]
    for gate in generators:
        np.testing.assert_allclose(gate @ gate, np.eye(len(gate)), atol=1e-14)
    for left, right in zip(generators, generators[1:]):
        np.testing.assert_allclose(left @ right @ left, right @ left @ right, atol=1e-14)


@pytest.mark.parametrize("m", range(1, 15))
def test_right_cayley_walk_hits_lower_bound_with_every_pair_crossed_once(m):
    tokens = list(range(1, m + 1))
    seen = {tokens[-1]}
    crossed = set()
    walk = tight_right_adjacent_walk(m)
    assert len(walk) == m * (m - 1) // 2
    for j in walk:
        pair = tuple(sorted((tokens[j - 1], tokens[j])))
        assert pair not in crossed
        crossed.add(pair)
        tokens[j - 1], tokens[j] = tokens[j], tokens[j - 1]
        seen.add(tokens[-1])
    assert seen == set(range(1, m + 1))
    assert tokens == list(range(m, 0, -1))


@pytest.mark.parametrize("n", [1, 4, 16, 64, 256, 1024])
def test_explicit_resource_sums_have_exponent_five_halves(n):
    calls, work = amplification_resource_skeleton(n)
    assert calls <= 5 * n ** 1.5
    assert work <= 9 * n ** 2.5


def test_impossible_normalization_and_invalid_parameters_rejected():
    with pytest.raises(ValueError, match="sqrt"):
        direct_induction((1,), steps=0)
    with pytest.raises(ValueError):
        direct_induction((), scratch_dimension=0)
