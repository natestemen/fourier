"""Independent regressions for promise-only amplification and composition.

These are finite algebraic checks, not evidence of the asymptotic theorem.
The generic examples deliberately have arbitrary, badly different unitary
completions away from their promised input columns. No state is projected or
reset between steps in the scratch-reuse test.
"""

from math import pi
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.linalg import block_diag, null_space

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from direct_induction_gate_probe import compile_induction  # noqa: E402


def _unitary(rng, dimension):
    matrix = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
        size=(dimension, dimension))
    unitary, _ = np.linalg.qr(matrix)
    return unitary


def _generic_block(rank, rounds, seed):
    """Build W from its promised columns, without using the QFT reference."""
    rng = np.random.default_rng(seed)
    ambient = 3 * rank + 2
    identity = np.eye(ambient, dtype=complex)
    source = identity[:, :rank]
    target = identity[:, rank:2 * rank]
    failure = identity[:, 2 * rank:3 * rank]
    transform = _unitary(rng, rank)
    theta = pi / (4 * rounds + 2)
    columns = np.sin(theta) * target @ transform + np.cos(theta) * failure
    completion = null_space(columns.conj().T)
    w = np.column_stack((columns, completion @ _unitary(rng, ambient - rank)))
    np.testing.assert_allclose(w.conj().T @ w, identity, atol=2e-15)
    return source, target, transform, w


def _amplified(w, source, target, rounds, *, include_minus=True):
    identity = np.eye(len(w), dtype=complex)
    source_reflection = 2 * source @ source.conj().T - identity
    target_reflection = 2 * target @ target.conj().T - identity
    iterate = w @ source_reflection @ w.conj().T @ target_reflection
    if include_minus:
        iterate = -iterate
    return np.linalg.matrix_power(iterate, rounds) @ w @ source


@pytest.mark.parametrize("rank,rounds", [(1, 1), (3, 2), (4, 5)])
def test_two_distinct_full_promises_amplify_complex_unitaries_with_correct_phase(rank, rounds):
    source, target, transform, w = _generic_block(rank, rounds, 71 + rank)
    expected = target @ transform
    np.testing.assert_allclose(_amplified(w, source, target, rounds), expected, atol=8e-15)
    # A missing per-round minus sign is not harmless in an isometry comparison.
    without_minus = _amplified(w, source, target, rounds, include_minus=False)
    np.testing.assert_allclose(without_minus, (-1) ** rounds * expected, atol=8e-15)
    if rounds % 2:
        assert np.linalg.norm(without_minus - expected, 2) == pytest.approx(2)


@pytest.mark.parametrize("rank,rounds", [(1, 3), (3, 2), (4, 5)])
def test_source_only_error_bound_allows_order_one_completion_changes(rank, rounds):
    source, target, transform, w = _generic_block(rank, rounds, 102 + rank)
    # This unitary fixes the source pointwise and negates its entire complement.
    alternate = w @ (2 * source @ source.conj().T - np.eye(len(w)))
    assert np.linalg.norm(alternate - w, 2) == pytest.approx(2)
    np.testing.assert_allclose((alternate - w) @ source, 0, atol=0)
    np.testing.assert_allclose(_amplified(alternate, source, target, rounds),
                               target @ transform, atol=1e-14)

    # Then perturb a promised column by a small ambient plane rotation. The
    # completion is still far away; a whole-unitary error bound is unavailable.
    delta = 0.003
    perturbation = np.eye(len(w), dtype=complex)
    perturbation[np.ix_([0, rank], [0, rank])] = [
        [np.cos(delta), -np.sin(delta)], [np.sin(delta), np.cos(delta)]]
    approximate = perturbation @ alternate
    eta = np.linalg.norm((approximate - w) @ source, 2)
    assert 0 < eta < 0.004
    assert np.linalg.norm(approximate - w, 2) > 1.99
    error = np.linalg.norm(_amplified(approximate, source, target, rounds)
                           - target @ transform, 2)
    assert error <= (4 * rounds + 1) * eta + 2e-14


def test_varying_rank_coherent_control_sectors_do_not_add_errors_or_lose_phase():
    blocks = [_generic_block(rank, 3, 200 + rank) for rank in (1, 2, 4)]
    source, target, transform, w = [block_diag(*(block[index] for block in blocks))
                                    for index in range(4)]
    actual = _amplified(w, source, target, 3)
    expected = target @ transform
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    # A reference can be entangled with the sector label; this compares the
    # full matrix with no phase fitting or measurement of the control register.
    reference_dimension = 3
    np.testing.assert_allclose(np.kron(actual, np.eye(reference_dimension)),
                               np.kron(expected, np.eye(reference_dimension)), atol=1e-14)


@pytest.mark.parametrize("seed", [3, 19, 83])
def test_clean_stage_bounds_compose_with_same_dirty_scratch_and_no_reset(seed):
    rng = np.random.default_rng(seed)
    rank, count = 3, 4
    clean = np.eye(2 * rank, dtype=complex)[:, :rank]
    dirty = np.eye(2 * rank, dtype=complex)[:, rank:]
    actual, expected = clean.copy(), clean.copy()
    error_budget = 0.0
    for step in range(count):
        transform = _unitary(rng, rank)
        # An arbitrary off-promise completion is permitted at every stage.
        completion = _unitary(rng, rank)
        ideal_extension = block_diag(transform, np.eye(rank))
        alternate = block_diag(transform, completion)
        delta = 0.001 * (step + 1)
        mixing = np.block([[np.cos(delta) * np.eye(rank), -np.sin(delta) * np.eye(rank)],
                           [np.sin(delta) * np.eye(rank), np.cos(delta) * np.eye(rank)]])
        approximate = mixing @ alternate
        eta = np.linalg.norm((approximate - ideal_extension) @ clean, 2)
        error_budget += eta
        actual = approximate @ actual
        expected = ideal_extension @ expected
        assert np.linalg.norm(actual - expected, 2) <= error_budget + 2e-14
        if step == 0:
            assert np.linalg.norm(dirty.conj().T @ actual, 2) > 0.0009
    # Keep the complete physical output, including leaked scratch. Discarding
    # leakage and renormalizing would not test the theorem's clean-isometry norm.
    assert np.linalg.norm(dirty.conj().T @ actual, 2) > 1e-4
    np.testing.assert_allclose(actual.conj().T @ actual, np.eye(rank), atol=3e-15)


def test_emitted_amplification_schedule_uses_literal_column_adjoints():
    compiled = compile_induction(2, fraction_bits=3, angle_bits=3)
    inverse_column = tuple(gate.inverse() for gate in reversed(compiled.column_gates))
    emitted = compiled.circuit.gates
    cursor = len(compiled.column_gates)
    assert tuple(emitted[:cursor]) == compiled.column_gates
    for _ in range(compiled.rounds):
        for block in (compiled.target_reflection, inverse_column,
                      compiled.source_reflection, compiled.column_gates):
            assert tuple(emitted[cursor:cursor + len(block)]) == block
            cursor += len(block)
        assert emitted[cursor].op == "phase"
        assert emitted[cursor].angle == pi
        cursor += 1
    assert cursor == len(emitted)
