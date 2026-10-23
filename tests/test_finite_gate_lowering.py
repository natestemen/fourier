"""Finite WORD and actual-matrix audits, never projective-error comparisons."""

from collections import Counter
from math import acos, pi, sqrt
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from reversible_gate_ir import Circuit, Gate, inverse_gates  # noqa: E402
from finite_gate_lowering import (  # noqa: E402
    CCX_WORD, FINITE_OPS, RotationSynthesizer, SynthesisFailure, controlled_matrix,
    lower_circuit, one_qubit_word_matrix, ry_matrix, simulate_lowered,
)


def _matrix(word, qubits):
    circuit = Circuit()
    circuit.qubit_count = qubits
    circuit.append(word)
    matrix = np.zeros((1 << qubits, 1 << qubits), dtype=complex)
    for column in range(1 << qubits):
        for row, amplitude in circuit.simulate({column: 1}).items():
            matrix[row, column] = amplitude
    return matrix


@pytest.fixture(scope="module")
def synthesizer():
    return RotationSynthesizer(1e-5, basis_depth=20, max_recursion_degree=4)


def test_exact_ccx_decomposition_has_seven_t_gates_and_no_relative_or_global_phase():
    actual = _matrix(CCX_WORD, 3)
    ideal = np.eye(8, dtype=complex)
    ideal[:, [3, 7]] = ideal[:, [7, 3]]
    np.testing.assert_allclose(actual, ideal, atol=8e-16)
    counts = Counter(gate.op for gate in CCX_WORD)
    assert counts["t"] + counts["tdg"] == 7
    assert counts["cx"] == 6 and counts["h"] == 2
    assert set(counts) <= FINITE_OPS


@pytest.mark.parametrize("angle", [pi / 2, pi, 2 * pi])
def test_exact_half_angle_shortcuts_and_projective_phase_cancellation(angle, synthesizer):
    record = synthesizer.synthesize(angle)
    actual = _matrix(record.controlled_word, 2)
    target = controlled_matrix(ry_matrix(angle))
    np.testing.assert_allclose(actual, target, atol=3e-15)
    assert record.operator_error < 3e-15
    # The control-zero branch is the exact inverse word sandwich, not an
    # independently synthesized inverse or an uncorrected SK global phase.
    np.testing.assert_allclose(actual[np.ix_([0, 2], [0, 2])], np.eye(2), atol=3e-15)
    if angle == pi:
        raw = one_qubit_word_matrix(record.half_word)
        assert np.linalg.norm(raw - ry_matrix(angle / 2), 2) > .1


def test_tolerance_is_certified_by_actual_four_dimensional_matrix_without_phase_fit(synthesizer):
    record = synthesizer.synthesize(.5)
    assert record.meets_tolerance and record.operator_error < 1e-5
    assert record.recursion_degree <= 4
    assert len(record.controlled_word) == 2 * len(record.half_word) + 2
    actual = _matrix(record.controlled_word, 2)
    target = controlled_matrix(ry_matrix(.5))
    error = np.linalg.norm(actual - target, 2)
    assert error == pytest.approx(record.operator_error, abs=3e-12)
    np.testing.assert_allclose(actual, record.full_matrix, atol=3e-12)
    assert set(record.gate_counts) <= FINITE_OPS


def test_negative_public_angles_are_exact_adjoint_words_not_resynthesized(synthesizer):
    positive = synthesizer.synthesize(.5)
    negative = synthesizer.synthesize(-.5)
    assert negative.controlled_word == inverse_gates(positive.controlled_word)
    np.testing.assert_allclose(negative.active_matrix, positive.active_matrix.conj().T, atol=0)
    assert synthesizer.synthesize(.5) is positive


def test_uncontrolled_rotation_uses_clean_auxiliary_and_preserves_global_phase(synthesizer):
    circuit = Circuit()
    target = circuit.allocate(1)[0]
    circuit.ry(pi, target)
    plan = lower_circuit(circuit, synthesizer=synthesizer)
    assert plan.qubit_count == 2 and plan.auxiliary == 1
    materialized = plan.materialize()
    actual = np.zeros((4, 2), dtype=complex)
    for column in range(2):
        for row, amplitude in materialized.simulate({column: 1}).items():
            actual[row, column] = amplitude
    expected = np.zeros_like(actual)
    expected[:2] = ry_matrix(pi)
    np.testing.assert_allclose(actual, expected, atol=3e-15)
    with pytest.raises(ValueError, match="auxiliary"):
        plan.simulate({2: 1})


def test_lazy_counts_and_block_acceleration_match_actual_finite_word_execution(synthesizer):
    circuit = Circuit()
    a, b, target = circuit.allocate(3)
    circuit.h(a)
    circuit.t(b)
    circuit.ccx(a, b, target)
    circuit.ry(.5, target, control=b)
    circuit.ry(pi / 2, a)
    circuit.ry(-.5, target, control=b)
    circuit.global_phase(pi)
    plan = lower_circuit(circuit, synthesizer=synthesizer)
    assert not any(gate.op in {"ccx", "ry", "cry"} for gate in plan.iter_gates())
    materialized = plan.materialize(max_gates=100_000)
    assert plan.gate_counts == materialized.gate_counts
    assert plan.elementary_gate_count == len(materialized.gates) - plan.global_phase_instruction_count
    rng = np.random.default_rng(41)
    vector = rng.normal(size=8) + 1j * rng.normal(size=8)
    vector /= np.linalg.norm(vector)
    initial = dict(enumerate(vector))
    exact_word = materialized.simulate(initial, cutoff=1e-14)
    stats = {}
    accelerated = simulate_lowered(plan, initial, cutoff=1e-14, stats=stats)
    all_indices = set(exact_word) | set(accelerated)
    difference = np.sqrt(sum(abs(exact_word.get(index, 0) - accelerated.get(index, 0)) ** 2
                             for index in all_indices))
    assert difference < 4e-12
    assert stats["block_accelerated"]
    assert stats["finite_elementary_gate_count"] == plan.elementary_gate_count
    analog = circuit.simulate(initial)
    synthesis_difference = np.sqrt(sum(abs(analog.get(index, 0) - accelerated.get(index, 0)) ** 2
                                       for index in set(analog) | set(accelerated)))
    assert synthesis_difference <= plan.operator_error_bound + 1e-12
    assert plan.operator_error_bound < 2.1e-5
    with pytest.raises(ValueError, match="materialization"):
        plan.materialize(max_gates=1)


def test_unmet_tolerance_raises_with_achieved_error_instead_of_silent_success():
    synthesizer = RotationSynthesizer(1e-12, basis_depth=3, max_recursion_degree=0)
    with pytest.raises(SynthesisFailure) as caught:
        synthesizer.synthesize(.5)
    assert caught.value.best.operator_error > caught.value.best.tolerance
    assert .5 not in synthesizer.records


@pytest.mark.parametrize("angle", [.03125, .0625, .125, 2 * acos(1 / sqrt(3))])
def test_degree_five_fallback_certifies_the_four_depth_twenty_precision_misses(angle):
    synthesizer = RotationSynthesizer(1e-5, basis_depth=20, max_recursion_degree=5)
    record = synthesizer.synthesize(angle)
    assert record.recursion_degree == 5
    assert record.operator_error < 1e-7
    assert np.linalg.norm(record.full_matrix - controlled_matrix(ry_matrix(angle)), 2) < 1e-7
    assert len(record.controlled_word) > 90_000


def test_arbitrary_global_phase_metadata_is_not_claimed_as_a_finite_gate():
    circuit = Circuit()
    circuit.global_phase(.123)
    with pytest.raises(ValueError, match="integer-pi"):
        lower_circuit(circuit)


def test_lowered_program_snapshots_source_against_append_and_same_length_edits(synthesizer):
    circuit = Circuit()
    wire = circuit.allocate(1)[0]
    circuit.x(wire)
    plan = lower_circuit(circuit, synthesizer=synthesizer)
    circuit.gates[0] = Gate("z", (0,))
    circuit.z(wire)
    assert tuple(plan.iter_gates()) == (Gate("x", (0,)),)
    assert plan.simulate() == {1: 1}
