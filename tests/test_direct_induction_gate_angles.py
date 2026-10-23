"""Small exhaustive audits for the gate-level direct-induction compiler."""

from fractions import Fraction
from math import atan2, pi, sqrt
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from alternative_qft_probe import addable_rows, partitions  # noqa: E402
from direct_induction_gate_angles import (  # noqa: E402
    apply_ratio_rotation,
    apply_weight_tree,
    compile_branch_state,
    compute_directed_weights,
    compute_ratio_angle_word,
    xor_directed_weights,
)
from direct_induction_precision import downward_weight_numerators  # noqa: E402
from reversible_gate_ir import Circuit, basis_index, word_value  # noqa: E402


def _corner_contents(shape):
    addable = tuple(
        (shape[row - 1] if row <= len(shape) else 0) + 1 - row
        for row in addable_rows(shape)
    )
    removable = tuple(
        length - row
        for row, length in enumerate(shape, 1)
        if length > (shape[row] if row < len(shape) else 0)
    )
    return addable, removable


def _mask(wires):
    return sum(1 << wire.index for wire in wires)


def _flatten(words):
    return tuple(bit for word in words for bit in word)


def _assert_only_kept_wires(index, kept):
    assert index & ~_mask(kept) == 0


def _corner_registers(circuit, capacity, word_width):
    addable = tuple(
        circuit.allocate(word_width, f"x{i}", fresh=True) for i in range(capacity)
    )
    removable = tuple(
        circuit.allocate(word_width, f"y{i}", fresh=True)
        for i in range(capacity - 1)
    )
    addable_active = circuit.allocate(capacity, "x_active", fresh=True)
    removable_active = circuit.allocate(capacity - 1, "y_active", fresh=True)
    return addable, removable, addable_active, removable_active


def _corner_assignment(shape, m, capacity, registers):
    addable_words, removable_words, addable_active, removable_active = registers
    addable, removable = _corner_contents(shape)
    shifted_x = tuple(value + m for value in addable) + (0,) * (capacity - len(addable))
    shifted_y = tuple(value + m for value in removable) + (0,) * (
        capacity - 1 - len(removable)
    )
    assignments = [
        *((word, value) for word, value in zip(addable_words, shifted_x)),
        *((word, value) for word, value in zip(removable_words, shifted_y)),
        (addable_active, (1 << len(addable)) - 1),
        (removable_active, (1 << len(removable)) - 1),
    ]
    return addable, removable, assignments


@pytest.mark.parametrize("m,capacity", [(3, 2), (4, 4)])
def test_directed_products_are_actual_boolean_arithmetic_and_match_reference(m, capacity):
    circuit = Circuit()
    registers = _corner_registers(circuit, capacity, word_width=4)
    weights = compute_directed_weights(
        circuit, *registers, fraction_bits=8
    )
    assert set(circuit.gate_counts) <= {"x", "cx", "ccx"}
    assert weights.report.gate_count == len(circuit.gates)
    assert weights.report.capacity == capacity

    inputs = _flatten(registers[0]) + _flatten(registers[1]) + registers[2] + registers[3]
    kept = inputs + weights.flat_wires
    for shape in partitions(m - 1):
        addable, removable, assignments = _corner_assignment(
            shape, m, capacity, registers
        )
        initial = basis_index(assignments)
        final, phase = circuit.simulate_basis(initial)
        expected = downward_weight_numerators(addable, removable, 8)
        assert tuple(word_value(final, word) for word in weights.words) == (
            expected + (0,) * (capacity - len(expected))
        )
        assert all(word_value(final, word) == value for word, value in assignments)
        assert phase == 1
        _assert_only_kept_wires(final, kept)


def test_weight_xor_oracle_erases_its_external_outputs_with_identical_schedule():
    circuit = Circuit()
    registers = _corner_registers(circuit, 2, word_width=3)
    weights = compute_directed_weights(circuit, *registers, fraction_bits=6)
    erase = xor_directed_weights(
        circuit, *registers, weights.words, fraction_bits=6
    )
    assert erase.gate_count == weights.report.gate_count
    shape = (2,)
    _addable, _removable, assignments = _corner_assignment(shape, 3, 2, registers)
    initial = basis_index(assignments)
    final, phase = circuit.simulate_basis(initial)
    assert final == initial and phase == 1


def test_computed_angle_word_tracks_runtime_masses_without_a_shape_table():
    circuit = Circuit()
    left = circuit.allocate(9, "left", fresh=True)
    right = circuit.allocate(9, "right", fresh=True)
    angle, report = compute_ratio_angle_word(
        circuit, left, right, angle_fraction_bits=8
    )
    assert set(circuit.gate_counts) <= {"x", "cx", "ccx"}
    assert report.gate_count == len(circuit.gates)
    scale = 1 << 8
    pairs = (
        (0, 0), (1, 0), (0, 1), (1, 1), (1, 2), (2, 1),
        (7, 31), (31, 7), (1, 255), (255, 1),
    )
    observed = []
    for left_value, right_value in pairs:
        initial = basis_index(((left, left_value), (right, right_value)))
        final, phase = circuit.simulate_basis(initial)
        encoded = word_value(final, angle)
        observed.append(encoded)
        if right_value == 0:
            assert encoded == 0
        elif left_value == 0:
            assert encoded == round(pi * scale / 2)
        else:
            exact = atan2(sqrt(right_value), sqrt(left_value))
            assert abs(encoded / scale - exact) < 0.02
        assert word_value(final, left) == left_value
        assert word_value(final, right) == right_value
        assert phase == 1
        _assert_only_kept_wires(final, left + right + angle)
    assert len(set(observed)) > 5


def _rotation_amplitudes(circuit, initial, target):
    state = circuit.simulate({initial: 1}, cutoff=1e-13)
    result = {}
    for index, amplitude in state.items():
        result[word_value(index, (target,))] = amplitude
    return state, result


def test_ratio_rotation_has_exact_endpoints_accurate_interior_and_clean_work():
    circuit = Circuit()
    left = circuit.allocate(9, "left", fresh=True)
    right = circuit.allocate(9, "right", fresh=True)
    enabled = circuit.allocate(1, "enabled", fresh=True)[0]
    target = circuit.allocate(1, "target", fresh=True)[0]
    report = apply_ratio_rotation(
        circuit, left, right, target, angle_fraction_bits=8, control=enabled
    )
    assert report.atan_terms == 7
    assert circuit.gate_counts["cry"] == 10  # nine angle bits plus exact endpoint
    assert set(circuit.gate_counts) <= {"x", "cx", "ccx", "cry"}
    kept = left + right + (enabled, target)

    # The (1, D**2-1) cases are exactly the magnitude data reused by a
    # seminormal Young-generator block: cos(theta)=1/abs(D).
    for left_value, right_value in (
        (1, 1), (1, 2), (2, 1), (7, 31), (31, 7), (1, 3), (1, 8), (1, 48)
    ):
        initial = basis_index(((left, left_value), (right, right_value), ((enabled,), 1)))
        state, amplitudes = _rotation_amplitudes(circuit, initial, target)
        assert amplitudes[0].imag == pytest.approx(0, abs=1e-13)
        assert amplitudes[1].imag == pytest.approx(0, abs=1e-13)
        total = left_value + right_value
        assert amplitudes[0].real == pytest.approx(sqrt(left_value / total), abs=0.008)
        assert amplitudes[1].real == pytest.approx(sqrt(right_value / total), abs=0.008)
        assert sum(abs(value) ** 2 for value in amplitudes.values()) == pytest.approx(1)
        for index in state:
            _assert_only_kept_wires(index, kept)

    for left_value, right_value, expected_target in ((0, 0, 0), (3, 0, 0), (0, 3, 1)):
        initial = basis_index(((left, left_value), (right, right_value), ((enabled,), 1)))
        state, amplitudes = _rotation_amplitudes(circuit, initial, target)
        assert amplitudes == pytest.approx({expected_target: 1})
        for index in state:
            _assert_only_kept_wires(index, kept)

    disabled = basis_index(((left, 1), (right, 2)))
    state, amplitudes = _rotation_amplitudes(circuit, disabled, target)
    assert amplitudes == pytest.approx({0: 1})
    for index in state:
        _assert_only_kept_wires(index, kept)


def test_four_leaf_tree_uses_computed_node_angles_and_uncomputes_all_mass_work():
    circuit = Circuit()
    weights = tuple(circuit.allocate(9, f"weight{i}", fresh=True) for i in range(4))
    target = circuit.allocate(2, "branch", fresh=True)
    report = apply_weight_tree(
        circuit, weights, target, angle_fraction_bits=8
    )
    assert len(report.node_reports) == 3
    assert report.capacity == 4 and report.mass_width == 11
    assert circuit.gate_counts["cry"] == 30
    kept = _flatten(weights) + target

    for values in ((10, 20, 30, 40), (17, 0, 0, 0), (0, 0, 5, 0), (1, 2, 0, 0)):
        initial = basis_index(tuple(zip(weights, values)))
        state = circuit.simulate({initial: 1}, cutoff=1e-13)
        amplitudes = {word_value(index, target): amplitude for index, amplitude in state.items()}
        total = sum(values)
        for branch in range(4):
            expected = sqrt(Fraction(values[branch], total)) if total else int(branch == 0)
            assert amplitudes.get(branch, 0).real == pytest.approx(float(expected), abs=0.008)
            assert amplitudes.get(branch, 0).imag == pytest.approx(0, abs=1e-13)
        assert sum(abs(value) ** 2 for value in state.values()) == pytest.approx(1)
        for index in state:
            assert tuple(word_value(index, word) for word in weights) == values
            _assert_only_kept_wires(index, kept)


@pytest.mark.parametrize("m,capacity", [(3, 2), (4, 4)])
def test_end_to_end_branch_compiler_erases_weights_and_prepares_all_small_shapes(m, capacity):
    circuit = Circuit()
    registers = _corner_registers(circuit, capacity, word_width=4)
    target = circuit.allocate(capacity.bit_length() - 1, "branch", fresh=True)
    report = compile_branch_state(
        circuit,
        *registers,
        target,
        fraction_bits=8,
        angle_fraction_bits=8,
    )
    assert report.weight_compute.gate_count == report.weight_erase.gate_count
    assert len(report.tree.node_reports) == capacity - 1
    assert set(circuit.gate_counts) <= {"x", "cx", "ccx", "cry"}
    assert report.gate_count == len(circuit.gates)
    # These intentionally loose caps document that the executable small audit
    # stays polynomial rather than hiding an exponential angle table.
    assert report.gate_count < 500_000
    assert report.qubit_count < 100_000

    inputs = _flatten(registers[0]) + _flatten(registers[1]) + registers[2] + registers[3]
    kept = inputs + target
    for shape in partitions(m - 1):
        addable, removable, assignments = _corner_assignment(
            shape, m, capacity, registers
        )
        initial = basis_index(assignments)
        state = circuit.simulate({initial: 1}, cutoff=1e-13)
        numerators = downward_weight_numerators(addable, removable, 8)
        total = sum(numerators)
        amplitudes = {word_value(index, target): amplitude for index, amplitude in state.items()}
        for branch in range(capacity):
            expected = sqrt(Fraction(numerators[branch], total)) if branch < len(numerators) else 0
            assert amplitudes.get(branch, 0).real == pytest.approx(float(expected), abs=0.008)
            assert amplitudes.get(branch, 0).imag == pytest.approx(0, abs=1e-13)
        assert sum(abs(value) ** 2 for value in state.values()) == pytest.approx(1)
        for index in state:
            _assert_only_kept_wires(index, kept)


def test_public_width_and_alias_contracts_reject_nonuniform_calls():
    circuit = Circuit()
    x = (circuit.allocate(3), circuit.allocate(3))
    y = (circuit.allocate(3),)
    active_x = circuit.allocate(2)
    active_y = circuit.allocate(1)
    with pytest.raises(ValueError, match="power of two"):
        compute_directed_weights(
            circuit, x + (circuit.allocate(3),), y, active_x + (0,), active_y,
            fraction_bits=4,
        )
    with pytest.raises(ValueError, match="capacity minus one"):
        compute_directed_weights(
            circuit, x, (), active_x, (), fraction_bits=4
        )
    with pytest.raises(ValueError, match="at least two"):
        apply_ratio_rotation(circuit, x[0], x[1], y[0][0], angle_fraction_bits=1)
    with pytest.raises(ValueError, match="disjoint from the mass"):
        apply_ratio_rotation(circuit, x[0], x[1], x[0][0], angle_fraction_bits=4)
    with pytest.raises(ValueError, match="length log2"):
        apply_weight_tree(circuit, x, (), angle_fraction_bits=4)
    with pytest.raises(ValueError, match="disjoint from all corner"):
        compile_branch_state(
            circuit, x, y, active_x, active_y, (x[0][0],),
            fraction_bits=4, angle_fraction_bits=4,
        )
