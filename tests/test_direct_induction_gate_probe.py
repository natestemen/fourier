"""Integration contracts for actual gate-emitted induction components."""

from math import isqrt
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from alternative_qft_probe import addable_rows, partitions  # noqa: E402
from direct_induction_gate_probe import (  # noqa: E402
    append_uniform_tag, audit_induction, compile_induction, compute_corner_registers,
)
from reversible_gate_ir import Circuit, basis_index, word_value  # noqa: E402


@pytest.mark.parametrize("m", range(2, 7))
def test_coherently_extracted_corners_match_independent_diagram_scan(m):
    circuit = Circuit()
    shape_registers = tuple(circuit.allocate(m.bit_length()) for _ in range(m))
    kmax = (1 + isqrt(8 * m - 7)) // 2
    capacity = 1 << (kmax - 1).bit_length()
    corners = compute_corner_registers(circuit, shape_registers, capacity)
    for shape in partitions(m - 1):
        initial = basis_index((register, shape[i] if i < len(shape) else 0)
                              for i, register in enumerate(shape_registers))
        output, phase = circuit.simulate_basis(initial)
        rows = addable_rows(shape)
        expected_x = [(shape[r - 1] if r <= len(shape) else 0) + 1 - r + m for r in rows]
        expected_y = [length - r + m for r, length in enumerate(shape, 1)
                      if length > (shape[r] if r < len(shape) else 0)]
        assert [word_value(output, word) for word in corners.addable][:len(rows)] == expected_x
        assert [word_value(output, word) for word in corners.removable][:len(rows) - 1] == expected_y
        assert [word_value(output, word) for word in corners.rows][:len(rows)] == list(rows)
        assert [word_value(output, (bit,)) for bit in corners.addable_active] == [1] * len(rows) + [0] * (capacity - len(rows))
        mask = sum(1 << wire.index for wire in corners.owned)
        assert output & ~mask == initial and phase == 1
        reverse = Circuit()
        reverse.qubit_count = circuit.qubit_count
        reverse.append(gate.inverse() for gate in reversed(corners.compute_gates))
        recovered, phase = reverse.simulate_basis(output)
        assert recovered == initial and phase == 1


@pytest.mark.parametrize("m", [2, 3, 4, 5, 7])
def test_public_uniform_tag_has_correct_full_unitary_inverse(m):
    circuit = Circuit()
    tag = circuit.allocate(m.bit_length())
    append_uniform_tag(circuit, tag, m)
    output = circuit.simulate()
    expected = {basis_index(((tag, i),)): 1 / np.sqrt(m) for i in range(1, m + 1)}
    assert set(output) == set(expected)
    assert max(abs(output[index] - expected[index]) for index in expected) < 2e-15
    # Check the specified completion, not just its prepared first column.
    inverse = circuit.inverse()
    for value in range(1 << len(tag)):
        initial = basis_index(((tag, value),))
        restored = inverse.simulate(circuit.simulate({initial: 1.}), cutoff=1e-14)
        assert set(restored) == {initial}
        assert abs(restored[initial] - 1) < 3e-15


@pytest.mark.parametrize("m,bits,tolerance,columns", [(2, 3, .059, 2), (3, 5, .044, 6)])
def test_complete_emitted_induction_on_every_source_column(m, bits, tolerance, columns):
    from gate_block_simulator import BlockSimulator

    compiled = compile_induction(m, fraction_bits=bits, angle_bits=bits)
    audit = audit_induction(compiled, cutoff=1e-13, simulator=BlockSimulator(compiled.circuit))
    assert audit["source_columns_checked"] == columns
    assert audit["clean_operator_error"] < tolerance
    assert audit["outside_target_operator_norm"] <= audit["clean_operator_error"] + 1e-14
    assert audit["isometry_defect"] < 2e-12
    assert audit["pruning_operator_error_bound"] < 1e-10
    assert set(compiled.circuit.gate_counts) <= {"x", "cx", "ccx", "z", "ry", "cry", "phase"}
    # Count the actual schedule, including ALL complete promise reflections.
    assert len(compiled.circuit.gates) == (
        (2 * compiled.rounds + 1) * len(compiled.column_gates)
        + compiled.rounds * (len(compiled.source_reflection) + len(compiled.target_reflection) + 1))


def test_complete_two_box_induction_has_actual_finite_gate_lowering():
    from finite_gate_lowering import FINITE_OPS, lower_circuit
    from gate_block_simulator import BlockSimulator

    compiled = compile_induction(2, fraction_bits=3, angle_bits=3)
    lowered = lower_circuit(compiled.circuit, tolerance=3e-5, max_recursion_degree=5)
    simulator = BlockSimulator(lowered.source, qubit_count=lowered.qubit_count,
        rotation_resolver=lambda gate: lowered.rotation_matrices.get(gate.angle)
        if gate.op in ("ry", "cry") else None)
    audit = audit_induction(compiled, cutoff=1e-13, simulator=simulator)
    assert set(lowered.gate_counts) <= FINITE_OPS
    assert not {"ry", "cry", "ccx"} & set(lowered.gate_counts)
    assert lowered.qubit_count == compiled.circuit.qubit_count + 1
    assert audit["clean_operator_error"] < .059 + lowered.operator_error_bound
    assert audit["isometry_defect"] < 2e-9
    assert audit["pruning_operator_error_bound"] < 1e-10
