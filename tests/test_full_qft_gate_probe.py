"""Whole-circuit Fourier checks, including dirty residuals between stages."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from alternative_qft_probe import (  # noqa: E402
    add_row, addable_rows, outer_boundary, partitions, tableaux,
)
from full_qft_gate_probe import (  # noqa: E402
    append_outer_boundary, audit_full_qft, compile_full_qft, run_experiment,
)
from gate_block_simulator import BlockSimulator  # noqa: E402
from reversible_gate_ir import Circuit, basis_index, word_value  # noqa: E402


def _boundary(level):
    circuit = Circuit()
    width = level.bit_length()
    shape = tuple(circuit.allocate(width) for _ in range(level))
    rows = tuple(circuit.allocate(width) for _ in range(level))
    right_last = circuit.allocate(width)
    data_count = circuit.qubit_count
    append_outer_boundary(circuit, shape, rows, right_last)
    return circuit, shape, rows, right_last, data_count


@pytest.mark.parametrize("level", [2, 3, 4])
def test_boundary_all_valid_shapes_tableaux_and_right_path_extensions(level):
    circuit, shape_register, row_register, right_last, _ = _boundary(level)
    inverse = circuit.inverse()
    for shape in partitions(level - 1):
        for added in addable_rows(shape):
            endpoint = add_row(shape, added)
            for q in tableaux(endpoint):
                initial = basis_index(
                    [(register, shape[i] if i < len(shape) else 0)
                     for i, register in enumerate(shape_register)] + list(zip(row_register, q)))
                output, phase = circuit.simulate_basis(initial)
                expected = basis_index(
                    [(register, endpoint[i] if i < len(endpoint) else 0)
                     for i, register in enumerate(shape_register)]
                    + list(zip(row_register, q)) + [(right_last, added)])
                assert output == expected and phase == 1
                assert inverse.simulate_basis(output) == (initial, 1)
                for r in tableaux(shape):
                    reference = outer_boundary(shape, q, r)
                    assert reference.new_right_path == r + (word_value(output, right_last),)


def test_boundary_added_row_is_not_assumed_to_be_last_left_tableau_row():
    circuit, shape, rows, right_last, _ = _boundary(3)
    initial = basis_index([(shape[0], 2), (rows[0], 1), (rows[1], 2), (rows[2], 1)])
    output, phase = circuit.simulate_basis(initial)
    assert phase == 1
    assert word_value(output, right_last) == 2
    assert word_value(output, rows[-1]) == 1


def test_boundary_is_unitary_on_malformed_data_and_dirty_shared_helpers():
    circuit, _, _, _, data_count = _boundary(3)
    inverse = circuit.inverse()
    cases = [0, (1 << data_count) - 1, (1 << circuit.qubit_count) - 1,
             (1 << (circuit.qubit_count - 1)) | 0b11011110101]
    for initial in cases:
        output, phase = circuit.simulate_basis(initial)
        assert phase == 1
        assert inverse.simulate_basis(output) == (initial, 1)


@pytest.fixture(scope="module")
def full_three():
    compiled = compile_full_qft(3, fraction_bits=5, angle_bits=5)
    return compiled, audit_full_qft(compiled)


def test_full_s3_every_input_column_against_direct_representation_qft(full_three):
    compiled, report = full_three
    assert report["source_columns_checked"] == report["group_order"] == 6
    assert report["clean_operator_error"] < .0441
    assert report["outside_target_operator_norm"] <= report["clean_operator_error"] + 1e-14
    assert report["isometry_defect"] < 3e-12
    assert report["pruning_operator_error_bound"] < 1e-10
    assert set(compiled.circuit.gate_counts) <= {"x", "cx", "ccx", "z", "ry", "cry", "phase"}


def test_full_s3_complex_superposition_and_exact_adjoint(full_three):
    _, report = full_three
    assert report["complex_superposition_error"] <= report["clean_operator_error"] + 1e-12
    assert report["complex_superposition_linearity_defect"] < 1e-12
    assert report["complex_superposition_inverse_roundtrip_error"] < 2e-12
    assert report["inverse_roundtrip_pruning_bound"] < 1e-10


def test_work_is_reused_and_complete_boundaries_are_charged(full_three):
    compiled, report = full_three
    assert len(compiled.circuit.gates) == 4 + sum(
        stage.induction_gate_count + stage.boundary_gate_count for stage in compiled.stages)
    assert report["shared_work_qubits"] == max(
        max(stage.induction_work_qubits, stage.boundary_work_qubits) for stage in compiled.stages)
    assert report["shared_work_qubits"] < sum(stage.induction_work_qubits for stage in compiled.stages)
    assert all(stage.boundary_gate_count > 0 for stage in compiled.stages)
    assert not report["intermediate_projection_or_reset"]
    first, second = compiled.stages
    first_work = {i for g in compiled.circuit.gates[first.start:first.induction_end]
                  for i in g.wires if i >= compiled.work_start}
    second_work = {i for g in compiled.circuit.gates[second.start:second.induction_end]
                   for i in g.wires if i >= compiled.work_start}
    assert len(first_work & second_work) == min(len(first_work), len(second_work))


def test_dirty_stage_residual_is_carried_and_not_projected_before_reuse(full_three):
    compiled, _ = full_three
    source = compiled.input_index((1, 1, 1))
    cut = compiled.stages[0].end
    prefix = BlockSimulator(compiled.circuit.gates[:cut], compiled.circuit.qubit_count)
    suffix = BlockSimulator(compiled.circuit.gates[cut:], compiled.circuit.qubit_count)
    intermediate = prefix.simulate({source: 1}, cutoff=1e-13)
    used_digits = sum(1 << wire.index for word in compiled.digits[:2] for wire in word)
    forbidden = compiled.work_mask() | used_digits
    projected = {index: amplitude for index, amplitude in intermediate.items() if not index & forbidden}
    residual = np.sqrt(sum(abs(amplitude) ** 2 for index, amplitude in intermediate.items() if index & forbidden))
    assert residual > 1e-4
    assert abs(sum(abs(z) ** 2 for z in intermediate.values()) - 1) < 2e-13
    unprojected_output = suffix.simulate(intermediate, cutoff=1e-13)
    projected_output = suffix.simulate(projected, cutoff=1e-13)
    difference = np.sqrt(sum(abs(unprojected_output.get(i, 0) - projected_output.get(i, 0)) ** 2
                             for i in unprojected_output.keys() | projected_output.keys()))
    # A unitary preserves the discarded component's norm. The actual compiler
    # retains it; projection would change the full Fourier circuit measurably.
    assert abs(difference - residual) < 2e-12
    assert abs(sum(abs(z) ** 2 for z in unprojected_output.values()) - 1) < 2e-13
    direct = BlockSimulator(compiled.circuit).simulate({source: 1}, cutoff=1e-13)
    assert max(abs(direct.get(i, 0) - unprojected_output.get(i, 0))
               for i in direct.keys() | unprojected_output.keys()) < 2e-13


def test_single_element_group_is_exact_and_needs_no_work():
    compiled = compile_full_qft(1)
    report = audit_full_qft(compiled)
    assert report["group_order"] == 1
    assert report["shared_work_qubits"] == 0
    assert report["clean_operator_error"] == 0
    assert report["complex_superposition_inverse_roundtrip_error"] == 0


def test_compilation_does_not_enumerate_group_elements_tableaux_or_shapes(monkeypatch):
    import alternative_qft_probe
    import direct_induction_gate_probe

    def forbidden(*args, **kwargs):
        raise AssertionError("compiler called an enumerated reference")

    for name in ("partitions", "tableaux", "fourier_basis", "fourier_reference", "transversal_words"):
        monkeypatch.setattr(alternative_qft_probe, name, forbidden)
    monkeypatch.setattr(direct_induction_gate_probe, "partitions", forbidden)
    compiled = compile_full_qft(2, fraction_bits=3, angle_bits=3)
    assert len(compiled.stages) == 1


def test_whole_s2_has_actual_clifford_t_lowering_and_inverse():
    from finite_gate_lowering import FINITE_OPS

    report = run_experiment(2, fraction_bits=3, angle_bits=3, rotation_tolerance=3e-5)
    finite, native = report["finite"], report["native"]
    assert finite["fully_finite_gate_lowered"]
    assert set(finite["finite_gate_counts"]) <= FINITE_OPS
    assert not {"ry", "cry", "ccx"} & set(finite["finite_gate_counts"])
    assert finite["logical_qubits"] == native["logical_qubits"] + 1
    assert finite["clean_operator_error"] < native["clean_operator_error"] + finite["synthesis_operator_error_bound"]
    assert finite["complex_superposition_inverse_roundtrip_error"] < 3e-9


@pytest.mark.parametrize("degree,bits", [(0, 7), (-1, 7), (2, 1), (2, 25)])
def test_invalid_public_parameters_rejected(degree, bits):
    with pytest.raises(ValueError):
        compile_full_qft(degree, fraction_bits=bits)
