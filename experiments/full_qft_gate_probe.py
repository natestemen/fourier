"""Entire small symmetric-group QFT as one unprojected emitted gate circuit.

Input is the canonical transversal word, not a one-line permutation. Each
induction acts coherently on the shape register. Exact Boolean boundaries
compute the new endpoint, append the correct added row to the right tableau,
erase the old endpoint, and reuse one common work pool at the next stage.
There is no measurement, postselection, work reset, or intermediate projection.
In particular, dirty approximation residuals enter subsequent stages unchanged.

Enumeration occurs only in the audit: its independent Fourier matrix uses
Young matrices for bubble-sorted group permutations, not induction matrices.
This deliberately unoptimized small-instance compiler does not demonstrate
the asymptotic resource bound or a practical advantage.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial

import numpy as np

from direct_induction_gate_bookkeeping import append_cover_row_xor, append_endpoint_xor
from direct_induction_gate_probe import compile_induction
from reversible_gate_ir import BooleanBuilder, Circuit, Gate, basis_index


def _flat(words):
    return tuple(bit for word in words for bit in word)


def append_outer_boundary(circuit, shape, rows, right_last):
    """Exact reversible boundary, clean on the ideal induction target promise.

    Invalid/dirty inputs still undergo the same unitary gate sequence. Scratch
    release is an allocator annotation, not a reset, projection, or simulation
    operation. ``right_last`` is initially blank on the valid input promise.
    """
    m, width = len(shape), len(shape[0])
    if len(rows) != m or any(len(word) != width for word in shape + rows):
        raise ValueError("boundary shapes and rows must have matching capacities")
    if len(right_last) != width:
        raise ValueError("right tableau slot has the wrong word width")
    circuit.reserve(_flat(shape + rows) + tuple(right_last))
    endpoint = tuple(circuit.allocate(width, "new-endpoint") for _ in range(m))
    append_endpoint_xor(circuit, rows, endpoint)
    append_cover_row_xor(circuit, shape, endpoint, right_last)
    # Reconstruct lambda using only mu and the newly appended right-path row.
    # Do not use Q's last row: it need not be the box added to the old shape.
    builder = BooleanBuilder(circuit)
    old = []
    for r, word in enumerate(endpoint, 1):
        decrement = builder.eq(right_last, builder.const(r, width))
        old.append(builder.sub(word, (decrement,) + (0,) * (width - 1), width))
    builder.copy_then_uncompute(_flat(tuple(old)), _flat(shape))
    for old_word, new_word in zip(shape, endpoint):
        for old_bit, new_bit in zip(old_word, new_word):
            circuit.swap(old_bit, new_bit)
    circuit.release(_flat(endpoint))


def _boundary_circuit(level, width):
    circuit = Circuit()
    shape = tuple(circuit.allocate(width) for _ in range(level))
    rows = tuple(circuit.allocate(width) for _ in range(level))
    right_last = circuit.allocate(width)
    data = _flat(shape + rows) + right_last
    append_outer_boundary(circuit, shape, rows, right_last)
    return circuit, data


def _remap(gates, mapping):
    """Intern repeated mapped gates; the actual chronological trace is kept."""
    cache = {}
    result = []
    for gate in gates:
        mapped = cache.get(gate)
        if mapped is None:
            mapped = Gate(gate.op, tuple(mapping[i] for i in gate.wires), gate.angle)
            cache[gate] = mapped
        result.append(mapped)
    return result


@dataclass(frozen=True)
class QFTStage:
    level: int
    start: int
    induction_end: int
    end: int
    induction_work_qubits: int
    boundary_work_qubits: int
    induction_gate_count: int
    boundary_gate_count: int


@dataclass
class CompiledQFT:
    degree: int
    fraction_bits: int
    angle_bits: int
    circuit: Circuit
    digits: tuple
    shape: tuple
    left: tuple
    right: tuple
    work_start: int
    stages: tuple[QFTStage, ...]

    def input_index(self, word):
        if len(word) != self.degree or any(not 1 <= i <= m for m, i in enumerate(word, 1)):
            raise ValueError("not a valid canonical transversal word")
        return basis_index(zip(self.digits, word))

    def output_index(self, shape, left, right):
        assignments = [(register, shape[i] if i < len(shape) else 0)
                       for i, register in enumerate(self.shape)]
        assignments += list(zip(self.left, left)) + list(zip(self.right, right))
        return basis_index(assignments)

    def reference_columns(self):
        # Reference-only enumeration and matrix evaluation. Neither compiler
        # path calls these functions or depends on their answers.
        from alternative_qft_probe import fourier_basis, fourier_reference, transversal_words

        sources = tuple(self.input_index(word) for word in transversal_words(self.degree))
        targets = tuple(self.output_index(*entry) for entry in fourier_basis(self.degree))
        return sources, targets, fourier_reference(self.degree)

    def work_mask(self):
        return ((1 << self.circuit.qubit_count) - 1) ^ ((1 << self.work_start) - 1)


def compile_full_qft(degree=3, *, fraction_bits=7, angle_bits=7):
    """Emit the full recurrence with one physical work pool and persistent data."""
    if not isinstance(degree, int) or degree < 1:
        raise ValueError("degree must be a positive integer")
    if not 2 <= fraction_bits <= 24 or not 2 <= angle_bits <= 20:
        raise ValueError("prototype precision outside the supported audit range")
    circuit = Circuit()
    width = degree.bit_length()
    arrays = lambda label: tuple(circuit.allocate(width, label) for _ in range(degree))
    digits, shape, left, right = (arrays(name) for name in ("input-digit", "shape", "left", "right"))
    work_start = circuit.qubit_count
    # S_1 has a single input digit 1 and a single output Fourier label.
    for word in (digits[0], shape[0], left[0], right[0]):
        circuit.x(word[0])
    stages = []
    for m in range(2, degree + 1):
        start = len(circuit.gates)
        induction = compile_induction(m, fraction_bits=fraction_bits, angle_bits=angle_bits)
        local_width = len(induction.tag)
        local_data = _flat(induction.shape + induction.rows) + induction.tag
        global_data = (_flat(tuple(word[:local_width] for word in shape[:m] + left[:m]))
                       + digits[m - 1][:local_width])
        mapping = {a.index: b.index for a, b in zip(local_data, global_data)}
        local_work_start = len(local_data)
        work_count = induction.circuit.qubit_count - local_work_start
        mapping.update((i, work_start + i - local_work_start)
                       for i in range(local_work_start, induction.circuit.qubit_count))
        circuit.append(_remap(induction.circuit.gates, mapping))
        induction_end = len(circuit.gates)
        boundary, boundary_data = _boundary_circuit(m, width)
        global_boundary_data = _flat(shape[:m] + left[:m]) + right[m - 1]
        mapping = {a.index: b.index for a, b in zip(boundary_data, global_boundary_data)}
        boundary_work_start = len(boundary_data)
        boundary_work_count = boundary.qubit_count - boundary_work_start
        mapping.update((i, work_start + i - boundary_work_start)
                       for i in range(boundary_work_start, boundary.qubit_count))
        circuit.append(_remap(boundary.gates, mapping))
        stages.append(QFTStage(m, start, induction_end, len(circuit.gates), work_count,
                               boundary_work_count, induction_end - start,
                               len(circuit.gates) - induction_end))
        # Reuse means remapping the SAME physical wires. No state is erased,
        # reset, restricted to a promise, or projected at this Python boundary.
    return CompiledQFT(degree, fraction_bits, angle_bits, circuit, digits, shape,
                       left, right, work_start, tuple(stages))


def _state_distance(first, second):
    return float(np.sqrt(sum(abs(first.get(i, 0) - second.get(i, 0)) ** 2
                             for i in first.keys() | second.keys())))


def audit_full_qft(compiled, *, cutoff=1e-13, simulator=None, inverse_simulator=None,
                   check_roundtrip=True):
    """Audit every group input, one complex superposition, and exact-adjoint return.

    The spectral norm comparison includes every final work bit. It certifies
    arbitrary superpositions relative to the computed floating-point columns,
    not merely basis-state probabilities. Pruning is bounded separately;
    floating-point roundoff is not interval-certified.
    """
    from gate_block_simulator import BlockSimulator

    sources, targets, reference = compiled.reference_columns()
    simulator = BlockSimulator(compiled.circuit) if simulator is None else simulator
    outputs, pruning, supports = [], [], []
    for source in sources:
        stats = {}
        outputs.append(simulator.simulate({source: 1 + 0j}, cutoff=cutoff, stats=stats))
        pruning.append(stats.get("discarded_norm_bound", 0.0))
        supports.append(stats.get("max_support", len(outputs[-1])))
    ambient = sorted(set(targets).union(*(output.keys() for output in outputs)))
    positions = {index: j for j, index in enumerate(ambient)}
    actual = np.zeros((len(ambient), len(sources)), complex)
    ideal = np.zeros_like(actual)
    for j, output in enumerate(outputs):
        for index, amplitude in output.items():
            actual[positions[index], j] = amplitude
        for i, target in enumerate(targets):
            ideal[positions[target], j] = reference[i, j]
    target_set = set(targets)
    dirty = [j for j, index in enumerate(ambient) if index not in target_set]
    # Complex phases ensure this tests amplitudes/linearity, not probabilities.
    coefficients = np.asarray([complex(j + 1, (-1) ** j * (len(sources) - j))
                               for j in range(len(sources))])
    coefficients /= np.linalg.norm(coefficients)
    initial = dict(zip(sources, coefficients))
    stats = {}
    superposed = simulator.simulate(initial, cutoff=cutoff, stats=stats)
    expected_superposed = dict(zip(targets, reference @ coefficients))
    column_superposed = {}
    for coefficient, output in zip(coefficients, outputs):
        for index, amplitude in output.items():
            column_superposed[index] = column_superposed.get(index, 0j) + coefficient * amplitude
    result = {
        "degree": compiled.degree,
        "fraction_bits": compiled.fraction_bits,
        "angle_bits": compiled.angle_bits,
        "source_columns_checked": len(sources),
        "group_order": factorial(compiled.degree),
        "logical_qubits": simulator.qubit_count,
        "persistent_data_qubits": compiled.work_start,
        "shared_work_qubits": compiled.circuit.qubit_count - compiled.work_start,
        "native_instruction_count": len(compiled.circuit.gates),
        "native_gate_counts": compiled.circuit.gate_counts,
        "stage_resources": [
            {"level": stage.level, "induction_gate_count": stage.induction_gate_count,
             "boundary_gate_count": stage.boundary_gate_count,
             "induction_work_qubits": stage.induction_work_qubits,
             "boundary_work_qubits": stage.boundary_work_qubits} for stage in compiled.stages],
        "clean_operator_error": float(np.linalg.norm(actual - ideal, 2)),
        "outside_target_operator_norm": float(np.linalg.norm(actual[dirty], 2)) if dirty else 0.0,
        "isometry_defect": float(np.linalg.norm(actual.conj().T @ actual - np.eye(len(sources)), 2)),
        "pruning_operator_error_bound": float(np.linalg.norm(pruning)),
        "maximum_intermediate_support": max(supports),
        "maximum_final_support": max(map(len, outputs)),
        "simulation_cutoff": cutoff,
        "complex_superposition_error": _state_distance(superposed, expected_superposed),
        "complex_superposition_linearity_defect": _state_distance(superposed, column_superposed),
        "complex_superposition_pruning_bound": stats.get("discarded_norm_bound", 0.0),
        "intermediate_projection_or_reset": False,
        "fully_finite_gate_lowered": False,
    }
    if check_roundtrip:
        if inverse_simulator is None:
            inverse_simulator = BlockSimulator(compiled.circuit.inverse())
        reverse_stats = {}
        returned = inverse_simulator.simulate(superposed, cutoff=cutoff, stats=reverse_stats)
        result["complex_superposition_inverse_roundtrip_error"] = _state_distance(returned, initial)
        result["inverse_roundtrip_pruning_bound"] = (
            stats.get("discarded_norm_bound", 0.0) + reverse_stats.get("discarded_norm_bound", 0.0))
    return result


def run_experiment(degree=3, *, fraction_bits=9, angle_bits=9, finite=True,
                   rotation_tolerance=1e-5, cutoff=1e-13):
    """Deterministic report; no environment-sensitive timings are included."""
    import platform
    from gate_block_simulator import BlockSimulator

    compiled = compile_full_qft(degree, fraction_bits=fraction_bits, angle_bits=angle_bits)
    result = {
        "material_passport": {
            "origin_skill": "academic-research-suite / experiment-agent",
            "mode": "implementation-and-reproducibility", "date": "2026-10-23",
            "verification_status": "complete-group-input-column simulation",
            "version": "full-qft-gates-v1"},
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "scope": f"Entire S_{degree} QFT on canonical transversal inputs, including every outer boundary.",
        "reference": "Young orthogonal matrices evaluated by bubble sorting each group permutation; no induction recurrence.",
        "simulation": "Actual emitted reversible blocks; finite rotation matrices multiplied from actual finite gate words.",
        "limitations": ["Small-instance correctness check, not an asymptotic resource fit or practical speedup.",
                        "Floating-point roundoff is not included in the pruning bound.",
                        "Unoptimized retained-history arithmetic; qubits exclude error correction.",
                        "Input conversion from one-line permutations is not compiled."],
        "native": audit_full_qft(compiled, cutoff=cutoff),
    }
    if finite:
        import qiskit
        from finite_gate_lowering import lower_circuit

        result["environment"]["qiskit"] = qiskit.__version__
        lowered = lower_circuit(compiled.circuit, tolerance=rotation_tolerance,
                                basis_depth=20, max_recursion_degree=5)
        # Missing finite words must fail loudly, never fall back to an analog
        # rotation in a report labelled as a fully finite-gate simulation.
        resolver = lambda gate: lowered.rotation_matrices[gate.angle] if gate.op in ("ry", "cry") else None
        simulator = BlockSimulator(lowered.source, qubit_count=lowered.qubit_count,
                                   rotation_resolver=resolver)
        inverse = BlockSimulator(tuple(g.inverse() for g in reversed(lowered.source.gates)),
                                 qubit_count=lowered.qubit_count, rotation_resolver=resolver)
        checked = audit_full_qft(compiled, cutoff=cutoff, simulator=simulator, inverse_simulator=inverse)
        checked.update({
            "fully_finite_gate_lowered": True,
            "finite_elementary_gate_count": lowered.elementary_gate_count,
            "finite_gate_counts": lowered.gate_counts,
            "t_and_tdg_count": lowered.gate_counts.get("t", 0) + lowered.gate_counts.get("tdg", 0),
            "synthesis_operator_error_bound": lowered.operator_error_bound,
            "requested_per_rotation_error": rotation_tolerance,
        })
        result["finite"] = checked
    return result


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--fraction-bits", type=int, default=9)
    parser.add_argument("--angle-bits", type=int, default=9)
    parser.add_argument("--native-only", action="store_true")
    parser.add_argument("--rotation-tolerance", type=float, default=1e-5)
    parser.add_argument("--cutoff", type=float, default=1e-13)
    args = parser.parse_args()
    print(json.dumps(run_experiment(args.degree, fraction_bits=args.fraction_bits,
        angle_bits=args.angle_bits, finite=not args.native_only,
        rotation_tolerance=args.rotation_tolerance, cutoff=args.cutoff), indent=2, sort_keys=True))
