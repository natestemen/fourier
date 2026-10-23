"""Lazy, phase-safe Clifford+T lowering of the elementary reversible IR.

Only public one-qubit angles enter Solovay--Kitaev synthesis.  No classical
table indexed by a quantum shape or arithmetic value occurs.  Each returned
rotation contains its actual finite gate word; counts come from those words.

For CRy(phi), synthesize V approximately Ry(phi/2), retaining ONLY the H/T/Tdg
word and deliberately dropping Qiskit's global-phase metadata. Chronological
V, CX, V^dagger, CX is exactly identity on control zero, and X V^dagger X V on
control one. Thus V's projective phase cancels. Uncontrolled rotations use the
same construction with one shared auxiliary set to one, then restored to zero.
Negative angles lower to the EXACT ADJOINT of the positive-angle word.

simulate_lowered accelerates finite rotation blocks using matrices multiplied
from their emitted words; all arithmetic still executes its individual Boolean
gates. It does not replace an arithmetic routine by a classical oracle. CCX
blocks use their independently checked exact 7-T decomposition. The extra
rotation auxiliary is an invariant clean register in this accelerated path.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Iterable, Iterator, Sequence

import numpy as np

from reversible_gate_ir import Circuit, Gate, inverse_gates


FINITE_OPS = frozenset({"x", "z", "h", "s", "sdg", "t", "tdg", "cx", "phase"})


def ry_matrix(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle / 2), math.sin(angle / 2)
    return np.array([[cosine, -sine], [sine, cosine]], dtype=complex)


def one_qubit_word_matrix(word: Iterable[Gate]) -> np.ndarray:
    """Multiply the actual finite word, without discarding its global phase."""
    matrices = {
        "x": np.array([[0, 1], [1, 0]], complex),
        "z": np.diag([1, -1]).astype(complex),
        "h": np.array([[1, 1], [1, -1]], complex) / math.sqrt(2),
        "s": np.diag([1, 1j]), "sdg": np.diag([1, -1j]),
        "t": np.diag([1, np.exp(1j * math.pi / 4)]),
        "tdg": np.diag([1, np.exp(-1j * math.pi / 4)]),
    }
    result = np.eye(2, dtype=complex)
    for gate in word:
        if gate.op == "phase":
            result *= np.exp(1j * float(gate.angle))
        elif gate.op in matrices and gate.wires == (0,):
            result = matrices[gate.op] @ result
        else:
            raise ValueError("expected a finite one-qubit word on local wire zero")
    return result


def _remap(word: Iterable[Gate], wires: Sequence[int]) -> tuple[Gate, ...]:
    return tuple(Gate(gate.op, tuple(wires[index] for index in gate.wires), gate.angle)
                 for gate in word)


def controlled_matrix(active: np.ndarray) -> np.ndarray:
    """Little-endian local control=0, target=1, including exact control-zero I."""
    result = np.eye(4, dtype=complex)
    result[np.ix_([1, 3], [1, 3])] = active
    return result


# Local wires: first control=0, second control=1, target=2.
CCX_WORD = (
    Gate("h", (2,)), Gate("cx", (1, 2)), Gate("tdg", (2,)),
    Gate("cx", (0, 2)), Gate("t", (2,)), Gate("cx", (1, 2)),
    Gate("tdg", (2,)), Gate("cx", (0, 2)), Gate("t", (1,)),
    Gate("t", (2,)), Gate("h", (2,)), Gate("cx", (0, 1)),
    Gate("t", (0,)), Gate("tdg", (1,)), Gate("cx", (0, 1)),
)


@dataclass(frozen=True)
class SynthesizedRotation:
    angle: float
    half_word: tuple[Gate, ...]
    controlled_word: tuple[Gate, ...]
    active_matrix: np.ndarray
    operator_error: float
    tolerance: float
    basis_depth: int
    recursion_degree: int

    @property
    def meets_tolerance(self) -> bool:
        return self.operator_error <= self.tolerance

    @property
    def gate_counts(self) -> dict[str, int]:
        return dict(Counter(gate.op for gate in self.controlled_word))

    @property
    def full_matrix(self) -> np.ndarray:
        return controlled_matrix(self.active_matrix)


class SynthesisFailure(RuntimeError):
    def __init__(self, best: SynthesizedRotation):
        self.best = best
        super().__init__(f"public rotation {best.angle:.17g}: achieved actual operator error "
                         f"{best.operator_error:.6g}, exceeding tolerance {best.tolerance:.6g} "
                         f"at maximum allowed recursion degree; no uncertified lowering returned")


@lru_cache(maxsize=8)
def _sk_engine(depth: int):
    from qiskit.synthesis import SolovayKitaevDecomposition
    return SolovayKitaevDecomposition(basis_gates=["h", "t", "tdg"], depth=depth)


def _record(angle, half_word, tolerance, depth, degree):
    half_word = tuple(half_word)
    finite_matrix = one_qubit_word_matrix(half_word)
    pauli_x = np.array([[0, 1], [1, 0]], complex)
    active = pauli_x @ finite_matrix.conj().T @ pauli_x @ finite_matrix
    mapped = _remap(half_word, (1,))
    word = mapped + (Gate("cx", (0, 1)),) + inverse_gates(mapped) + (Gate("cx", (0, 1)),)
    error = float(np.linalg.norm(active - ry_matrix(angle), 2))
    return SynthesizedRotation(angle, half_word, word, active, error, tolerance, depth, degree)


class RotationSynthesizer:
    """Tolerance-checked public dictionary; failure is explicit, never hidden."""

    def __init__(self, tolerance: float = 1e-5, *, basis_depth: int = 20,
                 max_recursion_degree: int = 4):
        if not 0 < tolerance < 1 or basis_depth < 1 or not 0 <= max_recursion_degree <= 5:
            raise ValueError("invalid synthesis precision/depth/degree")
        self.tolerance = tolerance
        self.basis_depth = basis_depth
        self.max_recursion_degree = max_recursion_degree
        self.records: dict[float, SynthesizedRotation] = {}

    def synthesize(self, angle: float) -> SynthesizedRotation:
        angle = float(angle)
        if not math.isfinite(angle):
            raise ValueError("nonfinite public angle")
        if angle in self.records:
            return self.records[angle]
        if angle < 0:
            positive = self.synthesize(-angle)
            record = SynthesizedRotation(angle, inverse_gates(positive.half_word),
                inverse_gates(positive.controlled_word), positive.active_matrix.conj().T,
                positive.operator_error, self.tolerance, self.basis_depth, positive.recursion_degree)
            self.records[angle] = record
            return record

        # Ry(k*pi/4), the desired HALF rotation here, is projectively S H T^k H S†.
        integer = round(angle / (math.pi / 2))
        if abs(angle - integer * (math.pi / 2)) <= 1e-14:
            half_word = (() if integer == 0 else
                         (Gate("sdg", (0,)), Gate("h", (0,)))
                         + (Gate("t", (0,)),) * (integer % 8)
                         + (Gate("h", (0,)), Gate("s", (0,))))
            record = _record(angle, half_word, self.tolerance, self.basis_depth, -1)
            if record.meets_tolerance:
                self.records[angle] = record
                return record
        best = None
        engine = _sk_engine(self.basis_depth)
        for degree in range(self.max_recursion_degree + 1):
            circuit = engine.run(ry_matrix(angle / 2), degree)
            # Intentionally omit circuit.global_phase: it cancels in the sandwich.
            word = []
            for instruction in circuit.data:
                name = instruction.operation.name
                if name not in {"h", "t", "tdg"} or len(instruction.qubits) != 1:
                    raise ValueError(f"unexpected Solovay--Kitaev instruction {name}")
                word.append(Gate(name, (0,)))
            record = _record(angle, word, self.tolerance, self.basis_depth, degree)
            if best is None or record.operator_error < best.operator_error:
                best = record
            if record.meets_tolerance:
                self.records[angle] = record
                return record
        assert best is not None
        raise SynthesisFailure(best)


class LoweredProgram:
    """Lazy complete finite program, sharing public templates across occurrences."""

    def __init__(self, circuit: Circuit, synthesizer: RotationSynthesizer):
        # Snapshot only references to immutable Gate objects, never expanded
        # finite words. Later appends or same-length edits of the source cannot
        # invalidate lazy counts or change the represented program.
        self.source = Circuit()
        self.source.qubit_count = circuit.qubit_count
        self.source.labels = dict(circuit.labels)
        self.source.gates = tuple(circuit.gates)
        self.synthesizer = synthesizer
        self.source_gate_count = len(self.source.gates)
        self.auxiliary = circuit.qubit_count if any(gate.op == "ry" for gate in circuit.gates) else None
        self.qubit_count = circuit.qubit_count + int(self.auxiliary is not None)
        self.rotation_matrices: dict[float, np.ndarray] = {}
        counts = Counter()
        self.operator_error_bound = 0.0
        for gate, repetitions in Counter(circuit.gates).items():
            if gate.op in {"ry", "cry"}:
                record = synthesizer.synthesize(float(gate.angle))
                self.rotation_matrices[float(gate.angle)] = record.active_matrix
                self.operator_error_bound += repetitions * record.operator_error
                counts.update({op: repetitions * count for op, count in record.gate_counts.items()})
                if gate.op == "ry":
                    counts["x"] += 2 * repetitions
            elif gate.op == "ccx":
                counts.update({op: repetitions * count for op, count in Counter(g.op for g in CCX_WORD).items()})
            elif gate.op in FINITE_OPS:
                if gate.op == "phase" and not math.isclose(float(gate.angle) / math.pi,
                        round(float(gate.angle) / math.pi), abs_tol=1e-12):
                    raise ValueError("only integer-pi global phase metadata is allowed in finite lowering")
                counts[gate.op] += repetitions
            else:
                raise ValueError(f"cannot lower instruction {gate.op!r}")
        self.gate_counts = dict(counts)
        self.elementary_gate_count = sum(count for op, count in counts.items() if op != "phase")
        self.global_phase_instruction_count = counts.get("phase", 0)

    def _unchanged(self):
        if len(self.source.gates) != self.source_gate_count:
            raise RuntimeError("source circuit changed after finite lowering")

    def iter_blocks(self) -> Iterator[tuple[Gate, ...]]:
        self._unchanged()
        for gate in self.source.gates:
            if gate.op == "ccx":
                yield _remap(CCX_WORD, gate.wires)
            elif gate.op == "cry":
                record = self.synthesizer.records[float(gate.angle)]
                yield _remap(record.controlled_word, gate.wires)
            elif gate.op == "ry":
                record = self.synthesizer.records[float(gate.angle)]
                yield ((Gate("x", (self.auxiliary,)),)
                       + _remap(record.controlled_word, (self.auxiliary, gate.wires[0]))
                       + (Gate("x", (self.auxiliary,)),))
            else:
                yield (gate,)

    def iter_gates(self) -> Iterator[Gate]:
        for block in self.iter_blocks():
            yield from block

    def materialize(self, *, max_gates: int = 1_000_000) -> Circuit:
        if self.elementary_gate_count + self.global_phase_instruction_count > max_gates:
            raise ValueError("expanded program exceeds explicit materialization limit; use iter_gates")
        circuit = Circuit()
        circuit.qubit_count = self.qubit_count
        circuit.append(self.iter_gates())
        return circuit

    def simulate(self, state=None, *, cutoff: float = 0.0, stats: dict | None = None):
        """Block-accelerated execution of matrices from the actual finite words."""
        self._unchanged()
        if self.auxiliary is not None and state is not None and any(
                index & (1 << self.auxiliary) for index in state):
            raise ValueError("rotation auxiliary must enter in its invariant zero state")
        result = self.source.simulate(state, cutoff=cutoff, stats=stats,
                                      rotation_matrices=self.rotation_matrices)
        if stats is not None:
            stats["finite_elementary_gate_count"] = self.elementary_gate_count
            stats["rotation_synthesis_error_bound"] = self.operator_error_bound
            stats["block_accelerated"] = True
        return result


def lower_circuit(circuit: Circuit, *, tolerance: float = 1e-5, basis_depth: int = 20,
                  max_recursion_degree: int = 4,
                  synthesizer: RotationSynthesizer | None = None) -> LoweredProgram:
    return LoweredProgram(circuit, synthesizer or RotationSynthesizer(tolerance,
        basis_depth=basis_depth, max_recursion_degree=max_recursion_degree))


def simulate_lowered(circuit: Circuit | LoweredProgram, state=None, *, cutoff: float = 0.0,
                     stats: dict | None = None, tolerance: float = 1e-5,
                     basis_depth: int = 20, max_recursion_degree: int = 4):
    program = circuit if isinstance(circuit, LoweredProgram) else lower_circuit(circuit,
        tolerance=tolerance, basis_depth=basis_depth, max_recursion_degree=max_recursion_degree)
    return program.simulate(state, cutoff=cutoff, stats=stats)
