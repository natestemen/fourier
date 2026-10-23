"""Explicit reversible/rotation gate IR and small sparse-state simulator.

There are no arithmetic or lookup-oracle instructions: BooleanBuilder emits
only X, CX and CCX. Words are little endian. Integer Bits are ONLY constants
zero and one, never wire indices. Public-angle Ry/controlled-Ry remain analog
instructions until a separate finite-gate lowering pass is applied.

Temporary reuse is explicit: release asserts that the caller has restored a
wire to zero. BooleanBuilder does this only after reversing its own complete
gate trace. Do not mutate a builder's inputs while its computation is live.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import cmath
import heapq
import math
from typing import Iterable, TypeAlias


@dataclass(frozen=True, order=True)
class Wire:
    index: int


Bit: TypeAlias = Wire | int
Word: TypeAlias = tuple[Bit, ...]


@dataclass(frozen=True)
class Gate:
    op: str
    wires: tuple[int, ...] = ()
    angle: float | None = None

    def inverse(self) -> Gate:
        if self.op in {"ry", "cry", "phase"}:
            return Gate(self.op, self.wires, -float(self.angle))
        conjugates = {"t": "tdg", "tdg": "t", "s": "sdg", "sdg": "s"}
        return Gate(conjugates.get(self.op, self.op), self.wires, self.angle)


class Circuit:
    def __init__(self):
        self.gates: list[Gate] = []
        self.qubit_count = 0
        self.labels: dict[int, str] = {}
        self._free: set[int] = set()
        self._free_heap: list[int] = []

    def allocate(self, count: int, label: str = "", *, fresh: bool = False) -> tuple[Wire, ...]:
        if count < 0:
            raise ValueError("negative allocation")
        result = []
        for position in range(count):
            if self._free and not fresh:
                while self._free_heap[0] not in self._free:
                    heapq.heappop(self._free_heap)
                index = heapq.heappop(self._free_heap)
                self._free.remove(index)
            else:
                index = self.qubit_count
                self.qubit_count += 1
            if label:
                self.labels[index] = f"{label}[{position}]"
            result.append(Wire(index))
        return tuple(result)

    def reserve(self, wires: Iterable[Wire]) -> tuple[Wire, ...]:
        """Prevent reuse; return formerly-free wires for an eventual release."""
        formerly_free = []
        for wire in wires:
            self._check(wire)
            if wire.index in self._free:
                formerly_free.append(wire)
            self._free.discard(wire.index)
        return tuple(formerly_free)

    def release(self, wires: Iterable[Wire]) -> None:
        """Caller asserts every released wire has returned to zero."""
        for wire in wires:
            self._check(wire)
            if wire.index in self._free:
                raise ValueError("wire already released")
            self._free.add(wire.index)
            heapq.heappush(self._free_heap, wire.index)

    def _check(self, wire: Wire) -> int:
        if not isinstance(wire, Wire) or not 0 <= wire.index < self.qubit_count:
            raise ValueError("wire is not allocated in this circuit")
        return wire.index

    def _emit(self, op: str, wires: Iterable[Wire] = (), angle=None) -> None:
        indices = tuple(self._check(wire) for wire in wires)
        if len(set(indices)) != len(indices):
            raise ValueError("gate operands must be distinct")
        if angle is not None and not math.isfinite(float(angle)):
            raise ValueError("angle must be finite")
        self.gates.append(Gate(op, indices, None if angle is None else float(angle)))

    def x(self, target: Wire) -> None:
        self._emit("x", (target,))

    def cx(self, control: Wire, target: Wire) -> None:
        self._emit("cx", (control, target))

    def ccx(self, first: Wire, second: Wire, target: Wire) -> None:
        if first == second:
            self.cx(first, target)
        else:
            self._emit("ccx", (first, second, target))

    def z(self, target: Wire) -> None:
        self._emit("z", (target,))

    def h(self, target: Wire) -> None:
        self._emit("h", (target,))

    def t(self, target: Wire) -> None:
        self._emit("t", (target,))

    def tdg(self, target: Wire) -> None:
        self._emit("tdg", (target,))

    def s(self, target: Wire) -> None:
        self._emit("s", (target,))

    def sdg(self, target: Wire) -> None:
        self._emit("sdg", (target,))

    def ry(self, angle: float, target: Wire, control: Bit | None = None) -> None:
        if control == 0 or float(angle) == 0:
            return
        if control is None or control == 1:
            self._emit("ry", (target,), angle)
        elif isinstance(control, Wire):
            self._emit("cry", (control, target), angle)
        else:
            raise ValueError("control must be a Wire or a Boolean constant")

    def global_phase(self, angle: float) -> None:
        if float(angle) != 0:
            self._emit("phase", angle=angle)

    def xor_into(self, source: Bit, target: Wire) -> None:
        if isinstance(source, Wire):
            self.cx(source, target)
        elif source == 1:
            self.x(target)
        elif source != 0:
            raise ValueError("integer Bits must be zero or one")

    def swap(self, left: Wire, right: Wire) -> None:
        if left != right:
            self.cx(left, right)
            self.cx(right, left)
            self.cx(left, right)

    def append(self, other: Circuit | Iterable[Gate]) -> None:
        source = list(other.gates if isinstance(other, Circuit) else other)
        if isinstance(other, Circuit):
            self.qubit_count = max(self.qubit_count, other.qubit_count)
        for gate in source:
            if not isinstance(gate, Gate):
                raise TypeError("append expects Gate objects")
            if gate.wires:
                self.qubit_count = max(self.qubit_count, max(gate.wires) + 1)
                # Appending an arbitrary trace does not assert its work is clean.
                self._free.difference_update(gate.wires)
            self.gates.append(gate)

    def inverse(self) -> Circuit:
        result = Circuit()
        result.qubit_count = self.qubit_count
        result.labels = dict(self.labels)
        result.gates = [gate.inverse() for gate in reversed(self.gates)]
        return result

    @property
    def gate_counts(self) -> dict[str, int]:
        return dict(Counter(gate.op for gate in self.gates))

    def simulate(self, state: dict[int, complex] | None = None, *, cutoff: float = 0.0,
                 stats: dict | None = None, rotation_matrices: dict | None = None) -> dict[int, complex]:
        """Apply every emitted gate, retaining the entire physical basis index."""
        if cutoff < 0:
            raise ValueError("negative cutoff")
        current = {0: 1.0 + 0j} if state is None else dict(state)
        if stats is not None:
            stats.setdefault("discarded_norm_bound", 0.0)
            stats.setdefault("max_support", len(current))
            stats.setdefault("gate_count", 0)
        if any(not isinstance(index, int) or index < 0 or index.bit_length() > self.qubit_count
               for index in current):
            raise ValueError("state index exceeds the physical register")
        for gate in self.gates:
            if stats is not None:
                stats["gate_count"] += 1
            updated: dict[int, complex] = {}

            def accumulate(index, amplitude):
                updated[index] = updated.get(index, 0j) + amplitude

            if gate.op == "phase":
                phase = cmath.exp(1j * float(gate.angle))
                current = {index: phase * amplitude for index, amplitude in current.items()}
                continue
            target = gate.wires[-1]
            target_mask = 1 << target
            controls_mask = sum(1 << control for control in gate.wires[:-1])
            if gate.op in {"ry", "cry"}:
                cosine, sine = math.cos(float(gate.angle) / 2), math.sin(float(gate.angle) / 2)
                rotation_matrix = (rotation_matrices[float(gate.angle)]
                                   if rotation_matrices is not None else None)
            if gate.op in {"x", "cx", "ccx"}:
                # These instructions are permutations: distinct input indices
                # cannot collide, so no amplitude aggregation is necessary.
                updated = {(index ^ target_mask if index & controls_mask == controls_mask else index): amplitude
                           for index, amplitude in current.items()}
            for index, amplitude in (() if gate.op in {"x", "cx", "ccx"} else current.items()):
                controls_on = index & controls_mask == controls_mask
                if gate.op in {"x", "cx", "ccx"}:
                    accumulate(index ^ target_mask if controls_on else index, amplitude)
                elif gate.op in {"z", "t", "tdg", "s", "sdg"}:
                    angles = {"z": math.pi, "t": math.pi / 4, "tdg": -math.pi / 4,
                              "s": math.pi / 2, "sdg": -math.pi / 2}
                    phase = (-1 if gate.op == "z" else cmath.exp(1j * angles[gate.op]))
                    accumulate(index, amplitude * (phase if index & target_mask else 1))
                elif gate.op == "h":
                    scale = amplitude / math.sqrt(2)
                    accumulate(index & ~target_mask, scale)
                    accumulate(index | target_mask, -scale if index & target_mask else scale)
                elif gate.op in {"ry", "cry"}:
                    if not controls_on:
                        accumulate(index, amplitude)
                        continue
                    if rotation_matrix is None:
                        accumulate(index, cosine * amplitude)
                        accumulate(index ^ target_mask, (-sine if index & target_mask else sine) * amplitude)
                    else:
                        column = int(bool(index & target_mask))
                        accumulate(index, rotation_matrix[column, column] * amplitude)
                        accumulate(index ^ target_mask, rotation_matrix[1 - column, column] * amplitude)
                else:
                    raise ValueError(f"unknown gate opcode {gate.op!r}")
            if stats is not None:
                stats["max_support"] = max(stats["max_support"], sum(amplitude != 0 for amplitude in updated.values()))
                stats["discarded_norm_bound"] += math.sqrt(sum(
                    abs(amplitude) ** 2 for amplitude in updated.values() if abs(amplitude) <= cutoff))
            current = {index: amplitude for index, amplitude in updated.items()
                       if abs(amplitude) > cutoff}
        return current

    def simulate_basis(self, index: int = 0) -> tuple[int, complex]:
        """Fast exact-basis path for Boolean and diagonal gate instructions.

        Rejects a superposition-producing gate rather than silently discarding
        branches. A disabled controlled rotation is allowed.
        """
        if index < 0 or index.bit_length() > self.qubit_count:
            raise ValueError("state index exceeds the physical register")
        phase = 1.0 + 0j
        diagonal_angles = {"t": math.pi / 4, "tdg": -math.pi / 4,
                           "s": math.pi / 2, "sdg": -math.pi / 2}
        for gate in self.gates:
            if gate.op == "phase":
                phase *= cmath.exp(1j * float(gate.angle))
                continue
            mask = 1 << gate.wires[-1]
            if gate.op in {"x", "cx", "ccx"}:
                controls = sum(1 << control for control in gate.wires[:-1])
                if index & controls == controls:
                    index ^= mask
            elif gate.op == "z":
                if index & mask:
                    phase = -phase
            elif gate.op in diagonal_angles:
                if index & mask:
                    phase *= cmath.exp(1j * diagonal_angles[gate.op])
            elif gate.op == "cry" and not index & (1 << gate.wires[0]):
                continue
            else:
                raise ValueError("simulate_basis encountered a superposition-producing gate")
        return index, phase


class BooleanBuilder:
    """Bit-blasted acyclic Boolean computation with clean inverse/reuse.

    Every result is a constant, an unchanged input, or a fresh temporary.
    copy_then_uncompute requires independently allocated output targets.
    External circuit gates are not part of this builder's inverse trace.
    """

    def __init__(self, circuit: Circuit):
        self.circuit = circuit
        self.temporary_wires: list[Wire] = []
        self.compute_gates: list[Gate] = []
        self._closed = False

    def _fresh(self) -> Wire:
        if self._closed:
            raise RuntimeError("builder is already uncomputed")
        wire = self.circuit.allocate(1, "bool")[0]
        self.temporary_wires.append(wire)
        return wire

    def _call(self, method, *args) -> None:
        start = len(self.circuit.gates)
        method(*args)
        self.compute_gates.extend(self.circuit.gates[start:])

    @staticmethod
    def const(value: int, width: int) -> Word:
        if width < 0 or value < 0:
            raise ValueError("unsigned constants require nonnegative value and width")
        return tuple((value >> position) & 1 for position in range(width))

    @staticmethod
    def _bit(bit: Bit) -> None:
        if not isinstance(bit, Wire) and bit not in (0, 1):
            raise ValueError("integer Bits must be zero or one")

    @staticmethod
    def _word(word: Iterable[Bit], width: int) -> Word:
        bits = tuple(word)
        return (bits + (0,) * width)[:width]

    def bnot(self, bit: Bit) -> Bit:
        self._bit(bit)
        if isinstance(bit, int):
            return 1 - bit
        result = self._fresh()
        self._call(self.circuit.x, result)
        self._call(self.circuit.cx, bit, result)
        return result

    def bxor(self, left: Bit, right: Bit) -> Bit:
        self._bit(left)
        self._bit(right)
        if left == right:
            return 0
        if left == 0:
            return right
        if right == 0:
            return left
        if left == 1:
            return self.bnot(right)
        if right == 1:
            return self.bnot(left)
        result = self._fresh()
        self._call(self.circuit.cx, left, result)
        self._call(self.circuit.cx, right, result)
        return result

    def band(self, left: Bit, right: Bit) -> Bit:
        self._bit(left)
        self._bit(right)
        if left == 0 or right == 0:
            return 0
        if left == 1 or left == right:
            return right
        if right == 1:
            return left
        result = self._fresh()
        self._call(self.circuit.ccx, left, right, result)
        return result

    def bor(self, left: Bit, right: Bit) -> Bit:
        self._bit(left)
        self._bit(right)
        if left == 1 or right == 1:
            return 1
        if left == 0 or left == right:
            return right
        if right == 0:
            return left
        result = self._fresh()
        self._call(self.circuit.cx, left, result)
        self._call(self.circuit.cx, right, result)
        self._call(self.circuit.ccx, left, right, result)
        return result

    def all_bits(self, bits: Iterable[Bit]) -> Bit:
        result: Bit = 1
        for bit in bits:
            result = self.band(result, bit)
        return result

    def any_bits(self, bits: Iterable[Bit]) -> Bit:
        result: Bit = 0
        for bit in bits:
            result = self.bor(result, bit)
        return result

    def eq(self, left: Iterable[Bit], right: Iterable[Bit]) -> Bit:
        left, right = tuple(left), tuple(right)
        width = max(len(left), len(right))
        return self.all_bits(self.bnot(self.bxor(a, b))
                             for a, b in zip(self._word(left, width), self._word(right, width)))

    def ult(self, left: Iterable[Bit], right: Iterable[Bit]) -> Bit:
        left, right = tuple(left), tuple(right)
        width = max(len(left), len(right))
        borrow: Bit = 0
        for a, b in zip(self._word(left, width), self._word(right, width)):
            borrow = self.bor(self.band(self.bnot(a), b),
                              self.band(self.bnot(self.bxor(a, b)), borrow))
        return borrow

    def ule(self, left: Iterable[Bit], right: Iterable[Bit]) -> Bit:
        return self.bnot(self.ult(right, left))

    def mux(self, select: Bit, true_word: Iterable[Bit], false_word: Iterable[Bit]) -> Word:
        self._bit(select)
        true_word, false_word = tuple(true_word), tuple(false_word)
        width = max(len(true_word), len(false_word))
        true_word, false_word = self._word(true_word, width), self._word(false_word, width)
        if select == 0:
            return false_word
        if select == 1:
            return true_word
        return tuple(self.bxor(f, self.band(select, self.bxor(t, f)))
                     for t, f in zip(true_word, false_word))

    def add(self, left: Iterable[Bit], right: Iterable[Bit], width: int | None = None) -> Word:
        left, right = tuple(left), tuple(right)
        width = max(len(left), len(right)) if width is None else width
        carry: Bit = 0
        result = []
        for a, b in zip(self._word(left, width), self._word(right, width)):
            different = self.bxor(a, b)
            result.append(self.bxor(different, carry))
            carry = self.bor(self.band(a, b), self.band(different, carry))
        return tuple(result)

    def sub(self, left: Iterable[Bit], right: Iterable[Bit], width: int | None = None) -> Word:
        left, right = tuple(left), tuple(right)
        width = max(len(left), len(right)) if width is None else width
        borrow: Bit = 0
        result = []
        for a, b in zip(self._word(left, width), self._word(right, width)):
            different = self.bxor(a, b)
            result.append(self.bxor(different, borrow))
            borrow = self.bor(self.band(self.bnot(a), b),
                              self.band(self.bnot(different), borrow))
        return tuple(result)

    def mul(self, left: Iterable[Bit], right: Iterable[Bit], width: int | None = None) -> Word:
        left, right = tuple(left), tuple(right)
        width = len(left) + len(right) if width is None else width
        result: Word = (0,) * width
        for shift, bit in enumerate(right[:width]):
            partial = (0,) * shift + tuple(self.band(a, bit) for a in left[:width - shift])
            result = self.add(result, partial, width)
        return result

    def divmod(self, dividend: Iterable[Bit], divisor: Iterable[Bit],
               width: int | None = None) -> tuple[Word, Word]:
        dividend, divisor = tuple(dividend), tuple(divisor)
        width = max(len(dividend), len(divisor)) if width is None else width
        internal = max(len(dividend), len(divisor), width)
        dividend = self._word(dividend, internal)
        divisor = self.mux(self.any_bits(divisor), self._word(divisor, internal + 1),
                           self.const(1, internal + 1))
        remainder: Word = (0,) * (internal + 1)
        quotient: list[Bit] = [0] * internal
        for position in range(internal - 1, -1, -1):
            shifted = (dividend[position],) + remainder[:internal]
            enough = self.bnot(self.ult(shifted, divisor))
            remainder = self.mux(enough, self.sub(shifted, divisor, internal + 1), shifted)
            quotient[position] = enough
        return tuple(quotient[:width]), remainder[:width]

    def uncompute(self) -> None:
        if self._closed:
            raise RuntimeError("builder is already uncomputed")
        self.circuit.append(gate.inverse() for gate in reversed(self.compute_gates))
        self.circuit.release(self.temporary_wires)
        self._closed = True

    def copy_then_uncompute(self, bits: Iterable[Bit], targets: Iterable[Wire]) -> None:
        bits, targets = tuple(bits), tuple(targets)
        if len(bits) != len(targets):
            raise ValueError("copy widths differ")
        if set(targets) & set(self.temporary_wires):
            raise ValueError("output targets must not be builder temporaries")
        for bit, target in zip(bits, targets):
            self.circuit.xor_into(bit, target)
        self.uncompute()


def word_value(index: int, word: Iterable[Bit]) -> int:
    """Decode a basis index for diagnostics; not a circuit instruction."""
    return sum((((index >> bit.index) & 1) if isinstance(bit, Wire) else bit) << position
               for position, bit in enumerate(word))


def basis_index(assignments: Iterable[tuple[Iterable[Wire], int]]) -> int:
    """Construct a simulator input index for disjoint little-endian words."""
    result = 0
    used = set()
    for word, value in assignments:
        word = tuple(word)
        if value < 0 or value >= 1 << len(word):
            raise ValueError("basis value does not fit word")
        for position, wire in enumerate(word):
            if wire.index in used:
                raise ValueError("basis assignments overlap")
            used.add(wire.index)
            result |= ((value >> position) & 1) << wire.index
    return result


def inverse_gates(gates: Iterable[Gate]) -> tuple[Gate, ...]:
    return tuple(gate.inverse() for gate in reversed(tuple(gates)))


def apply_gates(gates: Iterable[Gate], state: dict[int, complex] | None = None,
                *, qubit_count: int | None = None, cutoff: float = 0.0,
                stats: dict | None = None) -> dict[int, complex]:
    """Simulate a gate slice without requiring its original Circuit owner."""
    circuit = Circuit()
    circuit.append(gates)
    circuit.qubit_count = max(circuit.qubit_count, qubit_count or 0,
                              max((index.bit_length() for index in state), default=0)
                              if state is not None else 0)
    return circuit.simulate(state, cutoff=cutoff, stats=stats)
