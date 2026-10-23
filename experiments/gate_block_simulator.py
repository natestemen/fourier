"""Exact sparse simulation acceleration for an unchanged elementary gate IR.

Maximal classical-permutation/diagonal runs are evaluated on each complete
physical basis index, with their resulting permutation and phase memoized.
Only H/Ry/CRy branch amplitudes. This is a SIMULATOR optimization: the original
gate trace is neither modified nor replaced for circuit/resource accounting.
Every X/CX/CCX and state-dependent diagonal operation in a block is executed
on a cache miss, including operations on dirty ancillas.

Boolean instructions use predecoded byte offsets and eight-bit masks, not
one enormous Python integer mask per gate. This keeps decoding memory linear
in gate count even when the circuit has tens of thousands of physical wires.
Basis integers are converted to/from a bytearray only at block boundaries.

With cutoff=0 no amplitude is intentionally discarded. Positive cutoff is
applied at block/rotation boundaries; the sum of discarded l2 norms is a
triangle-inequality error bound relative to unpruned unitary evolution. It
does not include floating-point roundoff. No renormalization is performed.

Optional rotation_resolver(gate) returns a unitary 2x2 matrix, or None for the
native matrix. This simulation-only hook can evaluate actual compiled finite
rotation words without changing the stored gate trace. For CRy it replaces
only the controlled-on target action. The caller remains responsible for
certifying that its matrix represents the intended finite-word construction.
"""

from __future__ import annotations

import cmath
from dataclasses import dataclass, field
import math
import sys

from reversible_gate_ir import Circuit, Gate


_CLASSICAL = frozenset(("x", "cx", "ccx", "z", "t", "tdg", "s", "sdg", "phase"))
_ARITY = {"x": 1, "cx": 2, "ccx": 3, "z": 1, "t": 1, "tdg": 1,
          "s": 1, "sdg": 1, "phase": 0, "h": 1, "ry": 1, "cry": 2}
_EIGHTH = math.sqrt(0.5)
_PHASES = (1 + 0j, _EIGHTH + 1j * _EIGHTH, 1j, -_EIGHTH + 1j * _EIGHTH,
           -1 + 0j, -_EIGHTH - 1j * _EIGHTH, -1j, _EIGHTH - 1j * _EIGHTH)


@dataclass(eq=False)
class _ClassicalBlock:
    instructions: tuple
    global_phase: complex
    gate_count: int
    cache: dict[int, tuple[int, complex]] = field(default_factory=dict)

    def evaluate(self, index, byte_count):
        bits = bytearray(index.to_bytes(byte_count, "little"))
        phase_turns = 0
        for instruction in self.instructions:
            opcode = instruction[0]
            if opcode == 0:
                bits[instruction[1]] ^= instruction[2]
            elif opcode == 1:
                if bits[instruction[3]] & instruction[4]:
                    bits[instruction[1]] ^= instruction[2]
            elif opcode == 2:
                if ((bits[instruction[3]] & instruction[4])
                        and (bits[instruction[5]] & instruction[6])):
                    bits[instruction[1]] ^= instruction[2]
            elif bits[instruction[1]] & instruction[2]:
                phase_turns += instruction[3]
        return int.from_bytes(bits, "little"), self.global_phase * _PHASES[phase_turns & 7]


@dataclass(frozen=True)
class _Rotation:
    target: int
    controls: tuple[int, ...]
    matrix: tuple[tuple[complex, complex], tuple[complex, complex]]
    replaced: bool


def _validate_gate(gate):
    if not isinstance(gate, Gate) or gate.op not in _ARITY:
        raise ValueError(f"unsupported elementary gate: {gate!r}")
    if len(gate.wires) != _ARITY[gate.op]:
        raise ValueError(f"wrong operand count for {gate.op}")
    if any(not isinstance(wire, int) or wire < 0 for wire in gate.wires):
        raise ValueError("gate wire indices must be nonnegative integers")
    if len(set(gate.wires)) != len(gate.wires):
        raise ValueError("gate operands must be distinct")
    if gate.op in ("ry", "cry", "phase"):
        if gate.angle is None or not math.isfinite(float(gate.angle)):
            raise ValueError("rotation/global-phase angle must be finite")


def _decode_block(gates):
    instructions = []
    global_phase = 1 + 0j
    phase_steps = {"z": 4, "t": 1, "tdg": -1, "s": 2, "sdg": -2}
    for gate in gates:
        op, wires = gate.op, gate.wires
        if op == "phase":
            # Multiplication avoids overflowing a sum of individually finite
            # phase angles, and these scalar phases commute with the block.
            global_phase *= cmath.exp(1j * float(gate.angle))
            continue
        target = wires[-1]
        target_byte, target_mask = target >> 3, 1 << (target & 7)
        if op == "x":
            instruction = (0, target_byte, target_mask)
        elif op == "cx":
            instruction = (1, target_byte, target_mask, wires[0] >> 3, 1 << (wires[0] & 7))
        elif op == "ccx":
            instruction = (2, target_byte, target_mask, wires[0] >> 3, 1 << (wires[0] & 7),
                           wires[1] >> 3, 1 << (wires[1] & 7))
        else:
            instruction = (3, target_byte, target_mask, phase_steps[op])
        instructions.append(instruction)
    return _ClassicalBlock(tuple(instructions), global_phase, len(gates))


def _unitary_matrix(matrix):
    try:
        rows = tuple(tuple(complex(entry) for entry in row) for row in matrix)
    except (TypeError, ValueError) as error:
        raise ValueError("rotation resolver must return a 2x2 matrix or None") from error
    if len(rows) != 2 or any(len(row) != 2 for row in rows):
        raise ValueError("rotation resolver must return a 2x2 matrix")
    if any(not math.isfinite(z.real) or not math.isfinite(z.imag) for row in rows for z in row):
        raise ValueError("rotation matrix entries must be finite")
    for i in range(2):
        for j in range(2):
            overlap = sum(rows[k][i].conjugate() * rows[k][j] for k in range(2))
            if abs(overlap - int(i == j)) > 1e-10:
                raise ValueError("rotation resolver returned a nonunitary matrix")
    return rows


def _decode_rotation(gate, resolver):
    resolved = None if resolver is None else resolver(gate)
    replaced = resolved is not None
    if resolved is not None:
        matrix = _unitary_matrix(resolved)
    elif gate.op == "h":
        matrix = ((_EIGHTH + 0j, _EIGHTH + 0j), (_EIGHTH + 0j, -_EIGHTH + 0j))
    else:
        cosine, sine = math.cos(float(gate.angle) / 2), math.sin(float(gate.angle) / 2)
        matrix = ((cosine + 0j, -sine + 0j), (sine + 0j, cosine + 0j))
    return _Rotation(gate.wires[-1], gate.wires[:-1], matrix, replaced)


class BlockSimulator:
    """Reusable exact-gate simulator with bounded, cross-call block memoization.

    ``max_cache_bytes`` bounds an estimated cache footprint (default 128 MiB).
    Once full, new results are evaluated but not retained; existing entries
    remain reusable. The estimate includes the basis integers and a generous
    per-entry Python-container allowance. It is not a process-memory limit.
    """

    def __init__(self, circuit_or_gates, qubit_count=None, *, rotation_resolver=None,
                 max_cache_bytes=128 * 1024 * 1024):
        if isinstance(circuit_or_gates, Circuit):
            gates = tuple(circuit_or_gates.gates)
            count = circuit_or_gates.qubit_count if qubit_count is None else qubit_count
        else:
            gates = tuple(circuit_or_gates)
            count = qubit_count
        for gate in gates:
            _validate_gate(gate)
        minimum_count = max((max(gate.wires) + 1 for gate in gates if gate.wires), default=0)
        count = minimum_count if count is None else count
        if not isinstance(count, int) or count < minimum_count:
            raise ValueError("qubit_count must contain all gate operands")
        if not isinstance(max_cache_bytes, int) or max_cache_bytes < 0:
            raise ValueError("max_cache_bytes must be a nonnegative integer")
        self.gates = gates
        self.qubit_count = count
        self.byte_count = (count + 7) // 8
        self.max_cache_bytes = max_cache_bytes
        self.cache_estimated_bytes = 0
        self.cache_entries = 0
        self.steps = []
        self.blocks = []
        self.rotation_replacements = 0
        block_intern = {}
        rotation_intern = {}
        pending = []

        def finish_block():
            if not pending:
                return
            key = tuple(pending)
            block = block_intern.get(key)
            if block is None:
                block = _decode_block(key)
                block_intern[key] = block
                self.blocks.append(block)
            self.steps.append(block)
            pending.clear()

        for gate in gates:
            if gate.op in _CLASSICAL:
                pending.append(gate)
            else:
                finish_block()
                rotation = rotation_intern.get(gate)
                if rotation is None:
                    rotation = _decode_rotation(gate, rotation_resolver)
                    rotation_intern[gate] = rotation
                self.steps.append(rotation)
                self.rotation_replacements += int(rotation.replaced)
        finish_block()
        self.steps = tuple(self.steps)
        self.blocks = tuple(self.blocks)

    def clear_cache(self):
        for block in self.blocks:
            block.cache.clear()
        self.cache_estimated_bytes = self.cache_entries = 0

    def _basis_image(self, block, index, stats):
        cached = block.cache.get(index)
        if cached is not None:
            stats["block_cache_hits"] += 1
            return cached
        result = block.evaluate(index, self.byte_count)
        stats["block_basis_evaluations"] += 1
        stats["decoded_boolean_steps"] += len(block.instructions)
        cost = sys.getsizeof(index) + sys.getsizeof(result[0]) + sys.getsizeof(result[1]) + 192
        if self.cache_estimated_bytes + cost <= self.max_cache_bytes:
            block.cache[index] = result
            self.cache_estimated_bytes += cost
            self.cache_entries += 1
        return result

    @staticmethod
    def _prune(state, cutoff, stats):
        nonzero = sum(amplitude != 0 for amplitude in state.values())
        stats["max_support"] = max(stats["max_support"], nonzero)
        if cutoff == 0:
            return {index: amplitude for index, amplitude in state.items() if amplitude != 0}
        discarded = [abs(amplitude) for amplitude in state.values()
                     if amplitude != 0 and abs(amplitude) <= cutoff]
        if discarded:
            stats["discarded_norm_bound"] += math.hypot(*discarded)
            stats["pruned_states"] += len(discarded)
            stats["prune_events"] += 1
        return {index: amplitude for index, amplitude in state.items() if abs(amplitude) > cutoff}

    def simulate(self, state=None, *, cutoff=0.0, stats=None):
        """Apply the actual gate trace, preserving every physical ancilla bit.

        A reused ``stats`` dictionary accumulates counters and the pruning
        bound across calls. Pass a fresh dictionary for an individual column's
        certificate. Caches are shared even when the statistics dictionary is
        fresh. The return value has the same sparse-state format as Circuit.
        """
        if not math.isfinite(cutoff) or cutoff < 0:
            raise ValueError("cutoff must be finite and nonnegative")
        current = {0: 1 + 0j} if state is None else dict(state)
        if any(not isinstance(index, int) or index < 0 or index.bit_length() > self.qubit_count
               for index in current):
            raise ValueError("state index exceeds the physical register")
        if any(not math.isfinite(complex(value).real) or not math.isfinite(complex(value).imag)
               for value in current.values()):
            raise ValueError("state amplitudes must be finite")
        stats = {} if stats is None else stats
        for key in ("gate_count", "block_applications", "block_basis_evaluations", "block_cache_hits",
                    "decoded_boolean_steps", "rotation_applications", "rotation_replacements",
                    "pruned_states", "prune_events"):
            stats.setdefault(key, 0)
        stats.setdefault("discarded_norm_bound", 0.0)
        stats.setdefault("max_support", sum(amplitude != 0 for amplitude in current.values()))
        stats["max_support"] = max(stats["max_support"], sum(value != 0 for value in current.values()))
        stats["gate_count"] += len(self.gates)
        stats["rotation_replacements"] += self.rotation_replacements
        for step in self.steps:
            if isinstance(step, _ClassicalBlock):
                stats["block_applications"] += 1
                updated = {}
                for index, amplitude in current.items():
                    target, phase = self._basis_image(step, index, stats)
                    updated[target] = amplitude * phase  # A classical block is bijective.
            else:
                stats["rotation_applications"] += 1
                updated = {}
                target_mask = 1 << step.target
                matrix = step.matrix
                for index, amplitude in current.items():
                    if not all((index >> wire) & 1 for wire in step.controls):
                        updated[index] = updated.get(index, 0j) + amplitude
                        continue
                    column = (index >> step.target) & 1
                    zero, one = index & ~target_mask, index | target_mask
                    updated[zero] = updated.get(zero, 0j) + matrix[0][column] * amplitude
                    updated[one] = updated.get(one, 0j) + matrix[1][column] * amplitude
            current = self._prune(updated, cutoff, stats)
        stats["memoized_entries"] = self.cache_entries
        stats["cache_estimated_bytes"] = self.cache_estimated_bytes
        stats["unique_classical_blocks"] = len(self.blocks)
        stats["classical_block_occurrences"] = sum(isinstance(step, _ClassicalBlock) for step in self.steps)
        return current
