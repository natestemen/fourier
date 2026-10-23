"""Gate-level direct induction, with arithmetic rather than shape-angle tables.

This experimental compiler emits only the elementary operations of
reversible_gate_ir: Boolean X/CX/CCX, phase gates, and public-angle Ry/CRy.
Every shape-dependent coefficient, cache, predicate and angle is computed
by those Boolean gates. Analog public rotations are explicitly reported;
their Clifford+T lowering is a separate pass, never counted as free T gates.

The intended validation sizes are M=2 and M=3. Compilation itself is not a
large-n benchmark and the deliberately straightforward bit-blasting retains
substantial arithmetic history. The standalone theorem, not a small-size
fit to these counts, supplies the asymptotic bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, ceil, isqrt, pi, sqrt
from time import perf_counter

import numpy as np

from alternative_qft_probe import induction_bases, induction_reference, partitions
from reversible_gate_ir import BooleanBuilder, Circuit, Gate, Wire, basis_index


def _flat(words):
    return tuple(bit for word in words for bit in word)


def _inverse(gates):
    return tuple(gate.inverse() for gate in reversed(gates))


def _append_clean_inverse(circuit, gates):
    """Replay a clean XOR oracle's inverse without leaking allocator ownership."""
    touched = tuple(Wire(index) for index in {i for gate in gates for i in gate.wires})
    formerly_free = circuit.reserve(touched)
    circuit.append(_inverse(gates))
    # These were zero scratch before replay and are zero again afterward.
    circuit.release(formerly_free)


def _phase_bit(circuit, bit):
    if bit == 1:
        circuit.global_phase(pi)
    elif isinstance(bit, Wire):
        circuit.z(bit)


def _controlled_swap(circuit, control, left, right):
    for a, b in zip(left, right):
        if control == 1:
            circuit.swap(a, b)
        elif isinstance(control, Wire):
            circuit.cx(a, b)
            circuit.ccx(control, b, a)
            circuit.cx(a, b)


def _compact_corners(builder, records):
    """Stable public bitonic schedule: active records first, then origin."""
    values = list(records)
    size = len(values)
    assert size and size & (size - 1) == 0
    block = 2
    while block <= size:
        stride = block // 2
        while stride:
            for i in range(size):
                j = i ^ stride
                if j <= i:
                    continue
                a, b = values[i], values[j]
                ka, kb = a[3] + (builder.bnot(a[0]),), b[3] + (builder.bnot(b[0]),)
                swap = builder.ult(kb, ka) if i & block == 0 else builder.ult(ka, kb)
                left = (builder.mux(swap, (b[0],), (a[0],))[0],) + tuple(
                    builder.mux(swap, b[field], a[field]) for field in (1, 2, 3))
                right = (builder.mux(swap, (a[0],), (b[0],))[0],) + tuple(
                    builder.mux(swap, a[field], b[field]) for field in (1, 2, 3))
                values[i], values[j] = left, right
            stride //= 2
        block *= 2
    return values


@dataclass(frozen=True)
class CornerRegisters:
    addable: tuple
    removable: tuple
    addable_active: tuple
    removable_active: tuple
    rows: tuple
    owned: tuple
    compute_gates: tuple[Gate, ...]


def compute_corner_registers(circuit, shape, capacity):
    """Fixed scan/compaction of a quantum padded shape; contents offset by M."""
    m, width = len(shape), len(shape[0])
    content_width = (2 * m + 1).bit_length()
    arrays = lambda count, bits, label: tuple(circuit.allocate(bits, label) for _ in range(count))
    x = arrays(capacity, content_width, "addable")
    y = arrays(capacity - 1, content_width, "removable")
    active_x = circuit.allocate(capacity, "addable-active")
    active_y = circuit.allocate(capacity - 1, "removable-active")
    rows = arrays(capacity, width, "addable-row")
    owned = _flat(x + y + rows) + active_x + active_y
    start = len(circuit.gates)
    builder = BooleanBuilder(circuit)
    size = 1 << (max(m, capacity) - 1).bit_length()
    add, remove = [], []
    for r in range(size):
        if r < m:
            a = 1 if r == 0 else builder.ult(shape[r], shape[r - 1])
            below = shape[r + 1] if r + 1 < m else builder.const(0, width)
            d = builder.ult(below, shape[r])
            xv = builder.add(shape[r], builder.const(m - r, content_width), content_width)
            yv = builder.add(shape[r], builder.const(m - r - 1, content_width), content_width)
            row = builder.const(r + 1, width)
        else:
            a = d = 0
            xv = yv = builder.const(0, content_width)
            row = builder.const(0, width)
        origin = builder.const(r, max(1, (size - 1).bit_length()))
        add.append((a, xv, row, origin))
        remove.append((d, yv, row, origin))
    add, remove = _compact_corners(builder, add), _compact_corners(builder, remove)
    outputs = (_flat(tuple(record[1] for record in add[:capacity]))
               + _flat(tuple(record[1] for record in remove[:capacity - 1]))
               + _flat(tuple(record[2] for record in add[:capacity]))
               + tuple(record[0] for record in add[:capacity])
               + tuple(record[0] for record in remove[:capacity - 1]))
    builder.copy_then_uncompute(outputs, owned)
    return CornerRegisters(x, y, active_x, active_y, rows, owned,
                           tuple(circuit.gates[start:]))


def _ordinal_to_row(circuit, ordinal, row_target, corners):
    builder = BooleanBuilder(circuit)
    result = [0] * len(row_target)
    for i, row in enumerate(corners.rows):
        match = builder.eq(ordinal, builder.const(i, len(ordinal)))
        for b, value in enumerate(row):
            result[b] = builder.bxor(result[b], builder.band(match, value))
    builder.copy_then_uncompute(result, row_target)
    builder = BooleanBuilder(circuit)
    inverse = [0] * len(ordinal)
    for i, row in enumerate(corners.rows):
        match = builder.band(corners.addable_active[i], builder.eq(row_target, row))
        for b in range(len(ordinal)):
            if i >> b & 1:
                inverse[b] = builder.bxor(inverse[b], match)
    builder.copy_then_uncompute(inverse, ordinal)


def append_branching(circuit, shape, last_row, *, fraction_bits, angle_bits):
    from direct_induction_gate_angles import compute_directed_weights, apply_weight_tree

    m = len(shape)
    max_corners = (1 + isqrt(8 * m - 7)) // 2
    capacity = 1 << (max_corners - 1).bit_length()
    ordinal = circuit.allocate((capacity - 1).bit_length(), "corner-ordinal")
    corners = compute_corner_registers(circuit, shape, capacity)
    start = len(circuit.gates)
    weights = compute_directed_weights(circuit, corners.addable, corners.removable,
        corners.addable_active, corners.removable_active, fraction_bits=fraction_bits)
    weight_gates = tuple(circuit.gates[start:])
    report = apply_weight_tree(circuit, weights.words, ordinal,
                              angle_fraction_bits=angle_bits)
    _ordinal_to_row(circuit, ordinal, last_row, corners)
    _append_clean_inverse(circuit, weight_gates)
    circuit.release(_flat(weights.words))
    _append_clean_inverse(circuit, corners.compute_gates)
    circuit.release(corners.owned)
    circuit.release(ordinal)
    return report


def _axial_data(builder, rows, cache, j):
    width = len(rows[0]) + 2
    positive = builder.add(cache[j + 1], rows[j], width)
    negative = builder.add(cache[j], rows[j + 1], width)
    sign = builder.ult(positive, negative)
    distance = builder.mux(sign, builder.sub(negative, positive, width),
                           builder.sub(positive, negative, width))
    return sign, distance


def append_young_generator(circuit, rows, cache, tag, j, *, angle_bits):
    """Canonical-pair implementation; j is a public ZERO-based array position."""
    from direct_induction_gate_angles import apply_ratio_rotation

    orientation, guard = circuit.allocate(2, "generator-flags")
    builder = BooleanBuilder(circuit)
    sign, distance = _axial_data(builder, rows, cache, j)
    selected = builder.band(builder.any_bits(tag), builder.ule(tag, builder.const(j + 1, len(tag))))
    nonsingleton = builder.band(selected, builder.ult(builder.const(1, len(distance)), distance))
    singleton_minus = builder.all_bits((selected, sign, builder.eq(distance, builder.const(1, len(distance)))))
    _phase_bit(circuit, singleton_minus)
    builder.copy_then_uncompute((builder.band(nonsingleton, sign), nonsingleton), (orientation, guard))
    _controlled_swap(circuit, orientation, rows[j], rows[j + 1])
    _controlled_swap(circuit, orientation, cache[j], cache[j + 1])

    builder = BooleanBuilder(circuit)
    _, distance = _axial_data(builder, rows, cache, j)
    squared = builder.mul(distance, distance, 2 * len(distance))
    right = builder.sub(squared, builder.const(1, len(squared)), len(squared))
    _phase_bit(circuit, orientation)  # Z first: A_D = Ry(2 arccos(1/D)) Z.
    apply_ratio_rotation(circuit, builder.const(1, len(squared)), right, orientation,
                         angle_fraction_bits=angle_bits, control=guard)
    builder.uncompute()
    _controlled_swap(circuit, orientation, rows[j], rows[j + 1])
    _controlled_swap(circuit, orientation, cache[j], cache[j + 1])
    builder = BooleanBuilder(circuit)
    sign, distance = _axial_data(builder, rows, cache, j)
    selected = builder.band(builder.any_bits(tag), builder.ule(tag, builder.const(j + 1, len(tag))))
    nonsingleton = builder.band(selected, builder.ult(builder.const(1, len(distance)), distance))
    builder.copy_then_uncompute((builder.band(nonsingleton, sign), nonsingleton), (orientation, guard))
    circuit.release((orientation, guard))


def append_uniform_tag(circuit, tag, m):
    """Specific public binary-tree completion, including the unused zero tag."""
    def node(start, size, prefix):
        if size == 1:
            return
        half = size // 2
        left = sum(1 <= i <= m for i in range(start, start + half))
        right = sum(1 <= i <= m for i in range(start + half, start + size))
        target = tag[(size - 1).bit_length() - 1]
        builder = BooleanBuilder(circuit)
        control = builder.all_bits(bit if value else builder.bnot(bit) for bit, value in prefix)
        if right:
            if left:
                circuit.ry(2 * atan2(sqrt(right), sqrt(left)), target, control)
            else:
                circuit.xor_into(control, target)
        builder.uncompute()
        node(start, half, prefix + ((target, 0),))
        node(start + half, half, prefix + ((target, 1),))
    node(0, 1 << len(tag), ())


@dataclass
class CompiledInduction:
    level: int
    fraction_bits: int
    angle_bits: int
    circuit: Circuit
    column_gates: tuple[Gate, ...]
    source_reflection: tuple[Gate, ...]
    target_reflection: tuple[Gate, ...]
    shape: tuple
    rows: tuple
    tag: tuple
    flag: Wire
    work: tuple
    rounds: int
    compile_seconds: float

    def input_index(self, shape, tag, word):
        assignments = [(self.tag, tag)]
        assignments += [(register, shape[i] if i < len(shape) else 0)
                        for i, register in enumerate(self.shape)]
        assignments += [(register, word[i] if i < len(word) else 0)
                        for i, register in enumerate(self.rows)]
        return basis_index(assignments)

    def reference_columns(self):
        sources, targets, blocks = [], [], []
        for shape in partitions(self.level - 1):
            inputs, outputs = induction_bases(shape)
            sources.extend(self.input_index(shape, i, p) for i, p in inputs)
            targets.extend(self.input_index(shape, 0, q) for _, q in outputs)
            blocks.append(induction_reference(shape))
        from scipy.linalg import block_diag
        return tuple(sources), tuple(targets), block_diag(*blocks)


def compile_induction(level=3, *, fraction_bits=7, angle_bits=7):
    from direct_induction_gate_bookkeeping import append_cache_xor, append_promise_phase

    if not isinstance(level, int) or level < 2:
        raise ValueError("the nontrivial prototype starts at level 2")
    if not 2 <= fraction_bits <= 24 or not 2 <= angle_bits <= 20:
        raise ValueError("prototype precision outside the supported audit range")
    started = perf_counter()
    circuit = Circuit()
    width = level.bit_length()
    shape = tuple(circuit.allocate(width, "shape") for _ in range(level))
    rows = tuple(circuit.allocate(width, "tableau") for _ in range(level))
    tag = circuit.allocate(width, "tag")
    work_start = circuit.qubit_count
    flag = circuit.allocate(1, "attenuation")[0]
    cache = tuple(circuit.allocate(width, "columns") for _ in range(level))
    append_branching(circuit, shape, rows[-1], fraction_bits=fraction_bits, angle_bits=angle_bits)
    append_cache_xor(circuit, rows, cache)
    for j in reversed(range(level - 1)):
        append_young_generator(circuit, rows, cache, tag, j, angle_bits=angle_bits)
    append_cache_xor(circuit, rows, cache)
    # Compile the chosen H once and append its exact gate inverse.
    start = len(circuit.gates)
    append_uniform_tag(circuit, tag, level)
    forward_h = tuple(circuit.gates[start:])
    del circuit.gates[start:]
    _append_clean_inverse(circuit, forward_h)
    rounds = ceil(sqrt(level))
    theta = pi / (4 * rounds + 2)
    attenuation = sqrt(level) * np.sin(theta)
    circuit.ry(2 * np.arccos(attenuation), flag)
    column = tuple(circuit.gates)
    column_work = tuple(Wire(i) for i in range(work_start, circuit.qubit_count))
    # Predicates must not borrow any of the work they are checking for zero.
    circuit.reserve(column_work)
    start = len(circuit.gates)
    append_promise_phase(circuit, shape, rows, tag, flag,
                         tuple(w for w in column_work if w != flag), kind="source")
    source = tuple(circuit.gates[start:])
    start = len(circuit.gates)
    append_promise_phase(circuit, shape, rows, tag, flag,
                         tuple(w for w in column_work if w != flag), kind="target")
    target = tuple(circuit.gates[start:])
    circuit.gates = list(column)
    for _ in range(rounds):
        circuit.append(target)
        circuit.append(_inverse(column))
        circuit.append(source)
        circuit.append(column)
        circuit.global_phase(pi)
    return CompiledInduction(level, fraction_bits, angle_bits, circuit, column, source,
        target, shape, rows, tag, flag, column_work, rounds, perf_counter() - started)


def audit_induction(compiled, *, cutoff=0.0, columns=None, simulator=None, progress=None):
    """Compare the entire emitted-gate output, including every work bit."""
    sources, targets, reference = compiled.reference_columns()
    selected = tuple(range(len(sources))) if columns is None else tuple(columns)
    outputs, pruning_bounds, supports = [], [], []
    simulator = compiled.circuit if simulator is None else simulator
    for j in selected:
        stats = {}
        output = simulator.simulate({sources[j]: 1.0 + 0j}, cutoff=cutoff, stats=stats)
        outputs.append(output)
        pruning_bounds.append(stats.get("discarded_norm_bound", 0.0))
        supports.append(stats.get("max_support", len(output)))
        if progress is not None:
            progress(j, len(selected), stats)
    ambient = set(targets)
    for output in outputs:
        ambient.update(output)
    positions = {value: index for index, value in enumerate(sorted(ambient))}
    actual = np.zeros((len(positions), len(selected)), complex)
    ideal = np.zeros_like(actual)
    for out_j, source_j in enumerate(selected):
        for index, amplitude in outputs[out_j].items():
            actual[positions[index], out_j] = amplitude
        for i, target_index in enumerate(targets):
            ideal[positions[target_index], out_j] = reference[i, source_j]
    clean_rows = {positions[index] for index in targets}
    dirty = [i for i in range(len(positions)) if i not in clean_rows]
    return {
        "level": compiled.level, "fraction_bits": compiled.fraction_bits,
        "angle_bits": compiled.angle_bits, "source_columns_checked": len(selected),
        "logical_qubits": compiled.circuit.qubit_count,
        "column_work_qubits": len(compiled.work),
        "column_gate_count": len(compiled.column_gates),
        "source_reflection_gate_count": len(compiled.source_reflection),
        "target_reflection_gate_count": len(compiled.target_reflection),
        "amplified_gate_count": len(compiled.circuit.gates),
        "gate_counts": compiled.circuit.gate_counts,
        "clean_operator_error": float(np.linalg.norm(actual - ideal, ord=2)),
        "outside_target_operator_norm": float(np.linalg.norm(actual[dirty], ord=2)) if dirty else 0.,
        "isometry_defect": float(np.linalg.norm(actual.conj().T @ actual - np.eye(len(selected)), ord=2)),
        "maximum_sparse_support": max(map(len, outputs)), "simulation_cutoff": cutoff,
        "maximum_intermediate_support": max(supports),
        "pruning_operator_error_bound": float(np.linalg.norm(pruning_bounds)),
        "compile_seconds": compiled.compile_seconds,
        "fully_finite_gate_lowered": False,
    }


def run_experiment(level=3, *, fraction_bits=9, angle_bits=9, finite=True,
                   rotation_tolerance=1e-5, cutoff=1e-13):
    """Reproducible, complete-column small-instance result (no timing metrics)."""
    import platform
    import qiskit
    from gate_block_simulator import BlockSimulator

    compiled = compile_induction(level, fraction_bits=fraction_bits, angle_bits=angle_bits)
    native = audit_induction(compiled, cutoff=cutoff, simulator=BlockSimulator(compiled.circuit))
    native.pop("compile_seconds")
    result = {
        "material_passport": {
            "origin_skill": "academic-research-suite / experiment-agent",
            "mode": "implementation-and-reproducibility",
            "date": "2026-10-23",
            "verification_status": "complete-source-column simulation",
            "version": "direct-induction-gates-v1",
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "qiskit": qiskit.__version__},
        "scope": (f"One complete level-{level} induction step across "
                  f"{len(partitions(level - 1))} shapes; not the whole S_n QFT."),
        "simulation": "Exact emitted Boolean blocks; finite rotation matrices multiplied from actual gate words.",
        "limitations": ["Small-instance measured error, not an asymptotic performance fit.",
                        "Floating-point roundoff is not included in the pruning bound.",
                        "Unoptimized retained-history arithmetic; qubits exclude error-correction overhead.",
                        "Public rotation dictionary is allowed; no shape-indexed coefficient or angle table."],
        "native": native,
    }
    if finite:
        from finite_gate_lowering import lower_circuit
        lowered = lower_circuit(compiled.circuit, tolerance=rotation_tolerance,
                                basis_depth=20, max_recursion_degree=5)
        simulator = BlockSimulator(lowered.source, qubit_count=lowered.qubit_count,
            rotation_resolver=lambda gate: lowered.rotation_matrices.get(gate.angle)
            if gate.op in ("ry", "cry") else None)
        checked = audit_induction(compiled, cutoff=cutoff, simulator=simulator)
        checked.pop("compile_seconds")
        checked.update({
            "fully_finite_gate_lowered": True,
            "logical_qubits": lowered.qubit_count,
            "finite_elementary_gate_count": lowered.elementary_gate_count,
            "finite_gate_counts": lowered.gate_counts,
            "t_and_tdg_count": lowered.gate_counts.get("t", 0) + lowered.gate_counts.get("tdg", 0),
            "synthesis_operator_error_bound": lowered.operator_error_bound,
            "requested_per_rotation_error": rotation_tolerance,
            "rotation_records": [
                {"angle": r.angle, "operator_error": r.operator_error,
                 "recursion_degree": r.recursion_degree, "finite_gates": len(r.controlled_word)}
                for angle, r in sorted(lowered.synthesizer.records.items()) if angle >= 0],
        })
        result["finite"] = checked
    return result


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=int, default=3)
    parser.add_argument("--fraction-bits", type=int, default=9)
    parser.add_argument("--angle-bits", type=int, default=9)
    parser.add_argument("--native-only", action="store_true",
                        help="skip the explicit Clifford+T lowering")
    parser.add_argument("--rotation-tolerance", type=float, default=1e-5)
    parser.add_argument("--cutoff", type=float, default=1e-13,
                        help="sparse simulation threshold; discarded norm is reported")
    args = parser.parse_args()
    print(json.dumps(run_experiment(args.level, fraction_bits=args.fraction_bits,
        angle_bits=args.angle_bits, finite=not args.native_only,
        rotation_tolerance=args.rotation_tolerance, cutoff=args.cutoff), indent=2, sort_keys=True))
