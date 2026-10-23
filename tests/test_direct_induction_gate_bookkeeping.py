"""Exhaustive bit-plane evaluation of the emitted elementary Boolean gates.

The test interpreter applies each actual X/CX/CCX/Z gate to many classical
basis inputs in parallel using Python integer bit planes. It never calls the
compiler's predicate or simulates an opaque operation. Reference predicates
are deliberately separate and may use quadratic scans.
"""

from math import pi
from pathlib import Path
import random
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from direct_induction_gate_bookkeeping import (  # noqa: E402
    append_cache_xor, append_cover_row_xor, append_endpoint_xor, append_promise_phase,
)
from reversible_gate_ir import Circuit, basis_index  # noqa: E402


def _flatten(words):
    return tuple(bit for word in words for bit in word)


def _planes(values, width):
    result = [0] * width
    for position, value in enumerate(values):
        mask = 1 << position
        for bit in range(width):
            if (value >> bit) & 1:
                result[bit] |= mask
    return result


def _apply_gate_planes(circuit, input_wires, cases):
    """Exact many-basis-input interpreter, including relative phase signs."""
    mask = (1 << len(cases)) - 1
    values = [0] * circuit.qubit_count
    for wire, plane in zip(input_wires, _planes(cases, len(input_wires))):
        values[wire.index] = plane
    negative = 0
    for gate in circuit.gates:
        wires = gate.wires
        if gate.op == "x":
            values[wires[0]] ^= mask
        elif gate.op == "cx":
            values[wires[1]] ^= values[wires[0]]
        elif gate.op == "ccx":
            values[wires[2]] ^= values[wires[0]] & values[wires[1]]
        elif gate.op == "z":
            negative ^= values[wires[0]]
        elif gate.op == "phase":
            turns = round(gate.angle / pi)
            assert abs(gate.angle - turns * pi) < 1e-12
            if turns & 1:
                negative ^= mask
        else:
            raise AssertionError(f"not an elementary Boolean/phase gate: {gate.op}")
    return values, negative


def _check_images(circuit, wires, cases, expected_cases, expected_negative):
    values, negative = _apply_gate_planes(circuit, wires, cases)
    expected = [0] * circuit.qubit_count
    for wire, plane in zip(wires, _planes(expected_cases, len(wires))):
        expected[wire.index] = plane
    assert values == expected, "input corruption, incorrect output, or uncleared helper"
    assert negative == sum(int(sign) << i for i, sign in enumerate(expected_negative))


def _word_values(encoded, count, width):
    mask = (1 << width) - 1
    return tuple((encoded >> (i * width)) & mask for i in range(count))


def _reference_columns(rows, active):
    return tuple(sum(rows[j] == rows[i] for j in range(i + 1)) if i < active else 0
                 for i in range(len(rows)))


def _reference_tableau(rows, active):
    m = len(rows)
    endpoint = tuple(sum(row == r for row in rows[:active]) for r in range(1, m + 1))
    valid = all(1 <= row <= m for row in rows[:active])
    for end in range(1, active + 1):
        prefix = tuple(sum(row == r for row in rows[:end]) for r in range(1, m + 1))
        valid &= all(prefix[i] >= prefix[i + 1] for i in range(m - 1))
    return endpoint, valid


def _reference_promise(shape, rows, tag, flag, work, kind):
    m = len(rows)
    endpoint, valid = _reference_tableau(rows, m - 1 if kind == "source" else m)
    clean = flag == 0 and not any(work)
    if kind == "source":
        return valid and endpoint == shape and rows[-1] == 0 and 1 <= tag <= m and clean
    delta = tuple(b - a for a, b in zip(shape, endpoint))
    cover = (sum(shape) == m - 1 and all(shape[i] >= shape[i + 1] for i in range(m - 1))
             and all(0 <= x <= m for x in shape) and delta.count(1) == 1
             and all(d in (0, 1) for d in delta))
    return valid and cover and tag == 0 and clean


def _reference_cover(old, new):
    m = len(old)
    delta = tuple(b - a for a, b in zip(old, new))
    valid = (sum(old) == m - 1 and sum(new) == m
             and all(old[i] >= old[i + 1] and new[i] >= new[i + 1] for i in range(m - 1))
             and all(0 <= v <= m for v in old + new)
             and all(d in (0, 1) for d in delta) and delta.count(1) == 1)
    return (1 + delta.index(1), True) if valid else (0, False)


def _cache_circuit(m, width, active):
    circuit = Circuit()
    rows = tuple(circuit.allocate(width, "row") for _ in range(m))
    cache = tuple(circuit.allocate(width, "cache") for _ in range(m))
    append_cache_xor(circuit, rows, cache, active_length=active)
    return circuit, rows, cache


def _promise_circuit(m, width, kind, work_count=2):
    circuit = Circuit()
    shape = tuple(circuit.allocate(width, "shape") for _ in range(m))
    rows = tuple(circuit.allocate(width, "row") for _ in range(m))
    tag = circuit.allocate(width, "tag")
    flag = circuit.allocate(1, "flag")[0]
    work = circuit.allocate(work_count, "work")
    append_promise_phase(circuit, shape, rows, tag, flag, work, kind)
    return circuit, shape, rows, tag, flag, work


@pytest.mark.parametrize("m", [1, 2, 3])
def test_cache_all_row_encodings_active_lengths_and_dirty_xor_outputs(m):
    width = max(1, m.bit_length())
    for active in range(m + 1):
        circuit, rows, cache = _cache_circuit(m, width, active)
        assert set(circuit.gate_counts) <= {"x", "cx", "ccx"}
        wires = _flatten(rows + cache)
        for start in range(0, 1 << len(wires), 4096):
            cases = list(range(start, min(start + 4096, 1 << len(wires))))
            expected = []
            for case in cases:
                row_values = _word_values(case, m, width)
                columns = _reference_columns(row_values, active)
                correction = sum(column << (i * width + m * width)
                                 for i, column in enumerate(columns))
                expected.append(case ^ correction)
            _check_images(circuit, wires, cases, expected, [False] * len(cases))


@pytest.mark.parametrize("m", [1, 2, 3])
def test_endpoint_all_row_encodings_active_lengths_and_dirty_outputs(m):
    width = max(1, m.bit_length())
    for active in range(m + 1):
        circuit = Circuit()
        rows = tuple(circuit.allocate(width, "row") for _ in range(m))
        endpoint = tuple(circuit.allocate(width, "endpoint") for _ in range(m))
        append_endpoint_xor(circuit, rows, endpoint, active_length=active)
        wires = _flatten(rows + endpoint)
        for start in range(0, 1 << len(wires), 4096):
            cases = list(range(start, min(start + 4096, 1 << len(wires))))
            expected = []
            for case in cases:
                histogram, _ = _reference_tableau(_word_values(case, m, width), active)
                expected.append(case ^ sum(v << (i * width + m * width)
                                            for i, v in enumerate(histogram)))
            _check_images(circuit, wires, cases, expected, [False] * len(cases))


@pytest.mark.parametrize("m", [1, 2, 3])
def test_cover_row_and_validity_xor_every_shape_encoding_and_dirty_output(m):
    width = max(1, m.bit_length())
    circuit = Circuit()
    old = tuple(circuit.allocate(width, "old") for _ in range(m))
    new = tuple(circuit.allocate(width, "new") for _ in range(m))
    row = circuit.allocate(width, "added_row")
    valid = circuit.allocate(1, "valid")[0]
    append_cover_row_xor(circuit, old, new, row, valid)
    wires = _flatten(old + new) + row + (valid,)
    for start in range(0, 1 << len(wires), 4096):
        cases = list(range(start, min(start + 4096, 1 << len(wires))))
        expected = []
        for case in cases:
            added_row, is_cover = _reference_cover(_word_values(case, m, width),
                                                   _word_values(case >> (m * width), m, width))
            expected.append(case ^ (added_row << (2 * m * width))
                             ^ (int(is_cover) << ((2 * m + 1) * width)))
        _check_images(circuit, wires, cases, expected, [False] * len(cases))


def test_cover_row_without_valid_output_and_with_wider_row_target():
    circuit = Circuit()
    old = tuple(circuit.allocate(2, "old") for _ in range(3))
    new = tuple(circuit.allocate(2, "new") for _ in range(3))
    row = circuit.allocate(4, "added_row")
    append_cover_row_xor(circuit, old, new, row)
    index = basis_index(list(zip(old, (2, 0, 0))) + list(zip(new, (2, 1, 0))) + [(row, 12)])
    assert circuit.simulate_basis(index) == (index ^ (2 << row[0].index), 1 + 0j)


@pytest.mark.parametrize("m", [1, 2, 3])
@pytest.mark.parametrize("kind", ["source", "target"])
def test_complete_promise_phase_every_small_encoding_and_dirty_work(m, kind):
    width = max(1, m.bit_length())
    circuit, shape, rows, tag, flag, work = _promise_circuit(m, width, kind)
    wires = _flatten(shape + rows) + tag + (flag,) + work
    assert set(circuit.gate_counts) <= {"x", "cx", "ccx", "z", "phase"}
    assert circuit.gate_counts.get("z", 0) + circuit.gate_counts.get("phase", 0) == 1
    for start in range(0, 1 << len(wires), 4096):
        cases = list(range(start, min(start + 4096, 1 << len(wires))))
        signs = []
        for case in cases:
            shape_values = _word_values(case, m, width)
            row_values = _word_values(case >> (m * width), m, width)
            tag_value = (case >> (2 * m * width)) & ((1 << width) - 1)
            flag_value = (case >> ((2 * m + 1) * width)) & 1
            work_values = _word_values(case >> ((2 * m + 1) * width + 1), len(work), 1)
            signs.append(not _reference_promise(shape_values, row_values, tag_value,
                                                flag_value, work_values, kind))
        _check_images(circuit, wires, cases, cases, signs)


@pytest.mark.parametrize("m", [4, 5, 8])
def test_non_power_of_two_padding_and_wide_malformed_cache_words(m):
    width = m.bit_length() + 1
    circuit, rows, cache = _cache_circuit(m, width, m)
    rng = random.Random(811 + m)
    wires = _flatten(rows + cache)
    cases = [rng.randrange(1 << len(wires)) for _ in range(32)]
    expected = []
    for case in cases:
        columns = _reference_columns(_word_values(case, m, width), m)
        expected.append(case ^ sum(column << (i * width + m * width)
                                   for i, column in enumerate(columns)))
    _check_images(circuit, wires, cases, expected, [False] * len(cases))


@pytest.mark.parametrize("kind", ["source", "target"])
def test_superposition_preserves_amplitudes_and_applies_the_coherent_sign(kind):
    circuit, shape, rows, tag, flag, work = _promise_circuit(2, 2, kind, work_count=1)
    # Encode one accepted input and one dirty-work input of the same coherent shape.
    shape_value = 1
    rows_value, tag_value = (1, 1) if kind == "source" else (1 | (2 << 2), 0)
    accepted = shape_value | (rows_value << 4) | (tag_value << 8)
    rejected = accepted | (1 << work[0].index)
    state = {accepted: 0.6 + 0j, rejected: 0.8j}
    output = circuit.simulate(state)
    assert output == {accepted: 0.6 + 0j, rejected: -0.8j}


@pytest.mark.parametrize("kind", ["source", "target"])
def test_distinct_valid_shapes_are_coherent_inputs_to_one_compiled_circuit(kind):
    circuit, shape, rows, tag, flag, work = _promise_circuit(3, 2, kind, work_count=1)
    shapes = ((2, 0, 0), (1, 1, 0))
    row_values = (((1, 1, 0), (1, 2, 0)) if kind == "source"
                  else ((1, 2, 1), (1, 2, 3)))
    indices = []
    for i in range(2):
        assignments = list(zip(shape, shapes[i])) + list(zip(rows, row_values[i]))
        assignments += [(tag, 1 + 2 * i if kind == "source" else 0), ((flag,), 0), (work, 0)]
        indices.append(basis_index(assignments))
    dirty = indices[1] | (1 << work[0].index)
    initial = {indices[0]: 0.3 + 0.4j, indices[1]: -0.5j, dirty: 0.5 + 0.5j}
    expected = dict(initial)
    expected[dirty] *= -1
    assert circuit.simulate(initial) == expected


def test_repeated_cache_xor_reuses_clean_history_and_is_its_own_inverse():
    circuit, rows, cache = _cache_circuit(2, 2, 2)
    first_width = circuit.qubit_count
    append_cache_xor(circuit, rows, cache)
    assert circuit.qubit_count == first_width
    wires = _flatten(rows + cache)
    cases = list(range(1 << len(wires)))
    _check_images(circuit, wires, cases, cases, [False] * len(cases))


@pytest.mark.parametrize("kind", ["source", "target"])
def test_freed_work_wires_are_not_borrowed_as_reflection_helpers(kind):
    circuit = Circuit()
    shape = (circuit.allocate(1, "shape"),)
    rows = (circuit.allocate(1, "row"),)
    tag = circuit.allocate(1, "tag")
    flag = circuit.allocate(1, "flag")[0]
    work = circuit.allocate(3, "recycled_work")
    circuit.release(work)  # Main column compiler can mark its clean scratch reusable.
    append_promise_phase(circuit, shape, rows, tag, flag, work, kind)
    wires = _flatten(shape + rows) + tag + (flag,) + work
    cases = list(range(1 << len(wires)))
    signs = [not _reference_promise((case & 1,), ((case >> 1) & 1,),
                                    (case >> 2) & 1, (case >> 3) & 1,
                                    _word_values(case >> 4, 3, 1), kind)
             for case in cases]
    _check_images(circuit, wires, cases, cases, signs)
    first_width = circuit.qubit_count
    append_promise_phase(circuit, shape, rows, tag, flag, work, kind)
    assert circuit.qubit_count == first_width
    _check_images(circuit, wires, cases, cases, [False] * len(cases))


@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 8])
def test_compiled_gate_count_has_a_conservative_fixed_network_bound(m):
    width = max(1, m.bit_length())
    circuit, _, _ = _cache_circuit(m, width, m)
    levels = (2 * m - 1).bit_length()
    assert len(circuit.gates) <= 1000 * m * (levels + 1)**2 * width
    assert circuit.qubit_count <= 1000 * m * (levels + 1)**2 * width


def test_hardware_interface_validation_and_no_opaque_gates():
    circuit = Circuit()
    rows = (circuit.allocate(1, "row"),)
    target = (circuit.allocate(1, "target"),)
    with pytest.raises(ValueError):
        append_cache_xor(circuit, rows, rows)
    with pytest.raises(ValueError):
        append_cache_xor(circuit, rows, target, active_length=2)
    with pytest.raises(ValueError):
        append_cache_xor(circuit, (), ())
    with pytest.raises(ValueError):
        append_promise_phase(circuit, rows, rows, target[0], 0, (), kind="neither")
