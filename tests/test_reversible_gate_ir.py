"""Actual gate-stream tests; no arithmetic oracle in the simulator."""

from itertools import product
from math import pi
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from reversible_gate_ir import (  # noqa: E402
    BooleanBuilder, Circuit, Gate, Wire, apply_gates, basis_index, inverse_gates, word_value,
)


def _assert_clean(index, kept):
    mask = sum(1 << wire.index for wire in kept)
    assert index & ~mask == 0


@pytest.mark.parametrize("width", [1, 2, 3, 4])
@pytest.mark.parametrize("operation", ["add", "sub", "mul", "divmod", "eq", "ult", "ule", "mux"])
def test_unsigned_arithmetic_exhaustive_basis_action_and_clean_work(width, operation):
    circuit = Circuit()
    left = circuit.allocate(width, "a")
    right = circuit.allocate(width, "b")
    select = circuit.allocate(1, "select")
    output_width = 2 * width if operation in {"mul", "divmod"} else (
        1 if operation in {"eq", "ult", "ule"} else width)
    output = circuit.allocate(output_width, "out")
    builder = BooleanBuilder(circuit)
    if operation == "divmod":
        quotient, remainder = builder.divmod(left, right)
        bits = quotient + remainder
    elif operation == "mux":
        bits = builder.mux(select[0], left, right)
    else:
        value = getattr(builder, operation)(left, right)
        bits = value if isinstance(value, tuple) else (value,)
    builder.copy_then_uncompute(bits, output)
    assert set(circuit.gate_counts) <= {"x", "cx", "ccx"}
    assert sum(circuit.gate_counts.values()) == len(circuit.gates)
    for a, b in product(range(1 << width), repeat=2):
        for selected in ([0, 1] if operation == "mux" else [0]):
            initial = basis_index(((left, a), (right, b), (select, selected)))
            index, phase = circuit.simulate_basis(initial)
            mask = (1 << width) - 1
            expected = {"add": (a + b) & mask, "sub": (a - b) & mask,
                        "mul": a * b, "divmod": a // (b or 1) + ((a % (b or 1)) << width),
                        "eq": int(a == b), "ult": int(a < b), "ule": int(a <= b),
                        "mux": a if selected else b}[operation]
            assert word_value(index, output) == expected
            assert word_value(index, left) == a and word_value(index, right) == b
            assert phase == 1
            _assert_clean(index, left + right + select + output)


@pytest.mark.parametrize("a_width,b_width,out_width", [(2, 4, 3), (4, 2, 2), (1, 3, 5)])
def test_unequal_width_zero_extension_truncation_and_safe_division(a_width, b_width, out_width):
    circuit = Circuit()
    a = circuit.allocate(a_width)
    b = circuit.allocate(b_width)
    output = circuit.allocate(4 * out_width + 2)
    builder = BooleanBuilder(circuit)
    quotient, remainder = builder.divmod(a, b, width=out_width)
    results = (builder.add(a, b, width=out_width) + builder.mul(a, b, width=out_width)
               + quotient + remainder + (builder.eq(a, b), builder.ult(a, b)))
    builder.copy_then_uncompute(results, output)
    for first, second in product(range(1 << a_width), range(1 << b_width)):
        index, _ = circuit.simulate_basis(basis_index(((a, first), (b, second))))
        mask = (1 << out_width) - 1
        expected_words = ((first + second) & mask, (first * second) & mask,
                          (first // (second or 1)) & mask, (first % (second or 1)) & mask)
        for position, expected in enumerate(expected_words):
            assert word_value(index, output[position * out_width:(position + 1) * out_width]) == expected
        assert word_value(index, output[-2:]) == int(first == second) + 2 * int(first < second)
        _assert_clean(index, a + b + output)


def test_compute_copy_uncompute_is_xor_on_nonzero_output_and_its_own_inverse():
    circuit = Circuit()
    a, b, out = circuit.allocate(3)
    builder = BooleanBuilder(circuit)
    builder.copy_then_uncompute((builder.band(a, b),), (out,))
    for bits in product(range(2), repeat=3):
        initial = basis_index((((a,), bits[0]), ((b,), bits[1]), ((out,), bits[2])))
        final, _ = circuit.simulate_basis(initial)
        assert word_value(final, (out,)) == bits[2] ^ (bits[0] & bits[1])
        assert circuit.inverse().simulate_basis(final) == (initial, 1)
    doubled = Circuit()
    doubled.append(circuit)
    doubled.append(circuit)
    assert doubled.simulate_basis(7) == (7, 1)


def test_external_phase_is_not_accidentally_reversed_with_boolean_computation():
    circuit = Circuit()
    a, b = circuit.allocate(2)
    circuit.h(a)
    circuit.h(b)
    builder = BooleanBuilder(circuit)
    parity = builder.bxor(a, b)
    circuit.z(parity)
    builder.uncompute()
    expected = {0: .5, 1: -.5, 2: -.5, 3: .5}
    actual = circuit.simulate()
    assert set(actual) == set(expected)
    for index, amplitude in expected.items():
        assert actual[index] == pytest.approx(amplitude)
    recovered = circuit.inverse().simulate(actual, cutoff=1e-14)
    assert recovered == pytest.approx({0: 1})


def test_superposition_reference_entanglement_preserves_relative_and_global_phases():
    circuit = Circuit()
    reference, control, target = circuit.allocate(3)
    circuit.cx(reference, control)
    circuit.ry(.731, target, control=control)
    circuit.z(control)
    circuit.global_phase(.219)
    initial = {0: 1 / np.sqrt(2), 1: 1j / np.sqrt(2)}
    actual = circuit.simulate(initial)
    phase = np.exp(.219j)
    expected = {0: phase / np.sqrt(2),
                3: -1j * phase * np.cos(.731 / 2) / np.sqrt(2),
                7: -1j * phase * np.sin(.731 / 2) / np.sqrt(2)}
    assert actual == pytest.approx(expected)
    assert sum(abs(amplitude) ** 2 for amplitude in actual.values()) == pytest.approx(1)
    assert circuit.inverse().simulate(actual, cutoff=1e-14) == pytest.approx(initial)
    assert apply_gates(inverse_gates(circuit.gates), actual, cutoff=1e-14) == pytest.approx(initial)


def test_finite_gate_primitives_and_append_slice_inverse():
    circuit = Circuit()
    a, b, target = circuit.allocate(3)
    for wire in (a, b, target):
        circuit.h(wire)
    circuit.t(a)
    circuit.s(b)
    circuit.ccx(a, b, target)
    circuit.tdg(target)
    circuit.sdg(a)
    circuit.swap(a, b)
    state = circuit.simulate()
    assert sum(abs(amplitude) ** 2 for amplitude in state.values()) == pytest.approx(1)
    back = apply_gates(inverse_gates(circuit.gates), state, cutoff=1e-14)
    assert back == pytest.approx({0: 1})
    assert Gate("cry", (0, 1), .3).inverse() == Gate("cry", (0, 1), -.3)


def test_allocator_reuses_only_released_temporaries_and_can_reserve_work():
    circuit = Circuit()
    a, b = circuit.allocate(2, "persistent")
    first = BooleanBuilder(circuit)
    temp = first.band(a, b)
    first.uncompute()
    peak = circuit.qubit_count
    second = BooleanBuilder(circuit)
    same = second.band(a, b)
    assert same == temp and circuit.qubit_count == peak
    second.uncompute()
    previously_free = circuit.reserve((a, b, temp))
    assert previously_free == (temp,)
    helper = circuit.allocate(1, "helper")[0]
    assert helper.index == peak
    circuit.release((helper,))
    circuit.release(previously_free)
    assert circuit.allocate(1)[0] == temp
    assert circuit.allocate(1, fresh=True)[0].index == peak + 1


def test_append_conservatively_reserves_touched_free_wires():
    circuit = Circuit()
    wire = circuit.allocate(1)[0]
    circuit.release((wire,))
    circuit.append((Gate("x", (wire.index,)),))
    assert circuit.allocate(1)[0] != wire


def test_boolean_constants_have_no_hidden_wires_or_gates():
    circuit = Circuit()
    builder = BooleanBuilder(circuit)
    assert builder.add(builder.const(7, 3), builder.const(2, 3)) == (1, 0, 0)
    quotient, remainder = builder.divmod(builder.const(7, 3), builder.const(0, 3))
    assert quotient == (1, 1, 1) and remainder == (0, 0, 0)
    assert builder.all_bits(()) == 1 and builder.any_bits(()) == 0
    builder.uncompute()
    assert circuit.qubit_count == 0 and not circuit.gates


def test_invalid_calls_raise_instead_of_creating_opaque_or_lossy_operations():
    circuit = Circuit()
    a = circuit.allocate(1)[0]
    with pytest.raises(ValueError):
        circuit.cx(a, a)
    with pytest.raises(ValueError):
        circuit.xor_into(2, a)
    with pytest.raises(ValueError):
        circuit.simulate({2: 1})
    circuit.h(a)
    with pytest.raises(ValueError, match="superposition"):
        circuit.simulate_basis()
    builder = BooleanBuilder(circuit)
    temp = builder.bnot(a)
    with pytest.raises(ValueError, match="temporaries"):
        builder.copy_then_uncompute((a,), (temp,))
    builder.uncompute()
    with pytest.raises(RuntimeError):
        builder.uncompute()


def test_disabled_controls_and_zero_angle_emit_no_gate():
    circuit = Circuit()
    target = circuit.allocate(1)[0]
    circuit.ry(.5, target, control=0)
    circuit.ry(0, target)
    assert not circuit.gates
    circuit.ry(.5, target, control=1)
    assert circuit.gate_counts == {"ry": 1}


def test_reported_pruning_norm_bound_covers_the_complete_simulation_error():
    circuit = Circuit()
    first, second = circuit.allocate(2)
    circuit.ry(1e-5, first)
    circuit.ry(2e-5, second)
    circuit.cx(first, second)
    circuit.ry(-3e-5, first)
    exact = circuit.simulate()
    stats = {}
    pruned = circuit.simulate(cutoff=2e-5, stats=stats)
    difference = np.sqrt(sum(abs(exact.get(key, 0) - pruned.get(key, 0)) ** 2
                             for key in set(exact) | set(pruned)))
    assert 0 < stats["discarded_norm_bound"]
    assert difference <= stats["discarded_norm_bound"] + 1e-15
    assert stats["max_support"] >= len(pruned)
    assert stats["gate_count"] == len(circuit.gates)
    other_stats = {}
    assert apply_gates(circuit.gates, cutoff=2e-5, stats=other_stats) == pruned
    assert other_stats == stats
