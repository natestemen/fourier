"""Compare the accelerated evaluator with the original elementary-gate path."""

import cmath
import math
from pathlib import Path
import random
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from gate_block_simulator import BlockSimulator  # noqa: E402
from reversible_gate_ir import Circuit, Gate  # noqa: E402


def _distance(a, b):
    return math.hypot(*(abs(a.get(index, 0j) - b.get(index, 0j)) for index in a.keys() | b.keys()))


def _random_circuit(seed, count=60):
    rng = random.Random(seed)
    circuit = Circuit()
    wires = circuit.allocate(5, "including_dirty_work")
    choices = ("x", "cx", "ccx", "z", "t", "tdg", "s", "sdg", "h", "ry", "cry", "phase")
    for _ in range(count):
        op = rng.choice(choices)
        if op in ("x", "z", "t", "tdg", "s", "sdg", "h"):
            getattr(circuit, op)(rng.choice(wires))
        elif op in ("cx", "ccx"):
            getattr(circuit, op)(*rng.sample(wires, 2 if op == "cx" else 3))
        elif op == "ry":
            circuit.ry(rng.uniform(-3, 3), rng.choice(wires))
        elif op == "cry":
            control, target = rng.sample(wires, 2)
            circuit.ry(rng.uniform(-3, 3), target, control)
        else:
            circuit.global_phase(rng.uniform(-3, 3))
    state = {i: complex(rng.uniform(-1, 1), rng.uniform(-1, 1)) for i in range(1 << len(wires))}
    norm = math.hypot(*(abs(value) for value in state.values()))
    state = {i: value / norm for i, value in state.items()}
    return circuit, state


@pytest.mark.parametrize("seed", range(12))
def test_all_true_gate_types_dirty_work_and_coherence_match_original(seed):
    circuit, state = _random_circuit(seed)
    original = tuple(circuit.gates)
    simulator = BlockSimulator(circuit)
    stats = {}
    output = simulator.simulate(state, stats=stats)
    reference = circuit.simulate(state)
    assert _distance(output, reference) < 3e-14
    assert tuple(circuit.gates) == original
    assert stats["gate_count"] == len(original)
    assert stats["discarded_norm_bound"] == 0
    assert stats["max_support"] == 32


@pytest.mark.parametrize("seed", range(6))
def test_inverse_preserves_entire_physical_state(seed):
    circuit, state = _random_circuit(49 + seed)
    forward = BlockSimulator(circuit).simulate(state)
    backward = BlockSimulator(circuit.inverse()).simulate(forward)
    assert _distance(backward, state) < 3e-14


def test_identical_blocks_are_interned_and_reused_across_simulations():
    circuit = Circuit()
    a, b, c = circuit.allocate(3)
    circuit.cx(a, b)
    circuit.ccx(a, b, c)
    circuit.t(c)
    circuit.z(b)
    pattern = tuple(circuit.gates)
    circuit.h(c)
    circuit.append(pattern)
    circuit.h(c)
    circuit.append(pattern)
    simulator = BlockSimulator(circuit)
    assert len(simulator.blocks) == 1
    assert len(simulator.steps) == 5
    state = {0: 0.5, 3: 0.5j, 5: -0.5j, 7: -0.5}
    first_stats, second_stats = {}, {}
    first = simulator.simulate(state, stats=first_stats)
    second = simulator.simulate(state, stats=second_stats)
    assert first == second
    assert second_stats["block_basis_evaluations"] == second_stats["decoded_boolean_steps"] == 0
    assert second_stats["block_cache_hits"] > 0
    assert second_stats["memoized_entries"] == first_stats["memoized_entries"]
    simulator.clear_cache()
    assert simulator.cache_entries == simulator.cache_estimated_bytes == 0
    third_stats = {}
    assert simulator.simulate(state, stats=third_stats) == first
    assert third_stats["block_basis_evaluations"] > 0


@pytest.mark.parametrize("budget", [0, 400, 1200])
def test_cache_memory_budget_does_not_change_gate_action(budget):
    circuit, state = _random_circuit(71, 80)
    simulator = BlockSimulator(circuit.gates, qubit_count=circuit.qubit_count, max_cache_bytes=budget)
    stats = {}
    output = simulator.simulate(state, stats=stats)
    assert _distance(output, circuit.simulate(state)) < 3e-14
    assert stats["cache_estimated_bytes"] <= budget
    if budget == 0:
        assert stats["memoized_entries"] == 0


def test_high_physical_wires_use_only_small_instruction_masks():
    circuit = Circuit()
    wires = circuit.allocate(53000)
    circuit.cx(wires[0], wires[52999])
    circuit.ccx(wires[40000], wires[52999], wires[31000])
    circuit.t(wires[52999])
    circuit.z(wires[31000])
    circuit.x(wires[40000])
    simulator = BlockSimulator(circuit)
    assert len(simulator.blocks) == 1
    for instruction in simulator.blocks[0].instructions:
        assert all(abs(value) <= simulator.byte_count for value in instruction)
    initial = 1 | (1 << 40000) | (1 << 52998)
    assert _distance(simulator.simulate({initial: 1j}), circuit.simulate({initial: 1j})) < 1e-14


@pytest.mark.parametrize("cutoff", [0.03, 0.1, 0.25])
def test_accumulated_discarded_l2_norm_bounds_actual_pruning_error(cutoff):
    circuit, state = _random_circuit(814, 100)
    simulator = BlockSimulator(circuit)
    exact = simulator.simulate(state)
    stats = {}
    approximate = simulator.simulate(state, cutoff=cutoff, stats=stats)
    assert _distance(exact, approximate) <= stats["discarded_norm_bound"] + 3e-14
    assert stats["discarded_norm_bound"] >= 0
    assert stats["pruned_states"] >= stats["prune_events"]
    assert math.hypot(*(abs(value) for value in approximate.values())) <= 1 + 3e-14


def test_tiny_amplitudes_have_a_nonunderflowing_pruning_certificate():
    circuit = Circuit()
    bit = circuit.allocate(1)[0]
    circuit.h(bit)
    stats = {}
    output = BlockSimulator(circuit).simulate({0: 1e-300}, cutoff=1e-250, stats=stats)
    assert output == {}
    assert 0.999e-300 < stats["discarded_norm_bound"] < 1.001e-300


def _apply_matrix(state, target, matrix, control=None):
    result = {}
    mask = 1 << target
    for index, amplitude in state.items():
        if control is not None and not ((index >> control) & 1):
            result[index] = result.get(index, 0j) + amplitude
            continue
        column = (index >> target) & 1
        for row in range(2):
            output = (index & ~mask) | (row << target)
            result[output] = result.get(output, 0j) + matrix[row][column] * amplitude
    return result


def test_matrix_replacement_keeps_conditional_global_phases_and_compilation_counts():
    circuit = Circuit()
    control, target = circuit.allocate(2)
    circuit.ry(0.4, target)
    circuit.ry(0.7, target, control)
    circuit.ry(0.4, target)
    phase = cmath.exp(0.37j)
    matrix = ((phase / math.sqrt(2), phase / math.sqrt(2)),
              (phase / math.sqrt(2), -phase / math.sqrt(2)))
    calls = []

    def resolver(gate):
        calls.append(gate)
        return matrix

    simulator = BlockSimulator(circuit, rotation_resolver=resolver)
    assert len(calls) == 2  # The identical Ry is resolved only once.
    state = {0: 0.5, 1: 0.5j, 2: -0.5j, 3: -0.5}
    reference = _apply_matrix(state, target.index, matrix)
    reference = _apply_matrix(reference, target.index, matrix, control.index)
    reference = _apply_matrix(reference, target.index, matrix)
    stats = {}
    assert _distance(simulator.simulate(state, stats=stats), reference) < 1e-14
    assert stats["rotation_replacements"] == stats["gate_count"] == 3
    assert len(circuit.gates) == 3


def test_resolver_none_uses_native_and_empty_circuits_preserve_input():
    circuit, state = _random_circuit(57, 5)
    assert _distance(BlockSimulator(circuit, rotation_resolver=lambda gate: None).simulate(state),
                     circuit.simulate(state)) < 1e-14
    assert BlockSimulator([], qubit_count=5).simulate(state) == state
    assert BlockSimulator([]).simulate() == {0: 1 + 0j}
    assert BlockSimulator([Gate("phase", (), math.pi)]).simulate() == {0: cmath.exp(1j * math.pi)}


def test_large_finite_phase_angles_do_not_overflow_when_blocked():
    circuit = Circuit()
    circuit.global_phase(1e308)
    circuit.global_phase(1e308)
    assert BlockSimulator(circuit).simulate() == circuit.simulate()


def test_statistics_accumulate_but_cache_is_shared_with_fresh_stats():
    circuit, state = _random_circuit(31, 5)
    simulator = BlockSimulator(circuit)
    stats = {}
    simulator.simulate(state, stats=stats)
    simulator.simulate(state, stats=stats)
    assert stats["gate_count"] == 2 * len(circuit.gates)
    assert stats["block_cache_hits"] > 0


@pytest.mark.parametrize("gates", [
    [Gate("opaque_predicate", (0,))], [Gate("x", ())], [Gate("ccx", (0, 0, 1))],
    [Gate("x", (-1,))], [Gate("ry", (0,), None)], [Gate("phase", (), math.inf)],
])
def test_invalid_or_opaque_gate_traces_are_rejected(gates):
    with pytest.raises(ValueError):
        BlockSimulator(gates)


def test_bad_precision_width_state_and_nonunitary_substitutions_are_rejected():
    gate = Gate("ry", (2,), 0.3)
    with pytest.raises(ValueError):
        BlockSimulator([gate], qubit_count=2)
    with pytest.raises(ValueError):
        BlockSimulator([gate], rotation_resolver=lambda gate: ((1, 0), (0, 2)))
    with pytest.raises(ValueError):
        BlockSimulator([gate], max_cache_bytes=-1)
    simulator = BlockSimulator([gate])
    for invalid in (-1, 8):
        with pytest.raises(ValueError):
            simulator.simulate({invalid: 1})
    for cutoff in (-1, math.nan, math.inf):
        with pytest.raises(ValueError):
            simulator.simulate(cutoff=cutoff)
