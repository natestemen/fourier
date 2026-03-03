#!/usr/bin/env python3
"""Plot required R_aa + U3 layers vs synthesis precision epsilon.

This script uses the circuit template from the end of play.ipynb as the target
gate set (U3 + RXX/RYY/RZZ) and measures how many entangling layers are needed
to approximate each Fourier matrix to a target epsilon. For each epsilon in a
log-spaced range, it instantiates 100 Young-diagram Fourier matrices with 8
addable cells (3 qubits) and plots the average number of layers.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from bqskit.ir import Circuit
from bqskit.ir.gates import RXXGate, RYYGate, RZZGate, U3Gate

from compute_matrix import A_matrix
from helper import find_yds_with_fixed_addable_cells


def _build_template(num_qubits: int, entangling_layers: int) -> Circuit:
    circ = Circuit(num_qubits)
    pairs = [(0, 1), (0, 2), (1, 2)]

    for _ in range(entangling_layers):
        for q in range(num_qubits):
            circ.append_gate(U3Gate(), [q])
        for i, j in pairs:
            circ.append_gate(RXXGate(), [i, j])
            circ.append_gate(RYYGate(), [i, j])
            circ.append_gate(RZZGate(), [i, j])

    for q in range(num_qubits):
        circ.append_gate(U3Gate(), [q])
    return circ


def _instantiate(template: Circuit, target: np.ndarray) -> Circuit:
    try:
        return template.instantiate(target)
    except TypeError:
        return template.instantiate(target=target)


def _unitary_distance(U: np.ndarray, V: np.ndarray) -> float:
    vdot = np.vdot(V, U)
    if vdot == 0:
        phase = 1.0
    else:
        phase = vdot / abs(vdot)
    diff = U - phase * V
    return float(np.linalg.norm(diff) / np.linalg.norm(V))


def _free_parameter_count(circuit: Circuit) -> int:
    total = 0
    for op in circuit:
        num_params = getattr(op, "num_params", None)
        if num_params is None:
            num_params = getattr(op.gate, "num_params", 0)
        if callable(num_params):
            num_params = num_params()
        total += int(num_params)
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-diagram-size",
        type=int,
        required=True,
        help="Upper bound for Young diagram size enumeration.",
    )
    parser.add_argument(
        "--num-diagrams",
        type=int,
        default=100,
        help="Number of diagrams to sample per epsilon (default: 100).",
    )
    parser.add_argument(
        "--eps-start",
        type=float,
        default=1e-1,
        help="Largest epsilon in the sweep (default: 1e-1).",
    )
    parser.add_argument(
        "--eps-end",
        type=float,
        default=1e-6,
        help="Smallest epsilon in the sweep (default: 1e-6).",
    )
    parser.add_argument(
        "--eps-steps",
        type=int,
        default=6,
        help="Number of log-spaced epsilon points (default: 6).",
    )
    parser.add_argument(
        "--max-layers",
        type=int,
        default=6,
        help="Maximum entangling layers to try per epsilon (default: 6).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/plots/raau3_layers_vs_epsilon.png"),
        help="Path to save the plot image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    num_qubits = 3
    addable_cells = 1 << num_qubits

    diagrams = list(find_yds_with_fixed_addable_cells(addable_cells, args.max_diagram_size))
    if not diagrams:
        raise SystemExit("No diagrams found for the requested size bounds.")
    if len(diagrams) < args.num_diagrams:
        print(
            f"Warning: only {len(diagrams)} diagrams available; using all of them."
        )
    diagrams = diagrams[: args.num_diagrams]

    epsilons = np.logspace(math.log10(args.eps_start), math.log10(args.eps_end), args.eps_steps)

    avg_layers: list[float] = []
    max_layers: list[int] = []

    for eps in epsilons:
        per_diagram_layers: list[int] = []
        for diagram in diagrams:
            target = A_matrix(diagram)
            found_layers = None
            for layer_count in range(1, args.max_layers + 1):
                template = _build_template(num_qubits, layer_count)
                free_params = _free_parameter_count(template)
                if diagram is diagrams[0]:
                    unitary_dim = 1 << num_qubits
                    u8_params = unitary_dim * unitary_dim
                    print(
                        f"layers={layer_count} free_params={free_params} "
                        f"U(8) params={u8_params}"
                    )
                instantiated = _instantiate(template, target)
                unitary = np.asarray(instantiated.get_unitary(), dtype=np.complex128)
                distance = _unitary_distance(unitary, target)
                if distance <= eps:
                    found_layers = layer_count
                    break
            if found_layers is None:
                found_layers = args.max_layers + 1
            per_diagram_layers.append(found_layers)
        avg_layers.append(float(np.mean(per_diagram_layers)))
        max_layers.append(int(max(per_diagram_layers)))
        print(
            f"epsilon={eps:.1e} avg_layers={avg_layers[-1]:.2f} max_layers={max_layers[-1]}"
        )

    fig, ax = plt.subplots()
    ax.plot(epsilons, avg_layers, marker="o", label="Average layers")
    ax.plot(epsilons, max_layers, marker="x", label="Max layers")
    ax.set_xscale("log")
    ax.set_xlabel("epsilon")
    ax.set_ylabel("R_aa + U3 layers")
    ax.set_title("Layers needed vs synthesis precision")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
