#!/usr/bin/env python3
"""Entangling layers needed for RXX/RYY/RZZ + U3 synthesis vs precision ε.

Question: how many layers of the play.ipynb ansatz — a U3 column followed by
RXX, RYY, RZZ on every qubit pair — does BQSKit instantiation need to hit a
target precision ε on 3-qubit (8-addable) A-matrices?

Complements report.md, Finding 2: like the Givens→compiler benchmark, this
measures the cost of an unstructured (all-pairs two-qubit) ansatz on
A-matrices, here as layer count vs precision instead of CX count vs k.

Expected result: the required layers plateau at ≈4 for ε between 1e-2 and
1e-6, with the maximum over diagrams growing sharply only below ε ≈ 1e-7.

The circuit template is preserved verbatim from plot_raau3_layers.py (and
originally play.ipynb).  Defaults are sized for a fast bare run; the
archived data/raau3_layers_vs_epsilon.png used --num-diagrams 100 and a
sweep down to ε = 1e-10.

Outputs: data/raau3_layers.csv and data/plots/raau3_layers_vs_epsilon.png.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from bqskit.ir import Circuit
from bqskit.ir.gates import RXXGate, RYYGate, RZZGate, U3Gate

from fourier import a_matrix, diagrams_with_addable_cells

NUM_QUBITS = 3
PAIRS = [(0, 1), (0, 2), (1, 2)]


def build_template(entangling_layers: int) -> Circuit:
    """The play.ipynb ansatz: per layer, U3 on every qubit then RXX/RYY/RZZ
    on every pair; a final U3 column closes the circuit."""
    circ = Circuit(NUM_QUBITS)
    for _ in range(entangling_layers):
        for q in range(NUM_QUBITS):
            circ.append_gate(U3Gate(), [q])
        for i, j in PAIRS:
            circ.append_gate(RXXGate(), [i, j])
            circ.append_gate(RYYGate(), [i, j])
            circ.append_gate(RZZGate(), [i, j])
    for q in range(NUM_QUBITS):
        circ.append_gate(U3Gate(), [q])
    return circ


def unitary_distance(U: np.ndarray, V: np.ndarray) -> float:
    """Global-phase-invariant relative Frobenius distance between U and V."""
    vdot = np.vdot(V, U)
    phase = vdot / abs(vdot) if vdot != 0 else 1.0
    return float(np.linalg.norm(U - phase * V) / np.linalg.norm(V))


def layers_needed(target: np.ndarray, eps: float, max_layers: int) -> int:
    """Smallest layer count whose instantiated template reaches distance ≤ eps
    from `target`; max_layers + 1 if none does."""
    for layer_count in range(1, max_layers + 1):
        instantiated = build_template(layer_count).instantiate(target)
        unitary = np.asarray(instantiated.get_unitary(), dtype=np.complex128)
        if unitary_distance(unitary, target) <= eps:
            return layer_count
    return max_layers + 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-diagram-size", type=int, default=30,
                        help="upper bound for diagram size (default 30; "
                             "8-addable diagrams start at size 28)")
    parser.add_argument("--num-diagrams", type=int, default=8,
                        help="diagrams to sample per epsilon (default 8; "
                             "the archived plot used 100)")
    parser.add_argument("--eps-start", type=float, default=1e-1,
                        help="largest epsilon in the sweep (default 1e-1)")
    parser.add_argument("--eps-end", type=float, default=1e-6,
                        help="smallest epsilon in the sweep (default 1e-6)")
    parser.add_argument("--eps-steps", type=int, default=6,
                        help="number of log-spaced epsilon points (default 6)")
    parser.add_argument("--max-layers", type=int, default=6,
                        help="maximum entangling layers to try (default 6)")
    parser.add_argument("--output", type=Path,
                        default=Path("data/plots/raau3_layers_vs_epsilon.png"),
                        help="path for the plot image")
    args = parser.parse_args()

    diagrams = list(
        diagrams_with_addable_cells(1 << NUM_QUBITS, args.max_diagram_size)
    )
    if not diagrams:
        raise SystemExit("No diagrams found for the requested size bound.")
    if len(diagrams) < args.num_diagrams:
        print(f"Warning: only {len(diagrams)} diagrams available; using all of them.")
    diagrams = diagrams[: args.num_diagrams]

    for layer_count in range(1, args.max_layers + 1):
        free_params = sum(op.num_params for op in build_template(layer_count))
        print(f"layers={layer_count} free_params={free_params} "
              f"U(8) params={(1 << NUM_QUBITS) ** 2}")

    epsilons = np.logspace(
        math.log10(args.eps_start), math.log10(args.eps_end), args.eps_steps
    )

    avg_layers: list[float] = []
    max_layers: list[int] = []
    for eps in epsilons:
        per_diagram = [
            layers_needed(a_matrix(yd), eps, args.max_layers) for yd in diagrams
        ]
        avg_layers.append(float(np.mean(per_diagram)))
        max_layers.append(int(max(per_diagram)))
        print(f"epsilon={eps:.1e} avg_layers={avg_layers[-1]:.2f} "
              f"max_layers={max_layers[-1]}")

    csv_path = Path("data/raau3_layers.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epsilon", "avg_layers", "max_layers"])
        writer.writerows(zip(epsilons, avg_layers, max_layers))
    print(f"Wrote {csv_path}")

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


if __name__ == "__main__":
    main()
