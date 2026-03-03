#!/usr/bin/env python3
"""Compile random 4x4 A matrices into {u3, cry} and report multi-qubit gate counts."""
from __future__ import annotations

import argparse
import random

import numpy as np

import bqskit
from bqskit.compiler.machine import MachineModel
from bqskit.ir.gates import U3Gate, CRYGate


from compute_matrix import A_matrix
from helper import find_yds_with_fixed_addable_cells


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-qubits", type=int, default=2, help="Number of qubits.")
    parser.add_argument("--max-size", type=int, default=30, help="Max diagram size to search.")
    parser.add_argument("--count", type=int, default=10, help="Number of random matrices to compile.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (0 for random).")
    parser.add_argument("--opt", type=int, default=1, help="Transpile optimization level.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(None if args.seed == 0 else args.seed)

    if args.num_qubits < 1:
        raise SystemExit("--num-qubits must be >= 1.")
    addable = 1 << args.num_qubits
    diagrams = list(find_yds_with_fixed_addable_cells(addable, args.max_size))
    if not diagrams:
        raise SystemExit(f"No diagrams found with {addable} addable cells.")

    if len(diagrams) < args.count:
        print(f"Warning: only {len(diagrams)} diagrams available; using all of them.")
        sample = diagrams
    else:
        sample = rng.sample(diagrams, args.count)

    for i, d in enumerate(sample, start=1):
        mat = np.array(A_matrix(d), dtype=complex)
        circuit = bqskit.Circuit.from_unitary(mat)
        model = MachineModel(args.num_qubits, gate_set={U3Gate(), CRYGate()})
        compiled = bqskit.compile(circuit, model=model, optimization_level=args.opt)

        counts = {}
        multi = 0
        for op in compiled:
            name = op.gate.name
            counts[name] = counts.get(name, 0) + 1
            if op.gate.num_qudits > 1:
                multi += 1

        print("=" * 72)
        print(f"[{i}] diagram: {d.partition}")
        print(f"gate counts: {counts}")


if __name__ == "__main__":
    main()
