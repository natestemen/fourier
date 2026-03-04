#!/usr/bin/env python3
"""Compile random 2-qubit A matrices into {RY, RZ, CNOT} gate set using bqskit."""
from __future__ import annotations

import argparse
import random

import numpy as np
import bqskit
from bqskit.compiler.machine import MachineModel
from bqskit.ir.gates import RYGate, ZGate, CNOTGate

from compute_matrix import A_matrix
from helper import find_yds_with_fixed_addable_cells


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-size", type=int, default=30)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opt", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(None if args.seed == 0 else args.seed)

    diagrams = list(find_yds_with_fixed_addable_cells(4, args.max_size))
    if not diagrams:
        raise SystemExit("No diagrams found with 4 addable cells.")
    if len(diagrams) < args.count:
        sample = diagrams
    else:
        sample = rng.sample(diagrams, args.count)

    model = MachineModel(2, gate_set={RYGate(), ZGate(), CNOTGate()})

    for i, d in enumerate(sample, start=1):
        A = np.array(A_matrix(d), dtype=float)
        print("First det: ", np.linalg.det(A))
        circuit = bqskit.Circuit.from_unitary(A)
        compiled = bqskit.compile(circuit, model=model, optimization_level=args.opt)

        counts = {}
        cnot_count = 0
        for op in compiled:
            name = op.gate.name
            counts[name] = counts.get(name, 0) + 1
            if isinstance(op.gate, CNOTGate):
                cnot_count += 1

        print("=" * 72)
        print(f"[{i}] diagram: {getattr(d, 'partition', d)}")
        print(f"gate counts: {counts}")
        from bqskit.ext import bqskit_to_qiskit
        qc = bqskit_to_qiskit(compiled)

        print(np.linalg.det(compiled.get_unitary()))
        print(qc)
        # print(f"CNOT count: {cnot_count}")


if __name__ == "__main__":
    main()
