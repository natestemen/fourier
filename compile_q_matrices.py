#!/usr/bin/env python3
"""Build Q matrices from block-rotation procedure and compile them with bqskit."""
from __future__ import annotations

import argparse
import random

import numpy as np
import bqskit
from bqskit.compiler.machine import MachineModel
from bqskit.ir.gates import U3Gate, SqrtCNOTGate

from compute_matrix import A_matrix
from helper import find_yds_with_fixed_addable_cells


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-size", type=int, default=30)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eig-tol", type=float, default=1e-6)
    parser.add_argument("--opt", type=int, default=4)
    return parser.parse_args()


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _orthonormal_complement(V: np.ndarray) -> np.ndarray:
    basis = []
    P = V @ V.T
    for i in range(4):
        e = np.zeros(4)
        e[i] = 1.0
        v = e - P @ e
        if np.linalg.norm(v) > 1e-8:
            basis.append(v)
    M = np.stack(basis, axis=1)
    Q, _ = np.linalg.qr(M)
    return Q[:, :2]


def _build_q(A: np.ndarray, eig_tol: float) -> np.ndarray | None:
    w, V = np.linalg.eig(A)
    idx1 = np.argmin(np.abs(w - 1.0))
    idxm1 = np.argmin(np.abs(w + 1.0))
    if abs(w[idx1] - 1.0) > eig_tol or abs(w[idxm1] + 1.0) > eig_tol:
        return None

    v1 = _normalize(V[:, idx1].real)
    v2 = _normalize(V[:, idxm1].real)
    v2 = v2 - np.dot(v1, v2) * v1
    v2 = _normalize(v2)

    V12 = np.stack([v1, v2], axis=1)
    V34 = _orthonormal_complement(V12)
    v3, v4 = V34[:, 0], V34[:, 1]

    Q = np.column_stack([v1, v2, v3, v4])
    Q, _ = np.linalg.qr(Q)
    return Q


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

    # model = MachineModel(2, gate_set={U3Gate(), SqrtCNOTGate()})

    for i, d in enumerate(sample, start=1):
        A = np.array(A_matrix(d), dtype=float)
        Q = _build_q(A, args.eig_tol)
        print("=" * 72)
        print(f"[{i}] diagram: {getattr(d, 'partition', d)}")
        if Q is None:
            print("skipping: could not form Q.")
            continue

        circuit = bqskit.Circuit.from_unitary(Q)
        compiled = bqskit.compile(circuit, optimization_level=args.opt)

        counts = {}
        for op in compiled:
            name = op.gate.name
            counts[name] = counts.get(name, 0) + 1

        print(f"gate counts: {counts}")
        from bqskit.ext import bqskit_to_qiskit
        qc = bqskit_to_qiskit(compiled)
        print(qc)


if __name__ == "__main__":
    main()
