#!/usr/bin/env python3
"""Generate A matrices (sympy), multiply by CZ, compile with Qiskit, and interactively print circuits."""
from __future__ import annotations

import argparse
import random
from typing import Iterator

import numpy as np
import sympy as sp
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator

from symbolic_a_matrix import build_symbolic_a_matrix


CZ = np.diag([1, 1, 1, -1]).astype(complex)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-boxes", type=int, default=30, help="Max number of boxes.")
    parser.add_argument("--count", type=int, default=10, help="How many diagrams to compile.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (0 for random).")
    parser.add_argument(
        "--max-tries",
        type=int,
        default=10000,
        help="Maximum attempts to find valid diagrams.",
    )
    parser.add_argument(
        "--cz-left",
        action="store_true",
        help="Use CZ @ A instead of A @ CZ.",
    )
    parser.add_argument(
        "--no-cz",
        action="store_true",
        help="Do not apply CZ; use A directly.",
    )
    parser.add_argument(
        "--basis",
        default="u3,cx",
        help="Comma-separated basis gates for transpile (default: u3,cx).",
    )
    parser.add_argument("--opt", type=int, default=1, help="Qiskit optimization level.")
    return parser.parse_args()


def _iter_params(args: argparse.Namespace) -> Iterator[tuple[int, int, int, int, int, int]]:
    rng = random.Random(None if args.seed == 0 else args.seed)
    generated = 0
    tries = 0
    while generated < args.count and tries < args.max_tries:
        tries += 1
        if args.max_boxes < 6:
            # Minimum for w1>w2>w3 with h1=h2=h3=1 is 6 (3+2+1).
            break

        # Sample widths first (strict ordering).
        w3 = rng.randint(1, args.max_boxes - 2)
        w2 = rng.randint(w3 + 1, args.max_boxes - 1)
        w1 = rng.randint(w2 + 1, args.max_boxes)

        min_size = w1 + w2 + w3
        if min_size > args.max_boxes:
            continue

        # Iteratively allocate heights while respecting the remaining box budget.
        remaining = args.max_boxes
        max_h1 = (remaining - (w2 + w3)) // w1
        if max_h1 < 1:
            continue
        h1 = rng.randint(1, max_h1)
        remaining -= w1 * h1

        max_h2 = (remaining - w3) // w2
        if max_h2 < 1:
            continue
        h2 = rng.randint(1, max_h2)
        remaining -= w2 * h2

        max_h3 = remaining // w3
        if max_h3 < 1:
            continue
        h3 = rng.randint(1, max_h3)

        yield (w1, w2, w3, h1, h2, h3)
        generated += 1

    if generated < args.count:
        print(
            f"Warning: only generated {generated} diagrams within {args.max_tries} tries."
        )


def _is_unitary(mat: np.ndarray, tol: float = 1e-6) -> bool:
    eye = np.eye(mat.shape[0], dtype=complex)
    return np.allclose(mat.conj().T @ mat, eye, atol=tol)


def main() -> None:
    args = parse_args()
    A_sym, symbols = build_symbolic_a_matrix()
    basis_gates = [g.strip() for g in args.basis.split(",") if g.strip()]

    for idx, (w1, w2, w3, h1, h2, h3) in enumerate(_iter_params(args), start=1):
        subs = {
            symbols[0]: w1,
            symbols[1]: w2,
            symbols[2]: w3,
            symbols[3]: h1,
            symbols[4]: h2,
            symbols[5]: h3,
        }
        A = sp.Matrix(A_sym.subs(subs))
        A_num = np.array(A.evalf(), dtype=complex)

        if args.no_cz:
            U = A_num
        else:
            U = CZ @ A_num if args.cz_left else A_num @ CZ
        if not _is_unitary(U):
            print(f"[{idx}] Skipping non-unitary for params {w1,w2,w3,h1,h2,h3}")
            continue

        qc = QuantumCircuit(2)
        qc.unitary(Operator(U), [0, 1])
        compiled = transpile(qc, basis_gates=basis_gates, optimization_level=args.opt)

        print("=" * 80)
        print(f"[{idx}] params: w1={w1}, w2={w2}, w3={w3}, h1={h1}, h2={h2}, h3={h3}")
        if args.no_cz:
            print("CZ placement: none (A only)")
        else:
            print("CZ placement:", "CZ @ A" if args.cz_left else "A @ CZ")
        print(compiled)

        resp = input("Press Enter for next (or 'q' to quit): ").strip().lower()
        if resp == "q":
            break


if __name__ == "__main__":
    main()
