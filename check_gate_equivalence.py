#!/usr/bin/env python3
"""Check if the pictured gate is locally equivalent to random A matrices (up to global phase)."""
from __future__ import annotations

import argparse
import random
from typing import Iterator

import numpy as np
import sympy as sp
from qiskit.synthesis import TwoQubitWeylDecomposition

from symbolic_a_matrix import build_symbolic_a_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-boxes", type=int, default=30)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tries", type=int, default=10000)
    parser.add_argument("--theta-steps", type=int, default=61)
    parser.add_argument("--phi-steps", type=int, default=73)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument(
        "--cz-left",
        action="store_true",
        help="Use CZ @ A instead of A @ CZ before Weyl comparison.",
    )
    parser.add_argument(
        "--no-cz",
        action="store_true",
        help="Do not apply CZ (default is to apply CZ).",
    )
    return parser.parse_args()


def _iter_params(args: argparse.Namespace) -> Iterator[tuple[int, int, int, int, int, int]]:
    rng = random.Random(None if args.seed == 0 else args.seed)
    generated = 0
    tries = 0
    while generated < args.count and tries < args.max_tries:
        tries += 1
        if args.max_boxes < 6:
            break
        w3 = rng.randint(1, args.max_boxes - 2)
        w2 = rng.randint(w3 + 1, args.max_boxes - 1)
        w1 = rng.randint(w2 + 1, args.max_boxes)

        min_size = w1 + w2 + w3
        if min_size > args.max_boxes:
            continue

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


def _gate_matrix(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array(
        [
            [1, 0, 0, 0],
            [0, c, -s, 0],
            [0, s, c, 0],
            [0, 0, 0, 1],
        ],
        dtype=complex,
    )


def _weyl_coords(U: np.ndarray) -> np.ndarray:
    decomp = TwoQubitWeylDecomposition(U)
    return np.array([float(decomp.a), float(decomp.b), float(decomp.c)])


def main() -> None:
    args = parse_args()
    A_sym, symbols = build_symbolic_a_matrix()
    CZ = np.diag([1, 1, 1, -1]).astype(complex)

    # Precompute gate Weyl coords on a grid for coarse matching.
    thetas = np.linspace(0, np.pi, args.theta_steps)
    grid = []
    for th in thetas:
        W = _weyl_coords(_gate_matrix(th))
        grid.append((th, W))

    for idx, (w1, w2, w3, h1, h2, h3) in enumerate(_iter_params(args), start=1):
        subs = {
            symbols[0]: w1,
            symbols[1]: w2,
            symbols[2]: w3,
            symbols[3]: h1,
            symbols[4]: h2,
            symbols[5]: h3,
        }
        A = np.array(sp.Matrix(A_sym.subs(subs)).evalf(), dtype=complex)
        if args.no_cz:
            U = A
            cz_desc = "none"
        else:
            U = CZ @ A if args.cz_left else A @ CZ
            cz_desc = "CZ @ A" if args.cz_left else "A @ CZ"
        W_A = _weyl_coords(U)

        best = None
        best_dist = float("inf")
        for th, W in grid:
            dist = float(np.linalg.norm(W - W_A))
            if dist < best_dist:
                best_dist = dist
                best = (th, W)

        print("=" * 80)
        print(f"[{idx}] params: w1={w1}, w2={w2}, w3={w3}, h1={h1}, h2={h2}, h3={h3}")
        print(f"CZ placement: {cz_desc}")
        print(f"A Weyl: {W_A}")
        if best is None:
            print("No grid candidates.")
            continue
        th, W = best
        print(f"Best gate match: theta={th:.6f}")
        print(f"Gate Weyl: {W}")
        print(f"Distance: {best_dist:.6g}")
        if best_dist <= args.tol:
            print("LIKELY locally equivalent (within tolerance)")
        else:
            print("No match within tolerance")


if __name__ == "__main__":
    main()
