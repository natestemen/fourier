#!/usr/bin/env python3
"""Generate random 2-qubit A matrices and compute their eigenvalues."""
from __future__ import annotations

import argparse
import random
import cmath

import sympy as sp
from itertools import groupby

from symbolic_a_matrix import build_symbolic_a_matrix
from helper import find_yds_with_fixed_addable_cells


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-size", type=int, default=30, help="Max diagram size to search.")
    parser.add_argument("--count", type=int, default=10, help="Number of random diagrams to test.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (0 for random).")
    parser.add_argument("--sort", action="store_true", help="Sort eigenvalues by angle then magnitude.")
    parser.add_argument(
        "--approx",
        action="store_true",
        help="Also print a numerical approximation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(None if args.seed == 0 else args.seed)
    A_sym, symbols = build_symbolic_a_matrix()

    diagrams = list(find_yds_with_fixed_addable_cells(4, args.max_size))
    if not diagrams:
        raise SystemExit("No diagrams found with 4 addable cells.")

    if len(diagrams) < args.count:
        print(f"Warning: only {len(diagrams)} diagrams available; using all of them.")
        sample = diagrams
    else:
        sample = rng.sample(diagrams, args.count)

    for i, d in enumerate(sample, start=1):
        partition = list(getattr(d, "partition", d))
        groups = [(val, len(list(g))) for val, g in groupby(partition)]
        if len(groups) != 3:
            print("=" * 72)
            print(f"[{i}] diagram: {partition}")
            print("skipping: expected exactly 3 distinct row lengths for symbolic parameterization.")
            continue
        (w1, h1), (w2, h2), (w3, h3) = groups
        subs = {
            symbols[0]: w1,
            symbols[1]: w2,
            symbols[2]: w3,
            symbols[3]: h1,
            symbols[4]: h2,
            symbols[5]: h3,
        }
        mat = sp.Matrix(A_sym.subs(subs))
        eig_dict = mat.eigenvals()
        eigvals = []
        for val, mult in eig_dict.items():
            eigvals.extend([val] * int(mult))
        if args.sort:
            def _sort_key(z):
                zc = complex(sp.N(z))
                return (cmath.phase(zc), abs(zc))

            eigvals = sorted(eigvals, key=_sort_key)
        print("=" * 72)
        print(f"[{i}] diagram: {getattr(d, 'partition', d)}")
        print("eigenvalues:")
        for ev in eigvals:
            ev_s = sp.simplify(ev)
            if args.approx:
                ev_n = sp.N(ev_s, 16)
                print(f"  {ev_s}  ≈  {ev_n}")
            else:
                print(f"  {ev_s}")


if __name__ == "__main__":
    main()
