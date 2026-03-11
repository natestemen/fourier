#!/usr/bin/env python3
"""Compute Weyl (a,b,c) for the AC family and plot its Weyl-plane projection.

Family (n must be divisible by 3):
  AC = {(n, 0), (2n/3, n/3), (n/3, 2n/3), (0, n)}

Equivalent block parameters: w1=w2=w3=h1=h2=h3=n/3.
Row lengths are: 3m (m rows), 2m (m rows), m (m rows) where m = n/3.

Removable cells (explicit):
  RC = {(n-1, n/3-1), (2n/3-1, 2n/3-1), (n/3-1, n-1)}

This script uses the symbolic A-matrix parametrization (w1,w2,w3,h1,h2,h3)
instead of YoungDiagram to speed up evaluation.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qiskit.synthesis import TwoQubitWeylDecomposition
import sympy as sp

from symbolic_a_matrix import build_symbolic_a_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-min", type=int, default=3, help="Smallest n (multiple of 3).")
    parser.add_argument("--n-max", type=int, default=60, help="Largest n (multiple of 3).")
    parser.add_argument("--step", type=int, default=3, help="Step for n (default: 3).")
    parser.add_argument(
        "--max-size",
        type=int,
        default=40,
        help="Max diagram size for random sampling.",
    )
    parser.add_argument(
        "--random-count",
        type=int,
        default=150,
        help="Number of random diagrams to plot (0 to skip).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (0 for random).")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/plots/ac_family_weyl_plane.png"),
        help="Output path for the (b,|c|) plot.",
    )
    return parser.parse_args()


def _family_params(n: int) -> tuple[int, int, int, int, int, int] | None:
    if n <= 0 or n % 3 != 0:
        return None
    m = n // 3
    return (m, m, m, m, m, m)


def _weyl_coefficients(matrix: np.ndarray) -> tuple[float, float, float] | None:
    decomp = TwoQubitWeylDecomposition(np.asarray(matrix, dtype=np.complex128))
    return float(decomp.a), float(decomp.b), float(decomp.c)


def _add_weyl_bc_boundary(ax) -> None:
    pi4 = 0.25 * np.pi
    ax.plot([0, pi4], [0, 0], color="black", linewidth=1)
    ax.plot([0, pi4], [0, pi4], color="black", linewidth=1)
    ax.plot([pi4, pi4], [0, pi4], color="black", linewidth=1)


def _random_w_triples(max_size: int, w_min: int) -> list[tuple[int, int, int]]:
    triples: list[tuple[int, int, int]] = []
    for w3 in range(w_min, max_size // 3 + 1):
        max_w2 = (max_size - 3 * w3) // 2
        for w2 in range(w_min, max_w2 + 1):
            max_w1 = max_size - 2 * w2 - 3 * w3
            for w1 in range(w_min, max_w1 + 1):
                triples.append((w1, w2, w3))
    return triples


def _random_h_tuple(
    rng: random.Random,
    w1: int,
    w2: int,
    w3: int,
    max_size: int,
    h_min: int,
) -> tuple[int, int, int] | None:
    denom1 = w1 + w2 + w3
    denom2 = w2 + w3
    denom3 = w3

    h1_max = (max_size - denom2 * h_min - denom3 * h_min) // denom1
    if h1_max < h_min:
        return None
    h1 = rng.randint(h_min, h1_max)

    h2_max = (max_size - denom1 * h1 - denom3 * h_min) // denom2
    if h2_max < h_min:
        return None
    h2 = rng.randint(h_min, h2_max)

    h3_max = (max_size - denom1 * h1 - denom2 * h2) // denom3
    if h3_max < h_min:
        return None
    h3 = rng.randint(h_min, h3_max)

    return h1, h2, h3


def _random_params(
    max_size: int,
    count: int,
    seed: int,
) -> list[tuple[int, int, int, int, int, int]]:
    h_min = 2
    w_min = 2
    triples = _random_w_triples(max_size, w_min)
    if not triples:
        return []
    rng = random.Random() if seed == 0 else random.Random(seed)
    params: set[tuple[int, int, int, int, int, int]] = set()
    max_tries = max(500, count * 50)
    tries = 0
    while len(params) < count and tries < max_tries:
        tries += 1
        w1, w2, w3 = rng.choice(triples)
        h_tuple = _random_h_tuple(rng, w1, w2, w3, max_size, h_min)
        if h_tuple is None:
            continue
        h1, h2, h3 = h_tuple
        params.add((w1, w2, w3, h1, h2, h3))
    return list(params)


def main() -> None:
    args = parse_args()

    print("AC family:")
    print("  AC = {(n, 0), (2n/3, n/3), (n/3, 2n/3), (0, n)}")
    print("  RC = {(n-1, n/3-1), (2n/3-1, 2n/3-1), (n/3-1, n-1)}")

    A_sym, symbols = build_symbolic_a_matrix()
    A_func = sp.lambdify(symbols, A_sym, "numpy")
    w1_sym, w2_sym, w3_sym, h1_sym, h2_sym, h3_sym = symbols

    family_rows: list[dict[str, object]] = []
    skipped_invalid = 0
    for n in range(args.n_min, args.n_max + 1, 3):
        w1, w2, w3, h1, h2, h3 = _family_params(n)
        A_val = np.array(A_func(w1, w2, w3, h1, h2, h3), dtype=complex)
        a, b, c = _weyl_coefficients(A_val)
        row = {
            "n": n,
            "m": n // 3,
            "weyl_a": a,
            "weyl_b": b,
            "weyl_c": c,
        }
        family_rows.append(row)

    if skipped_invalid:
        print(f"Skipped {skipped_invalid} n values because symbolic A assumes w_i,h_i >= 2.")
    if not family_rows:
        raise SystemExit("No valid family points computed; adjust --n-min/--n-max.")

    # Limit of the family as n -> infinity (m = n/3 -> infinity).
    m_sym = sp.symbols("m", positive=True, integer=True)
    A_family = A_sym.subs(
        {
            w1_sym: m_sym,
            w2_sym: m_sym,
            w3_sym: m_sym,
            h1_sym: m_sym,
            h2_sym: m_sym,
            h3_sym: m_sym,
        }
    )
    try:
        A_lim = A_family.applyfunc(lambda expr: sp.limit(expr, m_sym, sp.oo))
        print("Limit matrix (n -> inf):")
        sp.pprint(A_lim, use_unicode=False)
        A_lim_val = np.array(A_lim.evalf(), dtype=complex)
        lim_abc = _weyl_coefficients(A_lim_val)
        if lim_abc is not None:
            a_lim, b_lim, c_lim = lim_abc
            print(
                f"Limit Weyl: a={a_lim:.6g} b={b_lim:.6g} c={c_lim:.6g} |c|={abs(c_lim):.6g}"
            )
    except Exception as exc:
        print(f"Limit computation failed: {exc}")

    random_points = []
    if args.random_count > 0:
        samples = _random_params(args.max_size, args.random_count, args.seed)
        if not samples:
            print("Warning: no random parameter samples fit the max-size constraint.")
        elif len(samples) < args.random_count:
            print(f"Warning: only {len(samples)} random samples fit the max size.")
        for w1, w2, w3, h1, h2, h3 in samples:
            A_val = np.array(A_func(w1, w2, w3, h1, h2, h3), dtype=complex)
            _, b, c = _weyl_coefficients(A_val)
            random_points.append((b, abs(c)))

    family_b = np.array([row["weyl_b"] for row in family_rows], dtype=float)
    family_c = np.array([abs(row["weyl_c"]) for row in family_rows], dtype=float)

    fig, ax = plt.subplots()
    _add_weyl_bc_boundary(ax)

    if random_points:
        xs = [p[0] for p in random_points]
        ys = [p[1] for p in random_points]
        ax.scatter(xs, ys, s=12, alpha=0.35, color="gray", label="random diagrams")

    ax.plot(family_b, family_c, color="tab:blue", linewidth=1.5, label="AC family")
    ax.scatter(family_b, family_c, color="tab:blue", s=20)

    ax.set_xlabel("b")
    ax.set_ylabel("|c|")
    ax.set_title("AC family on Weyl plane")
    ax.grid(True, alpha=0.3)
    ax.legend()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
