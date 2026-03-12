#!/usr/bin/env python3
"""Fast scatter of Weyl (b, |c|) using symbolic A-matrix sampling (no family)."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from qiskit.synthesis import TwoQubitWeylDecomposition

from symbolic_a_matrix import build_symbolic_a_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=1000, help="Number of points to plot.")
    parser.add_argument("--max-size", type=int, default=60, help="Max diagram size to sample.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (0 for random).")
    parser.add_argument(
        "--require-unitary",
        action="store_true",
        default=False,
        help="Skip matrices that are not numerically unitary (slower).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/plots/weyl_bc_random.png"),
        help="Output path for the plot image.",
    )
    return parser.parse_args()


def _is_unitary(mat: np.ndarray, tol: float = 1e-6) -> bool:
    eye = np.eye(mat.shape[0], dtype=complex)
    return np.allclose(mat.conj().T @ mat, eye, atol=tol)


def _size_from_params(w1: int, w2: int, w3: int, h1: int, h2: int, h3: int) -> int:
    return (w1 + w2 + w3) * h1 + (w2 + w3) * h2 + w3 * h3


def _valid_symbolic_params(w1: int, w2: int, w3: int, h1: int, h2: int, h3: int) -> bool:
    # Match symbolic generic assumptions for addable/removable cells.
    return min(w1, w2, w3, h1, h2, h3) >= 2


def _candidate_triples(max_size: int, w_min: int, h_min: int) -> list[tuple[int, int, int]]:
    triples: list[tuple[int, int, int]] = []
    for w3 in range(w_min, max_size // 3 + 1):
        max_w2 = (max_size - 3 * w3) // 2
        for w2 in range(w_min, max_w2 + 1):
            max_w1 = max_size - 2 * w2 - 3 * w3
            for w1 in range(w_min, max_w1 + 1):
                # Ensure at least minimal height fits.
                if _size_from_params(w1, w2, w3, h_min, h_min, h_min) <= max_size:
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


def _sample_params(
    count: int, max_size: int, seed: int, h_min: int, w_min: int
) -> list[tuple[int, int, int, int, int, int]]:
    triples = _candidate_triples(max_size, w_min, h_min)
    if not triples:
        return []

    rng = random.Random() if seed == 0 else random.Random(seed)
    params: list[tuple[int, int, int, int, int, int]] = []
    max_tries = max(1000, count * 50)
    tries = 0
    while len(params) < count and tries < max_tries:
        tries += 1
        w1, w2, w3 = rng.choice(triples)
        h_tuple = _random_h_tuple(rng, w1, w2, w3, max_size, h_min)
        if h_tuple is None:
            continue
        h1, h2, h3 = h_tuple
        if not _valid_symbolic_params(w1, w2, w3, h1, h2, h3):
            continue
        params.append((w1, w2, w3, h1, h2, h3))
    return params


def _add_weyl_bc_boundary(ax) -> None:
    pi4 = 0.25 * np.pi
    ax.plot([0, pi4], [0, 0], color="black", linewidth=1)
    ax.plot([0, pi4], [0, pi4], color="black", linewidth=1)
    ax.plot([pi4, pi4], [0, pi4], color="black", linewidth=1)


def main() -> None:
    args = parse_args()

    A_sym, symbols = build_symbolic_a_matrix()
    A_func = sp.lambdify(symbols, A_sym, "numpy")

    h_min = 2
    w_min = 2
    samples = _sample_params(args.count, args.max_size, args.seed, h_min, w_min)
    if not samples:
        raise SystemExit("No valid samples fit the max-size constraint.")
    if len(samples) < args.count:
        print(f"Warning: only {len(samples)} samples fit max size {args.max_size}.")

    bs: list[float] = []
    cs: list[float] = []
    for w1, w2, w3, h1, h2, h3 in samples:
        A_val = np.array(A_func(w1, w2, w3, h1, h2, h3), dtype=complex)
        if args.require_unitary and not _is_unitary(A_val):
            continue
        try:
            decomp = TwoQubitWeylDecomposition(A_val)
        except Exception:
            continue
        bs.append(float(decomp.b))
        cs.append(abs(float(decomp.c)))

    if not bs:
        raise SystemExit("No valid Weyl points computed.")

    fig, ax = plt.subplots()
    _add_weyl_bc_boundary(ax)
    ax.scatter(bs, cs, s=8, alpha=0.5, color="tab:blue")
    ax.set_xlabel("b")
    ax.set_ylabel("|c|")
    ax.set_title("Weyl (b, |c|) scatter (symbolic sampling)")
    ax.grid(True, alpha=0.3)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
