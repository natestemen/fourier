#!/usr/bin/env python3
"""Test commutant heuristic for density of A-matrix subgroup in O(N)."""
from __future__ import annotations

import argparse
import random

import numpy as np
import sympy as sp
from scipy.linalg import null_space

from symbolic_a_matrix import build_symbolic_a_matrix, build_symbolic_a_matrix_8addable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-qubits",
        type=int,
        choices=(2, 3),
        default=2,
        help="Number of qubits (2 -> 4x4, 3 -> 8x8).",
    )
    parser.add_argument("--count", type=int, default=12, help="Number of random matrices.")
    parser.add_argument("--max-size", type=int, default=60, help="Max diagram size to sample.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (0 for random).")
    parser.add_argument(
        "--w-min",
        type=int,
        default=2,
        help="Minimum width parameter w_i (default: 2).",
    )
    parser.add_argument(
        "--h-min",
        type=int,
        default=2,
        help="Minimum height parameter h_i (default: 2).",
    )
    parser.add_argument(
        "--require-unitary",
        action="store_true",
        default=False,
        help="Skip matrices that are not numerically unitary (slower).",
    )
    parser.add_argument(
        "--nullspace-rcond",
        type=float,
        default=1e-10,
        help="Relative condition threshold for nullspace computation.",
    )
    return parser.parse_args()


def _is_unitary(mat: np.ndarray, tol: float = 1e-6) -> bool:
    eye = np.eye(mat.shape[0], dtype=complex)
    return np.allclose(mat.conj().T @ mat, eye, atol=tol)


def _size_from_params(w1: int, w2: int, w3: int, h1: int, h2: int, h3: int) -> int:
    return (w1 + w2 + w3) * h1 + (w2 + w3) * h2 + w3 * h3


def _valid_symbolic_params(
    w1: int, w2: int, w3: int, h1: int, h2: int, h3: int, w_min: int, h_min: int
) -> bool:
    return min(w1, w2, w3) >= w_min and min(h1, h2, h3) >= h_min


def _candidate_triples(max_size: int, w_min: int, h_min: int) -> list[tuple[int, int, int]]:
    triples: list[tuple[int, int, int]] = []
    for w3 in range(w_min, max_size // 3 + 1):
        max_w2 = (max_size - 3 * w3) // 2
        for w2 in range(w_min, max_w2 + 1):
            max_w1 = max_size - 2 * w2 - 3 * w3
            for w1 in range(w_min, max_w1 + 1):
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
        if not _valid_symbolic_params(w1, w2, w3, h1, h2, h3, w_min, h_min):
            continue
        params.append((w1, w2, w3, h1, h2, h3))
    return params


def _sample_params_8(
    count: int, max_size: int, seed: int, h_min: int, w_min: int
) -> list[tuple[int, ...]]:
    # Widths w1..w7 with constraint: h_min * sum_{j=1}^7 j * w_j <= max_size.
    rng = random.Random() if seed == 0 else random.Random(seed)
    params: list[tuple[int, ...]] = []
    max_tries = max(2000, count * 80)
    tries = 0

    min_size = 28 * w_min * h_min
    if max_size < min_size:
        return []

    weights = [j for j in range(1, 8)]  # j=1..7

    while len(params) < count and tries < max_tries:
        tries += 1
        # Sample widths with the linear constraint.
        budget = max_size // h_min
        widths: list[int] = []
        used = 0
        for idx, weight in enumerate(weights, start=1):
            remaining_min = sum(weights[idx:]) * w_min
            max_w = (budget - used - remaining_min) // weight
            if max_w < w_min:
                widths = []
                break
            widths.append(rng.randint(w_min, max_w))
            used += weight * widths[-1]
        if not widths:
            continue

        # Compute suffix sums S_i = sum_{j=i}^7 w_j.
        suffix = [0] * 8
        for i in range(6, -1, -1):
            suffix[i] = suffix[i + 1] + widths[i]
        weights_h = suffix[:7]

        # Sample heights with constraint sum S_i * h_i <= max_size.
        heights: list[int] = []
        used_size = 0
        for i, wgt in enumerate(weights_h):
            remaining_min = sum(weights_h[i + 1 :]) * h_min
            max_h = (max_size - used_size - remaining_min) // wgt
            if max_h < h_min:
                heights = []
                break
            heights.append(rng.randint(h_min, max_h))
            used_size += wgt * heights[-1]
        if not heights:
            continue

        params.append((*widths, *heights))

    return params


def _commutant_system(mats: list[np.ndarray]) -> np.ndarray:
    if not mats:
        raise ValueError("Need at least one matrix.")
    n = mats[0].shape[0]
    eye = np.eye(n, dtype=complex)
    blocks = []
    for A in mats:
        blocks.append(np.kron(A, eye) - np.kron(eye, A.T))
    return np.vstack(blocks)


def main() -> None:
    args = parse_args()

    if args.num_qubits == 3:
        A_sym, symbols = build_symbolic_a_matrix_8addable()
    else:
        A_sym, symbols = build_symbolic_a_matrix()
    A_func = sp.lambdify(symbols, A_sym, "numpy")

    h_min = args.h_min
    w_min = args.w_min
    if h_min < 2 or w_min < 2:
        print(
            "Warning: w_min/h_min < 2 can invalidate the symbolic removable/addable assumptions."
        )
    if args.num_qubits == 3:
        params = _sample_params_8(args.count, args.max_size, args.seed, h_min, w_min)
    else:
        params = _sample_params(args.count, args.max_size, args.seed, h_min, w_min)
    if not params:
        min_size = 28 * w_min * h_min
        raise SystemExit(
            "No valid samples fit the max-size constraint "
            f"(min size is {min_size} for 3 qubits)."
        )
    if len(params) < args.count:
        print(f"Warning: only {len(params)} samples fit max size {args.max_size}.")

    matrices: list[np.ndarray] = []
    for param in params:
        A_val = np.array(A_func(*param), dtype=complex)
        if args.require_unitary and not _is_unitary(A_val):
            continue
        matrices.append(A_val)

    if not matrices:
        raise SystemExit("No valid matrices generated.")

    system = _commutant_system(matrices)
    ns = null_space(system, rcond=args.nullspace_rcond)
    dim = ns.shape[1]

    print(f"Commutant nullspace dimension: {dim}")
    if dim > 1:
        print("Warning: nontrivial commuting operator detected; likely not dense.")
    else:
        print("Commutant is trivial; heuristic density test passes.")


if __name__ == "__main__":
    main()
