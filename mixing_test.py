#!/usr/bin/env python3
"""Heuristic mixing test: random products of A matrices and rank of vec(M)."""
from __future__ import annotations

import argparse
import random

import numpy as np
import sympy as sp
from scipy.linalg import logm

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
    parser.add_argument("--count", type=int, default=8, help="Number of base matrices k.")
    parser.add_argument("--max-size", type=int, default=60, help="Max diagram size to sample.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (0 for random).")
    parser.add_argument("--length", type=int, default=6, help="Product length L.")
    parser.add_argument("--num-products", type=int, default=200, help="Number of products to sample.")
    parser.add_argument(
        "--method",
        choices=("logm", "diff"),
        default="diff",
        help="Use M=logm(P) or M=P-I.",
    )
    parser.add_argument(
        "--rank-tol",
        type=float,
        default=1e-8,
        help="Singular value threshold for rank estimation.",
    )
    parser.add_argument(
        "--svd-top",
        type=int,
        default=12,
        help="Number of top singular values to print.",
    )
    parser.add_argument(
        "--require-unitary",
        action="store_true",
        default=False,
        help="Skip matrices that are not numerically unitary (slower).",
    )
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

        suffix = [0] * 8
        for i in range(6, -1, -1):
            suffix[i] = suffix[i + 1] + widths[i]
        weights_h = suffix[:7]

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


def _random_product(rng: random.Random, mats: list[np.ndarray], length: int) -> np.ndarray:
    n = mats[0].shape[0]
    P = np.eye(n, dtype=complex)
    for _ in range(length):
        P = mats[rng.randrange(len(mats))] @ P
    return P


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
        min_size = 28 * w_min * h_min if args.num_qubits == 3 else 0
        msg = "No valid samples fit the max-size constraint."
        if args.num_qubits == 3:
            msg += f" (min size is {min_size} for 3 qubits)."
        raise SystemExit(msg)
    if len(params) < args.count:
        print(f"Warning: only {len(params)} samples fit max size {args.max_size}.")

    mats: list[np.ndarray] = []
    for param in params:
        A_val = np.array(A_func(*param), dtype=complex)
        if args.require_unitary and not _is_unitary(A_val):
            continue
        mats.append(A_val)
    if not mats:
        raise SystemExit("No valid matrices generated.")

    rng = random.Random() if args.seed == 0 else random.Random(args.seed + 1)
    n = mats[0].shape[0]
    eye = np.eye(n, dtype=complex)

    vecs = []
    for _ in range(args.num_products):
        P = _random_product(rng, mats, args.length)
        if args.method == "logm":
            M = logm(P)
        else:
            M = P - eye
        vecs.append(M.reshape(-1))

    V = np.vstack(vecs)
    svals = np.linalg.svd(V, compute_uv=False)
    rank = int(np.sum(svals > args.rank_tol))

    full_rank = min(V.shape)
    print(f"Collected {len(vecs)} products of length {args.length}")
    print(f"Matrix size: {n}x{n}; vec dimension: {n*n}")
    print(f"Rank estimate (tol={args.rank_tol:g}): {rank} / {full_rank}")
    print(f"Top singular values: {svals[: args.svd_top]}")
    print(f"Smallest singular value: {svals[-1]:.6g}")

    if rank < full_rank:
        print("Rank deficient: indicates a hidden invariant subspace.")
    else:
        print("Full rank: products explore the space; good sign for density.")


if __name__ == "__main__":
    main()
