#!/usr/bin/env python3
"""Compute KAK (Cartan) k-vector for the A matrix via Tucci's SVD-based magic-basis method."""
from __future__ import annotations

import argparse
from typing import Sequence

import sympy as sp

from symbolic_a_matrix import build_symbolic_a_matrix


MAGIC_BASIS = (sp.Matrix(
    [
        [1, 0, 0, 1],
        [0, sp.I, sp.I, 0],
        [0, -1, 1, 0],
        [sp.I, 0, 0, -sp.I],
    ]
) / sp.sqrt(2))

# Hadamard-like matrix Γ from Tucci (Eq. 33)
GAMMA = sp.Matrix(
    [
        [1, 1, 1, 1],
        [1, 1, -1, -1],
        [1, -1, 1, -1],
        [1, -1, -1, 1],
    ]
)

# User-provided simplification: P = H ⊕ H (Hadamard direct sum)
H2 = sp.Matrix([[1, 1], [1, -1]]) / sp.sqrt(2)
P_SYM = sp.diag(H2, H2)


def kak_k_vector_svd(A: sp.Matrix) -> tuple[sp.Matrix, sp.Matrix]:
    """Return (k0,k1,k2,k3) and theta from A using Tucci's SVD-based algorithm.

    Steps (per Tucci):
      X' = M^† X M.
      Let XR = Re(X'), XI = Im(X').
      Find orthogonal QL, QR s.t. DR = QL^T XR QR and DI = QL^T XI QR are diagonal.
      For full-rank A, we can use the EY theorem with A=XR, B=XI and P = H⊕H.
      Then e^{iΘ} = QL^T X' QR and (k0,k1,k2,k3)^T = Γ^T θ / 4.
    """
    M = MAGIC_BASIS
    Xp = M.H * A * M

    XR = Xp.applyfunc(sp.re)
    XI = Xp.applyfunc(sp.im)

    # SVD of XR (real), XR = UA * D * VA.T
    UA, D, VA = XR.singular_value_decomposition()

    # P = H ⊕ H (given)
    P = P_SYM

    # QL, QR as in EY when full rank and P simultaneously diagonalizes D,G
    QL = UA * P
    QR = VA * P

    E = (QL.T * Xp * QR).applyfunc(sp.simplify)
    diag = [E[i, i] for i in range(4)]
    theta = sp.Matrix([sp.atan2(sp.im(d), sp.re(d)) for d in diag])
    k = (GAMMA.T / 4) * theta
    return sp.simplify(k), sp.simplify(theta)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--w1", type=int)
    parser.add_argument("--w2", type=int)
    parser.add_argument("--w3", type=int)
    parser.add_argument("--h1", type=int)
    parser.add_argument("--h2", type=int)
    parser.add_argument("--h3", type=int)
    parser.add_argument(
        "--show-m",
        action="store_true",
        help="Print the magic basis matrix M and Gamma matrix.",
    )
    return parser.parse_args()


def _maybe_substitute(A: sp.Matrix, symbols: Sequence[sp.Symbol], args: argparse.Namespace) -> sp.Matrix:
    if None in (args.w1, args.w2, args.w3, args.h1, args.h2, args.h3):
        return A
    subs = {
        symbols[0]: args.w1,
        symbols[1]: args.w2,
        symbols[2]: args.w3,
        symbols[3]: args.h1,
        symbols[4]: args.h2,
        symbols[5]: args.h3,
    }
    return sp.simplify(A.subs(subs))


def main() -> None:
    args = parse_args()
    A_sym, symbols = build_symbolic_a_matrix()
    A = _maybe_substitute(A_sym, symbols, args)

    if args.show_m:
        print("Magic basis M:")
        sp.pprint(MAGIC_BASIS, use_unicode=False)
        print("Gamma:")
        sp.pprint(GAMMA, use_unicode=False)
        print()

    print("A matrix:")
    sp.pprint(A, use_unicode=False)
    print()

    if None in (args.w1, args.w2, args.w3, args.h1, args.h2, args.h3):
        raise SystemExit(
            "SVD-based KAK needs numeric values. Provide --w1 --w2 --w3 --h1 --h2 --h3."
        )

    k_vec, theta = kak_k_vector_svd(A)
    print("theta (phases of e^{iΘ}):")
    sp.pprint(theta, use_unicode=False)
    print("k-vector (k0,k1,k2,k3):")
    sp.pprint(k_vec, use_unicode=False)


if __name__ == "__main__":
    main()
