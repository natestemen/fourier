#!/usr/bin/env python3
"""Verify the hand-typed one-parameter symbolic A-matrix family M(n).

M(n) is the 4×4 orthogonal matrix with entries like
√(n(n+2)/((n−1)(n+1)(n+3))) that was written down by hand in play.ipynb.
Cross-checking against the library shows it is exactly A((n, 2, 1)) — the
one-parameter gun-like family obtained by growing the first row.

Question: is M(n) a genuine A-matrix family, and what are its invariants?
Checks ported from the notebook:
  1. every row and column norm simplifies to 1 (orthogonality),
  2. eigenvalues at n = 3 are {+1, −1, e^{±iθ}},
  3. magic-basis KAK angles (the phases of the magic-basis eigenvalues),
  4. the entrywise n → ∞ limit exists and is itself orthogonal.

Supports report.md Finding 1: a one-parameter A-matrix family with
eigenvalues {+1, −1, e^{±iθ}} — the spectrum that pins the Weyl coordinate
at a = π/4 while θ (i.e. b, c) moves with n.

Expected result: all norms print as 1, M(n) = A((n,2,1)) to ~1e-16,
eigenvalue pair phases ±θ, and a unitary n → ∞ limit.

Behavior change vs. the notebook: the magic-basis eigenvalue computation is
done at a concrete n (default 3) rather than symbolically in n — the fully
symbolic quartic does not terminate in reasonable time.
"""

from __future__ import annotations

import argparse

import numpy as np
import sympy as sp
from sympy import I, Matrix, sqrt
from yungdiagram import YoungDiagram

from fourier import a_matrix

N = sp.symbols("n", real=True, positive=True)

# The magic basis used in the notebook's KAK cell.
MAGIC = (
    Matrix([[1, 0, 0, I], [0, I, 1, 0], [0, I, -1, 0], [1, 0, 0, -I]]) / sqrt(2)
)


def mn_matrix() -> sp.Matrix:
    """The hand-typed family M(n); equals A((n, 2, 1)) for integer n ≥ 3."""
    n = N
    return Matrix(
        [
            [
                sqrt(n * (n + 2) / ((n - 1) * (n + 1) * (n + 3))),
                sqrt(
                    n**2 * (n + 2) ** 2 * (n - 2) / ((n - 1) ** 2 * (n + 1) ** 2 * (n + 3))
                ),
                sqrt(3 * n**2 * (n + 2) / (2 * (n - 1) ** 2 * (n + 1) * (n + 3))) / n,
                sqrt(3 * n * (n + 2) ** 2 / (2 * (n + 1) ** 2 * (n - 1) * (n + 3)))
                / (n + 2),
            ],
            [
                sqrt(3 * (n - 2) / (8 * (n - 1))),
                sqrt(3 * n * (n - 2) ** 2 * (n + 2) / (8 * (n - 1) ** 2 * (n + 1)))
                / (2 - n),
                sqrt(9 * n * (n - 2) / (16 * (n - 1) ** 2)),
                sqrt(9 * (n - 2) * (n + 2) / (16 * (n - 1) * (n + 1))) / 3,
            ],
            [
                sqrt(n / (4 * (n + 1))),
                sqrt(n**2 * (n - 2) * (n + 2) / (4 * (n + 1) ** 2 * (n - 1))) / (-n),
                -sqrt(3 * n**2 / (8 * (n + 1) * (n - 1))),
                sqrt(3 * n * (n + 2) / (8 * (n + 1) ** 2)),
            ],
            [
                sqrt(3 * (n + 2) / (8 * (n + 3))),
                -sqrt(3 * n * (n + 2) ** 2 * (n - 2) / (8 * (n + 1) * (n - 1) * (n + 3)))
                / (n + 2),
                -sqrt(9 * n * (n + 2) / (16 * (n - 1) * (n + 3))) / 3,
                -sqrt(9 * (n + 2) ** 2 / (16 * (n + 1) * (n + 3))),
            ],
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kak-n",
        type=int,
        default=3,
        help="Concrete n at which to compute the magic-basis KAK angles.",
    )
    parser.add_argument(
        "--check-n",
        type=int,
        default=5,
        help="Concrete n at which to cross-check M(n) against a_matrix((n, 2, 1)).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    M = mn_matrix()

    print("M(n) =")
    sp.pprint(M)

    # 0. Which family is this?  Numeric cross-check against the library.
    n0 = args.check_n
    M_num = np.array(M.subs(N, n0).evalf(), dtype=float)
    A_num = a_matrix(YoungDiagram([n0, 2, 1]))
    err = np.linalg.norm(M_num - A_num)
    print(f"\n|M({n0}) - a_matrix(({n0}, 2, 1))| = {err:.3e}")

    # 1. Row and column norms.
    print("\nRow/column norms (all should be 1):")
    for i in range(4):
        print(f"  row {i} norm^2:", sp.simplify(sum(M[i, j] ** 2 for j in range(4))))
    for j in range(4):
        print(f"  col {j} norm^2:", sp.simplify(sum(M[i, j] ** 2 for i in range(4))))

    # 2. Eigenvalues at n = 3 (the staircase (3, 2, 1)).
    print("\nEigenvalues at n = 3:")
    for val, mult in M.subs(N, 3).eigenvals().items():
        print(f"  ({mult}x)", val)

    # 3. Magic-basis KAK angles at a concrete n: conjugate into the magic
    # basis and read off the phases of the (unimodular) eigenvalues.
    nk = args.kak_n
    M_magic = sp.simplify(MAGIC.inv() * M.subs(N, nk) * MAGIC)
    eigvals = list(M_magic.eigenvals())
    print(f"\nMagic-basis eigenvalues at n = {nk}:")
    for val in eigvals:
        print("  ", sp.simplify(val))
    angles = [sp.simplify(sp.arg(val)) for val in eigvals]
    print("Canonical angles (phases of the magic-basis eigenvalues):")
    for ang in angles:
        print("  ", ang, f"= {float(ang):+.6f}")

    # 4. Entrywise n -> oo limit and its orthogonality.
    M_inf = M.applyfunc(lambda entry: sp.limit(entry, N, sp.oo))
    print("\nM(n -> oo) =")
    sp.pprint(M_inf)
    residual = sp.simplify(M_inf * M_inf.T - sp.eye(4))
    print("M_inf orthogonal:", residual == sp.zeros(4, 4))


if __name__ == "__main__":
    main()
