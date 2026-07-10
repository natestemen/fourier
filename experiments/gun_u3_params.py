#!/usr/bin/env python3
"""U3 parameters of the local gates of A(λ) along the gun family λ = (n+3, 3, 1).

The gun family is the generic 4-addable diagram with block widths (n, 2, 1)
and heights (1, 1, 1).  For each n the A-matrix is conjugated into the magic
basis, multiplied into the form K·(I⊗Z)·SWAP that is a pure tensor product
A⊗B, and the two single-qubit factors are decomposed as U3(θ, φ, λ) gates.

Question: how do the local (single-qubit) gates of the KAK decomposition vary
along a one-parameter family of diagrams, and do they converge as n → ∞?

Supports report.md Finding 1: the Weyl coordinate a = π/4 is pinned for the
whole family, so only the local gates and (b, c) move with n; this script
tracks the local side and its n → ∞ limit (computed symbolically).

Expected result: six smooth parameter curves saved to
data/plots/gun_u3_params.png, converging to the printed n → ∞ values.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from qiskit.quantum_info import Operator
from qiskit.synthesis import OneQubitEulerDecomposer

from fourier.amatrix import a_matrix_generic4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-min", type=int, default=3)
    parser.add_argument("--n-max", type=int, default=100)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument(
        "--output", type=Path, default=Path("data/plots/gun_u3_params.png")
    )
    return parser.parse_args()


def _magic_matrix() -> np.ndarray:
    M = np.array(
        [
            [1, 1j, 0, 0],
            [0, 0, 1j, 1],
            [0, 0, 1j, -1],
            [1, -1j, 0, 0],
        ],
        dtype=complex,
    )
    return M / np.sqrt(2)


def _swap() -> np.ndarray:
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=complex,
    )


def _factor_kron(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Factor a pure tensor product X = A ⊗ B via the rank-1 SVD of its
    row-major rearrangement."""
    T = X.reshape(2, 2, 2, 2)
    S = np.zeros((4, 4), dtype=complex)
    for i1 in range(2):
        for i0 in range(2):
            for j1 in range(2):
                for j0 in range(2):
                    S[i1 * 2 + j1, i0 * 2 + j0] = T[i1, i0, j1, j0]
    U, s, Vh = np.linalg.svd(S)
    a = U[:, 0] * np.sqrt(s[0])
    b = np.conj(Vh[0, :]) * np.sqrt(s[0])
    return a.reshape(2, 2), b.reshape(2, 2)


def _local_factors(
    U: np.ndarray, M: np.ndarray, SWAP: np.ndarray, IZ: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The det-normalized single-qubit factors A, B of M·U·M† · (I⊗Z) · SWAP."""
    K = M @ U @ M.conj().T
    A, B = _factor_kron(K @ IZ @ SWAP)
    phase = np.exp(-1j * np.angle(np.linalg.det(A)) / 2)
    return A * phase, B / phase


def main() -> None:
    args = parse_args()

    A_sym, symbols = a_matrix_generic4()
    w1, w2, w3, h1, h2, h3 = symbols
    gun_shape = {w2: 2, w3: 1, h1: 1, h2: 1, h3: 1}
    A_gun = A_sym.subs(gun_shape)  # 4×4 in the single symbol w1 = n

    M = _magic_matrix()
    SWAP = _swap()
    IZ = np.kron(np.eye(2, dtype=complex), np.diag([1.0 + 0j, -1.0]))
    decomp = OneQubitEulerDecomposer("U3")

    ns = list(range(args.n_min, args.n_max + 1, args.step))
    a_params, b_params = [], []
    for n in ns:
        U = np.array(A_gun.subs(w1, n).evalf(), dtype=complex)
        A, B = _local_factors(U, M, SWAP, IZ)
        a_params.append(decomp.angles(Operator(A)))
        b_params.append(decomp.angles(Operator(B)))

    a_t, a_p, a_l = map(list, zip(*a_params))
    b_t, b_p, b_l = map(list, zip(*b_params))

    # Exact n → ∞ limit of the family, then the same local factorization.
    U_inf_sym = A_gun.applyfunc(lambda expr: sp.limit(expr, w1, sp.oo))
    U_inf = np.array(U_inf_sym.evalf(), dtype=complex)
    A_inf, B_inf = _local_factors(U_inf, M, SWAP, IZ)
    a_inf = decomp.angles(Operator(A_inf))
    b_inf = decomp.angles(Operator(B_inf))

    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    series = [
        (axes[0, 0], a_t, "A theta"),
        (axes[0, 1], a_p, "A phi"),
        (axes[0, 2], a_l, "A lambda"),
        (axes[1, 0], b_t, "B theta"),
        (axes[1, 1], b_p, "B phi"),
        (axes[1, 2], b_l, "B lambda"),
    ]
    for ax, values, title in series:
        ax.plot(ns, values)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    all_vals = np.array(a_t + a_p + a_l + b_t + b_p + b_l, dtype=float)
    ymin, ymax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    for ax in axes.ravel():
        ax.set_ylim(ymin - 0.5, ymax + 0.5)

    fig.suptitle("Gun family U3 parameters vs n")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {args.output}")

    print("Limit n->inf:")
    print("A_inf:\n", A_inf)
    print("B_inf:\n", B_inf)
    print("A_inf U3(theta,phi,lam):", a_inf)
    print("B_inf U3(theta,phi,lam):", b_inf)


if __name__ == "__main__":
    main()
