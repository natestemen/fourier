#!/usr/bin/env python3
"""Plot U3(theta,phi,lam) parameters for A and B vs n in gun family."""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from qiskit.quantum_info import Operator
from qiskit.synthesis import OneQubitEulerDecomposer

from symbolic_a_matrix import build_symbolic_a_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-min", type=int, default=3)
    parser.add_argument("--n-max", type=int, default=100)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--output", default="data/plots/gun_u3_params.png")
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
    A = a.reshape(2, 2)
    B = b.reshape(2, 2)
    return A, B


def main() -> None:
    args = parse_args()
    A_sym, symbols = build_symbolic_a_matrix()

    M = _magic_matrix()
    SWAP = _swap()
    I2 = np.eye(2, dtype=complex)
    SZ = np.array([[1, 0], [0, -1]], dtype=complex)

    decomp = OneQubitEulerDecomposer("U3")

    ns = []
    a_t = []
    a_p = []
    a_l = []
    b_t = []
    b_p = []
    b_l = []

    for n in range(args.n_min, args.n_max + 1, args.step):
        subs = {
            symbols[0]: n,
            symbols[1]: 2,
            symbols[2]: 1,
            symbols[3]: 1,
            symbols[4]: 1,
            symbols[5]: 1,
        }
        U = np.array(sp.Matrix(A_sym.subs(subs)).evalf(), dtype=complex)
        K = M @ U @ M.conj().T
        X = K @ np.kron(I2, SZ) @ SWAP
        A, B = _factor_kron(X)

        # Normalize global phase
        phase = np.exp(-1j * np.angle(np.linalg.det(A)) / 2)
        A = A * phase
        B = B / phase

        a_theta, a_phi, a_lam = decomp.angles(Operator(A))
        b_theta, b_phi, b_lam = decomp.angles(Operator(B))

        ns.append(n)
        a_t.append(a_theta)
        a_p.append(a_phi)
        a_l.append(a_lam)
        b_t.append(b_theta)
        b_p.append(b_phi)
        b_l.append(b_lam)

    # Compute n -> infinity limit for A and B in gun family.
    n_sym = symbols[0]
    subs_inf = {
        symbols[1]: 2,
        symbols[2]: 1,
        symbols[3]: 1,
        symbols[4]: 1,
        symbols[5]: 1,
    }
    U_inf = sp.Matrix(A_sym.subs(subs_inf)).applyfunc(lambda expr: sp.limit(expr, n_sym, sp.oo))
    U_inf_num = np.array(U_inf.evalf(), dtype=complex)
    K_inf = M @ U_inf_num @ M.conj().T
    X_inf = K_inf @ np.kron(I2, SZ) @ SWAP
    A_inf, B_inf = _factor_kron(X_inf)
    phase = np.exp(-1j * np.angle(np.linalg.det(A_inf)) / 2)
    A_inf = A_inf * phase
    B_inf = B_inf / phase
    a_inf = decomp.angles(Operator(A_inf))
    b_inf = decomp.angles(Operator(B_inf))

    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    axes[0, 0].plot(ns, a_t, label="A theta")
    axes[0, 1].plot(ns, a_p, label="A phi")
    axes[0, 2].plot(ns, a_l, label="A lambda")
    axes[1, 0].plot(ns, b_t, label="B theta")
    axes[1, 1].plot(ns, b_p, label="B phi")
    axes[1, 2].plot(ns, b_l, label="B lambda")

    axes[0, 0].set_title("A theta")
    axes[0, 1].set_title("A phi")
    axes[0, 2].set_title("A lambda")
    axes[1, 0].set_title("B theta")
    axes[1, 1].set_title("B phi")
    axes[1, 2].set_title("B lambda")

    for ax in axes.ravel():
        ax.grid(True, alpha=0.3)
    all_vals = np.array(a_t + a_p + a_l + b_t + b_p + b_l, dtype=float)
    ymin = float(np.nanmin(all_vals))
    ymax = float(np.nanmax(all_vals))
    for ax in axes.ravel():
        ax.set_ylim(ymin-0.5, ymax+0.5)

    fig.suptitle("Gun family U3 parameters vs n")
    fig.tight_layout()
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {args.output}")
    print("Limit n->inf:")
    print("A_inf:\n", A_inf)
    print("B_inf:\n", B_inf)
    print("A_inf U3(theta,phi,lam):", a_inf)
    print("B_inf U3(theta,phi,lam):", b_inf)
    plt.show()


if __name__ == "__main__":
    main()
