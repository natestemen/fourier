#!/usr/bin/env python3
"""Compute A,B in theorem: M U M* = (A ⊗ B) SWAP (I ⊗ σz) for U ∈ O(4), det(U)=-1."""
from __future__ import annotations

import argparse
import numpy as np
import sympy as sp
from qiskit.quantum_info import Operator
from qiskit.synthesis import OneQubitEulerDecomposer

from symbolic_a_matrix import build_symbolic_a_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--w1", type=int, required=True)
    parser.add_argument("--w2", type=int, required=True)
    parser.add_argument("--w3", type=int, required=True)
    parser.add_argument("--h1", type=int, required=True)
    parser.add_argument("--h2", type=int, required=True)
    parser.add_argument("--h3", type=int, required=True)
    parser.add_argument("--tol", type=float, default=1e-6)
    return parser.parse_args()


def _magic_matrix() -> np.ndarray:
    # M = [[1, i, 0, 0], [0,0,i,1], [0,0,i,-1], [1,-i,0,0]] / sqrt(2)
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
    # SWAP on 2 qubits
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=complex,
    )


def _is_unitary(mat: np.ndarray, tol: float = 1e-6) -> bool:
    eye = np.eye(mat.shape[0], dtype=complex)
    return np.allclose(mat.conj().T @ mat, eye, atol=tol)


def _is_orthogonal(mat: np.ndarray, tol: float = 1e-6) -> bool:
    eye = np.eye(mat.shape[0], dtype=complex)
    return np.allclose(mat.T @ mat, eye, atol=tol)


def _factor_kron(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Factor X ≈ A ⊗ B via rank-1 SVD on reshaped tensor."""
    T = X.reshape(2, 2, 2, 2)
    # group indices: (i1,j1) x (i0,j0)
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
    subs = {
        symbols[0]: args.w1,
        symbols[1]: args.w2,
        symbols[2]: args.w3,
        symbols[3]: args.h1,
        symbols[4]: args.h2,
        symbols[5]: args.h3,
    }
    U = np.array(sp.Matrix(A_sym.subs(subs)).evalf(), dtype=complex)

    M = _magic_matrix()
    SWAP = _swap()
    I2 = np.eye(2, dtype=complex)
    SZ = np.array([[1, 0], [0, -1]], dtype=complex)

    print("U orthogonal:", _is_orthogonal(U, args.tol), "det(U):", np.linalg.det(U))

    K = M @ U @ M.conj().T
    X = K @ np.kron(I2, SZ) @ SWAP

    A, B = _factor_kron(X)

    # Normalize global phase so det(A), det(B) have unit magnitude
    phase = np.exp(-1j * np.angle(np.linalg.det(A)) / 2)
    A = A * phase
    B = B / phase

    print("A:")
    print(A)
    print("B:")
    print(B)
    decomposer = OneQubitEulerDecomposer("U3")
    a_params = decomposer.angles(Operator(A))
    b_params = decomposer.angles(Operator(B))
    print("A as U3(theta, phi, lam):", a_params)
    print("B as U3(theta, phi, lam):", b_params)

    # Verification
    recon = np.kron(A, B) @ SWAP @ np.kron(I2, SZ)
    err = np.max(np.abs(K - recon))
    print("max reconstruction error:", err)
    print("A unitary:", _is_unitary(A, args.tol), "B unitary:", _is_unitary(B, args.tol))


if __name__ == "__main__":
    main()
