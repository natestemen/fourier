#!/usr/bin/env python3
"""Test whether A-matrices are leaky entanglers (Peterson, Crooks & Smith 2019).

U ∈ SU(4) is leaky if there exists a nonzero h ∈ su(2) such that
    U · (h⊗I₂) · U†  ∈  k  =  su(2)⊗I₂ + I₂⊗su(2)   (local Lie algebra).

Infinitesimally, this says U can "absorb" a single-qubit rotation on the first
qubit by conjugating it into a pair of single-qubit rotations (one per qubit).

Algorithm:
----------
k⊥ (nonlocal part p) = span{ i·σₐ⊗σ_b : a,b ∈ {x,y,z} }  (9-dimensional).
M ∈ k  iff  Tr(M · P) = 0  for all P ∈ p.

Writing  A = Σ αₖ eₖ  (eₖ = iσₖ, the su(2) basis) and  Mₖ = U(eₖ⊗I₂)U†,
the condition becomes the 9×3 complex (→ 18×3 real) linear system

    L[ab, k] = Tr(Mₖ · (i·σₐ⊗σ_b)) = 0

U is leaky iff rank(L_real) < 3  (null space contains a nonzero real vector).

We test both "left" (h⊗I) and "right" (I⊗h) directions.

Validation: I⊗I → rank 0 (always leaky); CZ → rank 2 (leaky, 1D null space);
SWAP → rank 0 (leaky, fully local up to swap); generic gate → rank 3 (not leaky).
"""
from __future__ import annotations

import argparse

import numpy as np
from qiskit.synthesis import TwoQubitWeylDecomposition

from compute_matrix import A_matrix
from helper import find_yds_with_fixed_addable_cells

# ---------------------------------------------------------------------------
# Pauli matrices and su(2) basis
# ---------------------------------------------------------------------------
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
PAULIS = [SX, SY, SZ]

SU2_BASIS = [1j * p for p in PAULIS]          # {iσ_x, iσ_y, iσ_z}: anti-Hermitian traceless basis
NONLOCAL  = [np.kron(a, b) for a in PAULIS    # 9 σₐ⊗σ_b spanning the nonlocal part p ⊂ su(4)
                            for b in PAULIS]


# ---------------------------------------------------------------------------
# Core routine
# ---------------------------------------------------------------------------

def leakiness_rank(U: np.ndarray, direction: str = "left") -> tuple[int, float]:
    """Return (rank of L_real, smallest singular value) for the leakiness system.

    rank < 3  →  U is leaky in the given direction.

    direction: "left"  checks h⊗I₂  (first qubit)
               "right" checks I₂⊗h  (second qubit)
    """
    L = np.zeros((9, 3), dtype=complex)
    I2 = np.eye(2, dtype=complex)
    for k, ek in enumerate(SU2_BASIS):
        X  = np.kron(ek, I2) if direction == "left" else np.kron(I2, ek)
        Mk = U @ X @ U.conj().T
        for j, Gab in enumerate(NONLOCAL):
            # Condition: Tr(Mₖ · i·Gab) = 0  for all a,b
            L[j, k] = np.trace(Mk @ (1j * Gab))

    L_real = np.vstack([L.real, L.imag])   # 18×3 real system
    sv     = np.linalg.svd(L_real, compute_uv=False)
    tol    = 1e-8 * float(sv[0]) if sv[0] > 0 else 1e-8
    rank   = int(np.sum(sv > tol))
    return rank, float(sv[-1])


def null_direction(U: np.ndarray, direction: str = "left") -> np.ndarray | None:
    """Return the null vector (α₁, α₂, α₃) if leaky, else None.

    The returned vector gives  h = α₁·iσ_x + α₂·iσ_y + α₃·iσ_z  as the
    leaky su(2) direction.
    """
    L = np.zeros((9, 3), dtype=complex)
    I2 = np.eye(2, dtype=complex)
    for k, ek in enumerate(SU2_BASIS):
        X  = np.kron(ek, I2) if direction == "left" else np.kron(I2, ek)
        Mk = U @ X @ U.conj().T
        for j, Gab in enumerate(NONLOCAL):
            L[j, k] = np.trace(Mk @ (1j * Gab))

    L_real = np.vstack([L.real, L.imag])
    _, sv, Vt = np.linalg.svd(L_real)
    if sv[-1] < 1e-8 * sv[0]:
        return Vt[-1]    # last right-singular vector ≈ null vector
    return None


# ---------------------------------------------------------------------------
# Validation on known gates
# ---------------------------------------------------------------------------

def validate() -> None:
    """Check known leaky / non-leaky gates."""
    # Random SU(4) gate (should be non-leaky with probability 1)
    rng = np.random.default_rng(42)
    Q, _ = np.linalg.qr(rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4)))
    Q   /= np.linalg.det(Q) ** 0.25

    gates: dict[str, np.ndarray] = {
        "I⊗I  ": np.eye(4, dtype=complex),
        "CZ   ": np.diag([1.0, 1.0, 1.0, -1.0]).astype(complex),
        "SWAP ": np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex),
        "iSWAP": np.array([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]], dtype=complex),
        "rnd  ": Q,
    }

    print("=== Validation on known gates ===")
    print(f"{'gate':<10} {'left rank':>10} {'min sv (L)':>12}   {'right rank':>10} {'min sv (R)':>12}")
    print("-" * 60)
    for name, U in gates.items():
        r_l, s_l = leakiness_rank(U, "left")
        r_r, s_r = leakiness_rank(U, "right")
        leaky_l = "leaky" if r_l < 3 else "non-leaky"
        leaky_r = "leaky" if r_r < 3 else "non-leaky"
        print(f"  {name}   {r_l} ({leaky_l:<9}) {s_l:>12.2e}   {r_r} ({leaky_r:<9}) {s_r:>12.2e}")
    print()

    # Explicitly identify the leaky direction for CZ
    v = null_direction(gates["CZ   "], "left")
    if v is not None:
        names = ["iσ_x", "iσ_y", "iσ_z"]
        dominant = names[int(np.argmax(np.abs(v)))]
        print(f"  CZ leaky direction: α = {np.round(v, 4)}  →  dominant: {dominant}")
        print(f"  (Expected: iσ_z, since [CZ, Rz⊗I] = 0 up to local phases)")
    print()


# ---------------------------------------------------------------------------
# Main: run on A-matrices
# ---------------------------------------------------------------------------

def diagram_label(yd) -> str:
    p = yd.partition
    p = p() if callable(p) else p
    return str(tuple(p))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--addable", type=int, default=4,
                        help="Number of addable cells (default: 4 = 2-qubit)")
    parser.add_argument("--max-size", type=int, default=40)
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()

    if not args.no_validate:
        validate()

    n_leaky_l = n_leaky_r = n_leaky_either = n_total = 0

    hdr = f"{'Diagram':<22} {'a':>7} {'b':>7} {'c':>7}  {'L-rnk':>5} {'R-rnk':>5}  {'leaky?':<8}  {'min_sv_L':>10} {'min_sv_R':>10}"
    print(f"=== {args.addable}-addable A-matrices (max_size={args.max_size}) ===")
    print(hdr)
    print("-" * len(hdr))

    for yd in find_yds_with_fixed_addable_cells(args.addable, args.max_size):
        A = A_matrix(yd)
        if A.shape[0] != args.addable:
            continue

        U = A.astype(complex)

        r_l, s_l = leakiness_rank(U, "left")
        r_r, s_r = leakiness_rank(U, "right")
        leaky_l = r_l < 3
        leaky_r = r_r < 3
        leaky   = leaky_l or leaky_r

        try:
            # Adjust to SU(4) for Weyl decomposition (det must be +1)
            d = np.linalg.det(U)
            Usu4 = U / (d ** 0.25)
            weyl = TwoQubitWeylDecomposition(Usu4)
            a, b, c = weyl.a, weyl.b, weyl.c
        except Exception:
            a = b = c = float("nan")

        label = diagram_label(yd)
        leaky_str = ("L+R" if (leaky_l and leaky_r) else
                     "L  " if leaky_l else
                     "  R" if leaky_r else
                     "no ")
        print(f"{label:<22} {a:>7.4f} {b:>7.4f} {c:>7.4f}  "
              f"{r_l:>5} {r_r:>5}  {leaky_str:<8}  {s_l:>10.2e} {s_r:>10.2e}")

        n_total     += 1
        if leaky_l:     n_leaky_l     += 1
        if leaky_r:     n_leaky_r     += 1
        if leaky:       n_leaky_either += 1

    print("-" * len(hdr))
    print(f"\nResult ({n_total} diagrams):")
    print(f"  leaky (left  h⊗I):  {n_leaky_l}/{n_total}")
    print(f"  leaky (right I⊗h):  {n_leaky_r}/{n_total}")
    print(f"  leaky (either):     {n_leaky_either}/{n_total}")
    if n_leaky_either == n_total:
        print("\n  ALL A-matrices are leaky — consistent with the a=π/4 hypothesis.")
    elif n_leaky_either == 0:
        print("\n  NO A-matrices are leaky — a=π/4 does not imply leakiness here.")
    else:
        print("\n  Mixed result — leakiness does not hold universally for this family.")


if __name__ == "__main__":
    main()
