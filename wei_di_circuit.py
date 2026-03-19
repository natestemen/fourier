#!/usr/bin/env python3
"""Wei-Di optimal circuit synthesis for 4-addable A-matrices.

Wei & Di (arXiv:1203.0722): any X ∈ O(4) can be synthesized with:
  det=+1 (SO(4)):  2 CNOT + 6 Ry  (optimal)
  det=-1 (O(4)):   3 CNOT + 6 Ry  (optimal)

Circuit template (det=+1):
  X = [Ry(θ₁)⊗Ry(θ₂)] · CNOT · [Ry(b)⊗Ry(a)] · CNOT · [Ry(θ₃)⊗Ry(θ₄)]

For det=-1: append one extra CNOT (the 3-gate version).

The so(4) Lie algebra basis used by Wei & Di:
  l = span{iI⊗σ_y, iσ_y⊗I}           ← local: generates Ry⊗Ry only
  a = span{iσ_x⊗σ_y, iσ_y⊗σ_z}       ← 2-parameter Cartan subalgebra

Key results to verify for A-matrices:
  1. det = -1 always (eigenvalues {+1,-1,e^{iθ},e^{-iθ}} → product = -1)
  2. 3-CNOT + 6-Ry circuit fits to machine precision for every diagram
  3. Interaction parameters (a,b) trace out a 2D family matching Weyl (b,c)
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy.optimize import differential_evolution, minimize

from compute_matrix import A_matrix
from helper import find_yds_with_fixed_addable_cells

# ---------------------------------------------------------------------------
# Gate primitives (all real)
# ---------------------------------------------------------------------------

def Ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]])

# CNOT: control = qubit 0 (top/first), target = qubit 1 (bottom/second)
CNOT_01 = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0]], dtype=float)

# CNOT: control = qubit 1 (bottom/second), target = qubit 0 (top/first)
CNOT_10 = np.array([[1, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0],
                    [0, 1, 0, 0]], dtype=float)

CNOTS = [CNOT_01, CNOT_10]


# ---------------------------------------------------------------------------
# Circuit construction
# ---------------------------------------------------------------------------

def build_circuit(params: np.ndarray, cnot: np.ndarray, det_fix: bool) -> np.ndarray:
    """Build 4×4 circuit matrix from Wei-Di parameters.

    params = (θ₁, θ₂, a, b, θ₃, θ₄)
    Circuit = [Ry(θ₁)⊗Ry(θ₂)] · CNOT · [Ry(b)⊗Ry(a)] · CNOT · [Ry(θ₃)⊗Ry(θ₄)] [· CNOT]
    """
    t1, t2, a, b, t3, t4 = params
    K1 = np.kron(Ry(t1), Ry(t2))
    mid = np.kron(Ry(b), Ry(a))
    K2 = np.kron(Ry(t3), Ry(t4))
    C = K1 @ cnot @ mid @ cnot @ K2
    if det_fix:
        C = C @ cnot
    return C


def _loss(params: np.ndarray, target: np.ndarray,
          cnot: np.ndarray, det_fix: bool) -> float:
    diff = target - build_circuit(params, cnot, det_fix)
    return float(np.sum(diff * diff))


# ---------------------------------------------------------------------------
# Decomposition: numerical KAK extraction
# ---------------------------------------------------------------------------

def wei_di_decompose(
    X: np.ndarray,
    n_restarts: int = 40,
    seed: int = 0,
) -> tuple[np.ndarray, float, np.ndarray, bool]:
    """Extract Wei-Di parameters from X ∈ O(4).

    Returns (params, residual, cnot_used, det_fix)
    where params = (θ₁, θ₂, a, b, θ₃, θ₄).
    """
    rng = np.random.default_rng(seed)
    det_fix = bool(np.linalg.det(X) < 0)

    best_params = None
    best_loss = np.inf
    best_cnot = CNOT_01

    bounds = [(-np.pi, np.pi)] * 6

    for cnot in CNOTS:
        # Global search first (differential evolution, cheap)
        de_result = differential_evolution(
            _loss, bounds, args=(X, cnot, det_fix),
            seed=int(rng.integers(1 << 30)),
            maxiter=300, tol=1e-12, popsize=8,
            mutation=(0.5, 1.5), recombination=0.7,
        )
        if de_result.fun < best_loss:
            best_loss = de_result.fun
            best_params = de_result.x
            best_cnot = cnot

        # Local refinements from random starts
        for _ in range(n_restarts):
            x0 = rng.uniform(-np.pi, np.pi, 6)
            res = minimize(
                _loss, x0, args=(X, cnot, det_fix),
                method="L-BFGS-B",
                options={"maxiter": 2000, "ftol": 1e-20, "gtol": 1e-12},
            )
            if res.fun < best_loss:
                best_loss = res.fun
                best_params = res.x
                best_cnot = cnot

    # Final polish from best point
    res = minimize(
        _loss, best_params, args=(X, best_cnot, det_fix),
        method="L-BFGS-B",
        options={"maxiter": 5000, "ftol": 1e-24, "gtol": 1e-14},
    )
    best_params = res.x
    best_loss = res.fun

    return best_params, best_loss, best_cnot, det_fix


# ---------------------------------------------------------------------------
# Qiskit circuit builder
# ---------------------------------------------------------------------------

def to_qiskit_circuit(params: np.ndarray, cnot: np.ndarray, det_fix: bool,
                      label: str = "A"):
    """Build a Qiskit QuantumCircuit implementing the Wei-Di decomposition."""
    from qiskit import QuantumCircuit

    t1, t2, a, b, t3, t4 = params
    qc = QuantumCircuit(2, name=label)

    # Determine CNOT direction
    # CNOT_01: control=0, target=1
    ctrl, tgt = (0, 1) if np.allclose(cnot, CNOT_01) else (1, 0)

    # Layer 1: K1
    qc.ry(t1, 0)
    qc.ry(t2, 1)
    # Interaction
    qc.cx(ctrl, tgt)
    qc.ry(b, 0)
    qc.ry(a, 1)
    qc.cx(ctrl, tgt)
    # Layer 2: K2
    qc.ry(t3, 0)
    qc.ry(t4, 1)
    # Det correction
    if det_fix:
        qc.cx(ctrl, tgt)

    return qc


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate() -> None:
    """Check the circuit builder on a random SO(4) matrix."""
    from scipy.stats import ortho_group

    print("=== Validation ===")

    # Random SO(4) (det=+1)
    rng = np.random.default_rng(7)
    X_pos = ortho_group.rvs(4, random_state=rng)
    if np.linalg.det(X_pos) < 0:
        X_pos[0] *= -1

    # Random O(4) (det=-1)
    X_neg = ortho_group.rvs(4, random_state=rng)
    if np.linalg.det(X_neg) > 0:
        X_neg[0] *= -1

    for name, X in [("SO(4) det=+1", X_pos), ("O(4) det=-1", X_neg)]:
        params, res, cnot, det_fix = wei_di_decompose(X, n_restarts=20)
        recon = build_circuit(params, cnot, det_fix)
        n_cnot = 3 if det_fix else 2
        print(f"  {name}: residual={res:.2e}  n_CNOT={n_cnot}  "
              f"recon_err={np.max(np.abs(X - recon)):.2e}  "
              f"{'OK' if res < 1e-8 else 'FAIL'}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def diagram_label(yd) -> str:
    p = yd.partition
    p = p() if callable(p) else p
    return str(tuple(p))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-size", type=int, default=20)
    parser.add_argument("--restarts", type=int, default=40)
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument("--qiskit", action="store_true",
                        help="print Qiskit circuit for each diagram")
    parser.add_argument("--tol", type=float, default=1e-6,
                        help="residual tolerance for 'success'")
    args = parser.parse_args()

    if not args.no_validate:
        validate()

    n_total = n_ok = 0
    n_cnot2 = n_cnot3 = 0

    hdr = (f"{'Diagram':<22} {'det':>4} {'residual':>10}  "
           f"{'θ₁':>7} {'θ₂':>7} {'a':>7} {'b':>7} {'θ₃':>7} {'θ₄':>7}  "
           f"{'CNOT':>4} {'ok?':>4}")
    print(f"=== Wei-Di synthesis of 4-addable A-matrices (max_size={args.max_size}) ===")
    print(hdr)
    print("-" * len(hdr))

    for yd in find_yds_with_fixed_addable_cells(4, args.max_size):
        A = A_matrix(yd)
        if A.shape[0] != 4:
            continue

        d = np.linalg.det(A)
        params, residual, cnot, det_fix = wei_di_decompose(
            A, n_restarts=args.restarts)
        ok = residual < args.tol
        n_cnot = 3 if det_fix else 2
        label = diagram_label(yd)

        print(f"{label:<22} {d:>+.0f} {residual:>10.2e}  " +
              "  ".join(f"{p:>7.4f}" for p in params) +
              f"  {n_cnot:>4} {'✓' if ok else '✗'}")

        if args.qiskit and ok:
            qc = to_qiskit_circuit(params, cnot, det_fix, label)
            print(qc.draw(output="text", fold=120))
            print()

        n_total += 1
        if ok:
            n_ok += 1
        if det_fix:
            n_cnot3 += 1
        else:
            n_cnot2 += 1

    print("-" * len(hdr))
    print(f"\nResult ({n_total} diagrams):")
    print(f"  det=+1 (2 CNOT): {n_cnot2}")
    print(f"  det=-1 (3 CNOT): {n_cnot3}")
    print(f"  Synthesis OK:    {n_ok}/{n_total}")
    if n_cnot3 == n_total:
        print("\n  All A-matrices have det=-1  →  optimal circuit = 3 CNOT + 6 Ry")
    if n_ok == n_total:
        print("  All decompositions succeeded to within tolerance.")


if __name__ == "__main__":
    main()
