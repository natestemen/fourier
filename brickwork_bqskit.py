#!/usr/bin/env python3
"""Brickwork synthesis of A-matrices using BQSKit Circuit.instantiate.

Uses the same Wei-Di 1D brickwork architecture as brickwork_synth.py, but
delegates parameter optimization to BQSKit's built-in Circuit.instantiate
(QFactor by default — much faster than our custom L-BFGS-B).

Architecture: alternating layers of Wei-Di SO(4) bricks.
  Even layers: bricks on (0,1), (2,3), ...
  Odd  layers: bricks on (1,2), (3,4), ...
  Each brick: RY⊗RY → CNOT → RY⊗RY → CNOT → RY⊗RY  (6 Ry + 2 CNOT)

All A-matrices have det=−1.  Fix: prepend D=diag(1,...,−1) and fit D·A.

Examples:
  python brickwork_bqskit.py                          # n=2, sweep
  python brickwork_bqskit.py --qubits 3 --sweep       # all k=8 diagrams
  python brickwork_bqskit.py --qubits 3 --layers 7    # fixed depth
  python brickwork_bqskit.py --qubits 3 --accuracy    # residual vs layers
  python brickwork_bqskit.py --scale                  # scaling table n=2..4
"""
from __future__ import annotations

import argparse
import warnings
import time

import numpy as np
from bqskit import Circuit
from bqskit.ir.gates import RYGate, CNOTGate
from bqskit.qis import UnitaryMatrix

from compute_matrix import A_matrix
from helper import find_yds_with_fixed_addable_cells

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------------------------------------------------------------------
# Det-correction gate
# ---------------------------------------------------------------------------

CORRECTION_CNOT = {2: 1, 3: 6, 4: 14, 5: 30}


def correction_gate(n: int) -> np.ndarray:
    D = np.eye(2**n)
    D[-1, -1] = -1.0
    return D


def prepare_target(A: np.ndarray, n: int) -> tuple[np.ndarray, int]:
    """Return (SO-target = D·A, correction_cnot_count)."""
    if np.linalg.det(A) < 0:
        return correction_gate(n) @ A, CORRECTION_CNOT.get(n, 6)
    return A.copy(), 0


# ---------------------------------------------------------------------------
# BQSKit circuit template
# ---------------------------------------------------------------------------

def build_brickwork_circuit(n: int, n_layers: int) -> Circuit:
    """Build Wei-Di brickwork template with free Ry parameters.

    Gate order per brick (time order = left to right in circuit):
      RY(q) ⊗ RY(q+1)  →  CNOT(q→q+1)  →  RY(q) ⊗ RY(q+1)
                        →  CNOT(q→q+1)  →  RY(q) ⊗ RY(q+1)
    CNOT gates are fixed; all 6 RY angles per brick are free parameters.
    """
    circuit = Circuit(n)
    for layer in range(n_layers):
        for q in range(layer % 2, n - 1, 2):
            circuit.append_gate(RYGate(), [q])
            circuit.append_gate(RYGate(), [q + 1])
            circuit.append_gate(CNOTGate(), [q, q + 1])
            circuit.append_gate(RYGate(), [q])
            circuit.append_gate(RYGate(), [q + 1])
            circuit.append_gate(CNOTGate(), [q, q + 1])
            circuit.append_gate(RYGate(), [q])
            circuit.append_gate(RYGate(), [q + 1])
    return circuit


def count_bricks(n: int, n_layers: int) -> int:
    return sum(len(range(layer % 2, n - 1, 2)) for layer in range(n_layers))


def frobenius_residual(circuit: Circuit, target: np.ndarray) -> float:
    """||U_circuit − target||_F²."""
    U = np.array(circuit.get_unitary())
    diff = U - target
    return float(np.real(np.sum(diff.conj() * diff)))


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_layers(
    target_so: np.ndarray,
    n: int,
    n_layers: int,
    multistarts: int,
    seed: int = 0,
) -> tuple[float, float]:
    """Instantiate a fixed-depth brickwork to target_so.

    Returns (residual, elapsed_s).
    """
    T = UnitaryMatrix(target_so)
    circuit = build_brickwork_circuit(n, n_layers)
    t0 = time.perf_counter()
    circuit.instantiate(T, multistarts=multistarts, seed=seed)
    elapsed = time.perf_counter() - t0
    return frobenius_residual(circuit, target_so), elapsed


def fit_binary_search(
    target_so: np.ndarray,
    n: int,
    max_layers: int,
    tol: float,
    multistarts: int,
    seed: int = 0,
) -> tuple[float, int, float]:
    """Find minimum layers via binary search.

    Assumes convergence is monotone: if L layers converge, L+1 also converge.
    Uses O(log max_layers) fits instead of O(max_layers).

    Returns (residual, min_L, total_elapsed).
    If max_layers does not converge, returns (residual, max_layers, elapsed).
    """
    total_elapsed = 0.0

    def probe(L: int) -> tuple[float, bool]:
        nonlocal total_elapsed
        res, elapsed = fit_layers(target_so, n, L, multistarts, seed)
        total_elapsed += elapsed
        return res, res < tol

    # Verify that max_layers converges; if not, report failure immediately.
    res_hi, ok_hi = probe(max_layers)
    if not ok_hi:
        return res_hi, max_layers, total_elapsed

    # Binary search for the smallest converging L in [1, max_layers].
    # Invariant: hi always converges, lo-1 never converges (or lo=1).
    lo, hi = 1, max_layers
    best_res = res_hi

    while lo < hi:
        mid = (lo + hi) // 2
        res_mid, ok_mid = probe(mid)
        if ok_mid:
            hi = mid
            best_res = res_mid
        else:
            lo = mid + 1

    # lo == hi is the minimum layer count that converged.
    return best_res, lo, total_elapsed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def diagram_label(yd) -> str:
    p = yd.partition
    p = p() if callable(p) else p
    return str(tuple(p))


def so_dof(n: int) -> int:
    d = 2**n
    return d * (d - 1) // 2


DEFAULT_MAX_SIZE   = {2: 20, 3: 35, 4: 125, 5: 500}
DEFAULT_MAX_LAYERS = {2: 4,  3: 12, 4: 25,  5: 60}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--qubits",      type=int,   default=2)
    parser.add_argument("--sweep",       action="store_true",
                        help="Find minimum layers per diagram")
    parser.add_argument("--layers",      type=int,   default=None,
                        help="Fixed number of layers")
    parser.add_argument("--accuracy",    action="store_true",
                        help="Residual vs n_layers for the first diagram")
    parser.add_argument("--scale",       action="store_true",
                        help="Scaling table n=2..4")
    parser.add_argument("--max-layers",  type=int,   default=None)
    parser.add_argument("--max-size",    type=int,   default=None)
    parser.add_argument("--multistarts", type=int,   default=8,
                        help="BQSKit multistart count per layer (default 8)")
    parser.add_argument("--tol",         type=float, default=1e-6)
    parser.add_argument("--seed",        type=int,   default=0)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Scaling table
    # ------------------------------------------------------------------
    if args.scale:
        print("=== Brickwork (BQSKit) scaling: n qubits, first diagram ===")
        print()
        print(f"{'n':>3}  {'k=2^n':>6}  {'DOF':>6}  "
              f"{'min_L':>6}  {'bricks':>7}  {'CX_bw':>6}  "
              f"{'CX_fix':>7}  {'CX_tot':>7}  {'residual':>12}  {'t(s)':>7}")
        print("-" * 85)

        for n in range(2, 5):
            k        = 2**n
            dof      = so_dof(n)
            max_size = args.max_size or DEFAULT_MAX_SIZE.get(n, 20)
            max_L    = args.max_layers or DEFAULT_MAX_LAYERS.get(n, 20)
            corr_cx  = CORRECTION_CNOT.get(n, 6)

            yd = next(
                (y for y in find_yds_with_fixed_addable_cells(k, max_size)
                 if A_matrix(y).shape[0] == k),
                None,
            )
            if yd is None:
                print(f"  n={n}: no diagram found (max-size={max_size})")
                continue

            A = A_matrix(yd)
            T, cx_fix = prepare_target(A, n)
            label = diagram_label(yd)

            t0 = time.perf_counter()
            res, min_L, _ = fit_binary_search(T, n, max_L, args.tol,
                                      args.multistarts, args.seed)
            elapsed = time.perf_counter() - t0

            n_bricks = count_bricks(n, min_L)
            cx_bw    = 2 * n_bricks
            cx_tot   = cx_bw + cx_fix
            ok       = res < args.tol

            print(f"  {n:>1}  {k:>6}  {dof:>6}  "
                  f"{min_L:>6}  {n_bricks:>7}  {cx_bw:>6}  "
                  f"{cx_fix:>7}  {cx_tot:>7}  "
                  f"{res:>12.2e}  {elapsed:>7.1f}  "
                  f"{'✓' if ok else '✗'} ({label})")

        print()
        print("  CX_bw  = brickwork CNOT (2 per brick)")
        print("  CX_fix = det-correction gate CNOT (CZ/CCZ/CCCZ)")
        print("  CX_tot = total CNOT count for full circuit")
        print()
        print("  Comparison (Givens + Qiskit O(k²)):")
        for n in range(2, 5):
            k = 2**n
            givens_cx = int(0.75 * k * (k - 1))
            print(f"    n={n} k={k:>3}: Qiskit Givens ≈ {givens_cx} CX")
        return

    # ------------------------------------------------------------------
    # Per-qubit modes
    # ------------------------------------------------------------------
    n        = args.qubits
    k        = 2**n
    max_size = args.max_size or DEFAULT_MAX_SIZE.get(n, 20)
    max_L    = args.max_layers or DEFAULT_MAX_LAYERS.get(n, 20)
    corr_cx  = CORRECTION_CNOT.get(n, 6)

    print(f"=== Brickwork (BQSKit)  |  {n} qubits  {k}×{k} A-matrices ===")
    print("    Each brick:   Wei-Di SO(4)  =  2 CNOT + 6 Ry")
    print(f"    Det fix:      D = diag(1,...,-1)  =  {corr_cx} CNOT equiv.")
    print(f"    SO({k}) DOF = {so_dof(n)},  min bricks (theory) = {-(-so_dof(n)//6)}")
    print()

    # ------------------------------------------------------------------
    # Accuracy vs layers (single diagram)
    # ------------------------------------------------------------------
    if args.accuracy:
        yd = next(
            (y for y in find_yds_with_fixed_addable_cells(k, max_size)
             if A_matrix(y).shape[0] == k),
            None,
        )
        if yd is None:
            print(f"  No diagram found (max-size={max_size})")
            return

        A = A_matrix(yd)
        T, _ = prepare_target(A, n)
        label = diagram_label(yd)

        print(f"Residual vs layers  [{label}]:")
        print()
        print(f"  {'layers':>7}  {'bricks':>7}  {'CX_bw':>6}  "
              f"{'CX_tot':>7}  {'residual':>12}  {'t(s)':>7}")
        print("  " + "-" * 54)

        for L in range(1, max_L + 1):
            res, elapsed = fit_layers(T, n, L, args.multistarts, args.seed)
            n_bricks = count_bricks(n, L)
            cx_bw    = 2 * n_bricks
            cx_tot   = cx_bw + corr_cx
            ok       = res < args.tol
            print(f"  {L:>7}  {n_bricks:>7}  {cx_bw:>6}  "
                  f"{cx_tot:>7}  {res:>12.2e}  {elapsed:>7.1f}"
                  f"{'  ✓' if ok else ''}")
        return

    # ------------------------------------------------------------------
    # Fixed layers
    # ------------------------------------------------------------------
    if args.layers is not None:
        fixed_L  = args.layers
        n_bricks = count_bricks(n, fixed_L)
        cx_bw    = 2 * n_bricks
        cx_tot   = cx_bw + corr_cx

        print(f"Fixed {fixed_L} layers  ({n_bricks} bricks, {cx_tot} CNOT total):")
        hdr = (f"  {'Diagram':<26} {'det':>4}  {'residual':>12}  "
               f"{'CX_tot':>7}  {'ok?':>4}  {'t(s)':>7}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

        n_total = n_ok = 0
        for yd in find_yds_with_fixed_addable_cells(k, max_size):
            A = A_matrix(yd)
            if A.shape[0] != k:
                continue
            T, _ = prepare_target(A, n)
            res, elapsed = fit_layers(T, n, fixed_L, args.multistarts, args.seed)
            ok = res < args.tol
            print(f"  {diagram_label(yd):<26} {np.linalg.det(A):>+.0f}  "
                  f"{res:>12.2e}  {cx_tot:>7}  "
                  f"{'✓' if ok else '✗'}  {elapsed:>7.1f}")
            n_total += 1
            n_ok += ok

        print("  " + "-" * (len(hdr) - 2))
        print(f"\nResult: {n_ok}/{n_total} synthesized with {cx_tot} CNOT total")
        return

    # ------------------------------------------------------------------
    # Sweep (default)
    # ------------------------------------------------------------------
    hdr = (f"  {'Diagram':<26} {'det':>4}  "
           f"{'min_L':>6}  {'bricks':>7}  {'CX_bw':>6}  {'CX_tot':>7}  "
           f"{'residual':>12}  {'ok?':>4}  {'t(s)':>6}")
    print(f"Layer sweep (1..{max_L}), tol={args.tol:.0e}, max-size={max_size}:")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    n_total = n_ok = 0
    cx_list: list[int] = []

    for yd in find_yds_with_fixed_addable_cells(k, max_size):
        A = A_matrix(yd)
        if A.shape[0] != k:
            continue
        T, _ = prepare_target(A, n)
        res, min_L, elapsed = fit_binary_search(T, n, max_L, args.tol,
                                        args.multistarts, args.seed)
        ok = res < args.tol
        n_bricks = count_bricks(n, min_L)
        cx_bw    = 2 * n_bricks
        cx_tot   = cx_bw + corr_cx

        print(f"  {diagram_label(yd):<26} {np.linalg.det(A):>+.0f}  "
              f"{min_L:>6}  {n_bricks:>7}  {cx_bw:>6}  {cx_tot:>7}  "
              f"{res:>12.2e}  {'✓' if ok else '✗'}  {elapsed:>6.1f}")

        n_total += 1
        if ok:
            n_ok += 1
            cx_list.append(cx_tot)

    print("  " + "-" * (len(hdr) - 2))
    print(f"\nResult ({n_total} diagrams, {n_ok} converged):")
    if cx_list:
        print(f"  CNOT total: min={min(cx_list)}  max={max(cx_list)}  "
              f"mean={np.mean(cx_list):.1f}")
        print(f"  (brickwork + {corr_cx}-CNOT correction gate)")


if __name__ == "__main__":
    main()
