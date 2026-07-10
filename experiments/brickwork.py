#!/usr/bin/env python3
"""Brickwork-ansatz synthesis of A-matrices via BQSKit instantiation.

Question: how many layers of Wei–Di SO(4) bricks (6 Ry + 2 CNOT each,
arranged in a 1D brickwork) does it take to reproduce an n-qubit A-matrix,
and how does the resulting CNOT count compare to the O(k²) Givens→compiler
baseline of report.md, Finding 2?

Supports report.md, Open Direction 3 (break the k² compiler barrier) by
probing a structured ansatz; the n=2 case cross-checks Finding 5 (every
4-addable A-matrix is 3 CNOT + 6 Ry, here 1 det-fix CNOT + bricks).

Expected result: every 2-qubit diagram converges at a single brick — 3
CNOT total including the det fix, matching the Wei–Di optimum — and the
n=3 staircase at 7 layers ≈ 20 CNOT, versus ≈ 42 CX for Givens→Qiskit.

All A-matrices have det = −1 while the Ry/CNOT brickwork is special
orthogonal, so the fit targets D·A with D = diag(1, …, 1, −1) and the cost
of D (1 CZ / 6 CNOT / 14 CNOT for n = 2/3/4) is added to the totals.

This supersedes both brickwork_bqskit.py and the custom-optimizer
brickwork_synth.py.  Behavior changes: the default n=2 max-size is 12 (was
20) so a bare run finishes in well under two minutes; the old cosmetic
--sweep flag is gone (the per-diagram layer sweep is the default mode); and
the single-diagram modes use the staircase diagram directly instead of
taking the first catalog entry — the same diagram, but without enumerating
all partitions up to size 120 for n = 4, which made the old --scale mode
infeasible past n = 3.

Modes:
  (default)    minimum-layer sweep over every diagram
  --layers L   fixed depth for every diagram
  --accuracy   residual vs layer count for the staircase diagram
  --scale      staircase-diagram scaling table for n = 2..4
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
from bqskit import Circuit
from bqskit.ir.gates import CNOTGate, RYGate
from bqskit.qis import UnitaryMatrix

from fourier import a_matrix, diagrams_with_addable_cells, staircase

warnings.filterwarnings("ignore", category=DeprecationWarning)

# CNOT cost of the det-correction gate D = diag(1, ..., 1, −1) per qubit count.
CORRECTION_CNOT = {2: 1, 3: 6, 4: 14, 5: 30}

DEFAULT_MAX_SIZE = {2: 12, 3: 35, 4: 125, 5: 500}
DEFAULT_MAX_LAYERS = {2: 4, 3: 12, 4: 25, 5: 60}


def correction_gate(n: int) -> np.ndarray:
    D = np.eye(2**n)
    D[-1, -1] = -1.0
    return D


def prepare_target(A: np.ndarray, n: int) -> tuple[np.ndarray, int]:
    """(SO-target = D·A, correction CNOT count); A itself if det = +1."""
    if np.linalg.det(A) < 0:
        return correction_gate(n) @ A, CORRECTION_CNOT.get(n, 6)
    return A.copy(), 0


def build_brickwork_circuit(n: int, n_layers: int) -> Circuit:
    """Wei–Di brickwork template with free Ry parameters.

    Even layers place bricks on (0,1), (2,3), …; odd layers on (1,2), (3,4), ….
    Each brick is RY⊗RY → CNOT → RY⊗RY → CNOT → RY⊗RY with all 6 Ry angles
    free and the CNOTs fixed.
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


def so_dof(n: int) -> int:
    d = 2**n
    return d * (d - 1) // 2


def fit_layers(
    target_so: np.ndarray, n: int, n_layers: int, multistarts: int, seed: int = 0
) -> tuple[float, float]:
    """Instantiate a fixed-depth brickwork; (Frobenius² residual, seconds)."""
    circuit = build_brickwork_circuit(n, n_layers)
    t0 = time.perf_counter()
    circuit.instantiate(UnitaryMatrix(target_so), multistarts=multistarts, seed=seed)
    elapsed = time.perf_counter() - t0
    diff = np.array(circuit.get_unitary()) - target_so
    return float(np.real(np.sum(diff.conj() * diff))), elapsed


def fit_binary_search(
    target_so: np.ndarray,
    n: int,
    max_layers: int,
    tol: float,
    multistarts: int,
    seed: int = 0,
) -> tuple[float, int, float]:
    """Minimum converging layer count via binary search.

    Assumes monotone convergence in the layer count, so O(log max_layers)
    fits suffice.  Returns (residual, min_layers, total_seconds); if even
    max_layers fails, returns its residual with min_layers = max_layers.
    """
    total_elapsed = 0.0

    def probe(L: int) -> tuple[float, bool]:
        nonlocal total_elapsed
        res, elapsed = fit_layers(target_so, n, L, multistarts, seed)
        total_elapsed += elapsed
        return res, res < tol

    res_hi, ok_hi = probe(max_layers)
    if not ok_hi:
        return res_hi, max_layers, total_elapsed

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
    return best_res, lo, total_elapsed


def run_scale(args) -> None:
    print("=== Brickwork (BQSKit) scaling: n qubits, staircase diagram ===\n")
    print(f"{'n':>3}  {'k=2^n':>6}  {'DOF':>6}  "
          f"{'min_L':>6}  {'bricks':>7}  {'CX_bw':>6}  "
          f"{'CX_fix':>7}  {'CX_tot':>7}  {'residual':>12}  {'t(s)':>7}")
    print("-" * 85)

    for n in range(2, 5):
        k = 2**n
        max_L = args.max_layers or DEFAULT_MAX_LAYERS[n]

        yd = staircase(k - 1)
        T, cx_fix = prepare_target(a_matrix(yd), n)
        res, min_L, elapsed = fit_binary_search(
            T, n, max_L, args.tol, args.multistarts, args.seed
        )

        n_bricks = count_bricks(n, min_L)
        cx_bw = 2 * n_bricks
        print(f"  {n:>1}  {k:>6}  {so_dof(n):>6}  "
              f"{min_L:>6}  {n_bricks:>7}  {cx_bw:>6}  "
              f"{cx_fix:>7}  {cx_bw + cx_fix:>7}  "
              f"{res:>12.2e}  {elapsed:>7.1f}  "
              f"{'✓' if res < args.tol else '✗'} ({yd.partition})")

    print()
    print("  CX_bw  = brickwork CNOT (2 per brick)")
    print("  CX_fix = det-correction gate CNOT (CZ/CCZ/CCCZ)")
    print("  CX_tot = total CNOT count for full circuit")
    print()
    print("  Comparison (Givens + Qiskit O(k²)):")
    for n in range(2, 5):
        k = 2**n
        print(f"    n={n} k={k:>3}: Qiskit Givens ≈ {int(0.75 * k * (k - 1))} CX")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--qubits", type=int, default=2)
    parser.add_argument("--layers", type=int, default=None,
                        help="fixed number of layers for every diagram")
    parser.add_argument("--accuracy", action="store_true",
                        help="residual vs n_layers for the first diagram")
    parser.add_argument("--scale", action="store_true",
                        help="scaling table n=2..4")
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--max-size", type=int, default=None)
    parser.add_argument("--multistarts", type=int, default=8,
                        help="BQSKit multistart count per fit (default 8)")
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.scale:
        run_scale(args)
        return

    n = args.qubits
    k = 2**n
    max_size = args.max_size or DEFAULT_MAX_SIZE.get(n, 20)
    max_L = args.max_layers or DEFAULT_MAX_LAYERS.get(n, 20)
    corr_cx = CORRECTION_CNOT.get(n, 6)

    print(f"=== Brickwork (BQSKit)  |  {n} qubits  {k}×{k} A-matrices ===")
    print("    Each brick:   Wei-Di SO(4)  =  2 CNOT + 6 Ry")
    print(f"    Det fix:      D = diag(1,...,-1)  =  {corr_cx} CNOT equiv.")
    print(f"    SO({k}) DOF = {so_dof(n)},  min bricks (theory) = {-(-so_dof(n) // 6)}")
    print()

    if args.accuracy:
        yd = staircase(k - 1)
        T, _ = prepare_target(a_matrix(yd), n)
        print(f"Residual vs layers  [{yd.partition}]:\n")
        print(f"  {'layers':>7}  {'bricks':>7}  {'CX_bw':>6}  "
              f"{'CX_tot':>7}  {'residual':>12}  {'t(s)':>7}")
        print("  " + "-" * 54)
        for L in range(1, max_L + 1):
            res, elapsed = fit_layers(T, n, L, args.multistarts, args.seed)
            n_bricks = count_bricks(n, L)
            cx_bw = 2 * n_bricks
            print(f"  {L:>7}  {n_bricks:>7}  {cx_bw:>6}  "
                  f"{cx_bw + corr_cx:>7}  {res:>12.2e}  {elapsed:>7.1f}"
                  f"{'  ✓' if res < args.tol else ''}")
        return

    if args.layers is not None:
        n_bricks = count_bricks(n, args.layers)
        cx_tot = 2 * n_bricks + corr_cx

        print(f"Fixed {args.layers} layers  ({n_bricks} bricks, {cx_tot} CNOT total):")
        hdr = (f"  {'Diagram':<26} {'det':>4}  {'residual':>12}  "
               f"{'CX_tot':>7}  {'ok?':>4}  {'t(s)':>7}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

        n_total = n_ok = 0
        for yd in diagrams_with_addable_cells(k, max_size):
            A = a_matrix(yd)
            T, _ = prepare_target(A, n)
            res, elapsed = fit_layers(T, n, args.layers, args.multistarts, args.seed)
            ok = res < args.tol
            print(f"  {str(yd.partition):<26} {np.linalg.det(A):>+.0f}  "
                  f"{res:>12.2e}  {cx_tot:>7}  "
                  f"{'✓' if ok else '✗'}  {elapsed:>7.1f}")
            n_total += 1
            n_ok += ok

        print("  " + "-" * (len(hdr) - 2))
        print(f"\nResult: {n_ok}/{n_total} synthesized with {cx_tot} CNOT total")
        return

    # Default mode: minimum-layer sweep over every diagram.
    hdr = (f"  {'Diagram':<26} {'det':>4}  "
           f"{'min_L':>6}  {'bricks':>7}  {'CX_bw':>6}  {'CX_tot':>7}  "
           f"{'residual':>12}  {'ok?':>4}  {'t(s)':>6}")
    print(f"Layer sweep (1..{max_L}), tol={args.tol:.0e}, max-size={max_size}:")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    n_total = n_ok = 0
    cx_list: list[int] = []
    for yd in diagrams_with_addable_cells(k, max_size):
        A = a_matrix(yd)
        T, _ = prepare_target(A, n)
        res, min_L, elapsed = fit_binary_search(
            T, n, max_L, args.tol, args.multistarts, args.seed
        )
        ok = res < args.tol
        n_bricks = count_bricks(n, min_L)
        cx_bw = 2 * n_bricks
        cx_tot = cx_bw + corr_cx

        print(f"  {str(yd.partition):<26} {np.linalg.det(A):>+.0f}  "
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
