#!/usr/bin/env python3
"""Brickwork variational synthesis of A-matrices using Wei-Di 2-qubit blocks.

Architecture: 1D brickwork on n qubits.
  Even layers: bricks on (0,1), (2,3), (4,5), ...
  Odd layers:  bricks on (1,2), (3,4), (5,6), ...

Each brick = Wei-Di real SO(4) block (det=+1):
  [Ry(θ₁)⊗Ry(θ₂)] · CNOT · [Ry(b)⊗Ry(a)] · CNOT · [Ry(θ₃)⊗Ry(θ₄)]
  = 6 Ry gates + 2 CNOT per brick.

Since embedded bricks all have det=+1, the full circuit is always in SO(2^n).
Targets with det=−1 are handled by minimising min(‖C−A‖², ‖C+A‖²).

Examples:
  python brickwork_synth.py                   # 2 qubits, layer sweep
  python brickwork_synth.py --qubits 3 --sweep --max-size 28
  python brickwork_synth.py --qubits 3 --layers 6
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy.optimize import differential_evolution, minimize

from compute_matrix import A_matrix
from helper import find_yds_with_fixed_addable_cells

# ---------------------------------------------------------------------------
# Gate primitives  (all real)
# ---------------------------------------------------------------------------

def Ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]])


CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]], dtype=float)


def wei_di_block(params: np.ndarray) -> np.ndarray:
    """Wei-Di 2-CNOT block (det=+1 SO(4)).

    params = (θ₁, θ₂, a, b, θ₃, θ₄)
    Block = [Ry(θ₁)⊗Ry(θ₂)] · CNOT · [Ry(b)⊗Ry(a)] · CNOT · [Ry(θ₃)⊗Ry(θ₄)]
    """
    t1, t2, a, b, t3, t4 = params
    K1 = np.kron(Ry(t1), Ry(t2))
    mid = np.kron(Ry(b), Ry(a))
    K2 = np.kron(Ry(t3), Ry(t4))
    return K1 @ CNOT @ mid @ CNOT @ K2


def embed_gate(U2q: np.ndarray, q: int, n: int) -> np.ndarray:
    """Embed 4×4 gate acting on qubits (q, q+1) in the full 2^n × 2^n space."""
    before = np.eye(2**q)
    after  = np.eye(2**(n - q - 2))
    return np.kron(np.kron(before, U2q), after)


# ---------------------------------------------------------------------------
# Brickwork circuit
# ---------------------------------------------------------------------------

def brickwork_schedule(n: int, n_layers: int) -> list[int]:
    """Return ordered list of first-qubit indices q for each brick (acts on q, q+1)."""
    schedule: list[int] = []
    for layer in range(n_layers):
        start = layer % 2
        for q in range(start, n - 1, 2):
            schedule.append(q)
    return schedule


def brickwork_matrix(all_params: np.ndarray, schedule: list[int], n: int) -> np.ndarray:
    """Build full 2^n × 2^n brickwork matrix from flat parameter vector."""
    dim = 2**n
    C = np.eye(dim)
    for i, q in enumerate(schedule):
        U2q = wei_di_block(all_params[6 * i : 6 * i + 6])
        C = embed_gate(U2q, q, n) @ C
    return C


def _loss(all_params: np.ndarray, target: np.ndarray,
          schedule: list[int], n: int) -> float:
    C = brickwork_matrix(all_params, schedule, n)
    # Sign-agnostic: match C to ±target
    overlap = float(np.trace(C.T @ target))
    s = 1.0 if overlap >= 0.0 else -1.0
    diff = C - s * target
    return float(np.sum(diff * diff))


# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------

def fit_brickwork(
    target: np.ndarray,
    n: int,
    n_layers: int,
    n_restarts: int = 20,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Fit brickwork parameters to target. Returns (best_params, residual)."""
    rng = np.random.default_rng(seed)
    schedule = brickwork_schedule(n, n_layers)
    n_params = 6 * len(schedule)
    bounds = [(-np.pi, np.pi)] * n_params

    # Global search (DE)
    de = differential_evolution(
        _loss, bounds, args=(target, schedule, n),
        seed=int(rng.integers(1 << 30)),
        maxiter=600, tol=1e-12, popsize=10,
        mutation=(0.5, 1.5), recombination=0.7,
    )
    best_params, best_loss = de.x, de.fun

    # Local restarts from random initial points
    for _ in range(n_restarts):
        x0 = rng.uniform(-np.pi, np.pi, n_params)
        res = minimize(
            _loss, x0, args=(target, schedule, n),
            method="L-BFGS-B",
            options={"maxiter": 2000, "ftol": 1e-20, "gtol": 1e-12},
        )
        if res.fun < best_loss:
            best_loss, best_params = res.fun, res.x

    # Final high-precision polish
    res = minimize(
        _loss, best_params, args=(target, schedule, n),
        method="L-BFGS-B",
        options={"maxiter": 5000, "ftol": 1e-26, "gtol": 1e-14},
    )
    return res.x, float(res.fun)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def diagram_label(yd) -> str:
    p = yd.partition
    p = p() if callable(p) else p
    return str(tuple(p))


def cnot_count(n: int, n_layers: int) -> int:
    return 2 * len(brickwork_schedule(n, n_layers))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--qubits", type=int, default=2,
                        help="Number of qubits: 2→4×4, 3→8×8, 4→16×16 (default: 2)")
    parser.add_argument("--layers", type=int, default=None,
                        help="Fixed number of brickwork layers")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep layers 1..max-layers for every diagram")
    parser.add_argument("--max-layers", type=int, default=8)
    parser.add_argument("--max-size", type=int, default=20)
    parser.add_argument("--restarts", type=int, default=20)
    parser.add_argument("--tol", type=float, default=1e-6,
                        help="Residual threshold for 'OK' (default: 1e-6)")
    args = parser.parse_args()

    n = args.qubits
    dim = 2**n
    k = dim       # number of addable cells = matrix dimension

    print(f"=== Brickwork synthesis  |  {n} qubits, {dim}×{dim} A-matrices ===")
    print(f"    Each brick: Wei-Di SO(4)  =  2 CNOT + 6 Ry")
    print()

    # -----------------------------------------------------------------
    # Sweep mode: try 1..max_layers layers for every diagram
    # -----------------------------------------------------------------
    if args.sweep or (args.layers is None):
        max_L = args.max_layers
        layers_range = list(range(1, max_L + 1))

        # Header: one column per layer count
        col_w = 11
        hdr_layers = "  ".join(f"L={L:>2}" for L in layers_range)
        hdr = f"{'Diagram':<22}  {hdr_layers}"
        print(f"Residuals (✓ = < {args.tol:.0e}):")
        print(hdr)
        print("-" * len(hdr))

        n_total = 0
        min_ok_layers: dict[int, int] = {}  # layer → how many diagrams OK

        for yd in find_yds_with_fixed_addable_cells(k, args.max_size):
            A = A_matrix(yd)
            if A.shape[0] != k:
                continue

            label = diagram_label(yd)
            row_parts = [f"{label:<22}"]
            first_ok = None

            for L in layers_range:
                params, residual = fit_brickwork(
                    A, n, L,
                    n_restarts=args.restarts,
                    seed=n_total,
                )
                ok = residual < args.tol
                tag = "✓" if ok else " "
                row_parts.append(f"{residual:.2e}{tag}")
                if ok and first_ok is None:
                    first_ok = L
                min_ok_layers.setdefault(L, 0)
                if ok:
                    min_ok_layers[L] += 1

            print("  ".join(row_parts))
            n_total += 1

        print("-" * len(hdr))
        print()
        print("Summary:")
        for L in layers_range:
            n_ok = min_ok_layers.get(L, 0)
            nc = cnot_count(n, L)
            print(f"  L={L}  ({nc:>3} CNOT, {len(brickwork_schedule(n,L)):>2} bricks):  "
                  f"{n_ok}/{n_total} diagrams OK")
        return

    # -----------------------------------------------------------------
    # Fixed-layers mode
    # -----------------------------------------------------------------
    L = args.layers
    schedule = brickwork_schedule(n, L)
    n_bricks = len(schedule)
    n_cx = 2 * n_bricks
    n_total = n_ok = 0

    hdr = (f"{'Diagram':<22} {'det':>4} {'residual':>12}  "
           f"{'bricks':>6} {'CNOT':>5}  {'ok?':>4}")
    print(f"=== {k}-addable A-matrices, {L}-layer brickwork "
          f"({n_cx} CNOT / {n_bricks} bricks) ===")
    print(hdr)
    print("-" * len(hdr))

    for yd in find_yds_with_fixed_addable_cells(k, args.max_size):
        A = A_matrix(yd)
        if A.shape[0] != k:
            continue

        d = np.linalg.det(A)
        params, residual = fit_brickwork(A, n, L,
                                         n_restarts=args.restarts,
                                         seed=n_total)
        ok = residual < args.tol
        label = diagram_label(yd)
        print(f"{label:<22} {d:>+.0f} {residual:>12.2e}  "
              f"{n_bricks:>6} {n_cx:>5}  {'✓' if ok else '✗'}")

        n_total += 1
        if ok:
            n_ok += 1

    print("-" * len(hdr))
    print(f"\nResult: {n_ok}/{n_total} synthesized  "
          f"({n_cx} CNOT = {n_bricks} Wei-Di bricks × 2 CNOT)")


if __name__ == "__main__":
    main()
