#!/usr/bin/env python3
"""Brickwork variational synthesis of A-matrices using Wei-Di 2-qubit blocks.

Architecture: 1D brickwork on n qubits.
  Even layers: bricks on (0,1), (2,3), (4,5), ...
  Odd layers:  bricks on (1,2), (3,4), (5,6), ...

Each brick = Wei-Di real SO(4) block (det=+1):
  [Ry(θ₁)⊗Ry(θ₂)] · CNOT · [Ry(b)⊗Ry(a)] · CNOT · [Ry(θ₃)⊗Ry(θ₄)]
  = 6 Ry gates + 2 CNOT per brick.

All A-matrices have det=−1 (even dimensions).  Embedded 2-qubit gates always
have det = det(U_2q)^{2^{n-2}} = (±1)^{even} = +1 for n ≥ 3, so the brickwork
alone lives in SO(2^n) and cannot reach det=−1 targets directly.

Fix: prepend a single "det-correction" gate D = diag(1,...,1,−1) (det=−1).
  n=2: D = CZ  (4×4)   — 1 CNOT equiv.
  n=3: D = CCZ (8×8)   — 6 CNOT equiv.
  n=4: D = CCCZ(16×16) — ~14 CNOT equiv.

The brickwork fits SO-target T = D·A (det=+1). Full circuit = D · brickwork.

Note on CZ: adding CZ gates to the brickwork does NOT help for n≥3, because
  each embedded CZ has det = det(CZ_4x4)^{2^{n-2}} = (+1) for n≥3.

Optimization:
  Incremental warm-start: grow the circuit layer by layer.  The L-layer
  solution initialises the first L bricks for the (L+1)-layer run; new bricks
  are seeded near identity (params≈0).  This avoids cold-start global search
  at every depth and makes n=4+ tractable.

Examples:
  python brickwork_synth.py                          # n=2, 1 layer
  python brickwork_synth.py --qubits 3 --sweep       # find min layers per diagram
  python brickwork_synth.py --qubits 3 --layers 7    # all k=8 diagrams, 7 layers
  python brickwork_synth.py --scale                  # scaling table n=2..4
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.optimize import minimize

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
    """Wei-Di 2-CNOT SO(4) block.  params = (θ₁, θ₂, a, b, θ₃, θ₄)."""
    t1, t2, a, b, t3, t4 = params
    K1  = np.kron(Ry(t1), Ry(t2))
    mid = np.kron(Ry(b),  Ry(a))
    K2  = np.kron(Ry(t3), Ry(t4))
    return K1 @ CNOT @ mid @ CNOT @ K2


def embed_gate(U2q: np.ndarray, q: int, n: int) -> np.ndarray:
    """Embed 4×4 gate on qubits (q, q+1) in 2^n × 2^n space."""
    return np.kron(np.kron(np.eye(2**q), U2q), np.eye(2**(n - q - 2)))


# ---------------------------------------------------------------------------
# Determinant correction gate
#
# D = diag(1,...,1,−1) has det=−1 for any dimension.  Prepend to brickwork
# to handle det=−1 A-matrix targets.  Brickwork then fits T = D·A ∈ SO(2^n).
# ---------------------------------------------------------------------------

def correction_gate(n: int) -> np.ndarray:
    D = np.eye(2**n)
    D[-1, -1] = -1.0
    return D


# Approximate CNOT cost to implement D = diag(1,...,1,−1)
# (Barenco et al. n-qubit Toffoli decomposition)
CORRECTION_CNOT = {2: 1, 3: 6, 4: 14, 5: 30}


# ---------------------------------------------------------------------------
# Brickwork circuit
# ---------------------------------------------------------------------------

def brickwork_schedule(n: int, n_layers: int) -> list[int]:
    """Ordered list of first-qubit index q for each brick (acts on q, q+1)."""
    sched: list[int] = []
    for layer in range(n_layers):
        for q in range(layer % 2, n - 1, 2):
            sched.append(q)
    return sched


def brickwork_matrix(all_params: np.ndarray, schedule: list[int], n: int) -> np.ndarray:
    C = np.eye(2**n)
    for i, q in enumerate(schedule):
        C = embed_gate(wei_di_block(all_params[6*i : 6*i+6]), q, n) @ C
    return C


def _dRy(theta: float) -> np.ndarray:
    """Derivative of Ry(theta) with respect to theta."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return 0.5 * np.array([[-s, -c], [c, -s]])


def _loss_and_grad(all_params: np.ndarray, target: np.ndarray,
                   schedule: list[int], n: int) -> tuple[float, np.ndarray]:
    """Compute loss and exact gradient via backpropagation through the brickwork.

    Forward pass stores prefix products; backward pass accumulates ∂loss/∂params
    analytically — no finite differences, ~6x fewer matrix ops vs scipy default.
    """
    B    = len(schedule)
    dim  = 2**n

    # --- Forward pass: build each brick matrix and all prefix products ---
    bricks: list[np.ndarray] = []
    for i, q in enumerate(schedule):
        bricks.append(embed_gate(wei_di_block(all_params[6*i : 6*i+6]), q, n))

    # prefix[j] = M_j @ ... @ M_0 @ I   (prefix[0] = I, prefix[j+1] = M_j @ prefix[j])
    prefix = [np.eye(dim)]
    for M in bricks:
        prefix.append(M @ prefix[-1])

    C    = prefix[-1]
    diff = C - target                           # residual
    loss = float(np.sum(diff * diff))

    # --- Backward pass ---
    #
    # ∂loss/∂θₖʲ = 2·tr(diffᵀ · suffix_j · ∂M_j · prefix[j])
    #            = 2·np.sum(diff * (suffix_j @ ∂M_j @ prefix[j]).T)
    #
    # where suffix_j = M_{B-1} @ ... @ M_{j+1}  (suffix_{B-1} = I)
    # and suffix_{j-1} = suffix_j @ bricks[j]   (right-multiply to step back)
    grad   = np.zeros_like(all_params)
    suffix = np.eye(dim)                        # suffix_{B-1} = I

    for j in range(B - 1, -1, -1):
        q  = schedule[j]
        p6 = all_params[6*j : 6*j+6]
        t1, t2, a, b, t3, t4 = p6

        R1, R2, Ra, Rb, R3, R4 = Ry(t1), Ry(t2), Ry(a), Ry(b), Ry(t3), Ry(t4)
        dR1, dR2 = _dRy(t1), _dRy(t2)
        dRa, dRb = _dRy(a),  _dRy(b)
        dR3, dR4 = _dRy(t3), _dRy(t4)

        K1      = np.kron(R1, R2)
        dK1_dt1 = np.kron(dR1, R2)
        dK1_dt2 = np.kron(R1, dR2)
        mid     = np.kron(Rb, Ra)
        dmid_da = np.kron(Rb, dRa)
        dmid_db = np.kron(dRb, Ra)
        K2      = np.kron(R3, R4)
        dK2_dt3 = np.kron(dR3, R4)
        dK2_dt4 = np.kron(R3, dR4)

        # ∂loss/∂p_k = 2·tr(diffᵀ @ suffix @ ∂M @ prefix[j])
        #            = 2·np.sum(diff * (suffix @ ∂M @ prefix[j]))
        #   since np.sum(A * B) = tr(Aᵀ @ B)
        def _g(d_inner: np.ndarray) -> float:
            d_emb = embed_gate(d_inner, q, n)
            return 2.0 * float(np.sum(diff * (suffix @ d_emb @ prefix[j])))

        grad[6*j + 0] = _g(dK1_dt1 @ CNOT @ mid @ CNOT @ K2)
        grad[6*j + 1] = _g(dK1_dt2 @ CNOT @ mid @ CNOT @ K2)
        grad[6*j + 2] = _g(K1 @ CNOT @ dmid_da @ CNOT @ K2)
        grad[6*j + 3] = _g(K1 @ CNOT @ dmid_db @ CNOT @ K2)
        grad[6*j + 4] = _g(K1 @ CNOT @ mid @ CNOT @ dK2_dt3)
        grad[6*j + 5] = _g(K1 @ CNOT @ mid @ CNOT @ dK2_dt4)

        # suffix_{j-1} = suffix_j @ M_j
        suffix = suffix @ bricks[j]

    return loss, grad


def _loss(all_params: np.ndarray, target: np.ndarray,
          schedule: list[int], n: int) -> float:
    return _loss_and_grad(all_params, target, schedule, n)[0]


def _grad(all_params: np.ndarray, target: np.ndarray,
          schedule: list[int], n: int) -> np.ndarray:
    return _loss_and_grad(all_params, target, schedule, n)[1]


# ---------------------------------------------------------------------------
# Optimiser — incremental warm-start
#
# Core idea: after finding the best L-layer solution, initialise the (L+1)-layer
# run by appending near-identity bricks to the existing parameters.  This avoids
# expensive cold-start global search at every depth level.
# ---------------------------------------------------------------------------

_LBFGS = {"maxiter": 10000, "ftol": 1e-22, "gtol": 1e-13}


def _polish(params: np.ndarray, target: np.ndarray,
            schedule: list[int], n: int) -> tuple[np.ndarray, float]:
    res = minimize(_loss, params, args=(target, schedule, n),
                   jac=_grad, method="L-BFGS-B", options=_LBFGS)
    return res.x, float(res.fun)


def fit_incremental(
    target_so: np.ndarray,   # SO(2^n) target (det=+1 — already D·A)
    n: int,
    max_layers: int,
    tol: float,
    n_restarts: int,
    seed: int = 0,
) -> tuple[np.ndarray, float, int]:
    """Incrementally grow the brickwork, warm-starting each new layer.

    Returns (best_params, residual, min_layers_to_converge).
    If it does not converge within max_layers, returns best found.
    """
    rng        = np.random.default_rng(seed)
    warm_params: np.ndarray | None = None   # best params from previous depth

    for L in range(1, max_layers + 1):
        schedule = brickwork_schedule(n, L)
        n_params = 6 * len(schedule)
        best_x, best_f = None, np.inf

        # ---- candidate initial points ----
        candidates: list[np.ndarray] = []

        # 1. Warm-start: extend previous solution with near-identity new bricks
        if warm_params is not None:
            n_new = n_params - len(warm_params)
            # Try a few perturbations of the warm extension
            for noise in [0.0, 0.1, 0.3]:
                ext = rng.uniform(-noise, noise, n_new) if noise > 0 else np.zeros(n_new)
                candidates.append(np.concatenate([warm_params, ext]))

        # 2. Random restarts (fewer when warm-start is available)
        n_rand = n_restarts if warm_params is None else max(6, n_restarts // 2)
        for _ in range(n_rand):
            candidates.append(rng.uniform(-np.pi, np.pi, n_params))

        # ---- optimise each candidate ----
        for x0 in candidates:
            x, f = _polish(x0, target_so, schedule, n)
            if f < best_f:
                best_f, best_x = f, x

        warm_params = best_x  # carry forward to next depth

        if best_f < tol:
            return best_x, best_f, L

    return warm_params, best_f, max_layers   # didn't converge


def prepare_target(A: np.ndarray, n: int) -> tuple[np.ndarray, int]:
    """Return (SO-target = D·A, correction_cnot_count)."""
    if np.linalg.det(A) < 0:
        return correction_gate(n) @ A, CORRECTION_CNOT.get(n, 6)
    return A.copy(), 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def diagram_label(yd) -> str:
    p = yd.partition
    p = p() if callable(p) else p
    return str(tuple(p))


# DOF of SO(2^n) = 2^n(2^n-1)/2
def so_dof(n: int) -> int:
    d = 2**n
    return d * (d - 1) // 2


# Default max-sizes (minimum k=2^n diagram has size k*(k-1)/2 using staircase)
DEFAULT_MAX_SIZE = {2: 20, 3: 35, 4: 125, 5: 500}

# Default max-layers search bound
DEFAULT_MAX_LAYERS = {2: 4, 3: 12, 4: 25, 5: 60}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--qubits",     type=int, default=2)
    parser.add_argument("--layers",     type=int, default=None,
                        help="Fixed number of layers (skip sweep)")
    parser.add_argument("--sweep",      action="store_true",
                        help="Find minimum layers per diagram")
    parser.add_argument("--scale",      action="store_true",
                        help="Print scaling table: n=2..4, first diagram each")
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--max-size",   type=int, default=None)
    parser.add_argument("--restarts",   type=int, default=12)
    parser.add_argument("--tol",        type=float, default=1e-6)
    args = parser.parse_args()

    # -----------------------------------------------------------------
    # Scaling table
    # -----------------------------------------------------------------
    if args.scale:
        print("=== Brickwork scaling: n qubits, first diagram, incremental sweep ===")
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
            tol      = args.tol

            yd = next((y for y in find_yds_with_fixed_addable_cells(k, max_size)
                       if A_matrix(y).shape[0] == k), None)
            if yd is None:
                print(f"  n={n}: no diagram found (max-size={max_size})")
                continue

            A  = A_matrix(yd)
            T, cx_fix = prepare_target(A, n)
            label = diagram_label(yd)

            t0 = time.perf_counter()
            params, residual, min_L = fit_incremental(
                T, n, max_L, tol,
                n_restarts=args.restarts, seed=0)
            elapsed = time.perf_counter() - t0

            n_bricks = len(brickwork_schedule(n, min_L))
            cx_bw    = 2 * n_bricks
            cx_tot   = cx_bw + cx_fix
            ok       = residual < tol

            print(f"  {n:>1}  {k:>6}  {dof:>6}  "
                  f"{min_L:>6}  {n_bricks:>7}  {cx_bw:>6}  "
                  f"{cx_fix:>7}  {cx_tot:>7}  "
                  f"{residual:>12.2e}  {elapsed:>7.1f}  "
                  f"{'✓' if ok else '✗'} ({label})")

        print()
        print("  CX_bw  = brickwork CNOT (2 per brick)")
        print("  CX_fix = det-correction gate CNOT (CZ/CCZ/CCCZ)")
        print("  CX_tot = total CNOT count for full circuit")
        print()
        print("  Comparison (Givens + Qiskit O(k^2) from compile_benchmark.py):")
        for n in range(2, 5):
            k = 2**n
            # Empirical fit from compile_benchmark: CX ≈ c * k^2
            # n=2: ~12 CX, n=3: ~100+ CX, n=4: ~500+ CX (extrapolated)
            givens_cx = int(0.75 * k * (k - 1))  # rough estimate: 1.5 × k(k-1)/2
            print(f"    n={n} k={k:>3}: Qiskit Givens ≈ {givens_cx} CX")
        return

    # -----------------------------------------------------------------
    # Standard per-qubit modes
    # -----------------------------------------------------------------
    n        = args.qubits
    dim      = 2**n
    k        = dim
    max_size = args.max_size or DEFAULT_MAX_SIZE.get(n, 20)
    max_L    = args.max_layers or DEFAULT_MAX_LAYERS.get(n, 20)
    corr_cx  = CORRECTION_CNOT.get(n, 6)
    fixed_L  = args.layers

    print(f"=== Brickwork synthesis  |  {n} qubits  {dim}×{dim} A-matrices ===")
    print("    Each brick:   Wei-Di SO(4)  =  2 CNOT + 6 Ry")
    print(f"    Det fix:      D = diag(1,...,-1)  =  {corr_cx} CNOT equiv.")
    print(f"    SO({dim}) DOF = {so_dof(n)},  min bricks (theory) = {-(-so_dof(n)//6)}")
    print()

    # -----------------------------------------------------------------
    # Sweep / fixed-layer mode
    # -----------------------------------------------------------------
    if fixed_L is not None:
        # Fixed depth: run all diagrams without incremental sweep
        schedule = brickwork_schedule(n, fixed_L)
        n_bricks = len(schedule)
        cx_bw    = 2 * n_bricks
        cx_tot   = cx_bw + corr_cx

        hdr = (f"{'Diagram':<26} {'det':>4} {'residual':>12}  "
               f"{'bricks':>6} {'CX_bw':>6} {'CX_tot':>7}  {'ok?':>4}")
        print(f"Fixed {fixed_L} layers  ({n_bricks} bricks, {cx_tot} CNOT total):")
        print(hdr)
        print("-" * len(hdr))
        n_total = n_ok = 0
        for idx, yd in enumerate(find_yds_with_fixed_addable_cells(k, max_size)):
            A = A_matrix(yd)
            if A.shape[0] != k:
                continue
            T, _ = prepare_target(A, n)
            params, res, _ = fit_incremental(T, n, fixed_L, args.tol,
                                              args.restarts, seed=idx)
            ok = res < args.tol
            print(f"{diagram_label(yd):<26} {np.linalg.det(A):>+.0f} {res:>12.2e}  "
                  f"{n_bricks:>6} {cx_bw:>6} {cx_tot:>7}  {'✓' if ok else '✗'}")
            n_total += 1
            if ok:
                n_ok += 1
        print("-" * len(hdr))
        print(f"\nResult: {n_ok}/{n_total} synthesized with {cx_tot} CNOT total")

    else:
        # Sweep mode (default): find minimum layers per diagram
        hdr = (f"{'Diagram':<26} {'det':>4}  "
               f"{'min_L':>6}  {'bricks':>7}  {'CX_bw':>6}  {'CX_tot':>7}  "
               f"{'residual':>12}  {'ok?':>4}  {'t(s)':>6}")
        print(f"Layer sweep (1..{max_L}), tol={args.tol:.0e}, max-size={max_size}:")
        print(hdr)
        print("-" * len(hdr))

        n_total = n_ok = 0
        cx_list: list[int] = []

        for idx, yd in enumerate(find_yds_with_fixed_addable_cells(k, max_size)):
            A = A_matrix(yd)
            if A.shape[0] != k:
                continue
            T, _ = prepare_target(A, n)
            t0 = time.perf_counter()
            params, res, min_L = fit_incremental(
                T, n, max_L, args.tol, args.restarts, seed=idx)
            elapsed = time.perf_counter() - t0
            ok = res < args.tol
            n_bricks = len(brickwork_schedule(n, min_L))
            cx_bw    = 2 * n_bricks
            cx_tot   = cx_bw + corr_cx
            print(f"{diagram_label(yd):<26} {np.linalg.det(A):>+.0f}  "
                  f"{min_L:>6}  {n_bricks:>7}  {cx_bw:>6}  {cx_tot:>7}  "
                  f"{res:>12.2e}  {'✓' if ok else '✗'}  {elapsed:>6.1f}")
            n_total += 1
            if ok:
                n_ok += 1
                cx_list.append(cx_tot)

        print("-" * len(hdr))
        print(f"\nResult ({n_total} diagrams, {n_ok} converged):")
        if cx_list:
            print(f"  CNOT total: min={min(cx_list)}  max={max(cx_list)}  "
                  f"mean={np.mean(cx_list):.1f}")
            print(f"  (brickwork + {corr_cx}-CNOT correction gate)")


if __name__ == "__main__":
    main()
