#!/usr/bin/env python3
"""
Apply flagsynth's flag decomposition to A-matrices and compare gate counts.

The flag decomposition (Kottmann et al. 2026) recursively applies the cosine-sine
decomposition to produce a parameter-optimal circuit. For a 2^n × 2^n unitary it
gives 4^n - 1 rotations and ~1/2 * 4^n CNOTs (vs QSD).

For *real orthogonal* A-matrices the trailing diagonal turns out to be ±1 (Clifford
only), so the rotation budget lives entirely in the flag circuit.

Outputs per matrix:
  - RY count, RZ count, CZ count in the flag circuit
  - Whether all diagonal entries are ±1 (real orthogonal A-matrix property)
  - Reconstruction error
  - Comparison vs brickwork CNOT estimate

Usage:
    python flagsynth_amatrix.py                        # n=2,3 A-matrices + random SO
    python flagsynth_amatrix.py --qubits 2             # only k=4 (n=2) matrices
    python flagsynth_amatrix.py --qubits 3             # only k=8 (n=3) matrices
    python flagsynth_amatrix.py --qubits 2 3 --random  # include random SO for comparison
    python flagsynth_amatrix.py --max-size 60          # more diagrams per qubit count
    python flagsynth_amatrix.py --plot --qubits 2 3 4  # scaling plot (gate counts + timing)
    python flagsynth_amatrix.py --plot --plot-output fig.png  # save plot to custom path
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pennylane as qml
from scipy.stats import ortho_group

import flagsynth

from compute_matrix import A_matrix
from helper import find_yds_with_fixed_addable_cells
from yungdiagram import YoungDiagram

# CNOT cost of brickwork per layer (2 per Wei-Di brick).
# From brickwork_synth.py DEFAULT_MAX_LAYERS we know the minimum layers needed.
BRICKWORK_MIN_LAYERS = {2: 4, 3: 12, 4: 25}

# Approximate CNOT cost to implement D = diag(1,...,1,-1) used in brickwork.
CORRECTION_CNOT = {2: 1, 3: 6, 4: 14}


def sample_diagrams(k_addable: int, n_samples: int,
                    rng: np.random.Generator) -> list:
    """Directly construct YoungDiagrams with exactly k_addable addable cells.

    A diagram has k_addable addable cells iff it has exactly k_addable-1
    distinct part values.  We construct these by picking k_addable-1 distinct
    positive integers — one per row — without any partition enumeration.
    """
    n_distinct = k_addable - 1
    results = []
    seen: set = set()

    # Always include the minimal staircase (n_distinct, n_distinct-1, ..., 1)
    staircase = tuple(range(n_distinct, 0, -1))
    results.append(YoungDiagram(staircase))
    seen.add(staircase)

    attempts = 0
    while len(results) < n_samples and attempts < 10_000:
        attempts += 1
        vals = rng.choice(np.arange(1, n_distinct * 4 + 1),
                          size=n_distinct, replace=False)
        parts = tuple(sorted(vals.tolist(), reverse=True))
        if parts not in seen:
            seen.add(parts)
            results.append(YoungDiagram(parts))

    return results


def brickwork_cnot_estimate(n: int) -> int:
    """Rough CNOT count for brickwork at minimum required layers."""
    layers = BRICKWORK_MIN_LAYERS.get(n, 0)
    if layers == 0:
        return 0
    # bricks per layer: ceil((n-1)/2) for even, floor((n-1)/2) for odd
    n_bricks = 0
    for layer in range(layers):
        n_bricks += len(range(layer % 2, n - 1, 2))
    return 2 * n_bricks + CORRECTION_CNOT.get(n, 0)


# ---------------------------------------------------------------------------

def apply_flagsynth(U: np.ndarray) -> dict:
    """
    Run flagsynth on U and return gate count summary.

    Returns dict with keys:
        ops_count, ry, rz, cz, diag_is_pm1, recon_err, n
    """
    n = int(round(np.log2(U.shape[0])))
    wires = list(range(n))

    ops, diag = flagsynth.mux_multi_qubit_decomp(
        [U], mux_wires=[], target_wires=wires, n_b=1, break_down=True
    )

    ry  = sum(1 for o in ops if type(o).__name__ == "RY")
    rz  = sum(1 for o in ops if type(o).__name__ == "RZ")
    cz  = sum(1 for o in ops if type(o).__name__ in ("CZ", "CNOT"))

    # For real orthogonal inputs the diagonal should be ±1 (Clifford, no rotation cost).
    diag_pm1 = bool(np.max(np.abs(np.imag(diag))) < 1e-6 and
                    np.max(np.abs(np.abs(np.real(diag)) - 1.0)) < 1e-6)

    # Reconstruction check
    dev = qml.device("default.qubit", wires=wires)

    @qml.qnode(dev)
    def circuit():
        for op in ops:
            qml.apply(op)
        qml.DiagonalQubitUnitary(diag, wires=wires)
        return qml.state()

    U_rec = qml.matrix(circuit, wire_order=wires)()
    recon_err = float(np.max(np.abs(U_rec - U)))

    return {
        "n": n,
        "ops_count": len(ops),
        "ry": ry,
        "rz": rz,
        "cz": cz,
        "diag_pm1": diag_pm1,
        "recon_err": recon_err,
        "rz_angles": np.array([o.data[0] for o in ops if type(o).__name__ == "RZ"],
                               dtype=float),
    }


def print_stats(label: str, r: dict) -> None:
    n = r["n"]
    so_dof = (4**n - 2**n) // 2  # DOF of SO(2^n)
    bw_cx = brickwork_cnot_estimate(n)
    total_rots = r["ry"] + r["rz"]
    diag_note = "±1 (Clifford)" if r["diag_pm1"] else "complex"

    print(f"  {label}")
    print(f"    flag ops:  RY={r['ry']}  RZ={r['rz']}  CZ={r['cz']}")
    print(f"    total rots in flag: {total_rots}  (SO({2**n}) DOF = {so_dof})")
    print(f"    diagonal:  {diag_note}")
    print(f"    CZ vs brickwork CNOT est: {r['cz']} vs ~{bw_cx}")
    print(f"    recon err: {r['recon_err']:.2e}")

    # RZ angle distribution — are they structured?
    if len(r["rz_angles"]) > 0:
        angles_mod = np.abs(r["rz_angles"]) % np.pi
        near_zero   = np.sum(angles_mod < 0.05)
        near_pi2    = np.sum(np.abs(angles_mod - np.pi / 2) < 0.05)
        near_pi     = np.sum(np.abs(angles_mod - np.pi) < 0.05)
        generic     = len(angles_mod) - near_zero - near_pi2 - near_pi
        print(f"    RZ angle dist (mod π):  ~0={near_zero}  ~π/2={near_pi2}  ~π={near_pi}  generic={generic}")

    print()


# ---------------------------------------------------------------------------

def run_qubit_count(n: int, max_size: int, max_diagrams: int, include_random: bool) -> None:
    k = 2**n
    print(f"{'=' * 60}")
    print(f"n={n} qubits  (k={k}×{k} A-matrices)")
    print(f"{'=' * 60}")

    # --- A-matrices ---
    # For small k use the catalog; for large k construct directly to avoid
    # enumerating billions of partitions.
    min_size_needed = (k - 1) * k // 2
    if min_size_needed <= max_size:
        diagrams = list(find_yds_with_fixed_addable_cells(k, max_size))
        sample = diagrams[:max_diagrams]
    else:
        rng = np.random.default_rng(0)
        sample = sample_diagrams(k, max_diagrams, rng)

    if not sample:
        print("  No diagrams found.")
        return
    ry_counts, rz_counts, cz_counts = [], [], []

    for idx, yd in enumerate(sample):
        A = np.array(A_matrix(yd), dtype=float)
        label_parts = getattr(yd, "partition", yd)
        label = str(tuple(label_parts() if callable(label_parts) else label_parts))

        try:
            r = apply_flagsynth(A)
        except Exception as e:
            print(f"  {label}: ERROR — {e}")
            continue

        print_stats(f"A{label}", r)
        ry_counts.append(r["ry"])
        rz_counts.append(r["rz"])
        cz_counts.append(r["cz"])

    if ry_counts:
        print(f"  Summary over {len(ry_counts)} A-matrices:")
        print(f"    RY: {min(ry_counts)}–{max(ry_counts)}  mean {np.mean(ry_counts):.1f}")
        print(f"    RZ: {min(rz_counts)}–{max(rz_counts)}  mean {np.mean(rz_counts):.1f}")
        print(f"    CZ: {min(cz_counts)}–{max(cz_counts)}  mean {np.mean(cz_counts):.1f}")
        print()

    # --- Random SO(k) for comparison ---
    if include_random:
        print(f"  Random SO({k}) comparison (5 samples):")
        rng = np.random.default_rng(0)
        for i in range(5):
            R = ortho_group.rvs(k, random_state=rng)
            if np.linalg.det(R) < 0:
                R[:, 0] *= -1
            try:
                r = apply_flagsynth(R)
            except Exception as e:
                print(f"    random {i}: ERROR — {e}")
                continue
            print_stats(f"random SO({k}) #{i}", r)


# ---------------------------------------------------------------------------

def plot_scaling(qubit_range: list[int], max_size: int, max_diagrams: int,
                 output_path: str | None = None) -> None:
    """Plot flag decomp gate counts and timing as a function of matrix size."""
    import matplotlib.pyplot as plt

    rows: list[tuple] = []  # (n, ry, rz, cz, elapsed_s)

    rng = np.random.default_rng(0)
    for n in qubit_range:
        k = 2**n
        print(f"  n={n} (k={k}x{k})...", end=" ", flush=True)

        diagrams = sample_diagrams(k, max_diagrams, rng)
        if not diagrams:
            print("no diagrams found")
            continue

        sample = diagrams[:max_diagrams]
        n_ok = 0
        for yd in sample:
            A = np.array(A_matrix(yd), dtype=float)
            t0 = time.perf_counter()
            try:
                r = apply_flagsynth(A)
            except Exception as e:
                print(f"ERROR: {e}")
                continue
            elapsed = time.perf_counter() - t0
            rows.append((n, r["ry"], r["rz"], r["cz"], elapsed))
            n_ok += 1

        if n_ok:
            subset = [(ry, rz, cz, t) for (nn, ry, rz, cz, t) in rows if nn == n]
            ry0, rz0, cz0 = subset[0][0], subset[0][1], subset[0][2]
            t_mean = np.mean([s[3] for s in subset])
            print(f"RY={ry0}  RZ={rz0}  CZ={cz0}  t={t_mean:.2f}s/matrix  ({n_ok} matrices)")
        else:
            print("all failed")

    if not rows:
        print("No data to plot.")
        return

    data = np.array(rows, dtype=float)
    ns_unique = np.unique(data[:, 0]).astype(int)
    ks = 2**ns_unique

    def mean_col(col_idx):
        return np.array([data[data[:, 0] == n, col_idx].mean() for n in ns_unique])

    ry_m  = mean_col(1)
    rz_m  = mean_col(2)
    cz_m  = mean_col(3)
    t_m   = mean_col(4)

    # Theoretical reference curves
    k_th  = np.array([2**n for n in range(1, ns_unique.max() + 2)])
    so_dof = (k_th**2 - k_th) // 2          # SO(k) degrees of freedom
    flag_cz = (4**np.log2(k_th).astype(int) - 1) // 3  # (4^n - 1) / 3

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left: gate counts ---
    ax1.plot(ks, ry_m, "o-", label="RY (non-Clifford)")
    ax1.plot(ks, cz_m, "s-", label="CZ")
    ax1.plot(ks, rz_m, "^-", alpha=0.6, label="RZ (Clifford ×π)")
    ax1.plot(k_th, so_dof, "k--", alpha=0.4, label=r"$(k^2{-}k)/2$  [SO($k$) DOF]")
    ax1.set_xlabel("Matrix size $k = 2^n$")
    ax1.set_ylabel("Gate count")
    ax1.set_title("Flag decomposition gate counts for A-matrices")
    ax1.set_xticks(ks)
    ax1.set_xticklabels([f"$k={k}$\n($n={n}$)" for k, n in zip(ks, ns_unique)])
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- Right: timing ---
    ax2.plot(ks, t_m, "D-", color="tab:red")
    ax2.set_xlabel("Matrix size $k = 2^n$")
    ax2.set_ylabel("Time per matrix (s)")
    ax2.set_title("Flag decomp compute time per A-matrix")
    ax2.set_xticks(ks)
    ax2.set_xticklabels([f"$k={k}$\n($n={n}$)" for k, n in zip(ks, ns_unique)])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_path or "flagsynth_scaling.png"
    plt.savefig(path, dpi=150)
    print(f"\nPlot saved to {path}")


# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--qubits", type=int, nargs="+", default=[2, 3],
                   help="Qubit counts to test (default: 2 3)")
    p.add_argument("--max-size", type=int, default=40,
                   help="Max Young diagram size for catalog (default: 40)")
    p.add_argument("--max-diagrams", type=int, default=8,
                   help="Max A-matrices to test per qubit count (default: 8)")
    p.add_argument("--random", action="store_true",
                   help="Also run random SO matrices for comparison")
    p.add_argument("--plot", action="store_true",
                   help="Plot gate counts and timing vs matrix size")
    p.add_argument("--plot-output", type=str, default=None,
                   help="Output path for plot (default: flagsynth_scaling.png)")
    args = p.parse_args()

    if args.plot:
        print("Collecting data for scaling plot...")
        plot_scaling(args.qubits, args.max_size, args.max_diagrams, args.plot_output)
        return

    for n in args.qubits:
        if n < 2:
            print(f"Skipping n={n} (need n>=2 for flag decomposition).")
            continue
        if n > 4:
            print(f"Warning: n={n} may be slow.")
        run_qubit_count(n, args.max_size, args.max_diagrams, args.random)


if __name__ == "__main__":
    main()
