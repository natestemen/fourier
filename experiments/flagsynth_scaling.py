#!/usr/bin/env python3
"""Flag decomposition (flagsynth) of A-matrices: gate counts and scaling.

Question: what does the parameter-optimal flag decomposition (Kottmann et
al. 2026 — recursive cosine–sine synthesis, 4ⁿ−1 rotations for a 2ⁿ×2ⁿ
unitary) cost on A-matrices, and does their real-orthogonal structure show
up in the circuit?

Supports report.md, Open Direction 3 (break the k² compiler barrier): the
flag decomposition is exactly the CS-recursion circuit the report proposes
handing to compilers, so its RY/RZ/CZ counts are the structured-input
baseline to compare against the Finding 2 Givens→compiler counts.

Expected result: reconstruction error < 1e-6 everywhere, RY count tracking
the SO(k) dimension k(k−1)/2, and a trailing diagonal that is a pure
Clifford phase — ±1 for k = 8, ±i for k = 4.  (The ±1-only check inherited
from the old script prints "complex" for the ±i case.)

Outputs: with --plot, data/plots/flagsynth_scaling.png (moved from the old
script's default of ./flagsynth_scaling.png); otherwise a per-matrix report
on stdout.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pennylane as qml
from scipy.stats import ortho_group
from yungdiagram import YoungDiagram

import flagsynth

from fourier import a_matrix, diagrams_with_addable_cells, staircase

# CNOT cost of brickwork synthesis (see experiments/brickwork.py) at the
# minimum layer counts found there — 2 CNOT per Wei–Di brick plus the
# det-correction gate.
BRICKWORK_MIN_LAYERS = {2: 4, 3: 12, 4: 25}
CORRECTION_CNOT = {2: 1, 3: 6, 4: 14}


def sample_diagrams(k_addable: int, n_samples: int, rng: np.random.Generator) -> list:
    """Directly construct diagrams with exactly `k_addable` addable cells.

    A diagram has k_addable addable cells iff it has k_addable−1 distinct
    part values, so drawing distinct part values avoids enumerating
    partitions — essential for large k where the catalog is astronomical.
    """
    n_distinct = k_addable - 1
    first = staircase(n_distinct)
    results = [first]
    seen = {first.partition}

    attempts = 0
    while len(results) < n_samples and attempts < 10_000:
        attempts += 1
        vals = rng.choice(
            np.arange(1, n_distinct * 4 + 1), size=n_distinct, replace=False
        )
        parts = tuple(sorted(vals.tolist(), reverse=True))
        if parts not in seen:
            seen.add(parts)
            results.append(YoungDiagram(parts))
    return results


def brickwork_cnot_estimate(n: int) -> int:
    """Rough CNOT count for brickwork at its minimum required layers."""
    layers = BRICKWORK_MIN_LAYERS.get(n, 0)
    if layers == 0:
        return 0
    n_bricks = sum(len(range(layer % 2, n - 1, 2)) for layer in range(layers))
    return 2 * n_bricks + CORRECTION_CNOT.get(n, 0)


def apply_flagsynth(U: np.ndarray) -> dict:
    """Run flagsynth on U; gate counts, diagonal check, reconstruction error."""
    n = int(round(np.log2(U.shape[0])))
    wires = list(range(n))

    ops, diag = flagsynth.mux_multi_qubit_decomp(
        [U], mux_wires=[], target_wires=wires, n_b=1, break_down=True
    )

    ry = sum(1 for o in ops if type(o).__name__ == "RY")
    rz = sum(1 for o in ops if type(o).__name__ == "RZ")
    cz = sum(1 for o in ops if type(o).__name__ in ("CZ", "CNOT"))

    # For real orthogonal inputs the diagonal should be ±1 (Clifford).
    diag_pm1 = bool(
        np.max(np.abs(np.imag(diag))) < 1e-6
        and np.max(np.abs(np.abs(np.real(diag)) - 1.0)) < 1e-6
    )

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
        "rz_angles": np.array(
            [o.data[0] for o in ops if type(o).__name__ == "RZ"], dtype=float
        ),
    }


def print_stats(label: str, r: dict) -> None:
    n = r["n"]
    so_dof = (4**n - 2**n) // 2  # DOF of SO(2^n)
    diag_note = "±1 (Clifford)" if r["diag_pm1"] else "complex"

    print(f"  {label}")
    print(f"    flag ops:  RY={r['ry']}  RZ={r['rz']}  CZ={r['cz']}")
    print(f"    total rots in flag: {r['ry'] + r['rz']}  (SO({2**n}) DOF = {so_dof})")
    print(f"    diagonal:  {diag_note}")
    print(f"    CZ vs brickwork CNOT est: {r['cz']} vs ~{brickwork_cnot_estimate(n)}")
    print(f"    recon err: {r['recon_err']:.2e}")

    if len(r["rz_angles"]) > 0:
        angles_mod = np.abs(r["rz_angles"]) % np.pi
        near_zero = np.sum(angles_mod < 0.05)
        near_pi2 = np.sum(np.abs(angles_mod - np.pi / 2) < 0.05)
        near_pi = np.sum(np.abs(angles_mod - np.pi) < 0.05)
        generic = len(angles_mod) - near_zero - near_pi2 - near_pi
        print(f"    RZ angle dist (mod π):  ~0={near_zero}  ~π/2={near_pi2}  "
              f"~π={near_pi}  generic={generic}")
    print()


def run_qubit_count(
    n: int, max_size: int, max_diagrams: int, include_random: bool
) -> None:
    k = 2**n
    print("=" * 60)
    print(f"n={n} qubits  (k={k}×{k} A-matrices)")
    print("=" * 60)

    # Use the catalog when the minimal diagram fits under max_size; otherwise
    # construct diagrams directly to avoid enumerating billions of partitions.
    if (k - 1) * k // 2 <= max_size:
        sample = list(diagrams_with_addable_cells(k, max_size))[:max_diagrams]
    else:
        sample = sample_diagrams(k, max_diagrams, np.random.default_rng(0))

    if not sample:
        print("  No diagrams found.")
        return

    ry_counts, rz_counts, cz_counts = [], [], []
    for yd in sample:
        A = a_matrix(yd)
        label = str(yd.partition)
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


def plot_scaling(qubit_range: list[int], max_diagrams: int, output_path: Path) -> None:
    """Plot flag-decomposition gate counts and timing vs matrix size."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows: list[tuple] = []  # (n, ry, rz, cz, elapsed_s)
    rng = np.random.default_rng(0)
    for n in qubit_range:
        k = 2**n
        print(f"  n={n} (k={k}x{k})...", end=" ", flush=True)

        sample = sample_diagrams(k, max_diagrams, rng)
        n_ok = 0
        for yd in sample:
            A = a_matrix(yd)
            t0 = time.perf_counter()
            try:
                r = apply_flagsynth(A)
            except Exception as e:
                print(f"ERROR: {e}")
                continue
            rows.append((n, r["ry"], r["rz"], r["cz"], time.perf_counter() - t0))
            n_ok += 1

        if n_ok:
            subset = [row[1:] for row in rows if row[0] == n]
            ry0, rz0, cz0 = subset[0][0], subset[0][1], subset[0][2]
            t_mean = np.mean([s[3] for s in subset])
            print(f"RY={ry0}  RZ={rz0}  CZ={cz0}  t={t_mean:.2f}s/matrix  "
                  f"({n_ok} matrices)")
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

    ry_m, rz_m, cz_m, t_m = mean_col(1), mean_col(2), mean_col(3), mean_col(4)

    k_th = np.array([2**n for n in range(1, ns_unique.max() + 2)])
    so_dof = (k_th**2 - k_th) // 2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

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

    ax2.plot(ks, t_m, "D-", color="tab:red")
    ax2.set_xlabel("Matrix size $k = 2^n$")
    ax2.set_ylabel("Time per matrix (s)")
    ax2.set_title("Flag decomp compute time per A-matrix")
    ax2.set_xticks(ks)
    ax2.set_xticklabels([f"$k={k}$\n($n={n}$)" for k, n in zip(ks, ns_unique)])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--qubits", type=int, nargs="+", default=[2, 3],
                        help="qubit counts to test (default: 2 3)")
    parser.add_argument("--max-size", type=int, default=40,
                        help="max diagram size for the catalog (default: 40)")
    parser.add_argument("--max-diagrams", type=int, default=8,
                        help="max A-matrices per qubit count (default: 8)")
    parser.add_argument("--random", action="store_true",
                        help="also run random SO matrices for comparison")
    parser.add_argument("--plot", action="store_true",
                        help="plot gate counts and timing vs matrix size")
    parser.add_argument("--plot-output", type=Path,
                        default=Path("data/plots/flagsynth_scaling.png"),
                        help="output path for the --plot figure")
    args = parser.parse_args()

    if args.plot:
        print("Collecting data for scaling plot...")
        plot_scaling(args.qubits, args.max_diagrams, args.plot_output)
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
