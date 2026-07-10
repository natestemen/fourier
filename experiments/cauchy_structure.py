"""Demonstrate and verify the Cauchy structure of A-matrices.

Question: does the factorization A[:,1:] = diag(α)·C·diag(β) (C Cauchy,
displacement rank 1) really deliver the promised fast algorithms — an
O(k log² k) mat-vec in general, O(k log k) via FFT for staircases, and an
O(k)-depth CS circuit — and what do the operation counts look like?

The structure itself now lives in the library (fourier.amatrix.cauchy_form);
this script only exercises it:

1. displacement-rank check: diag(ac)·C − C·diag(rc) = 1·1ᵀ,
2. fast mat-vec correctness: CauchyForm.matvec / matvec_fast vs direct A@v,
3. staircase Toeplitz mat-vec: CauchyForm.matvec_toeplitz vs direct,
4. op-count table + scaling plot (data/plots/cauchy_scaling.png,
   data/cauchy_opcounts.csv),
5. the explicit CS decomposition of A(3,2,1): 6 Givens rotations.

Supports report.md, "Finding 3: Cauchy Structure and Its Consequences".

Expected result: all checks pass (errors ~1e-15); the CS decomposition of
A(3,2,1) reconstructs to <1e-14 with 2 CS angles + 4 sub-block rotations.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from yungdiagram import YoungDiagram

from fourier.amatrix import a_matrix, cauchy_form
from fourier.decompositions import cs_factor
from fourier.diagrams import staircase

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PLOT_DIR = DATA_DIR / "plots"

# Okabe–Ito colorblind-safe hues, fixed per series.
_COLORS = {
    "naive": "#000000",
    "fast": "#0072B2",
    "fft": "#009E73",
    "cs": "#D55E00",
}


# ── op-count models (report.md Finding 3) ─────────────────────────────────────


def op_count_fast(k: int) -> int:
    """O(k log² k) for the partial-fractions multipoint evaluation."""
    return int(k * np.log2(k + 1) ** 2) if k > 1 else 1


def op_count_fft(k: int) -> int:
    """O(k log k) for the Toeplitz FFT mat-vec (staircase only)."""
    return int(k * np.log2(k + 1)) if k > 1 else 1


def cs_gate_count(k: int) -> int:
    """Gates of the CS recursion T(k) = 2T(k/2) + k/2 — O(k log k) IF the
    sub-blocks recurse as A-matrices (the open conjecture)."""
    if k <= 2:
        return 1
    return 2 * cs_gate_count(k // 2) + k // 2


# ── verification sections ──────────────────────────────────────────────────────


def check_displacement(partitions: list[list[int]]) -> None:
    print("\n1. Displacement rank = 1 (all partitions):")
    for part in partitions:
        cf = cauchy_form(YoungDiagram(part))
        D = cf.displacement
        rank_D = np.linalg.matrix_rank(D, tol=1e-9)
        is_ones = np.allclose(D, np.ones_like(D))
        print(f"   {str(tuple(part)):20s}  k={len(cf.ac)},  "
              f"rank(X·C - C·Y) = {rank_D},  = 1·1ᵀ: {is_ones}")


def check_matvec(partitions: list[list[int]]) -> None:
    print("\n2. Fast mat-vec correctness (partial-fractions formula):")
    for part in partitions:
        d = YoungDiagram(part)
        cf = cauchy_form(d)
        v = np.random.default_rng(42).standard_normal(len(cf.ac))
        r_direct = a_matrix(d) @ v
        ok_factored = np.allclose(cf.matvec(v), r_direct)
        ok_fast = np.allclose(cf.matvec_fast(v), r_direct, atol=1e-8)
        print(f"   {tuple(part)}: factored≈direct={ok_factored}, fast≈direct={ok_fast}")


def check_staircase_toeplitz(max_k: int) -> None:
    print("\n3. Staircase Toeplitz FFT correctness:")
    rng = np.random.default_rng(0)
    for k in range(2, max_k + 1):
        d = staircase(k)
        cf = cauchy_form(d)
        v = rng.standard_normal(len(cf.ac))
        ok = np.allclose(cf.matvec_toeplitz(v), a_matrix(d) @ v, atol=1e-9)
        print(f"   k={k}  {d.partition}: FFT≈direct={ok}")


def op_count_table(ks: list[int]) -> list[dict]:
    print("\n4. Operation count comparison:")
    print(f"  {'k':>5} | {'Naive k²':>10} | {'Fast log²k':>12} | {'FFT k log k':>12} | {'Circuit (CS)':>13}")
    print("  " + "-" * 60)
    rows = []
    for k in ks:
        row = {
            "k": k,
            "naive": k * k,
            "fast": op_count_fast(k),
            "fft": op_count_fft(k),
            "cs": cs_gate_count(k),
        }
        rows.append(row)
        print(f"  {k:>5} | {row['naive']:>10,} | {row['fast']:>12,} | "
              f"{row['fft']:>12,} | {row['cs']:>13,}")
    return rows


def scaling_plot(rows: list[dict], out_path: Path) -> None:
    ks = [r["k"] for r in rows]
    series = [
        ("naive", "Naive O(k²)", "o"),
        ("fast", "Fast Cauchy O(k log²k)", "s"),
        ("fft", "Toeplitz FFT O(k log k)", "^"),
        ("cs", "CS circuit O(k log k)", "d"),
    ]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for key, label, marker in series:
        ax.loglog(ks, [r[key] for r in rows], marker=marker, color=_COLORS[key], label=label)
    ax.set_xlabel("Matrix dimension k", fontsize=11)
    ax.set_ylabel("Operations", fontsize=11)
    ax.set_title("Operation count: Cauchy A-matrix decomposition", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    for key, label, marker in series[1:]:
        speedup = [r["naive"] / r[key] for r in rows]
        ax2.semilogx(ks, speedup, marker=marker, color=_COLORS[key],
                     label=label.split(" O(")[0] + " speedup")
    ax2.set_xlabel("Matrix dimension k", fontsize=11)
    ax2.set_ylabel("Speedup over naive", fontsize=11)
    ax2.set_title("Speedup vs naive O(k²)", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out_path}")


def explicit_cs_321() -> None:
    print("\n5. EXPLICIT BUTTERFLY CIRCUIT for A(3,2,1) — k=4:")
    print("""
   A(3,2,1) has content sequence: 3 > 2 > 1 > 0 > -1 > -2 > -3
                                  A   R   A   R    A    R    A

   CS decomposition (p=2, q=2):
     Layer 1:  2 Givens rotations (the CS angles), mixing rows {0↔2}, {1↔3}
     Layer 2:  U₁, U₂ — one Givens each on rows {0,1} and rows {2,3}
     Layer 3:  V₁, V₂ — one Givens each on cols {0,1} and cols {2,3}
   TOTAL: 6 Givens rotations (same as naive at k=4 — advantage grows with k)
""")
    A321 = a_matrix(YoungDiagram([3, 2, 1]))
    cs = cs_factor(A321)
    print("   CS angles (radians):", np.round(cs.thetas, 5))
    print(f"   Reconstruction error: {np.max(np.abs(A321 - cs.matrix())):.2e}")
    print(f"   U₁ Givens angle: {np.degrees(np.arccos(cs.u1[0, 0])):.3f}°")
    print(f"   U₂ Givens angle: {np.degrees(np.arccos(cs.u2[0, 0])):.3f}°")
    print(f"   V₁ Givens angle: {np.degrees(np.arccos(cs.v1t[0, 0])):.3f}°")
    print(f"   V₂ Givens angle: {np.degrees(np.arccos(cs.v2t[0, 0])):.3f}°")
    print("   Total: 2 (CS) + 1+1 (U) + 1+1 (V) = 6 Givens rotations")


# ── CLI ────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--max-stair", type=int, default=5,
                   help="Largest staircase k for the Toeplitz FFT check (default: 5).")
    p.add_argument("--max-log-k", type=int, default=10,
                   help="Op-count table covers k = 2, 4, …, 2^N (default: 10 → k=1024).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  CAUCHY STRUCTURE VERIFICATION (library: fourier.amatrix.cauchy_form)")
    print("=" * 65)

    staircases = [list(range(k, 0, -1)) for k in range(2, 6)]
    check_displacement(staircases)
    check_matvec(staircases[:3])
    check_staircase_toeplitz(args.max_stair)

    ks = [2**i for i in range(1, args.max_log_k + 1)]
    rows = op_count_table(ks)

    csv_path = DATA_DIR / "cauchy_opcounts.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["k", "naive", "fast", "fft", "cs"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {csv_path}")

    scaling_plot(rows, PLOT_DIR / "cauchy_scaling.png")
    explicit_cs_321()


if __name__ == "__main__":
    main()
