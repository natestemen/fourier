"""Cost of accuracy: ULV rotation count as a function of the truncation δ.

Question: is the circuit cost logarithmic in 1/δ?  This is load-bearing for
the QFT-on-Sₙ application: the transform composes many A-matrix blocks and
operator errors add, so each block must be built at δ ≈ ε_total/#blocks —
the per-block accuracy shrinks with problem size, and only a log(1/δ) cost
keeps the near-linear k-scaling win intact under composition.

Method: for each δ from 10⁻² to 10⁻¹², factor staircase and random
A-matrices at several k, optimizing the leaf size b per (k, δ) over a small
grid (the mechanism is N ~ k·b(δ) with the required b(δ) ~ 2·r(δ) growing
logarithmically — a fixed b stalls once ranks reach it).  Plot the collapse
N/(k·log₂k) vs digits of accuracy: if N = c(δ)·k·log₂k with c linear in
log(1/δ), every k falls on one line.

Expected result: a straight collapse line, ~0.6–0.9 units of N/(k·log₂k)
per digit of accuracy, with measured errors tracking δ.  Writes
data/ulv_accuracy.csv and data/plots/ulv_accuracy.png.
"""

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from fourier.amatrix import random_content_a_matrix, staircase_a_matrix

from ulv_circuit import factorization_error, ulv_factor

DELTAS = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12]
KS = [512, 1024, 2048]
LEAVES = [24, 32, 48, 64, 80]

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
K_COLORS = dict(zip(KS, [BLUE, ORANGE, AQUA]))
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def one_run(job):
    k, kind, delta, leaf = job
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if kind == "staircase":
        A = staircase_a_matrix(k)
    else:
        A = random_content_a_matrix(k, np.random.default_rng(11 * k))
    _, M_final, count, _ = ulv_factor(A, b=leaf, delta=delta)
    return {
        "k": k,
        "kind": kind,
        "delta": delta,
        "leaf": leaf,
        "rotations": count,
        "error": factorization_error(M_final),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    parser.add_argument("--csv", type=Path, default=Path("data/ulv_accuracy.csv"))
    parser.add_argument("--plot", type=Path, default=Path("data/plots/ulv_accuracy.png"))
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    if args.plot_only:
        with args.csv.open() as f:
            rows = [
                {**r, "k": int(r["k"]), "delta": float(r["delta"]), "leaf": int(r["leaf"]),
                 "rotations": int(r["rotations"]), "error": float(r["error"])}
                for r in csv.DictReader(f)
            ]
    else:
        jobs = [
            (k, kind, d, b)
            for k in KS
            for kind in ("staircase", "random")
            for d in DELTAS
            for b in LEAVES
        ]
        jobs.sort(key=lambda j: -(j[0] * j[3]))
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(one_run, jobs))
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv} ({len(rows)} runs)")

    # best-leaf envelope per (k, kind, delta)
    best = {}
    for r in rows:
        key = (r["k"], r["kind"], r["delta"])
        if key not in best or r["rotations"] < best[key]["rotations"]:
            best[key] = r

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED, labelsize=9)
    for s in ax.spines.values():
        s.set_color(GRID)

    digits_all, norm_all = [], []
    for k in KS:
        for kind, filled in (("staircase", False), ("random", True)):
            pts = sorted(
                (r for (kk, kd, _), r in best.items() if kk == k and kd == kind),
                key=lambda r: -r["delta"],
            )
            digits = [np.log10(1 / r["delta"]) for r in pts]
            norm = [r["rotations"] / (k * np.log2(k)) for r in pts]
            digits_all += digits
            norm_all += norm
            if filled:
                ax.scatter(digits, norm, s=30, color=K_COLORS[k], linewidths=0,
                           label=f"k = {k}")
            else:
                ax.scatter(digits, norm, s=70, facecolors="none",
                           edgecolors=K_COLORS[k], linewidths=1.4)

    slope, icept = np.polyfit(digits_all, norm_all, 1)
    dd = np.linspace(min(digits_all), max(digits_all), 50)
    ax.plot(dd, icept + slope * dd, color=INK, lw=1.5,
            label=f"fit  {icept:.1f} + {slope:.2f}·digits")

    ax.set_xlabel("digits of accuracy  log₁₀(1/δ)", color=INK, fontsize=10)
    ax.set_ylabel("N / (k·log₂k)", color=INK, fontsize=10)
    ax.set_title("Cost of accuracy (leaf size optimized per δ; "
                 "rings = staircase, dots = random)", color=INK, fontsize=10.5)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)

    fig.tight_layout()
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, dpi=160)
    print(f"wrote {args.plot}")
    print(f"collapse fit: N/(k·log₂k) ≈ {icept:.2f} + {slope:.3f}·log₁₀(1/δ)")
    for k in KS:
        r = best[(k, "staircase", 1e-12)]
        print(f"  k={k}, δ=1e-12: N={r['rotations']:,} (b={r['leaf']}), "
              f"err={r['error']:.1e}, vs dense {k * (k - 1) // 2:,}")


if __name__ == "__main__":
    main()
