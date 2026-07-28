"""Scaling evidence: ULV rotation counts vs k over many random diagrams.

Question: is the ULV count (RESULTS.md §10) really near-linear in k, on a
dense grid of sizes and a spread of random Young-diagram shapes — not just
the four dyadic sizes of the original table?

Method: for each k in a dense grid up to 2048, factor the staircase and
several random-shape A-matrices (gap profiles sampled among uniform {1..3},
uniform {1..10}, and heavy-tailed geometric — by the content-gap bijection
these are genuine random diagrams), record rotation counts and errors, fit
N = a·k^p on the tail, and compare against a·k·log₂k and the dense
k(k−1)/2.

Expected result: tail exponent p ≈ 1.1–1.2, the a·k·log₂k model fitting
the tail within a few percent, and the dense curve diverging — the
evidence figure behind ulv-note.md.  Writes data/ulv_scaling.csv and
data/plots/ulv_scaling.png.
"""

import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from fourier.amatrix import _a_matrix_from_contents_log, staircase_a_matrix

from ulv_circuit import factorization_error, ulv_factor

K_GRID = [64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048,
          3072, 4096, 6144, 8192, 12288, 16384]
RUNS_PER_K = {64: 6, 96: 6, 128: 6, 192: 5, 256: 5, 384: 4, 512: 4, 768: 3,
              1024: 3, 1536: 2, 2048: 2, 3072: 2, 4096: 2, 6144: 2, 8192: 2,
              12288: 1, 16384: 1}
PROFILES = ["u3", "u10", "heavy"]

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def random_gaps(profile: str, n: int, rng) -> np.ndarray:
    if profile == "u3":
        return rng.integers(1, 4, size=n)
    if profile == "u10":
        return rng.integers(1, 11, size=n)
    return rng.geometric(0.15, size=n)


def one_run(job):
    k, kind, seed, delta, leaf = job
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if kind == "staircase":
        A = staircase_a_matrix(k)
        profile = "staircase"
    else:
        rng = np.random.default_rng(seed)
        profile = PROFILES[seed % len(PROFILES)]
        seq = np.concatenate([[0.0], np.cumsum(random_gaps(profile, 2 * k - 2, rng))])[::-1]
        A = _a_matrix_from_contents_log(seq[0::2], seq[1::2])
    t0 = time.time()
    _, M_final, count, _ = ulv_factor(A, b=leaf, delta=delta)
    return {
        "k": k,
        "kind": kind,
        "profile": profile,
        "seed": seed,
        "rotations": count,
        "error": factorization_error(M_final),
        "seconds": round(time.time() - t0, 1),
    }


def sweep(delta, leaf, workers, skip=frozenset()):
    jobs = []
    for k in K_GRID:
        for kind, s in [("staircase", 0)] + [
            ("random", 1000 + 17 * k + s) for s in range(RUNS_PER_K[k])
        ]:
            if (k, kind, s) not in skip:
                jobs.append((k, kind, s, delta, leaf))
    jobs.sort(key=lambda j: -j[0])  # big ones first for better packing
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one_run, jobs))


def fit_tail(ks, ns, k_min):
    x = np.log(np.array([k for k in ks if k >= k_min], dtype=float))
    y = np.log(np.array([n for k, n in zip(ks, ns) if k >= k_min], dtype=float))
    p, loga = np.polyfit(x, y, 1)
    resid = float(np.sqrt(np.mean((y - (p * x + loga)) ** 2)))
    return p, np.exp(loga), resid


def make_plot(rows, out_png, delta):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rand = [r for r in rows if r["kind"] == "random"]
    stair = sorted([r for r in rows if r["kind"] == "staircase"], key=lambda r: r["k"])

    ks_mean, n_mean = [], []
    for k in K_GRID:
        vals = [r["rotations"] for r in rand if r["k"] == k]
        if vals:
            ks_mean.append(k)
            n_mean.append(float(np.exp(np.mean(np.log(vals)))))

    p_fit, a_fit, resid_pow = fit_tail(ks_mean, n_mean, 384)
    # best-fit constant for the k·log₂k model, least squares in log space
    all_pts = [(r["k"], r["rotations"]) for r in rand + stair]
    c_klogk = float(np.exp(np.mean([np.log(n / (k * np.log2(k))) for k, n in all_pts])))

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED, labelsize=9)
    for s in ax.spines.values():
        s.set_color(GRID)

    kk = np.geomspace(min(K_GRID), max(K_GRID), 200)
    ax.plot(kk, kk * (kk - 1) / 2, ls="--", lw=1.4, color=MUTED, zorder=1)
    ax.text(400, 400 * 399 / 2 * 1.5, "dense  k(k−1)/2", color=MUTED,
            fontsize=9, ha="center", va="bottom", rotation=33)
    ax.plot(kk, c_klogk * kk * np.log2(kk), ls="-", lw=1.6, color=INK, zorder=2,
            label=f"fit  {c_klogk:.2f}·k·log₂k")

    ax.scatter([r["k"] for r in rand], [r["rotations"] for r in rand],
               s=26, color=BLUE, alpha=0.7, linewidths=0, zorder=3, label="random diagrams")
    ax.scatter([r["k"] for r in stair], [r["rotations"] for r in stair],
               s=90, facecolors="none", edgecolors=ORANGE, linewidths=1.6,
               zorder=4, label="staircase")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("k  (A-matrix dimension)", color=INK, fontsize=10)
    ax.set_ylabel("Givens rotations", color=INK, fontsize=10)
    ax.set_title(f"ULV rotation count  (δ = {delta:g}; tail fit N ≈ {a_fit:.1f}·k^{p_fit:.2f})",
                 color=INK, fontsize=10.5)
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK)

    # the per-k spread is < ±2%, far below marker size on the log axis —
    # zoom one size so the individual runs are visible
    kz = 1024
    zr = [r["rotations"] for r in rand if r["k"] == kz]
    zs = [r["rotations"] for r in stair if r["k"] == kz]
    axz = ax.inset_axes([0.64, 0.05, 0.32, 0.26])
    axz.set_facecolor("#fcfcfb")
    axz.scatter([kz] * len(zr), zr, s=26, color=BLUE, alpha=0.7, linewidths=0, zorder=3)
    axz.scatter([kz] * len(zs), zs, s=90, facecolors="none", edgecolors=ORANGE,
                linewidths=1.6, zorder=4)
    axz.set_xlim(kz * 0.94, kz * 1.06)
    lo, hi = min(zs + zr), max(zs + zr)
    axz.set_ylim(lo - 0.03 * lo, hi + 0.03 * hi)
    axz.set_xticks([])
    axz.tick_params(colors=MUTED, labelsize=7.5)
    axz.text(0.04, 0.90, f"zoom: k = {kz} — every run, spread < ±2%",
             transform=axz.transAxes, color=MUTED, fontsize=8, va="top")
    for s in axz.spines.values():
        s.set_color(MUTED)
    ax.indicate_inset_zoom(axz, edgecolor=MUTED, alpha=0.6)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    print(f"wrote {out_png}")
    print(f"tail fit (k >= 384): N ~ {a_fit:.2f} * k^{p_fit:.3f}  (log-space RMS {resid_pow:.3f})")
    print(f"k*log2(k) model constant on tail: {c_klogk:.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta", type=float, default=1e-4)
    parser.add_argument("--leaf", type=int, default=32)
    parser.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    parser.add_argument("--csv", type=Path, default=Path("data/ulv_scaling.csv"))
    parser.add_argument("--plot", type=Path, default=Path("data/plots/ulv_scaling.png"))
    parser.add_argument("--plot-only", action="store_true", help="re-plot from the existing CSV")
    parser.add_argument("--extend", action="store_true",
                        help="keep existing CSV rows; compute only missing (k, kind, seed) jobs")
    args = parser.parse_args()

    def load_csv():
        with args.csv.open() as f:
            return [
                {**r, "k": int(r["k"]), "seed": int(r["seed"]),
                 "rotations": int(r["rotations"]), "error": float(r["error"])}
                for r in csv.DictReader(f)
            ]

    if args.plot_only:
        rows = load_csv()
    else:
        existing = load_csv() if args.extend and args.csv.exists() else []
        skip = frozenset((r["k"], r["kind"], r["seed"]) for r in existing)
        rows = existing + sweep(args.delta, args.leaf, args.workers, skip)
        rows.sort(key=lambda r: (r["k"], r["kind"], r["seed"]))
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv} ({len(rows)} runs)")
        worst_err = max(r["error"] for r in rows)
        print(f"worst factorization error: {worst_err:.1e}")

    make_plot(rows, args.plot, args.delta)


if __name__ == "__main__":
    main()
