#!/usr/bin/env python3
"""Givens triangularization vs CS butterfly: gate count and depth vs k.

Question: does the recursive cosine–sine (CS) butterfly decomposition of an
A-matrix beat plain Givens triangularization in two-qubit gate count or in
parallel circuit depth?

Supports report.md, Finding 3 (final paragraph): across ~800 random
partitions the CS butterfly matches Givens in total gate count — both
O(k²) — but achieves O(k) parallel depth.

Expected result: identical gate counts (fit ≈ 0.28·k^2.2 over the sampled
range) with depth 2k−3 for Givens vs ≈ k for the CS butterfly — the
butterfly's win is a constant factor in depth, not in count.

Behavior change vs the old benchmark_circuit.py: depth is now
fourier.decompositions.parallel_depth — the longest path in the true
dependency DAG — whereas the old script's greedy layer packing could place
non-commuting rotations in the same layer and so understated the Givens
depth (it reported both methods near O(k)).  Depth here also counts plane
rotations only; the diag(±1) sign layer is excluded.

Outputs: data/givens_vs_cs.csv and data/plots/circuit_scaling.png.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from yungdiagram import YoungDiagram

from fourier import a_matrix, givens_factor, staircase
from fourier.decompositions import cs_butterfly, parallel_depth


def random_partition(n: int, rng: np.random.Generator) -> list[int]:
    parts, remaining = [], n
    while remaining > 0:
        p = int(rng.integers(1, remaining + 1))
        parts.append(p)
        remaining -= p
    return sorted(parts, reverse=True)


def collect(n_samples: int, max_n: int, seed: int) -> list[dict]:
    """Decompose the A-matrix of every sampled partition both ways.

    The sample is every staircase of size ≤ max_n plus `n_samples` random
    partitions of random size in [2, max_n], deduplicated.
    """
    rng = np.random.default_rng(seed)

    candidates: list[tuple[str, list[int]]] = []
    t = 1
    while t * (t + 1) // 2 <= max_n:
        candidates.append(("staircase", list(staircase(t).partition)))
        t += 1
    for _ in range(n_samples):
        n = int(rng.integers(2, max_n + 1))
        candidates.append(("random", random_partition(n, rng)))

    results = []
    seen: set[tuple[int, ...]] = set()
    for kind, part in candidates:
        key = tuple(part)
        if key in seen:
            continue
        seen.add(key)

        A = a_matrix(YoungDiagram(part))
        k = A.shape[0]
        if k < 2:
            continue

        g_giv, _ = givens_factor(A)
        g_cs, _ = cs_butterfly(A)
        results.append(
            dict(
                kind=kind,
                partition=part,
                n=sum(part),
                k=k,
                giv_gates=len(g_giv),
                giv_depth=parallel_depth(g_giv),
                cs_gates=len(g_cs),
                cs_depth=parallel_depth(g_cs),
            )
        )
    return results


def fit_power(x, y, label: str = "") -> tuple[float, float]:
    def model(x, a, b):
        return a * np.asarray(x, float) ** b

    (a, b), _ = curve_fit(model, x, y, p0=[1.0, 1.5], maxfev=5000)
    print(f"  {label:30s}  {a:.3f} · k^{b:.3f}")
    return float(a), float(b)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=800,
                        help="random partitions to draw (default 800)")
    parser.add_argument("--max-n", type=int, default=60,
                        help="largest partition size (default 60)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Collecting data  (samples={args.samples}, max_n={args.max_n}) …")
    data = collect(args.samples, args.max_n, args.seed)
    print(f"  {len(data)} unique partitions with k ≥ 2\n")

    csv_path = Path("data/givens_vs_cs.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["kind", "partition", "n", "k",
              "giv_gates", "giv_depth", "cs_gates", "cs_depth"]
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for d in data:
            writer.writerow({**d, "partition": str(tuple(d["partition"]))})
    print(f"Wrote {csv_path}\n")

    ks = np.array([d["k"] for d in data])
    color_n = np.array([d["n"] for d in data])

    def pw(x, a, b):
        return a * np.asarray(x, float) ** b

    print("Power-law fits  (y = a · k^b):")
    fit_gates_giv = fit_power(ks, [d["giv_gates"] for d in data], "Givens  gate count")
    fit_gates_cs = fit_power(ks, [d["cs_gates"] for d in data], "CS      gate count")
    fit_depth_giv = fit_power(ks, [d["giv_depth"] for d in data], "Givens  depth")
    fit_depth_cs = fit_power(ks, [d["cs_depth"] for d in data], "CS      depth")

    k_range = np.linspace(ks.min(), ks.max(), 300)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("A-matrix circuit decomposition: Givens vs CS butterfly")

    panels = [
        ("giv_gates", "cs_gates", fit_gates_giv, fit_gates_cs,
         "2-qubit gate count", "Gate count",
         (k_range * (k_range - 1) / 2, "k(k−1)/2")),
        ("giv_depth", "cs_depth", fit_depth_giv, fit_depth_cs,
         "circuit depth (parallel layers)", "Circuit depth",
         (k_range, "O(k)")),
    ]
    for ax, (giv_key, cs_key, fit_giv, fit_cs, ylabel, title, ref) in zip(axes, panels):
        sc = ax.scatter(ks, [d[giv_key] for d in data],
                        c=color_n, cmap="Blues", alpha=0.5, s=16, label="Givens")
        ax.scatter(ks, [d[cs_key] for d in data],
                   c=color_n, cmap="Oranges", alpha=0.5, s=16, label="CS butterfly")
        plt.colorbar(sc, ax=ax, label="partition size n")

        a, b = fit_giv
        ax.plot(k_range, pw(k_range, a, b), "b-", lw=2,
                label=f"Givens fit: {a:.2f}·k^{b:.2f}")
        a, b = fit_cs
        ax.plot(k_range, pw(k_range, a, b), "r--", lw=2,
                label=f"CS fit: {a:.2f}·k^{b:.2f}")
        ax.plot(k_range, ref[0], "k:", lw=1, alpha=0.4, label=ref[1])

        ax.set_xlabel("k  (A-matrix dimension)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.set_xscale("log")
        ax.set_yscale("log")

    plt.tight_layout()
    out = Path("data/plots/circuit_scaling.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved to {out}")


if __name__ == "__main__":
    main()
