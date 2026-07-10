#!/usr/bin/env python3
"""Compile A-matrix Givens circuits with Qiskit and BQSKit and fit CX vs k.

Question: does handing the naive k(k−1)/2-Givens circuit of an A-matrix to a
state-of-the-art compiler break the O(k²) two-qubit-gate barrier?

Supports report.md, Finding 2: Qiskit (u3+cx and rx/ry/rz+cx bases) and
BQSKit (optimization level 3) both compile the Givens circuit at ≈ a·k² CX.

Expected result: power-law fits of compiled CX count vs k have exponent
close to 2 for every compiler/basis — no compiler gets a free lunch.

The report numbers came from --max-k 15 --samples 300 --bqskit-samples 3
with BQSKit run at every k; the defaults here are much smaller (and BQSKit
is capped at --bqskit-max-k) so a bare run finishes in about a minute.
Behavior change vs the old compile_benchmark.py: the compiled-depth panel
was dropped — the library benchmark helpers report gate counts only.

Outputs: data/compile_benchmark.csv and data/plots/compiled_scaling.png.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from yungdiagram import YoungDiagram

from fourier import a_matrix, givens_factor
from fourier.circuits import bqskit_counts, givens_circuit, transpiled_counts

BASIS = {
    "u3+cx": ("u3", "cx"),
    "rx/ry/rz+cx": ("rx", "ry", "rz", "cx"),
}

STYLE = {
    "u3+cx": {"color": "steelblue", "ls": "-"},
    "rx/ry/rz+cx": {"color": "darkorange", "ls": "--"},
    "bqskit": {"color": "mediumseagreen", "ls": "-."},
}


def partitions_with_k(target_k: int, n_wanted: int, seed: int = 0) -> list[list[int]]:
    """Up to `n_wanted` partitions whose A-matrix has dimension `target_k`.

    A partition has target_k addable cells iff it has target_k−1 distinct part
    values.  The staircase (target_k−1, …, 1) is always included; the rest are
    random partitions of n near the staircase size, where hits are likely.
    """
    rng = random.Random(seed + target_k * 997)
    sc = list(range(target_k - 1, 0, -1))
    found = {tuple(sc)}
    result = [sc]

    n_lo = max(target_k, target_k * (target_k - 1) // 2)
    n_hi = max(n_lo * 4, 30)

    for _ in range(2_000_000):
        if len(result) >= n_wanted:
            break
        n = rng.randint(n_lo, n_hi)
        remaining, parts = n, []
        while remaining > 0:
            p = rng.randint(1, remaining)
            parts.append(p)
            remaining -= p
        part = tuple(sorted(parts, reverse=True))
        if part in found or len(set(part)) != target_k - 1:
            continue
        found.add(part)
        result.append(list(part))

    return result[:n_wanted]


def givens_qiskit_circuit(partition: list[int]):
    """The one-hot Givens circuit of A(partition), plus its dimension k."""
    A = a_matrix(YoungDiagram(partition))
    gates, signs = givens_factor(A)
    return givens_circuit(A.shape[0], gates, signs), A.shape[0]


def split_counts(counts: dict[str, int]) -> tuple[int, int]:
    """(cx, single-qubit) gate counts from a count_ops dict."""
    cx = counts.get("cx", 0)
    return cx, sum(v for name, v in counts.items() if name != "cx")


def collect(args) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = {name: [] for name in BASIS}
    rows["bqskit"] = []

    for k in range(2, args.max_k + 1):
        parts = partitions_with_k(k, args.samples, seed=args.seed)
        print(f"  k={k:3d}  ({len(parts)} partitions)", flush=True)

        for name, basis in BASIS.items():
            samples = []
            for part in parts:
                qc, _ = givens_qiskit_circuit(part)
                cx, sq = split_counts(
                    transpiled_counts(qc, basis, args.qiskit_opt_level)
                )
                samples.append((cx, sq))
            rows[name].append(
                {
                    "k": k,
                    "cx": np.mean([s[0] for s in samples]),
                    "sq": np.mean([s[1] for s in samples]),
                    "total": np.mean([s[0] + s[1] for s in samples]),
                }
            )

    if not args.skip_bqskit:
        bqskit_max_k = min(args.bqskit_max_k, args.max_k)
        print(f"\nBQSKit opt_level=3  ({args.bqskit_samples} samples/k, "
              f"k ≤ {bqskit_max_k}) …")
        for k in range(2, bqskit_max_k + 1):
            parts = partitions_with_k(k, args.bqskit_samples, seed=args.seed)
            print(f"  bqskit k={k:3d}  ({len(parts)} partitions) ",
                  end="", flush=True)
            samples = []
            for part in parts:
                qc, _ = givens_qiskit_circuit(part)
                try:
                    cx, sq = split_counts(bqskit_counts(qc))
                except Exception as e:  # a single BQSKit failure shouldn't kill the run
                    print(f"[err: {e}]", end="", flush=True)
                    continue
                samples.append((cx, sq))
                print(".", end="", flush=True)
            if samples:
                rows["bqskit"].append(
                    {
                        "k": k,
                        "cx": np.mean([s[0] for s in samples]),
                        "sq": np.mean([s[1] for s in samples]),
                        "total": np.mean([s[0] + s[1] for s in samples]),
                    }
                )
            print()

    return rows


def fit_power(ks, ys, label: str = "") -> tuple[float, float] | None:
    def model(x, a, b):
        return a * np.asarray(x, float) ** b

    if len(ks) < 2:
        print(f"  {label:35s}  too few points to fit")
        return None
    (a, b), _ = curve_fit(model, ks, ys, p0=[1.0, 2.0], maxfev=5000)
    print(f"  {label:35s}  {a:.3f} · k^{b:.3f}")
    return float(a), float(b)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-k", type=int, default=8,
                        help="largest A-matrix dimension (default 8; report used 15)")
    parser.add_argument("--samples", type=int, default=20,
                        help="Qiskit sample partitions per k (default 20; report used 300)")
    parser.add_argument("--bqskit-samples", type=int, default=1,
                        help="BQSKit sample partitions per k (default 1; report used 3)")
    parser.add_argument("--bqskit-max-k", type=int, default=4,
                        help="largest k handed to BQSKit (default 4 — BQSKit is slow)")
    parser.add_argument("--skip-bqskit", action="store_true",
                        help="skip the BQSKit pass entirely")
    parser.add_argument("--qiskit-opt-level", type=int, default=1,
                        help="Qiskit transpile optimization level (default 1, as in the report run)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"Compiling (max_k={args.max_k}, {args.samples} qiskit samples/k, "
          f"bqskit={'off' if args.skip_bqskit else args.bqskit_samples}) …\n")
    data = collect(args)

    csv_path = Path("data/compile_benchmark.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w") as fh:
        fh.write("compiler,k,cx,sq,total\n")
        for name, rows in data.items():
            for r in rows:
                fh.write(f"{name},{r['k']},{r['cx']},{r['sq']},{r['total']}\n")
    print(f"\nWrote {csv_path}")

    print("\nPower-law fits  (y = a · k^b):")
    fits: dict[str, dict[str, tuple[float, float] | None]] = {}
    for name, rows in data.items():
        if not rows:
            continue
        ks = [r["k"] for r in rows]
        fits[name] = {
            metric: fit_power(ks, [r[metric] for r in rows], f"{name}  {metric}")
            for metric in ("cx", "total")
        }

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Compiled A-matrix circuit: Givens → native gates")
    k_ref = np.linspace(2, args.max_k, 300)

    panels = [
        ("cx", "CX count", "CX (CNOT) count",
         (3 * k_ref * (k_ref - 1) / 2, "3·k(k−1)/2  (CX bound)")),
        ("total", "Total gates (SQ+CX)", "Total gate count", None),
    ]
    for ax, (metric, ylabel, title, ref) in zip(axes, panels):
        for name, rows in data.items():
            if not rows:
                continue
            ks = [r["k"] for r in rows]
            ys = [r[metric] for r in rows]
            c, ls = STYLE[name]["color"], STYLE[name]["ls"]
            ax.scatter(ks, ys, color=c, alpha=0.8, s=40, zorder=3, label=name)
            ab = fits.get(name, {}).get(metric)
            if ab is not None:
                ax.plot(k_ref, ab[0] * k_ref ** ab[1], color=c, ls=ls, lw=2,
                        label=f"{name}: {ab[0]:.2f}·k^{ab[1]:.2f}")
        if ref is not None:
            ax.plot(k_ref, ref[0], "k:", lw=1, alpha=0.35, label=ref[1])
        ax.set_xlabel("k  (A-matrix dimension)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7)

    plt.tight_layout()
    out = Path("data/plots/compiled_scaling.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
