#!/usr/bin/env python3
"""
compile_benchmark.py — transpile A-matrix Givens circuits to native gate sets
and fit compiled gate count / depth vs A-matrix dimension k.

Each A-matrix is decomposed into k(k-1)/2 Givens (2-qubit) rotations, then
the circuit is transpiled by Qiskit to two basis sets:
  • u3 + cx       (IBM-style: arbitrary single-qubit + CNOT)
  • rx/ry/rz + cx (Euler-angle single-qubit + CNOT)

Each 2-qubit Givens gate can require up to 3 CX, so the hard upper bound is
3·k(k−1)/2 CX gates.  Qiskit may cancel some after optimization.

Usage:
    python3 compile_benchmark.py              # k=2..15, 300 Qiskit / 3 BQSKit samples
    python3 compile_benchmark.py 20 8         # max_k  qiskit_samples_per_k
    python3 compile_benchmark.py 20 8 5       # max_k  qiskit_samples  bqskit_samples
    python3 compile_benchmark.py 15 300 0     # disable BQSKit (bqskit_samples=0)
"""

import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from qiskit import transpile as qk_transpile

from symbolic_a_matrix import build_symbolic_a_matrix_for_partition, _partition_addable_cells
from a_matrix_circuit import decompose, to_qiskit_circuit


# ── helpers ────────────────────────────────────────────────────────────────────

def n_addable(partition):
    return len(_partition_addable_cells(partition))


def staircase(t):
    return list(range(t, 0, -1))


def partitions_with_k(target_k, n_wanted=5, seed=0):
    """Return up to n_wanted partitions whose A-matrix has dimension target_k."""
    rng = random.Random(seed + target_k * 997)
    found = set()
    result = []

    # Staircase (t, t-1, ..., 1) always has t+1 addable cells
    sc = staircase(target_k - 1)
    if sc:
        try:
            if n_addable(sc) == target_k:
                key = tuple(sc)
                found.add(key)
                result.append(list(sc))
        except Exception:
            pass

    # For large k, random partitions of small n almost never have k addable cells.
    # The staircase (k-1,...,1) has n = k(k-1)/2, so centre the search there.
    n_lo = max(target_k, target_k * (target_k - 1) // 2)
    n_hi = max(n_lo * 4, 30)

    for _ in range(2_000_000):
        if len(result) >= n_wanted:
            break
        n = rng.randint(n_lo, n_hi)
        rem, parts = n, []
        while rem > 0:
            p = rng.randint(1, rem)
            parts.append(p)
            rem -= p
        part = tuple(sorted(parts, reverse=True))
        if part in found:
            continue
        try:
            if n_addable(list(part)) == target_k:
                found.add(part)
                result.append(list(part))
        except Exception:
            pass

    return result


# ── compilation ────────────────────────────────────────────────────────────────

BASIS = {
    "u3+cx":       ["u3", "cx"],
    "rx/ry/rz+cx": ["rx", "ry", "rz", "cx"],
}

BQSKIT_OPT_LEVEL = 3


def compile_partition_bqskit(partition):
    from bqskit import compile as bqskit_compile
    from bqskit.ext import qiskit_to_bqskit

    A = np.array(
        build_symbolic_a_matrix_for_partition(partition).tolist(), dtype=float
    )
    k = A.shape[0]
    gates = decompose(A)
    qc = to_qiskit_circuit(gates, k, [f"q{i}" for i in range(k)])
    bqc = qiskit_to_bqskit(qc)
    compiled = bqskit_compile(bqc, optimization_level=BQSKIT_OPT_LEVEL)
    cx = sum(1 for op in compiled.operations() if op.num_qudits == 2)
    sq = sum(1 for op in compiled.operations() if op.num_qudits == 1)
    return {"k": k, "cx": cx, "sq": sq, "total": cx + sq, "depth": compiled.depth}


def compile_partition(partition, basis_gates, opt_level=1):
    A = np.array(
        build_symbolic_a_matrix_for_partition(partition).tolist(), dtype=float
    )
    k = A.shape[0]
    gates = decompose(A)
    qc = to_qiskit_circuit(gates, k, [f"q{i}" for i in range(k)])
    tqc = qk_transpile(qc, basis_gates=basis_gates, optimization_level=opt_level)
    ops = tqc.count_ops()
    cx = ops.get("cx", 0)
    sq = sum(v for name, v in ops.items() if name != "cx")
    return {"k": k, "cx": cx, "sq": sq, "total": cx + sq, "depth": tqc.depth()}


# ── data collection ────────────────────────────────────────────────────────────

def collect(max_k=16, samples_per_k=5, bqskit_samples_per_k=3):
    rows = {name: [] for name in BASIS}
    rows["bqskit"] = []

    # ── Qiskit basis sets ──────────────────────────────────────────────────────
    for k in range(2, max_k + 1):
        parts = partitions_with_k(k, n_wanted=samples_per_k)
        if not parts:
            continue
        print(f"  k={k:3d}  ({len(parts)} partitions) ", end="", flush=True)

        for name, bg in BASIS.items():
            compiled = []
            for i, p in enumerate(parts):
                if i % 50 == 49:
                    print(".", end="", flush=True)
                try:
                    compiled.append(compile_partition(p, bg))
                except Exception:
                    pass
            if compiled:
                rows[name].append({
                    "k":     k,
                    "cx":    np.mean([r["cx"]    for r in compiled]),
                    "sq":    np.mean([r["sq"]    for r in compiled]),
                    "total": np.mean([r["total"] for r in compiled]),
                    "depth": np.mean([r["depth"] for r in compiled]),
                })

        print()

    # ── BQSKit (opt level 3, fewer samples — ~15 min per circuit) ─────────────
    if bqskit_samples_per_k > 0:
        print(f"\nBQSKit opt_level={BQSKIT_OPT_LEVEL}  ({bqskit_samples_per_k} samples/k) …")
        for k in range(2, max_k + 1):
            parts = partitions_with_k(k, n_wanted=bqskit_samples_per_k)
            if not parts:
                continue
            print(f"  bqskit k={k:3d}  ({len(parts)} partitions) ", end="", flush=True)
            compiled = []
            for p in parts:
                try:
                    compiled.append(compile_partition_bqskit(p))
                    print(".", end="", flush=True)
                except Exception as e:
                    print(f"[err: {e}]", end="", flush=True)
            if compiled:
                rows["bqskit"].append({
                    "k":     k,
                    "cx":    np.mean([r["cx"]    for r in compiled]),
                    "sq":    np.mean([r["sq"]    for r in compiled]),
                    "total": np.mean([r["total"] for r in compiled]),
                    "depth": np.mean([r["depth"] for r in compiled]),
                })
            print()

    return rows


# ── fitting ────────────────────────────────────────────────────────────────────

def fit_power(ks, ys, label=""):
    def model(x, a, b):
        return a * np.asarray(x, float) ** b
    try:
        (a, b), _ = curve_fit(model, ks, ys, p0=[1.0, 2.0], maxfev=5000)
        print(f"  {label:35s}  {a:.3f} · k^{b:.3f}")
        return float(a), float(b)
    except Exception as e:
        print(f"  {label:35s}  fit failed: {e}")
        return None, None


# ── plotting ───────────────────────────────────────────────────────────────────

STYLE = {
    "u3+cx":       {"color": "steelblue",  "ls": "-"},
    "rx/ry/rz+cx": {"color": "darkorange", "ls": "--"},
    "bqskit":      {"color": "mediumseagreen", "ls": "-."},
}


def main():
    args                 = sys.argv[1:]
    max_k                = int(args[0]) if len(args) > 0 else 15
    samples_per_k        = int(args[1]) if len(args) > 1 else 300
    bqskit_samples_per_k = int(args[2]) if len(args) > 2 else 3

    print(f"Compiling (max_k={max_k}, {samples_per_k} samples/k, "
          f"bqskit={bqskit_samples_per_k} samples/k) …\n")
    data = collect(max_k, samples_per_k, bqskit_samples_per_k)

    print("\nPower-law fits  (y = a · k^b):")
    fits = {}
    for name, rows in data.items():
        if not rows:
            continue
        ks = [r["k"] for r in rows]
        fits[name] = {}
        for metric in ("cx", "total", "depth"):
            ys = [r[metric] for r in rows]
            fits[name][metric] = fit_power(ks, ys, f"{name}  {metric}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Compiled A-matrix circuit: Givens → native gates")

    metrics     = ["cx",         "total",               "depth"]
    ylabels     = ["CX count",   "Total gates (SQ+CX)", "Depth"]
    titles      = ["CX (CNOT) count", "Total gate count", "Circuit depth"]

    k_ref = np.linspace(2, max_k, 300)
    refs  = [
        # CX upper bound: 3 Givens per 2q gate, k(k-1)/2 Givens
        (3 * k_ref * (k_ref - 1) / 2, "3·k(k−1)/2  (CX bound)"),
        (None, None),
        (k_ref, "O(k)"),
    ]

    for ax, metric, ylabel, title, (ref_y, ref_lbl) in zip(
        axes, metrics, ylabels, titles, refs
    ):
        for name, rows in data.items():
            if not rows:
                continue
            ks = [r["k"] for r in rows]
            ys = [r[metric] for r in rows]
            c  = STYLE[name]["color"]
            ls = STYLE[name]["ls"]
            ax.scatter(ks, ys, color=c, alpha=0.8, s=40, zorder=3, label=name)
            ab = fits.get(name, {}).get(metric, (None, None))
            if ab[0]:
                ax.plot(k_ref, ab[0] * k_ref ** ab[1], color=c, ls=ls, lw=2,
                        label=f"{name}: {ab[0]:.2f}·k^{ab[1]:.2f}")

        if ref_y is not None:
            ax.plot(k_ref, ref_y, "k:", lw=1, alpha=0.35, label=ref_lbl)

        pass
        ax.set_xlabel("k  (A-matrix dimension)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7)

    plt.tight_layout()
    out = "data/plots/compiled_scaling.png"
    plt.savefig(out, dpi=150)
    print(f"\nSaved to {out}")
    plt.show()


if __name__ == "__main__":
    main()
