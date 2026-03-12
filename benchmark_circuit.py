#!/usr/bin/env python3
"""
benchmark_circuit.py — compare Givens triangularization vs CS butterfly
decomposition of A-matrices, fitting gate-count / depth vs matrix size k.

Usage:
    python3 benchmark_circuit.py              # 800 samples, max partition size 60
    python3 benchmark_circuit.py 2000 80      # n_samples max_n
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.linalg import cossin as scipy_cossin

from symbolic_a_matrix import build_symbolic_a_matrix_for_partition
from a_matrix_circuit import decompose, Gate


# ── CS butterfly ───────────────────────────────────────────────────────────────

def decompose_cs(A: np.ndarray) -> list[Gate]:
    """
    Recursive Cosine-Sine butterfly decomposition.

    A = block_diag(U1,U2) @ CS @ block_diag(V1h,V2h)

    CS layer: k/2 disjoint rotations on pairs (i, i+k/2) — all parallelisable.
    U and V blocks act on disjoint halves, so they also run in parallel.

    Recurrences (equal split, k a power of 2):
      gate count:  T(k) = 4·T(k/2) + k/2  →  O(k²)
      depth:       T(k) = 2·T(k/2) + 1    →  O(k)

    Both are asymptotically the same as Givens, but structure may expose sparsity
    in the blocks for A-matrices with special symmetry.
    """
    gates: list[Gate] = []
    _cs_rec(A, list(range(A.shape[0])), gates)
    return gates


def _cs_rec(A: np.ndarray, idx: list[int], gates: list[Gate]) -> None:
    k = A.shape[0]

    if k == 1:
        if A[0, 0] < 0:
            gates.append(Gate(idx[0], idx[0], np.pi, "Z"))
        return

    # Odd or k=2: plain Givens
    if k == 2 or k % 2 != 0:
        for g in decompose(A):
            gates.append(Gate(idx[g.i], idx[g.j], g.theta, g.label))
        return

    p = k // 2
    (u1, u2), (c_arr, s_arr), (v1h, v2h) = scipy_cossin(A, p=p, q=p, separate=True)

    top = idx[:p]
    bot = idx[p:]

    # Vt first
    _cs_rec(v1h, top, gates)
    _cs_rec(v2h, bot, gates)

    # CS layer: p disjoint rotations
    for i in range(p):
        theta = np.arctan2(float(s_arr[i]), float(c_arr[i]))
        if abs(np.sin(theta)) > 1e-12:
            gates.append(Gate(top[i], bot[i], theta, "CS"))

    # U last
    _cs_rec(u1, top, gates)
    _cs_rec(u2, bot, gates)


# ── layer depth ────────────────────────────────────────────────────────────────

def circuit_depth(gates: list[Gate]) -> int:
    layer_used: list[set[int]] = []
    for g in gates:
        indices = {g.i} if g.i == g.j else {g.i, g.j}
        placed = False
        for used in layer_used:
            if not (used & indices):
                used |= indices
                placed = True
                break
        if not placed:
            layer_used.append(set(indices))
    return len(layer_used)


# ── partition generators ───────────────────────────────────────────────────────

def random_partition(n: int, rng: np.random.Generator) -> list[int]:
    parts, remaining = [], n
    while remaining > 0:
        p = int(rng.integers(1, remaining + 1))
        parts.append(p)
        remaining -= p
    return sorted(parts, reverse=True)


def staircase(t: int) -> list[int]:
    return list(range(t, 0, -1))


# ── data collection ────────────────────────────────────────────────────────────

def collect(n_samples: int = 800, max_n: int = 60, seed: int = 42) -> list[dict]:
    rng  = np.random.default_rng(seed)
    rows = []

    t = 1
    while t * (t + 1) // 2 <= max_n:
        rows.append(("staircase", staircase(t)))
        t += 1

    for _ in range(n_samples):
        n = int(rng.integers(2, max_n + 1))
        rows.append(("random", random_partition(n, rng)))

    results = []
    seen    = set()

    for kind, part in rows:
        key = tuple(part)
        if key in seen:
            continue
        seen.add(key)

        try:
            A_sym = build_symbolic_a_matrix_for_partition(part)
            A     = np.array(A_sym.tolist(), dtype=float)
            k     = A.shape[0]
            if k < 2:
                continue

            g_giv = decompose(A)
            g_cs  = decompose_cs(A)

            results.append(dict(
                kind=kind, partition=part, n=sum(part), k=k,
                giv_gates = len([g for g in g_giv if g.i != g.j]),
                giv_depth = circuit_depth(g_giv),
                cs_gates  = len([g for g in g_cs  if g.i != g.j]),
                cs_depth  = circuit_depth(g_cs),
            ))
        except Exception:
            pass

    return results


# ── fitting ────────────────────────────────────────────────────────────────────

def fit_power(x, y, label=""):
    def model(x, a, b):
        return a * np.array(x, dtype=float) ** b
    try:
        popt, _ = curve_fit(model, x, y, p0=[1.0, 1.5], maxfev=5000)
        a, b = float(popt[0]), float(popt[1])
        print(f"  {label:30s}  {a:.3f} · k^{b:.3f}")
        return a, b
    except Exception as e:
        print(f"  {label:30s}  fit failed: {e}")
        return None, None


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    args      = sys.argv[1:]
    n_samples = int(args[0]) if len(args) > 0 else 800
    max_n     = int(args[1]) if len(args) > 1 else 60

    print(f"Collecting data  (n_samples={n_samples}, max_n={max_n}) …")
    data = collect(n_samples, max_n)
    print(f"  {len(data)} unique partitions with k ≥ 2\n")

    ks = np.array([d["k"] for d in data])

    def pw(x, a, b):
        return a * np.array(x, dtype=float) ** b

    print("Power-law fits  (y = a · k^b):")
    ag_giv, bg_giv = fit_power(ks, [d["giv_gates"] for d in data], "Givens  gate count")
    ag_cs,  bg_cs  = fit_power(ks, [d["cs_gates"]  for d in data], "CS      gate count")
    ad_giv, bd_giv = fit_power(ks, [d["giv_depth"] for d in data], "Givens  depth")
    ad_cs,  bd_cs  = fit_power(ks, [d["cs_depth"]  for d in data], "CS      depth")

    k_range = np.linspace(ks.min(), ks.max(), 300)
    k_ref   = np.linspace(ks.min(), ks.max(), 300)

    color_n = np.array([d["n"] for d in data])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("A-matrix circuit decomposition: Givens vs CS butterfly")

    # ── gate count ─────────────────────────────────────────────────────────────
    ax = axes[0]
    sc = ax.scatter(ks, [d["giv_gates"] for d in data],
                    c=color_n, cmap="Blues", alpha=0.5, s=16, label="Givens")
    ax.scatter(ks, [d["cs_gates"] for d in data],
               c=color_n, cmap="Oranges", alpha=0.5, s=16, label="CS butterfly")
    plt.colorbar(sc, ax=ax, label="partition size n")

    if ag_giv:
        ax.plot(k_range, pw(k_range, ag_giv, bg_giv), "b-", lw=2,
                label=f"Givens fit: {ag_giv:.2f}·k^{bg_giv:.2f}")
    if ag_cs:
        ax.plot(k_range, pw(k_range, ag_cs, bg_cs), "r--", lw=2,
                label=f"CS fit: {ag_cs:.2f}·k^{bg_cs:.2f}")
    ax.plot(k_ref, k_ref * (k_ref - 1) / 2, "k:", lw=1, alpha=0.4, label="k(k−1)/2")

    ax.set_xlabel("k  (A-matrix dimension)")
    ax.set_ylabel("2-qubit gate count")
    ax.set_title("Gate count")
    ax.legend(fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")

    # ── depth ──────────────────────────────────────────────────────────────────
    ax = axes[1]
    sc = ax.scatter(ks, [d["giv_depth"] for d in data],
                    c=color_n, cmap="Blues", alpha=0.5, s=16, label="Givens")
    ax.scatter(ks, [d["cs_depth"] for d in data],
               c=color_n, cmap="Oranges", alpha=0.5, s=16, label="CS butterfly")
    plt.colorbar(sc, ax=ax, label="partition size n")

    if ad_giv:
        ax.plot(k_range, pw(k_range, ad_giv, bd_giv), "b-", lw=2,
                label=f"Givens fit: {ad_giv:.2f}·k^{bd_giv:.2f}")
    if ad_cs:
        ax.plot(k_range, pw(k_range, ad_cs, bd_cs), "r--", lw=2,
                label=f"CS fit: {ad_cs:.2f}·k^{bd_cs:.2f}")
    ax.plot(k_ref, k_ref, "k:", lw=1, alpha=0.4, label="O(k)")

    ax.set_xlabel("k  (A-matrix dimension)")
    ax.set_ylabel("circuit depth (parallel layers)")
    ax.set_title("Circuit depth")
    ax.legend(fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")

    plt.tight_layout()
    out = "data/plots/circuit_scaling.png"
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved to {out}")
    plt.show()


if __name__ == "__main__":
    main()