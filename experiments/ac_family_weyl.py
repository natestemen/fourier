"""Weyl-plane trajectory of the AC family (equal block widths and heights).

Question: where does the maximally symmetric "AC family" of 4-addable
diagrams — addable contents {n, n/3, −n/3, −n} for n divisible by 3, i.e.
block parameters w1=w2=w3=h1=h2=h3=n/3 — sit in the (b, |c|) Weyl plane, and
what is its n→∞ limit?

Supports report.md, Finding 1: a = π/4 for every 4-addable A-matrix, with
(b, c) varying continuously — the AC family is a distinguished curve inside
the scatter of random diagrams, converging to an interior limit point.

Expected result: the family traces a short curve converging to the limit
b ≈ 0.2527, |c| ≈ 0.0889 (Weyl coordinates of the exact m→∞ limit matrix),
against a background of randomly sampled generic diagrams.

Family (n must be divisible by 3, m = n/3):
  AC = {(n, 0), (2n/3, n/3), (n/3, 2n/3), (0, n)}
  RC = {(n-1, n/3-1), (2n/3-1, 2n/3-1), (n/3-1, n-1)}
Row lengths: 3m (m rows), 2m (m rows), m (m rows).

Uses the symbolic block parametrization (fourier.amatrix.a_matrix_generic4)
instead of exact diagrams for speed, exactly as the original did. Outputs:

  - data/plots/ac_family_weyl_plane.png
  - data/ac_family_weyl.csv

Replaces plot_ac_family_weyl.py. Behavior changes: the dead skipped_invalid
counter is dropped; --step is now honored (the original parsed it but
hardcoded a step of 3); n not divisible by 3 is skipped instead of crashing;
the n→∞ limit point is marked on the plot; no plt.show().
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from fourier.amatrix import a_matrix_generic4
from fourier.weyl import weyl_coordinates

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
PLOT_DIR = DATA_DIR / "plots"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-min", type=int, default=3, help="Smallest n (multiple of 3).")
    parser.add_argument("--n-max", type=int, default=60, help="Largest n (multiple of 3).")
    parser.add_argument("--step", type=int, default=3, help="Step for n (default: 3).")
    parser.add_argument(
        "--max-size",
        type=int,
        default=40,
        help="Max diagram size for random sampling.",
    )
    parser.add_argument(
        "--random-count",
        type=int,
        default=150,
        help="Number of random diagrams to plot (0 to skip).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (0 for random).")
    parser.add_argument(
        "--output",
        type=Path,
        default=PLOT_DIR / "ac_family_weyl_plane.png",
        help="Output path for the (b,|c|) plot.",
    )
    return parser.parse_args()


def random_w_triples(max_size: int, w_min: int) -> list[tuple[int, int, int]]:
    """All (w1, w2, w3) with w_i >= w_min fitting a diagram of <= max_size cells
    at minimal heights."""
    triples: list[tuple[int, int, int]] = []
    for w3 in range(w_min, max_size // 3 + 1):
        max_w2 = (max_size - 3 * w3) // 2
        for w2 in range(w_min, max_w2 + 1):
            max_w1 = max_size - 2 * w2 - 3 * w3
            for w1 in range(w_min, max_w1 + 1):
                triples.append((w1, w2, w3))
    return triples


def random_h_tuple(
    rng: random.Random,
    w1: int,
    w2: int,
    w3: int,
    max_size: int,
    h_min: int,
) -> tuple[int, int, int] | None:
    """Random (h1, h2, h3) with h_i >= h_min keeping the diagram size <= max_size."""
    denom1 = w1 + w2 + w3
    denom2 = w2 + w3
    denom3 = w3

    h1_max = (max_size - denom2 * h_min - denom3 * h_min) // denom1
    if h1_max < h_min:
        return None
    h1 = rng.randint(h_min, h1_max)

    h2_max = (max_size - denom1 * h1 - denom3 * h_min) // denom2
    if h2_max < h_min:
        return None
    h2 = rng.randint(h_min, h2_max)

    h3_max = (max_size - denom1 * h1 - denom2 * h2) // denom3
    if h3_max < h_min:
        return None
    h3 = rng.randint(h_min, h3_max)

    return h1, h2, h3


def random_params(
    max_size: int, count: int, seed: int
) -> list[tuple[int, int, int, int, int, int]]:
    """Up to `count` distinct generic (w, h) configurations of size <= max_size."""
    h_min = 2
    w_min = 2
    triples = random_w_triples(max_size, w_min)
    if not triples:
        return []
    rng = random.Random() if seed == 0 else random.Random(seed)
    params: set[tuple[int, int, int, int, int, int]] = set()
    max_tries = max(500, count * 50)
    tries = 0
    while len(params) < count and tries < max_tries:
        tries += 1
        w1, w2, w3 = rng.choice(triples)
        h_tuple = random_h_tuple(rng, w1, w2, w3, max_size, h_min)
        if h_tuple is None:
            continue
        h1, h2, h3 = h_tuple
        params.add((w1, w2, w3, h1, h2, h3))
    return list(params)


def main() -> None:
    args = parse_args()

    print("AC family:")
    print("  AC = {(n, 0), (2n/3, n/3), (n/3, 2n/3), (0, n)}")
    print("  RC = {(n-1, n/3-1), (2n/3-1, 2n/3-1), (n/3-1, n-1)}")

    A_sym, symbols = a_matrix_generic4()
    A_func = sp.lambdify(symbols, A_sym, "numpy")

    family_rows: list[dict[str, float | int]] = []
    for n in range(args.n_min, args.n_max + 1, args.step):
        if n % 3 != 0:
            continue
        m = n // 3
        A_val = np.array(A_func(m, m, m, m, m, m), dtype=complex)
        a, b, c = weyl_coordinates(A_val)
        family_rows.append({"n": n, "m": m, "weyl_a": a, "weyl_b": b, "weyl_c": c})

    if not family_rows:
        raise SystemExit("No valid family points computed; adjust --n-min/--n-max.")

    # Limit of the family as n -> infinity (m = n/3 -> infinity).
    m_sym = sp.symbols("m", positive=True, integer=True)
    A_family = A_sym.subs(dict.fromkeys(symbols, m_sym))
    A_lim = A_family.applyfunc(lambda expr: sp.limit(expr, m_sym, sp.oo))
    print("Limit matrix (n -> inf):")
    sp.pprint(A_lim, use_unicode=False)
    a_lim, b_lim, c_lim = weyl_coordinates(np.array(A_lim.evalf(), dtype=complex))
    print(f"Limit Weyl: a={a_lim:.6g} b={b_lim:.6g} c={c_lim:.6g} |c|={abs(c_lim):.6g}")

    random_points: list[tuple[float, float]] = []
    if args.random_count > 0:
        samples = random_params(args.max_size, args.random_count, args.seed)
        if not samples:
            print("Warning: no random parameter samples fit the max-size constraint.")
        elif len(samples) < args.random_count:
            print(f"Warning: only {len(samples)} random samples fit the max size.")
        for params in samples:
            A_val = np.array(A_func(*params), dtype=complex)
            _, b, c = weyl_coordinates(A_val)
            random_points.append((b, abs(c)))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / "ac_family_weyl.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "m", "weyl_a", "weyl_b", "weyl_c"])
        writer.writeheader()
        writer.writerows(family_rows)
    print(f"Saved family series to {csv_path}")

    family_b = np.array([row["weyl_b"] for row in family_rows], dtype=float)
    family_c = np.array([abs(row["weyl_c"]) for row in family_rows], dtype=float)

    fig, ax = plt.subplots()
    pi4 = 0.25 * np.pi
    ax.plot([0, pi4], [0, 0], color="black", linewidth=1)
    ax.plot([0, pi4], [0, pi4], color="black", linewidth=1)
    ax.plot([pi4, pi4], [0, pi4], color="black", linewidth=1)

    if random_points:
        xs = [p[0] for p in random_points]
        ys = [p[1] for p in random_points]
        ax.scatter(xs, ys, s=12, alpha=0.35, color="gray", label="random diagrams")

    ax.plot(family_b, family_c, color="tab:blue", linewidth=1.5, label="AC family")
    ax.scatter(family_b, family_c, color="tab:blue", s=20)
    ax.scatter(
        [b_lim], [abs(c_lim)], marker="*", s=120, color="tab:blue",
        edgecolors="black", linewidths=0.6, label="AC limit n→∞",
    )

    ax.set_xlabel("b")
    ax.set_ylabel("|c|")
    ax.set_title("AC family on Weyl plane")
    ax.grid(True, alpha=0.3)
    ax.legend()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
