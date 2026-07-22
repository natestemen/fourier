"""ULV counts and HSS ranks across a diverse family of Young-diagram shapes.

Question: do the near-linear ULV rotation counts (RESULTS.md §10) hold
across the space of diagram shapes, or only for the staircase?

Content gaps <-> (width, height) blocks bijectively, so gap profiles cover
the space of diagrams with k addable cells (bijection verified in
tests/test_amatrix.py at small k).  Profiles:
  staircase : all gaps 1 (minimal diagram, clustered contents)
  u3        : gaps uniform in {1..3} (the note's 'generic')
  u10       : gaps uniform in {1..10} (heterogeneous blocks)
  heavy     : geometric gaps, occasional huge blocks (heavy-tailed shapes)
  cluster   : two staircase arms separated by one gap of 200 (extreme void)
  comb      : alternating gaps 1 and 30 (wide flat blocks)

Expected result: max HSS rank 7-11 for every instance; rotation counts
within a few percent of the staircase at every k, with the extreme shapes
slightly cheaper (larger gaps = better-separated Cauchy nodes) - the
staircase is the observed worst case.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from hss_structure import hss_ranks
from ulv_circuit import factorization_error, ulv_factor

from fourier.amatrix import _a_matrix_from_contents_log


def gaps(profile: str, k: int, rng) -> np.ndarray:
    n = 2 * k - 2
    if profile == "staircase":
        return np.ones(n, dtype=int)
    if profile == "u3":
        return rng.integers(1, 4, size=n)
    if profile == "u10":
        return rng.integers(1, 11, size=n)
    if profile == "heavy":
        return rng.geometric(0.15, size=n)  # mean ~6.7, tail to ~50+
    if profile == "cluster":
        g = np.ones(n, dtype=int)
        g[n // 2] = 200
        return g
    if profile == "comb":
        g = np.ones(n, dtype=int)
        g[::2] = 30
        return g
    raise ValueError(profile)


def build(profile: str, k: int, rng) -> np.ndarray:
    seq = np.concatenate([[0.0], np.cumsum(gaps(profile, k, rng))])[::-1]
    return _a_matrix_from_contents_log(seq[0::2], seq[1::2])


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--delta", type=float, default=1e-4)
    parser.add_argument("--leaf", type=int, default=32)
    args = parser.parse_args()

    delta = args.delta
    profiles = ["staircase", "u3", "u10", "heavy", "cluster", "comb"]
    seeds = {k: (4 if k <= 256 else 3 if k <= 512 else 1) for k in args.sizes}

    print(f"delta={delta}, b={args.leaf}")
    print(f"{'profile':>10} {'k':>5} {'runs':>4} {'maxHSSrank':>10} "
          f"{'rotations (min..max)':>22} {'ratio':>6} {'err(max)':>9}")
    for k in args.sizes:
        naive = k * (k - 1) // 2
        for prof in profiles:
            n_runs = 1 if prof in ("staircase", "cluster", "comb") else seeds[k]
            counts, errs, hranks = [], [], []
            for seed in range(n_runs):
                rng = np.random.default_rng(100 + seed)
                A = build(prof, k, rng)
                hranks.append(max(r for _, r in hss_ranks(A, 1e-3).values()))
                ops, Mf, count, _ = ulv_factor(A, b=args.leaf, delta=delta)
                counts.append(count)
                errs.append(factorization_error(Mf))
            print(f"{prof:>10} {k:>5} {len(counts):>4} {max(hranks):>10} "
                  f"{min(counts):>10,} ..{max(counts):>9,} {max(counts) / naive:>6.2f} {max(errs):>9.1e}")
        print()


if __name__ == "__main__":
    main()
