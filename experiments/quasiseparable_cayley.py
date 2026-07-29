"""Numerical quasiseparable ranks of A and its skew Cayley generator.

This strengthens the dyadic HSS rank experiment in ``hss_structure.py``:
for every split ``i`` it measures the lower-left block ``M[i:, :i]``.  The
largest numerical rank is the epsilon-quasiseparable order.  Unitary
rank-structured matrices of order r admit Givens-weight representations with
O(k r) parameters/rotations, so bounded ranks here point to a direct
near-linear circuit rather than level-by-level ULV.

The fixed reflection D = diag(1, -1, ..., -1) puts A D in the Cayley chart.
Its generator S is exactly skew-symmetric.  A skew-structured approximation
to S maps back to an exactly orthogonal approximation of A, avoiding the
nonunitarity introduced by truncating A's off-diagonal blocks independently.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from fourier.amatrix import random_content_a_matrix, staircase_a_matrix
from fourier.structured import cayley_factorization


def maximum_cut_rank(matrix: np.ndarray, tolerance: float) -> tuple[int, int]:
    """Return maximum lower-left cut rank and the first split attaining it."""
    k = matrix.shape[0]
    best_rank = 0
    best_split = 0
    for split in range(1, k):
        singular_values = np.linalg.svd(
            matrix[split:, :split], compute_uv=False
        )
        rank = int(np.count_nonzero(singular_values > tolerance))
        if rank > best_rank:
            best_rank = rank
            best_split = split
    return best_rank, best_split


def collect_rows(
    sizes: list[int],
    tolerances: list[float],
    random_samples: int,
    seed: int,
) -> list[dict[str, int | float | str]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, int | float | str]] = []
    for k in sizes:
        matrices = [("staircase", 0, staircase_a_matrix(k))]
        matrices.extend(
            ("random-content", sample, random_content_a_matrix(k, rng))
            for sample in range(random_samples)
        )
        for family, sample, A in matrices:
            cayley = cayley_factorization(A)
            targets = (("A", A), ("cayley", cayley.generator))
            for tolerance in tolerances:
                for representation, matrix in targets:
                    rank, split = maximum_cut_rank(matrix, tolerance)
                    rows.append(
                        {
                            "k": k,
                            "family": family,
                            "sample": sample,
                            "representation": representation,
                            "tolerance": tolerance,
                            "max_cut_rank": rank,
                            "first_max_split": split,
                            "cayley_condition": cayley.gauge_condition,
                        }
                    )
            print(
                f"k={k:4d} {family:>14s}[{sample}]  "
                f"cond(I+AD)={cayley.gauge_condition:.3f}"
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument(
        "--tolerances", type=float, nargs="+", default=[1e-2, 1e-4, 1e-6, 1e-8]
    )
    parser.add_argument("--random-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = collect_rows(args.sizes, args.tolerances, args.random_samples, args.seed)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.output}")

    print("\nmaximum ranks")
    for row in rows:
        print(
            f"{row['k']:4d} {row['family']:>14s} {row['representation']:>7s} "
            f"eps={row['tolerance']:.0e} r={row['max_cut_rank']:2d} "
            f"split={row['first_max_split']:3d}"
        )


if __name__ == "__main__":
    main()
