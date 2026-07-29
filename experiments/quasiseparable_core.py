"""Sequential quasiseparable factorization by Grassmann core rotations.

This experiment takes a one-sided alternative to hierarchical ULV.  At every
step, the active coordinates consist of one survivor bundle from an original
prefix followed by untouched suffix coordinates.  The leading block couples
to the suffix through an upper-right (quasiseparable) block.  Its numerical
column space is compressed with the Grassmann-optimal core construction from
``ulv_core_transforms.py``; decoupled coordinate pairs retire immediately.

Exact count statement
---------------------
Suppose U is orthogonal and every prefix/suffix block U[:i, i:] has rank at
most s.  Run this algorithm with a block of b=s+1 coordinates.  If the current
coupling rank is r and d=b-r coordinates retire, then the step costs at most

    row compression:   r*d
    column retirement: b*d - d*(d+1)/2
    total:             2*r*d + d*(d-1)/2 <= 2*s*d.

The last inequality uses d=s+1-r.  Thus all sequential steps use at most 2*s
rotations per retired coordinate, and the final block has size at most
2*(s+1).  A conservative exact bound is therefore

    N < 2*s*k + 2*(s+1)^2.

This is a direct product of plane rotations, not a Givens-weight
representation with an unevaluated banded weight matrix.

Approximate use
---------------
For A-matrices, the prefix/suffix Cauchy blocks have rapidly decaying singular
values.  The default adaptive version grows its window only when the measured
rank saturates it.  Each discarded coupling tail has norm at most ``delta``.
The elementary worst-case error bound accumulates these tails linearly, so a
rigorous target epsilon can use delta=O(epsilon/k); the much smaller observed
accumulation is reported, not assumed.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fourier.amatrix import random_content_a_matrix, staircase_a_matrix
from fourier.decompositions import givens_factor

from ulv_circuit import factorization_error
from ulv_core_transforms import (
    _compress_core,
    _left_rotation,
    _retire_core,
)


@dataclass(frozen=True)
class StepCount:
    """Rank and rotation accounting for one prefix step."""

    size: int
    rank: int
    retired: int
    row_rotations: int
    column_rotations: int


def quasiseparable_core_factor(
    A,
    delta=1e-5,
    rank_bound=None,
    initial_window=8,
):
    """Reduce an orthogonal matrix to an approximate sign diagonal.

    ``rank_bound=s`` selects the theorem-backed fixed block size ``s+1`` and
    raises ``ValueError`` if a transformed coupling rank exceeds s.
    ``rank_bound=None`` uses an adaptive window, suitable when only numerical
    ranks are known.

    Returns ``(ops, M_final, count, step_counts)``.
    """
    k = A.shape[0]
    M = A.copy()
    ops = []
    active_rows = list(range(k))
    active_cols = list(range(k))
    steps = []

    if rank_bound is not None:
        if rank_bound < 0:
            raise ValueError("rank_bound must be nonnegative")
        window = rank_bound + 1
    else:
        if initial_window < 2:
            raise ValueError("initial_window must be at least 2")
        window = initial_window

    while len(active_rows) > 2 * window:
        m = len(active_rows)
        window = min(window, m // 2)

        while True:
            rows = active_rows[:window]
            cols = active_cols[:window]
            far_cols = active_cols[window:]
            E = M[np.ix_(rows, far_cols)]
            U, singular_values, _ = np.linalg.svd(E, full_matrices=False)
            r = int(np.sum(singular_values > delta))

            if rank_bound is not None:
                if r > rank_bound:
                    raise ValueError(
                        f"observed coupling rank {r} exceeds rank_bound={rank_bound}"
                    )
                break

            if r < window:
                break

            larger = min(2 * window, m // 2)
            if larger == window:
                break
            window = larger

        if r >= window:
            # No numerical compression at this tolerance; finish densely.
            break

        before = len(ops)
        local_order, _ = _compress_core(M, U[:, :r], rows, ops)
        # Keep logical row/column coordinate pairs together.  The physical
        # matrix is not permuted; only the active index lists change order.
        rows = [rows[index] for index in local_order]
        cols = [cols[index] for index in local_order]
        row_count = len(ops) - before

        d = window - r
        before = len(ops)
        _retire_core(M, rows, cols, d, ops)
        column_count = len(ops) - before
        steps.append(StepCount(window, r, d, row_count, column_count))

        active_rows = rows[d:] + active_rows[window:]
        active_cols = cols[d:] + active_cols[window:]

        if rank_bound is None:
            # The next block contains the whole survivor group and at least
            # one untouched coordinate.  Saturation triggers growth above.
            window = max(initial_window, r + 1)

    gates, _ = givens_factor(M[np.ix_(active_rows, active_cols)])
    for gate in reversed(gates):
        i, j = active_rows[gate.i], active_rows[gate.j]
        _left_rotation(M, i, j, -gate.theta, ops)

    return ops, M, len(ops), steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024, 2048])
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument(
        "--rank-bound",
        type=int,
        help="use the fixed theorem-backed block size rank_bound+1",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rng = np.random.default_rng(23)
    print(
        f"sequential quasiseparable core, delta={args.delta:g}, "
        f"window={args.window}, rank_bound={args.rank_bound}"
    )
    print(
        f"{'target':>10} {'k':>5} {'rotations':>10} {'dense':>10} "
        f"{'ratio':>7} {'error':>10} {'max rank':>9}"
    )

    rows = []
    for k in args.sizes:
        for label, A in (
            ("staircase", staircase_a_matrix(k)),
            ("generic", random_content_a_matrix(k, rng)),
        ):
            _, M_final, count, steps = quasiseparable_core_factor(
                A,
                delta=args.delta,
                rank_bound=args.rank_bound,
                initial_window=args.window,
            )
            dense = k * (k - 1) // 2
            max_rank = max((step.rank for step in steps), default=0)
            error = factorization_error(M_final)
            rows.append(
                {
                    "k": k,
                    "family": label,
                    "delta": args.delta,
                    "initial_window": args.window,
                    "rank_bound": args.rank_bound,
                    "rotations": count,
                    "dense": dense,
                    "ratio": count / dense,
                    "operator_error": error,
                    "max_rank": max_rank,
                }
            )
            print(
                f"{label:>10} {k:>5} {count:>10,} {dense:>10,} "
                f"{count / dense:>7.3f} {error:>10.2e} "
                f"{max_rank:>9}"
            )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
