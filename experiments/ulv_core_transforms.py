"""Whole-group ULV using incomplete (core) Givens transformations.

This is an isolated refinement of ``ulv_circuit.py``.  It changes two
things:

1. survivor groups are never split when the next level is blocked, matching
   the support invariant used in ``ulv-proof.md``;
2. local orthogonal matrices are not completed and then triangularized.
   Only the Stiefel factors that affect the active subspaces are reduced.

For a block of ``bb`` active coordinates whose coupling rank is ``r``, let
``d = bb-r``.  The row compression uses exactly the Grassmannian-sized bound

    d*r

plane rotations generically.  The construction finds ``r`` well-conditioned
survivor coordinates and nulls each of the other ``d`` coordinate rows with
``r`` rotations.  The right retirement transform then uses at most

    bb*d - d*(d + 1)/2.

The old implementation spends up to ``bb*(bb-1)`` rotations on two complete
``bb x bb`` factors.  At the count-optimal choice ``bb ~= 2r``, the new
leading cost is ``2.5r`` rotations per retired coordinate instead of ``4r``.

The approximation itself is unchanged: an SVD selects the coupling subspace,
the leading ``bb-r`` rows are retired, and the remaining active block is
finished densely.  Every emitted operation is an ordinary plane rotation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from scipy.linalg import qr

from fourier.amatrix import random_content_a_matrix, staircase_a_matrix
from fourier.decompositions import givens_factor

from ulv_circuit import _apply_left, _apply_right, factorization_error


@dataclass(frozen=True)
class BlockCount:
    """Rotation accounting for one compressed block."""

    size: int
    rank: int
    row_rotations: int
    column_rotations: int


def _left_rotation(M, i, j, theta, ops):
    """Apply G(i,j,theta) on the left and record it."""
    c, s = np.cos(theta), np.sin(theta)
    Mi, Mj = M[i, :].copy(), M[j, :].copy()
    M[i, :] = c * Mi - s * Mj
    M[j, :] = s * Mi + c * Mj
    ops.append(("L", i, j, theta))


def _right_rotation(M, i, j, theta, ops):
    """Apply G(i,j,theta) on the right and record it."""
    c, s = np.cos(theta), np.sin(theta)
    Mi, Mj = M[:, i].copy(), M[:, j].copy()
    M[:, i] = c * Mi + s * Mj
    M[:, j] = -s * Mi + c * Mj
    ops.append(("R", i, j, theta))


def _compress_core(M, coupling_basis, rows, ops):
    """Confine an r-dimensional coupling basis to r survivor coordinates.

    A generic r-plane in R^bb has Grassmann dimension r*(bb-r), and this
    routine attains that count.  Pivoted QR first chooses r independent
    coordinate rows.  For each other coordinate, the (r+1)-by-r matrix
    consisting of that row and the survivors has a one-dimensional left null
    space.  Mapping its null vector to the retiring coordinate takes r Givens
    rotations and makes that entire row of the coupling basis zero.

    Returns the local permutation ``retired + survivors``.  The caller must
    apply it to both the row and partner-column index lists so that retirement
    still lands on the global diagonal.
    """
    B = coupling_basis.copy()
    bb, r = B.shape
    if r == 0:
        return list(range(bb)), 0

    # Columns of B.T are coordinate rows of the coupling basis.  Pivoting
    # chooses an invertible, numerically stable r-by-r survivor submatrix.
    _, _, pivots = qr(B.T, mode="economic", pivoting=True)
    survivors = [int(index) for index in pivots[:r]]
    survivor_set = set(survivors)
    retired = [index for index in range(bb) if index not in survivor_set]

    before = len(ops)
    for target in retired:
        active = [target] + survivors
        S = B[active, :]

        # z.T @ S = 0.  If P z = e_0 for orthogonal P, then the first
        # row of P is z.T, so applying P makes the target row exactly zero.
        _, _, Vt = np.linalg.svd(S.T, full_matrices=True)
        z = Vt[-1, :].copy()
        if z[0] < 0:
            z *= -1

        for offset, survivor in enumerate(survivors, start=1):
            theta = -np.arctan2(z[offset], z[0])
            c, s = np.cos(theta), np.sin(theta)

            z0, zq = z[0], z[offset]
            z[0] = c * z0 - s * zq
            z[offset] = s * z0 + c * zq

            Bt, Bs = B[target, :].copy(), B[survivor, :].copy()
            B[target, :] = c * Bt - s * Bs
            B[survivor, :] = s * Bt + c * Bs
            _left_rotation(M, rows[target], rows[survivor], theta, ops)

    order = retired + survivors
    return order, len(ops) - before


def _retire_core(M, rows, cols, d, ops):
    """Map ``d`` confined rows to the first ``d`` partner columns.

    This is thin QR of ``M[rows[:d], cols].T`` performed directly with core
    transformations.  No arbitrary orthogonal complement is formed.
    """
    C = M[np.ix_(rows[:d], cols)].T.copy()
    bb = len(cols)
    before = len(ops)

    for col in range(d):
        pivot = col
        for target in range(pivot + 1, bb):
            x, y = C[pivot, col], C[target, col]
            if abs(y) <= 1e-15:
                continue
            theta = np.arctan2(y, x)
            c, s = np.cos(theta), np.sin(theta)
            Cp, Ct = C[pivot, :].copy(), C[target, :].copy()
            C[pivot, :] = c * Cp + s * Ct
            C[target, :] = -s * Cp + c * Ct
            _right_rotation(M, cols[pivot], cols[target], theta, ops)

    return len(ops) - before


def _whole_group_blocks(groups, target_size):
    """Merge adjacent survivor groups without ever splitting a group."""
    blocks = []
    current = None

    for rows, cols, lo, hi in groups:
        if current is None:
            current = (rows.copy(), cols.copy(), lo, hi)
        else:
            cr, cc, clo, chi = current
            current = (cr + rows, cc + cols, min(clo, lo), max(chi, hi))

        if len(current[0]) >= target_size:
            blocks.append(current)
            current = None

    if current is not None:
        if blocks:
            rows, cols, lo, hi = blocks[-1]
            cr, cc, clo, chi = current
            blocks[-1] = (rows + cr, cols + cc, min(lo, clo), max(hi, chi))
        else:
            blocks.append(current)

    return blocks


def ulv_core_factor(A, b=32, delta=1e-4, complete_factors=False):
    """Reduce an orthogonal matrix to an approximate sign diagonal.

    With ``complete_factors=False`` (the default), incomplete core
    transformations give the reduced rotation count.  Setting it to true
    runs the same whole-group recursion but decomposes complete local
    orthogonal factors, providing an apples-to-apples count baseline.

    Returns ``(ops, M_final, count, block_counts)``.
    """
    k = A.shape[0]
    M = A.copy()
    ops = []
    groups = [([i], [i], i, i) for i in range(k)]
    block_counts = []

    while sum(len(group[0]) for group in groups) > 2 * b:
        active_cols = [col for group in groups for col in group[1]]
        blocks = _whole_group_blocks(groups, b)
        new_groups = []
        progressed = False

        for rows, cols, lo, hi in blocks:
            bb = len(rows)
            own = set(cols)
            offcols = [col for col in active_cols if col not in own]
            E = M[np.ix_(rows, offcols)]
            U, singular_values, _ = np.linalg.svd(E, full_matrices=False)
            r = int(np.sum(singular_values > delta))

            if r >= bb:
                new_groups.append((rows, cols, lo, hi))
                continue

            before = len(ops)
            if complete_factors:
                # A full U is needed only for this comparison path.
                if U.shape[1] < bb:
                    U, _, _ = np.linalg.svd(E, full_matrices=True)
                Q = np.vstack([U[:, r:].T, U[:, :r].T])
                _apply_left(M, Q, rows, ops)
            else:
                order, _ = _compress_core(M, U[:, :r], rows, ops)
                rows = [rows[index] for index in order]
                cols = [cols[index] for index in order]
            row_count = sum(op[0] in ("L", "R") for op in ops[before:])

            d = bb - r
            before = len(ops)
            if complete_factors:
                W, _ = np.linalg.qr(M[np.ix_(rows[:d], cols)].T, mode="complete")
                _apply_right(M, W, cols, ops)
            else:
                _retire_core(M, rows, cols, d, ops)
            column_count = sum(op[0] in ("L", "R") for op in ops[before:])

            block_counts.append(BlockCount(bb, r, row_count, column_count))
            new_groups.append((rows[d:], cols[d:], lo, hi))
            progressed = True

        if not progressed:
            break
        groups = new_groups

    active_rows = [row for group in groups for row in group[0]]
    active_cols = [col for group in groups for col in group[1]]

    gates, _ = givens_factor(M[np.ix_(active_rows, active_cols)])
    for gate in reversed(gates):
        i, j = active_rows[gate.i], active_rows[gate.j]
        _left_rotation(M, i, j, -gate.theta, ops)

    count = sum(op[0] in ("L", "R") for op in ops)
    return ops, M, count, block_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--leaf", type=int, default=32)
    parser.add_argument("--delta", type=float, default=1e-4)
    args = parser.parse_args()

    rng = np.random.default_rng(17)
    print(f"whole-group ULV, leaf target={args.leaf}, delta={args.delta:g}")
    print(
        f"{'target':>10} {'k':>5} {'core':>9} {'complete':>9} "
        f"{'saving':>8} {'core err':>10} {'full err':>10}"
    )
    for k in args.sizes:
        for label, A in (
            ("staircase", staircase_a_matrix(k)),
            ("generic", random_content_a_matrix(k, rng)),
        ):
            _, M_core, n_core, _ = ulv_core_factor(A, args.leaf, args.delta)
            _, M_full, n_full, _ = ulv_core_factor(
                A, args.leaf, args.delta, complete_factors=True
            )
            print(
                f"{label:>10} {k:>5} {n_core:>9,} {n_full:>9,} "
                f"{1.0 - n_core / n_full:>7.1%} "
                f"{factorization_error(M_core):>10.2e} "
                f"{factorization_error(M_full):>10.2e}"
            )


if __name__ == "__main__":
    main()
