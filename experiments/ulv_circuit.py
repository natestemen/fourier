"""ULV factorization of A(λ) into explicitly counted Givens rotations.

Question: can the HSS structure of A-matrices (RESULTS.md §9) be converted
into an *actual* rotation sequence with sub-quadratic count?

Answer (RESULTS.md §10): yes.  This script factors A ≈ Lᵀ·D·Rᵀ where L, R
are explicit products of Givens rotations and D = diag(±1), by an
orthogonal-ULV elimination:

  per level, per leaf of b active rows/columns:
  1. row-compress — SVD the leaf's off-diagonal block row; a LOCAL b×b
     rotation confines all off-leaf coupling to the trailing r rows
     (r = ε-rank, ~6–13);
  2. retire — the leading b−r rows now live entirely in the leaf's own
     columns and are orthonormal; a local column rotation maps them to ±e.
     Orthogonality of the full matrix then forces the partner columns to ±e
     automatically, so both retire;
  3. shrink — only r rows/columns per leaf stay active; recurse.

Expected result (δ = 1e-4, b = 32): rotation counts ≈ 10.1k / 21.3k / 45.3k
at k = 256 / 512 / 1024 versus k(k−1)/2 = 32.6k / 130.8k / 523.8k — i.e.
31% / 16% / 8.7% of the dense count, with per-doubling growth ≈ 2.1 (near
linear), at operator error ≈ 1e-4.  Both prior exact negatives (RESULTS.md
§7, §8) are untouched: this is an ε-approximate factorization.
"""

import argparse

import numpy as np

from fourier.amatrix import random_content_a_matrix, staircase_a_matrix
from fourier.decompositions import givens_factor


def _apply_left(M, Q, rows, ops):
    """M[rows,:] ← Q·M[rows,:], Q decomposed into recorded Givens rotations.

    givens_factor gives Q = Gₙ···G₁·diag(signs): signs first, then gates."""
    gates, signs = givens_factor(Q)
    for t, s in enumerate(signs):
        if s < 0:
            M[rows[t], :] *= -1
            ops.append(("LS", rows[t]))
    for g in gates:
        i, j = rows[g.i], rows[g.j]
        c, s = np.cos(g.theta), np.sin(g.theta)
        Mi, Mj = M[i, :].copy(), M[j, :].copy()
        M[i, :] = c * Mi - s * Mj
        M[j, :] = s * Mi + c * Mj
        ops.append(("L", i, j, g.theta))


def _apply_right(M, W, cols, ops):
    """M[:,cols] ← M[:,cols]·W, W decomposed into recorded Givens rotations.

    W = Gₙ···G₁·diag(signs) applied from the right: Gₙ first, signs last."""
    gates, signs = givens_factor(W)
    for g in reversed(gates):
        i, j = cols[g.i], cols[g.j]
        c, s = np.cos(g.theta), np.sin(g.theta)
        Mi, Mj = M[:, i].copy(), M[:, j].copy()
        M[:, i] = c * Mi + s * Mj
        M[:, j] = -s * Mi + c * Mj
        ops.append(("R", i, j, g.theta))
    for t, s in enumerate(signs):
        if s < 0:
            M[:, cols[t]] *= -1
            ops.append(("RS", cols[t]))


def ulv_factor(A, b=32, delta=1e-4):
    """Reduce orthogonal A to ~diag(±1) by structured rotations.

    Returns (ops, M_final, count, rank_log): ops is the ordered transform
    list; count the number of Givens rotations; rank_log the per-level leaf
    ε-ranks.  The achievable circuit error is
    ‖A − Lᵀ·D·Rᵀ‖₂ = ‖M_final − diag(sign(diag))‖₂."""
    k = A.shape[0]
    M = A.copy()
    ops = []
    active_r = list(range(k))
    active_c = list(range(k))
    rank_log = []

    level = 0
    while len(active_r) > 2 * b:
        m = len(active_r)
        n_leaves = m // b
        bounds = [round(t * m / n_leaves) for t in range(n_leaves + 1)]
        new_r, new_c = [], []
        level_ranks = []

        for t in range(n_leaves):
            rows = active_r[bounds[t] : bounds[t + 1]]
            cols = active_c[bounds[t] : bounds[t + 1]]
            bb = len(rows)
            offcols = [c for c in active_c if c not in set(cols)]

            U, s, _ = np.linalg.svd(M[np.ix_(rows, offcols)])
            r = int(np.sum(s > delta))
            level_ranks.append(r)
            if r >= bb:
                new_r += rows
                new_c += cols
                continue

            # 1. left null space of the off-diagonal block into the leading rows
            _apply_left(M, np.vstack([U[:, r:].T, U[:, :r].T]), rows, ops)

            # 2. retire the bb−r decoupled rows against the leaf columns
            d = bb - r
            W, _ = np.linalg.qr(M[np.ix_(rows[:d], cols)].T, mode="complete")
            _apply_right(M, W, cols, ops)

            new_r += rows[d:]
            new_c += cols[d:]

        rank_log.append((level, m, level_ranks))
        if len(new_r) == m:  # no leaf compressed; hand the rest to the dense finish
            break
        active_r, active_c = new_r, new_c
        level += 1

    # dense finish on the remaining active block: apply Gᵀ in reverse order
    gates, _ = givens_factor(M[np.ix_(active_r, active_c)])
    for g in reversed(gates):
        i, j = active_r[g.i], active_r[g.j]
        c, s = np.cos(-g.theta), np.sin(-g.theta)
        Mi, Mj = M[i, :].copy(), M[j, :].copy()
        M[i, :] = c * Mi - s * Mj
        M[j, :] = s * Mi + c * Mj
        ops.append(("L", i, j, -g.theta))

    count = sum(1 for op in ops if op[0] in ("L", "R"))
    return ops, M, count, rank_log


def factorization_error(M_final):
    """‖M_final − diag(sign(diag))‖₂ = ‖A − Lᵀ·D·Rᵀ‖₂."""
    return float(np.linalg.norm(M_final - np.diag(np.sign(np.diag(M_final))), 2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    parser.add_argument("--leaf", type=int, default=32)
    parser.add_argument("--delta", type=float, default=1e-4)
    parser.add_argument("--ranks", action="store_true", help="print per-level leaf ranks")
    args = parser.parse_args()

    rng = np.random.default_rng(1)
    print(f"leaf b = {args.leaf}, truncation δ = {args.delta}")
    print(f"{'target':>10} {'k':>5} {'rotations':>10} {'k(k-1)/2':>9} {'ratio':>6} {'err':>9}")
    prev = {}
    for k in args.sizes:
        for label, A in [
            ("staircase", staircase_a_matrix(k)),
            ("generic", random_content_a_matrix(k, rng)),
        ]:
            ops, M_final, count, rank_log = ulv_factor(A, b=args.leaf, delta=args.delta)
            err = factorization_error(M_final)
            naive = k * (k - 1) // 2
            growth = f"  x{count / prev[label]:.2f}/doubling" if label in prev else ""
            print(
                f"{label:>10} {k:>5} {count:>10} {naive:>9} {count / naive:>6.2f} {err:>9.1e}{growth}"
            )
            prev[label] = count
            if args.ranks:
                for lvl, m, ranks in rank_log:
                    print(
                        f"       level {lvl}: m={m}, leaf ranks max {max(ranks)}, mean {np.mean(ranks):.1f}"
                    )


if __name__ == "__main__":
    main()
