"""ULV factorization with formula-only compression bases (no SVDs).

Question: can the ULV circuit's rotation angles (RESULTS.md §10) be computed
from the diagram's contents alone — the property coherent evaluation needs —
instead of from numerical SVDs of matrix data?

Method: the coupling of a block with row contents x ⊂ I and scalings α to
any outside column is a combination of α and α/(x − y) with y outside I.  So
an explicit "proxy pole" basis
    U = W_c · [α | α/(x − ŷ₁) | α/(x − ŷ₂) | …],
with poles ŷ placed geometrically outside I and W_c the accumulated product
of previously recorded rotations, spans the coupling without ever reading
the matrix.  A pivoted QR of U (still formula-only) picks the rank.  Blocks
must merge whole survivor groups — never split them — so that support
intervals stay disjoint and all far poles lie outside I.

Expected result (b = 32, δ = 10⁻⁴): counts within 1.4× of the SVD-based
`ulv_circuit.ulv_factor` at equal or better error — e.g. k = 1024
staircase: 57,697 rotations vs 45,328 (SVD) vs 523,776 (dense), error
~1×10⁻⁴.  Formula ranks run 13–16 where SVD needs 8–13.  Consequence: every
angle is a feed-forward arithmetic function of the contents (plus earlier
angles); the only matrix-data touch left is the retire step, whose input is
the explicit diagonal block.
"""

import argparse

import numpy as np
from scipy.linalg import qr

from fourier.amatrix import random_content_a_matrix, staircase_a_matrix
from fourier.decompositions import givens_factor

from ulv_circuit import _apply_left, _apply_right, factorization_error, ulv_factor


def proxy_poles(lo, hi, span, d0=0.5, growth=1.5):
    """Poles placed geometrically outside [lo, hi], out to ~2× the full
    content span."""
    poles = []
    d = d0
    while d < 2.2 * span:
        poles.append(hi + d)
        poles.append(lo - d)
        d *= growth
    return np.array(poles)


def formula_rotation(W_c, x_under, alpha_under, span, delta, trunc=1e-2):
    """The block's compression rotation, from formulas only.

    Returns (Q, r): an orthogonal bb×bb transform whose leading bb−r rows are
    orthogonal to the coupling's column space, and the retained rank r —
    both determined by the contents, not by matrix data."""
    poles = proxy_poles(x_under.min(), x_under.max(), span)
    Phi = np.column_stack([alpha_under] + [alpha_under / (x_under - p) for p in poles])
    U = W_c @ Phi
    Q, R, _ = qr(U, mode="full", pivoting=True)
    diag = np.abs(np.diag(R))
    r = min(int(np.sum(diag > trunc * delta * diag[0])), W_c.shape[0])
    return np.vstack([Q[:, r:].T, Q[:, :r].T]), r


def ulv_factor_explicit(A, ac, b=32, delta=1e-4):
    """ULV reduction of A to ~diag(±1) with formula-derived rotations.

    ac = addable contents (descending) aligned with A's rows.  Returns
    (ops, M_final, count, rank_log, worst_decouple) where worst_decouple is
    the largest off-block norm among rows the algorithm retired — the
    internal consistency check on the formula bases."""
    k = A.shape[0]
    M = A.copy()
    RowOps = np.eye(k)  # accumulated left rotations, so W_c is reconstructible
    alpha = A[:, 0].copy()
    span = float(ac[0] - ac[-1])
    ops = []
    rank_log = []
    worst_decouple = 0.0

    def apply_left_tracked(Q, rows):
        n0 = len(ops)
        _apply_left(M, Q, rows, ops)
        for op in ops[n0:]:  # replay on the accumulator
            if op[0] == "LS":
                RowOps[op[1], :] *= -1
            elif op[0] == "L":
                _, i, j, th = op
                c, s = np.cos(th), np.sin(th)
                Ri, Rj = RowOps[i, :].copy(), RowOps[j, :].copy()
                RowOps[i, :] = c * Ri - s * Rj
                RowOps[j, :] = s * Ri + c * Rj

    # groups are indivisible survivor bundles; merging only whole groups
    # keeps support intervals disjoint, so every far column's content lies
    # outside a block's pole interval.
    groups = [([i], [i], i, i) for i in range(k)]

    level = 0
    while sum(len(g[0]) for g in groups) > 2 * b:
        m = sum(len(g[0]) for g in groups)

        blocks = []
        cur_r, cur_c, cur_lo, cur_hi = [], [], None, None
        for rows_g, cols_g, lo_g, hi_g in groups:
            cur_r += rows_g
            cur_c += cols_g
            cur_lo = lo_g if cur_lo is None else min(cur_lo, lo_g)
            cur_hi = hi_g if cur_hi is None else max(cur_hi, hi_g)
            if len(cur_r) >= b:
                blocks.append((cur_r, cur_c, cur_lo, cur_hi))
                cur_r, cur_c, cur_lo, cur_hi = [], [], None, None
        if cur_r:
            if blocks:
                pr, pc, plo, phi = blocks[-1]
                blocks[-1] = (pr + cur_r, pc + cur_c, min(plo, cur_lo), max(phi, cur_hi))
            else:
                blocks.append((cur_r, cur_c, cur_lo, cur_hi))

        active_c_all = [c for g in groups for c in g[1]]
        new_groups = []
        level_ranks = []
        progressed = False

        for rows, cols, s_lo, s_hi in blocks:
            bb = len(rows)
            under = np.arange(s_lo, s_hi + 1)
            W_c = RowOps[np.ix_(rows, under)]

            Q, r = formula_rotation(W_c, ac[under], alpha[under], span, delta)
            level_ranks.append(r)
            if r >= bb:
                new_groups.append((rows, cols, s_lo, s_hi))
                continue

            apply_left_tracked(Q, rows)

            d = bb - r
            offcols = [c for c in active_c_all if c not in set(cols)]
            worst_decouple = max(
                worst_decouple,
                float(np.linalg.norm(M[np.ix_(rows[:d], offcols)], 2)),
            )
            Wq, _ = np.linalg.qr(M[np.ix_(rows[:d], cols)].T, mode="complete")
            _apply_right(M, Wq, cols, ops)

            new_groups.append((rows[d:], cols[d:], s_lo, s_hi))
            progressed = True

        rank_log.append((level, m, level_ranks))
        if not progressed:
            break
        groups = new_groups
        level += 1

    active_r = [i for g in groups for i in g[0]]
    active_c = [c for g in groups for c in g[1]]

    gates, _ = givens_factor(M[np.ix_(active_r, active_c)])
    for g in reversed(gates):
        i, j = active_r[g.i], active_r[g.j]
        c, s = np.cos(-g.theta), np.sin(-g.theta)
        Mi, Mj = M[i, :].copy(), M[j, :].copy()
        M[i, :] = c * Mi - s * Mj
        M[j, :] = s * Mi + c * Mj
        ops.append(("L", i, j, -g.theta))

    count = sum(1 for op in ops if op[0] in ("L", "R"))
    return ops, M, count, rank_log, worst_decouple


def targets(kind, k, rng):
    if kind == "staircase":
        m = k - 1
        ac = np.array([m - 2.0 * i for i in range(k)])
        return ac, staircase_a_matrix(k)
    gaps = rng.integers(1, 4, size=2 * k - 2)
    seq = np.concatenate([[0.0], np.cumsum(gaps)])[::-1].astype(float)
    from fourier.amatrix import _a_matrix_from_contents_log

    return seq[0::2], _a_matrix_from_contents_log(seq[0::2], seq[1::2])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--delta", type=float, default=1e-4)
    parser.add_argument("--leaf", type=int, default=32)
    args = parser.parse_args()

    rng = np.random.default_rng(3)
    print(f"delta={args.delta}, b={args.leaf}")
    print(
        f"{'target':>10} {'k':>5} {'formula rots':>12} {'svd rots':>9} {'ratio':>6} "
        f"{'err(formula)':>12} {'err(svd)':>9} {'decouple':>9}"
    )
    for k in args.sizes:
        for kind in ("staircase", "generic"):
            ac, A = targets(kind, k, rng)
            _, Mf_e, count_e, _, wd = ulv_factor_explicit(A, ac, b=args.leaf, delta=args.delta)
            _, Mf_s, count_s, _ = ulv_factor(A, b=args.leaf, delta=args.delta)
            print(
                f"{kind:>10} {k:>5} {count_e:>12,} {count_s:>9,} {count_e / count_s:>6.2f} "
                f"{factorization_error(Mf_e):>12.1e} {factorization_error(Mf_s):>9.1e} {wd:>9.1e}"
            )


if __name__ == "__main__":
    main()
