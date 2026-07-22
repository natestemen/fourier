"""Numerical verification of ulv-proof.md Lemmas 1 and 2.

Question: do the two load-bearing lemmas of the proof draft hold on real
A-matrices, and is the explicit construction of Lemma 2 valid?

Lemma 1 (interlacing geometry): far-column contents sit >= 3 above /
>= 1 below the block's row content interval.  Expected: margins exactly
3 and 1 on the staircase (the inequalities are tight there).

Lemma 2 (dyadic-Taylor rank bound): the piecewise Taylor approximant of an
off-diagonal block achieves rank <= 2p*ceil(log2(2*len)) + 1 with
||E - E~||_2 <= 3*sqrt(k)*2^-p.  Expected: OK at every p, with slack
(the bound is conservative; the error decays like (1/3)^p as derived).
"""

import numpy as np

from fourier.amatrix import _a_matrix_from_contents_log


def build(kind, k, rng):
    if kind == "staircase":
        m = k - 1
        ac = np.array([m - 2.0 * i for i in range(k)])
        rc = np.array([m - 1.0 - 2.0 * j for j in range(k - 1)])
    else:
        gaps = rng.integers(1, 4, size=2 * k - 2)
        seq = np.concatenate([[0.0], np.cumsum(gaps)])[::-1].astype(float)
        ac, rc = seq[0::2], seq[1::2]
    return ac, rc, _a_matrix_from_contents_log(ac, rc)


def dyadic_pieces(lo, hi):
    """Piece boundaries refined toward both ends: lengths 1,1,2,4,... per side."""
    mid = (lo + hi) / 2
    cuts = {lo, hi}
    d = 1.0
    x = hi
    while x - d > mid:
        x -= d
        cuts.add(x)
        d *= 2
    d = 1.0
    x = lo
    while x + d < mid:
        x += d
        cuts.add(x)
        d *= 2
    return sorted(cuts)


def taylor_approx(E, x, y_far, const_col_idx, p):
    """Rank-counted approximant: exact constant column + per-piece p-term
    Taylor of 1/(x−y), applied through the entrywise identity
    E~ = E * (x−y) * f~."""
    Etil = np.zeros_like(E)
    rank = 0
    if const_col_idx is not None:
        Etil[:, const_col_idx] = E[:, const_col_idx]
        rank += 1
    cuts = dyadic_pieces(x.min(), x.max())
    ycols = [j for j in range(E.shape[1]) if j != const_col_idx]
    yv = y_far
    for a, bnd in zip(cuts[:-1], cuts[1:]):
        rows = np.where((x >= a - 1e-9) & (x <= bnd + 1e-9))[0] if bnd == cuts[-1] else \
               np.where((x >= a - 1e-9) & (x < bnd - 1e-9))[0]
        if len(rows) == 0:
            continue
        c = (a + bnd) / 2
        # f~(x,y) = -sum_{s<p} (x-c)^s / (y-c)^{s+1}
        ftil = np.zeros((len(rows), len(yv)))
        for s in range(p):
            ftil += -np.power(x[rows][:, None] - c, s) / np.power(yv[None, :] - c, s + 1)
        f = 1.0 / (x[rows][:, None] - yv[None, :])
        Etil[np.ix_(rows, ycols)] = E[np.ix_(rows, ycols)] * ftil / f
        rank += min(p, len(rows))
    return Etil, rank


def main():
    rng = np.random.default_rng(7)
    k = 256
    for kind in ("staircase", "generic"):
        ac, rc, A = build(kind, k, rng)
        print(f"=== {kind}, k={k} ===")

        # Lemma 1
        worst_left, worst_right = np.inf, np.inf
        for (s, t) in [(0, 31), (64, 95), (112, 143), (224, 255)]:
            top, bot = ac[s], ac[t]
            for pcol in range(1, k):
                yc = rc[pcol - 1]
                if s <= pcol <= t:
                    continue
                if pcol <= s - 1:
                    worst_left = min(worst_left, yc - top)
                else:
                    worst_right = min(worst_right, bot - yc)
        print(f"  Lemma 1: min left margin {worst_left} (claim >= 3), "
              f"min right margin {worst_right} (claim >= 1)")

        # Lemma 2
        s, t = 112, 143
        rows = np.arange(s, t + 1)
        cols = [j for j in range(k) if not (s <= j <= t)]
        E = A[np.ix_(rows, cols)]
        x = ac[rows]
        const_idx = cols.index(0) if 0 in cols else None
        y_far = rc[np.array([j - 1 for j in cols if j != 0])]
        ell = x.max() - x.min()
        print(f"  block rows {s}..{t}, interval length {ell:.0f}, "
              f"pieces {len(dyadic_pieces(x.min(), x.max())) - 1}")
        print(f"  {'p':>3} {'rank':>5} {'bound 2p*log+1':>14} {'err':>10} "
              f"{'claim 3sqrt(k)2^-p':>18} {'true sig_(r+1)':>14}")
        sv = np.linalg.svd(E, compute_uv=False)
        for p in (2, 4, 6, 8, 10):
            Etil, rank = taylor_approx(E, x, y_far, const_idx, p)
            err = float(np.linalg.norm(E - Etil, 2))
            bound_rank = 2 * p * int(np.ceil(np.log2(2 * ell))) + 1
            claim = 3 * np.sqrt(k) * 2.0 ** (-p)
            true_tail = sv[rank] if rank < len(sv) else 0.0
            ok = "OK" if (err <= claim and rank <= bound_rank) else "VIOLATION"
            print(f"  {p:>3} {rank:>5} {bound_rank:>14} {err:>10.2e} "
                  f"{claim:>18.2e} {true_tail:>14.2e}  {ok}")
        print()


if __name__ == "__main__":
    main()
