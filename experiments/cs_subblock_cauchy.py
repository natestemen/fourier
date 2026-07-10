"""Exact test: are CS sub-blocks of A(λ) A-matrices of ANY real contents?

Settles open directions #1/#2 of report.md (RESULTS.md Finding 7).  Unlike
the older catalog/optimization search (cs_subblock_match.py), this is an
exact decision procedure, invariant under the full gauge freedom of the CS
decomposition (per-plane column signs and plane permutations) and under the
affine reparametrization of contents that leaves an A-matrix unchanged.

Method: if M = A(ac, rc) with constant column c0, then with α = |M[:,c0]| and
N[i,j] = αᵢ/M[i,j] (j ≠ c0), the matrix W[i,j] = N[i,j] − N[0,j] equals
(acᵢ − ac₀)/βⱼ and therefore has RANK ONE.  Rank-1-ness is testable by SVD
(gauge-invariantly), recovers the contents in closed form, and its second
singular value certifies non-membership.

Expected result: k=6 blocks all pass vacuously (every generic 3×3 orthogonal
is a real-content A-matrix — a dimension coincidence); k=8 blocks ALL FAIL
(0 of 425×4 up to size 36), and --split-search shows no alternative CS
bipartition works either.  The CS circuit recursion is falsified.
"""

import argparse
from itertools import combinations

import numpy as np
from yungdiagram import YoungDiagram

from fourier.amatrix import a_matrix, a_matrix_from_contents
from fourier.decompositions import cs_factor
from fourier.diagrams import diagrams_with_addable_cells


def amatrix_content_test(
    M: np.ndarray, rank_tol: float = 1e-8, recon_tol: float = 1e-8
) -> dict:
    """Decide whether orthogonal M equals an A-matrix of some real content
    sequence, up to column signs, with any column as the constant column.

    Returns {'is_amatrix', 'const_col', 'contents', 'rank_ratio', 'recon_err'};
    rank_ratio is the best σ₂/σ₁ over admissible constant columns (∞ when no
    column is sign-uniform — a cheap necessary condition).
    """
    p = M.shape[0]
    best = {
        "is_amatrix": False,
        "rank_ratio": np.inf,
        "const_col": None,
        "contents": None,
        "recon_err": np.inf,
    }

    for c0 in range(p):
        col = M[:, c0]
        if np.any(np.abs(col) < 1e-12) or not (np.all(col > 0) or np.all(col < 0)):
            continue
        alpha = np.abs(col)

        rest = [j for j in range(p) if j != c0]
        if np.any(np.abs(M[:, rest]) < 1e-14):
            continue
        N = alpha[:, None] / M[:, rest]
        W = (N - N[0, :][None, :])[1:, :]

        sv = np.linalg.svd(W, compute_uv=False)
        ratio = sv[1] / sv[0] if len(sv) > 1 and sv[0] > 0 else 0.0
        best["rank_ratio"] = min(best["rank_ratio"], ratio)
        if ratio > rank_tol:
            continue

        # recover contents in the gauge ac₀ = 0 (scale is free)
        U_, S_, Vt_ = np.linalg.svd(W)
        ac = np.concatenate([[0.0], U_[:, 0] * S_[0]])
        beta = 1.0 / Vt_[0, :]
        rc = -N[0, :] * beta

        row_order = np.argsort(-ac)
        col_order = np.argsort(-rc)
        A_test = a_matrix_from_contents(ac[row_order], rc[col_order])
        M_perm = M[np.ix_(row_order, [c0] + [rest[j] for j in col_order])]
        col_signs = np.sign(M_perm[0, :]) * np.sign(A_test[0, :])
        err = float(np.max(np.abs(M_perm - A_test * col_signs[None, :])))

        if err < best["recon_err"]:
            best.update(
                recon_err=err,
                const_col=c0,
                contents=(ac[row_order], rc[col_order]),
            )
        if err < recon_tol:
            best["is_amatrix"] = True
            return best

    return best


def scan(k: int, max_size: int) -> None:
    print(f"=== k = {k} (sub-blocks {k // 2}×{k // 2}), diagrams up to size {max_size} ===")
    n_diag = 0
    n_match = dict.fromkeys(("U1", "U2", "V1", "V2"), 0)
    closest_miss = np.inf
    for yd in diagrams_with_addable_cells(k, max_size):
        cs = cs_factor(a_matrix(yd))
        n_diag += 1
        for name, block in [("U1", cs.u1), ("U2", cs.u2), ("V1", cs.v1t.T), ("V2", cs.v2t.T)]:
            res = amatrix_content_test(block)
            if res["is_amatrix"]:
                n_match[name] += 1
            elif res["rank_ratio"] != np.inf:
                closest_miss = min(closest_miss, res["rank_ratio"])
    if n_diag == 0:
        print(f"no {k}-addable diagrams exist up to size {max_size} "
              f"(minimum is {(k - 1) * k // 2}); raise --max-size")
        return
    print(f"diagrams tested: {n_diag} (4 blocks each)")
    print(f"blocks that ARE real-content A-matrices: {n_match}")
    print(f"closest miss certificate (σ₂/σ₁): {closest_miss:.3e}")


def split_search(partition: list[int]) -> None:
    """Exhaustive search over all (row-subset, column-subset) CS bipartitions."""
    yd = YoungDiagram(partition)
    A = a_matrix(yd)
    k = A.shape[0]
    p = k // 2

    n_full = 0
    best = 0
    row_subsets = [list(c) for c in combinations(range(k), p) if 0 in c]  # fix row 0 (symmetry)
    col_subsets = [list(c) for c in combinations(range(k), p)]
    for rt in row_subsets:
        for ct in col_subsets:
            rperm = rt + [i for i in range(k) if i not in rt]
            cperm = ct + [j for j in range(k) if j not in ct]
            cs = cs_factor(A[np.ix_(rperm, cperm)])
            n = sum(
                amatrix_content_test(b)["is_amatrix"]
                for b in (cs.u1, cs.u2, cs.v1t.T, cs.v2t.T)
            )
            best = max(best, n)
            if n == 4:
                n_full += 1
                print(f"  FULL PASS rows_top={rt} cols_top={ct}")
    total = len(row_subsets) * len(col_subsets)
    print(f"{tuple(partition)}: splits with all 4 blocks structured: {n_full}/{total}; best partial = {best}/4")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--addable", type=int, default=8, help="k (default 8, the decisive case)")
    parser.add_argument("--max-size", type=int, default=36)
    parser.add_argument(
        "--split-search",
        type=str,
        default=None,
        metavar="PARTITION",
        help="comma-separated partition; exhaustively try every CS bipartition of its A-matrix",
    )
    args = parser.parse_args()

    if args.split_search:
        split_search([int(x) for x in args.split_search.split(",")])
    else:
        scan(args.addable, args.max_size)


if __name__ == "__main__":
    main()
