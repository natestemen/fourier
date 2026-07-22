"""A-matrices are orthogonal HSS matrices with k-independent ε-ranks.

Question: does A(λ) carry hierarchical low-rank structure that an
approximate circuit factorization could exploit?

Finding (RESULTS.md §9): yes.  Every HSS block row (a dyadic row block
against all other columns) has ε-rank bounded by a constant independent of k
(≤ 10 at ε = 1e-3 for k up to 512, vs full rank k/2 for random SO(k)), and
the rank grows only like log(1/ε).  The mechanism is the Cauchy kernel
1/(c(a) − c(r)): index-separated blocks are content-separated, so their
singular values decay exponentially.

Expected output: a rank table per level, flat in k for A-matrices and full
for the random control; and the log(1/ε) sweep.

Large-k A-matrices are built directly from contents in log-space (the hook
formula overflows float64 beyond k ≈ 200); the construction is validated
against `fourier.amatrix.a_matrix` at small k in tests and reproduces the
staircase matrix to 1e-15.
"""

import argparse

import numpy as np
from scipy.stats import ortho_group

from fourier.amatrix import a_matrix, random_content_a_matrix, staircase_a_matrix
from fourier.diagrams import staircase


def hss_ranks(A: np.ndarray, eps: float, min_block: int = 8) -> dict[int, tuple[int, int]]:
    """{level: (block size, max ε-rank of any HSS block row at that level)}."""
    k = A.shape[0]
    out = {}
    size, level = k // 2, 0
    while size >= min_block:
        ranks = []
        for start in range(0, k, size):
            rows = slice(start, start + size)
            cols = np.r_[0:start, start + size : k]
            sv = np.linalg.svd(A[rows, cols], compute_uv=False)
            ranks.append(int(np.sum(sv > eps)))
        out[level] = (size, max(ranks))
        size //= 2
        level += 1
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--eps-sweep", action="store_true",
                        help="also sweep eps at the largest size")
    args = parser.parse_args()

    rng = np.random.default_rng(9)

    err = np.max(np.abs(a_matrix(staircase(15)) - staircase_a_matrix(16)))
    print(f"construction check vs a_matrix(staircase(15)): max diff = {err:.1e}\n")

    print(f"=== max HSS block-row ε-rank per dyadic level (ε = {args.eps}) ===")
    for k in args.sizes:
        targets = [("staircase", staircase_a_matrix(k)), ("generic", random_content_a_matrix(k, rng))]
        if k <= 256:
            targets.append(("random SO", ortho_group.rvs(k, random_state=rng)))
        for label, A in targets:
            prof = "  ".join(f"s={sz}:r≤{r}" for sz, r in hss_ranks(A, args.eps).values())
            print(f"  {label:10s} k={k:4d}  {prof}")
        print()

    if args.eps_sweep:
        k = max(args.sizes)
        A = staircase_a_matrix(k)
        print(f"=== ε-dependence at k = {k} (staircase) ===")
        for eps in (1e-2, 1e-3, 1e-6, 1e-9, 1e-12):
            ranks = [r for _, r in hss_ranks(A, eps).values()]
            print(f"  ε={eps:.0e}: max rank per level {ranks}")


if __name__ == "__main__":
    main()
