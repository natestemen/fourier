"""Empirical minimal Givens-rotation count for A-matrices (RESULTS.md §8).

Question: an A-matrix has only 2k−3 effective parameters (contents modulo the
affine gauge), yet the generic factorization uses k(k−1)/2 rotations — does
some fixed pattern of m ≪ k(k−1)/2 plane rotations reproduce A(λ) exactly?

Method: for each target, scan m upward; per m, L-BFGS the angles of several
candidate plane patterns (triangulation prefixes/suffixes, nearest-neighbour
sweeps, random patterns) from random restarts.  det = −1 targets absorb one
uncounted reflection.  A random SO(k) control calibrates the search.

Result (July 2026): NO gap below full — for every diagram tested at k = 4…7
the residual decays smoothly and reaches machine precision only at
m = k(k−1)/2, exactly like the random control.  A-matrices are empirically
circuit-generic despite their low parameter count.  (Caveat: the pattern
search is heuristic; a "magic" pattern could in principle be missed.)
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import ortho_group

from fourier.amatrix import a_matrix
from fourier.diagrams import diagrams_with_addable_cells, staircase


def product_of_rotations(planes, thetas, k):
    M = np.eye(k)
    for (i, j), t in zip(planes, thetas):
        c, s = np.cos(t), np.sin(t)
        row_i, row_j = M[i, :].copy(), M[j, :].copy()
        M[i, :] = c * row_i - s * row_j
        M[j, :] = s * row_i + c * row_j
    return M


def fit_pattern(A, planes, restarts, rng, tol):
    k = A.shape[0]

    def loss(thetas):
        return float(np.sum((A - product_of_rotations(planes, thetas, k)) ** 2))

    best = np.inf
    for _ in range(restarts):
        res = minimize(
            loss,
            rng.uniform(-np.pi, np.pi, len(planes)),
            method="L-BFGS-B",
            options={"maxiter": 3000, "ftol": 1e-22, "gtol": 1e-14},
        )
        best = min(best, res.fun)
        if best < tol**2:
            break
    return float(np.sqrt(best))


def candidate_patterns(k, m, rng, n_random):
    tri = [(j, i) for j in range(k - 1) for i in range(j + 1, k)]
    pats = []
    if m <= len(tri):
        pats.append(tri[:m])
        pats.append(tri[-m:])
    sweep = [(i, i + 1) for i in range(k - 1)]
    pat = []
    while len(pat) < m:
        pat.extend(sweep)
    pats.append(pat[:m])
    all_planes = [(i, j) for i in range(k) for j in range(i + 1, k)]
    for _ in range(n_random):
        idx = rng.integers(0, len(all_planes), size=m)
        pats.append([all_planes[t] for t in idx])
    return pats


def minimal_m(A, label, rng, restarts, n_random, tol):
    k = A.shape[0]
    if np.linalg.det(A) < 0:
        A = A @ np.diag([1.0] * (k - 1) + [-1.0])  # free reflection, not counted
    full = k * (k - 1) // 2
    residuals = {}
    m_star = None
    for m in range(k - 1, full + 1):
        best = np.inf
        for planes in candidate_patterns(k, m, rng, n_random):
            best = min(best, fit_pattern(A, planes, restarts, rng, tol))
            if best < tol:
                break
        residuals[m] = best
        if best < tol:
            m_star = m
            break
    print(
        f"{label:>28s}  full={full}  m*={m_star}  "
        + " ".join(f"m{m}:{residuals[m]:.1e}" for m in sorted(residuals))
    )
    return m_star, residuals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k-max", type=int, default=7)
    parser.add_argument("--restarts", type=int, default=6)
    parser.add_argument("--random-patterns", type=int, default=5)
    parser.add_argument("--tol", type=float, default=1e-7)
    parser.add_argument("--output", type=Path, default=Path("data/min_givens_results.json"))
    args = parser.parse_args()

    rng = np.random.default_rng(7)
    out = {}
    for k in range(4, args.k_max + 1):
        targets = [("staircase", a_matrix(staircase(k - 1)))]
        for yd in diagrams_with_addable_cells(k, 3 * k + 6):
            if tuple(yd.partition) != tuple(staircase(k - 1).partition):
                targets.append((str(yd.partition), a_matrix(yd)))
            if len(targets) >= 3:
                break
        targets.append(("random SO", ortho_group.rvs(k, random_state=rng)))
        for label, A in targets:
            m_star, res = minimal_m(
                A, f"k={k} {label}", rng, args.restarts, args.random_patterns, args.tol
            )
            out[f"k{k}:{label}"] = {
                "m_star": m_star,
                "residuals": {str(m): v for m, v in res.items()},
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
