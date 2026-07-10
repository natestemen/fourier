"""Algebraic relationships between small A-matrices — and a negative recursion.

Question: are the smallest A-matrices related by exact conjugacy or embedded
products, and does the family recurse as A(λ') = M·(1 ⊕ A(λ)) with a
*structured* orthogonal M?

Three parts, all exact (sympy) except where noted:

1. 2×2 conjugacy identities: A(1)·A(2)·A(1) = A(1,1) and
   A(1)·A(1,1)·A(1) = A(2) — the level-2 matrices are conjugate under A(1).
2. Embedded-product searches for A(2,1): every 3-fold and 4-fold product of
   2×2 A-matrices embedded into 3×3 coordinate planes is compared against
   A(2,1) (numeric prefilter, symbolic confirmation of hits), plus
   column-level factorization probes and the A(1)-conjugates of A(2,1).
3. The M(1 ⊕ A_k) recursion check: for random 3-addable diagrams λ',
   M = A(λ')·(1 ⊕ A(1,1))ᵀ.  Since 1 ⊕ A is exactly orthogonal, an
   orthogonal M always exists trivially; the meaningful (negative) question
   is whether M is ever itself an A-matrix or otherwise structured, so M is
   also compared against the full 3-addable catalog.

Supports report.md, Finding 1 ("I also checked whether A_{k+1} could be
written as M(1 ⊕ A_k) … but it does not hold in general") and the framing of
"The Core Unsolved Problem".

Expected result: the level-2 conjugacy identities hold exactly; no embedded
product reproduces A(2,1); M is orthogonal by construction but never matches
a 3-addable A-matrix — in fact det M = −1 (det A' = +1, det B = −1) while
every 3-addable A-matrix has det +1, so the op-norm distance floor is 2.

Behavior notes vs the old scripts: the embedding searches use a numeric
prefilter with symbolic confirmation of hits (same search space, identical
conclusions, much faster than simplifying all 243 products); the recursion
check computes M = A'·Bᵀ directly instead of solving normal equations
(identical since B is orthogonal) and adds the catalog comparison.
"""

from __future__ import annotations

import argparse
import random
from itertools import product

import numpy as np
import sympy as sp
from yungdiagram import YoungDiagram

from fourier.amatrix import a_matrix, a_matrix_symbolic
from fourier.diagrams import diagrams_with_addable_cells

PAIRS = [(0, 1), (0, 2), (1, 2)]


def embed(M: sp.Matrix, i: int, j: int, n: int = 3) -> sp.Matrix:
    """Embed a 2×2 matrix into the (i, j) coordinate plane of an n×n identity."""
    E = sp.eye(n)
    E[i, i] = M[0, 0]
    E[i, j] = M[0, 1]
    E[j, i] = M[1, 0]
    E[j, j] = M[1, 1]
    return E


def numeric(M: sp.Matrix) -> np.ndarray:
    return np.array(M.evalf().tolist(), dtype=float)


# ── part 1: structure and 2×2 conjugacy ────────────────────────────────────────


def basic_structure(named: list[tuple[str, sp.Matrix]]) -> None:
    print("=== Basic structure ===")
    for name, M in named:
        det = sp.simplify(M.det())
        symmetric = sp.simplify(M - M.T) == sp.zeros(*M.shape)
        involution = sp.simplify(M**2 - sp.eye(M.shape[0])) == sp.zeros(*M.shape)
        print(f"  {name}: det={det}, symmetric={symmetric}, involution={involution}")
    print()


def conjugacy_identities(A1: sp.Matrix, A2: sp.Matrix, A11: sp.Matrix) -> None:
    conj = sp.simplify(A1 * A2 * A1)
    print("=== A(1) · A(2) · A(1) ===")
    sp.pprint(conj)
    print(f"  == A(1,1)? {conj == sp.simplify(A11)}")
    print()

    conj2 = sp.simplify(A1 * A11 * A1)
    print("=== A(1) · A(1,1) · A(1) ===")
    sp.pprint(conj2)
    print(f"  == A(2)? {conj2 == sp.simplify(A2)}")
    print()
    print("So A(1,1) and A(2) are conjugate under A(1): both are level-2")
    print("partitions, and A(1) is the level-1 matrix.")
    print()


# ── part 2: embedded-product searches for A(2,1) ───────────────────────────────


def _search_products(
    factors: list[tuple[str, sp.Matrix]],
    target_sym: sp.Matrix,
    target_num: np.ndarray,
    description: str,
) -> None:
    """Compare every embedded product of `factors` against the target:
    numeric prefilter (allclose to 1e-10), symbolic confirmation of hits."""
    print(f"Searching: A(2,1) = {description} ...")
    numeric_factors = [numeric(M) for _, M in factors]

    found = False
    for positions in product(PAIRS, repeat=len(factors)):
        P_num = np.eye(3)
        for (i, j), F in zip(positions, numeric_factors):
            E = np.eye(3)
            E[i, i], E[i, j], E[j, i], E[j, j] = F[0, 0], F[0, 1], F[1, 0], F[1, 1]
            P_num = P_num @ E
        if not np.allclose(P_num, target_num, atol=1e-10):
            continue
        # Symbolic confirmation of the numeric hit.
        P = sp.eye(3)
        for (i, j), (_, M) in zip(positions, factors):
            P = P * embed(M, i, j)
        if sp.simplify(P) == target_sym:
            names = ", ".join(f"{name} in {pos}" for (name, _), pos in zip(factors, positions))
            print(f"  FOUND: {names}")
            found = True
    if not found:
        print("  (none found)")
    print()


def embedding_searches(
    A1: sp.Matrix, A2: sp.Matrix, A11: sp.Matrix, A21: sp.Matrix
) -> None:
    A21_sym = sp.simplify(A21)
    A21_num = numeric(A21)

    print("=== Can we extend the conjugacy to A(2,1)? ===")
    print("A(2,1) is 3×3, so we embed the 2×2 matrices into coordinate planes.")
    print()

    _search_products([("A1", A1), ("A2", A2), ("A1", A1)], A21_sym, A21_num,
                     "embed(A1,ij) · embed(A2,kl) · embed(A1,mn)")
    _search_products([("A11", A11), ("A2", A2), ("A11", A11)], A21_sym, A21_num,
                     "embed(A11,ij) · embed(A2,kl) · embed(A11,mn)")

    print("Searching: A(2,1) = 4-fold embedded products ...")
    combos = [
        [("A1", A1), ("A2", A2), ("A11", A11), ("A2", A2)],
        [("A2", A2), ("A1", A1), ("A2", A2), ("A1", A1)],
        [("A2", A2), ("A11", A11), ("A2", A2), ("A11", A11)],
    ]
    for combo in combos:
        _search_products(combo, A21_sym, A21_num,
                         " · ".join(name for name, _ in combo))


def column_factorization(
    A1: sp.Matrix, A2: sp.Matrix, A11: sp.Matrix, A21: sp.Matrix
) -> None:
    print("=== Column-level factorization ===")
    for p in PAIRS:
        if sp.simplify(embed(A11, *p) * A21.col(2)) == sp.simplify(A21.col(1)):
            print(f"embed(A11,{p}) · A21[:,2] == A21[:,1]")
    for p in PAIRS:
        if sp.simplify(embed(A2, *p) * A21.col(1)) == sp.simplify(A21.col(2)):
            print(f"embed(A2,{p}) · A21[:,1] == A21[:,2]")
    for p in PAIRS:
        if sp.simplify(embed(A1, *p) * A21.col(1)) == sp.simplify(A21.col(2)):
            print(f"embed(A1,{p}) · A21[:,1] == A21[:,2]")
        if sp.simplify(embed(A1, *p) * A21.col(2)) == sp.simplify(A21.col(1)):
            print(f"embed(A1,{p}) · A21[:,2] == A21[:,1]")
    print()

    print("A(2,1) · embed(A1, *) for each pair:")
    for p in PAIRS:
        prod_ = sp.simplify(A21 * embed(A1, *p))
        print(f"  A21 · embed(A1,{p}) =")
        sp.pprint(prod_)
        print()

    print("=== Conjugates of A(2,1) by embed(A1,*) ===")
    for p in PAIRS:
        E = embed(A1, *p)
        conj = sp.simplify(E * A21 * E)
        print(f"  embed(A1,{p}) · A21 · embed(A1,{p}) =")
        sp.pprint(conj)
        print(f"  Is this A(2,1)? {conj == sp.simplify(A21)}")
        print()


# ── part 3: the M(1 ⊕ A_k) recursion check ─────────────────────────────────────


def recursion_check(count: int, max_size: int, seed: int, tol: float) -> None:
    print("=" * 72)
    print("M(1 ⊕ A) recursion check:  A(λ') = M · (1 ⊕ A(1,1)),  M = A(λ')·Bᵀ")
    print("=" * 72)
    rng = random.Random(None if seed == 0 else seed)

    dA = YoungDiagram([1, 1])
    A = a_matrix(dA)
    B = np.eye(3)
    B[1:, 1:] = A

    catalog = [
        (str(d.partition), a_matrix(d)) for d in diagrams_with_addable_cells(3, max_size)
    ]
    diagrams_ap = list(diagrams_with_addable_cells(3, max_size))
    sample = diagrams_ap if len(diagrams_ap) < count else rng.sample(diagrams_ap, count)

    print(f"Fixed A diagram: {dA.partition}")
    print("A:\n", A)
    print("B = 1 ⊕ A:\n", B)

    n_amatrix = 0
    for i, dAp in enumerate(sample, start=1):
        Ap = a_matrix(dAp)
        M = Ap @ B.T  # exact solution of A' = M·B, since B is orthogonal
        ortho_err = np.linalg.norm(M @ M.T - np.eye(3), ord="fro")
        recon_err = np.linalg.norm(M @ B - Ap, ord="fro")

        dists = [(label, float(np.linalg.norm(Ad - M, 2))) for label, Ad in catalog]
        best_label, best_dist = min(dists, key=lambda x: x[1])

        print("=" * 72)
        print(f"[{i}] A' diagram: {dAp.partition}")
        print(f"orthonormality error ‖MMᵀ − I‖: {ortho_err:.3e}")
        print(f"reconstruction error ‖MB − A'‖: {recon_err:.3e}")
        print(M)
        if best_dist <= tol:
            n_amatrix += 1
            print(f"M IS a 3-addable A-matrix: {best_label}  (op_norm={best_dist:.2e})")
        else:
            print(f"M is not a 3-addable A-matrix "
                  f"(closest: {best_label}, op_norm={best_dist:.4f})")

    print("=" * 72)
    print(f"SUMMARY: {n_amatrix} / {len(sample)} factors M matched a 3-addable A-matrix.")


# ── CLI ────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--count", type=int, default=10,
                   help="Number of random 3-addable A' for the recursion check (default: 10).")
    p.add_argument("--max-size", type=int, default=20,
                   help="Max diagram size for the recursion check (default: 20).")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed; 0 = random (default: 0).")
    p.add_argument("--tol", type=float, default=1e-8,
                   help="Operator-norm tolerance for the M catalog match (default: 1e-8).")
    p.add_argument("--skip-symbolic", action="store_true",
                   help="Skip the symbolic identity/embedding sections.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.skip_symbolic:
        A1 = a_matrix_symbolic([1])
        A2 = a_matrix_symbolic([2])
        A11 = a_matrix_symbolic([1, 1])
        A21 = a_matrix_symbolic([2, 1])

        basic_structure([("A(1)", A1), ("A(2)", A2), ("A(1,1)", A11), ("A(2,1)", A21)])
        conjugacy_identities(A1, A2, A11)
        embedding_searches(A1, A2, A11, A21)
        column_factorization(A1, A2, A11, A21)

    recursion_check(args.count, args.max_size, args.seed, args.tol)


if __name__ == "__main__":
    main()
