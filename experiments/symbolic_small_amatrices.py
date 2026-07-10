#!/usr/bin/env python3
"""Closed-form 2- and 3-addable A-matrices and their composition products.

Question: what do the smallest A-matrices look like as exact functions of
the diagram's block widths and heights, and does composing embedded
2-addable matrices (A ⊕ 1)(1 ⊕ A′), or the three-factor product
M = C₂₃(h₂,w₂)·C₁₂(h₁,w₁)·C₂₃(h₁−h₂, w₁+w₂), reproduce a 3-addable
A-matrix?

Supports report.md "The Core Unsolved Problem: Recursive Decomposition" —
these closed forms are the k = 2, 3 base cases of the conjectured recursion,
and the composition products test its simplest instance.

Expected result: exact symbolic matrices (cross-checked against
fourier.a_matrix_symbolic on concrete diagrams), and composition products
that do NOT reproduce the 3-addable A-matrix (Frobenius distance ~2.0 at
the spot-check point) — no clean recursion at k = 3.

Parametrization (matches fourier.amatrix.a_matrix_generic4's block
convention; the library only exposes the generic builders for k = 4, 8):
  2-addable: the w×h rectangle, addable cells (w,0), (0,h).
  3-addable: blocks of widths (w₁, w₂) and heights (h₁, h₂), i.e. row
  lengths w₁+w₂ (h₁ times) then w₂ (h₂ times).

Behavior notes vs. print_symbolic_two_addable.py /
print_symbolic_two_three_addable.py: output is merged into one script, and
the concrete-diagram cross-checks plus the numeric composition spot check
are new.
"""

from __future__ import annotations

import argparse

import numpy as np
import sympy as sp

from fourier.amatrix import a_matrix_symbolic


def _content(cell: tuple[sp.Expr, sp.Expr]) -> sp.Expr:
    x, y = cell
    return x - y


def _ratio_addable(a, addable, removable) -> sp.Expr:
    """f(λ+a) / (m·f(λ)) through contents only (hook-walk identity)."""
    ca = _content(a)
    num = sp.prod(_content(r) - ca for r in removable)
    den = sp.prod(_content(x) - ca for x in addable if x != a)
    return sp.simplify(num / den)


def _ratio_removable(r, addable_r, removable_r) -> sp.Expr:
    """(m−1)·f(λ−r) / (m·f(λ)) through contents of λ−r (hook-walk identity)."""
    cr = _content(r)
    num = sp.prod(_content(x) - cr for x in addable_r if x != r)
    den = sp.prod(_content(x) - cr for x in removable_r)
    return sp.simplify(num / den)


def two_addable_symbolic(w: sp.Expr, h: sp.Expr) -> sp.Matrix:
    """Exact 2×2 A-matrix of the w×h rectangle (assumes w, h ≥ 2 so that
    removing the corner neither merges blocks nor deletes a row)."""
    a1, a2 = (w, 0), (0, h)
    addable = [a1, a2]
    r1 = (w - 1, h - 1)

    addable_r1 = [a1, r1, a2]
    removable_r1 = [(w - 1, h - 2), (w - 2, h - 1)]

    add_ratios = [_ratio_addable(a, addable, [r1]) for a in addable]
    rem_ratio = _ratio_removable(r1, addable_r1, removable_r1)

    A = sp.zeros(2, 2)
    for i, a in enumerate(addable):
        A[i, 0] = sp.sqrt(add_ratios[i])
        A[i, 1] = sp.sqrt(add_ratios[i] * rem_ratio) / (_content(a) - _content(r1))
    return sp.simplify(A)


def three_addable_symbolic(
    w1: sp.Expr, w2: sp.Expr, h1: sp.Expr, h2: sp.Expr
) -> sp.Matrix:
    """Exact 3×3 A-matrix of the two-block diagram with row lengths w₁+w₂
    (h₁ rows) then w₂ (h₂ rows); assumes all wᵢ, hᵢ ≥ 2."""
    a1, a2, a3 = (w1 + w2, 0), (w2, h1), (0, h1 + h2)
    addable = [a1, a2, a3]
    r1 = (w1 + w2 - 1, h1 - 1)
    r2 = (w2 - 1, h1 + h2 - 1)
    removable = [r1, r2]

    addable_r1 = [a1, r1, a2, a3]
    removable_r1 = [(w1 + w2 - 1, h1 - 2), (w1 + w2 - 2, h1 - 1), r2]
    addable_r2 = [a1, a2, r2, a3]
    removable_r2 = [r1, (w2 - 1, h1 + h2 - 2), (w2 - 2, h1 + h2 - 1)]

    add_ratios = [_ratio_addable(a, addable, removable) for a in addable]
    rem_ratios = [
        _ratio_removable(r1, addable_r1, removable_r1),
        _ratio_removable(r2, addable_r2, removable_r2),
    ]

    A = sp.zeros(3, 3)
    for i, a in enumerate(addable):
        A[i, 0] = sp.sqrt(add_ratios[i])
        for j, r in enumerate(removable, start=1):
            A[i, j] = sp.sqrt(add_ratios[i] * rem_ratios[j - 1]) / (
                _content(a) - _content(r)
            )
    return sp.simplify(A)


def _show(name: str, matrix: sp.Matrix, latex: bool) -> None:
    print(f"\n{name} =")
    sp.pprint(matrix, use_unicode=False)
    if latex:
        print("LaTeX:")
        print(sp.latex(matrix))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-latex", action="store_true", help="Skip the LaTeX printouts."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    latex = not args.no_latex

    w, h = sp.symbols("w h", integer=True, positive=True)
    w1, w2, h1, h2 = sp.symbols("w1 w2 h1 h2", integer=True, positive=True)

    A2 = two_addable_symbolic(w, h)
    A3 = three_addable_symbolic(w1, w2, h1, h2)

    # Cross-check the generic builders against the library on concrete
    # diagrams: the 3×2 rectangle [3,3] and the two-block diagram [5,3,3].
    rect_diff = sp.simplify(A2.subs({w: 3, h: 2}) - a_matrix_symbolic([3, 3]))
    assert rect_diff == sp.zeros(2, 2), rect_diff
    block_diff = sp.simplify(
        A3.subs({w1: 2, w2: 3, h1: 1, h2: 2}) - a_matrix_symbolic([5, 3, 3])
    )
    assert block_diff == sp.zeros(3, 3), block_diff
    print("Cross-checks vs a_matrix_symbolic([3,3]) and ([5,3,3]): OK")

    _show("2-addable A(w,h)", A2, latex)
    _show("2-addable A(w1,h1)", two_addable_symbolic(w1, h1), latex)
    _show("2-addable A(w2,h1+h2)", two_addable_symbolic(w2, h1 + h2), latex)
    _show("3-addable A(w1,w2,h1,h2)", A3, latex)

    # ── composition products (from print_symbolic_two_addable.py) ────────────
    I1 = sp.Matrix([[1]])
    A_w1h1 = two_addable_symbolic(w1, h1)
    A_w2h2 = two_addable_symbolic(w2, h2)

    prod = sp.simplify(sp.diag(A_w1h1, I1) * sp.diag(I1, A_w2h2))
    _show("(A ⊕ 1)(1 ⊕ A')", prod, latex)

    # M = C23(h2,w2) C12(h1,w1) C23(h1-h2, w1+w2), where C12 = A ⊕ 1 and
    # C23 = 1 ⊕ A act on index pairs (1,2) and (2,3).
    C12 = sp.diag(A_w1h1, I1)
    C23_1 = sp.diag(I1, A_w2h2)
    C23_2 = sp.diag(I1, two_addable_symbolic(w1 + w2, h1 - h2))
    M = sp.simplify(C23_1 * C12 * C23_2)
    _show("M = C23(h2,w2) C12(h1,w1) C23(h1-h2, w1+w2)", M, latex)

    # Numeric spot check: is M the 3-addable A-matrix at matching parameters?
    point = {w1: 2, w2: 3, h1: 3, h2: 1}  # needs h1 > h2 for the third factor
    M_num = np.array(M.subs(point).evalf(), dtype=float)
    A3_num = np.array(A3.subs(point).evalf(), dtype=float)
    dist = np.linalg.norm(M_num - A3_num)
    print(f"\n|M - A3| at (w1,w2,h1,h2)=(2,3,3,1): {dist:.4f}  (0 would mean recursion)")


if __name__ == "__main__":
    main()
