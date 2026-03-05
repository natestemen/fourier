#!/usr/bin/env python3
"""Print fully symbolic 2x2 A matrix for 2-addable-cell Young diagram.

For two addable cells, the diagram is a rectangle parameterized by width w and height h.
Addable cells: (w, 0) and (0, h)
Removable cell: (w-1, h-1)
"""
from __future__ import annotations

import sympy as sp


def content(cell):
    x, y = cell
    return x - y


def ratio_addable(a, addable, removable):
    ca = content(a)
    num = sp.prod(content(r) - ca for r in removable)
    den = sp.prod(content(x) - ca for x in addable if x != a)
    return sp.simplify(num / den)


def ratio_removable(r, addable_r, removable_r):
    cr = content(r)
    num = sp.prod(content(x) - cr for x in addable_r if x != r)
    den = sp.prod(content(x) - cr for x in removable_r)
    return sp.simplify(num / den)


def main() -> None:
    w, h = sp.symbols("w h", integer=True, positive=True)

    # For a rectangle w x h: addable cells and removable cells.
    a1 = (w, 0)
    a2 = (0, h)
    addable = [a1, a2]

    r1 = (w - 1, h - 1)
    removable = [r1]

    # For lambda - r1: addable/removable sets
    # After removing r1, addable cells include r1, (w,0), (0,h)
    addable_r1 = [a1, r1, a2]
    removable_r1 = [
        (w - 1, h - 2),  # bottom of width-w block (requires h>1)
        (w - 2, h - 1),  # bottom of new width-(w-1) block
    ]

    # Ratios
    add_ratios = [ratio_addable(a, addable, removable) for a in addable]
    rem_ratio = ratio_removable(r1, addable_r1, removable_r1)

    # Build A: columns [dummy, r1]
    A = sp.zeros(2, 2)
    for i, a in enumerate(addable):
        A[i, 0] = sp.sqrt(add_ratios[i])
        A[i, 1] = sp.sqrt(add_ratios[i] * rem_ratio) / (content(a) - content(r1))

    print("A(w,h) =")
    sp.pprint(A, use_unicode=False)
    print("\nLaTeX:")
    print(sp.latex(A))

    # Build A and A' with independent parameters (w1,h1) and (w2,h2)
    w1, h1, w2, h2 = sp.symbols("w1 h1 w2 h2", integer=True, positive=True)
    A1 = A.subs({w: w1, h: h1})
    A2 = A.subs({w: w2, h: h2})

    I1 = sp.Matrix([[1]])
    Aop1 = sp.diag(A1, I1)  # A ⊕ 1
    IopA2 = sp.diag(I1, A2)  # 1 ⊕ A'
    prod = sp.simplify(Aop1 * IopA2)

    print("\n(A ⊕ 1)(1 ⊕ A') =")
    sp.pprint(prod, use_unicode=False)
    print("\nLaTeX:")
    print(sp.latex(prod))

    # Product M = C23(h2,w2) C12(h1,w1) C23(h1-h2, w1+w2)
    # where C12 = A ⊕ 1 and C23 = 1 ⊕ A acting on indices (1,2) and (2,3).
    w1, h1, w2, h2 = sp.symbols("w1 h1 w2 h2", integer=True, positive=True)
    A_h1w1 = A.subs({w: w1, h: h1})
    A_h2w2 = A.subs({w: w2, h: h2})
    A_h1mh2_w1pw2 = A.subs({w: w1 + w2, h: h1 - h2})

    C12 = sp.diag(A_h1w1, I1)
    C23_1 = sp.diag(I1, A_h2w2)
    C23_2 = sp.diag(I1, A_h1mh2_w1pw2)

    M = sp.simplify(C23_1 * C12 * C23_2)
    print("\nM = C23(h2,w2) C12(h1,w1) C23(h1-h2, w1+w2) =")
    sp.pprint(M, use_unicode=False)
    print("\nLaTeX:")
    print(sp.latex(M))


if __name__ == "__main__":
    main()
