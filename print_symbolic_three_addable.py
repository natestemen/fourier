#!/usr/bin/env python3
"""Print fully symbolic 3x3 A matrix for a 3-addable-cell Young diagram.

Parametrization:
  addable cells = {(w1+w2, 0), (w1, h2), (0, h1)}
  removable cells = {(w1+w2-1, h2-1), (w1-1, h1-1)}
Assumes h1 > h2 > 0 and w1, w2 > 0.
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
    w1, w2, h1, h2 = sp.symbols("w1 w2 h1 h2", integer=True, positive=True)

    # Addable / removable for lambda
    a1 = (w1 + w2, 0)
    a2 = (w1, h2)
    a3 = (0, h1)
    addable = [a1, a2, a3]

    r1 = (w1 + w2 - 1, h2 - 1)
    r2 = (w1 - 1, h1 - 1)
    removable = [r1, r2]

    # For lambda - r1
    addable_r1 = [a1, r1, a2, a3]
    removable_r1 = [
        (w1 + w2 - 1, h2 - 2),  # bottom of top block (requires h2 > 1)
        (w1 + w2 - 2, h2 - 1),  # bottom of new width-(w1+w2-1) block
        r2,
    ]

    # For lambda - r2
    addable_r2 = [a1, a2, r2, a3]
    removable_r2 = [
        r1,
        (w1 - 1, h1 - 2),      # bottom of tall block (requires h1 > 1)
        (w1 - 2, h1 - 1),      # bottom of new width-(w1-1) block
    ]

    # Ratios
    add_ratios = [ratio_addable(a, addable, removable) for a in addable]
    rem_ratio_1 = ratio_removable(r1, addable_r1, removable_r1)
    rem_ratio_2 = ratio_removable(r2, addable_r2, removable_r2)

    # Build A: columns [dummy, r1, r2]
    A = sp.zeros(3, 3)
    for i, a in enumerate(addable):
        A[i, 0] = sp.sqrt(add_ratios[i])
        A[i, 1] = sp.sqrt(add_ratios[i] * rem_ratio_1) / (content(a) - content(r1))
        A[i, 2] = sp.sqrt(add_ratios[i] * rem_ratio_2) / (content(a) - content(r2))

    print("A(w1,w2,h1,h2) =")
    sp.pprint(A, use_unicode=False)
    print("\nLaTeX:")
    print(sp.latex(A))


if __name__ == "__main__":
    main()
