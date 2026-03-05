#!/usr/bin/env python3
"""Print symbolic A matrices for 2-addable and 3-addable Young diagrams.

2-addable (rectangle): parameters (w,h)
3-addable: parameters (w1,w2,h1,h2) with
  addable cells = {(w1+w2, 0), (w2, h1), (0, h1+h2)}
  removable cells = {(w1+w2-1, h1-1), (w2-1, h1+h2-1)}

Also prints the 2x2 matrices for (w1,h1) and (w2,h2).
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


def build_two_addable_A(w, h):
    a1 = (w, 0)
    a2 = (0, h)
    addable = [a1, a2]

    r1 = (w - 1, h - 1)
    removable = [r1]

    addable_r1 = [a1, r1, a2]
    removable_r1 = [
        (w - 1, h - 2),
        (w - 2, h - 1),
    ]

    add_ratios = [ratio_addable(a, addable, removable) for a in addable]
    rem_ratio = ratio_removable(r1, addable_r1, removable_r1)

    A = sp.zeros(2, 2)
    for i, a in enumerate(addable):
        A[i, 0] = sp.sqrt(add_ratios[i])
        A[i, 1] = sp.sqrt(add_ratios[i] * rem_ratio) / (content(a) - content(r1))
    return sp.simplify(A)


def build_three_addable_A(w1, w2, h1, h2):
    # addable cells = {(w1+w2, 0), (w2, h1), (0, h1+h2)}
    a1 = (w1 + w2, 0)
    a2 = (w2, h1)
    a3 = (0, h1 + h2)
    addable = [a1, a2, a3]

    # removable cells = {(w1+w2-1, h1-1), (w2-1, h1+h2-1)}
    r1 = (w1 + w2 - 1, h1 - 1)
    r2 = (w2 - 1, h1 + h2 - 1)
    removable = [r1, r2]

    # For lambda - r1
    addable_r1 = [a1, r1, a2, a3]
    removable_r1 = [
        (w1 + w2 - 1, h1 - 2),
        (w1 + w2 - 2, h1 - 1),
        r2,
    ]

    # For lambda - r2
    addable_r2 = [a1, a2, r2, a3]
    removable_r2 = [
        r1,
        (w2 - 1, h1 + h2 - 2),
        (w2 - 2, h1 + h2 - 1),
    ]

    add_ratios = [ratio_addable(a, addable, removable) for a in addable]
    rem_ratio_1 = ratio_removable(r1, addable_r1, removable_r1)
    rem_ratio_2 = ratio_removable(r2, addable_r2, removable_r2)

    A = sp.zeros(3, 3)
    for i, a in enumerate(addable):
        A[i, 0] = sp.sqrt(add_ratios[i])
        A[i, 1] = sp.sqrt(add_ratios[i] * rem_ratio_1) / (content(a) - content(r1))
        A[i, 2] = sp.sqrt(add_ratios[i] * rem_ratio_2) / (content(a) - content(r2))
    return sp.simplify(A)


def main() -> None:
    w, h = sp.symbols("w h", integer=True, positive=True)
    w1, w2, h1, h2 = sp.symbols("w1 w2 h1 h2", integer=True, positive=True)

    A2 = build_two_addable_A(w, h)
    A2_w1h1 = build_two_addable_A(w1, h1)
    A2_w2h2 = build_two_addable_A(w2, h1 + h2)
    A3 = build_three_addable_A(w1, w2, h1, h2)

    print("2-addable A(w,h) =")
    sp.pprint(A2, use_unicode=False)
    print("\nLaTeX:")
    print(sp.latex(A2))

    print("\n2-addable A(w1,h1) =")
    sp.pprint(A2_w1h1, use_unicode=False)
    print("\nLaTeX:")
    print(sp.latex(A2_w1h1))

    print("\n2-addable A(w2,h1+h2) =")
    sp.pprint(A2_w2h2, use_unicode=False)
    print("\nLaTeX:")
    print(sp.latex(A2_w2h2))

    print("\n3-addable A(w1,w2,h1,h2) =")
    sp.pprint(A3, use_unicode=False)
    print("\nLaTeX:")
    print(sp.latex(A3))


if __name__ == "__main__":
    main()
