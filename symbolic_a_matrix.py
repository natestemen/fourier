import sympy as sp


def content(cell: tuple[sp.Expr, sp.Expr]) -> sp.Expr:
    """Return the content of a cell with coordinates (x, y)."""
    x, y = cell
    return x - y


def ratio_addable(a, addable, removable):
    """Compute d_{lambda+a} / (m d_lambda) using addable/removable of lambda."""
    ca = content(a)
    num = sp.prod(content(r) - ca for r in removable)
    den = sp.prod(content(x) - ca for x in addable if x != a)
    return sp.simplify(num / den)


def ratio_removable(r, addable_r, removable_r):
    """Compute (m-1) d_{lambda-r} / d_lambda using addable/removable of lambda-r."""
    cr = content(r)
    num = sp.prod(content(x) - cr for x in addable_r if x != r)
    den = sp.prod(content(x) - cr for x in removable_r)
    return sp.simplify(num / den)


def build_symbolic_a_matrix():
    # Symbols and assumptions
    w1, w2, w3, h1, h2, h3 = sp.symbols(
        "w_1 w_2 w_3 h_1 h_2 h_3",
        integer=True,
        positive=True,
    )

    # Generic-shape assumptions (see notes below).
    # Given parametrization: h_i > 0, w_1 > 0, w_2 > 0, w_3 >= 1.
    # Row lengths are (w1+w2+w3), (w2+w3), (w3).
    # The formulas below additionally assume removing r_j does not merge blocks
    # (e.g., (w1+w2+w3) - 1 != (w2+w3), (w2+w3) - 1 != w3) and does not delete a row (h_j > 1).

    # Addable and removable cells for lambda.
    a1 = (w1 + w2 + w3, 0)
    a2 = (w2 + w3, h1)
    a3 = (w3, h1 + h2)
    a4 = (0, h1 + h2 + h3)
    addable = [a1, a2, a3, a4]

    r1 = (w1 + w2 + w3 - 1, h1 - 1)
    r2 = (w2 + w3 - 1, h1 + h2 - 1)
    r3 = (w3 - 1, h1 + h2 + h3 - 1)
    removable = [r1, r2, r3]

    # Addable/removable for lambda - r1.
    addable_r1 = [a1, r1, a2, a3, a4]
    removable_r1 = [
        (w1 + w2 + w3 - 1, h1 - 2),  # bottom of top block (requires h1 > 1)
        (w1 + w2 + w3 - 2, h1 - 1),  # bottom of new length-(w1+w2+w3-1) block
        r2,
        r3,
    ]

    # Addable/removable for lambda - r2.
    addable_r2 = [a1, a2, r2, a3, a4]
    removable_r2 = [
        r1,
        (w2 + w3 - 1, h1 + h2 - 2),  # bottom of middle block (requires h2 > 1)
        (w2 + w3 - 2, h1 + h2 - 1),  # bottom of new length-(w2+w3-1) block
        r3,
    ]

    # Addable/removable for lambda - r3.
    addable_r3 = [a1, a2, a3, r3, a4]
    removable_r3 = [
        r1,
        r2,
        (w3 - 1, h1 + h2 + h3 - 2),  # bottom of last block (requires h3 > 1)
        (w3 - 2, h1 + h2 + h3 - 1),  # bottom of new length-(w3-1) block
    ]

    # Ratios for addable cells.
    addable_ratios = [ratio_addable(a, addable, removable) for a in addable]

    # Ratios for removable cells (for each r_j).
    removable_ratios = [
        ratio_removable(r1, addable_r1, removable_r1),
        ratio_removable(r2, addable_r2, removable_r2),
        ratio_removable(r3, addable_r3, removable_r3),
    ]

    # Build A matrix: 4x4 with columns [dummy, r1, r2, r3].
    A = sp.zeros(4, 4)

    for i, a in enumerate(addable):
        A[i, 0] = sp.sqrt(addable_ratios[i])

    for j, r in enumerate([r1, r2, r3], start=1):
        rr = removable_ratios[j - 1]
        for i, a in enumerate(addable):
            A[i, j] = sp.sqrt(addable_ratios[i] * rr) / (content(a) - content(r))

    return A, (w1, w2, w3, h1, h2, h3)


def main():
    A, symbols = build_symbolic_a_matrix()
    print("Symbols:")
    print(symbols)
    print("\nA matrix (symbolic):")
    sp.pprint(A)#, use_unicode=False)


if __name__ == "__main__":
    main()
