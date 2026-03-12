import sympy as sp
import sys


def content(cell: tuple[sp.Expr, sp.Expr]) -> sp.Expr:
    """Return the content of a cell with coordinates (x, y)."""
    x, y = cell
    return x - y


def ratio_addable(a, addable, removable):
    """Compute d_{lambda+a} / (m d_lambda) using addable/removable of lambda."""
    ca = content(a)
    num = sp.prod(content(r) - ca for r in removable)
    den = sp.prod(content(x) - ca for x in addable if x != a)
    return num / den


def ratio_removable(r, addable_r, removable_r):
    """Compute (m-1) d_{lambda-r} / d_lambda using addable/removable of lambda-r."""
    cr = content(r)
    num = sp.prod(content(x) - cr for x in addable_r if x != r)
    den = sp.prod(content(x) - cr for x in removable_r)
    return num / den


def _partition_addable_cells(partition: list[int]) -> list[tuple[sp.Expr, sp.Expr]]:
    rows = [sp.Integer(x) for x in partition if x > 0]
    addable: list[tuple[sp.Expr, sp.Expr]] = []
    for y, row_len in enumerate(rows):
        if y == 0 or row_len < rows[y - 1]:
            addable.append((row_len, sp.Integer(y)))
    addable.append((sp.Integer(0), sp.Integer(len(rows))))
    return addable


def _partition_removable_cells(partition: list[int]) -> list[tuple[sp.Expr, sp.Expr]]:
    rows = [sp.Integer(x) for x in partition if x > 0]
    removable: list[tuple[sp.Expr, sp.Expr]] = []
    for y, row_len in enumerate(rows):
        is_last = y == len(rows) - 1
        next_len = rows[y + 1] if not is_last else sp.Integer(-1)
        if is_last or row_len > next_len:
            removable.append((row_len - 1, sp.Integer(y)))
    return removable


def _remove_cell(partition: list[int], cell: tuple[sp.Expr, sp.Expr]) -> list[int]:
    x, y = cell
    y_int = int(y)
    rows = [int(v) for v in partition if v > 0]
    rows[y_int] -= 1
    rows = [v for v in rows if v > 0]
    return rows


def _add_cell(partition: list[int], cell: tuple[sp.Expr, sp.Expr]) -> list[int]:
    x, y = cell
    y_int = int(y)
    rows = [int(v) for v in partition if v > 0]
    if y_int == len(rows):
        rows.append(1)
    else:
        rows[y_int] += 1
    return rows


def _hook_length_count(partition: list[int]) -> sp.Expr:
    rows = [int(v) for v in partition if v > 0]
    n = sum(rows)
    if n == 0:
        return sp.Integer(1)
    hooks = []
    for y, row_len in enumerate(rows):
        for x in range(row_len):
            arm = row_len - x - 1
            leg = sum(1 for yy in range(y + 1, len(rows)) if rows[yy] > x)
            hooks.append(arm + leg + 1)
    return sp.factorial(n) / sp.prod(sp.Integer(h) for h in hooks)


def build_symbolic_a_matrix_for_partition(partition: list[int]) -> sp.Matrix:
    addable = _partition_addable_cells(partition)
    removable = _partition_removable_cells(partition)

    f_lambda = _hook_length_count(partition)
    m = sum(partition) + 1

    f_add = []
    for a in addable:
        f_add.append(_hook_length_count(_add_cell(partition, a)))

    f_rem = []
    for r in removable:
        f_rem.append(_hook_length_count(_remove_cell(partition, r)))

    A = sp.zeros(len(addable), len(removable) + 1)
    for i, a in enumerate(addable):
        A[i, 0] = sp.sqrt(f_add[i] / (m * f_lambda))

    for j, r in enumerate(removable, start=1):
        rr = (m - 1) * f_rem[j - 1] / (m * f_lambda)
        for i, a in enumerate(addable):
            A[i, j] = sp.sqrt(f_add[i] * rr / f_lambda) / (content(a) - content(r))

    return sp.simplify(A)


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


def build_symbolic_a_matrix_8addable():
    """Build symbolic 8x8 A matrix for 8-addable Young diagrams.

    Parameters:
      w_1..w_7, h_1..h_7 (positive integers).

    Addable cells:
      a_1 = (sum_{i=1}^7 w_i, 0)
      a_2 = (sum_{i=2}^7 w_i, h_1)
      ...
      a_7 = (w_7, sum_{i=1}^6 h_i)
      a_8 = (0, sum_{i=1}^7 h_i)

    Removable cells:
      r_j = (sum_{i=j}^7 w_i - 1, sum_{i=1}^j h_i - 1) for j=1..7.

    Assumptions for the removable/addable structure of lambda - r_j:
      w_i >= 2 and h_i >= 2 so that removing r_j does not merge blocks
      or delete a row.
    """
    w = sp.symbols("w_1:8", integer=True, positive=True)
    h = sp.symbols("h_1:8", integer=True, positive=True)

    # Prefix sums of heights and suffix sums of widths.
    h_prefix = [sp.Integer(0)]
    for idx in range(7):
        h_prefix.append(h_prefix[-1] + h[idx])
    w_suffix = [sp.Integer(0)] * 8
    for idx in range(6, -1, -1):
        w_suffix[idx] = w_suffix[idx + 1] + w[idx]

    # Addable cells for lambda.
    addable = [(w_suffix[j], h_prefix[j]) for j in range(7)]
    addable.append((sp.Integer(0), h_prefix[7]))

    # Removable cells for lambda.
    removable = [(w_suffix[j] - 1, h_prefix[j + 1] - 1) for j in range(7)]

    addable_ratios = [ratio_addable(a, addable, removable) for a in addable]

    # Ratios for removable cells (for each r_j).
    removable_ratios = []
    for j, rj in enumerate(removable):
        addable_rj = addable[: j + 1] + [rj] + addable[j + 1 :]
        xj, yj = rj
        removable_rj = (
            removable[:j]
            + [(xj, yj - 1), (xj - 1, yj)]
            + removable[j + 1 :]
        )
        removable_ratios.append(ratio_removable(rj, addable_rj, removable_rj))

    # Build A matrix: 8x8 with columns [dummy, r1..r7].
    A = sp.zeros(8, 8)
    for i, a in enumerate(addable):
        A[i, 0] = sp.sqrt(addable_ratios[i])

    for j, r in enumerate(removable, start=1):
        rr = removable_ratios[j - 1]
        for i, a in enumerate(addable):
            A[i, j] = sp.sqrt(addable_ratios[i] * rr) / (content(a) - content(r))

    return A, (*w, *h)


def main():
    if len(sys.argv) > 1:
        raw = " ".join(sys.argv[1:]).replace(",", " ")
        parts = [int(x) for x in raw.split() if x.strip()]
        if not parts:
            raise SystemExit("Provide a partition like: 2 1 (or 2,1)")
        A = build_symbolic_a_matrix_for_partition(parts)
        print(f"Partition {tuple(parts)} A matrix (symbolic):")
        sp.pprint(A, use_unicode=False)
        print("\nLaTeX:")
        print(sp.latex(A))
        return

    A, symbols = build_symbolic_a_matrix()
    print("Symbols:")
    print(symbols)
    print("\nA matrix (symbolic):")
    sp.pprint(A)  # use_unicode=False


if __name__ == "__main__":
    main()
