"""Symbolic A matrix for the partition (2, 1)."""
import sympy as sp

# Content of a cell (x=col, y=row): x - y
def content(cell):
    x, y = cell
    return x - y

# Partition (2, 1): 3 addable cells, 2 removable cells
# Addable: end of each row + new row
addable  = [(2, 0), (1, 1), (0, 2)]
removable = [(1, 0), (0, 1)]

# Number of standard tableaux (hook-length formula)
# f(2,1) = 3!/(3·1·1) = 2
# f(3,1) = 4!/(4·2·1·1) = 3
# f(2,2) = 4!/(3·2·2·1) = 2
# f(2,1,1) = 4!/(4·1·2·1) = 3
# f(1,1) = 2!/(2·1) = 1
# f(2,) = 2!/(2·1) = 1
f_lam = sp.Integer(2)
f_add = {(2, 0): sp.Integer(3), (1, 1): sp.Integer(2), (0, 2): sp.Integer(3)}
f_rem = {(1, 0): sp.Integer(1), (0, 1): sp.Integer(1)}

m = sp.Integer(4)  # diagram.size + 1 = 3 + 1

A = sp.zeros(3, 3)

# Column 0: dummy (no removal)
for j, a in enumerate(addable):
    A[j, 0] = sp.sqrt(f_add[a] / (m * f_lam))

# Columns 1, 2: each removable cell
for i, r in enumerate(removable, start=1):
    rr = f_rem[r]
    for j, a in enumerate(addable):
        factor = sp.sqrt((m - 1) * f_add[a] * rr / (m * f_lam**2))
        A[j, i] = sp.simplify(factor / (content(a) - content(r)))

print("A matrix for partition (2, 1):")
sp.pprint(A)
print()
print("Simplified entries:")
for i in range(3):
    for j in range(3):
        print(f"  A[{i},{j}] = {A[i,j]}")
