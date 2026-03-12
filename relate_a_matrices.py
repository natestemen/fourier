"""Explore algebraic relationships between A(2,1), A(2,), A(1,1), and A(1,)."""
import sympy as sp
from symbolic_a_matrix import build_symbolic_a_matrix_for_partition

A1  = build_symbolic_a_matrix_for_partition([1])
A2  = build_symbolic_a_matrix_for_partition([2])
A11 = build_symbolic_a_matrix_for_partition([1, 1])
A21 = build_symbolic_a_matrix_for_partition([2, 1])

print("=== Basic structure ===")
for name, M in [("A(1,)", A1), ("A(2,)", A2), ("A(1,1)", A11), ("A(2,1)", A21)]:
    print(f"  {name}: det={sp.simplify(M.det())}, symmetric={M==M.T}, involution={M**2==sp.eye(M.shape[0])}")
print()

# ─── Key relationship: A(1,1) = A(1,)*A(2,)*A(1,) ────────────────────────────
conj = sp.simplify(A1 * A2 * A1)
print("=== A(1,) * A(2,) * A(1,) ===")
sp.pprint(conj)
print(f"  == A(1,1)? {conj == A11}")
print()

# Also check the other natural conjugation
conj2 = sp.simplify(A1 * A11 * A1)
print("=== A(1,) * A(1,1) * A(1,) ===")
sp.pprint(conj2)
print(f"  == A(2,)? {conj2 == A2}")
print()

# So A(1,1) and A(2,) are conjugate under A(1,), which makes sense:
# both are level-2 partitions, and A(1,) is the level-1 matrix.
print("=== Can we extend this to A(2,1)? ===")
print("A(2,1) is 3x3, so we need 3x3 extensions of A(2,) and/or A(1,).")
print()

# Natural 3x3 embeddings: A(2,) acting on two of the three 'addable content' axes
# The addable contents of (2,1) are {2, 0, -2} (row indices 0,1,2).
# A(2,) has addable contents {2, -1}, A(1,1) has {1, -2}.
# There is no perfect row-alignment, but we can try all (i,j) embeddings.

def embed(M, i, j, n=3):
    E = sp.eye(n)
    E[i,i]=M[0,0]; E[i,j]=M[0,1]; E[j,i]=M[1,0]; E[j,j]=M[1,1]
    return E

pairs = [(0,1),(0,2),(1,2)]

# Check: A(2,1) = E(A1,ij) * E(A2,kl) * E(A1,mn) for any embeddings?
print("Searching: A(2,1) = embed(A1,ij) * embed(A2,kl) * embed(A1,mn) ...")
found = False
for p1 in pairs:
    for p2 in pairs:
        for p3 in pairs:
            P = sp.simplify(embed(A1,*p1)*embed(A2,*p2)*embed(A1,*p3))
            if P == A21:
                print(f"  FOUND: A1 in {p1}, A2 in {p2}, A1 in {p3}")
                found = True
if not found:
    print("  (none found)")
print()

# Try with A(1,1) too (since A(1,1) = A1*A2*A1):
print("Searching: A(2,1) = embed(A11,ij) * embed(A2,kl) * embed(A11,mn) ...")
found = False
for p1 in pairs:
    for p2 in pairs:
        for p3 in pairs:
            P = sp.simplify(embed(A11,*p1)*embed(A2,*p2)*embed(A11,*p3))
            if P == A21:
                print(f"  FOUND: A11 in {p1}, A2 in {p2}, A11 in {p3}")
                found = True
if not found:
    print("  (none found)")
print()

# Try 4-fold products:
print("Searching: A(2,1) = E1 * E2 * E11 * E2 (4 factors) ...")
found = False
for p1 in pairs:
    for p2 in pairs:
        for p3 in pairs:
            for p4 in pairs:
                for combo in [
                    (embed(A1,*p1),embed(A2,*p2),embed(A11,*p3),embed(A2,*p4)),
                    (embed(A2,*p1),embed(A1,*p2),embed(A2,*p3),embed(A1,*p4)),
                    (embed(A2,*p1),embed(A11,*p2),embed(A2,*p3),embed(A11,*p4)),
                ]:
                    P = sp.simplify(combo[0]*combo[1]*combo[2]*combo[3])
                    if P == A21:
                        print(f"  FOUND: {p1},{p2},{p3},{p4}")
                        sp.pprint(P)
                        found = True
if not found:
    print("  (none found)")
print()

# ─── Alternative: check column-by-column factorization ────────────────────────
# A(2,1)[:,0] = dummy col, A(2,1)[:,1] = (1,1)-branch, A(2,1)[:,2] = (2,)-branch
# A(1,1)[:,0] = dummy col of A(1,1), A(1,1)[:,1] = (1,)-branch
# A(2,)[:,0]  = dummy col of A(2,),  A(2,)[:,1]  = (1,)-branch
#
# Is there a "stitching" relationship where A(2,1) columns come from
# A(1,1) and A(2,) columns multiplied by A(1,)?

print("=== Column-level factorization ===")
# Hypothesis: A21[:,1] = (3x2 extension of A11) * A1 * something?
# Try: is A21[:,1] = E(A11,some_pair) * A21[:,2]  (relating the two branch cols)?
for p in pairs:
    v = sp.simplify(embed(A11,*p) * A21.col(2))
    if v == A21.col(1):
        print(f"embed(A11,{p}) * A21[:,2] == A21[:,1]")

for p in pairs:
    v = sp.simplify(embed(A2,*p) * A21.col(1))
    if v == A21.col(2):
        print(f"embed(A2,{p}) * A21[:,1] == A21[:,2]")

for p in pairs:
    v = sp.simplify(embed(A1,*p) * A21.col(1))
    if v == A21.col(2):
        print(f"embed(A1,{p}) * A21[:,1] == A21[:,2]")
    v = sp.simplify(embed(A1,*p) * A21.col(2))
    if v == A21.col(1):
        print(f"embed(A1,{p}) * A21[:,2] == A21[:,1]")
print()

# Check: does A(2,1) * embed(A1,ij) = something nice?
print("A(2,1) * embed(A1, *) for each pair:")
for p in pairs:
    prod = sp.simplify(A21 * embed(A1, *p))
    print(f"  A21 * embed(A1,{p}) =")
    sp.pprint(prod)
    print()

# Check: embed(A1,*) * A(2,1) * embed(A1,*)?
print("=== Conjugates of A(2,1) by embed(A1,*) ===")
for p in pairs:
    E = embed(A1,*p)
    conj = sp.simplify(E * A21 * E)
    print(f"  embed(A1,{p}) * A21 * embed(A1,{p}) =")
    sp.pprint(conj)
    print(f"  Is this A(2,1)? {conj == A21}")
    print()
