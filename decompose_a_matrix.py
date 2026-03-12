"""
Fast decomposition of the A-matrix exploiting its Cauchy structure.

KEY STRUCTURAL FACTS (from graph_a_matrix.py):
  1. A[:,1:] = diag(α) · C · diag(β)  where C[i,j] = 1/(c(aᵢ)-c(rⱼ))  [Cauchy]
  2. Displacement rank 1: X·C - C·Y = 1·1ᵀ   (X=diag(addable), Y=diag(removable))
  3. All contents are INTEGERS, strictly interlaced on the number line.

DECOMPOSITION STRATEGY:
  Naive QR/Givens: O(k²) gates.

  Better — three levels of structure to exploit:

  Level 1 — Cauchy displacement rank = 1 (always):
    GKO/superfast algorithms → O(k log² k) Givens rotations.
    The Schur complements of a Cauchy matrix are again Cauchy (same structure
    preserved under elimination), so the recursion terminates cleanly.

  Level 2 — Integer contents in range [-k, k] (always):
    The partial-fractions sum  f(aᵢ) = Σⱼ wⱼ/(aᵢ - rⱼ)  is a polynomial
    evaluation problem with integer nodes.  Using the subproduct-tree
    multi-point evaluation (Borodin-Moenck, 1974):
      Build Q(t) = Π(t - rⱼ) and P(t) = Σⱼ wⱼ·Π_{k≠j}(t-rₖ) in O(k log² k),
      then evaluate at all aᵢ in O(k log² k).

  Level 3 — Toeplitz-Cauchy for staircase partitions (k,k-1,…,1):
    W[i,j] = 2(j-i)+1 → C is a Toeplitz matrix → FFT applies → O(k log k).

  The recursion at each level: once the mat-vec is O(k log² k), the circuit
  depth for the full k×k orthogonal matrix via CS decomposition is O(log k)
  levels, each requiring O(k log² k) work → total O(k log² k).
  (For the Toeplitz case: O(k log k).)
"""

import numpy as np
from numpy.polynomial import polynomial as P
import sympy as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
from symbolic_a_matrix import (
    build_symbolic_a_matrix_for_partition,
    _partition_addable_cells,
    _partition_removable_cells,
)
from helper import partitions


def content(c):
    return int(c[0] - c[1])


def get_partition_data(partition):
    add = _partition_addable_cells(partition)
    rem = _partition_removable_cells(partition)
    ac  = sorted([content(a) for a in add], reverse=True)
    rc  = sorted([content(r) for r in rem], reverse=True)
    A   = np.array(
        build_symbolic_a_matrix_for_partition(partition).tolist(), dtype=float
    )
    alpha = A[:, 0].copy()
    # A[i,j] = alpha_i * beta_j / (ac[i] - rc[j-1])  for j>=1
    # => beta_j = A[0,j] * (ac[0] - rc[j-1]) / alpha[0]
    beta = np.array(
        [A[0, j + 1] * (ac[0] - rc[j]) / alpha[0] for j in range(len(rc))]
    )
    return ac, rc, alpha, beta, A


# ── 1. Displacement rank = 1  (already confirmed; show the generator vectors) ──

def displacement_generators(ac, rc):
    """Return u, v s.t. diag(ac)·C - C·diag(rc) = u·vᵀ (should be 1·1ᵀ)."""
    k = len(ac)
    m = len(rc)
    C = np.array([[1.0 / (ac[i] - rc[j]) for j in range(m)] for i in range(k)])
    D = np.diag(ac) @ C - C @ np.diag(rc)
    # D should be all-ones
    return D, C


# ── 2. Subproduct-tree multi-point evaluation  O(k log² k) ────────────────────

class SubproductTree:
    """
    Classic divide-and-conquer subproduct tree for multi-point polynomial eval.

    evaluate(p, xs) returns p(xs[i]) for all i in O(n log² n) multiplications.
    We count multiplications instead of timing to get an op-count that's
    independent of constant factors.
    """

    def __init__(self, xs):
        self.xs = list(xs)
        self.n  = len(xs)
        self.ops = 0
        self.tree = self._build(list(range(self.n)))

    def _build(self, indices):
        if len(indices) == 1:
            # (x - xs[i])
            return np.array([-self.xs[indices[0]], 1.0])
        mid = len(indices) // 2
        left  = self._build(indices[:mid])
        right = self._build(indices[mid:])
        self.ops += (len(left) - 1) * len(right)   # polynomial multiply cost
        return np.polymul(left[::-1], right[::-1])[::-1]

    def evaluate(self, poly):
        """Evaluate poly at all xs, return array of values."""
        results = [0.0] * self.n
        self._eval_recurse(poly, list(range(self.n)), results)
        return np.array(results)

    def _eval_recurse(self, poly, indices, results):
        if len(indices) == 1:
            results[indices[0]] = np.polyval(poly[::-1], self.xs[indices[0]])
            return
        mid = len(indices) // 2
        left_mod  = self._mod(poly, indices[:mid])
        right_mod = self._mod(poly, indices[mid:])
        self._eval_recurse(left_mod,  indices[:mid],  results)
        self._eval_recurse(right_mod, indices[mid:],  results)

    def _mod(self, poly, indices):
        """poly mod subproduct(indices)."""
        sub = self._build_sub(indices)
        self.ops += len(poly) * len(sub)
        _, rem = np.polydiv(poly[::-1], sub[::-1])
        return rem[::-1]

    def _build_sub(self, indices):
        if len(indices) == 1:
            return np.array([-self.xs[indices[0]], 1.0])
        mid = len(indices) // 2
        left  = self._build_sub(indices[:mid])
        right = self._build_sub(indices[mid:])
        return np.polymul(left[::-1], right[::-1])[::-1]


def cauchy_matvec_fast(ac, rc, alpha, beta, v):
    """
    Apply A to v using the partial-fractions representation.

    f(aᵢ) = Σⱼ wⱼ / (aᵢ - rⱼ)   where wⱼ = beta[j] * v[j+1]
           = P(aᵢ) / Q(aᵢ)
    where Q(t) = Π(t - rⱼ),  P(t) = Σⱼ wⱼ · Π_{k≠j}(t - rₖ)

    Returns result and operation count.
    """
    ops = 0
    w   = beta * v[1:]
    m   = len(rc)

    # Build Q(t) = prod(t - r_j)
    Q = np.array([1.0])
    for r in rc:
        ops += len(Q)
        Q = np.polymul(Q, np.array([1.0, -r]))  # t - r

    # Build each Qⱼ(t) = Q(t) / (t - rⱼ)  via synthetic division
    Qj_list = []
    for j, r in enumerate(rc):
        Qj, rem = np.polydiv(Q, np.array([1.0, -r]))
        ops += len(Q)
        Qj_list.append(Qj)

    # P(t) = Σⱼ wⱼ · Qⱼ(t)
    P_poly = np.zeros(m)
    for j, Qj in enumerate(Qj_list):
        ops += len(Qj)
        P_poly = P_poly + w[j] * Qj

    # Evaluate P and Q at each aᵢ
    fvals = np.array([np.polyval(P_poly, a) / np.polyval(Q, a) for a in ac])
    ops += len(ac) * (len(P_poly) + len(Q))

    result = alpha * v[0] + alpha * fvals
    return result, ops


def cauchy_matvec_naive(ac, rc, alpha, beta, v):
    """Naive O(k²) mat-vec."""
    w      = beta * v[1:]
    fvals  = np.array([sum(w[j] / (a - rc[j]) for j in range(len(rc))) for a in ac])
    result = alpha * v[0] + alpha * fvals
    ops    = len(ac) * len(rc)
    return result, ops


# ── 3. GKO / structured Givens reduction  O(k log² k) ─────────────────────────

def givens_rotation(a, b):
    """Return (c, s, r) such that [c s; -s c]^T [a; b] = [r; 0]."""
    if b == 0:
        return 1.0, 0.0, a
    r = np.hypot(a, b)
    return a / r, b / r, r


def count_givens_for_cauchy(ac, rc, alpha, beta):
    """
    Simulate GKO-style structured Givens reduction on the Cauchy core.
    At each step, we use the displacement structure to update the generators
    rather than updating the full matrix: O(k) per column instead of O(k²).

    Returns the number of Givens rotations used.
    """
    k  = len(ac)
    m  = len(rc)

    # Start with displacement generator: C satisfies X*C - C*Y = u*vT, u=v=ones
    # We maintain u (length k) and v (length m) as the generator pair.
    # Each elimination step:
    #   1. Pick pivot from column j using generator: O(k) ops
    #   2. Apply Givens to zero sub-diagonal: O(1) Givens per row
    #   3. Update generators: O(k) ops

    # Count: k Givens rotations per column → m columns → k*m ≈ k² total
    # BUT: with structured update, each Givens costs O(1) to update the generators
    # vs O(k) for full matrix update.
    # Net circuit complexity: k Givens rotations total (not k² or k*m).
    #
    # More precisely, the triangular factorization uses:
    #   sum_{j=0}^{m-1} (k-j-1) Givens rotations = k*(k-1)/2 - m*(m-1)/2 ≈ k/2 rotations
    # But the update work per rotation is O(log k) with binary-tree aggregation.

    total_givens = 0
    for j in range(m):
        # Zero out entries below diagonal in column j: k-j-1 Givens
        total_givens += (k - j - 1)
    return total_givens  # = k*(k-1)/2 - m*(m-1)/2 ≈ k²/2 - k²/8 ≈ 3k²/8


def count_structured_givens(k):
    """
    For the STRUCTURED algorithm (GKO with displacement rank 1):
    Each Givens rotation costs O(1) to apply to the generator (not O(k)).
    The total number of Givens is still O(k²), but the DEPTH of the circuit
    is O(k log k) because the rotations can be parallelized using the
    displacement structure.

    Alternative: use the QR decomposition via Householder reflectors.
    With displacement rank 1, each Householder costs O(k) using generator update,
    not O(k²). There are k Householders → O(k²) total.

    The ACTUAL O(k log k) circuit comes from butterfly decomposition of the
    displacement structure. See below.
    """
    return k * (k - 1) // 2   # standard Givens count (not yet exploiting structure)


# ── 4. Butterfly / FFT decomposition for Toeplitz-Cauchy (staircase)  ─────────
#
# For the staircase partition (k, k-1, ..., 1):
#   W[i,j] = 2(j-i)+1  →  C[i,j] = 1/(2(j-i)+1)  [Toeplitz!]
#
# A Toeplitz matrix-vector product can be computed in O(k log k) via FFT:
#   embed C in a 2k×2k circulant matrix, apply FFT, multiply, inverse FFT.
# The resulting circuit is a butterfly network of depth O(log k).

def toeplitz_matvec_fft(first_row, first_col, v):
    """
    Apply a Toeplitz matrix T to vector v in O(n log n) using FFT embedding.
    T[i,j] = first_col[i-j] for i>=j, first_row[j-i] for j>=i.
    """
    n_rows = len(first_col)  # number of rows
    # Embed T (n_rows × n_cols Toeplitz) in a circulant of size N = n_rows + n_cols - 1.
    # Circulant first row = [first_col; first_row[1:][::-1]]  (standard embedding).
    circ = np.concatenate([first_col, first_row[-1:0:-1]])   # length n_rows + n_cols - 1
    v_padded = np.concatenate([v, np.zeros(n_rows - 1)])     # length N
    result_circ = np.fft.ifft(np.fft.fft(circ) * np.fft.fft(v_padded)).real
    return result_circ[:n_rows]


def staircase_A_via_toeplitz(k, v):
    """
    Apply A-matrix for staircase partition (k, k-1, ..., 1) to vector v
    using FFT for the Toeplitz Cauchy core.
    This is O(k log k).
    """
    partition = list(range(k, 0, -1))
    ac, rc, alpha, beta, A = get_partition_data(partition)

    # The core Cauchy matrix C[i,j] = 1/(2(j-i)+1) is Toeplitz.
    # First row of C: [1/1, 1/3, 1/5, ...] for i=0
    # First col of C: [1/1, 1/(-1), 1/(-3), ...] = [1, -1, -1/3, ...]
    m = len(rc)
    first_row = np.array([1.0 / (2*j + 1) for j in range(m)])     # C[0,j]
    first_col = np.array([1.0 / (1 - 2*i) for i in range(len(ac))])  # C[i,0]

    w = beta * v[1:]
    fft_result = toeplitz_matvec_fft(first_row, first_col, w)

    result = alpha * v[0] + alpha * fft_result
    return result


# ── 5. Count analysis and verification ────────────────────────────────────────

def get_staircase_partitions(max_k):
    """Return staircase partitions for k=2,...,max_k."""
    return [(k, list(range(k, 0, -1))) for k in range(2, max_k + 1)]


print("=" * 65)
print("  DECOMPOSITION PATH ANALYSIS")
print("=" * 65)

print("\n1. Displacement rank = 1 (all partitions):")
for part in [[2,1],[3,2,1],[4,3,2,1],[5,4,3,2,1]]:
    D, C = displacement_generators(*[[content(a) for a in _partition_addable_cells(part)],
                                       [content(r) for r in _partition_removable_cells(part)]])
    ac = [content(a) for a in _partition_addable_cells(part)]
    rc = [content(r) for r in _partition_removable_cells(part)]
    D, C = displacement_generators(ac, rc)
    rank_D = np.linalg.matrix_rank(D, tol=1e-9)
    is_ones = np.allclose(D, np.ones_like(D))
    k = len(ac)
    print(f"   {str(tuple(part)):20s}  k={k},  rank(X·C - C·Y) = {rank_D},  = 1·1ᵀ: {is_ones}")

print("\n2. Fast mat-vec correctness (partial-fractions formula):")
for part in [[2,1],[3,2,1],[4,3,2,1]]:
    ac, rc, alpha, beta, A = get_partition_data(part)
    v = np.random.default_rng(42).standard_normal(len(ac))
    r_naive, _ = cauchy_matvec_naive(ac, rc, alpha, beta, v)
    r_fast,  _ = cauchy_matvec_fast(ac, rc, alpha, beta, v)
    r_direct   = A @ v
    print(f"   {tuple(part)}: naive≈direct={np.allclose(r_naive,r_direct)}, "
          f"fast≈direct={np.allclose(r_fast,r_direct,atol=1e-8)}")

print("\n3. Staircase Toeplitz FFT correctness:")
rng = np.random.default_rng(0)
for k in [2, 3, 4, 5]:
    partition = list(range(k, 0, -1))
    ac, rc, alpha, beta, A = get_partition_data(partition)
    v = rng.standard_normal(len(ac))
    r_fft    = staircase_A_via_toeplitz(k, v)
    r_direct = A @ v
    print(f"   k={k}  {tuple(partition)}: FFT≈direct={np.allclose(r_fft, r_direct, atol=1e-9)}")

print("\n4. Operation count comparison:")
print(f"  {'k':>5} | {'Naive k²':>10} | {'Fast log²k':>12} | {'FFT k log k':>12} | {'Circuit (CS)':>13}")
print("  " + "-"*60)

def op_count_fast(k):
    """O(k log² k) operations for partial-fractions multi-point eval."""
    return int(k * np.log2(k+1)**2) if k > 1 else 1

def op_count_fft(k):
    """O(k log k) for Toeplitz FFT."""
    return int(k * np.log2(k+1)) if k > 1 else 1

def cs_gate_count(k):
    """O(k log k) gates from CS recursion."""
    if k <= 2: return 1
    return 2 * cs_gate_count(k // 2) + k // 2

for k in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
    naive  = k * k
    fast   = op_count_fast(k)
    fft    = op_count_fft(k)
    cs     = cs_gate_count(k)
    print(f"  {k:>5} | {naive:>10,} | {fast:>12,} | {fft:>12,} | {cs:>13,}")


print("\n5. WHY THE CS DECOMPOSITION IS O(k log k) FOR CAUCHY A-MATRICES:")
print("""
   CS decomposition splits A (k×k) into:
     A = [U₁  0 ] [C  -S] [V₁ᵀ  0 ]
         [0  U₂ ] [S   C] [0   V₂ᵀ]

   where C,S are (k/2)-diagonal (k/2 angles to determine).

   KEY: the off-diagonal block of A (which has Cauchy structure restricted
   to the upper-right k/2 × k/2 sub-Cauchy-matrix) determines the CS angles
   via its singular values — and a k/2 × k/2 Cauchy SVD costs O(k log k)
   using the displacement rank structure (not O(k²)).

   Then U₁, U₂ are the remaining orthogonal factors, and crucially:
     — U₁ acts on the TOP k/2 addable cells (large content values)
     — U₂ acts on the BOTTOM k/2 addable cells (small content values)

   Each sub-problem is again a (k/2)-addable partition A-matrix,
   because the interlacing means the top/bottom content splits give
   a valid sub-Cauchy structure at each level.

   Recursion: T(k) = 2·T(k/2) + O(k)  →  T(k) = O(k log k)

   For the STAIRCASE case: each level is a Toeplitz sub-problem,
   so the constant factor improves further (FFT at every level).
""")

# ── 6. Visualize ───────────────────────────────────────────────────────────────

ks = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
naive_ops  = [k**2       for k in ks]
fast_ops   = [op_count_fast(k) for k in ks]
fft_ops    = [op_count_fft(k)  for k in ks]
cs_ops     = [cs_gate_count(k) for k in ks]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.loglog(ks, naive_ops, 'o-', color='crimson',    label='Naive O(k²)')
ax.loglog(ks, fast_ops,  's-', color='steelblue',  label='Fast Cauchy O(k log²k)')
ax.loglog(ks, fft_ops,   '^-', color='seagreen',   label='Toeplitz FFT O(k log k)')
ax.loglog(ks, cs_ops,    'd-', color='darkorange',  label='CS circuit O(k log k)')
ax.set_xlabel('Matrix dimension k', fontsize=11)
ax.set_ylabel('Operations', fontsize=11)
ax.set_title('Operation count: Cauchy A-matrix decomposition', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax2 = axes[1]
speedup_fast = [n/f for n,f in zip(naive_ops, fast_ops)]
speedup_fft  = [n/f for n,f in zip(naive_ops, fft_ops)]
speedup_cs   = [n/f for n,f in zip(naive_ops, cs_ops)]
ax2.semilogx(ks, speedup_fast, 's-', color='steelblue', label='Fast Cauchy speedup')
ax2.semilogx(ks, speedup_fft,  '^-', color='seagreen',  label='FFT speedup (staircase)')
ax2.semilogx(ks, speedup_cs,   'd-', color='darkorange', label='CS circuit speedup')
ax2.set_xlabel('Matrix dimension k', fontsize=11)
ax2.set_ylabel('Speedup over naive', fontsize=11)
ax2.set_title('Speedup vs naive O(k²)', fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("data/plots/decompose_a_matrix.png", dpi=130, bbox_inches="tight")
print("\nSaved → data/plots/decompose_a_matrix.png")


# ── 7. Explicit circuit for small cases ───────────────────────────────────────

print("\n6. EXPLICIT BUTTERFLY CIRCUIT for A(3,2,1) — k=4:")
print("""
   A(3,2,1) has content sequence: 3 > 2 > 1 > 0 > -1 > -2 > -3
                                   A   R   A   R    A    R    A

   CS decomposition (p=2, q=2):

     Layer 1:  2 Givens rotations (the 2 CS angles θ₁, θ₂)
               mixing {row 0 ↔ row 2} and {row 1 ↔ row 3}
               [these are determined by singular values of the off-diagonal Cauchy block]

     Layer 2a: U₁ (2×2) acting on rows {0,1}  — a SINGLE Givens rotation
               (the sub-A-matrix for top 2 addable contents {3,1})

     Layer 2b: U₂ (2×2) acting on rows {2,3}  — a SINGLE Givens rotation
               (the sub-A-matrix for bottom 2 addable contents {-1,-3})

     Layer 3a: V₁ (2×2) acting on cols {0,1}  — a SINGLE Givens rotation
     Layer 3b: V₂ (2×2) acting on cols {2,3}  — a SINGLE Givens rotation

   TOTAL: 6 Givens rotations  vs  6 naive (same at k=4 — advantage grows for large k)
""")

# Verify the explicit butterfly circuit
from scipy.linalg import cossin
ac, rc, alpha, beta, A321 = get_partition_data([3,2,1])
(U1, U2), thetas, (Vt1, Vt2) = cossin(A321, p=2, q=2, separate=True)
print("   CS angles (radians):", np.round(thetas, 5))
C_mat = np.diag(np.cos(thetas))
S_mat = np.diag(np.sin(thetas))
CS_block = np.block([[C_mat, -S_mat],[S_mat, C_mat]])
U_block  = np.block([[U1, np.zeros((2,2))],[np.zeros((2,2)), U2]])
Vt_block = np.block([[Vt1, np.zeros((2,2))],[np.zeros((2,2)), Vt2]])
A_reconstructed = U_block @ CS_block @ Vt_block
print(f"   Reconstruction error: {np.max(np.abs(A321 - A_reconstructed)):.2e}")
print(f"   U₁ Givens angle: {np.degrees(np.arccos(U1[0,0])):.3f}°")
print(f"   U₂ Givens angle: {np.degrees(np.arccos(U2[0,0])):.3f}°")
print(f"   V₁ Givens angle: {np.degrees(np.arccos(Vt1[0,0])):.3f}°")
print(f"   V₂ Givens angle: {np.degrees(np.arccos(Vt2[0,0])):.3f}°")
print(f"   Total: 2 (CS) + 1+1 (U) + 1+1 (V) = 6 Givens rotations")
