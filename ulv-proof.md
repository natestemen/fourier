# Proof draft: near-linear approximate Givens factorizations of A-matrices

*Companion to `ulv-note.md`.  Status: complete modulo the two bookkeeping
points flagged in §7; constants not optimized.  July 2026.*

## 1. Statement

**Theorem.**  Let λ be a partition with k addable cells, A = A(λ) ∈ O(k)
its branching matrix, and L the content span (the difference between the
largest and smallest addable-cell contents; L ≤ |λ| + 1).  For every
ε ∈ (0, ½) there exist plane rotations G₁, …, G_N and D = diag(±1) with

    ‖A − G_N ··· G₁ · D‖₂ ≤ ε,
    N ≤ C · k · log(2L) · (log k + log(1/ε)),

for an absolute constant C.  In particular, for any family of diagrams with
|λ| ≤ poly(k) — e.g. staircases, or any shapes with polynomially bounded
parts — N = O(k · log k · (log k + log 1/ε)), versus k(k−1)/2 exactly.

The witnessing factorization is the ULV elimination of `ulv-note.md`; the
proof bounds its truncation ranks (§4), shows the bound survives the
recursion (§5), and accounts errors and rotation counts (§6).

## 2. Notation and standing facts

Write x₀ > y₀ > x₁ > y₁ > ⋯ > x_{k−1} for the interlaced integer contents
of addable (x) and removable (y) cells.  The entries are

    A[i, 0] = αᵢ,      A[i, j] = αᵢ βⱼ / (xᵢ − y_{j−1})   (j ≥ 1),

with α the (positive, unit-norm) constant column.  Facts used throughout:

- (F1) A is orthogonal, so every submatrix has spectral norm ≤ 1 and
  Frobenius norm ≤ √k; every |αᵢ| ≤ 1.
- (F2) Contents are distinct integers, so |xᵢ − yⱼ| ≥ 1 always, and
  strict interlacing holds.
- (F3) If E = W·X·V with ‖W‖, ‖V‖ ≤ 1 then σ_{r+1}(E) ≤ σ_{r+1}(X)
  for every r.  (Split X at rank r and use submultiplicativity on the tail.)
- (F4) Any b×b orthogonal matrix is a product of ≤ b(b−1)/2 plane
  rotations and a sign diagonal (classical Givens reduction).

## 3. Lemma 1 (interlacing geometry)

*Let {s, …, t} be a contiguous set of positions, defining the block's rows
(contents x_s, …, x_t, spanning the interval [x_t, x_s]) and the block's
own columns.  Every column at a position p ∉ {s, …, t}, p ≥ 1, carries a
content y_{p−1} at distance ≥ 1 from [x_t, x_s]:*

    p ≤ s−1  ⟹  y_{p−1} ≥ x_s + 3,        p ≥ t+1  ⟹  y_{p−1} ≤ x_t − 1.

**Proof.**  Interlacing places y_{p−1} strictly between x_{p−1} and x_p;
all contents are distinct integers.

*Left side (p ≤ s−1).*  y_{p−1} > x_p and integrality give
y_{p−1} ≥ x_p + 1.  Monotonicity gives x_p ≥ x_{s−1}, and interlacing puts
y_{s−1} strictly between x_{s−1} and x_s, so x_{s−1} ≥ x_s + 2.  Chaining:
y_{p−1} ≥ x_{s−1} + 1 ≥ x_s + 3.

*Right side (p ≥ t+1).*  y_{p−1} < x_{p−1} and p−1 ≥ t give
y_{p−1} ≤ x_{p−1} − 1 ≤ x_t − 1.  ∎

(The asymmetry is an artifact of the position convention and only helps.
The constant column p = 0 carries no content and costs one extra rank in
Lemma 2.  Note the block's *own* column at position s carries y_{s−1},
which lies just above the row interval — irrelevant, since own columns are
excluded from the coupling.)

## 4. Lemma 2 (rank of an off-diagonal block of A itself)

*Let I be contiguous rows with contents in an interval of length ℓ ≤ L,
and let J be any set of columns whose contents lie outside that interval
(distance ≥ 1; the constant column may be included).  Then for every
p ≥ 1 there is a matrix Ẽ of rank ≤ 2p·⌈log₂(2ℓ)⌉ + 1 with*

    ‖A[I, J] − Ẽ‖₂ ≤ 3·√k·2^{−p}.

**Proof.**  Handle the constant column exactly (rank 1).  For the rest,
E[i, j] = αᵢ βⱼ f(xᵢ, yⱼ) with f(x, y) = 1/(x − y).

Dyadic decomposition of the row interval: refine toward each end with
pieces P₁, P₂, … of lengths 1, 1, 2, 4, …, from each end; at most
2⌈log₂(2ℓ)⌉ pieces cover the interval, and each piece P of diameter d(P)
sits at distance ≥ d(P) from every content outside the interval (a piece
of length 2^{j} is preceded by pieces of total length 2^{j} toward its own
end, and the opposite exterior is farther still; the innermost pieces have
length 1 and the exterior is at distance ≥ 1 by Lemma 1).

Fix a piece P with center c and radius ρ = d(P)/2, and any outside content
y: then |y − c| ≥ d(P) + ... ≥ 3ρ, so the geometric series

    1/(x − y) = − Σ_{s≥0} (x − c)^s / (y − c)^{s+1},   |x − c| ≤ ρ,

converges with ratio |x−c|/|y−c| ≤ 1/3.  Truncating after p terms leaves a
pointwise error ≤ (1/3)^p · (3/2) · |1/(x−y)| — i.e. an **entrywise
relative** error ≤ (3/2)·3^{−p}, because |1/(x−y)| ≥ 1/(|y−c|(1+1/3)).

The truncation is a sum of p separated products (x−c)^s · (y−c)^{−s−1}, so
on the rows of piece P it is a rank-p matrix; stacking over pieces gives
rank ≤ 2p⌈log₂(2ℓ)⌉, and the diagonal scalings α, β multiply into the
factors without changing rank.  For the error, entrywise
|E − Ẽ| ≤ (3/2)·3^{−p}·|E| gives ‖E − Ẽ‖₂ ≤ ‖E − Ẽ‖_F ≤
(3/2)·3^{−p}·‖E‖_F ≤ (3/2)·3^{−p}·√k by (F1).  Absorbing constants (and
3^{−p} ≤ 2^{−p}) yields the claim.  ∎

**Corollary (truncation rank).**  σ_{r+1}(A[I, J]) ≤ δ whenever
r ≥ r*(δ) := 2⌈log₂(3√k/δ)⌉·⌈log₂(2L)⌉ + 1 = O(log L · log(k/δ)).

**Remark (sharpness).**  The Beckermann–Townsend Zolotarev bound for real
Cauchy matrices gives the same product form with constant ~1/π² and no √k;
it predicts r*(10⁻³) ≈ 6–7 at L ≈ 500, matching the measured ranks 6–9.
The elementary bound above is ~10× looser in the constant but
self-contained.  The dyadic-Taylor approximant is itself an explicit
piecewise-polynomial basis in x — an alternative certificate for the
formula-basis variant of `experiments/ulv_explicit_basis.py`.

## 5. Lemma 3 (persistence through the recursion)

*Run the ULV elimination with whole-group merging (blocks are unions of
complete survivor groups).  Then at every level, every block's coupling
matrix E satisfies σ_{r+1}(E) ≤ σ_{r+1}(A[I, J]) for some contiguous
original row range I whose contents contain the block's support interval,
and some column set J with contents outside it.  Hence the Corollary of
Lemma 2 applies verbatim at every level.*

**Proof.**  Induct on levels, maintaining the invariant: each surviving
group g carries a contiguous original row range S(g) ("support"), the
ranges of distinct groups are disjoint and ordered, and every rotation
recorded so far that involves a row of g acts within S(g).

Initially S({i}) = {i}.  A block formed by merging adjacent whole groups
has support S = ∪ S(g), contiguous by the ordering invariant.  Its current
rows are (rows of an orthogonal transform local to S) applied to the
original rows S, so the coupling to the current far columns factors as
E = W · A[S, J] · V, where W collects the S-local row transforms
restricted to the block's rows (‖W‖ ≤ 1, rows of an orthogonal matrix) and
V collects the far-leaf-local column transforms and column selections
(‖V‖ ≤ 1).  The far columns' original positions lie outside S — here
whole-group merging is essential: since supports are disjoint and ordered,
active columns of *other* groups have positions outside S, so by Lemma 1
their contents lie outside S's content interval.  (Splitting a survivor
group across two blocks breaks exactly this step: the two blocks' supports
overlap, a neighbor's active column can carry a content interior to the
block's interval, and no exterior approximation applies — observed
numerically as an O(1) failure before the merge rule was imposed.)
Now (F3) gives σ_{r+1}(E) ≤ σ_{r+1}(A[S, J]).  After the block is
processed, its survivors form one group with support S, preserving the
invariant.  ∎

## 6. Error and count accounting; proof of the theorem

Fix leaf size b and truncation δ (chosen below).

**Per-block cost.**  Each processed block applies one b×b row rotation and
one b×b column rotation: ≤ b(b−1) ≤ b² plane rotations by (F4).

**Per-block error.**  The compression rotation is exact; by Lemmas 2–3 the
rows it designates as decoupled have off-block residual ≤ σ_{r*+1}(E) ≤ δ
in spectral norm.  The retire rotation is exact and confines those rows to
the leaf; declaring them ±e discards (i) the ≤ δ off-block residual and
(ii) the deviation of the confined rows from exact orthonormality within
the leaf, which is ≤ 2δ since the full matrix is orthogonal and the rows
are unit vectors up to the discarded mass.  By orthogonality the partner
columns are then ±e up to ≤ cδ.  Total discarded mass per block ≤ c₀δ in
spectral norm, for an absolute c₀ (B2, §7).

**Level sizes.**  Take b = 2r*(δ) + 2.  A processed block of size b keeps
r* + 1 ≤ b/2 survivors, so the active dimension at level ℓ is
m_ℓ ≤ k·2^{−ℓ}, the number of levels is ≤ log₂ k, and the total number of
processed blocks is B ≤ Σ_ℓ (m_ℓ/b + 1) ≤ 2k/b + log₂ k.

**Assembly.**  All applied transforms are exact isometries, so the
discarded pieces add linearly:

    ‖A − Lᵀ D Rᵀ‖₂ ≤ c₀ δ B ≤ c₀ δ (2k/b + log₂ k).

Choose δ = ε·b/(4c₀k).  Then the error is ≤ ε, and

    r*(δ) = O( log L · log(k/δ) ) = O( log L · (log k + log 1/ε) ),

using log(1/δ) ≤ log k + log(1/ε) + log(4c₀) (the b in δ only helps).  The
rotation count is

    N ≤ b²·B + (2b)² ≤ 2kb + O(b² log k) = O(k · r*(δ))
      = O( k · log(2L) · (log k + log 1/ε) ).                        ∎

Two readings of the bound: for fixed accuracy, N = Õ(k); for accuracy
shrinking with the application's block count (ε ≤ 1/k, the natural regime
when many A-matrices compose), N = O(k · log L · log(1/ε)).

## 7. Bookkeeping point to finalize

- **(B2) The constant c₀.**  The retire-step accounting (truncated
  residual + orthonormality deviation of the confined rows +
  partner-column cleanup) should be traced to a concrete c₀; the argument
  gives c₀ ≤ 5 without effort.  Mechanical; does not affect the form of
  the theorem.

## 8. Numerical verification

`experiments/verify_proof_lemmas.py` checks the two load-bearing lemmas on
real A-matrices (k = 256, staircase and random-content): Lemma 1's margins
come out exactly 3 (left) and 1 (right) on the staircase — the
inequalities are tight there — and Lemma 2's explicit construction
satisfies both its rank and error claims at every tested p, with the error
decaying at the derived (1/3)^p rate and actual ranks well under the
bound.

## 9. Remarks

1. **Worst case.**  The bound depends on the shape only through log L.
   Empirically the staircase (L = 2k − 2, the densest packing) is the
   worst case among all shapes tested, consistent with the proof: larger
   gaps only increase separations, and L enters logarithmically.
2. **ε → 0.**  r*(δ) → k/2 as δ → 0 recovers the dense count, matching
   the empirical exact-genericity of A-matrices.
3. **Measured vs proved constants.**  The theorem's constants are loose
   (the elementary Lemma 2 costs ~10× over Zolotarev); the implementation
   achieves 45,328 rotations at k = 1024, δ = 10⁻⁴ — about 2.8·k·r*
   with the *measured* r* ≈ 13 — so the practical constant is small.
4. The proof certifies the SVD-based algorithm; the piecewise-polynomial
   approximant of Lemma 2 doubles as an explicit basis, so the same bound
   (with the same form, slightly larger constants) covers the formula-only
   variant.
