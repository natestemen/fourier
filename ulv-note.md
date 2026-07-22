# ε-approximate Givens factorizations of A-matrices with near-linear counts

*Working note, July 2026.  Numerical evidence + constructive algorithm; no theorems yet.
Code: the `fourier` repo (`experiments/ulv_circuit.py` and companions; see §Reproducibility).*

## Setting

A(λ) is the k×k orthogonal branching-rule matrix of a partition λ with k
addable cells; recall its non-constant part is diag(α)·C·diag(β) with
C[i,j] = 1/(c(aᵢ) − c(rⱼ)) Cauchy on the interlaced contents.  The shared
baseline is the exact Givens factorization, A = G_N ··· G_1 · diag(±1) with
N = k(k−1)/2 plane rotations; compilers do not improve the exponent.

This note: an ε-approximate factorization in the *same* gate alphabet —
plane rotations plus a sign diagonal — with measured near-linear counts:
45,328 rotations vs 523,776 at k = 1024, at operator error ~10⁻⁴.

## Observation: bounded hierarchical ranks

Halve the index range recursively and measure the ε-rank (singular values
above ε) of each block row A[I, Iᶜ] — the "HSS ranks" in numerical-linear-
algebra terms:

- **Uniform in k**: at ε = 10⁻³, every level has rank ≤ 10 for k = 64…512
  (staircase and random-content diagrams alike); Haar SO(k) is full-rank
  (k/2) at every level.
- **Logarithmic in accuracy**: at k = 256 the worst level rank is
  6, 9, 15, 20, 25 at ε = 10⁻², 10⁻³, 10⁻⁶, 10⁻⁹, 10⁻¹².

Mechanically unsurprising: index-separated blocks are content-separated,
where the Cauchy kernel is smooth, so their singular values decay
exponentially — the property behind fast classical Cauchy solvers.  The new
step is spending it on the orthogonal matrix itself.

## Construction

The linear algebra is standard and we claim no novelty for it: it is an
adaptation of the ULV solver for hierarchically semiseparable systems
(Chandrasekaran–Gu–Pals, SIMAX 2006); representing rank-structured
matrices by plane rotations goes back to the Givens-weight representation
(Delvaux–Van Barel, SIMAX 2007, incl. the unitary case).  The contribution
is the rank observation above — a fact about A(λ), not about matrices —
and the counted-circuit consequence.

The high-level idea: work through the matrix in blocks of b ≈ 32
consecutive rows.  Each block's coupling to the rest of the matrix — the
submatrix A[I, Iᶜ] of its rows against all columns outside the block, the
object measured above — has numerical rank only r ≈ 6–13, so a few Givens
rotations *within the block* confine that coupling to r of its rows.  The other b−r rows — and, because
the matrix stays orthogonal throughout, their partner columns — become
exact unit vectors and drop out.  Only the r-coupled fraction survives to
the next round; repeat until a small remainder is finished densely.  Every
rotation is recorded, giving A ≈ Lᵀ · diag(±1) · Rᵀ — read as a circuit,
one flat sequence of two-level Givens gates, exactly as in the classical
factorization, just on different planes and far fewer of them: O(b²)
rotations per block, k/b blocks, geometrically shrinking rounds ⇒ O(k·b)
total versus k(k−1)/2.  All rotations act within a block, which is also
convenient for limited connectivity.

## Measured counts

**Test set.**  A(λ) depends only on the interlaced content sequence, and
content-gap sequences biject with diagrams (consecutive gaps are the block
heights and widths), so sampling gap profiles samples genuine Young
diagrams — we verified this bijection numerically at small k against the
hook-length construction (agreement 10⁻¹⁵).  Six shape profiles per k,
spanning the qualitative extremes:

- staircase λ = (k−1, …, 1) — all gaps 1, the densest possible contents;
- random gaps uniform in {1..3} and in {1..10} (3–4 seeds each);
- heavy-tailed geometric gaps — occasional very large blocks;
- a two-arm shape split by a single gap of 200 (extreme content void);
- a comb — alternating gaps 1 and 30 (wide flat blocks).

33 instances across k = 256, 512, 1024, each built exactly from its
contents in log space (the hook-length route overflows float64 past
k ≈ 200; the construction is orthogonal to ~10⁻¹² at k = 512).

**Outcome.**  Max HSS rank 7–11 over every instance; rotation counts within
±4% of the table below at every k (k = 1024 range: 43,391–46,192); errors
all ≈ δ.  The extreme shapes are slightly *cheaper* — larger gaps mean
better-separated Cauchy nodes — so the staircase is the observed worst
case, and the table reports it.

The "Givens rotations" column counts ordinary plane rotations in the
recorded factorization — the identical primitive to the classical
G_N ··· G_1 · diag(±1), so the comparison with k(k−1)/2 is gate-for-gate.
Parameters: block size b = 32, truncation δ = 10⁻⁴.  Error is exactly
‖A − Lᵀ·D·Rᵀ‖₂ — no hidden slack:

| k | Givens rotations (this work) | k(k−1)/2 (exact) | ratio | error | growth/doubling |
|---|---|---|---|---|---|
| 128 | 4,403 | 8,128 | 0.54 | 9×10⁻⁵ | — |
| 256 | 10,139 | 32,640 | 0.31 | 1.0×10⁻⁴ | ×2.30 |
| 512 | 21,324 | 130,816 | 0.16 | 1.2×10⁻⁴ | ×2.10 |
| 1024 | 45,328 | 523,776 | **0.087** | 1.4×10⁻⁴ | ×2.13 |

Quadratic would be ×4.0 per doubling; ×2.1 is near-linear, consistent with
the O(r(ε)·k·polylog k) the rank bounds predict.  Accuracy is cheap:
δ = 10⁻⁶ at k = 512 costs 26% more rotations for error 7.7×10⁻⁷.  Known
constant-factor headroom ≥ 2× (local transforms are decomposed fully where
~r·b rotations suffice).

## Angles from formulas (no SVDs)

The construction above picks its rotations from SVDs of matrix data.  For
settings where the angles must be *computed* as functions of the diagram —
e.g. coherently, with λ in a register — a formula-only variant works: the
coupling of a block with row contents x and scalings α is spanned by α and
α/(x − y) with y outside the block's content interval, so an explicit
"proxy pole" basis (poles placed geometrically outside the interval, pushed
through the previously emitted rotations) replaces the SVD; a pivoted QR of
this explicit matrix picks the rank.  Measured (same b, δ): counts grow to
1.24–1.40× the SVD version — e.g. 57,697 rotations at k = 1024, still 11%
of dense — at equal or better error, with formula ranks 13–16 where the SVD
needs 8–13.  Every angle is then a feed-forward arithmetic function of the
contents and earlier angles; only the retire step still reads the working
matrix, and its input is the explicit diagonal block.
(`experiments/ulv_explicit_basis.py`.)

## Caveats

1. The Õ(k) scaling is measured to k = 1024; a proof draft of
   N = O(k · log L · (log k + log 1/ε)) — L the content span, so
   O(k·log k·(log k + log 1/ε)) for polynomial-size shapes — is in the
   companion `ulv-proof.md` (elementary: interlacing geometry + dyadic
   Taylor rank bounds + ULV accounting; one constant left to trace).
2. Crossover vs the dense factorization is at k ≈ 100 — an asymptotic
   result.
3. ε → 0 recovers the quadratic count as truncated singular values
   re-enter; the method is inherently approximate.

## Reproducibility

`experiments/hss_structure.py` (rank tables, ε-sweep),
`experiments/ulv_circuit.py` (the factorization; the table above is its
default output), `experiments/ulv_diversity.py` (the shape sweep), and
`experiments/ulv_explicit_basis.py` (the formula-only variant), with the
library under `src/fourier/`.  Everything regenerates in minutes on a
laptop except the k = 1024 rows (~10 min).
