# Results

Consolidated findings on circuit decompositions of A-matrices — the orthogonal
matrices A(λ) arising from the symmetric-group branching rule, relevant to
implementing the QFT on Sₙ.  The original research narrative is `report.md`
(March 2026); this file is the maintained summary, with pointers to the code
that reproduces each claim. Repository matrix identities are checked by the
test suite (`uv run pytest`); literature-based circuit bounds and open
precision questions are identified separately.

## Definitions

For a Young diagram λ with k addable cells, A(λ) is the k×k orthogonal matrix
with rows indexed by addable cells (content descending), a distinguished
"constant" column 0, and columns 1…k−1 indexed by removable cells (content
descending).  Construction: `fourier.amatrix.a_matrix` (numeric),
`a_matrix_symbolic` (exact), `a_matrix_generic4/8` (whole-family symbolic).

---

## 1. Every 4-addable A-matrix lies on the a = π/4 Weyl face

The KAK invariant a is pinned at π/4 (the maximally-entangling face containing
CNOT/CZ) for the entire 4-addable family; (b, c) vary continuously.
Established symbolically (magic-basis KAK on the generic family) and
numerically (every diagram up to large sizes).

- Spectrum structure: eigenvalues {+1, −1, e^{iθ}, e^{−iθ}}, so det = −1, and
  A conjugates orthogonally to diag(1, −1) ⊕ R(θ)
  (`fourier.weyl.block_rotation_form`).
- The Weyl point cloud is exactly rank 2 (PCA): `experiments/weyl_pointcloud.py`.
- Leakiness (Peterson–Crooks–Smith) does **not** explain it: all 8,231
  four-addable A-matrices tested to max_size 30 are non-leaky (rank 3), unlike
  CZ/iSWAP which share the face: `experiments/leakiness.py`.
- Reproduce: `experiments/weyl_scan.py`, `experiments/symbolic_weyl_scan.py`,
  `experiments/family_weyl.py` (named one-parameter families with n→∞ limits),
  tests `tests/test_weyl.py`.

## 2. Compilers do not beat O(k²) on the Givens circuit

Handing the Givens triangularization (k(k−1)/2 rotations) to Qiskit (u3+cx,
opt 3) or BQSKit (opt 3) yields compiled CX counts scaling as ≈ a·k² with
exponent ≈ 2 across k = 2…15 (~300 partitions per k).  BQSKit is not
meaningfully better than Qiskit.  The structure must be exploited *before*
compilation.  Reproduce: `experiments/compile_benchmark.py`.

Approximate ansatz-based synthesis buys a constant factor, not an exponent:
brickwork instantiation reaches the Wei–Di optimum at k = 4 (1 brick =
3 CNOT) and ~20 CNOT vs ~42 for the k = 8 staircase at accuracy threshold
(`experiments/brickwork.py`); flagsynth at k = 8 needs 28 Ry = dim SO(8),
consistent with Finding 8's circuit-genericity (`experiments/flagsynth_scaling.py`).

## 3. Cauchy structure — displacement rank 1 — fast classical algorithms

The non-constant part factors as A[:,1:] = diag(α)·C·diag(β) with
C[i,j] = 1/(c(aᵢ)−c(rⱼ)) a Cauchy matrix; addable/removable contents strictly
interlace, and diag(ac)·C − C·diag(rc) = 𝟙𝟙ᵀ (displacement rank 1).
Consequences, all verified numerically:

- O(k log² k) mat-vec via partial fractions / GKO-style generator updates
  (`fourier.amatrix.CauchyForm.matvec_fast`);
- O(k log k) mat-vec for the staircase λ = (k, …, 1), whose Cauchy core is
  Toeplitz (`CauchyForm.matvec_toeplitz`);
- CS (cosine–sine) decomposition with angles determined by the singular
  values of an off-diagonal Cauchy sub-block.

Reproduce: `experiments/cauchy_structure.py`; tests `tests/test_amatrix.py`.

## 4. Recursive Givens reduction A(λ) → A(λ′): no systematic pattern

Exhaustive search over (pivot, target) Givens reductions compared against the
full lower catalog finds only sporadic depth-1/2 matches; hook-ratio-guided
beam search is inconsistent.  A-matrices behave like generic orthogonal
matrices under Givens reduction.  Reproduce: `experiments/reduction_search.py`,
`experiments/hook_guided_search.py`.

## 5. Optimal 2-qubit circuit: 3 CNOT + 6 Ry, exactly

All 4-addable A-matrices have det = −1, so Wei & Di's O(4) synthesis
(arXiv:1203.0722) applies and is provably optimal: 3 CNOT + 6 Ry.  All 311
diagrams to max_size 15 decompose with residual < 10⁻¹⁶ (vs ~12 CX from naive
Givens → Qiskit).  Library: `fourier.decompositions.wei_di_fit`,
`fourier.circuits.wei_di_circuit`.  Reproduce: `experiments/wei_di_synthesis.py`;
tests `tests/test_decompositions.py`, `tests/test_circuits.py`.

## 6. CS butterfly: same gate count, half the depth

The recursive CS butterfly (`fourier.decompositions.cs_butterfly`) matches
Givens in gate count (O(k²)) but has dependency-DAG depth ≈ k vs ≈ 2k for
column-major Givens.  (The old `benchmark_circuit.py` used a flawed greedy
depth that reordered non-commuting gates; the library's `parallel_depth` is
the true dependency depth.)  Reproduce: `experiments/givens_vs_cs.py`.

---

## 7. NEW (July 2026): the CS sub-block conjecture is FALSE

The open question of report.md (directions #1 and #2) — whether the CS
sub-blocks U₁, U₂, V₁, V₂ of A(λ) are themselves A-matrices of smaller
diagrams — is now settled, negatively, by an **exact decision procedure**
rather than optimization:

1. **Affine invariance**: `a_matrix_from_contents(ac, rc)` is invariant under
   ac → s·ac + t, rc → s·rc + t, so the right question is whether a block is
   an A-matrix of *any real* content sequence — strictly weaker than "of a
   genuine diagram".
2. **Gauge invariance**: the CS decomposition fixes the sub-blocks only up to
   per-plane column signs and plane permutations; earlier searches
   (`cs_subblock_test.py`) checked only global sign/transpose/row-perms.
3. **Exact test**: with α the constant column and N[i,j] = αᵢ/M[i,j], the
   matrix W[i,j] = N[i,j] − N[0,j] = (acᵢ − ac₀)/βⱼ must have **rank 1**
   (invariant under the whole gauge group).  Rank-1-ness decides membership
   and recovers the contents in closed form; the second singular value is a
   certificate for non-membership.

Results (`experiments/cs_subblock_cauchy.py`):

- k = 6 (3×3 blocks): all 596×4 blocks pass — but vacuously: dim SO(3) = 3
  equals the affine-reduced content family, so *every* generic 3×3 orthogonal
  is a real-content A-matrix.  k = 6 carries no information.
- **k = 8 (4×4 blocks): 0 of 425×4 sub-blocks are A-matrices of any real
  content sequence** (closest certificate σ₂/σ₁ ≈ 3.6×10⁻⁴, against machine
  precision ~10⁻¹²).
- **All alternative splits fail as well**: for the staircase (7,6,5,4,3,2,1)
  and (8,6,5,4,3,2,1,1), an exhaustive search over all 2,450 (row-subset ×
  column-subset) CS bipartitions finds **no** split in which all four
  sub-blocks are structured; the best split has 1 of 4.  The even–odd
  (radix-2 FFT) split fails for 402/425 diagrams on all four blocks.
- The real-Schur eigenvector matrices of A(λ) are not A-matrix-structured
  either (0/139 at k = 8), closing the spectral-factorization variant.

**Consequence**: the Cauchy/displacement structure of A(λ) does not propagate
through *any* tested CS bipartition.  Moreover, the old gate-count recurrence
was itself incorrect: a CS factorization contains four recursive matrices
\(U_1,U_2,V_1,V_2\), so its count recurrence is
\(T(k)=4T(k/2)+k/2\); \(2T(k/2)+k/2\) describes parallel depth, not total
rotations.  Thus sub-block closure alone would not have yielded an
`O(k log k)` rotation count.  The fast *classical* algorithms of Finding 3
are unaffected.

## 8. UPDATED: the exact quadratic barrier is open; symmetric contents save half

The earlier claim that tested A-matrices required all \(k(k-1)/2\) rotations
was based on two triangle patterns, one nearest-neighbor sweep, and only five
random words per length.  A pruned exhaustive `k=4` search now finds exact
five-rotation decompositions (plus a free sign) for both the staircase and
other content instances.  The staircase identity is pinned exactly in
`tests/test_structured.py`, so the old six-rotation conclusion is false.

There is also a general exact construction.  If the contents are centrally
symmetric — including every staircase and all self-conjugate diagrams — row
reversal intertwines A with a signed reversal of its nonconstant columns.
Two sets of disjoint π/4 pair rotations split A into orthogonal parity blocks
of sizes \(p=\lceil k/2\rceil\) and \(q=\lfloor k/2\rfloor\).  Dense
factorization only inside those blocks costs

\[
N_{\rm sym}
=k-1+\frac{p(p-1)}2+\frac{q(q-1)}2
=\frac{k^2}{4}+O(k),
\]

asymptotically half the generic count.  Implementation:
`fourier.structured.symmetric_content_factorization`.

For the full real-content family, the only rigorous parameter-count lower
bound for a finite collection of fixed plane words is \(2k-3\), because the
contents have \(2k-3\) degrees of freedom after affine gauge.  The gap from
this linear lower bound to the dense quadratic upper bound remains open.
`experiments/five_givens_search.py` distinguishes isolated fits from
family-wide charts using continuation and tangent spaces.  The original
`experiments/min_givens_count.py` remains useful negative evidence for its
sampled topologies, not a minimal-count result.

---

## 9. NEW (July 2026): A-matrices are orthogonal HSS matrices — a route to Õ(k) approximate circuits

The first *positive* structural result beyond the Cauchy factorization
itself.  Measuring the ε-rank of every HSS block row (each dyadic row block
against all other columns) of A(λ):

- **The rank is small over the measured range**: ≤ 10 at ε = 10⁻³ for every
  level, for staircase and random-content A-matrices from k = 64 to 512 —
  while a random SO(k) is full-rank (k/2) at every level.  Independence from
  k is an empirical observation here, not an asymptotic theorem.
- **The rank grows only logarithmically in the accuracy**: at k = 256 the
  worst level rank is 6 (ε = 10⁻²), 9 (10⁻³), 15 (10⁻⁶), 20 (10⁻⁹),
  25 (10⁻¹²) — the classic Cauchy-kernel bound, since index-separated blocks
  are content-separated and 1/(c(a) − c(r)) is smooth there.

So A(λ) is empirically an **orthogonal hierarchically semiseparable (HSS)
matrix with slowly growing numerical ranks**.  Structured-factorization
theory (ULV decompositions of HSS matrices; Givens-weight representations of
rank-structured orthogonal matrices) converts bounded HSS rank into
factorizations using **O(r·k·polylog k) plane rotations** — i.e.
ε-approximate circuits with Õ(k) two-qubit gates in the one-hot encoding,
versus k(k−1)/2 exactly.

This is consistent with the exact CS negative and the open exact-count
problem in Finding 8; the HSS route is inherently approximate, with the
ε → 0 limit restoring the
quadratic count (r(ε) → k/2 as the top-level CS spectrum fills in).

The naive Givens order does not see this structure (dropping sub-10⁻³ angles
from the standard triangularization of a k = 256 A-matrix removes only ~0.4%
of rotations); a nested-basis ULV elimination is required — built in
Finding 10.  Reproduce: `experiments/hss_structure.py`.  (Large-k A-matrices
are built from their contents in log space — `fourier.amatrix.
staircase_a_matrix` / `random_content_a_matrix` — since the hook formula
overflows beyond k ≈ 200.)

## 10. NEW (July 2026): constructive ULV circuit — near-linear rotation counts

The ULV factorization predicted by Finding 9 was implemented
(`experiments/ulv_circuit.py`): an orthogonal-ULV elimination that reduces
A(λ) to diag(±1) with explicitly recorded Givens rotations.  Per leaf of b
active rows: a local rotation confines all off-leaf coupling to r rows
(r = the ε-rank), a local column rotation retires the other b−r rows to ±e,
and orthogonality of the full matrix retires the partner columns for free;
only r rows/columns per leaf stay active for the next level.

Measured counts (b = 32, truncation δ = 10⁻⁴; both staircase and
random-content diagrams):

| k | ULV rotations | k(k−1)/2 | ratio | operator error |
|---|---|---|---|---|
| 128 | 4,403 | 8,128 | 0.54 | 9×10⁻⁵ |
| 256 | 10,139 | 32,640 | 0.31 | 1.0×10⁻⁴ |
| 512 | 21,324 | 130,816 | 0.16 | 1.2×10⁻⁴ |
| 1024 | 45,328 | 523,776 | **0.087** | 1.4×10⁻⁴ |

The per-doubling growth falls to ≈ 2.1 by k = 1024 — near-linear scaling
(quadratic would be 4.0) — while the error stays pinned at the truncation
level.  A dense scaling sweep (`experiments/ulv_scaling.py`, 73 runs, 17
sizes to k = 16384, random shapes from three gap profiles per size) fits
the k ≥ 384 tail as N ≈ 25·k^1.09 (log-RMS 0.010) with N ≈ 4.7·k·log₂k
tracking the whole tail; at k = 16384 the count is 923,550 vs 134,209,536
dense (**0.69%**), error 2.5×10⁻⁴, with the error growing only linearly in
the block count as Finding 11's accounting predicts.  Figure:
`data/plots/ulv_scaling.png`.

Cost of accuracy (`experiments/ulv_accuracy.py`, 240 runs, δ = 10⁻²…10⁻¹²,
leaf size optimized per δ): all k collapse onto
N ≈ (1.5 + 0.73·log₁₀(1/δ))·k·log₂k — linear in the digits of accuracy
over ten orders of magnitude, 11% of dense even at twelve digits
(k = 2048), achieved errors tracking δ.  Composition consequence: M blocks
at total error ε need δ = ε/M each, costing only +0.73·log₁₀M per block.
Figure: `data/plots/ulv_accuracy.png`.  The result is shape-robust: since content-gap sequences biject with
diagrams (gaps = block heights/widths; `tests/test_amatrix.py`), a sweep
over six shape profiles — staircase, uniform gaps {1..3} and {1..10},
heavy-tailed geometric gaps, a two-arm shape with a 200-gap void, and a
comb — 33 instances at k = 256…1024, gives max HSS rank 7–11 everywhere and
counts within ±4% of the table, with the extreme shapes slightly cheaper
(better content separation); the staircase is the observed worst case
(`experiments/ulv_diversity.py`).  In the one-hot encoding each rotation is one two-qubit Givens gate,
so this is an ε-approximate circuit for A(λ) with **Õ(k) gates**, the
scaling the original report hoped for from the (falsified) CS recursion —
obtained instead from the HSS structure, at the price of ε ≈ 10⁻⁴ accuracy.

The local constant has now been reduced in
`experiments/ulv_core_transforms.py`.  This variant also preserves whole
survivor groups, matching the invariant assumed by the proof.  For a block
of size \(b\), coupling rank \(r\), and \(d=b-r\) retiring coordinates, it
uses \(rd\) row rotations (the dimension of the relevant Grassmannian) and
at most \(bd-d(d+1)/2\) retirement rotations, without completing either
local orthogonal factor.  Paired whole-group results at `b=32`,
`delta=1e-4` are:

| k | core rotations | completed-local rotations | saving | operator error |
|---:|---:|---:|---:|---:|
| 128 | 3,043 | 4,403 | 30.9% | 9.08×10⁻⁵ |
| 256 | 6,745 | 10,139 | 33.5% | 1.04×10⁻⁴ |
| 512 | 14,397 | 21,207 | 32.1% | 1.28×10⁻⁴ |
| 1024 | 30,452 | 45,556 | 33.2% | 1.44×10⁻⁴ |
| 2048 | 62,150 | 91,648 | 32.2% | 1.59×10⁻⁴ |

The core and completed-local paths have the same error in every paired run.

## 11. NEW (July 2026): proof draft for the Õ(k) count

`ulv-proof.md` drafts the theorem: for any λ with k addable cells and
content span L, A(λ) is ε-approximated by a product of
N = O(k · log(2L) · (log k + log 1/ε)) plane rotations and a sign diagonal
— O(k · log k · (log k + log 1/ε)) for any polynomial-size shape family.
Structure: (i) an interlacing lemma placing all far-column contents at
distance ≥ 1 from a block's content interval; (ii) an elementary
dyadic-Taylor rank bound for off-diagonal blocks (rank
2p·⌈log₂ 2ℓ⌉ + 1 at error 3√k·2⁻ᵖ; the Beckermann–Townsend Zolotarev
bound gives the sharp constant); (iii) a persistence lemma showing the
bound survives the ULV recursion exactly when blocks merge whole survivor
groups — formalizing the failure mode observed empirically; (iv) linear
error accumulation across ≤ 2k/b blocks.  Lemmas (i)–(ii) are numerically
verified (`experiments/verify_proof_lemmas.py`); Lemma 1 is tight on the
staircase.  Remaining: one mechanical constant trace (§7 of the draft).

## 12. NEW: arrowhead/Jacobi structure and a skew-Cayley route

The Cauchy formula gives an exact spectral factorization not previously used
in the repository.  With \(X=\operatorname{diag}(x)\),

\[
A^T X A=
\begin{pmatrix}
\sum_i\alpha_i^2x_i & \beta^T\\
\beta & \operatorname{diag}(y)
\end{pmatrix}.
\]

Equivalently, let \(P_x\) be the transition-measure orthogonal-polynomial
transform and \(J=P_x^T X P_x\) its Jacobi matrix.  The spectrum of
\(J_{1:,1:}\) is exactly the removable contents, and

\[
A=P_x(1\oplus P_y^T)
\]

up to column signs.  For staircases these are Hahn transforms with parameters
`(-1/2,-1/2)` and `(+1/2,+1/2)`.  This is implemented and tested in
`fourier.structured`; it is an exact structural decomposition, though the
known `O(k log² k)` polynomial-transform arithmetic is not yet a Givens
circuit.

A second route uses \(D=\operatorname{diag}(1,-1,\ldots,-1)\) and the skew
Cayley generator

\[
S=(I-AD)(I+AD)^{-1}.
\]

Any skew approximation to \(S\) maps back to an exactly orthogonal
approximation to A.  Scanning every lower-left cut at tolerance `1e-4` gives
maximum ranks `5,6,6,7` for staircase sizes `32,64,128,256`, and
`5,6,7,7` for one random-content sequence per size; `cond(I+AD)` stays below
`1.75`.  This motivates a direct quasiseparable/Givens-weight construction,
but the globally consistent skew compression and product-only conversion
remain to be implemented.  Reproduce:
`experiments/quasiseparable_cayley.py`; full derivations and references are
in `NEW_METHODS.md`.

The all-cut ranks also lead to a simpler construction that is already
implemented.  `experiments/quasiseparable_core.py` sequentially compresses a
prefix/suffix coupling and retires decoupled coordinate pairs.  If an
orthogonal matrix has exact upper quasiseparable rank \(s\), the explicit
Givens count satisfies

\[
N<2sk+2(s+1)^2.
\]

At `delta=1e-5`, staircase counts for `k=256,512,1024,2048` are
`3,836`, `8,615`, `18,937`, and `39,627`, with operator errors
`2.0e-5`, `2.7e-5`, `4.0e-5`, and `5.9e-5`.  At `k=1024` this is `3.6%`
of dense and substantially smaller than hierarchical core ULV.  The exact
count theorem and recorded-operation replay are tested; the small observed
error accumulation and the numerical-rank scaling remain empirical.

## 13. NEW (October 2026): improvements in coherent total cost

The exact Givens baseline now has a **quadratic-arithmetic angle generator**.
`experiments/structured_qr_probe.py` uses the rank-one displacement
generators directly, rather than constructing and triangularizing a dense
matrix. It emits the same k(k−1)/2 rotations in a fixed adjacent word,
with O(k²) scalar work including the transition weights. At k=512 the
observed maximum-entry residual is about 3.4e−13 for a staircase and
9.6e−12 for a wide-gap random profile. This specializes established
fast Cauchy QR. A uniform finite-precision stability bound is still absent;
O(k) mutable classical memory is not an O(k) reversible-workspace bound.

For staircases, `fourier.staircase_lift` gives an **exact reduction to two
Krawtchouk transforms** on dimension 2k−1. The branch containing the
constant column is handled and uncomputed coherently; there is no
postselection. Tests check the whole input isometry, its clean flags, and
the SU(2) phase convention. At k=128, spectral residual is 1.42e−13,
and forbidden-output leakage is below 3e−16. Combining this reduction with
the 2026 binary SU(2)/Krawtchouk transform theorems yields a restricted-family
approximate bound of `O(log³(k/ε) log(2/ε))` quantum gates **including
coefficient arithmetic**, with O(log(k/ε)) qubits. The underlying fast
quantum-Hermite circuit is not implemented in this repository. An elementary
Dicke encoder gives an alternative O(k polylog(k/ε)) coherent construction
with O(k) physical work qubits.

`fourier.projective_transport` extends any such fast family by exact signed
permutations: invert the content coordinate around any removable node,
moving that column to the distinguished constant position. The permutations
require only arithmetic on (k,q), where q selects the removable node.
The fast staircase orbit includes nonuniform, nonsymmetric Young diagrams,
but does **not** include every diagram. Recognizing the orbit from arbitrary
coherent input data is a separate cost.

The quantum-gate bound is not a Givens-count bound; a binary gate can act on
many matrix coordinates simultaneously. Proofs, sources, full comparisons,
reproduction commands, and precision qualifications are in
[`NEW_METHODS.md`](NEW_METHODS.md), §§6–8. These findings alone do not
establish a universal subquadratic coherent bit-complexity bound; the next
research pass below gives a stronger architecture.

## 14. NEW: general near-linear overlap encoding and two fast families

The strongest general candidate now avoids a Givens product. For every
A-matrix in the repository convention, the sign structure implies

\[
 \||A|\|_2\le\lceil\log_2 k\rceil+1.
\]

An elementary proof flips the first column, splits the signed off-diagonal
part into dyadic sibling levels of norm at most one, and adds the diagonal.
A finite positive Neumann sum then produces row/column scaling vectors in
O(log k) positive Cauchy matvecs. Weighted square-root-entry isometries have
overlap A/μ with μ=O(log k). Oblivious amplitude amplification implements A
in O(log k) calls to these preparations; each costs O(k) scalar operations
and state-preparation rotations, before precision/control factors.

The missing weight-preprocessing bottleneck is addressed by signed
logarithmic potentials: charge −1 on addable nodes and +1 on removable
nodes. Their values give log α² and −log β². A dyadic multipole routine
computes all weights in O(k p log(2L)) arithmetic; its inverse-distance
variant computes the balancing products at the same cost. Here p is only
logarithmic in input bounds and desired accuracy. Padded sorting-network
joins replace the classical box dictionary in the proposed coherent
implementation; no free QRAM is assumed.

The resulting **derived bound** is
k·poly(log k,log L,log(1/ε)) gates and workspace, including parameter
arithmetic and uncomputation, given coherent ordered corner lists.
Conversion from another diagram encoding must be charged separately.
Uniform padded schedules, signed phases and clean-isometry error bounds
are essential. The weighted embedding and amplitude amplification are
established literature tools, not inventions of this project; full
attribution and the A-specific argument appear in `NEW_METHODS.md`, §10.

Implementation status:

- `experiments/log_weight_multipole.py`: numerical weights, positive
  matvecs and balancing without a dense matrix. At k=1024, log-weight
  errors were below 4.0e−12 for the tested staircase, random and wide-gap
  profiles. The analytic series-tail bound is separate from float rounding.
- `experiments/coherent_overlap_encoding.py`: independent dense checks of
  both isometries, fixed-phase amplification and complete failure amplitude.
  Combined multipole/overlap checks through k=64 gave operator errors below
  8.4e−14 and failure norms below 9.5e−15 for the recorded three families.
- `experiments/reversible_join_probe.py`: fixed bitonic-network batched
  lookup with source/query masks, carry history, XOR output and explicit
  compute-copy-uncompute. This verifies reversible basis-state logic and
  operation counts, not gate-level phase synthesis.

The full finite-precision reversible circuit is **not yet implemented or
certified**. Dense validation of an isometry is not a benchmark of the
claimed fast quantum implementation. The bound also does not establish a
subquadratic exact Givens decomposition.

Two restricted-family additions complement the general route:

1. Rectangular staircases (fixed step height h and width w): remove a
   near-pole diagonal from a fractional-Fourier kernel, then restore it
   exactly. Uniform normalization O(√k) yields a derived Õ(√k) coherent
   bound, including gamma-ratio coefficient arithmetic. See
   `experiments/hilbert_block_encoding.py`; no full circuit is compiled.
2. Cotangent-spaced contents: an exact two-QFT/diagonal-phase identity has
   O(log² k+b log k) ideal gates for power-of-two k and a coherent b-bit
   parameter. Small actual Qiskit circuits, phases and clean ancillas are
   tested in `experiments/rectangular_staircase_probe.py`. The nodes are
   usually irrational; scaled/rounded integer approximants give a specific
   family of Young diagrams, not a generic approximation method.

A negative search result: retaining only O(k log k) of the largest
departures from π/2 in the generator-QR word did not maintain small error.
At a retained budget 8k log₂ k, staircase errors grew from about 0.076 at
k=128 to 0.306 at k=256. This pruning heuristic is not a scalable result.

The research workflow keeps derived identities, numerical evidence,
compiled small circuits and unimplemented resource estimates separate.

## 15. Compact streaming induction: a subcubic full-QFT theorem draft

Combining the general overlap core with canonical compact-shape operations
gives **n^(5/2)·poly(log n,log(1/ε)) gates** and
**n·poly(log n,log(1/ε)) workspace** for the complete approximate S_n QFT.
The formal statement and proof are in
[`subcubic-qft-theorem.tex`](subcubic-qft-theorem.tex); see
[`NEW_METHODS.md`, §13](NEW_METHODS.md#13-a-subcubic-full-qft-compact-streaming-induction)
for the research summary. This is an externally unreviewed circuit-existence
theorem draft; a complete fault-tolerant compiler and practical resource
constants remain unimplemented. Numerical agreement alone is not the proof.

The new ingredients are:

- Canonical Frobenius shape registers with floor(sqrt M) public slots,
  fixed scans for box edits, and fixed sorting networks for corner extraction.
  Compute/swap/inverse erases old shapes without retaining edit history.
- A one-box difference token replacing a second evolving endpoint shape.
  It covers diagonal A mixing, identity detours, inactive/star branches and
  clean equality-flag erasure. Its local inverse needs no stored history.
- A uniform fixed-precision core argument: exact rational truncated
  log/inverse potentials with polynomial word bounds, stable Horner
  balancing, source-preserving preparations and full-workspace projector
  reflections. This replaces the earlier informal global rounding claim.
  No free QRAM or classical preprocessing for superposed labels is assumed.

| Work at outer induction size M | Total cost, hiding precision logarithms |
|---|---:|
| One compact inner step, including its A matrix | Õ(√M) |
| All M inner steps and external-boundary conversions | Õ(M^(3/2)) |
| Full QFT, summing M=1,...,n | **Õ(n^(5/2))** |

A simpler unbalanced-overlap version yields **Õ(n^(11/4))** without the
Neumann balancing argument, independently crossing the cubic exponent.

Validation: compact schedules cover all 2,714 partitions through 20 boxes
and 19,246 box-edit edges. The streaming program verifies induction blocks
against direct Young-orthogonal representation matrices, including arbitrary
complex superpositions and inverses reconstructed from clean outputs. Its
full 720×720 S_6 Fourier matrix has max entry discrepancy 6.94e−17 and
orthogonality infinity error 1.01e−14. A test forbids expanded-shape access
inside inner steps. Directed-binary interval tests enclose exact rational
potential/weight answers and certify positive padding for stored scalings.

Reproduce with `tests/test_compact_shapes.py`,
`tests/test_streaming_induction.py`, `tests/test_certified_multipole.py`, and
`tests/test_precision_budget.py`. Code: `src/fourier/compact_shapes.py` and
the correspondingly named experiment files. The research-verification
workflow separated mathematical bounds, exact reversible-logic checks,
interval certificates and dense reference tests throughout this result.
The theorem-development pass adds 18 exact-rational precision checks,
13 quantum-interface checks and 23 wrapper checks. That checkpoint passed
**601 tests**.
The quantum checks include a counterexample to checking only part of the
preparation scratch during reflection; the theorem requires all of it.

## 16. Beyond 5/2: a transport lemma, a simpler matching route, and limits

The complete coherent QFT bound remains **Õ(n^(5/2))**, not a smaller
exponent. The current search adds:

| Result | Bound | What it does and does not establish |
|---|---|---|
| Aligned one-box transport | rank(E_p)≤2p+4 and ||A_new−A_old−E_p||≤(4/(3√3))4^(−p) | Uniform approximation theorem; no cheap coherent basis preparation yet |
| Orthogonal transport correction | Error ≤(8/(3√3))4^(−p), supported on ≤4p+8 dimensions | A genuine orthogonal update exists; constructing its supporting subspace coherently is still charged |
| Direct induction plus coherent amplification | Õ(M^(3/2)) per induction; Õ(n^(5/2)) overall | Independent matching construction, avoiding the fast A-matrix machinery |
| One-line permutation ↔ canonical transversal encoding | O(n log⁴(n+1)) gates, including cleanup | Improves the previous quadratic interface, not the dominant QFT exponent |
| Fresh-input local A obstruction | Ω(k) gates at δ<c/√N on a fixed-N Young family with k=Θ(√N) | Rules out a generic o(k) fresh-input replacement; **not** an amortized or full-QFT lower bound |
| Right-adjacent walk through all M left cosets | At least M(M−1)/2 swaps | Rules out improving that Beals implementation by reordering representatives alone |

The transport theorem follows from the exact one-box resolvent update
f_new(z)/f_old(z)=(z−c)²/((z−c)²−1) and an exponentially convergent
divided-difference expansion. Its exact rank is unbounded; small approximate
rank is the useful phenomenon. A polar-factor argument upgrades the
approximation to a true orthogonal correction. A separate exact witness
shows that storing every original-prefix corner table may require
Ω(M^(3/2)) words, and that old-path caches cannot simply be erased under
control of the new output label.

Proofs, coherent-access caveats and next targets are in `NEW_METHODS.md`, §14.
New reference code and tests have matching names: `prefix_amortization_probe`,
`sublinear_core_probe`, `alternative_qft_probe`, and `transversal_encoding_probe`.
The expanded full suite passes **766 tests**. None of these probes is a
compiled fault-tolerant circuit. The next general-exponent target
is a clean reusable transport representation, a different induction block
encoding, or a multiplicity-aware balanced subgroup merge—not an assumed
constant-cost angle or memory oracle.

## 17. Optimality audit: 5/2 is tight for two modular architectures

The new lower bound is **restricted, not an unrestricted QFT optimality
theorem**. Full proofs are in `qft-optimality-audit.tex`; a research summary
is in `NEW_METHODS.md`, §15.

| Circuit class | Lower bound established here | Crucial qualification |
|---|---|---|
| Fresh coherent local A primitive | Ω(k)=Ω(√N), even at constant clean error δ≤1/144 | Shape provided only in explicit fresh parameter words; no retained shape-dependent cache |
| Separate full-promise streaming modules | Ω(n^(5/2)) | Each prefix module separately correct; no shared preprocessing or gate fusion |
| Tagged-column scalar-polynomial amplification | Ω(n^(5/2)) | Prescribed physical W calls separately implemented; each outer induction has its own correctness contract |
| Arbitrary bounded-arity clean S_n QFT circuit | Ω(n log n) | Does not prove 5/2 optimal; no claim of an exhaustive strongest-known bound |

The strengthened local proof constructs genuine equal-size Young diagrams
with disjoint Frobenius input blocks. Every selected squared amplitude is
at least 1/(32m); flipping its corresponding bit changes that amplitude
by a factor 7/9. If u input blocks are untouched, averaging over their
bits gives δ²≥u/(2592m). First-row padding extends this to every
sufficiently large size, not just a sparse sequence. Separately counted
local-module bounds then sum to n^(5/2)/(40√2 a)+O(n²), where a is gate
arity. The amplified route has independent Ω(√M) polynomial-degree and
Ω(M) physical-column-implementation obstructions.

The research audit explicitly separates these proofs from finite checks
and rejects the unjustified step of multiplying local bounds for arbitrary
circuits. The remaining route below 5/2 is global reuse/fusion or a different
architecture, not a faster isolated fresh-input local primitive.

Validation: **976 tests pass**, including 210 new lower-bound checks. The
standalone proof note compiles successfully; its three main proof areas
were checked in separate internal audits. External review remains needed.

## 18. A simpler standalone coherent 5/2 theorem

[`simple-qft-theorem.tex`](simple-qft-theorem.tex) develops the direct
induction/amplification route into a self-contained circuit-existence
proof draft. Its bound is **Õ(n^(5/2)) elementary gates and Õ(n) live
qubits**, suppressing polynomial logarithms in n and inverse error.
It includes coherent shape-dependent arithmetic and clean isometry error,
not just a count of ideal representation rotations.

The key cost chain is **Õ(M) per column call × O(√M) amplification
rounds**, summed over M=1,…,n. Unlike the earlier theorem draft, this
proof does not depend on fast A-matrix synthesis, multipole weight
algorithms, or compact Frobenius shape bookkeeping.

New proof components and reference checks:

- Branching probabilities factor into ratios in (0,1). Directed fixed-point
  products have explicitly bounded error and logarithmic word precision;
  all shape-dependent weights take O(k² polylog)=Õ(M) work.
- Binary mass-tree preparation and local Young rotations have explicit
  endpoint-safe angle algorithms and synthesis budgets.
- Fixed sorting networks build and erase tableau caches, validate full
  source/target promises, and recover the old endpoint at the outer boundary.
  Their reversible tapes are checked even on malformed input words.
- The complete two-promise amplification proof has error (4r+1)η from
  source-isometry error η, with exact Boolean reflections and separate
  predicate helpers. The global error is budgeted across all stages.

The note compiles successfully and has separate internal arithmetic,
bookkeeping, and quantum-contract audits. That checkpoint passed **1193 tests**,
including **217 new checks** for this construction. This is not external peer review,
a priority claim, a practical performance result, or a proof that 5/2 is
unrestrictedly optimal. Those implementations are reference verifiers;
Finding 19 adds a small complete finite-gate induction compiler. Independent
review of the shorter proof remains outstanding.

## Finding 19: complete small Clifford+T induction, including coherent arithmetic

**Question:** can the simpler proof's complete induction step be emitted as
elementary gates, with all shape-dependent arithmetic and cleanup charged?

**Result:** yes, for the tested three-box step. The compiler handles both
input shapes with the same circuit. All six valid columns match the
independent ideal induction matrix to **0.00505832 clean operator error**,
including relative phases and unwanted work-register amplitudes.

The unoptimized program contains **17,138,782 Clifford+T elementary gates**,
including **7,778,300 T/T† gates**, on **76,009 logical qubits**. These are
actual emitted-program counts, not a black-box arithmetic estimate. The
public rotation dictionary is synthesized; shape-dependent angle words
are calculated reversibly. Complete source/target reflections include all
column work. The recorded simulation-pruning error bound is below
1.51 × 10⁻¹²; floating-point roundoff is separately unvalidated.

See `experiments/direct_induction_gate_probe.py`, the saved report
`data/direct_induction_gate_M3_b9_c9.json`, and `NEW_METHODS.md`, §17.
Two consecutive full reports are byte-identical, and **1346 tests pass**,
including **153 new gate-level checks**.
This closes the **small complete induction implementation** milestone.
It does not demonstrate a practical speedup, compile the entire S_n QFT,
prove the asymptotic bound numerically, or establish publication priority.

## Open directions (updated)

0. **New priority:** externally review the simpler direct-induction theorem
   and improve the now-measured constants of the complete induction compiler.
   Reuse arithmetic scratch and reduce synthesis overhead, then assemble
   a whole small QFT including the outer recurrence and cleanup. Measure
   total gates **and workspace**. The earlier balanced-overlap/compact-streaming
   pipeline remains an alternative, but is no longer a prerequisite for
   the simpler 5/2 theorem. Priority review and useful concrete constants
   remain separate tasks.
1. ~~CS sub-block recursion~~ — **falsified** (Finding 7).
2. ~~Constructive HSS/ULV circuit~~ — **done** (Finding 10).  Next:
   ~~replace completed local factors with core transforms~~ — **done** in
   `ulv_core_transforms.py`; next push δ → 10⁻⁶, independently replay the
   operation log, and finish the error constants.
3. Coherent angle computation — **partially verified classically**
   (`experiments/ulv_explicit_basis.py`): replacing the SVD compression
   with an explicit proxy-pole basis (α and α/(x − ŷ) with geometrically
   placed poles, pushed through the previously emitted rotations; rank from
   a pivoted QR of the explicit basis) preserves the construction at
   1.24–1.40× the SVD counts and equal-or-better error — 57,697 rotations
   at k = 1024 (11% of dense).  Every angle is a feed-forward arithmetic
   function of the contents and earlier angles; only the retire step reads
   the working matrix, and its input is the explicit diagonal block.  Key
   implementation fact: blocks must merge whole survivor groups (proper HSS
   tree) — position-based re-chunking overlaps support intervals and puts
   neighbor poles inside the interval, which the outside-pole basis cannot
   represent.  Remaining for full coherence: express the retire step's QR
   through the same accumulated-formula pipeline, bound the arithmetic depth
   and precision per angle, and remove adaptive rank/pivot decisions. The
   October generator-QR result supplies a fixed-word quadratic-arithmetic
   exact baseline, not yet a fast low-rank coherent construction.
4. The one-hot encoding wastes exponential Hilbert space: k addable cells fit
   in ⌈log₂ k⌉ qubits.  The binary-encoded Givens gates become multi-controlled
   rotations; Finding 9 suggests the right gates to control are the ULV
   generators, not the raw Givens angles (the standard triangularization's
   angle vectors are not Walsh-compressible — probed July 2026).
5. The exact count is open.  Search `k=5` support-complete words near the
   \(2k-3\) lower bound, and seek orthogonal butterfly/scattering
   realizations of the Jacobi/Hahn and projective-Cauchy factorizations.
6. Implement a globally consistent skew-quasiseparable approximation to the
   Cayley generator, then compare it directly with sequential cores and
   core ULV.
7. Compile and resource-estimate the clean staircase lift; check reversible
   family recognition and uniform varying-k control in the surrounding QFT.
   Separately prove precision bounds for the general generator recurrence.
