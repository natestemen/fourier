# Results

Consolidated findings on circuit decompositions of A-matrices — the orthogonal
matrices A(λ) arising from the symmetric-group branching rule, relevant to
implementing the QFT on Sₙ.  The original research narrative is `report.md`
(March 2026); this file is the maintained summary, with pointers to the code
that reproduces each claim.  Every structural fact listed here is also pinned
down by the test suite (`uv run pytest`).

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

## Open directions (updated)

1. ~~CS sub-block recursion~~ — **falsified** (Finding 7).
2. ~~Constructive HSS/ULV circuit~~ — **done** (Finding 10).  Next:
   ~~replace completed local factors with core transforms~~ — **done** in
   `ulv_core_transforms.py`; next push δ → 10⁻⁶, independently replay the
   operation log, and finish the error constants.
3. ~~Coherent angle computation~~ — **verified classically**
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
   through the same accumulated-formula pipeline (mechanical), and bound
   the arithmetic depth per angle.
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
