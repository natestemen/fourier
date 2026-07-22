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
diagrams, which would close a T(k) = 2T(k/2) + O(k) circuit recursion — is
now settled, negatively, by an **exact decision procedure** rather than
optimization:

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
through *any* CS bipartition.  The O(k log k) circuit recursion in its
conjectured form is dead; the fast *classical* algorithms of Finding 3 are
unaffected.  Any sub-quadratic circuit for A(λ) must come from a different
mechanism than block-recursive CS structure.

## 8. NEW (July 2026): A-matrices are circuit-generic — no gap below k(k−1)/2

An A-matrix has only 2k−3 effective parameters, yet scanning the minimal
number of plane rotations m that reproduces A(λ) to machine precision
(optimizing angles over triangulation/sweep/random plane patterns per m,
det = −1 absorbed by one uncounted reflection) finds **no gap below the full
count**: for every diagram tested at k = 4…7 the best residual decays
smoothly (e.g. k = 6: 0.83 at m = 5 → 0.25 at m = 14) and reaches machine
precision only at m = k(k−1)/2 — the same profile as a random SO(k) control.

Despite the low-dimensional parametrization, A-matrices behave like *generic*
orthogonal matrices at the circuit level, quantitatively confirming the
suspicion recorded in the original report.  (Caveat: the pattern search is
heuristic — a structured pattern family it does not sample could in principle
be missed.)  Reproduce: `experiments/min_givens_count.py`.

---

## 9. NEW (July 2026): A-matrices are orthogonal HSS matrices — a route to Õ(k) approximate circuits

The first *positive* structural result beyond the Cauchy factorization
itself.  Measuring the ε-rank of every HSS block row (each dyadic row block
against all other columns) of A(λ):

- **The rank is bounded independently of k**: ≤ 10 at ε = 10⁻³ for every
  level, for staircase and random-content A-matrices from k = 64 to 512 —
  while a random SO(k) is full-rank (k/2) at every level.
- **The rank grows only logarithmically in the accuracy**: at k = 256 the
  worst level rank is 6 (ε = 10⁻²), 9 (10⁻³), 15 (10⁻⁶), 20 (10⁻⁹),
  25 (10⁻¹²) — the classic Cauchy-kernel bound, since index-separated blocks
  are content-separated and 1/(c(a) − c(r)) is smooth there.

So A(λ) is an **orthogonal hierarchically semiseparable (HSS) matrix with
HSS rank r(ε) = O(log 1/ε), uniformly in k**.  Structured-factorization
theory (ULV decompositions of HSS matrices; Givens-weight representations of
rank-structured orthogonal matrices) converts bounded HSS rank into
factorizations using **O(r·k·polylog k) plane rotations** — i.e.
ε-approximate circuits with Õ(k) two-qubit gates in the one-hot encoding,
versus k(k−1)/2 exactly.

This is consistent with both negatives: Finding 7 rules out *exact* CS
recursion and Finding 8 rules out *exact* small rotation counts; the HSS
route is inherently approximate, with the ε → 0 limit restoring the
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
level.  The result is shape-robust: since content-gap sequences biject with
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
Constant-factor headroom remains: the implementation decomposes each local
b×b transform fully (b(b−1)/2 rotations) where ~r·b would do.

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

## Open directions (updated)

1. ~~CS sub-block recursion~~ — **falsified** (Finding 7).
2. ~~Constructive HSS/ULV circuit~~ — **done** (Finding 10).  Next:
   tighten the constants (targeted r·b-rotation local transforms instead of
   full b²/2), push δ → 10⁻⁶ and map the cost of accuracy, and port the
   factorization to the binary (⌈log₂ k⌉-qubit) encoding, where each
   rotation costs O(log k) gates and the ULV locality should keep the
   controlled versions cheap.
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
3. The one-hot encoding wastes exponential Hilbert space: k addable cells fit
   in ⌈log₂ k⌉ qubits.  The binary-encoded Givens gates become multi-controlled
   rotations; Finding 9 suggests the right gates to control are the ULV
   generators, not the raw Givens angles (the standard triangularization's
   angle vectors are not Walsh-compressible — probed July 2026).
4. The two exact negatives (Findings 7, 8) say the win is approximate or
   asymptotic: ε-approximate HSS circuits (Finding 9), approximate ansatz
   synthesis (brickwork's constant-factor win, Finding 2), n → ∞ family
   limits, and gate sharing in the global QFT-on-Sₙ construction.
