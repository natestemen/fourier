# fourier

Circuit decompositions of **A-matrices**: the k×k orthogonal matrices A(λ)
attached to a Young diagram λ with k addable cells by the symmetric-group
branching rule.  They are the local building blocks of the quantum Fourier
transform on Sₙ; the research question is whether their combinatorial
structure admits circuits below the generic O(k²) two-qubit-gate bound.

**What is known so far lives in [`RESULTS.md`](RESULTS.md)**, with the latest
structured-factorization work in [`NEW_METHODS.md`](NEW_METHODS.md).
[`report.md`](report.md) is the original March-2026 narrative;
[`ulv-note.md`](ulv-note.md) is a shareable expert note on the July-2026
HSS/ULV decomposition result, and [`ulv-proof.md`](ulv-proof.md) is the
proof draft for its Õ(k) rotation count.  Headlines:
all 4-addable A-matrices sit on the a = π/4 Weyl face and synthesize optimally
as 3 CNOT + 6 Ry; the family carries Cauchy/displacement-rank-1 structure
giving O(k log² k) *classical* algorithms; compilers do not beat O(k²) gates;
the CS sub-block recursion conjecture is falsified exactly; and A-matrices
are orthogonal HSS matrices with small observed ε-ranks, which a constructive
ULV factorization converts into ε-approximate circuits with near-linear
rotation counts.  New exact results include a Jacobi/Hahn factorization and
an asymptotic factor-of-two rotation saving for symmetric-content diagrams.

October update: a generator-QR prototype reduces general-content angle
arithmetic to O(k²), at the unchanged exact Givens count. An exact staircase
reduction to two Krawtchouk transforms, combined with 2026 binary SU(2)
circuits, gives a literature-based polylogarithmic approximate coherent
gate bound for that restricted family. Signed projective transport extends
it to some nonsymmetric diagrams. The reduction and clean flag handling are
tested; the fast quantum primitive is not compiled here. See
[`NEW_METHODS.md`](NEW_METHODS.md) for precision, scope, and encoding caveats.

Latest research pass: a **balanced overlap encoding** now gives a derived
near-linear approximate coherent resource bound for general integer
contents, including weight computation and cleanup. Its key bound is
‖|A|‖₂≤ceil(log₂ k)+1; logarithmic/inverse-distance multipole computations
supply the parameters without a dense matrix. Numerical components and a
fixed reversible-join logic verifier are tested, but the full arithmetic
and state-preparation circuit is not compiled. This
route uses near-linear workspace, not only a log-size data register.
There are also new rectangular-staircase Õ(√k) resource estimates and an
explicit two-QFT cotangent family with tested small Qiskit circuits.
These are enlarged-register quantum constructions, not shorter exact
Givens words. See [`NEW_METHODS.md`](NEW_METHODS.md), §§10–12.

Full-QFT theorem draft: **compact streaming induction** removes the repeated
linear-size shape handling. With the fixed-precision overlap construction,
the draft proves an **Õ(n^(5/2)) gate / Õ(n) workspace** bound for the
complete S_n QFT, including coherent angle computation. The formal theorem,
supporting lemmas and proof are in
[`subcubic-qft-theorem.tex`](subcubic-qft-theorem.tex), with a research summary in
[`NEW_METHODS.md`, §13](NEW_METHODS.md#13-a-subcubic-full-qft-compact-streaming-induction).
The full S_6 transform is checked against independent representation matrices;
directed-binary interval code certifies reference multipole weights and
balancing. The theorem uses a separate exact-rational truncation argument
with explicit polynomial word bounds. Its three internal audits added
54 targeted tests; that checkpoint passed 601 tests. This remains an
externally unreviewed circuit-existence theorem draft, not an already
compiled fault-tolerant implementation or practical benchmark.

Search beyond 5/2: the general exponent is unchanged. A new one-box transport
lemma gives an error-δ correction supported on **O(log(1/δ)) dimensions**;
turning its dense, shape-dependent subspace into a reusable coherent circuit
remains open. There is also a simpler direct-induction/amplification route
matching **Õ(n^(5/2))**, independent of fast A-matrix preprocessing, and a
clean **Õ(n)** permutation/transversal conversion. Exact fresh-input and
transversal-walk obstructions delimit the routes that cannot work as simple
local replacements. That checkpoint passed **766 tests**. See
[`NEW_METHODS.md`, §14](NEW_METHODS.md#14-pushing-below-52-shared-work-structure-and-explicit-obstructions).

Optimality audit: **5/2 is exponent-optimal for two explicitly restricted
modular architectures**, not proved optimal for arbitrary coherent QFT
circuits. The fresh local Ω(k) obstruction now holds even at small constant
error and every sufficiently large diagram size. The unrestricted lower
bound established here is Ω(n log n). Proofs and exact scope are in
[`qft-optimality-audit.tex`](qft-optimality-audit.tex) and `NEW_METHODS.md`, §15.
That checkpoint passed **976 tests**, including 210 new lower-bound checks.

Simpler theorem consolidation: [`simple-qft-theorem.tex`](simple-qft-theorem.tex)
now gives a standalone **Õ(n^(5/2)) gate / Õ(n) workspace** proof via direct
induction and full-unitary amplitude amplification. It does **not** depend
on fast A-matrix decomposition, multipoles, or compact shape encoding.
New bounded-product precision estimates, fixed-network reversible tableau
bookkeeping, and full-promise amplification checks account for coherent
angle computation, validation, and cleanup. See `NEW_METHODS.md`, §16,
and `RESULTS.md`, Finding 18. This is the preferred simpler theorem draft
for external review. That checkpoint passed **1193 tests**, including
**217 new checks** for this simpler construction.

Gate-level milestone: a **complete three-box induction step** now compiles
to an explicit, lazily emitted Clifford+T program, including shape-dependent
arithmetic and complete work-register reflections. All six valid input
columns, across both shapes, were checked with measured clean operator error
**0.00506**. The first, unoptimized compiler uses **17,138,782 elementary gates**
and **76,009 logical qubits**: an implementation check, not a practical speedup
or a compiled full QFT. See `NEW_METHODS.md`, §17, `RESULTS.md`, Finding 19,
and [`data/direct_induction_gate_M3_b9_c9.json`](data/direct_induction_gate_M3_b9_c9.json).
That checkpoint passed **1346 tests**, including **153 new gate-level checks**.
External proof review and publication-priority review remain separate tasks.

Audit and full-QFT milestone: the simpler theorem now gives explicit
**O(n^(5/2) L^5) gates / O(n L^3) live qubits**, where
L = 2 + ceil(log₂(n+1)) + ceil(log₂(1/ε)). Public classical compilation
costs n^(5/2) poly(L) **bit operations**, using a finite-precision synthesis
theorem. A guarded integer-arithmetic reference and independent amplification
tests accompany the audit. The entire **S₃ QFT** now compiles, including
right-tableau updates and shared dirty-workspace reuse without reset.
All six inputs give measured clean operator error **0.005074**, using
**29,545,004 Clifford+T gates and 76,019 logical qubits**. These large,
unoptimized counts are a correctness milestone, not a practical speedup.
The prototype is not yet the theorem's arbitrary-precision compiler.
See `NEW_METHODS.md`, §18, `RESULTS.md`, Finding 20, and
[`data/full_qft_gate_S3_b9_c9.json`](data/full_qft_gate_S3_b9_c9.json).
The full suite passes **1398 tests**, including **52 new precision,
composition, and whole-QFT checks**. The theorem source compiles successfully.

## Setup

```sh
uv sync          # installs the package (src/fourier) and all dependencies
uv run pytest    # structural, numerical, and circuit-regression tests
```

Dependencies of note: [`yungdiagram`](https://github.com/natestemen/yungDiagram)
(Young-diagram combinatorics), Qiskit, BQSKit, sympy.  `flagsynth` is a
private SSH git dependency (XanaduAI) used only by
`experiments/flagsynth_scaling.py`.

## Layout

```
src/fourier/          the library — one canonical implementation of everything
  diagrams.py         partition generators, k-addable enumeration, staircase
  amatrix.py          A(λ): numeric, exact-symbolic, generic-family symbolic,
                      real-content extension, Cauchy form + fast mat-vecs
  decompositions.py   Givens factorization, CS decomposition, CS butterfly,
                      dependency depth, Wei–Di optimal 2-qubit synthesis
  structured.py       arrowhead/Jacobi, skew-Cayley, and symmetric-content
                      parity factorizations
  staircase_lift.py   exact two-Krawtchouk lift, clean flag erasure, Dicke reference
  projective_transport.py  signed-permutation transport between content families
  compact_shapes.py  fixed-capacity reversible Frobenius shape/corner schedules
  weyl.py             Weyl coordinates, symbolic KAK (Tucci), leakiness,
                      block-rotation normal form
  circuits.py         one-hot Givens circuits, Wei–Di circuit, Qiskit/BQSKit
                      compile + gate-count helpers
tests/                pytest suite for all of the above
experiments/          thin scripts, one per research question (see below)
data/                 kept datasets (expensive to regenerate) and plots
```

The A-matrix convention (row/column ordering, constant column) is documented
once, in `src/fourier/amatrix.py`, and everything follows it.

## Experiments

Each script has a docstring stating the question it answers, the finding it
supports, and the expected result; all write plots to `data/plots/` and CSVs
to `data/`.  Run with `uv run python experiments/<name>.py --help`.

| Theme | Scripts |
|---|---|
| Weyl chamber (Finding 1) | `weyl_scan.py`, `symbolic_weyl_scan.py`, `family_weyl.py`, `param_family_weyl.py`, `ac_family_weyl.py`, `weyl_pointcloud.py`, `weyl_region.py`, `fsim_weyl.py`, `leakiness.py`, `block_rotation.py` |
| Compiler baselines (Finding 2) | `compile_benchmark.py`, `generate_u3_dataset.py`, `gun_u3_params.py` |
| Cauchy structure (Finding 3) | `cauchy_structure.py` |
| Recursion searches (Finding 4, 7) | `reduction_search.py`, `hook_guided_search.py`, `cs_subblock_match.py`, `cs_subblock_cauchy.py`, `relate_a_matrices.py`, `mixing_rank.py` |
| HSS structure (Findings 9–11) | `hss_structure.py`, `ulv_circuit.py`, `ulv_diversity.py`, `ulv_explicit_basis.py`, `ulv_scaling.py`, `ulv_accuracy.py`, `verify_proof_lemmas.py` |
| New structured routes | `ulv_core_transforms.py`, `quasiseparable_core.py`, `quasiseparable_cayley.py`, `five_givens_search.py` |
| Coherent-cost improvements | `structured_qr_probe.py`, `staircase_krawtchouk_lift.py` |
| New overlap/Fourier routes | `coherent_overlap_encoding.py`, `log_weight_multipole.py`, `reversible_join_probe.py`, `hilbert_block_encoding.py`, `rectangular_staircase_probe.py` |
| Subcubic full QFT | `streaming_induction.py`, `certified_multipole.py`, `precision_budget.py` |
| Simpler direct-induction theorem | `direct_induction_precision.py`, `direct_induction_bookkeeping.py`; quantum contracts in `tests/test_direct_induction_contract.py` |
| Complete gate-level induction | `direct_induction_gate_probe.py`, `direct_induction_gate_angles.py`, `direct_induction_gate_bookkeeping.py`, `reversible_gate_ir.py`, `finite_gate_lowering.py`, `gate_block_simulator.py` |
| Full small QFT and explicit precision audit | `full_qft_gate_probe.py`, `direct_qft_precision_schedule.py`; generic composition regressions in `tests/test_qft_composition_audit.py` |
| Beyond the 5/2 exponent | `prefix_amortization_probe.py`, `sublinear_core_probe.py`, `alternative_qft_probe.py`, `transversal_encoding_probe.py` |
| Restricted 5/2 optimality | `constant_error_local_probe.py`, `modular_lower_bound_probe.py`, `amplification_lower_bound_probe.py` |
| Synthesis (Findings 5, 6, 8) | `wei_di_synthesis.py`, `givens_vs_cs.py`, `brickwork.py`, `flagsynth_scaling.py`, `raau3_layers.py`, `min_givens_count.py` |
| Symbolic / small cases | `symbolic_small_amatrices.py`, `symbolic_mn_family.py`, `random_3qubit.py` |

## Data

| Path | What | Regenerate |
|---|---|---|
| `data/4_addable_size_23_u3_cnot.csv` | per-gate u3 params + Weyl coords, BQSKit-compiled, every 4-addable diagram to size 23 | expensive — `experiments/generate_u3_dataset.py` |
| `data/u3_params.csv` | u3 params, BQSKit opt-4, 4-addable to size 20 | expensive — same script |
| `data/weyl_4_addable_size_25.csv` | Weyl (a,b,c) per diagram to size 25 | minutes — `experiments/weyl_scan.py` |
| `data/quasiseparable_core.csv` | sequential core-Givens count/error sweep | seconds — `experiments/quasiseparable_core.py` |
| `data/quasiseparable_cayley.csv` | all-cut ranks of A and its Cayley generator | seconds — `experiments/quasiseparable_cayley.py` |
| `data/direct_induction_gate_M3_b9_c9.json` | complete M=3 native/Clifford+T induction audit, all six columns | `experiments/direct_induction_gate_probe.py --level 3 --fraction-bits 9 --angle-bits 9` |
| `data/full_qft_gate_S3_b9_c9.json` | entire S₃ native/Clifford+T QFT, all six group inputs, shared workspace and boundaries | `experiments/full_qft_gate_probe.py --degree 3 --fraction-bits 9 --angle-bits 9` |
| `data/plots/` | figures; regenerable by the experiment named in each file |

## History

This repo began as ~70 one-off scripts plus a notebook; it was reorganized
(library + experiments + tests) in July 2026.  The pre-reorganization state
is preserved in git history — every deleted script is recoverable from the
commit tagged `pre-reorg` (or the commit history before the reorganization
commit).
