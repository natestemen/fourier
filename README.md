# fourier

Circuit decompositions of **A-matrices**: the k×k orthogonal matrices A(λ)
attached to a Young diagram λ with k addable cells by the symmetric-group
branching rule.  They are the local building blocks of the quantum Fourier
transform on Sₙ; the research question is whether their combinatorial
structure admits circuits below the generic O(k²) two-qubit-gate bound.

**What is known so far lives in [`RESULTS.md`](RESULTS.md)** (with
[`report.md`](report.md) as the original March-2026 narrative,
[`ulv-note.md`](ulv-note.md) as a shareable expert note on the July-2026
HSS/ULV decomposition result, and [`ulv-proof.md`](ulv-proof.md) as the
proof draft for its Õ(k) rotation count).  Headlines:
all 4-addable A-matrices sit on the a = π/4 Weyl face and synthesize optimally
as 3 CNOT + 6 Ry; the family carries Cauchy/displacement-rank-1 structure
giving O(k log² k) *classical* algorithms; compilers do not beat O(k²) gates;
the CS sub-block recursion conjecture is falsified exactly; and A-matrices
are orthogonal HSS matrices with k-independent ε-ranks, which a constructive
ULV factorization converts into ε-approximate circuits with near-linear
rotation counts (8.7% of the dense count at k = 1024, error ~10⁻⁴).

## Setup

```sh
uv sync          # installs the package (src/fourier) and all dependencies
uv run pytest    # 70+ tests pinning every structural claim
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
| HSS structure (Findings 9–11) | `hss_structure.py`, `ulv_circuit.py`, `ulv_diversity.py`, `ulv_explicit_basis.py`, `verify_proof_lemmas.py` |
| Synthesis (Findings 5, 6, 8) | `wei_di_synthesis.py`, `givens_vs_cs.py`, `brickwork.py`, `flagsynth_scaling.py`, `raau3_layers.py`, `min_givens_count.py` |
| Symbolic / small cases | `symbolic_small_amatrices.py`, `symbolic_mn_family.py`, `random_3qubit.py` |

## Data

| Path | What | Regenerate |
|---|---|---|
| `data/4_addable_size_23_u3_cnot.csv` | per-gate u3 params + Weyl coords, BQSKit-compiled, every 4-addable diagram to size 23 | expensive — `experiments/generate_u3_dataset.py` |
| `data/u3_params.csv` | u3 params, BQSKit opt-4, 4-addable to size 20 | expensive — same script |
| `data/weyl_4_addable_size_25.csv` | Weyl (a,b,c) per diagram to size 25 | minutes — `experiments/weyl_scan.py` |
| `data/plots/` | figures; regenerable by the experiment named in each file |

## History

This repo began as ~70 one-off scripts plus a notebook; it was reorganized
(library + experiments + tests) in July 2026.  The pre-reorganization state
is preserved in git history — every deleted script is recoverable from the
commit tagged `pre-reorg` (or the commit history before the reorganization
commit).
