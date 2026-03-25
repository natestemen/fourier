# Executive Summary: A-Matrix Circuit Decomposition

*March 2026*

---

## Background

The A-matrices for Young diagrams arise from the branching rule of the symmetric group: for a partition λ with k addable cells, A(λ) is a k×k orthogonal matrix whose (i,j) entry encodes the transition amplitude between the irrep λ and its neighbors under one step of the branching rule. Each row is indexed by an addable cell and each column by a removable cell (plus one dummy "constant" column).

The question I set out to investigate is whether these matrices — which are highly structured and determined entirely by combinatorial data — admit a circuit decomposition that is substantially more efficient than the naive Givens triangularization, which uses k(k−1)/2 two-qubit gates and has O(k²) sequential depth.

---

## Finding 1: Weyl Chamber — All 2-Qubit A-Matrices Have a = π/4

The first concrete thing I found is about the two-qubit (k=4) case. Every 4-addable A-matrix, regardless of which Young diagram it comes from, has the same KAK invariant: **a = π/4**. The other two Weyl coordinates (b, c) vary continuously across the family of diagrams — and I mapped them out for a large sample, both symbolically and numerically — but a is always pinned at π/4.

This is a meaningful geometric fact: a = π/4 is the boundary of the Weyl chamber associated with maximally entangling unitaries (like CNOT and CZ). All 4-addable A-matrices live on this face. I verified this symbolically via the magic-basis KAK decomposition and numerically by computing Qiskit's `TwoQubitWeylDecomposition` for every 4-addable diagram up to large sizes.

I then checked whether the 4×4 matrices have a simple eigenvalue structure consistent with this. They do: every 4-addable A-matrix has eigenvalues {+1, −1, e^{iθ}, e^{−iθ}} and can be conjugated into the block form diag(1, −1) ⊕ R(θ) by an orthogonal change of basis. The angle θ encodes (b, c). This block decomposition holds up to numerical precision across all tested cases.

I also checked whether A_{k+1} could be written as M(1 ⊕ A_k) for some orthogonal M — this would give a clean recursive relationship — but it does not hold in general.

One natural hypothesis for *why* a = π/4 is leakiness (Peterson, Crooks & Smith 2019): a gate U is leaky if there exists a nonzero h ∈ su(2) such that U·(h⊗I)·U† stays in the local Lie algebra k = su(2)⊗I + I⊗su(2). CZ and iSWAP are canonical leaky gates, both of which live on the a=π/4 face. The test reduces to checking whether a 18×3 real linear system has a nontrivial null space. Running this on all 8,231 four-addable A-matrices with max_size=30 gives: **0 leaky out of 8,231** — every single one has full rank 3, with minimum singular values solidly above zero (~0.9–2.3). A-matrices are genuinely non-leaky despite sharing the a=π/4 Weyl coordinate with CZ. The explanation for a=π/4 must be structural (real orthogonality, combinatorial constraint on entries) rather than the gate-algebra property of leakiness.

---

## Finding 2: Qiskit and BQSKit Both Compile at O(k²) — No Free Lunch

The natural thing to try first is: decompose A into Givens rotations (k(k−1)/2 of them), build a Qiskit circuit, and let a state-of-the-art compiler optimize it. I benchmarked this across k = 2 to 15, with ~300 sample partitions per k, using two compilers:

- **Qiskit** with `u3+cx` and `rx/ry/rz+cx` gate sets
- **BQSKit** at optimization level 3

Power-law fits to the compiled CX count confirm that both scale as **≈ a·k²** with exponent very close to 2. BQSKit does not do meaningfully better than Qiskit despite being a much more aggressive optimizer. The conclusion is that simply handing the Givens circuit to an existing compiler doesn't break the quadratic barrier — you need to exploit the structure at the level of the decomposition itself.

---

## Finding 3: Cauchy Structure and Its Consequences

The most important structural fact about A-matrices is that the non-constant part factors as:

```
A[:,1:] = diag(α) · C · diag(β)
```

where C[i,j] = 1 / (c(aᵢ) − c(rⱼ)) is a **Cauchy matrix**, with rows indexed by addable-cell contents and columns by removable-cell contents. The key features:

1. **Strict interlacing**: addable and removable contents alternate on the integer number line (e.g., for (3,2,1): 3A > 2R > 1A > 0R > −1A > −2R > −3A). This is guaranteed by the branching structure of Young diagrams — not an accident.
2. **All denominators positive**: interlacing means every c(aᵢ) − c(rⱼ) > 0, so C is a well-conditioned, sign-definite Cauchy matrix.
3. **Displacement rank 1**: diag(ac)·C − C·diag(rc) = **1**·**1**ᵀ. Verified for all tested partitions.

This displacement-rank-1 structure is the key to fast algorithms. Three levels of exploitation were worked out and verified numerically:

**GKO / superfast Cauchy algorithms → O(k log² k)**
Schur complements of a Cauchy matrix are again Cauchy (structure is preserved under Gaussian elimination). So instead of updating the full matrix — O(k²) per Givens rotation — you update only the displacement generator vectors — O(k) per step. The total work for the full triangular factorization drops from O(k³) to O(k log² k). This yields an O(k log² k) mat-vec, verified numerically.

**Toeplitz special case for staircase partitions → O(k log k)**
For the staircase partition λ = (k, k−1, …, 1), the weight matrix W[i,j] = 2(j−i)+1 is Toeplitz. The Cauchy core C is therefore a Toeplitz matrix, and a Toeplitz mat-vec can be computed in O(k log k) via FFT embedding in a circulant. Correctness verified for k = 2, 3, 4, 5.

**CS (cosine-sine) decomposition → O(k log k) circuit depth**
The CS decomposition splits A into:

```
A = [U₁ ⊕ U₂] · [C −S; S C] · [V₁ᵀ ⊕ V₂ᵀ]
```

The k/2 cosine-sine angles are determined by the singular values of an off-diagonal Cauchy sub-block, which costs O(k log k) using the displacement-rank structure. The U and V sub-blocks act on the top/bottom k/2 addable cells respectively. The recursion T(k) = 2T(k/2) + O(k) gives O(k log k) total **if** the sub-blocks are again A-matrices of smaller diagrams. That open question is the crux of the unsolved part.

For the specific case (3,2,1) [k=4], the CS decomposition was computed explicitly: 6 Givens rotations (2 CS angles + 2×1 for U + 2×1 for V), reconstruction error < 1e-14. The circuit depth improves from O(k²) sequential to O(k) parallel layers even in the generic case.

Separately, the gate-count benchmark (Givens vs CS butterfly across ~800 random partitions) confirmed: CS butterfly matches Givens in total gate count (both O(k²)) but achieves O(k) circuit depth, which is better for hardware with limited connectivity.

---

## The Core Unsolved Problem: Recursive Decomposition

The biggest question I worked on — and did not resolve — is whether the A-matrix family is **closed under some natural reduction**:

**Conjecture**: There exists a sequence of Givens rotations mapping A(λ) → A(λ') where λ' is obtained from λ by removing one addable cell.

If true, this would give a recursive decomposition of depth k−2 down to a 2×2 A-matrix, with each level requiring a manageable number of rotations. It would also explain why the CS sub-blocks have the right shape.

**Exhaustive search** tried every (pivot_row, target_col) Givens reduction on A(λ) and compared the result against every known (k−1)-addable A-matrix. Occasional matches were found at depths 1 and 2 — meaning specific reduction paths do land on lower-dimensional A-matrices — but no systematic pattern emerged. The match rate is too low and too irregular to suggest a general rule.

**Hook-guided beam search** tested the hypothesis that hook length ratios h₁:h₂ predict which 2×2 target the full k→2 reduction chain reaches. Hook-ratio-derived targets were enumerated, and a beam search with column-alignment scoring was used to find reduction paths. Results were inconsistent: some diagrams yield paths, most do not, and no combinatorial rule tied the successful cases to their hook lengths.

**The fundamental difficulty** seems to be that A-matrices, despite their Cauchy structure, behave like generic orthogonal matrices under Givens reduction. The Cauchy structure is a global property of the full matrix — it does not obviously descend to submatrices obtained by arbitrary row eliminations. The best commit message summary: "I'm starting to think these are potentially generic orthogonals."

---

## Finding 5: Optimal 2-Qubit Circuit — 3 CNOT + 6 Ry (Exact)

Wei & Di (arXiv:1203.0722) prove that any O(4) gate with det=−1 can be synthesized with at most **3 CNOT + 6 Ry** gates, and this is optimal (lower bound from Shende et al.). All 4-addable A-matrices have det=−1 (eigenvalues {+1, −1, e^{iθ}, e^{−iθ}} multiply to −1), so this result applies directly.

The so(4) Cartan decomposition underlying the circuit uses:
```
l = span{iI⊗σ_y, iσ_y⊗I}         ← local algebra: Ry-only, no Rz
a = span{iσ_x⊗σ_y, iσ_y⊗σ_z}     ← 2D Cartan subalgebra
```
Every X ∈ O(4) factors as `[Ry(θ₁)⊗Ry(θ₂)] · CNOT · [Ry(b)⊗Ry(a)] · CNOT · [Ry(θ₃)⊗Ry(θ₄)] · CNOT`. The 2D Cartan subalgebra is consistent with the Weyl finding: a=π/4 is fixed, only (b,c) vary, leaving exactly 2 free interaction parameters.

Numerically extracted the 6 parameters for every 4-addable A-matrix up to max_size=15 (311 diagrams). All 311 decomposed to residual < 10⁻¹⁶. Example circuit for A(3,2,1):

```
q_0: ─Ry(0.80)──■──Ry(2.24)──■──Ry(-2.96)──■──
q_1: ─Ry(-2.69)─X──Ry(1.75)──X──Ry(0.66)───X──
```
3 CX + 6 Ry = 9 gates total. Compare to naive Givens → Qiskit: ~12 CX. Wei-Di is the provably optimal synthesis for any 4-addable A-matrix. Script: `wei_di_circuit.py`.

---

## Open Directions

**1. Recursive decomposition via CS sub-blocks (most promising)**
The CS decomposition always gives two sub-blocks U₁ and U₂ of the right dimensions. The question is whether there exists a labeling of the k/2 addable cells in each half that makes these sub-blocks equal to the A-matrix of some smaller diagram. If yes, the O(k log k) recursion closes. This would require matching the CS singular values to hook-length-derived expressions — a computation that is tractable symbolically for small k.

**2. Explicit hardware-ready butterfly circuit**
Make the CS recursion concrete: write the full gate sequence, track qubit labels, and verify or falsify the sub-block A-matrix claim computationally for k = 6, 8, 10. This is a finite computation that should be decidable.

**3. Break the k² compiler barrier**
Use the Cauchy SVD + CS decomposition — rather than Givens triangularization — as the input to Qiskit/BQSKit and check whether the structured circuit is compiled more efficiently. The hypothesis is that the parallel CS layers expose cancellations that the sequential Givens circuit hides.

---

*Code for all of the above lives in `/fourier/`. Key files: `decompose_a_matrix.py` (Cauchy structure and fast algorithms), `benchmark_circuit.py` (Givens vs CS), `compile_benchmark.py` (Qiskit/BQSKit scaling), `exhaustive_reduce.py` and `hook_guided_reduce.py` (recursive decomposition search), `symbolic_kak.py` and `scan_symbolic_weyl.py` (Weyl chamber analysis).*
