# New decomposition methods (July 2026)

This note records the methods found after re-auditing the exact and
approximate A-matrix decompositions.  The main conclusion is that the exact
quadratic barrier is not established.  There is now an exact half-count
construction for every centrally symmetric content sequence, an exact
Jacobi/Hahn factorization for every A-matrix, and a roughly one-third
constant reduction in the existing approximate ULV circuit.  A new
sequential quasiseparable construction gives the strongest approximate
counts and an exact `O(k s)` theorem for rank-`s` orthogonal matrices.

## Summary

| Method | Scope | Guarantee | Rotation status |
|---|---|---|---|
| Parity split | Staircases and all centrally symmetric/self-conjugate diagrams | Exact | About half the dense count |
| Jacobi/associated-polynomial factorization | Every A-matrix | Exact | Fast arithmetic is known; a fast Givens circuit remains open |
| Grassmann/core ULV | Every tested content profile | Approximate, explicit Givens sequence | 30–34% fewer rotations than matched full-local ULV at fixed error |
| Sequential quasiseparable cores | Every tested content profile | Exact `O(k s)` theorem; approximate on A | Best new counts: 18,937 at `k=1024`, error `4.0e-5` |
| Skew-Cayley quasiseparable route | Every tested content profile | Exact orthogonality after any skew truncation | Promising direct `O(k r)` target; conversion is not implemented |
| Five-Givens charts | `k = 4` examples | Exact | Refutes the old six-rotation heuristic claim |

## End-to-end comparison: rotations versus angle generation

The near-linear rotation counts do **not** yet give a near-linear coherent
implementation.  The distinction is important here: in the larger
\(S_n\)-QFT, the diagram/content data can be in superposition, so the angles
must be evaluated reversibly, applied, and uncomputed.  A classical SVD
performed before the quantum computation is not an angle oracle.

The table below separates those two resources.  Let \(b\) be the local
ULV/window size, \(r\) its retained HSS rank, \(d=b-r\), \(s\) the maximum
prefix/suffix quasiseparable rank, \(L\) the content span, and \(I\) the
number of nonlinear-optimizer iterations.  Time bounds are fixed-precision
real-arithmetic costs for the implementations currently in this repository;
they include a dense working matrix but suppress bit complexity.  Signs are
free, and every “rotation” is an abstract two-level Givens rotation.

| Status | Method | Scope / guarantee | Plane rotations \(N\) | Current classical work to obtain the angles | Coherent-angle assessment |
|---|---|---|---:|---:|---|
| Previous | Dense column Givens | Exact, every \(A\) | \(k(k-1)/2\) | \(\Theta(k^3)\) time, \(\Theta(k^2)\) memory | Fixed feed-forward word and reversible `atan2` are possible in principle, but they require a dense mutable state and cubic arithmetic |
| Previous | Recursive CS butterfly | Exact, every \(A\) | \(k(k-1)/2\); depth \(O(k)\) | \(\Theta(k^3)\) time, \(\Theta(k^2)\) memory | Current route uses global CSD/SVD and gauge choices; no efficient content-local oracle is known |
| Previous | Compiler / variational brickwork | Small-\(k\) approximation only | No subquadratic bound | At least \(O(INk)\) for dense objective/gradient evaluations; \(O(Ik^3)\) when \(N=\Theta(k^2)\) | Target-dependent nonlinear optimization cannot serve as a uniform coherent angle routine |
| Previous | Completed-local SVD ULV | Approximate, every \(A\) | Empirically \(\approx4.7k\log_2 k\) at fixed \(\delta\).  The draft \(O(k\log(2L)(\log k+\log(1/\epsilon)))\) bound requires the later whole-group merge invariant, not the original position re-chunking | \(O(k^2b+kb^2)\) time, \(\Theta(k^2)\) memory | Not ready: repeated wide SVDs, threshold ranks, and a dense working matrix |
| Previous | Proxy-pole / explicit-basis ULV | Approximate, every \(A\) | \(1.24\)–\(1.40\) times the SVD-ULV count in tests | \(O(kN+k^2\operatorname{poly}(b,r))\) time, \(\Theta(k^2)\) memory in the prototype | **Partial:** compression spaces are formula-derived and feed-forward; retirement QR, pivots, rank padding, reversible depth, and uncomputation remain |
| New | Symmetry/parity split | Exact for centrally symmetric contents | \(k-1+\binom{\lceil k/2\rceil}{2}+\binom{\lfloor k/2\rfloor}{2}=k^2/4+O(k)\) | \(\Theta(k^3)\) time, \(\Theta(k^2)\) memory | The \(k-1\) outer angles are fixed \(\pi/4\); the two dense half-size blocks still need coherent dense QR |
| New | Jacobi/Hahn factorization | Exact factor identity for every \(A\) | No short Givens result; densifying both factors gives \((k-1)^2\) | Current general constructor: \(\Theta(k^3)\) time and \(\Theta(k^2)\) memory.  Staircase recurrence data: \(O(k)\) | The known \(O(k\log^2 k)\) transform algorithm applies a classical transform; it does not generate an orthogonal Givens network |
| New | Grassmann/core ULV | Approximate, every \(A\) | Local \(2rd+d(d-1)/2\); same asymptotic draft bound as ULV and 30–34% fewer rotations in matched tests | \(O(k^2b+kr^3)\) time, \(\Theta(k^2)\) memory | Lower rotation constant only so far: the prototype still obtains coupling spaces and survivor pivots from SVD/QR |
| New | Sequential quasiseparable cores | Exact for rank-\(s\) orthogonal \(U\); approximate for \(A\) | \(N<2sk+2(s+1)^2\); conditionally \(O(k\,s_{\epsilon/k})\) for approximation | Fixed-\(s\) prototype: \(O(k^2s^2+ks^3)\) time, \(\Theta(k^2)\) memory | Best count, but not ready: it still uses repeated suffix SVDs and an adaptive word |
| New | Skew-Cayley route | Proposed approximate route; skew truncation preserves orthogonality | Conditional target \(O(kr)\); no product emitted yet | \(\Theta(k^3)\) time and \(\Theta(k^2)\) memory merely to form the current dense Cayley generator | No angle routine yet; global skew compression and product conversion are both missing |
| Previous structure, not a decomposition | Cauchy / Toeplitz fast matvec | Applies \(A\) to a classical vector | N/A | Theoretical \(O(k\log^2 k)\) generally and \(O(k\log k)\) for staircases | Produces no angles and its internal arithmetic is not a unitary amplitude circuit |

The \(k=4\) staircase's exact five-Givens word is an additional \(O(1)\)
closed-form special case, not a scalable family-wide method.

There are two defensible targets for replacing the dense angle generators.
These are research targets, not achieved complexities:

| Candidate | Conditional arithmetic target | Dependency depth / live state | Missing step |
|---|---:|---:|---|
| Formula-basis tree ULV, preferably with the new core transforms | \(\widetilde O(kr^2)\) | Target \(\widetilde O(r^2\log k)\) arithmetic depth; structured rather than \(k^2\) state | Deterministic proxy bases, formula-only retirement, fixed rank padding, and reversible local QR |
| Generator-based sequential quasiseparable cores | \(O(ks^2)\) once rank-\(s\) generators are available | Sequential \(O(ks^2)\) dependency depth; \(O(s^2)\) live local state plus streamed generators | Construct the generators directly from contents and replace SVD/pivoted survivor selection by a fixed stable core recurrence |
| Staircase Hahn route | \(O(k\log^2 k)\) would be the natural target | Potentially polylogarithmic if an orthogonal butterfly exists | No orthogonal butterfly or Givens factorization is known |
| Global skew-quasiseparable Cayley route | \(\widetilde O(kr^2)\) work and \(O(kr)\) emitted rotations are plausible | Depends on the missing structured solve/factorization | Build one consistent skew approximation, then convert it to a product without densifying |

Several qualifications are essential for using either table in the full
algorithm:

1. Building all transition weights from arbitrary interlaced contents costs
   \(\Theta(k^2)\) in the current code, before any factorization.  The
   identity \(\alpha_i^2=q(x_i)/p'(x_i)\) suggests a
   product/remainder-tree target of \(\widetilde O(k)\), but no stable
   reversible implementation is present.  Staircase weights avoid this
   bottleneck through a constant-work hypergeometric ratio recurrence.
2. If the larger QFT uses \(M\) such matrices with total allowed error
   \(\epsilon_{\rm tot}\), take a per-matrix budget
   \(\epsilon_A=O(\epsilon_{\rm tot}/M)\).  For \(N\) rotations, the
   sufficient per-angle accuracy is
   \(O(\epsilon_A/N)\), hence
   \(P=\Theta(\log_2(NM/\epsilon_{\rm tot}))\) bits plus conditioning guard
   bits.  Each arithmetic count above must then be multiplied by the cost
   of reversible \(P\)-bit division, square root, `atan2`, and
   uncomputation.
3. Numerical ranks, SVD signs, pivoted-QR permutations, and adaptive window
   sizes depend on the diagram.  In a coherent superposition, the circuit
   must use a fixed worst-case word (padding inactive slots with zero
   angles) or carry and uncompute all branch decisions.  Thus the relevant
   \(r\) or \(s\) is the worst case over every diagram sharing that circuit,
   not the favorable rank of one classical instance.
4. The ULV experiments use a local threshold \(\delta\).  For rigorous
   global error \(\epsilon_A\), use roughly
   \(\delta=\Theta(\epsilon_A b/k)\) hierarchically and conservatively
   \(\delta=\Theta(\epsilon_A/k)\) sequentially.  Fixed-\(\delta\)
   empirical fits are not fixed-global-error asymptotics.
5. A plane rotation is a two-qubit gate only in the one-hot encoding.  On
   \(\lceil\log_2k\rceil\) binary address qubits it becomes a controlled
   two-level operation with a separate address/control cost.

The practical conclusion is that the **sequential quasiseparable method has
the best rotation count**, while **formula-basis tree ULV currently has the
more credible coherent-depth story**.  Neither is end-to-end coherent yet.

## 1. Exact arrowhead and Jacobi factorizations

Let the addable and removable contents strictly interlace as

\[
x_0>y_0>x_1>\cdots>y_{k-2}>x_{k-1}.
\]

Write

\[
w_i=\frac{\prod_j(x_i-y_j)}
          {\prod_{\ell\ne i}(x_i-x_\ell)},\qquad
\gamma_j=-\frac{\prod_i(y_j-x_i)}
               {\prod_{\ell\ne j}(y_j-y_\ell)},
\]

with the signs understood in the interlaced order, so both quantities are
positive.  With \(\alpha_i=\sqrt{w_i}\) and
\(\beta_j=\sqrt{\gamma_j}\),

\[
A_{i0}=\alpha_i,\qquad
A_{i,j+1}=\frac{\alpha_i\beta_j}{x_i-y_j}.
\]

The constant column can be included in the Cauchy structure rather than
treated separately.  For any \(\eta\) away from the nodes, put

\[
u_i=\frac1{x_i-\eta},\qquad
v_0=0,\qquad v_{j+1}=\frac1{y_j-\eta}.
\]

With suitable diagonal scales \(r,c\), implemented by
`projective_cauchy_from_contents`,

\[
A=\operatorname{diag}(r)
\left[\frac1{u_i-v_j}\right]
\operatorname{diag}(c),\qquad
\operatorname{diag}(u)A-A\operatorname{diag}(v)=rc^T.
\]

Thus the *full square matrix* has projective displacement rank one; the
distinguished column is simply the Cauchy node at infinity before the
Möbius change of coordinate.

The full matrix, including its constant column, obeys the exact intertwining
identity

\[
\operatorname{diag}(x)A=AH,\qquad
H=A^T\operatorname{diag}(x)A
 =
\begin{pmatrix}
s & \beta^T\\
\beta & \operatorname{diag}(y)
\end{pmatrix},
\]

where \(s=\sum_i x_i-\sum_j y_j=\sum_iw_ix_i\).  Thus \(A\) is the
normalized eigenvector transform of a symmetric arrowhead matrix.  This is
implemented by `fourier.structured.arrowhead_from_contents`.

There is an even more useful polynomial form.  Apply symmetric Lanczos to
\(\operatorname{diag}(x)\), starting at \(\alpha\).  It produces the
orthogonal-polynomial evaluation transform \(P_x\) and a Jacobi matrix

\[
J=P_x^T\operatorname{diag}(x)P_x.
\]

The removable contents are exactly the eigenvalues of \(J_{1:,1:}\).  If
\(P_y\) diagonalizes that trailing Jacobi matrix, then, up to free column
signs,

\[
\boxed{A=P_x(1\oplus P_y^T).}
\]

`fourier.structured.jacobi_factorization_from_contents` constructs this
factorization from the nodes alone and the tests verify it on diagram and
real-content instances.

For the staircase, with \(N=k-1\),

\[
w_i=\frac{(1/2)_i}{i!}\frac{(1/2)_{N-i}}{(N-i)!},
\]

so \(P_x\) is a symmetric Hahn transform with parameters
\((-1/2,-1/2)\); the associated transform has parameters
\((+1/2,+1/2)\).  The Jacobi diagonal is zero and

\[
b_0^2=\frac{N(N+1)}2,\qquad
b_j^2=\frac{(N-j)(N+j+1)}4\quad(1\le j<N).
\]

This gives an exact Hahn/Clebsch–Gordan route.  Driscoll, Healy and Rockmore
give `O(k log^2 k)` arithmetic for any discrete orthogonal-polynomial
transform satisfying a three-term recurrence.  Their recurrence circuit is
not a product of orthogonal two-level rotations, so this is not yet an
`O(k log^2 k)` Givens decomposition.

## 2. Exact half-count decomposition for symmetric contents

Suppose, after a common translation,

\[
x_i=-x_{k-1-i},\qquad y_j=-y_{k-2-j}.
\]

This includes every staircase and every self-conjugate diagram.  Let \(R\)
reverse the rows and let \(C=1\oplus(-R_{\rm tail})\).  The Cauchy formula
gives

\[
RA=AC.
\]

Both involutions are diagonalized by disjoint \(\pi/4\) rotations on
reversed coordinate pairs.  If \(p=\lceil k/2\rceil\) and
\(q=\lfloor k/2\rfloor\), those fixed basis changes give

\[
A=U(B_+\oplus B_-)V^T,\qquad
B_+\in O(p),\quad B_-\in O(q).
\]

There are \(\lfloor k/2\rfloor+\lfloor(k-1)/2\rfloor=k-1\) fixed rotations.
Factoring the two blocks densely therefore uses

\[
\boxed{
N_{\rm sym}
=k-1+\frac{p(p-1)}2+\frac{q(q-1)}2
=\frac{k^2}{4}+O(k),
}
\]

versus \(k(k-1)/2\) generically.  The implementation is
`fourier.structured.symmetric_content_factorization`.

At \(k=4\), the count is five rather than six.  An explicit staircase word is

\[
\begin{aligned}
&(1,3),(2,3),(0,1),(0,3),(1,2),\\
&\theta=\left(
\frac\pi4,\arctan\sqrt{\frac5{27}},\arctan\sqrt{\frac35},
-\frac{3\pi}4,-\frac{3\pi}4
\right),
\end{aligned}
\]

followed by `diag(-1,1,1,1)`.  This identity is pinned by
`tests/test_structured.py`.  `experiments/five_givens_search.py` also
enumerates the pruned length-five topology set.  A five-rotation word fitted
to one generic diagram does not form a family-wide chart: its tangent-space
mismatch is about `0.26`, and continuation fails.  The general exact problem
therefore remains open.

The only rigorous fixed-word lower bound currently available is the family
dimension \(2k-3\), not \(k(k-1)/2\).  Contents are recoverable up to their
two-parameter affine gauge, so a finite collection of words with fewer than
\(2k-3\) angles cannot cover the real-content family.

## 3. Fewer rotations in the existing ULV construction

The old ULV implementation completed and densely decomposed two local
\(b\times b\) orthogonal matrices.  It also repartitioned the flat survivor
list, whereas the proof requires previously formed survivor groups to remain
indivisible.

`experiments/ulv_core_transforms.py` fixes both issues.  It merges whole
groups and operates only on the active subspaces.  If a block has coupling
rank \(r\) and retires \(d=b-r\) coordinates:

- choosing `r` survivor coordinate rows and nulling each other row costs
  exactly \(rd\) rotations, the dimension of \(\mathrm{Gr}(r,b)\);
- thin retirement QR costs at most
  \(bd-d(d+1)/2\) rotations.

The total local cost is

\[
2rd+\frac{d(d-1)}2.
\]

At the common choice \(b=2r\), this is
\((5r^2-r)/2\), compared with \(b(b-1)=4r^2-2r\) for the two completed
factors.

Paired results at `b=32`, `delta=1e-4` are:

| k | core rotations | completed-local rotations | saving | operator error |
|---:|---:|---:|---:|---:|
| 128 | 3,043 | 4,403 | 30.9% | 9.08e-5 |
| 256 | 6,745 | 10,139 | 33.5% | 1.04e-4 |
| 512 | 14,397 | 21,207 | 32.1% | 1.28e-4 |
| 1,024 | 30,452 | 45,556 | 33.2% | 1.44e-4 |
| 2,048 | 62,150 | 91,648 | 32.2% | 1.59e-4 |

The completed and core versions have the same measured error in each paired
run; only the representation of the local orthogonal subspaces changes.
With a separately tuned leaf size `b=12`, the staircase uses 82,451
rotations at `k=4096`, error `2.46e-4`.  That tuned number should not be
compared as a pure constant-factor result with the old fixed-`b=32` table.

## 4. Direct sequential quasiseparable Givens circuit

`experiments/quasiseparable_core.py` removes the hierarchy altogether.  The
active coordinates are a survivor bundle from a processed prefix followed by
an untouched suffix.  Compress the first `b` active rows against the suffix
using the same Grassmann cores, retire the decoupled row/column pairs, and
advance the prefix.

There is a simple exact theorem.  Suppose \(U\in O(k)\) has upper
quasiseparable rank \(s\):

\[
\operatorname{rank} U[:i,i:]\le s\qquad(1\le i<k).
\]

Choose \(b=s+1\).  If the current coupling rank is \(r\) and
\(d=b-r\) pairs retire, the step costs

\[
rd+\left(bd-\frac{d(d+1)}2\right)
=2rd+\frac{d(d-1)}2
\le 2sd.
\]

All row and column transformations are prefix-local, so the next
prefix/suffix rank is still at most \(s\).  At least one pair retires per
step, and the dense final block has size at most \(2(s+1)\).  Therefore

\[
\boxed{N<2sk+2(s+1)^2}
\]

ordinary Givens rotations plus signs.  This is a product-only construction:
there is no unevaluated banded weight matrix.  The test suite verifies the
bound and operation-log replay on an exactly rank-one quasiseparable
orthogonal family.

For A-matrices, use numerical ranks and grow the window only when it
saturates.  At `delta=1e-5`:

| k | staircase rotations | error | random-content rotations | error |
|---:|---:|---:|---:|---:|
| 128 | 1,661 | 1.96e-5 | 1,801 | 1.44e-5 |
| 256 | 3,836 | 2.00e-5 | 3,915 | 2.03e-5 |
| 512 | 8,615 | 2.67e-5 | 8,994 | 2.18e-5 |
| 1,024 | 18,937 | 3.99e-5 | 19,805 | 2.73e-5 |
| 2,048 | 39,627 | 5.90e-5 | 44,011 | 3.69e-5 |

At `k=1024`, the staircase count is `3.6%` of the dense `523,776` and
`62%` of the improved hierarchical core-ULV count, at a smaller local
threshold.  With a fixed `s=10`, the `k=2048` staircase uses `39,901`
rotations, below the theorem bound `41,202`.

For approximation, each discarded coupling tail has norm at most `delta`.
A conservative global target can take `delta=O(epsilon/k)`; the substantially
smaller observed accumulation is empirical.  The prototype still computes
dense SVDs.  Replacing those with the existing Cauchy proxy-pole bases is the
main step needed for fast classical angle generation.

## 5. Skew-Cayley route to a structure-preserving alternative

Use the fixed free reflection

\[
D=\operatorname{diag}(1,-1,\ldots,-1),\qquad Q=AD,
\]

and form

\[
S=(I-Q)(I+Q)^{-1}.
\]

For all tested A-matrices, `I+Q` is well conditioned; \(S\) is exactly
skew-symmetric.  Conversely,

\[
A=(I+S)^{-1}(I-S)D.
\]

This is useful because any skew-symmetric approximation \(\widehat S\)
maps back to an exactly orthogonal matrix.  Moreover,

\[
\|Q(S)-Q(\widehat S)\|_2
\le 2\|S-\widehat S\|_2,
\]

since the resolvent norm of `I + S` is at most one for real skew \(S\).

`experiments/quasiseparable_cayley.py` measures every lower-left cut, not
only dyadic HSS blocks.  At tolerance `1e-4`, the maximum cut ranks of the
staircase Cayley generator for `k=32,64,128,256` are `5,6,6,7`; one
random-content sample gives `5,6,7,7`.  The observed conditions of `I+AD`
stay between `1.49` and `1.75`.  At tolerance `1e-8`, the staircase ranks
are `8,9,11,12`.

The sequential construction above already turns A's numerical
quasiseparability into a product of rotations.  Cayley remains an attractive
alternative because skew compression preserves orthogonality by
construction and has the clean `2 epsilon` error bound.  The missing step is
important: separately truncating cut blocks does not produce one globally
consistent skew matrix.  A valid implementation should construct that
global skew approximation, apply the Cayley map, and compare its product
against sequential cores.

## 6. Recommended next tests

1. Replace the sequential factorization's dense SVDs with proxy-pole/Cauchy
   bases and independently replay the resulting formula-generated angles.
2. Implement the global skew-quasiseparable Cayley approximation and compare
   its emitted Givens list against sequential cores and core ULV.
3. Search for an orthogonal butterfly realization of the two staircase Hahn
   transforms.  Fast arithmetic alone is not enough.
4. At `k=5`, exhaust/prune seven- and eight-rotation support-complete words.
   The lower bound is seven; sparse random search has already found nine
   versus ten dense, but this is not a minimality result.
5. Test projective/scattering coordinates.  Treating the constant column as
   a Cauchy node at infinity puts the family on two alternating point sets on
   the real projective line modulo `PGL(2)`, again of dimension `2k-3`.
6. Separate one-hot abstract rotations from binary-encoded controlled
   two-level gates in every resource table.

## Primary references

- [Fasino, “Orthogonal Cauchy-like matrices”](https://link.springer.com/article/10.1007/s11075-022-01391-y)
- [Driscoll–Healy–Rockmore, “Fast Discrete Polynomial Transforms”](https://digitalcommons.dartmouth.edu/facoa/2066/)
- [Delvaux–Van Barel, “A Givens-Weight Representation for Rank Structured Matrices”](https://doi.org/10.1137/060654967)
- [Delvaux–Van Barel, “Unitary Rank Structured Matrices”](https://doi.org/10.1016/j.cam.2007.03.020)
- [Delvaux–Frederix–Van Barel, hierarchical to unitary-weight conversion](https://etna.ricam.oeaw.ac.at/vol.33.2008-2009/pp163-188.dir/pp163-188.pdf)
- [Stewart, stability of orthogonal transformations of rank-structured matrices](https://epubs.siam.org/doi/10.1137/15M1041900)
- [Beckermann–Townsend, displacement structure and singular-value bounds](https://epubs.siam.org/doi/10.1137/19M1244433)
