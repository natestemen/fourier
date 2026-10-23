# New decomposition methods (updated October 2026)

## Latest search: beyond the 5/2 exponent

The best complete coherent bound established in this repository remains
**n^(5/2)·poly(log n,log(1/ε))**. This search has not established a smaller
exponent. It has produced a useful new structural theorem: adding one box
to a diagram changes its A transform, after aligning the surviving content
labels, by a correction of **rank O(log(1/δ)) to operator error δ**.
This is a potential tool for sharing work between steps, not yet a fast
coherent update circuit. Fresh-input access and cache-cleanup obstructions
explain why the distinction matters. See §14 for proofs, tests, and the
remaining algorithmic targets.

## Latest October 22 pass: a full subcubic QFT theorem draft

The current formal statement and proof are in
[`subcubic-qft-theorem.tex`](subcubic-qft-theorem.tex). They supersede the
earlier informal global rounding estimate below. The numerical proof now
uses exact rational truncated potentials with explicit polynomial word
bounds, rounding weights, iterates and probability leaves back to dyadics
at specified boundaries. The compiled document also states the valid
encoded domain, the full-workspace reflection condition, and the complete
streaming invariant. Independent external mathematical review is pending.

The compact streaming wrapper and finite-precision analysis now give a
constructive **n^(5/2)·poly(log n,log(1/ε))** gate bound for the complete
S_n QFT, with **n·poly(log n,log(1/ε))** workspace. Parameter generation,
shape handling, coherent controls and cleanup are included. A simpler
unbalanced version gives exponent 11/4 independently of the balancing step.

This is an algorithmic/circuit-existence result with a written proof and
verified reference components, **not a complete compiled fault-tolerant
circuit or a practical runtime benchmark**. The whole S_6 Fourier matrix
matches an independent representation-theoretic construction. See §13 for
the theorem, uniform precision/access model, reversible invariants,
validation and limitations.

## Earlier October 22 pass: a general near-linear coherent route

The strongest new candidate is **balanced row/column overlap encoding**.
For arbitrary integer Young contents, its derived resource bound is

\[
  k\,\operatorname{poly}(\log(2k),\log(2L),\log(1/\epsilon)),
\]

including parameter computation, controlled application and uncomputation.
Here k is the matrix dimension, L is the content span (at least one), and
ε is the allowed error of the complete clean implementation of one A.
The input model is a coherent, ordered corner list; extracting that list
from a different diagram encoding is an additional cost. Workspace is also
near-linear up to these logarithmic factors, **not** just O(log k) qubits.

This is an algorithmic construction with tested numerical components and
separate internal proof audits, **not a fully compiled or roundoff-certified
reversible circuit**. It is not a new exact Givens factorization. The key
change from ULV is to encode A as the overlap of two easily prepared sets
of states, then amplify that overlap. Classical fast matrix multiplication
alone would not justify a quantum gate bound; §10 supplies the state
preparation, normalization, data-access and cleanup argument.

In this table, a tilde hides powers of log k, log L and log(1/ε). The new
application and parameter columns are both included in the total.

| Method | Scope | Transformation work | Parameter work | Total coherent status |
|---|---|---|---|---|
| Generator QR, previous pass | All | Θ(k²) exact abstract Givens rotations | O(k²) scalar operations | Numerical implementation; uniform precision bound still open |
| Low-rank ULV/sequential cores, previous work | General profiles | Near-linear approximate rotation counts in theory/experiments under stated rank bounds | Existing constructors at least quadratic | Short word does not yet give a fast coherent implementation |
| **Balanced overlap + multipole preprocessing** | **All integer Young contents** | O(k log k) state-preparation rotations, plus arithmetic/control overhead | **Õ(k)** preprocessing and coefficient work | **Derived Õ(k) total bound; components tested, reversible compiler/roundoff certification unfinished** |
| Fractional-Fourier encoding | Rectangular staircases, step height h and width w | Õ(√k) block-encoding/amplification work | Polylogarithmic per call, included in total | Derived Õ(√k) total bound; scalar and matrix prototypes, no full circuit |
| Two-QFT cotangent family | Explicit restricted real-content family; rounded/scaled integer approximants | O(log² k + b log k) ideal gates for k a power of two and a coherent b-bit parameter | Included; b=O(log(1/ε)) | Actual small Qiskit circuits tested; integer approximation and recognition qualifications in §12 |

The overlap construction draws on established weighted quantum-walk
embeddings and oblivious amplitude amplification; those tools are not new.
The new contribution to this repository is their combination with the
A-specific sign bound, a finite positive balancing construction, and fast
coherent parameter computation. No claim of priority over the literature
is made. The earlier findings below are retained as a research history.

## Earlier October 22 update: count coherent arithmetic, not just rotations

Three additions change the earlier comparison:

1. **General contents:** a rank-one Cauchy-generator QR prototype emits the
   same `k(k-1)/2` exact Givens word with **O(k²)** scalar arithmetic,
   including transition-weight construction, instead of dense QR's cubic
   work. The word is fixed and needs no data-dependent pivoting or SVD.
   Worst-case finite-precision stability is not established.
2. **Staircases:** an exact reduction to **two Krawtchouk transforms** has
   been derived, including clean erasure of the constant-column branch flag.
   Combining this identity with the efficient binary SU(2)/Krawtchouk
   transforms in the cited 2026 literature gives a **polylogarithmic
   approximate quantum circuit including coefficient arithmetic**. This is
   a theorem composition, not a compiled implementation in this repository,
   and not a polylogarithmic Givens word on the original k coordinates.
3. **A larger fast family:** a projective change of content coordinates
   transports an A-matrix by signed permutations only. It extends the
   staircase construction to specific nonuniform, nonsymmetric diagrams,
   but does not cover arbitrary diagrams.

Let `P` denote working precision and `C(P)` the cost of bounded-precision
reversible scalar arithmetic (including elementary functions). Let
`F(k,ε) = O(log³(k/ε) log(2/ε))`. Here ε is per-matrix operator error.

| Route | Applicable family | Applying the transform | Parameter generation / total coherent cost | Status |
|---|---|---|---|---|
| Dense Givens baseline | All | k(k−1)/2 abstract plane rotations | Θ(k³) scalar operations before precision overhead | Existing implementation |
| Generator QR | All strictly interlaced contents | k(k−1)/2 abstract plane rotations | O(k²) scalar operations; reversible O(k² C(P)) target with a computation tape, **P not yet bounded uniformly** | Implemented numerical prototype; exact-arithmetic identity |
| Sequential low-rank cores | All tested profiles | O(ks) plane rotations for prescribed rank s | Current constructor O(k²s²+ks³); no fast coherent generator yet | Existing numerical prototype |
| Two-Krawtchouk lift, streamed Dicke implementation | Staircase | O(k) variable qubit rotations, not original-space Givens | k·poly(log k, log(1/ε)) reversible arithmetic and gates; O(k) physical work qubits plus arithmetic | Exact reduction and encoder tested; arithmetic circuit not compiled |
| Two-Krawtchouk lift, binary SU(2) implementation | Staircase | F(k,ε) quantum gates in the cited gate model | Included in F; no k-long angle table. O(log(k/ε)) qubits | Literature-based approximate complexity, exact reduction tested |
| Projectively transported fast family | Signed-permutation orbit described in §8 | Same as chosen base transform, plus poly(log k) gates | Only reversible index comparisons/subtractions and signs, given (k,q) | Exact identity and diagram examples tested; recognition cost separate |

The two gate-count models in this table are intentionally distinct. Binary
quantum gates act on log-size registers and can mix many matrix coordinates
at once. None of these results establishes a subquadratic **exact Givens**
factorization for all A-matrices. Finite Clifford+T synthesis can add further
polylogarithmic overhead to the cited arbitrary-rotation gate model.

Details, proof ideas, numerical checks, and coherence restrictions appear
in §§6–8. The July comparison below is retained as the baseline; its dense
angle-generation counts do not include the new generator QR.

## July baseline

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

Among these July methods, the **sequential quasiseparable method has the
best rotation count**, while **formula-basis tree ULV has the more credible
coherent-depth story**. Neither is end-to-end coherent yet. The October
update adds a stronger staircase-specific quantum circuit and a general
quadratic-arithmetic exact baseline.

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

## 6. General-content generator QR: quadratic angle arithmetic

`experiments/structured_qr_probe.py` specializes the generator QR idea of
[Delvaux–Gemignani–Van Barel, §3](https://www.cs.kuleuven.be/publicaties/rapporten/tw/TW469.pdf)
to the real, displacement-rank-one orthogonal matrix. This adapts established
structured QR; no claim of a new general QR theorem is intended.

Choose η above the largest addable content and use the square projective
Cauchy representation already implemented in §1. Reverse its columns.
At every growing suffix-row/prefix-column subproblem, maintain the two
vectors u=Qᵀg and vᵀ=hᵀR⁻¹. The symmetric matrix Qᵀdiag(x)Q has lower
triangle uᵢvⱼ and diagonal yᵢ+uᵢvᵢ. Thus this dense conjugated diagonal
matrix can be represented by two vectors and its known nodes.

When bordering by one row, initialize u=(g_new,u_old), v=(0,v_old).
At adjacent position j, compute

```text
a = x_new - y[j]
b = u[j+1] v[j] - u[j] v[j+1]
(c,s) = (-b,a)/hypot(a,b).
```

Rotate both generator pairs by `[[c,s],[-s,c]]`. At the end of the row
insertion, update `v[last] += (x_new-y[last])/u[last]`. The kernel
equation `ac+bs=0` is the next QR displacement condition; the last
diagonal determines the final generator. The carried-diagonal invariant
`d-u[j]v[j]=x_new` eliminates d from the actual recurrence, reducing
cancellation. Every bordering step uses n−1 rotations and O(n) scalar work.
Summing gives exactly k(k−1)/2 rotations and O(k²) arithmetic, without
constructing A. The positive column scales need not be evaluated: the
final sign diagonal can be recovered from v alone.

In exact arithmetic `a` is nonzero because the row and column nodes are
disjoint. The final component `u[last]` is nonzero because the bordered
matrix `[C[:,:-1],g]` is Cauchy–Vandermonde and nonsingular. These facts
prevent exact-arithmetic breakdown, but do not bound numerical conditioning.

There is also a useful quantitative bound. Let L=max(1,original content
span), δ be the smallest original interlacing gap, and η=x_max+L.
Telescoping interlaced residue products give αᵢ²≥δ²/L² and therefore
`min |g_i| ≥ δ/(2L²)` and `||g||₂ ≤ 1/L`. At every reversed-order
n-row stage the first n−1 chosen poles interlace the n row nodes. In these
projective coordinates define

\[
t_i=\frac{\prod_{j<n-1}(x_i-y_j)}{\prod_{\ell\ne i}(x_i-x_\ell)}.
\]

Then tᵢ>0 and Σtᵢ=1. The last QR vector is proportional to
zᵢ=tᵢ/gᵢ, so `|u[last]|=1/||z||₂ ≥ min |g_i|`. Orthogonal rotations
preserve the generator norms and each final-v increment is at most
2L/δ, giving `||u||₂≤1/L`, `||v||₂≤2nL/δ`, and
`hypot(a,b)≥δ/(4L²)`. Thus neither generator magnitudes nor division
denominators need exponential range. This is **not yet** an accumulated
rounding-error bound: polynomial sensitivity per update can still compound
over the quadratic number of updates. The guard-bit question remains open.

The classical mutable generator state is O(k), **not** the full reversible
space bound. Storing angles and the intermediate computation history uses
O(k²P) bits, plus elementary-arithmetic workspace/history. Compute all
angles, apply the factor word in its prescribed order, and run parameter
generation backward to erase it; the diagram register is preserved. This
is a coherent architecture, not yet a reversible gate implementation.

At k=512, the prototype emits 130,816 rotations. Maximum-entry residuals
against the independent content formula were approximately 3.4e−13
(staircase), 4.0e−13 (random integer gaps), and 9.6e−12 (real gaps ranging
over exp([-3,3])). These are numerical observations, not error theorems.
The reversed column order matters strongly for stability. Generator
cancellation and small denominators remain: there is no proven bound
`P=polylog(k,content_span,1/ε)` for this implementation. Consequently the
quadratic **real-arithmetic** result should not yet be advertised as a
quadratic worst-case coherent **bit-complexity** theorem.

Reproduce: `uv run python experiments/structured_qr_probe.py --sizes 64 128 512`.
Tests: `tests/test_structured_qr_probe.py`.

## 7. Staircase: an exact clean reduction to two Krawtchouk transforms

Set N=k−1 and M=2N. Let K be the tensor Hadamard H⊗M restricted to
normalized Dicke states, one state for each weight w=0,…,M. This real,
symmetric, orthogonal (2k−1)-dimensional transform is Krawtchouk.
Define the diagonal sign S to be +1 for w≤N and −1 for w>N. Form columns

\[
E_0=|N\rangle,\qquad
E_{j+1}=(-1)^j S K|2j+1\rangle\quad (0\le j<N).
\]

Then, in this repository's content convention,

\[
\boxed{A_{ij}=(-1)^i(KE_j)_{2i}.}
\]

All odd-weight output amplitudes are zero. The implementation and proof
are in `src/fourier/staircase_lift.py`. A useful short proof is to introduce
the spin-N matrix J with off-diagonal
`J[w,w+1]=sqrt((w+1)(M-w))/2`. It obeys
`K J K = diag(N-w)`. Odd columns of K are antisymmetric under w↔M−w;
S makes them symmetric, orthogonal to the central basis state. The
commutator `[J,S]` has support only on the central edge, so the transformed
columns have exactly the Cauchy denominators
`(N−2i)−(N−2j−1)`. Their normalization and signs fix the desired A.

**Clean branch handling.** Encode input column 0 as weight N and every
other column j+1 as weight 2j+1, retaining a flag f=[column≠0]. Apply
the first K controlled on f, then S. At this point the constant branch is
exactly |N⟩, whereas the other branch has no central-weight component.
XORing `[w≠N]` therefore clears f before the final K. Relabel the even
output weight by w/2, retaining its zero parity bit, and apply the row
sign. This is coherent for arbitrary superpositions; no measurement or
postselection is used. Tests simulate the entire flagged isometry.

Two clean K implementations with operator errors δ₁ and δ₂ give total
error at most δ₁+δ₂, including all work registers. The first K is
controlled, so its phase must be fixed, not discarded as an irrelevant
global phase. In the standard spin convention,
`K=exp(-iπ J_y/2) diag((-1)^w)`; this equality has a separate test.

### Two implementations of the K primitive

An elementary implementation encodes a binary weight w into M physical
qubits in the uniform Dicke state, applies M Hadamards, and decodes.
The encoder streams M rotations with angle `2 asin(sqrt(r/ell))`, where
r counts the remaining ones and ell counts the remaining sites. Compute
and uncompute the angle **before** changing r. The counter ends at zero.
Four encoders for two K calls use at most 8(k−1) variable rotations and
O(k) work qubits, with polynomial-logarithmic reversible arithmetic per
rotation. This is already near-linear in both relevant resources, but
does not stay inside the original k-dimensional space.

A stronger implementation follows from
[Iyer–Jain–Jordan–Somma, Theorem 1.1 and §5.1](https://arxiv.org/html/2602.15180v1#S5.SS1):
the spin-N SU(2) rotation has a clean binary-register circuit using
`O(log³(k/ε) log(2/ε))` gates, including arithmetic. This paper states an
operator-norm approximation with ancillas returned close to zero.
[Prakash et al., Theorem 3.12](https://arxiv.org/html/2610.03581v1#S3.SS1.SSS3)
explicitly gives an efficient binary Krawtchouk transform. Combining either
primitive with our reduction gives the bound in the October table.

This consequence relies on those published-preprint constructions; the
repository does **not** implement or independently reproduce their quantum
Hermite-transform subroutine. It does implement the exact matrix identity,
phase convention, flag erasure, and elementary encoder reference.
For coherent varying k, use a common cutoff for k_max and compute the
two-mode occupation embedding from k reversibly; preserve sector-dependent
phases. The resource bound is then F(k_max,ε), not a different circuit size
on each coherent branch. The spin rotation angle needs
O(log(k_max/ε)) bits because the spin generator has norm k−1; the cited
arithmetic bound includes this precision. The uniform-sector justification
uses the truncated-space bound in
Prakash et al., Theorem 2.2, rather than treating a separately compiled
circuit for each k as a free controlled oracle. Family recognition from a
general quantum diagram register is a separate cost. The Hahn theorem in
that paper only covers nonnegative integer parameters and cannot directly
be substituted for our half-integer Hahn parameters.

Numerics at k=128: spectral residual versus the independent content
formula 1.42e−13; orthogonality residual 2.07e−15; odd-output leakage
2.15e−16; central flag leakage zero in the integer-sum reference.
These dense numerical constructions verify the identity, not quantum
runtime. Reproduce: `uv run python experiments/staircase_krawtchouk_lift.py`.

## 8. Projective transport: extending the fast family at no angle cost

For arbitrary interlaced x,y, choose a removable content z=y_q and set

\[
x'_i=\frac1{x_i-z},\qquad
\{y'_j\}=\{0\}\cup\left\{\frac1{y_j-z}:j\ne q\right\}.
\]

Sort both sequences in descending order. Their A-matrix is the original
A-matrix with signed permutations only. In zero-based indices the old row
and column orders are

```text
rows:    q,...,0,  k-1,...,q+1
columns: q+1, q,...,0, k-1,...,q+2.
```

The first q+1 rows have positive sign; the remaining rows negative.
The first column has positive sign, the next q columns negative, and all
remaining columns positive. Thus

\[
A(x',y')=D_r A(x,y)[\text{rows},\text{columns}]D_c.
\]

To prove this, make the old z-column the new constant column and use row
signs to make it positive. Ratios of each remaining column to this column
are precisely Cauchy functions of x'; the column signs make their scales
positive. Orthogonality fixes the normalization. This establishes an exact
matrix identity, not just preservation of displacement rank.

No new angles are needed. These permutations consist of reversed intervals
and one moved index, so a binary circuit uses comparisons and subtraction
from (k,q), not a stored arbitrary permutation table. This adds only
poly(log k) gates even when q is coherent. Construction of the classical
coordinate arrays is O(k); identifying such a structure from arbitrary
input arrays is **not** claimed to be polylogarithmic.

Positive rescaling and translation leave A unchanged. Rational transformed
nodes can therefore be scaled to integers and centered so
`sum(x')=sum(y')`, giving a genuine Young diagram. For example the k=3
staircase transforms into contents x'=[5,−3,−7], y'=[−1,−4], the diagram
`(5,5,5,5,5,5,3)`. This is nonsymmetric and has uneven corner spacings,
yet its A-matrix uses the staircase circuit plus signs and shuffles.
Clearing rational denominators can produce much larger diagram sizes;
polylogarithmic matrix application does not mean polylogarithmic time to
read or construct an expanded diagram description.

Implementation: `src/fourier/projective_transport.py`; tests cover all
removable-pole choices on staircase and random contents through k=32,
plus the genuine diagram example. This provides a restricted fast family,
not a transformation of every A-matrix into a staircase.

## 9. Recommended next tests

The later §§10–12 change the priority: first implement and certify the
balanced-overlap pipeline's reversible arithmetic and clean state
preparation, then compare gates and workspace with generator QR. The
following earlier rotation-focused tests remain useful secondary work.

1. Replace the sequential factorization's dense SVDs with proxy-pole/Cauchy
   bases and independently replay the resulting formula-generated angles.
2. Implement the global skew-quasiseparable Cayley approximation and compare
   its emitted Givens list against sequential cores and core ULV.
3. Compile a small clean two-K staircase circuit and audit its total
   reversible arithmetic and precision; compare the elementary Dicke
   implementation with the binary SU(2)/quantum-Hermite route.
4. At `k=5`, exhaust/prune seven- and eight-rotation support-complete words.
   The lower bound is seven; sparse random search has already found nine
   versus ten dense, but this is not a minimality result.
5. Bound generator growth and required precision in the new QR baseline.
   Then use its fixed small-state recurrences to replace dense local QR
   inside a low-rank method, while retaining a global error guarantee.
6. Separate one-hot abstract rotations from binary-encoded controlled
   two-level gates in every resource table.

## 10. General balanced overlap encoding: why the total can be near-linear

### 10.1 A logarithmic bound from the sign pattern

Write x for descending addable contents, y for removable contents:

\[
x_0>y_0>x_1>\cdots>y_{k-2}>x_{k-1},\qquad
A_{i0}=\alpha_i,\quad A_{i,j+1}=\frac{\alpha_i\beta_j}{x_i-y_j},
\]

where α and β are positive. Set B=|A| entrywise, d=ceil(log₂ k), and
Q=A diag(−1,1,…,1). All strict-upper entries of Q are nonnegative; all
strict-lower entries are nonpositive. Pad to a power of two if necessary.
At each of d binary splitting levels, the signed off-diagonal part is a
direct sum of sibling blocks

\[
 \begin{pmatrix}0&Q_{LR}\\-Q_{RL}&0\end{pmatrix}.
\]

Each block has norm max(‖Q_LR‖,‖Q_RL‖)≤1, since Q is orthogonal. Every
off-diagonal position belongs to exactly one level. The absolute diagonal
has norm at most one. The triangle inequality therefore proves

\[
 \boxed{\|B\|_2\le d+1.}
\]

This bound is uniform over diagrams, not an empirical low-rank assumption.
It also holds for identity-padded matrices. It does not say that every
unweighted row sum is O(log k); row sums can be larger, which is why the
next balancing step matters.

### 10.2 Positive balancing without solving an eigenproblem

Set C=d+1, λ=2C, H=[[0,B],[Bᵀ,0]] and

\[
 v=\sum_{t=0}^{T}(H/\lambda)^t\mathbf1,
 \qquad T=\left\lceil\log_2(2\sqrt{2k})\right\rceil.
\]

All entries of v are at least one, and ‖v‖₂≤2√(2k). Since ‖H/λ‖≤1/2,
the omitted next term has every entry at most 1/2. Thus
Hv/λ=v−1+(H/λ)^(T+1)1≤v. Split v=(l,r). Then

\[
 Br\le\lambda l,\qquad B^T l\le\lambda r.
\]

Only O(log k) products by B and Bᵀ are required. They do **not** require
forming B: the tail is a positive inverse-distance kernel 1/|x−y| with
diagonal weights, plus the constant column.

### 10.3 Overlap, amplification, and clean output

Choose m=ceil(πλ/2) and μ=csc(π/(4m+2)). Then μ≥2λ=O(log k).
Prepare two isometries in a shared larger register:

\[
 P|i\rangle=\sum_j\sqrt{B_{ij}r_j/(\mu l_i)}|i,j,g\rangle
       +\text{row-private padding},
\]
\[
 Q|j\rangle=\sum_i\operatorname{sgn}(A_{ij})
       \sqrt{B_{ij}l_i/(\mu r_j)}|i,j,g\rangle
       +\text{column-private padding}.
\]

The padding probabilities are at least 1/2. The row-padding and
column-padding spaces must be orthogonal. The positive scaling ratios
cancel in each overlap, giving P†Q=A/μ. With R_P=2PP†−I and similarly R_Q,

\[
 (-R_QR_P)^m Q=P A.
\]

Apply inverse P preparation last. This implements A with clean auxiliary
registers in ideal arithmetic, with no measurement or postselection. Keep
the displayed minus sign: omitting a diagram-controlled global phase is
not harmless. The amplification identity is the standard oblivious
amplification mechanism, not a new claim about ordinary Grover search.

Each preparation costs O(k) bounded-precision scalar operations and tree
rotations: retrieve the source-index data with one O(k) scan, then traverse
the other k coefficient registers at fixed positions. Build a binary
probability-sum tree, perform its rotations, and reverse the tree
arithmetic. The source index is preserved, enabling this cleanup even
after the target index is in superposition. There is no O(1) QRAM lookup.
Repeating O(log k) times gives O(k log k) tree rotations; reversible
coefficient arithmetic, binary controls and gate synthesis add logarithmic
factors. These are rotations on an enlarged register, not k-coordinate
Givens rotations. Generating the angle associated with a *coherent*
source index does not mean materializing k different classical angle lists.

The weighted square-root-entry embedding has a clear antecedent in
[Childs, §2, equations (4)–(5)](https://arxiv.org/html/0810.0312v3#S2).
The clean amplification identity is supported by
[Berry et al., Lemma 3.6](https://arxiv.org/html/1312.1414v2#S3).
Here the finite positive Neumann sum supplies adequate scaling vectors
without assuming a Perron eigenvector oracle.

### 10.4 Computing the weights and balancing vectors

Place charges −1 at x nodes and +1 at y nodes. The self-excluded logarithmic
potential Σ_s q_s log|t−s| equals log α_i² at x_i and −log β_j² at y_j.
Use dyadic boxes of widths 1,2,4,… above span L. Interaction lists contain
at most three boxes per target per level, with radius/distance≤1/3.
For n=2k−1 nodes and p terms, log-series truncation is bounded by
3n·3^(−p−1)/(2(p+1)). Moments and evaluations cost O(kp log(2L)).
Integer nearest neighbors have log distance zero, and self terms are
excluded; no singular kernel is evaluated.

The same boxes evaluate Σ q_s/|t−s|. The inverse-distance expansion is a
geometric series; for positive charges its relative truncation error is
at most 3^(−p−1). Adjacent unit-distance contributions are included exactly.
Consequently the O(log k) balancing products are also near-linear. The
implementation uses a conservative absolute truncation budget instead.

The floating reference uses dictionaries for box lookup. A coherent
implementation instead uses fixed, padded records: compute segmented box
moments on sorted nodes, produce at most three queries per target, sort
source/query records by box ID with a fixed sorting network, broadcast
moments within equal-key runs, and reverse the sorting. Keep and eventually
erase comparison bits and scan history. O(k log² k) compare/exchange steps
per level add logarithmic overhead, not a QRAM assumption. Copying a
computed basis-encoded value is coherent copying of a classical function,
not cloning an arbitrary quantum state.

`experiments/reversible_join_probe.py` verifies this lookup schedule on
basis-encoded records: sort, scan, unsort, copy fixed original query slots,
sort, reverse scan, unsort. Its four fixed bitonic passes use O(k log² k)
comparators and restore all records, carry history and swap bits. Payload
width is explicit. This is reversible-logic verification, not a compiled
quantum implementation or independent proof of gate-synthesis phases.

All weights, scaling vectors and preprocessing history are functions of
the unchanged diagram register. Reverse preprocessing after applying A.
The input diagram can therefore remain in superposition throughout. For
varying k or L, use global maxima, validity flags, identity padding and
one common μ/m; never use branch-dependent classical circuit lengths.

### 10.5 Precision and current validation limits

For unit-separated integer contents, α_i≥1/L and 1≤β_j≤L. The first
bound follows by telescoping the interlaced product for α_i²; the second
follows from β_j^(−2)=Σ_i α_i²/(x_i−y_j)². Hence every nonzero matrix
entry is at least 1/L². Together with 1≤l_i,r_j≤2√(2k), this excludes
exponentially tiny nonzero probabilities at fixed input bit length.

Finite stored l,r cancel **exactly** in the overlap. They need only preserve
the padding inequalities. An absolute vector error ≤0.01/√k suffices with
the factor-two slack. A conservative coefficient target is relative error
η=O(min{1/k², ε/[k log(2k)]}); the first term controls balancing error.
Each positive subtree sum then has relative error O(η), and each angle
2 atan2(√right,√left) changes by O(η), even for unbalanced branches.
Telescoping O(k log k) such rotations bounds the full output/garbage error.
Exact zero leaves in padded sectors bypass 0/0. Budget scalar rounding and
rotation synthesis separately. Compare the approximate circuit with an
ideal one using exact A and the **same stored** l,r; do not claim that a
nonorthogonal approximation to A is amplified exactly.

The exact-rational construction in the standalone theorem establishes
P=poly(log k,log L,log(1/ε)) working bits by explicit denominator accounting;
bounded intermediate magnitudes alone would not justify that conclusion.
Including fixed sorting,
elementary-function evaluation and the retained reversible tape gives
k·poly(log k,log L,log(1/ε)) gates and work qubits. This is the derived
resource architecture; explicit fault-tolerant constants and a certified
finite-precision implementation remain to be supplied.

Implemented checks: `experiments/log_weight_multipole.py` tests both
potentials, positive matvecs, and balancing without constructing A;
`experiments/coherent_overlap_encoding.py` independently tests the dense
isometries, coherent amplification, relative phases, and failure norms.
Combined end-to-end tests compare against the independent content formula,
including random integer gaps and gaps spanning many scales. The dense
overlap test is O(k³) storage; it verifies the identity and is **not** the
claimed quantum implementation. Floating numerical agreement does not
certify the finite-arithmetic error bound.

## 11. Rectangular staircases: fractional Fourier encoding

A rectangular staircase replaces each ordinary staircase step by a fixed
height h and width w. After affine normalization its contents are
x_i=k−1−i, y_j=k−1−j−θ, θ=h/(h+w). Its residue weights have closed gamma
ratios. A Fourier transform, index-dependent phases and inverse Fourier
transform produce a sine-denominator kernel approximating 1/(x_i−y_j).

The important refinement is to remove the near-pole diagonal and implement
it exactly as a separate weighted diagonal. The remaining Fourier
multiplier has absolute value <7 uniformly in θ, avoiding a 1/θ overhead.
Combining the pieces gives normalization

\[
 \Gamma=1+\alpha_{\max}
     [7e\sqrt{(k-1)\theta(1-\theta)}+\sqrt{k}]=O(\sqrt{k}).
\]

The corresponding singular-value amplification route uses Õ(√k) total
gates, with coefficient evaluation included. For fixed θ, the sharper
power is k^((1−min(θ,1−θ))/2), up to logarithmic factors. Ordinary
staircases (θ=1/2) already have the stronger polylogarithmic two-Krawtchouk
route; this construction matters for unequal step widths and heights.

Coefficient evaluation uses constant-many gamma ratios at coherent
indices, not a k-entry table. A fixed argument shift plus a truncated
Stirling expansion costs polynomially in precision and log input size.
The positive-real remainder bound is given by
[NIST DLMF §5.11(ii)](https://dlmf.nist.gov/5.11#ii). Robust amplification
must include the sine-kernel and arithmetic errors; the approximate kernel
is not itself orthogonal. General singular-value amplification is an
established tool described by
[Gilyén et al.](https://arxiv.org/abs/1806.01838).

`experiments/hilbert_block_encoding.py` validates the kernel, normalization,
gamma ratios and fixed-loop scalar expansion. It does not compile the
complete arithmetic/amplification circuit. For coherent θ, use a common
worst-case normalization and keep all parameter-dependent phases.

## 12. An explicit two-QFT cotangent family

For 0<ρ<1, set

\[
 x_i=\cot[\pi(i+\rho)/k],\quad
 y_j=\cot(\pi j/k),\quad j=1,\ldots,k-1.
\]

The associated A has an exact factorization
A=D F Λ_ρ F† D† G, with F the normalized positive-exponent Fourier
matrix, D_ii=exp(iπi/k), G=diag(1,−1,…,−1), and
(Λ_ρ)_mm=exp[iπρ(2m+1−k)/k]. Summing the geometric series proves the
identity. Thus for power-of-two k, two ordinary qubit QFTs and simple
phase gates implement A in O(log² k) ideal gates. A coherent b-bit ρ adds
O(b log k) controlled phases; its scalar phase must also be retained.
Since ‖∂A/∂ρ‖≤π, b=O(log(1/ε)) suffices independently of k.

These cotangent contents are generally irrational, not literal Young
contents. Rounding Sx,Sy to integers and translating to
Σx=Σy gives genuine diagrams when interlacing is preserved. A conservative
perturbation bound gives sufficient scale
S=O(k^(5/2)/(min(ρ,1−ρ) ε)). This is an explicitly generated restricted
family, not a method to approximate arbitrary Young diagrams. Certified
generation must also bound numerical cotangent evaluation error.

Unlike the other new routes, `experiments/rectangular_staircase_probe.py`
contains actual clean Qiskit circuits, including a coherent parameter
register. Small circuits are checked sector by sector against the dense
formula. At k=128, fixed ρ, one elementary-gate compilation uses 174 CNOTs
and 242 one-qubit gates on 13 qubits; with b=10 coherent parameter bits it
uses 314 CNOTs and 452 one-qubit gates on 23 qubits. These are ideal
arbitrary-rotation counts, not Clifford+T counts. Family recognition and
conversion from arbitrary diagram encodings are separate costs.

## 13. A subcubic full QFT: compact streaming induction

### 13.1 Result and status

**Theorem draft, not a compiled fault-tolerant circuit or externally
reviewed result.** The standalone
[`subcubic-qft-theorem.tex`](subcubic-qft-theorem.tex) is the current full
proof; this section remains its research-summary version.
Combining the fixed-word version of §10 with the compact streaming wrapper
below gives, for every n and 0<ε≤1, a uniform coherent circuit for the S_n
QFT with

\[
 G(n,\epsilon)=n^{5/2}\,\operatorname{poly}(\log(n+1),\log(1/\epsilon)),
 \qquad W(n,\epsilon)=n\,\operatorname{poly}(\log(n+1),\log(1/\epsilon)).
\]

The error is on the complete output, including work registers: the ideal
output has every temporary register zero. With a constant-factor tightening
of the error budget this also bounds channel diamond distance by ε. This is
an approximate gate-count result, not an exact short Givens word, and does
not claim an improved physical depth or a practical crossover.

The result uses the canonical cycle transversals t_i^(m)=(i,...,m) of
[Bruinsma–Grinko–Ozols, Proposition 5.8 and §§5–6](https://arxiv.org/html/2608.28569v2#S5).
It does not pay for the alternative-transversal correction from their §5.4.
The existing outer relative-path conversion may be retained.
If the input instead supplies a one-line permutation, convert it by fixed
scans: at size m let i=g(m), apply t_i^(-1) to every stored image
(i maps to m, images greater than i decrease by one), then recurse on the
first m−1 images. Reversing these operations reconstructs the permutation.
The conversion costs O(n² polylog n), below the claimed QFT bound.

Evidence levels should not be conflated:

- The exponent follows from the constructive algorithms and error bounds
  in §§10 and 13, using standard reversible simulation of bounded-word
  classical arithmetic and synthesis of controlled rotations.
- `compact_shapes.py` explicitly verifies word-level reversible schedules.
  `certified_multipole.py` certifies numerical enclosures by directed integer
  arithmetic and executes padded reversible data joins.
- `streaming_induction.py` verifies the whole transform against independent
  representation matrices; its dense A calls are verification oracles, not
  an implementation of the fast core.
- No complete Toffoli/Clifford+T circuit, optimized constants, or external
  independent mathematical review is supplied. The theorem and proof here
  remain research results to scrutinize, rather than a claim that numerical
  tests alone establish an asymptotic speedup.

### 13.2 A canonical compact shape register

For a public outer induction size M, let D=floor(sqrt(M)). If λ has d
diagonal boxes, store d and its Frobenius coordinates

\[
 a_i=\lambda_i-i,\qquad b_i=\lambda'_i-i,\qquad 1\le i\le d.
\]

They count boxes right of and below each diagonal box. Since d²≤|λ|≤M,
both arrays fit D slots. Active entries are strictly decreasing nonnegative
integers; inactive slots are zero. Each word has O(log(M+1)) bits.
This is canonical, not a history-dependent cache.

For any row r, a fixed scan gives

\[
 \lambda_r=\begin{cases}a_r+r,&r\le d,\\
 \sum_{i=1}^{d}[b_i+i\ge r],&r>d.
 \end{cases}
\]

Adding a valid box (r,s) increments a_r if s>r, increments b_s if s<r,
or appends (0,0) and increments d if s=r. All indexed changes are implemented
by scanning the D fixed positions and comparing their public index with
the coherent row/column. Removal reverses the change.

To make an in-place update clean, compute f_r(λ)=λ+box_r into a fresh
compact register, reverse the arithmetic scratch, swap old/new registers,
and XOR g_r(f_r(λ))=λ into the old register, where g_r removes row r.
Both directions cost O(D) word operations. The stable row label is retained
throughout. A corner ordinal cannot replace it without an inverse relabeling.

All corner candidates can be read from neighboring arm/leg differences.
Equivalently, with δ_t denoting a unit signed record at content t,

\[
 \nu_\lambda=\delta_0+\sum_{i=1}^{d}
   (\delta_{a_i+1}-\delta_{a_i}+\delta_{-b_i-1}-\delta_{-b_i})
 =\sum_{x\in AC}\delta_x-\sum_{y\in RC}\delta_y.
\]

Sorting/aggregating O(D) records yields all corners. The implemented version
constructs the already-cancelled candidates from neighboring differences
and uses fixed bitonic sort/copy/unsort. Comparator decisions are retained
and erased. Row↔corner conversion scans fixed slots, then uses the inverse
lookup to erase the previous label. These primitives use
O(D polylog M) gates and workspace, with no quantum RAM assumption.

One can also update corner records via
ν_(λ+box_c)−ν_λ=δ_(c−1)+δ_(c+1)−2δ_c. This identity is tested, but the
proved streaming construction does not require a numerically updated weight
cache: weights are recomputed canonically from the current compact shape.
That avoids roundoff-dependent edit history becoming entangled with outputs.

### 13.3 Streaming state and invariant

For U_Ind^(M), the original input is a tableau path p of length M−1 and
a transversal index i. Its original endpoint λ is available as an external
control. Use a word register initially (0,p_1,...,p_(M−1)), a compact prefix
initially empty, a token a=0, and a flag f=i. Zero is the star/inactive symbol,
not a physical row.

At inner level j>1, first add the still-unprocessed original row p_(j−1)
to the compact prefix. If a=0, swap the previous star word slot with this
row and decrement f. At j=1 there is no prefix update, swap or decrement.
Apply the local embedding only if f=1.

On an active non-star input write

\[
 \lambda_j=\kappa+p,\qquad \widetilde\lambda_j=\kappa+a.
\]

Here κ is the original prefix one level earlier, λ_j is the new original
prefix, and the previous induced endpoint is κ+a. The token is the extra
box relative to κ; it is **not generally the last output tableau symbol**.
The invariant implies λ_j=tildeλ_j exactly when p=a.

- If p≠a, the Mackey detour block is the one-dimensional identity. Output
  q=p and carry a. The new induced endpoint is λ_j+a.
- If p=a (including the activation star/star case), erase the duplicate
  token by XOR with p, relabel p from star/removable-row to column index,
  apply A_(λ_j), relabel its output to addable row q, and copy q into a.
  The endpoint is again λ_j+a.

Compute the equality flag before erasing the duplicate. After the block,
the same flag equals [a=q], both on the diagonal and detour sectors, so it
can be erased. The controlling compact shape stays unchanged throughout
the A call and all its preprocessing uncomputation.

Before activation, the word is the original path with a trailing moving
star, and f=i−j+1. At activation, the swap puts star in the current matrix
input, selecting its constant column. After activation, f=1 and all tokens
are positive rows. Thus an invalid i>M exits as the original path followed
by star, unchanged endpoint, and flag i−M+1, exactly the extended induction
map needed inside the recursion.

This is an induction proof of the complete recursion, not an argument that
only its diagonal blocks match. It includes the identity detours and all
relative signs of A. No measurement of the diagram, token or activation
level is made.

### 13.4 Inverse and outer-boundary cleanup

The local inverse first applies A^T when f=1 and a=q, undoing the
copy/erase procedure. A recovered token a=0 identifies the previous inactive
branch: increment f and undo its swap. Then remove the restored original
row from the compact prefix. No previous shape, flag or swap history is
needed. These operations are reversible on the promised subspace; standard
XOR-oracle constructions extend them to full unitary circuits on arbitrary
bitstrings. Invalid encodings need not share the clean-output guarantee.

At the induction boundary the prefix is exactly the original endpoint λ.
The output endpoint is μ=λ+a for valid inputs. Retain a in the new right
path slot before erasing the internal token. Copy the compact prefix,
add token a to the copy, and output μ. Recover and erase a using the exact
checksum

\[
 W(\lambda)=\sum_r r\lambda_r
 =\sum_{i=1}^{d}\left[i(1+a_i+b_i)+\frac{b_i(b_i+1)}2\right],
 \qquad a=W(\mu)-W(\lambda).
\]

This identity is used only under the one-box promise; W is not a hash that
distinguishes arbitrary diagrams. Erase the prefix against the external λ,
and erase f=1. To remove the external old λ in the full QFT, append the
new row a to the right tableau path; recover λ by removing that row from μ,
then XOR it out. This is the usual relative-path interface, not a discarded
shape register. Invalid induction labels retain the specified output flag.

Boundary conversion to/from the paper's full row lists is allowed once per
outer induction. Even the simple O(MD) scan-based conversions cost
O(M^(3/2) polylog M), already within the target. They are never performed
inside the M inner steps. Keeping compact endpoints throughout is possible
but not necessary for the exponent claimed here.

### 13.5 Closing the finite-precision core

The following fills in the precision/access assumptions of §10 sufficiently
for a uniform circuit-existence bound. Let K be a public padded dimension
and L≥1 a public content-span bound. In an outer M induction take
K=2^ceil(log₂(2floor(sqrt M)+2)), L=2M+1; pad A with identity on inactive
indices. All loops, normalizations and bit widths depend on K,L,δ, not on a
measured diagram sector.

**Polynomial conditioning.** For integer strictly interlacing contents,
the products on each side of x_i telescope, giving
α_i²≥1/[(x_0−x_i)(x_i−x_last)] for interior nodes and α_i²≥1/L at
endpoints. Hence α_i≥1/L. Column normalization gives
β_j^(−2)=Σ_i α_i²/(x_i−y_j)²∈[L^(−2),1], so 1≤β_j≤L and every
nonzero |A_ij|≥L^(−2). The k=1 case is A=[1].

**Stable balancing.** Put C=ceil(log₂K)+1, λ=2C, and evaluate the finite
Neumann sum by the Horner recurrence v_0=1, v_(t+1)=1+Hv_t/λ. Use a fixed
T chosen as the least integer with 2^(2T)≥8K. If the approximate positive matrix has entrywise relative
error d≤1/(10000K), then ||Htilde−H||≤dC: this follows from
|Htilde−H|≤dH and monotonicity of the norm of a nonnegative matrix.
If each approximate matvec has component error at most τ≤λ/(10000K),
including subsequent division/addition roundoff in matvec units, contraction
gives

\[
 \|\widehat v-v\|_2\le4\sqrt{2K}(d+\tau/\lambda)<0.002/\sqrt K.
\]

For e=||vhat−v||_2, weighted sums using stored l,r are bounded by
λ+e(λ+1)/(1−e): every row and column of |A| has Euclidean norm one,
so each component of BΔr or B^TΔl is at most e. The distinction from
an infinity-norm-only error bound is important here.
Choose the public integer amplification count m=2λ and
μ=csc(π/(4m+2)); then 2λ≤μ≤4λ+1 and padding remains greater than 0.49.
The exact comparison isometries use exact A and these **same stored l,r**,
so the scaling vectors cancel exactly. The circuit does not need an exact
Perron vector and does not amplify a nonorthogonal approximation as if it
were unitary.

**Fixed-word potential computation: revised proof.** Use exact rational
evaluation of the finite multipole formulas and round only their final
outputs. At each of D=ceil(log₂(L+1)) dyadic levels, normalized source
coordinates have |z|≤1. With b-bit dyadic charges, the jth moment has a
common denominator dividing 2^(b+ℓj), so summing N sources increases its
numerator width by only O(log N). It does not multiply N unrelated
denominators. Six padded queries per target cover at most three active
far boxes; fixed sorting joins and retained reverse-sort history provide
the same coherent access as before. Two additional joins handle distance
one, and self terms are excluded.

Far-field ratios satisfy |r|≤1/3. The logarithmic tail is at most
3NQ·3^(−p−1)/(2(p+1)); the inverse tail is at most NQ·3^(−p−1).
Interactions partition the sources, so no extra factor D is needed here.
Range-reduced logarithms use an R-term rational atanh series, with total
error at most NQ(D+5)3^(−2R). Choose p,R to make each error at most ζ/4,
then round the final potential to dyadic error at most ζ/2.

Each far polynomial has a common denominator involving only dyadic moment
denominators, one distance raised to O(p), p!, and one logarithm-series
denominator of O(R(D+log R)) bits. Each target sums only O(D) interactions.
The standalone theorem gives an explicit polynomial numerator/denominator
envelope for this whole calculation. Thus exact rational arithmetic still
has K·polylog(K,L,1/δ) bit cost. This replaces the previous unsupported
specific claim P=4096B²; no global floating-point stability assertion is
needed for the theorem.

Exponentials use exact rational Taylor polynomials on a small argument
followed by O(log log L) squarings. Their total rational word growth is
polynomial in log L and requested precision. Square roots use integer
square roots; tree angles use a bounded-ratio half-angle arctangent series.
Public π and μ have fixed-series constructions. Weights, every balancing
iterate, and every probability leaf are rounded back to common dyadics
before long sums or further iterations, preventing denominator growth
across K leaves or across iterations. None requires an exact-real oracle.

The existing `certified_multipole.py` is a separate directed-interval
reference implementation, not an implementation of this exact-rational
proof algorithm. `tests/test_theorem_precision_audit.py` checks rational
truncation, interaction partitioning, and common denominator bounds;
`precision_budget.py` supplies explicit downstream probability budgets.

**Preparation and amplification error.** Stored l,r can be bounded in
[1/2,4K]. A nonzero good leaf probability is therefore at least
1/[8K(4λ+1)L²]. Exact inactive leaves remain zero. Relative error η in
positive leaf masses induces O(η) error in every tree angle, even for very
unbalanced siblings: for θ=2atan(sqrt(R/L)), |dθ/dlog(R/L)|≤1/2.
Padding has constant slack. There are O(K) tree rotations per preparation.
Allocate arithmetic and rotation synthesis errors so each preparation
isometry differs by at most ξ=O(δ/log(K+1)).

Only isometry closeness is needed. If ||P−Ptilde||≤ξ then
||PP†−Ptilde Ptilde†||≤2ξ and the reflection error is ≤4ξ. The amplified
operation, including initial Q and final inverse P preparation, has error
at most (8m+2)ξ. This avoids assuming operator closeness of arbitrary
unitary extensions on dirty scratch states. Keep the minus sign in
(-R_QR_P)^m; diagram-controlled phases cannot be dropped.
Each reflection must check the entire preparation input-clean subspace,
including target, tags, angle words and retained arithmetic tape—not just
the good flag. Read-only preprocessing tables are external controls.

Every arithmetic block is totalized on invalid encodings: validity flags
select safe dummy denominators and fixed finite outputs. Quantum gates
remain unitary even on leakage states. The approximation guarantee is
needed only on the ideal promised path, not on every invalid bitstring.
The rational word bounds establish a constructive precision schedule;
the complete compiled reversible implementation remains future work.

**Reversibility and resources.** Fixed-word scalar computations can be
made reversible by retaining each operation's work, copying its answer,
and reversing the computation. This costs a constant number of forward/
backward runs and workspace proportional to the bounded-word running time.
All coefficient/balancing tables and tapes depend on the unchanged compact
shape. Source-dependent preparation arithmetic preserves its source index,
so its scratch can also be erased after the target is in superposition.
After amplification, reverse preprocessing while the shape still agrees.
Sorting, elementary-function evaluation, angle generation, controlled
rotation synthesis and all retained tape together cost
K·poly(log K,log L,log(1/δ)) gates and workspace. This establishes the
clean core interface required by §13.3 without a free preprocessing oracle.
Identity-padded indices use their analytic diagonal entries and scalar
Horner recurrence; zero-padded content slots are never passed to Cauchy
denominators. A quantum angle word controls a fixed sequence of elementary
binary-angle rotations, whose public angles can be compiled offline to a
finite gate set with polylogarithmic overhead, for example by the
[Solovay–Kitaev algorithm](https://arxiv.org/abs/quant-ph/0505030).

### 13.6 Summing the actual work

At outer size M, use one public compact capacity D=floor(sqrt M) for all M
inner stages. This is important: the simulator does not shrink its arrays
to match a branch or each inner level. Each inner stage costs
sqrt(M)·poly(log M,log(1/δ)), including shape updates, comparisons,
row/index relabeling, matrix computation/application and cleanup.
There are M stages. Boundary work costs at most the same M^(3/2) factor.
The old outer T_M operation costs only O(M polylog M). Therefore

\[
 G(n,\epsilon)\le
 \sum_{M=1}^{n} M^{3/2}\operatorname{poly}(\log(n+1),\log(1/\epsilon))
 =n^{5/2}\operatorname{poly}(\log(n+1),\log(1/\epsilon)).
\]

Choose a common local complete-output error δ=ε/[4n(n+1)] for at most
n(n+1)/2 core calls. Telescoping unitaries bounds total operator error by
their sum, including leakage into invalid/work states; the ideal path
always stays in the promised subspaces. The operator-to-channel bound
contributes at most another factor two. Thus precision adds logarithms,
not another power of n. Input/output paths require O(n log n) qubits;
only one stage's sqrt(n)·polylog workspace is live at a time.

As an independent fallback, omit balancing and use unweighted overlap
normalization O(sqrt K). The same certified fast weight preprocessing then
gives K^(3/2)·polylog local work and a full n^(11/4)·polylog QFT. This
already breaks cubic without the balancing lemma; the positive Neumann
construction is the extra step reaching exponent 5/2.

### 13.7 Reproducible validation and limits

- `src/fourier/compact_shapes.py` and its tests exhaust all 2,714 partitions
  with at most 20 boxes and 19,246 valid addition/removal edges. Tests cover
  diagonal growth/shrinkage, fixed capacities, arbitrary XOR outputs,
  oriented-edge inversion, corner ordering and clean sort/relabel scratch.
- `experiments/streaming_induction.py` compares all induction blocks through
  six-box input diagrams with the independent representation-theoretic
  definition. It tests valid/invalid mixtures, inverse reconstruction from
  clean outputs, and forbids expanded-shape conversion inside inner steps.
  The entire 720×720 S_6 QFT agrees with direct Young-orthogonal matrices
  to max entry error 6.94e−17; its orthogonality infinity error is 1.01e−14.
- `experiments/certified_multipole.py` certifies signed log and inverse
  potentials, residue weights, Horner balancing, scalar functions and tree angles with
  outward integer interval arithmetic, analytic tails and padded data joins.
  Exact rational inverse-potential and squared-weight answers are enclosed.
  Each balancing matvec encloses exact H times the current stored dyadic
  vector. A contraction accumulates the certified midpoint-rounding errors,
  yielding an instance certificate for positive padding without floating
  comparisons to a guessed ideal matrix.
  High-precision log/angle comparisons are regression checks in addition
  to, not substitutes for, the interval remainder proof.
- `experiments/precision_budget.py` checks public probability budgets,
  Horner stability, fixed-sector padding and projector-based amplification
  error bounds. Dense calculations are test references, not fast subroutines.

The next implementation milestone is a compiled clean small A circuit using
the bounded-word arithmetic, followed by explicit fault-tolerant resources.
The theorem-development pass added 54 targeted checks across
`test_theorem_precision_audit.py`, `test_theorem_quantum_audit.py`, and
`test_theorem_wrapper_audit.py`. That checkpoint passed 601 tests. These
checks support the proof audit but do not replace mathematical proof or
external review.
The present tests do not determine whether this asymptotic construction is
competitive at experimentally accessible n, nor certify every gate in an
unwritten compiler. The distinction between an algorithmic bound and a
compiled implementation is intentional.

## 14. Pushing below 5/2: shared-work structure and explicit obstructions

The general full-QFT exponent is **still 5/2**. The results below narrow the
next problem; they are not a lower bound of 5/2 for arbitrary QFT circuits.
All circuit statements use ordinary bounded-arity gates, coherent diagram
inputs, clean work registers, and no free quantum RAM.

### 14.1 A one-box change has an exponentially accurate low-rank correction

Let λ′ be obtained from λ by adding the box of content c. Define the
boundary resolvent

\[
 f_\lambda(z)=\frac{\prod_{y\in RC(\lambda)}(z-y)}
                    {\prod_{x\in AC(\lambda)}(z-x)}.
\]

The exact boundary update gives

\[
 \frac{f_{\lambda'}(z)}{f_\lambda(z)}
   =\frac{(z-c)^2}{(z-c)^2-1}.
\]

Consequently, a one-box edit changes **every persistent residue**, not just
the few corner records that appear or disappear. Set
ρ(z)=sqrt(1−(z−c)^(−2)). On persistent nodes,
α′_x=α_x/ρ(x) and β′_y=β_yρ(y). Persistent nodes satisfy |z−c|≥2.

Pad both old and new A matrices by identity to K=max(k,k′). Independently
permute their rows so common addable contents come first, and their columns
so common removable contents come first. Call these aligned orthogonal
matrices A and B. The distinguished constant column, changed contents and
padding coordinates are exceptional. There are at most four exceptional
rows and columns in total. On the common block,

\[
 B-A=D_{1/\rho(x)}\left[A D_{\rho(y)}-D_{\rho(x)}A\right].
\]

**Constructive approximation lemma.** For every integer p≥0 there is an
explicit matrix E_p such that

\[
 \operatorname{rank}E_p\le 2p+4,\qquad
 \|B-A-E_p\|_2\le \eta_p,
 \qquad \eta_p=\frac{4}{3\sqrt3}\,4^{-p}.
\]

To construct it, keep the exceptional rows and columns exactly. On the
common block replace ρ(z) in the bracket by the degree-p binomial series
in (z−c)^(−2), while retaining the exact outer D_(1/ρ(x)). Its scalar tail
is at most 4^(−p)/3 and 1/ρ(x)≤2/√3. The common submatrix of A has norm
at most one, giving the stated dimension-independent operator bound.
For u=x−c and v=y−c, each divided difference

\[
 \frac{v^{-2j}-u^{-2j}}{u-v}
   =\sum_{a=1}^{2j}u^{-a}v^{-(2j+1-a)}
\]

uses row powers drawn from u^(−1),…,u^(−2p). Collecting equal row powers
therefore gives rank at most 2p, not p(p+1). The exceptional rows/columns
contribute at most four additional ranks. The p=0 case is included.

**A genuine orthogonal correction exists.** Put R_p=E_p A^T. Then
T=BA^T is orthogonal and ||T−(I+R_p)||≤η_p<1. Let V_p be the orthogonal
polar factor of I+R_p. Singular values lie within η_p of one, so

\[
 \|V_p-T\|_2\le2\eta_p.
\]

Moreover V_p is exactly the identity outside
S=range(R_p)+range(R_p^T), whose dimension is at most **4p+8**. Thus an
accuracy-δ correction acts nontrivially on only O(log(1/δ)) dimensions,
uniformly over all Young-diagram box additions. This is a structural
decomposition, not merely a numerical observation of small singular values.

**Not a constant-rank exact update.** For a staircase followed by a top-row
box addition, the common-block kernel, up to nonsingular diagonal factors,
is (ρ(y)−ρ(x))/(x−y). Choose r common rows and more than 2r common columns.
A row dependency would imply ρ(y)Q(y)=P(y) at all those columns, with
degrees at most r−1. Squaring and clearing the factor y² (after shifting
c to zero) makes y²P²−(y²−1)Q² a degree-at-most-2r polynomial with more
than 2r roots. It would vanish identically, impossible for nonzero P,Q
because of the odd zero multiplicity at y=1. Hence these r rows are
independent. In particular the exact update rank grows at least as
floor((k−3)/2). Approximation, rather than an exact constant-size edit, is
essential to this route.

**Coherent-cost gap.** Describing the small subspace S is not the same as
preparing its basis coherently. Its vectors are dense and depend on the
diagram. Generic preparation still reads Ω(k) parameters; a numerical SVD
or polar decomposition is not a free controlled circuit. The alignment
permutations also need coherent implementations. Further, successive
streaming A calls act on different path slots and are separated by token
copy/equality operations, so the A^T factors cannot simply be cancelled.
The outstanding task is to exploit this transport while sharing a clean,
reusable representation across those interfaces.

`experiments/prefix_amortization_probe.py` constructs E_p explicitly without
SVD; SVD is used only for numerical verification and the polar-factor
reference. Tests cover every addition edge through nine-box shapes,
increasing/unchanged/decreasing corner counts, and larger staircase and
rectangular-staircase examples. These checks accompany the analytic proof;
they are not the proof of its uniform bound.

### 14.2 A fresh-input local gate-access lower bound

There is an exact fixed-size family showing that the current Õ(k) local
cost is optimal in its power of k for a fresh-input primitive, at sufficiently
small inverse-polynomial error. This does not concern amortized calls.

For m≥1, let k=4m+1 and choose independent bits s_0,…,s_(m−1). Set

\[
 x_i=8m-4i,\quad
 y_j=x_j-a_{j\bmod4}+s_{\lfloor j/4\rfloor}d_{j\bmod4},
 \qquad a=(2,2,1,3),\quad d=(1,-1,-1,1).
\]

All lists strictly interlace, sum x=sum y=0, and describe genuine diagrams
with the **same** N=32m²+3m boxes and span L=16m. Flipping bit s_r from
zero to one changes only its four removable-node words, but multiplies
α_(4r)² by 49/81, and α_(4r) by 7/9. Since α_(4r)≥1/L, the two positive
first-column outputs differ in norm by at least 2/(9L).

Suppose a fixed clean circuit never touches the four-word block for bit r.
After stripping off the unchanged parameter registers, its actual output
on the constant-column input is identical for those two parameter choices.
If each ideal output were approximated within δ<1/(9L), the triangle
inequality would give separation less than 2/(9L), a contradiction.
Every block must therefore be touched. A gate of arity at most q can touch
at most q such blocks, giving at least ceil(m/q)=**Ω(k)** gates.
Fresh preprocessing is included in this count; retaining correlated work
between calls invalidates the argument's fresh-input hypothesis.

The result also applies to compact Frobenius data, not just a redundant
corner-list encoding. Keep only the first floor(m/2) bits variable. The
Durfee side is 4m, all leg words remain fixed, and each bit changes only a
disjoint block of eight arm words. The selected addable corner always lies
in row 8r+1, so the argument survives row-labeled outputs as well.

The exact witness and its proof are in `experiments/sublinear_core_probe.py`.
Tests exhaust m≤6 and check larger families using exact rational residues,
including the same-box-count and disjoint-Frobenius-support conditions.
This is **not** a full-QFT lower bound and cannot be multiplied by n² to
claim that the 5/2 exponent is optimal.

### 14.3 Why straightforward prefix caching does not remove √n

Two separate problems occur.

First, there are valid tableau paths with Θ(√M) corners at Θ(M) prefixes:
build a staircase with d(d+1)≤M, then continue adding to its first row.
Materializing every prefix's corner list therefore writes Ω(M^(3/2))
words on that family. Batched sorting may accelerate row-occurrence
counts, but cannot make this explicit output smaller. A compressed cache
is not excluded by the argument.

Second, the original path is changed coherently by induction. Keeping a
copy of all its prefixes is not harmless workspace. For example, original
paths (1,2,1) and (1,1,2) have different two-box prefixes, yet both have a
nonzero amplitude at the same four-box output word (1,2,1,1) of shape (3,1).
A cleanup controlled only by that output label cannot erase both distinct
cache records unitarily. More general cleanup that mixes output labels is
not ruled out, but must be supplied and counted. The probe records both
amplitudes, whose squared magnitudes are 3/32 and 1/32.

### 14.4 Reordering adjacent-generator transversals cannot solve the problem

For H=S_(M−1) fixing M, a left coset tH is labeled by t(M). Right
multiplication by an adjacent transposition swaps adjacent positions in
the one-line permutation t. View the starting permutation as ordered
tokens. Exposing every token in the last position requires every pair
of initially ordered tokens to cross at least once. Each adjacent swap
crosses exactly one pair. Any continuous right-adjacent walk visiting
all M cosets thus needs at least **M(M−1)/2** swaps, and successive
right-end extraction attains this bound.

This covers arbitrary representative choices, their ordering, and bridge
word cancellations in that particular implementation of Beals' loop.
It does not lower-bound arbitrary circuits implementing a long cycle,
nor a different QFT construction. Cached tableau coordinates already make
individual adjacent Young generators polylogarithmic in the published
construction; rediscovering that cache is not a new speedup. See
[Bruinsma–Grinko–Ozols, §4 and Appendix A.2](https://arxiv.org/html/2608.28569v2#S4).

### 14.5 A simpler independent route matching the 5/2 exponent

The direct induction formula gives another route to **Õ(n^(5/2))**, without
the fast local A decomposition or multipole balancing. This is an internally
audited construction, not a claim of literature priority or a smaller bound.
It combines the induction formula in
[Bruinsma–Grinko–Ozols, §3](https://arxiv.org/html/2608.28569v2#S3) with
[oblivious amplitude amplification](https://arxiv.org/abs/1312.1414).

At level M, fix the external input shape λ. The source labels (i,P), with
i∈[M] and P a standard tableau of λ, and the target labels (μ,Q), with
μ covering λ and Q a standard tableau of μ, both span M·d_λ dimensions.
Let F_λ be the induction unitary. Prepare the branching state

\[
 |P,\star\rangle\longmapsto
 \sum_{\mu\succ\lambda}\sqrt{\frac{d_\mu}{M d_\lambda}}
 |P\to\mu\rangle,
\]

apply R_μ(t_i) controlled on the retained transversal i, then coherently
project that tag onto its uniform state by a uniform-state inverse circuit
and a zero-register projector. The resulting block is **F_λ/√M**. The
projection is a mathematical block description, not an actual measurement.

More explicitly, let I and J embed the source and target in the same
physical register space, with zero scratch and the respective star/erased
register conventions. Set r=ceil(sqrt M), θ=π/(4r+2), and μ=csc θ≥sqrt M.
An additional flag rotation attenuates the zero-flag amplitude by sqrt M/μ.
The resulting unitary W obeys J^†WI=F_λ/μ. With
R_I=2II^†−I and R_J=2JJ^†−I, the exact rotation identity is

\[
 (-W R_I W^\dagger R_J)^r W I=J F_\lambda.
\]

Thus all flags, the old transversal tag and temporary work end clean,
including relative phases. Both projectors test **all** zero work and their
valid source/target promises. They are not merely tests of a success flag.
If ||W̃I−WI||≤η, the complete output error is at most (4r+1)η; only
isometry closeness is required, not closeness of arbitrary unitary
extensions. This follows by bounding the two image projectors by 2η and
telescoping the r reflections.

**Cost including arithmetic.** There are O(sqrt M) successor coefficients.
The direct residue products for all of them take O(M) scalar work, with
polylogarithmic precision overhead. A fixed sorting network builds all
tableau column coordinates in Õ(M) gates. Applying the adjacent generators
s_(M−1),…,s_1 with threshold controls [i≤j] realizes the selected canonical
transversal t_i=(i,…,M) in Õ(M) gates, and preserves those cached coordinates.
Recomputing the cache from the final tableau erases it. Validity predicates
and full reflections use fixed scans/sorts of O(M) words. Specifically,
within-row ranks give column coordinates; sorting by (column,row) checks
consecutive row coordinates and increasing labels in each column. Row and
column group counts, joined at diagonal cells, give the Frobenius endpoint
coordinates. Thus validation and endpoint conversion do not scan all
tableau prefixes or repeatedly query a compact shape. For target membership,
the row-checksum difference identifies a candidate added row, and comparing
the compact endpoint against λ plus that box checks the one-box promise.
The outer endpoint/right-path conversion costs Õ(M) once, as before.

Therefore W, its inverse, and each full reflection cost Õ(M), including
coherent coefficient work. There are O(sqrt M) amplification iterations,
so induction costs Õ(M^(3/2)) and the full sum costs **Õ(n^(5/2))**.
Workspace is Õ(n). Precision η=ε/O(n^(3/2)) suffices after summing the
complete-output errors. No free parameter oracle or postselection is used.

**A normalization obstruction for this particular embedding.** If normalized
induction columns retain tag i and one fixed normalized erasure bra v is
used, the block is F_λ diag(conj(v_i)). Producing a scalar F_λ/μ requires
|v_i|=1/μ for every i, hence μ≥sqrt M. Allowing per-column attenuation does
not help: it only increases the required |v_i|. Reweighting this tag alone
cannot remove the square-root amplification factor. This is not a lower
bound for other block encodings.

`experiments/alternative_qft_probe.py` constructs this route independently
of the Mackey recursion and A-matrix code. Tests compare complete induction
outputs and small full QFTs, including clean ancillas, inverse action,
arbitrary complex reference entanglement, differing unitary completions,
and a negative test for omitting scratch from a reflection. Dense matrices
are validation tools only, not the claimed efficient implementation.
The full S_5 QFT has operator-norm discrepancy 2.48e−15; the six-box
induction block with input shape (3,2) has complete-output discrepancy
3.42e−15. Independent internal audits checked the source/target predicates,
phase convention and resource accounting.

### 14.6 An actual peripheral improvement: near-linear permutation encoding

The previous one-line-permutation interface used O(n² polylog n) scans.
Both directions can instead be performed cleanly in **O(n log⁴(n+1))**
ordinary gates, with O(n log³(n+1)) work bits. This improves the interface,
not the present 5/2 full-QFT exponent.

For the canonical cycles t_i^(m)=(i,…,m), write
g=t_(i_n)^(n)⋯t_(i_1)^(1). The code is

\[
 i_m=1+\#\{j<m:g(j)<g(m)\}.
\]

Compute these prefix ranks using a public divide-by-index tree. At a node,
sort its records by value and scan the number of left-child records seen.
Add that count to each right-child record's rank, then reverse the sort.
Every earlier/smaller pair is counted once, at its lowest separating node.

For the inverse, a node represents a public value interval with s records
in descending original-index order. Let L=floor(s/2) be the number of
remaining lower-half values. For each record with current rank r, assign
it to the lower half and decrement L if r≤L; otherwise assign it to the
upper half and replace r by r−L. This splits the remaining order statistic
exactly. The child sizes are always the public floor/ceiling sizes, even
for odd s. Sort by (child, descending original index) and recurse to
singleton intervals, where the values are known. A final sort restores
original-index order.

All sorts are fixed sorting networks with retained comparator bits.
Decoder branch bits are retained at every tree depth. Copy the answer,
then reverse every operation and erase those bits by recomputing their
original comparisons. Clean XOR oracles for the two bijections give a
clean in-place conversion by forward-oracle, register swap, inverse-oracle.
There is no data-dependent circuit schedule or indexed-memory oracle.
The O(n log³ n) comparators over the public tree each handle O(log n)-bit
records, giving the advertised gate bound including cleanup.

`experiments/transversal_encoding_probe.py` explicitly simulates the
reversible tapes. Tests cover every permutation through n=7, agreement
with direct cycle multiplication, arbitrary XOR outputs, data-independent
operation counts, and random odd/even sizes through n=127. This is a
word-level reversible reference, not a compiled gate circuit.

### 14.7 What would constitute the next exponent improvement?

The new transport theorem suggests a concrete research target: maintain a
coherent, clean description of the O(log(1/δ))-dimensional update subspaces
through a whole induction, and implement their corrections without a new
Θ(k)-word scan at every step. It must also accommodate changing path slots
and clean up the old basis data. That would evade the fresh-input bound;
just replacing each standalone A by another fresh standalone circuit will
not.

A separate sufficient target is a clean balanced merge for
S_a×S_b inside S_(a+b), of cost Õ((a+b)^γ) with γ<5/2. A genuinely balanced
recurrence would then give T(n)=2T(n/2)+Õ(n^γ)=Õ(n^γ) for γ>1.
This is a conditional target, not a constructed merge: Littlewood–Richardson
multiplicity labels, coherent coefficients, and the final basis conversion
must all be implemented. An exponentially long coset loop is not a solution.

Validation at this checkpoint: **766 tests pass**, including 165 added for
the four new probes. Independent internal audits checked the transport
proof, reversible encoding schedule and amplified-induction interface.
These remain research derivations and reference implementations, not
external mathematical review or a compiled fault-tolerant circuit.

## 15. Trying to prove 5/2 optimal: restricted theorems and the remaining gap

The conclusion is **not** an unrestricted Ω(n^(5/2)) QFT lower bound.
We can prove that exponent optimal for two precisely specified modular
architectures. The full proof, including the limits of its applicability,
is in `qft-optimality-audit.tex`.

### 15.1 A stronger local obstruction: constant error, every large size

The hard family in §14.2 can be strengthened substantially. Let m be its
number of four-node blocks, so N_m=32m²+3m and k=4m+1. Comparing its
residues to the midpoint family y_j=x_j−2 gives, uniformly over all bits,

\[
 \alpha_{4r}^2\ge\frac1{8m}\left(\frac7{12}\right)^2\frac{59}{64}
 >\frac1{32m}.
\]

For a block offset t, the ratio to the midpoint residue is either
1−5/[(16t+10)(16t+14)] or 1−5/[(16t+2)(16t+6)]. Two exceptional factors
are at least 7/12; the other deficits have a summable inverse-square tail.
This is a uniform proof, not an extrapolation from small examples.

Flipping bit r changes its selected positive amplitude from h_r to 7h_r/9.
For a uniformly random bit this has conditional variance h_r²/81, at least
1/(2592m). If u disjoint parameter blocks are untouched by the circuit,
its relevant output amplitudes cannot depend on those bits. Averaging
the squared clean-output error over them therefore gives

\[
 \delta^2\ge u/(2592m).
\]

Thus at the **constant** accuracy δ≤1/144 at most floor(m/8) blocks can
remain untouched. The argument projects onto correct parameter labels
and zero workspace, so it does not assume exact parameter preservation
by an approximate circuit.

For every N≥35 choose maximal m with N_m≤N and pad the first row by
p=N−N_m boxes. Freeze bit zero and vary only 1≤r<floor(m/2). The padding
changes only a common first Frobenius arm. At the selected persistent
corner, putting d=16r, it multiplies the residue by

\[
 \frac{d(d+p-1)}{(d-1)(d+p)}\ge1.
\]

The variance bound and 7/9 amplitude ratio survive. There are
floor(m/2)−1 variable disjoint eight-arm blocks. Hence an a-qubit-arity
circuit implementing the fresh Frobenius-controlled local primitive needs
at least

\[
 \left\lceil\frac{\max(0,\lfloor m/2\rfloor-1-\lfloor m/8\rfloor)}a
 \right\rceil=\Omega(\sqrt N)=\Omega(k)
\]

gates, even at constant error. This includes coherent preprocessing and
cleanup, and assumes no other shape-dependent input or free memory oracle.

### 15.2 When this proves exponent-optimality

Suppose each pair of outer size M and prefix size N<M has a separately
implemented local module. Each must work on **every** valid N-box shape,
have fresh shape-independent workspace, and meet the local error contract.
No external tableau/cache can be read, and no gates are shared, fused or
cancelled across modules. Then the costs really can be summed:

\[
 G\ge\sum_{N=35}^{n-1}(n-N)\,G_N^{\rm lower}
 =\frac{n^{5/2}}{40\sqrt2\,a}+O(n^2)=\Omega(n^{5/2}).
\]

Different modules may use different hard witness shapes: full-promise
correctness of each fixed module justifies that. It does **not** show that
one tableau path encounters all hard cases. The independent clean-core
implementation is exponent-optimal within this explicit interface class,
up to logarithmic factors. An optimizing global compiler need not preserve
the class.

### 15.3 The matching amplified-induction route also has a restricted bound

For §14.5's tagged normalized-column embedding, scalar normalization
requires μ≥sqrt M. For a bounded odd amplification polynomial p of degree
at most q with |p(1/μ)−1|≤ε<1, Bernstein's trigonometric inequality gives

\[
 q\ge\frac{1-\epsilon}{\arcsin(1/\mu)}
 \ge\frac{2(1-\epsilon)}\pi\sqrt M.
\]

This follows by applying the derivative inequality to p(sin t), which
vanishes at zero. It is a scalar-polynomial amplification bound, not a
generic search lower bound transplanted to a known Fourier matrix.

There is also a genuine Ω(M) gate obstruction for the **prescribed full
normalized-column action** W in the fixed row-word encoding. For incoming
shape (M−1), the induced representation is the natural permutation
representation. Its Young-basis vectors are the uniform vector and
h_j=((j−1)e_j−Σ_(i<j)e_i)/sqrt(j(j−1)). On tag i=j, output tableau slot
j has row 2 with probability (j−1)/j, while its input has row 1.
Every slot j=2,…,M−1 therefore needs a touched distinguishing bit at
complete-output error <1/sqrt2. This proves Ω(M) gates for W, not merely
an O(M) implementation cost.

If the Ω(sqrt M) calls to W/W† are separately realized without fusion,
each counted module works for every incoming shape (including (M−1)),
and each ideal outer-stage template independently meets its amplification contract,
the costs multiply to Ω(M^(3/2)); summing gives Ω(n^(5/2)). An arbitrary
block-encoding completion, compressed tableau encoding, shared calls,
or direct synthesis of the induction unitary is outside this theorem.

### 15.4 Why this does not settle unrestricted optimality

Clean coherent circuits may share shape-dependent workspace or combine
many mathematical factors into one circuit. As a simple illustration,
T repetitions of a controlled rotation by θ times the Hamming weight of
k bits can be fused into one rotation by Tθ times that weight. Each fresh
call has an Ω(k) access bound at sufficient precision, yet the composition
can compute the weight once and uncompute it once, in Õ(k) total gates.
This is a counterexample to the proposed general multiplication principle,
not a shortcut we have constructed for the QFT.

For arbitrary bounded-arity clean circuits we establish only

\[
 G\ge\big[\log_2(n!)+2\log_2(1-\delta^2/2)\big]/a
 =\Omega(n\log n).
\]

The inverse QFT maps the trivial-irrep basis label to a uniform superposition
of n! group labels. An a-qubit gate grows computational-basis support by
at most 2^a; constant approximation error still requires a constant fraction
of n! support. A separate Schmidt-rank argument gives Ω(n log n) gates
crossing a physical right-tableau cut, even with free single-register
operations. Neither argument reaches exponent 5/2. No claim is made that
this audit proves the strongest existing unrestricted lower bound.

The missing theorem must survive shared preprocessing and global changes
of intermediate basis. Merely strengthening the standalone local bound,
counting coset steps, or citing classical two-coordinate FFT lower bounds
does not supply it. Beating 5/2 remains possible by leaving the restricted
architectures.

New exact/dense reference checks are `constant_error_local_probe`,
`modular_lower_bound_probe`, and `amplification_lower_bound_probe`, with
matching test files. These check identities and resource accounting, not
compiled quantum circuits. The local variance argument and both composition
contracts received independent internal audits; external review is still
needed. The complete suite now passes **976 tests**, including **210 new
lower-bound checks**.

## 16. A standalone simpler 5/2 theorem

The direct-induction route from §14.5 is now developed in
[`simple-qft-theorem.tex`](simple-qft-theorem.tex). The claimed bound is

\[
G(n,\epsilon)=n^{5/2}\operatorname{poly}(\log(n+1),\log(1/\epsilon)),
\qquad
S(n,\epsilon)=n\operatorname{poly}(\log(n+1),\log(1/\epsilon)).
\]

The error is at most ε/2 in clean isometry norm, hence at most ε in channel
diamond distance on the valid input subspace. The construction is uniform
over coherent shape superpositions and charges both coefficient arithmetic
and circuit application. No QRAM, shape-dependent advice, postselection,
or free exact rotations are assumed. This is a **standalone research proof
draft**, not an independently reviewed theorem or a completed gate compiler.

The proof is simpler than §13 because it bypasses local A-matrix synthesis.
At size M, prepare a branching column, apply the selected Young
representation transversal, and erase the transversal tag in the uniform
basis. The good block is F_λ/√M for the complete unitary induction map F_λ.
Public attenuation permits exact ideal amplification in O(√M) rounds.
Every column call and every complete promise reflection costs Õ(M), so
the stage costs Õ(M^(3/2)) and the sum costs Õ(n^(5/2)). The previous
multipole/overlap and compact-shape machinery is not needed for this proof.

| Obligation | What the standalone proof now supplies |
|---|---|
| Coherent branching coefficients | Each probability is a product of k−1 ratios in (0,1); all k products cost O(k² polylog)=Õ(M) |
| Finite precision | Directed multiply/divide rounding gives state error ≤√(2k(k−1)2^−b); b=O(log M+log(1/η)) suffices |
| Variable angles | Fixed integer mass trees, endpoint-safe square root/arctangent arithmetic, and bit-controlled synthesized rotations |
| Tableau action | Clean column cache; explicit sign-canonicalized two-tableau reflection; M−1 public-index generators |
| Complete source/target reflections | Fixed sorting networks and scans check tableau validity, endpoint relation, tag, flag, and **all** column-unitary work |
| Reversible cleanup | Dedicated clean predicate helpers; cache erased from the final tableau; old endpoint recovered from the new shape and appended right-path row |
| Approximate amplification | Isometry error η becomes at most (4r+1)η for r rounds, independent of the arbitrary unitary completion |
| Global error and space | η=ε/[2n(4⌈√n⌉+1)]; reused near-linear workspace; an explicit Fourier-coefficient recurrence |

The new bounded-factor identity, in descending content order, is

\[
p_i=\prod_{j<i}\frac{y_j-x_i}{x_j-x_i}
     \prod_{j\ge i}\frac{x_i-y_j}{x_i-x_{j+1}}.
\]

For each factor a/d, update the fixed-point integer
N←⌊Na/d⌋ starting from N=2^b. The intermediate never exceeds 2^b,
and each update adds less than 2^−b absolute error. This avoids a
conditioning assumption or huge exact products. The angle verifier uses
exact rational enclosures; the theorem's bounded-word implementation uses
fixed-point Horner evaluation with guard bits, not growing rational
denominators.

Reference checks are in `experiments/direct_induction_precision.py` and
`experiments/direct_induction_bookkeeping.py`, with matching tests.
`tests/test_direct_induction_contract.py` independently checks the cached
generator, product order, promises, amplification errors, and outer cleanup.
The bookkeeping simulator retains and erases sorting decisions on malformed
inputs too, and records a data-independent near-linear operation schedule.
These are word-level and finite algebraic checks, not a full quantum compiler.
At this checkpoint the complete suite passed **1193 tests**, including **217 new checks**:
111 precision checks, 77 bookkeeping checks, and 29 quantum-contract checks.
The standalone TeX source compiles successfully. Separate internal audits
reviewed the written arithmetic, bookkeeping, and quantum-interface proofs.

The next subsection supplies the small compiled induction milestone.
External proof review, publication priority, and unrestricted optimality
are not established here.

## 17. Complete small induction circuit: coherent arithmetic through Clifford+T

`experiments/direct_induction_gate_probe.py` now compiles a complete
three-box induction step, including the **two possible input shapes in
superposition**, and checks all six valid source columns against an
independent representation-theoretic matrix. The comparison includes
relative phases and every work register; it is not a postselected comparison
of only the successful output block. Checking the full matrix controls every
superposition of these inputs, not just their individual basis-state errors.

The emitted circuit calculates corners with fixed sorting/compaction
networks, branching probabilities with bounded-product integer arithmetic,
and angle words with fixed-point square root/arctangent routines. It then
applies bit-controlled rotations, cached Young generators, tag erasure,
attenuation, and full-unitary amplitude amplification. Both promise
reflections test the complete work-register condition with dedicated helpers.
Shape-dependent quantities are computed and uncomputed **inside the circuit**;
there is no shape-indexed angle table or free arithmetic operation.

The finite gate program is a lazy iterator over actual Clifford+T gates,
not an estimated synthesis cost. Toffolis have an explicit exact seven-T
decomposition. Variable rotations use a public angle dictionary synthesized
with Qiskit's Solovay–Kitaev implementation. A controlled rotation uses a
phase-cancelling pair of one-qubit words and CNOTs; its negative-angle partner
is the exact adjoint of the whole controlled word. This avoids turning
irrelevant one-qubit global phases into relative phases. A single shared
auxiliary wire handles uncontrolled rotations and returns exactly to zero.

At nine fractional weight bits and nine fractional angle bits:

| Measured quantity | Complete M=3 induction |
|---|---:|
| Valid source columns checked | 6, across both shapes |
| Clifford+T elementary gates | 17,138,782 |
| T and T† gates | 7,778,300 |
| Logical qubits, excluding error correction | 76,009 |
| Clean full-output operator error | 0.0050583193 |
| Norm outside the ideal target subspace | 0.0050565597 |
| Sum of local synthesis-error bounds | 0.0005277950 |
| Accumulated simulation-pruning operator bound | 1.51 × 10⁻¹² |

The native circuit, before finite-gate synthesis, contains 2,038,394
instructions (including two global-sign annotations), uses 76,008 wires,
and has clean error 0.0050616357. The tiny decrease in measured error after
synthesis is incidental cancellation, not a synthesis advantage. Each
public rotation meets the requested numerical tolerance 10⁻⁵; four angles
required a deeper synthesis recursion than the first attempt allowed.
The final program's two global-sign annotations are not elementary gates.

The simulator accelerates repeated classical reversible blocks by executing
their actual gates once per encountered complete basis state, then caching
that result. Finite rotation matrices are multiplied from their actual
synthesized words. Separate tests compare both accelerations with literal
gate-by-gate execution, including phases, dirty work bits, and adjoints.
This is a simulation optimization, not a circuit oracle or a reduction in
the counted gates. Pruning errors are accumulated without renormalization.
The finite-word numerical isometry defect is 2.56 × 10⁻¹⁰, reflecting
floating-point matrix multiplication; finite gate words themselves are
unitary. Roundoff is **not interval-certified** by the reported pruning bound.

The saved complete report is
`data/direct_induction_gate_M3_b9_c9.json`. Reproduce it with:

```sh
.venv/bin/python experiments/direct_induction_gate_probe.py --level 3 --fraction-bits 9 --angle-bits 9
```

The recorded environment is Python 3.13.5, NumPy 2.4.2, and Qiskit 2.3.0.
Two consecutive complete runs produced byte-identical JSON reports. The
full suite passes **1346 tests**, including **153 new gate-level checks**:
45 reversible-IR, 15 finite-synthesis, 9 arithmetic-angle, 33 bookkeeping,
38 block-simulation, and 13 full-induction integration tests. The updated
standalone theorem source also compiles successfully.
The implementation tests cover arithmetic, fixed-network bookkeeping,
allocation and helper reuse, complete source-column comparisons, and
phase-safe finite synthesis. An allocator ownership leak discovered during
integration was repaired before recording these resource figures.

**Scope:** this is one complete induction stage, not the assembled S_n QFT,
an asymptotic scaling fit, a useful concrete speedup, or independent proof
review. The very large constants come from deliberately unoptimized
retained-history arithmetic and generic finite synthesis. The analytic
Õ(n^(5/2)) theorem remains a separate argument. Next, replace retained
arithmetic histories with reusable scratch/in-place arithmetic, assemble a
whole small QFT with the outer encoding/cleanup recurrence, and obtain
external review of the shorter proof.

## Primary references

- [Bruinsma–Grinko–Ozols, the symmetric-group QFT and induction recurrence](https://arxiv.org/abs/2608.28569)
- [Dawson–Nielsen, constructive finite-gate synthesis](https://arxiv.org/abs/quant-ph/0505030)

- [Childs, weighted quantum-walk embeddings](https://arxiv.org/abs/0810.0312)
- [Berry et al., oblivious amplitude amplification](https://arxiv.org/abs/1312.1414)
- [Gilyén et al., quantum singular-value transformation and amplification](https://arxiv.org/abs/1806.01838)
- [Queffélec–Zarouf, Bernstein's inequality for polynomials, Theorem 1.1](https://arxiv.org/abs/1903.10801)
- [Ailon, Fourier lower bounds in a two-coordinate linear circuit model](https://arxiv.org/abs/1305.4745)
- [Cleve–Watrous, cyclic-QFT circuits and depth lower bounds](https://arxiv.org/abs/quant-ph/0006004)
- [NIST DLMF, gamma asymptotics and error bounds](https://dlmf.nist.gov/5.11)

- [Iyer–Jain–Jordan–Somma, efficient high-dimensional SU(n) circuits](https://arxiv.org/abs/2602.15180)
- [Prakash et al., quantum orthogonal-polynomial transforms](https://arxiv.org/abs/2610.03581)
- [Delvaux–Gemignani–Van Barel, fast QR of Cauchy-like matrices](https://www.cs.kuleuven.be/publicaties/rapporten/tw/TW469.pdf)
- [Fasino, “Orthogonal Cauchy-like matrices”](https://link.springer.com/article/10.1007/s11075-022-01391-y)
- [Driscoll–Healy–Rockmore, “Fast Discrete Polynomial Transforms”](https://digitalcommons.dartmouth.edu/facoa/2066/)
- [Delvaux–Van Barel, “A Givens-Weight Representation for Rank Structured Matrices”](https://doi.org/10.1137/060654967)
- [Delvaux–Van Barel, “Unitary Rank Structured Matrices”](https://doi.org/10.1016/j.cam.2007.03.020)
- [Delvaux–Frederix–Van Barel, hierarchical to unitary-weight conversion](https://etna.ricam.oeaw.ac.at/vol.33.2008-2009/pp163-188.dir/pp163-188.pdf)
- [Stewart, stability of orthogonal transformations of rank-structured matrices](https://epubs.siam.org/doi/10.1137/15M1041900)
- [Beckermann–Townsend, displacement structure and singular-value bounds](https://epubs.siam.org/doi/10.1137/19M1244433)
