"""An exact symmetric-qubit lift of the staircase A-matrix.

For a k-by-k staircase matrix let N=k-1 and M=2N.  Write K for
Hadamard**tensor(M) restricted to the Dicke basis (one state per Hamming
weight), and S[w,w]=+1 for w<=N, -1 otherwise.  The following columns form
an isometry into the reversal-even subspace::

    E[:, 0]   = |N>
    E[:, j+1] = (-1)**j S K |2j+1>,  0 <= j < N.

The exact identity, in amatrix.py's descending-content convention, is::

    A[i, j] = (-1)**i (K E)[2i, j].

All odd-weight output amplitudes vanish.  This is a circuit identity, not
a subquadratic algorithm for constructing a dense matrix: the routines
below deliberately construct small reference matrices for verification.
The reduction uses only two clean, coherent K primitives, one controlled;
any faster implementation of K immediately replaces the elementary Dicke
construction described below.

Proof sketch
------------
Let J have off-diagonal J[w,w+1]=sqrt((w+1)(M-w))/2.  Then
K J K=diag(N-w).  Reversal R obeys R K=K diag((-1)**w).  Hence K's odd
columns are reversal-odd, vanish at w=N, and S makes them reversal-even.
The resulting E columns, together with |N>, are orthonormal.  Applying K
therefore maps E onto the even-weight subspace.

Only the central edge of J fails to commute with S.  If psi=K|2j+1>,
then (J-(N-2j-1)) S psi = c_j |N>, where
c_j=2 J[N-1,N] psi[N-1].  The sign of c_j is (-1)**j, since the odd
coefficient of (1-z)**(N-1)(1+z)**(N+1) is
2*(-1)**j*binomial(N-1,j).  After the column sign, the nonconstant columns
thus have the Cauchy form alpha[i]*positive_constant/(N-2i-(N-2j-1)).
The constant column is positive after the row signs and satisfies
alpha[i]**2=binomial(2i,i)*binomial(2N-2i,N-i)/4**N.  These are exactly
the staircase transition weights.  Normalization fixes the remaining
positive constants, proving equality with the repository A-matrix.

Clean coherent realization
--------------------------
Start with the input label and a flag f=[label!=0].  Encode the label as
N if f=0, and 2*(label-1)+1 if f=1, retaining f.  This is injective even
when N is odd.  Apply the column sign and K controlled on f, then S.
Now f=[weight!=N] on the whole supported state: the constant branch is
|N>, while the other branch has no central-weight amplitude.  XOR this
predicate to erase f.  Apply K, divide the necessarily even output
weight by two (retain the zero parity bit), and apply the alternating row
sign.  No measurement or postselection is used, including for
superpositions containing column 0.  The input encoding is reversible
because label=0 for f=0, and label=(weight+1)/2 for f=1; it therefore
requires no persistent copy of the original label.

For approximate primitives all ancillas remain part of the output error
analysis.  Two clean operator-norm K approximations of errors delta_1 and
delta_2 give total isometry error at most delta_1+delta_2: the flag and
parity arithmetic are exact unitary operations.  At finite precision the
flag/parity bits need only be close to zero, not identically zero.  A
controlled implementation must preserve the specified phase of K; a
primitive defined only up to an unknown global phase is insufficient.

Implement K as U_D**dagger H**tensor(M) U_D.  The clean Dicke encoder U_D
starts with a counter r=weight and M zero qubits.  For remaining length
ell=M,...,1, prepare the next bit with amplitudes sqrt(1-r/ell),sqrt(r/ell),
then subtract that bit from r.  Every completed branch has r=0; every
weight-w string has amplitude 1/sqrt(binomial(M,w)).  Compute and erase
the angle before modifying r.  Inverting this circuit cleans the M-qubit
workspace after each K.  At most four encoders, or 4M variable rotations,
are needed for the two K calls (plus controlled Hadamards and arithmetic).

The ideal identity is exact, but finite gate synthesis is approximate.
To obtain operator error epsilon, each rotation can have angle error
O(epsilon/M); b=O(log(M/epsilon)) angle bits suffice.  A straightforward
reversible bounded-precision evaluation of 2*asin(sqrt(r/ell)), with exact
endpoint branches and extra O(log M) guard bits, costs poly(log M,b) bit
operations per rotation.  This gives M*poly(log M,log(1/epsilon)) gates
and M+poly(log M,log(1/epsilon)) work qubits under standard reversible
arithmetic and rotation-synthesis assumptions.  No compiled arithmetic
circuit or fault-tolerant resource count is supplied here.  In particular,
this is not an O(k) two-level-Givens factorization on the original space.
"""

from dataclasses import dataclass
from math import comb, sqrt

import numpy as np
import numpy.typing as npt

__all__ = ["KrawtchoukLift", "dicke_rotation", "staircase_lift", "symmetric_hadamard"]

FloatArray = npt.NDArray[np.float64]


def symmetric_hadamard(n_qubits: int) -> FloatArray:
    """Return H**tensor(n_qubits) on normalized Dicke states.

    Entries are computed using exact integer Krawtchouk sums before the
    final float conversion, avoiding cancellation in floating-point sums.
    The returned matrix is indexed by weight in increasing order.  Dense
    construction takes cubic arithmetic work; it is a validation oracle,
    not the proposed quantum implementation.
    """
    if not isinstance(n_qubits, (int, np.integer)) or n_qubits < 0:
        raise ValueError("n_qubits must be a nonnegative integer")
    m = int(n_qubits)
    binomial = [comb(m, w) for w in range(m + 1)]
    scale = 2.0 ** (-m / 2)
    transform = np.empty((m + 1, m + 1))
    for w in range(m + 1):
        for t in range(w, m + 1):
            polynomial = sum(
                (-1) ** j * comb(w, j) * comb(m - w, t - j)
                for j in range(max(0, t - m + w), min(w, t) + 1)
            )
            value = scale * sqrt(binomial[w] / binomial[t]) * polynomial
            transform[w, t] = transform[t, w] = value
    return transform


@dataclass(frozen=True)
class KrawtchoukLift:
    """Reference matrices for the clean staircase circuit described above."""

    k: int
    hadamard: FloatArray

    @property
    def n_qubits(self) -> int:
        return 2 * (self.k - 1)

    @property
    def fold_signs(self) -> FloatArray:
        return np.where(np.arange(self.n_qubits + 1) <= self.k - 1, 1.0, -1.0)

    def prepared_columns(self) -> FloatArray:
        """E after the first controlled K, fold, and coherent flag erasure."""
        prepared = np.zeros((self.n_qubits + 1, self.k))
        prepared[self.k - 1, 0] = 1.0
        prepared[:, 1:] = (
            self.fold_signs[:, None]
            * self.hadamard[:, 1::2]
            * (-1.0) ** np.arange(self.k - 1)
        )
        return prepared

    def lifted_matrix(self) -> FloatArray:
        """Return the full Dicke-weight output before compression and signs."""
        return self.hadamard @ self.prepared_columns()

    def matrix(self) -> FloatArray:
        """Return the staircase A-matrix in the repository's convention."""
        return (-1.0) ** np.arange(self.k)[:, None] * self.lifted_matrix()[::2]


def staircase_lift(k: int) -> KrawtchoukLift:
    """Build an exact-identity reference for a k-by-k staircase A-matrix."""
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer")
    return KrawtchoukLift(int(k), symmetric_hadamard(2 * (int(k) - 1)))


def dicke_rotation(remaining_ones: int, remaining_sites: int) -> tuple[float, float]:
    """Cosine/sine amplitudes for one step of the clean Dicke encoder.

    The rotation angle in the conventional R_y(theta) gate is twice the
    arctangent of sine/cosine.  Its arithmetic must be undone before the
    remaining-ones counter is decremented by the emitted bit.
    """
    if (
        not isinstance(remaining_sites, (int, np.integer))
        or not isinstance(remaining_ones, (int, np.integer))
        or remaining_sites < 1
        or not 0 <= remaining_ones <= remaining_sites
    ):
        raise ValueError("expected integers 0 <= remaining_ones <= remaining_sites")
    return (
        sqrt((remaining_sites - remaining_ones) / remaining_sites),
        sqrt(remaining_ones / remaining_sites),
    )
