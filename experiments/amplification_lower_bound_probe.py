r"""Audits for a restricted direct-induction amplification lower bound.

This module does NOT prove an unrestricted S_n QFT gate lower bound.  The
analytic statements below concern the tagged-column / scalar singular-value
amplification architecture and a separately compiled, bounded-locality W.
Tests check finite instances and accounting; they are not the proof.

1. Scalar-query obstruction
---------------------------
Normalized tag erasure gives a scalar block F/mu only if mu >= sqrt(M).
Consider a scalar QSVT amplifier whose successful block is p(1/mu) F,
where p is odd, has degree <= q, and |p(x)| <= 1 for every x in [-1,1].
For target error eps < 1, |p(1/mu)-1| <= eps.  The trigonometric polynomial
T(t)=p(sin(t)) has degree <= q, T(0)=0, and ||T||_infty <= 1.  Bernstein's
inequality gives ||T'||_infty <= q, hence

    q >= (1-eps)/asin(1/mu) >= 2(1-eps) mu/pi = Omega(sqrt(M)).

This permits arbitrary scalar phases chosen using the known mu.  It does
not cover a circuit bypassing the block encoding using explicit F data.
The relevant Bernstein statement is Theorem 1.1 in Queffelec--Zarouf,
https://arxiv.org/html/1903.10801.  The scalar QSVT framework is from
Gilyen--Su--Low--Wiebe, https://arxiv.org/abs/1806.01838.

2. Physical support obstruction for the prescribed W
---------------------------------------------------
Require the same coherent W_M module to work on every valid incoming shape;
we may therefore test its action on lambda=(M-1), for M>=2.  This full-shape
promise applies separately to every counted forward or inverse module.
Identify the induced representation with C^M.  Its
Young basis is u=(1,...,1)/sqrt(M) and, for 2 <= j <= M,

    h_j = ((j-1)e_j - sum_{a<j} e_a)/sqrt(j(j-1)).

The branching state u/sqrt(M) + sqrt((M-1)/M) h_M equals e_M.  The
canonical transversal t_i sends e_M to e_i, so the tableau part of
W I|i> is the i-th column of the matrix whose rows are u and h_j.  The
attenuation flag and tag unpreparation act on other registers and do not
alter its tableau marginal.  The h_j coordinate is the tableau whose
only second-row label is j.  On input i=j its probability is (j-1)/j.

Every source tableau slot j=2,...,M-1 initially stores row 1.  Choose
one bit distinguishing encodings of row 1 and row 2 in each slot.  If
no gate touches that bit, the output probability of the row-2 event
is zero.  Its vector distance from the required W output is at least
sqrt((j-1)/j) >= 1/sqrt(2).  Thus a circuit approximating W on all source
inputs with isometry error eta < 1/sqrt(2) must touch M-2 distinct bits.
With gates acting on at most b qubits it needs at least ceil((M-2)/b)
gates.  Output row slots have a fixed physical encoding; unbounded-arity
gates, compressed encodings and free data-dependent rewiring are excluded.

3. Where the bounds may be multiplied
------------------------------------
Only in a modular architecture with non-overlapping, separately implemented
W/W^dagger calls may the two costs be multiplied.  Each forward call must
implement the prescribed W on the full promised source; an inverse call
is the inverse circuit, or an approximation of the full W^dagger.  No
fusion, cross-call cancellation or global resynthesis is included in this
model.  Here q counts calls in the ideal scalar-polynomial template, whose
own successful block obeys the stated error contract; approximate compiled
calls need not give an exactly polynomial block.  At each outer level M,
the preceding bounds give Omega(M^(3/2))
gates and their sum is Omega(n^(5/2)) for constant bounded error.  Together
with the existing upper bound, this is exponent-optimality of this
architecture up to logarithmic factors, NOT of the symmetric-group QFT.
"""

from __future__ import annotations

from math import asin, ceil, isfinite, sqrt

import numpy as np


def one_row_induction(m: int) -> np.ndarray:
    """Closed form: rows are the trivial component followed by h_2,...,h_M."""
    if not isinstance(m, int) or m < 1:
        raise ValueError("M must be a positive integer")
    result = np.zeros((m, m))
    result[0, :] = 1 / sqrt(m)
    for j in range(2, m + 1):
        result[j - 1, :j - 1] = -1 / sqrt(j * (j - 1))
        result[j - 1, j - 1] = sqrt((j - 1) / j)
    return result


def scalar_query_lower_bound(normalization: float, error: float = 0.1) -> float:
    """Necessary real-valued q bound for the bounded-odd-polynomial model.

    This does not test whether a proposed polynomial is QSVT-realizable.
    A necessary query count is the ceiling of the returned value.
    """
    if not isfinite(normalization) or normalization < 1:
        raise ValueError("normalization must be finite and at least one")
    if not isfinite(error) or not 0 <= error < 1:
        raise ValueError("error must belong to [0,1)")
    return (1 - error) / asin(1 / normalization)


def compulsory_row_slots(m: int, isometry_error: float = 0.1) -> tuple[int, ...]:
    """Slots certified to require a touched distinguishing bit in W.

    The selected input tag can differ for each slot, since the same circuit
    must work on every promised input.  Equality is deliberately excluded:
    the projection-distance argument only contradicts a strictly smaller
    error.  For error < 1/sqrt(2), these are all slots 2,...,M-1.
    """
    if not isinstance(m, int) or m < 1:
        raise ValueError("M must be a positive integer")
    if not isfinite(isometry_error) or not 0 <= isometry_error < 1:
        raise ValueError("isometry error must belong to [0,1)")
    return tuple(j for j in range(2, m) if isometry_error < sqrt((j - 1) / j))


def physical_w_gate_lower_bound(m: int, locality: int = 2,
                               isometry_error: float = 0.1) -> int:
    """Support-only gate bound for a separately implemented physical W."""
    if not isinstance(locality, int) or locality < 1:
        raise ValueError("gate locality must be a positive integer")
    return ceil(len(compulsory_row_slots(m, isometry_error)) / locality)


def modular_gate_lower_bound(n: int, *, locality: int = 2,
                            target_error: float = 0.1,
                            isometry_error: float = 0.1) -> float:
    """Lower-bound sum ONLY under the modular contract in the module docstring.

    No integer rounding of query counts is needed for this conservative
    bound.  mu=sqrt(M) is the most favorable allowed normalization.
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    return sum(scalar_query_lower_bound(sqrt(m), target_error)
               * physical_w_gate_lower_bound(m, locality, isometry_error)
               for m in range(1, n + 1))
