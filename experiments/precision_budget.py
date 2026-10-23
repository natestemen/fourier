"""Explicit sufficient error budgets for the balanced A-matrix architecture.

This is a mathematical budget calculator and dense stability verifier, NOT a
quantum circuit or a certificate for the floating multipole implementation.
The budgets are public functions of padded dimension K, integer content span
L, and requested local error epsilon. They must not be chosen after measuring
or inspecting a coherent diagram sector.

The useful stability change is Horner evaluation of the positive Neumann sum:
v_0 = 1; v_{t+1} = 1 + H v_t / lambda. The exact result is the same finite
sum as in coherent_overlap_encoding.py, but roundoff obeys a contraction.

Let C = ceil(log2 K)+1, lambda=2C. If B=abs(A), then ||B||<=C. If every
nonzero entry of Btilde differs relatively by at most d, entrywise positivity
gives ||Btilde-B||<=d C; this does not cost an extra factor K. With d<=1/2,
||Htilde/lambda||<=3/4. If each matvec component has additive error at most t,
Horner's vector error is bounded, independently of its number of steps, by
4 sqrt(2K) (d+t/lambda). Here t includes division/addition roundoff after
rescaling back into the matvec's units.

Stored balancing vectors cancel EXACTLY in row/column overlaps. They only
have to preserve positive padding. We compare an approximate circuit with
the ideal circuit using exact A and those same stored balancing vectors.
We do not claim exact amplification of a nonorthogonal approximation to A.
"""

from dataclasses import dataclass
from fractions import Fraction
from math import log, pi, sin, sqrt

import numpy as np


def _ceil_log2_ratio(value: Fraction) -> int:
    """Smallest nonnegative b such that 2**b >= value, without float logs."""
    if value <= 1:
        return 0
    b = max(0, value.numerator.bit_length() - value.denominator.bit_length())
    if (value.denominator << b) < value.numerator:
        b += 1
    return b


@dataclass(frozen=True)
class PrecisionBudget:
    padded_dimension: int
    content_span: int
    epsilon: Fraction
    absolute_norm_bound: int
    neumann_lambda: int
    neumann_steps: int
    amplification_steps: int
    normalization_upper_bound: int
    tree_leaves: int
    rotation_applications: int
    balancing_entry_relative_error: Fraction
    balancing_matvec_absolute_error: Fraction
    leaf_relative_error: Fraction
    individual_rotation_operator_error: Fraction
    log_weight_absolute_error: Fraction
    nonzero_leaf_probability_lower_bound: Fraction
    probability_arithmetic_absolute_error: Fraction
    probability_fraction_bits: int

    @property
    def normalization(self) -> float:
        """Display/reference value only; a circuit needs certified constants."""
        return 1 / sin(pi / (4 * self.amplification_steps + 2))

    @property
    def scaling_vector_error_bound(self) -> float:
        return 4 * sqrt(2 * self.padded_dimension) * (
            float(self.balancing_entry_relative_error)
            + float(self.balancing_matvec_absolute_error) / self.neumann_lambda)

    @property
    def padded_probability_lower_bound(self) -> float:
        """Padding guarantee for the exact comparison circuit, rounded down."""
        error = self.scaling_vector_error_bound
        lam = self.neumann_lambda
        # Each row of |A| (and of H) has Euclidean norm exactly one,
        # since A is orthogonal, including identity padding. Consequently
        # ||H delta_v||_infinity <= ||delta_v||_2 = error. The +1 here
        # is intentional: using the spectral norm bound C would be valid
        # but unnecessarily weaker.
        weighted_sum_bound = lam + error * (lam + 1) / (1 - error)
        return 1 - weighted_sum_bound / (2 * lam)

    @property
    def circuit_error_bound(self) -> Fraction:
        # Relative leaf errors eta change a binary-tree angle by <=4 eta.
        # R_y operator distance is <= half its angle error. The other term
        # covers angle computation AND synthesis, not either one separately.
        return self.rotation_applications * (
            2 * self.leaf_relative_error
            + self.individual_rotation_operator_error)


def precision_budget(padded_dimension: int, content_span: int,
                     epsilon=Fraction(1, 1000)) -> PrecisionBudget:
    """Return conservative public budgets with no dependence on actual data.

    ``probability_fraction_bits`` only budgets final probability arithmetic;
    it is NOT a blanket certificate for every FMM operation. A certified
    potential implementation must additionally propagate intervals to meet
    the log-weight / matvec tolerances in this record.
    """
    if (not isinstance(padded_dimension, int) or padded_dimension < 1
            or not isinstance(content_span, int) or content_span < 1):
        raise ValueError("dimension and content span must be positive integers")
    epsilon = Fraction(epsilon)
    if not 0 < epsilon <= 1:
        raise ValueError("epsilon must lie in (0, 1]")
    k, length = padded_dimension, content_span
    c = (k - 1).bit_length() + 1
    lam = 2 * c
    # 2**(2T) >= 8K ensures the omitted next Neumann term is <=1/2.
    terms = ((8 * k - 1).bit_length() + 1) // 2
    # Integer choice avoids a floating ceil(pi*lambda/2). It obeys
    # 2lambda <= csc(pi/(4m+2)) <= 4lambda+1.
    m = 2 * lam
    mu_upper = 4 * lam + 1
    leaves = 1 << k.bit_length()  # power of two >= K+1
    # Actual preparation/inverse-preparation calls number 4m+2. Double
    # this for a conservative projector-level perturbation argument:
    # ||PP†-Ptilde Ptilde†|| <= 2||P-Ptilde||, so a reflection costs
    # <=4||P-Ptilde||. This only needs an isometry error guarantee on
    # clean work inputs, not an unjustified full-unitary guarantee on
    # every invalid or nonzero-scratch basis state.
    rotations = (8 * m + 4) * (leaves - 1)
    delta_b = Fraction(1, 10_000 * k)
    tau = Fraction(lam, 10_000 * k)
    eta = epsilon / (16 * rotations)
    rotation_error = epsilon / (8 * rotations)
    log_error = min(eta / 64, delta_b / 16)
    # l,r in [1/2,4K], mu<=mu_upper, and every nonzero |Aij|>=L^-2.
    p_min = Fraction(1, 8 * k * mu_upper * length * length)
    arithmetic_error = eta * p_min / (128 * leaves)
    bits = _ceil_log2_ratio(1 / arithmetic_error)
    return PrecisionBudget(k, length, epsilon, c, lam, terms, m, mu_upper,
                           leaves, rotations, delta_b, tau, eta,
                           rotation_error, log_error, p_min,
                           arithmetic_error, bits)


def horner_scaling(matrix, budget: PrecisionBudget, *, approximate_matrix=None,
                   component_errors=None):
    """Dense stress-test reference for the stable recurrence, not fast code.

    ``component_errors`` has shape (T, 2K), bounded by the budget's tau.
    These errors are inserted before division by lambda. The reference
    leaves verification of the supplied matrix approximation to the caller.
    """
    matrix = np.asarray(matrix, dtype=float)
    k = budget.padded_dimension
    if matrix.shape != (k, k):
        raise ValueError("matrix shape must equal the public padded dimension")
    b = np.abs(matrix if approximate_matrix is None else approximate_matrix)
    if b.shape != (k, k):
        raise ValueError("approximate matrix has the wrong shape")
    errors = (np.zeros((budget.neumann_steps, 2 * k)) if component_errors is None
              else np.asarray(component_errors, dtype=float))
    if errors.shape != (budget.neumann_steps, 2 * k):
        raise ValueError("expected one error vector per Horner step")
    left, right = np.ones(k), np.ones(k)
    for error in errors:
        left, right = (1 + (b @ right + error[:k]) / budget.neumann_lambda,
                       1 + (b.T @ left + error[k:]) / budget.neumann_lambda)
    return left, right


def relative_mass_angle_bound(eta: float) -> float:
    """Uniform binary-tree angle bound from relative sibling-mass error.

    For theta=2 atan sqrt(R/L), dtheta/dlog(R/L)<=1/2. Thus perturbing
    positive L and R by at most eta relatively changes theta by at most
    0.5*log((1+eta)/(1-eta)). Exact zero masses must stay exactly zero.
    The implementation uses a more conservative 4*eta in its budget.
    """
    if not 0 <= eta < 1:
        raise ValueError("relative error must be in [0,1)")
    return 0.5 * log((1 + eta) / (1 - eta))
