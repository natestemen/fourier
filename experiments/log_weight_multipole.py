"""Near-linear approximate residue weights via a signed logarithmic potential.

For integer, strictly interlaced contents x,y, place charge -1 at each x
and +1 at each y. Excluding self, the potential sum q*log|t-source| is
log(alpha**2) at an x node and -log(beta**2) at a y node.

This reference uses uniform dyadic boxes and normalized multipole moments.
For each target box, interact with the children of its parent's neighbors,
excluding itself and its immediate neighbors. At most three source boxes
occur per level, and radius/distance <= 1/3. Unit leaf boxes contain at
most one integer node; remaining adjacent-node log distances are log(1)=0.

For n=2k-1 points and p retained terms, an analytic potential-error bound is
    3*n*3**(-(p+1))/(2*(p+1)).
This only bounds series truncation: ordinary floating-point error is not
certified here. Moments and all evaluations use O(n*p*log(2L)) arithmetic,
with O(n*p*log(2L)) retained reference data, L the content span.

The Python dictionary is CLASSICAL, not a free quantum-memory oracle.
A proposed coherent implementation replaces it by batched sorted joins:
at each level sorted point IDs permit segmented moment sums, then <=3n
target requests and <=n source records are sorted by box ID, values are
broadcast, and the sort is reversed. Fixed sorting networks add polylog(n)
overhead without QRAM. These reversible networks are NOT compiled here.
The program verifies numerical potentials and operation counts, not that
full coherent gate bound.

The inverse-distance variant also supplies B=abs(A) matvecs and a positive
Neumann scaling without forming B. Together with coherent_overlap_encoding,
this gives the proposed k*polylog(k,L,1/epsilon) architecture described in
NEW_METHODS.md, section 10. reversible_join_probe checks a fixed sorted-join
compute-copy-uncompute schedule separately; it is not yet integrated as a
quantum circuit here. All content spans in this float reference are <=2**50.
"""

import argparse
from dataclasses import dataclass
from math import ceil, log

import numpy as np


@dataclass
class MultipoleWeights:
    alpha: np.ndarray
    beta: np.ndarray
    log_alpha_squared: np.ndarray
    log_beta_squared: np.ndarray
    order: int
    levels: int
    moment_terms: int
    evaluation_terms: int
    occupied_boxes: int
    log_error_bound: float


@dataclass
class Potential:
    values: np.ndarray
    order: int
    levels: int
    moment_terms: int
    evaluation_terms: int
    occupied_boxes: int
    series_error_bound: float


def multipole_potential(positions, charges, *, order, kernel="log"):
    """Evaluate a log or inverse-distance potential at sorted integer nodes.

    Self terms are omitted. Charges may be signed. For ``inverse``, each
    far interaction uses sum_m M_m*(radius/D)**m / abs(D). The omitted
    geometric tail is at most ||charges||_1*3**(-order-1): |D|>=3/2
    for separated unit boxes. Adjacent unit-distance nodes are added exactly.
    These are truncation bounds, not floating-point roundoff certificates.
    """
    raw_positions = list(positions)
    if not raw_positions or any(not np.isfinite(x) or int(x) != x for x in raw_positions):
        raise ValueError("positions must be nonempty finite integers")
    raw_positions = [int(x) for x in raw_positions]
    if any(a >= b for a, b in zip(raw_positions, raw_positions[1:])):
        raise ValueError("positions must be strictly ascending")
    span = raw_positions[-1] - raw_positions[0]
    if span > 2**50:
        raise ValueError("floating reference requires span <=2**50")
    positions = np.array([x-raw_positions[0] for x in raw_positions], dtype=np.int64)
    charges = np.asarray(charges, dtype=float)
    n = len(positions)
    if charges.shape != (n,) or not np.all(np.isfinite(charges)):
        raise ValueError("expected one finite charge per position")
    if not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError("order must be a positive integer")
    if kernel not in ("log", "inverse"):
        raise ValueError("kernel must be log or inverse")
    depth = int(span).bit_length()
    potentials = np.zeros(n)
    term_numbers = np.arange(1, order + 1)
    moment_terms = evaluation_terms = occupied_boxes = 0
    for level in range(depth):
        width = 1 << level
        ids = positions // width
        boundaries = np.r_[0, 1 + np.flatnonzero(np.diff(ids)), n]
        moments = {}
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            box = int(ids[start])
            normalized = (positions[start:stop] % width) * (2.0 / width) - 1.0
            powers = np.ones(stop - start)
            vector = np.empty(order + 1)
            for m in range(order + 1):
                vector[m] = np.dot(charges[start:stop], powers)
                powers *= normalized
            moments[box] = vector
            moment_terms += (stop - start) * (order + 1)
        occupied_boxes += len(moments)
        for i, (position, target) in enumerate(zip(positions, ids)):
            parent = int(target) // 2
            for parent_offset in (-1, 0, 1):
                for child in (0, 1):
                    source = 2 * (parent + parent_offset) + child
                    if abs(source - int(target)) <= 1 or source not in moments:
                        continue
                    vector = moments[source]
                    d = 2.0 * (int(position) - source * width) / width - 1.0
                    ratio = 1.0 / d
                    if abs(ratio) > 1 / 3 + 1e-15:
                        raise ArithmeticError("invalid interaction separation")
                    if kernel == "log":
                        potentials[i] += (
                            vector[0] * (log(abs(d)) + level * log(2) - log(2))
                            - np.dot(vector[1:], ratio ** term_numbers / term_numbers)
                        )
                    else:
                        potentials[i] += (2.0 / width / abs(d)) * (
                            vector[0] + np.dot(vector[1:], ratio ** term_numbers))
                    evaluation_terms += order + 1
    if kernel == "inverse":
        neighbors = np.flatnonzero(np.diff(positions) == 1)
        potentials[neighbors] += charges[neighbors + 1]
        potentials[neighbors + 1] += charges[neighbors]
    mass = float(np.abs(charges).sum())
    bound = (truncation_bound(mass, order) if kernel == "log" else
             mass * 3.0 ** (-(order + 1)))
    return Potential(potentials, int(order), depth, moment_terms,
                     evaluation_terms, occupied_boxes, bound)


def truncation_bound(n_points, order):
    """Uniform absolute log-potential error from the omitted series tail."""
    return 1.5 * n_points * 3.0 ** (-(order + 1)) / (order + 1)


def multipole_weights(ac, rc, *, log_tolerance=1e-10, order=None):
    """Approximate all alpha,beta without forming a dense Cauchy matrix.

    The floating reference restricts integer coordinates to a span <=2**50.
    Absolute offsets are removed before floating arithmetic, so large common
    integer translations do not hurt accuracy. For larger spans the proposed
    method requires a multiprecision implementation, which is not supplied.
    """
    ac, rc = list(ac), list(rc)
    k = len(ac)
    if k == 0 or len(rc) != k - 1:
        raise ValueError("expected k>=1 addable and k-1 removable contents")
    merged = [item for pair in zip(ac, rc) for item in pair] + [ac[-1]]
    if any(not np.isfinite(t) or int(t) != t for t in merged):
        raise ValueError("contents must be finite integers")
    merged = [int(t) for t in merged]
    if any(a <= b for a, b in zip(merged, merged[1:])):
        raise ValueError("contents must strictly interlace")
    if not np.isfinite(log_tolerance) or log_tolerance <= 0:
        raise ValueError("log_tolerance must be positive and finite")
    span = merged[0] - merged[-1]
    if span > 2**50:
        raise ValueError("floating reference requires content span <=2**50")
    n = len(merged)
    if order is None:
        order = max(1, ceil(log(max(1.0, 1.5 * n / log_tolerance), 3)))
    if not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError("order must be a positive integer")
    # Ascending positions retain their original alternating signs.
    positions = np.array([t - merged[-1] for t in merged[::-1]], dtype=np.int64)
    charges = np.where(np.arange(n) % 2 == 0, -1.0, 1.0)
    result = multipole_potential(positions, charges, order=order)
    potentials = result.values[::-1]
    log_alpha = potentials[::2]
    log_beta = -potentials[1::2]
    return MultipoleWeights(
        np.exp(0.5 * log_alpha), np.exp(0.5 * log_beta),
        log_alpha, log_beta, int(order), result.levels, result.moment_terms,
        result.evaluation_terms, result.occupied_boxes, result.series_error_bound,
    )


def absolute_matvec(ac, rc, alpha, beta, vector, *, transpose=False,
                    tolerance=1e-10):
    """Apply B=|A| or B.T using inverse-distance potentials, never dense B.

    Tolerance bounds series truncation in each output entry, assuming exact
    input weights/vector. Finite arithmetic and input error are additional.
    Positive charges are the intended balancing use, but signed inputs work.
    """
    ac, rc = list(ac), list(rc)
    k = len(ac)
    alpha, beta, vector = map(lambda x: np.asarray(x, dtype=float),
                              (alpha, beta, vector))
    if (k == 0 or len(rc) != k-1 or alpha.shape != (k,)
            or beta.shape != (k-1,) or vector.shape != (k,)):
        raise ValueError("inconsistent contents, weights, or vector")
    if (not all(np.all(np.isfinite(v)) for v in (alpha, beta, vector))
            or np.any(alpha <= 0) or np.any(beta <= 0)
            or not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError("expected positive finite weights and tolerance")
    merged = [item for pair in zip(ac, rc) for item in pair] + [ac[-1]]
    charges = np.zeros(2*k-1)
    if transpose:
        charges[::2] = alpha * vector
        output_scale = max(1.0, float(np.max(beta, initial=0)))
    else:
        charges[1::2] = beta * vector[1:]
        output_scale = max(1.0, float(np.max(alpha)))
    mass = float(np.abs(charges).sum())
    order = max(1, ceil(log(max(1.0, mass * output_scale / tolerance), 3)))
    potential = multipole_potential(merged[::-1], charges[::-1],
                                    order=order, kernel="inverse")
    values = potential.values[::-1]
    if transpose:
        result = np.r_[np.dot(alpha, vector), beta * values[1::2]]
    else:
        result = alpha * (vector[0] + values[::2])
    return result, potential


@dataclass
class MultipoleScaling:
    left: np.ndarray
    right: np.ndarray
    weights: MultipoleWeights
    absolute_norm_bound: float
    neumann_steps: int
    amplification_steps: int
    normalization: float
    moment_terms: int
    evaluation_terms: int


def multipole_positive_scaling(ac, rc, *, tolerance=1e-10):
    """Compute the logarithmic-normalization positive scalings without dense A.

    See coherent_overlap_encoding.a_matrix_positive_scaling for the proof.
    Every Neumann step consists of two fast inverse-distance matvecs. Both
    the weight and matvec truncation tolerances are specified here; they do
    not certify floating roundoff. The dense result is an independent check.
    A circuit must use fixed padded sizes, orders and precision, derived from
    global k,L bounds rather than classical branching on quantum data.
    """
    ac, rc = list(ac), list(rc)
    weights = multipole_weights(ac, rc, log_tolerance=tolerance)
    k = len(ac)
    bound = float((k-1).bit_length() + 1)
    lam = 2 * bound
    terms = int(np.ceil(np.log2(2 * np.sqrt(2 * k))))
    u, w = np.ones(k), np.ones(k)
    left, right = u.copy(), w.copy()
    moment_terms, evaluation_terms = weights.moment_terms, weights.evaluation_terms
    for _ in range(terms):
        new_u, stats_u = absolute_matvec(ac, rc, weights.alpha, weights.beta, w,
                                        tolerance=tolerance)
        new_w, stats_w = absolute_matvec(ac, rc, weights.alpha, weights.beta, u,
                                        transpose=True, tolerance=tolerance)
        u, w = new_u / lam, new_w / lam
        left += u
        right += w
        moment_terms += stats_u.moment_terms + stats_w.moment_terms
        evaluation_terms += stats_u.evaluation_terms + stats_w.evaluation_terms
    steps = int(np.ceil(np.pi * lam / 2))
    normalization = float(1 / np.sin(np.pi / (4 * steps + 2)))
    return MultipoleScaling(left, right, weights, bound, terms, steps,
                            normalization, moment_terms, evaluation_terms)


def direct_log_weights(ac, rc):
    """Independent quadratic reference, not used by the multipole routine."""
    ac, rc = np.asarray(ac, dtype=float), np.asarray(rc, dtype=float)
    log_alpha = np.array([
        np.log(np.abs(x - rc)).sum()
        - np.log(np.abs(x - np.delete(ac, i))).sum()
        for i, x in enumerate(ac)
    ])
    log_beta = np.array([
        np.log(np.abs(y - ac)).sum()
        - np.log(np.abs(y - np.delete(rc, j))).sum()
        for j, y in enumerate(rc)
    ])
    return log_alpha, log_beta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[16, 64, 256, 1024])
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()
    rng = np.random.default_rng(20261023)
    print("family,k,levels,order,log_error,series_bound,moment_terms,evaluation_terms")
    for family in ("staircase", "random", "wide"):
        for k in args.sizes:
            gaps = (np.ones(2*k-2, dtype=int) if family == "staircase" else
                    rng.integers(1, 20, size=2*k-2) if family == "random" else
                    2 ** rng.integers(0, 16, size=2*k-2))
            merged = np.r_[0, np.cumsum(gaps)][::-1]
            ac, rc = merged[::2], merged[1::2]
            result = multipole_weights(ac, rc, log_tolerance=args.tolerance)
            la, lb = direct_log_weights(ac, rc)
            error = max(np.max(np.abs(result.log_alpha_squared-la)),
                        np.max(np.abs(result.log_beta_squared-lb), initial=0))
            print(f"{family},{k},{result.levels},{result.order},{error:.6g},"
                  f"{result.log_error_bound:.6g},{result.moment_terms},"
                  f"{result.evaluation_terms}")


if __name__ == "__main__":
    main()
