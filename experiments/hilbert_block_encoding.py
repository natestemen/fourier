"""Verification oracle for a fractional-Fourier block encoding of A.

This is NOT a quantum-circuit implementation. It tests the elementary
identities behind a coherent algorithm for rectangular staircases, whose
unit-spaced contents are x_i=N-i and y_j=N-j-theta, 0<theta<1.

Let delta be theta reduced to [-1/2,1/2], and s=theta-delta in {0,1}.
Embed tail column j at integer j+s. The Cauchy denominator is then
j+s-i+delta. A fractional Fourier shift, with diagonal phase gauges, has
kernel sin(pi*delta)/(M*sin(pi*(j-i+delta)/M)). Scaling gives a csc kernel
which approximates 1/(j-i+delta). Its near-singular diagonal is removed in
the Fourier domain and restored exactly as a separate weighted diagonal.
The remaining Fourier multiplier has absolute value <7 uniformly for
0<abs(delta)<=1/2. Thus small theta needs no 1/theta amplification cost.

Adding the constant column by |uniform_k><0| gives an LCU normalization

    Gamma = 1 + 7*a_max*b_bound + sqrt(k)*a_max,

where a_max is obtained from the two endpoint weights and
b_bound=e*sqrt(N*theta*(1-theta)). Gamma=O(sqrt(k)) uniformly, and for
fixed theta it is O(k**((1-min(theta,1-theta))/2)). A QSVT / oblivious
amplification construction uses O(Gamma*log(1/epsilon)) block queries.
Only constant-many gamma-ratio coefficients are evaluated in superposition
per query. Standard recurrence plus a shifted Stirling expansion computes
log-gamma to P bits in poly(P,log(k),log(1/min(theta,1-theta))) bit work.
For diagram parameters theta=h/(h+w), these are all logarithmic in input
size and precision. There is no QRAM or k-long precomputed angle table.

For |j+s-i+delta|<=L<=M/2, the entry kernel error is <=3*L/M**2.
Consequently ||A_tilde-A||<=3*L*||beta||_2/M**2. The approximation is not
orthogonal; singular-vector transformation / robust amplification is
needed, not postselection alone. Allocate kernel and arithmetic errors
below the final tolerance with amplification robustness included.

The dense arrays here check the identities, bounds, and normalization;
they are never included in the claimed circuit cost. The generic integer
content cotangent route is also included to expose its worse normalization.
"""

from dataclasses import dataclass
import argparse

import numpy as np
from scipy.special import gammaln
import mpmath as mp

from fourier.amatrix import _a_matrix_from_contents_log
from fourier.structured import _transition_weights, _validate_contents


def log_gamma_stirling_probe(z, bits: int):
    """Fixed-loop scalar arithmetic prototype and analytic truncation bound.

    Shift by R=bits+8, use R Stirling correction terms, then subtract R
    elementary logarithms. For positive z, DLMF 5.11(ii) bounds the
    truncation remainder by the first omitted term. The returned bound
    excludes finite-arithmetic rounding (the mpmath context uses guard
    bits). This exhibits the fixed, polynomial-in-precision loop schedule;
    it is not a reversible gate implementation. Bernoulli numbers depend
    only on the circuit precision and can be prepared independently.
    """
    if not isinstance(bits, int) or bits < 2:
        raise ValueError("bits must be an integer at least two")
    with mp.workprec(4 * bits + 256):
        z = mp.mpf(z)
        if z <= 0:
            raise ValueError("positive gamma argument required")
        count = bits + 8
        shifted = z + count
        answer = ((shifted - mp.mpf("0.5")) * mp.log(shifted) - shifted
                  + mp.log(2 * mp.pi) / 2)
        for j in range(1, count + 1):
            answer += mp.bernoulli(2 * j) / (2 * j * (2 * j - 1) * shifted**(2 * j - 1))
        for j in range(count):
            answer -= mp.log(z + j)
        order = 2 * count + 2
        bound = abs(mp.bernoulli(order)) / (order * (order - 1) * shifted**(order - 1))
        return +answer, +bound


def rectangular_scales(k: int, theta: float):
    """Closed alpha and beta residues; no dense A is constructed."""
    if not isinstance(k, (int, np.integer)) or k < 2:
        raise ValueError("k must be an integer at least two")
    if not np.isfinite(theta) or not 0 < theta < 1:
        raise ValueError("theta must lie strictly between zero and one")
    n = k - 1
    i = np.arange(k)
    log_alpha2 = (gammaln(i + 1 - theta) - gammaln(1 - theta)
                  + gammaln(n - i + theta) - gammaln(theta)
                  - gammaln(i + 1) - gammaln(n - i + 1))
    j = np.arange(n)
    log_beta2 = (gammaln(j + 1 + theta) - gammaln(theta)
                 + gammaln(n - j + 1 - theta) - gammaln(1 - theta)
                 - gammaln(j + 1) - gammaln(n - j))
    return np.exp(log_alpha2 / 2), np.exp(log_beta2 / 2)


def fractional_shift_matrix(m: int, delta: float):
    """Dense oracle for a gauged, real fractional Fourier shift."""
    if m < 2 or not 0 < abs(delta) <= 0.5:
        raise ValueError("expected m>=2 and 0<abs(delta)<=1/2")
    index = np.arange(m)
    f = np.exp(-2j * np.pi * index[:, None] * index[None, :] / m) / np.sqrt(m)
    phases = np.exp(1j * delta * np.pi * (2 * index - (m - 1)) / m)
    gauge = np.exp(-1j * np.pi * index / m)
    return gauge[:, None] * ((f * phases) @ f.conj().T) * gauge.conj()[None, :]


def offdiagonal_multipliers(m: int, delta: float):
    """Fourier multiplier of the csc core after removing its diagonal.

    expm1 and sinc avoid cancellation in the theta-small limit. The scalar
    residual still suffers machine cancellation for abs(delta)<<1e-7;
    its analytic limit is zero. Tests avoid using double precision as a
    certificate for that extreme. A reversible implementation needs P-bit
    arithmetic with guard bits, not these float64 routines.
    """
    if m < 2 or not 0 < abs(delta) <= 0.5:
        raise ValueError("expected m>=2 and 0<abs(delta)<=1/2")
    t = np.pi * (2 * np.arange(m) - (m - 1)) / m
    scale = np.pi / np.sin(np.pi * delta)
    correction = (1 / np.sinc(delta) - 1 / np.sinc(delta / m)) / delta
    return scale * np.expm1(1j * delta * t) + correction


@dataclass(frozen=True)
class RectangularBlockEncodingProbe:
    exact: np.ndarray
    approximate: np.ndarray
    normalization_bound: float
    kernel_error_bound: float
    alpha: np.ndarray
    beta: np.ndarray
    period: int
    delta: float
    column_shift: int


def rectangular_probe(k: int, theta: float, period: int):
    """Construct dense checks of the proposed three-term block encoding."""
    alpha, beta = rectangular_scales(k, theta)
    if not isinstance(period, (int, np.integer)) or period < 2 * k:
        raise ValueError("period must be an integer at least 2*k")
    n = k - 1
    shift = int(theta > 0.5)
    delta = theta - shift
    r = np.arange(n)[None, :] + shift - np.arange(k)[:, None]
    differences = r + delta
    core = np.pi / (period * np.sin(np.pi * differences / period))
    # The close-to-pole diagonal is implemented directly, with norm <=1.
    core[r == 0] = 1 / delta
    approximate = np.column_stack((alpha, alpha[:, None] * core * beta[None, :]))
    exact = _a_matrix_from_contents_log(n - np.arange(k, dtype=float),
                                       n - theta - np.arange(n, dtype=float))
    a_max = max(alpha[0], alpha[-1])
    b_bound = np.e * np.sqrt(n * theta * (1 - theta))
    normalization = 1 + a_max * (7 * b_bound + np.sqrt(k))
    error_bound = 3 * np.max(np.abs(differences)) * np.linalg.norm(beta) / period**2
    return RectangularBlockEncodingProbe(exact, approximate, normalization,
                                        error_bound, alpha, beta, period,
                                        delta, shift)


def integer_cotangent_probe(ac, rc, period: int):
    """Generic integer-contents cot kernel; its normalization can be large.

    h_M(r)=(pi/M)cot(pi*r/M) has Fourier multipliers
    -i*pi*(1-2*m/M) for m!=0, and zero at m=0, hence norm <pi.
    Its diagonal scaling gives normalization
    max(alpha)*(pi*max(beta)+sqrt(k)), not polylog universally.
    """
    ac, rc = _validate_contents(ac, rc)
    if not (np.all(ac == np.rint(ac)) and np.all(rc == np.rint(rc))):
        raise ValueError("integer cotangent embedding requires integer contents")
    span = ac[0] - ac[-1]
    if period < 2 * span:
        raise ValueError("period must be at least twice the content span")
    alpha = np.sqrt(_transition_weights(ac, rc))
    beta = 1 / np.linalg.norm(alpha[:, None] / (ac[:, None] - rc[None, :]), axis=0)
    differences = ac[:, None] - rc[None, :]
    core = (np.pi / period) / np.tan(np.pi * differences / period)
    approximation = np.column_stack((alpha, alpha[:, None] * core * beta[None, :]))
    normalization = alpha.max() * (np.pi * beta.max() + np.sqrt(len(ac)))
    # Conservative cot error: |1/r-(pi/M)cot(pi*r/M)| <= 4|r|/M^2.
    bound = 4 * span * np.linalg.norm(beta) / period**2
    return approximation, normalization, bound


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 32, 128])
    parser.add_argument("--theta", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.9])
    args = parser.parse_args()
    print("k theta period spectral_error error_bound normalization")
    for k in args.sizes:
        for theta in args.theta:
            period = 1 << int(np.ceil(np.log2(128 * k)))
            result = rectangular_probe(k, theta, period)
            error = np.linalg.norm(result.approximate - result.exact, ord=2)
            print(k, theta, period, f"{error:.6g}",
                  f"{result.kernel_error_bound:.6g}", f"{result.normalization_bound:.6g}")


if __name__ == "__main__":
    main()
