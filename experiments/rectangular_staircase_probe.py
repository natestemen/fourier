"""Fast-family probes: rectangular staircases and a Fourier/cotangent family.

The rectangular staircase has equally spaced addable nodes and a fractional
offset rho in (0,1) for removable nodes. Its transition weights are Hahn
weights with (alpha,beta)=(-rho,rho-1), outside the nonnegative-integer
domain of the currently cited quantum Hahn theorem. This file's Hahn route
does not give a new fast circuit for those rectangular staircases; see the
separate hilbert_block_encoding experiment for a fractional-Fourier route.

A different, exact fast family is obtained from angularly uniform nodes::

    x_i = cot(pi*(i+rho)/k),  y_j = cot(pi*j/k),  j=1,...,k-1.

Write D=diag(exp(i*pi*i/k)), F[i,m]=exp(2*pi*i*i*m/k)/sqrt(k),
L[m,m]=exp(i*pi*rho*(2*m+1-k)/k), and G=diag(1,-1,...,-1).
Then the repository's A-matrix is exactly D F L F^dagger D^dagger G.
Indeed the geometric sum gives the real kernel

    (D F L F^dagger D^dagger)[i,j]
      = sin(pi*rho)/(k*sin(pi*(i-j+rho)/k)).

Its columns are orthonormal (also immediate from the phase factorization).
Column zero is positive; all other columns divided by column zero have
positive Cauchy form after G. Thus uniqueness of the normalized Cauchy
columns identifies it with A(x,y). This is an exact identity, not a claim
of a novel Fourier diagonalization theorem.

For k=2**n, two n-qubit QFTs and O(n) fixed-parameter phase gates give
O(log(k)**2) ideal one-/two-qubit gates. With a coherent b-bit rho, its
diagonal phase is bilinear in rho and the Fourier index, so O(b*log(k))
controlled phase gates suffice, plus O(b) gates for its sector-dependent
scalar phase. Retain that scalar phase when rho is coherent. Approximate
rotation synthesis adds its usual logarithmic precision overhead; no
matrix-sized angle list or quantum-Hermite primitive is required.

The cotangent nodes are usually irrational and therefore not literal
Young-diagram contents. Rounding large common positive multiples and then
centering gives genuine integer interlacing, with a conservative explicit
operator-error certificate. This creates a restricted family of diagrams
whose A-matrices have fast approximations; it is not a generic solver.
Recognizing this family from arbitrary input arrays is a separate cost.
All dense arrays here are independent verification oracles. The separate
cotangent_qft_circuit constructor builds actual circuits, with fixed or
coherent parameters, rather than using those arrays as quantum oracles.
"""

from dataclasses import dataclass
import argparse
import math

import numpy as np
from scipy.special import gammaln

from fourier.amatrix import _a_matrix_from_contents_log


def _validate(k, rho):
    if not isinstance(k, (int, np.integer)) or k < 2:
        raise ValueError("k must be an integer at least two")
    if not np.isfinite(rho) or not 0 < rho < 1:
        raise ValueError("rho must lie strictly between zero and one")


def rectangular_contents(k, rho):
    """Return unit-spaced nodes with fractional removable offset rho."""
    _validate(k, rho)
    return k - 1.0 - np.arange(k), k - 1.0 - rho - np.arange(k - 1)


def rectangular_weights(k, rho):
    """Hahn weights: (1-rho)_i (rho)_(k-1-i)/(i! (k-1-i)!)."""
    _validate(k, rho)
    i = np.arange(k)
    n = k - 1
    logs = (gammaln(i + 1 - rho) - gammaln(1 - rho)
            + gammaln(n - i + rho) - gammaln(rho)
            - gammaln(i + 1) - gammaln(n - i + 1))
    return np.exp(logs)


def cotangent_contents(k, rho):
    """Return the interlacing contents of the exact Fourier family."""
    _validate(k, rho)
    x = 1 / np.tan(np.pi * (np.arange(k) + rho) / k)
    y = 1 / np.tan(np.pi * np.arange(1, k) / k)
    return x, y


def cotangent_kernel(k, rho):
    """Construct the closed real A kernel, only for validation."""
    _validate(k, rho)
    i, j = np.arange(k)[:, None], np.arange(k)[None, :]
    c = np.sin(np.pi * rho) / (k * np.sin(np.pi * (i - j + rho) / k))
    c[:, 1:] *= -1
    return c


def fourier_circuit_matrix(k, rho):
    """Dense oracle for the two-QFT circuit, including its scalar phase."""
    _validate(k, rho)
    index = np.arange(k)
    f = np.exp(2j * np.pi * index[:, None] * index[None, :] / k) / np.sqrt(k)
    d = np.exp(1j * np.pi * index / k)
    eigenvalues = np.exp(1j * np.pi * rho * (2 * index + 1 - k) / k)
    c = d[:, None] * ((f * eigenvalues) @ f.conj().T) * d.conj()[None, :]
    c[:, 1:] *= -1
    return c


def cotangent_qft_circuit(k, rho=None, *, rho_bits=None):
    """Build the actual clean QFT circuit for power-of-two k.

    Exactly one of a fixed rho or a coherent rho_bits register is required.
    With b=rho_bits, the parameter register encodes rho=R/2**b in
    little-endian order. Qubits are data, then optional parameter, then
    zeroed work. The work bits implement G with a linear Toffoli chain
    and return exactly to zero. No matrix-sized oracle is used.

    The circuit uses O(n**2+n*b) ideal elementary gates and O(n+b) qubits,
    n=log2(k). The phase of each rho sector is retained. Since the largest
    rho derivative of an eigenphase is below pi, rho rounding contributes
    at most pi*2**(-b) operator error, independently of k.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import QFTGate

    if not isinstance(k, (int, np.integer)) or k < 2 or k & (k - 1):
        raise ValueError("circuit requires power-of-two k>=2")
    if (rho is None) == (rho_bits is None):
        raise ValueError("specify exactly one of rho and rho_bits")
    n = int(k).bit_length() - 1
    if rho_bits is None:
        _validate(k, rho)
        b = 0
    else:
        if not isinstance(rho_bits, (int, np.integer)) or rho_bits < 1:
            raise ValueError("rho_bits must be a positive integer")
        b = int(rho_bits)
    work_count = n - 1 if n > 1 else 0
    circuit = QuantumCircuit(n + b + work_count, name="cotangent_A")
    data = list(range(n))

    # G=+1 on zero and -1 elsewhere, with a clean linear-size AND chain.
    if n == 1:
        circuit.z(0)
    else:
        work = list(range(n + b, n + b + work_count))
        circuit.x(data)
        chain = [(0, 1, work[0])]
        chain.extend((work[j - 2], j, work[j - 1]) for j in range(2, n))
        for controls in chain:
            circuit.ccx(*controls)
        circuit.z(work[-1])
        for controls in reversed(chain):
            circuit.ccx(*controls)
        circuit.x(data)
        circuit.global_phase += np.pi

    for j in range(n):
        circuit.p(-np.pi * 2**j / k, j)
    circuit.append(QFTGate(n).inverse(), data)
    if b:
        for r in range(b):
            coefficient = 2.0 ** (r - b)
            # A parameter-dependent scalar is not a disposable global phase.
            circuit.p(np.pi * coefficient * (1 - k) / k, n + r)
            for j in range(n):
                circuit.cp(2 * np.pi * coefficient * 2**j / k, n + r, j)
    else:
        circuit.global_phase += np.pi * rho * (1 - k) / k
        for j in range(n):
            circuit.p(2 * np.pi * rho * 2**j / k, j)
    circuit.append(QFTGate(n), data)
    for j in range(n):
        circuit.p(np.pi * 2**j / k, j)
    return circuit


def perturbation_bound(k, min_gap, max_node_error):
    """Conservative spectral bound for perturbing all interlaced nodes.

    If each node moves by at most e<delta/2, each difference changes by
    relative error at most 2e/delta. Put h=-log(1-2e/delta). Product
    residues give |Delta log(alpha)| <= (k-1)h, and column normalization
    gives |Delta log(beta)| <= kh. Every A entry therefore changes by
    at most (exp(2kh)-1)*abs(A_ij). Its Frobenius bound is sqrt(k) times
    that factor, and hence also bounds operator norm. Both A matrices
    are orthogonal, so clipping the bound at two is valid.
    """
    if min_gap <= 0 or max_node_error < 0:
        raise ValueError("positive gap and nonnegative error required")
    relative = 2 * max_node_error / min_gap
    if relative >= 1:
        return 2.0
    exponent = -2 * k * np.log1p(-relative)
    if exponent >= np.log1p(2 / np.sqrt(k)):
        return 2.0
    return float(np.sqrt(k) * np.expm1(exponent))


@dataclass(frozen=True)
class RoundedCotangentDiagram:
    addable: np.ndarray
    removable: np.ndarray
    scale: int
    center: int
    max_node_error: float
    error_bound: float

    def blocks(self):
        """Compressed (row width, multiplicity) pairs; never expand rows."""
        widths = [int(self.addable[0])]
        for j in range(len(self.removable) - 1):
            widths.append(widths[-1] - int(self.removable[j] - self.addable[j + 1]))
        return tuple((width, int(x - y)) for width, x, y in
                     zip(widths, self.addable[:-1], self.removable))


def rounded_cotangent_diagram(k, rho, scale):
    """Construct centered integer contents and certify proximity to fast A.

    The certificate treats the computed floating-point cotangent nodes as
    the reference nodes. This numerical constructor is not interval
    arithmetic: include independent evaluation-error bounds when using it
    as a rigorous finite-precision diagram-generation routine.
    """
    x, y = cotangent_contents(k, rho)
    if not isinstance(scale, (int, np.integer)) or scale < 1:
        raise ValueError("scale must be a positive integer")
    if scale * max(np.max(np.abs(x)), np.max(np.abs(y))) > 2**50:
        raise ValueError("scale too large for this floating-point verification oracle")
    xi, yi = np.rint(scale * x).astype(np.int64), np.rint(scale * y).astype(np.int64)
    merged = np.empty(2 * k - 1, dtype=np.int64)
    merged[::2], merged[1::2] = xi, yi
    if np.any(merged[:-1] <= merged[1:]):
        raise ValueError("scale too small to preserve strict interlacing")
    error = max(np.max(np.abs(xi / scale - x)), np.max(np.abs(yi / scale - y)))
    original = np.empty(2 * k - 1)
    original[::2], original[1::2] = x, y
    gap = np.min(original[:-1] - original[1:])
    center = int(np.sum(xi) - np.sum(yi))
    return RoundedCotangentDiagram(xi - center, yi - center, int(scale), center,
                                   float(error), perturbation_bound(k, gap, error))


def sufficient_rounding_scale(k, rho, epsilon):
    """A priori polynomial scale guaranteeing the analytic rounding bound.

    Since |cot'(theta)|>=1, every gap is at least pi*min(rho,1-rho)/k.
    Rounding at scale S moves every unscaled node by at most 1/(2S).
    Consequently S=O(k**(5/2)/(min(rho,1-rho)*epsilon)) suffices.
    This controls the mathematical rounding, not transcendental evaluation.
    """
    _validate(k, rho)
    if not 0 < epsilon < 2:
        raise ValueError("epsilon must lie strictly between zero and two")
    gap_lower_bound = np.pi * min(rho, 1 - rho) / k
    relative = -np.expm1(-np.log1p(epsilon / np.sqrt(k)) / (2 * k))
    return math.ceil(1 / (gap_lower_bound * relative))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128])
    parser.add_argument("--rho", type=float, default=0.37)
    parser.add_argument("--epsilon", type=float, default=1e-3)
    args = parser.parse_args()
    print("k,exact_circuit_error,scale,rounded_error,certified_bound,content_bits")
    for k in args.sizes:
        x, y = cotangent_contents(k, args.rho)
        target = _a_matrix_from_contents_log(x, y)
        circuit = fourier_circuit_matrix(k, args.rho)
        exact_error = np.linalg.norm(circuit - target, 2)
        scale = sufficient_rounding_scale(k, args.rho, args.epsilon)
        rounded = rounded_cotangent_diagram(k, args.rho, scale)
        target_rounded = _a_matrix_from_contents_log(rounded.addable, rounded.removable)
        rounded_error = np.linalg.norm(circuit - target_rounded, 2)
        bits = int(max(abs(int(x)) for x in rounded.addable)).bit_length() + 1
        print(f"{k},{exact_error:.5g},{scale},{rounded_error:.5g},"
              f"{rounded.error_bound:.5g},{bits}")


if __name__ == "__main__":
    main()
