"""Exact Cauchy-generator Givens synthesis with quadratic arithmetic.

This is a rank-one, real-node specialization of the generator QR construction
in Delvaux, Gemignani, Van Barel, *Fast QR factorization of Cauchy-like
matrices*, Section 3, https://www.cs.kuleuven.be/publicaties/rapporten/tw/TW469.pdf
(published in LAA 428 (2008), 697--711, doi:10.1016/j.laa.2007.08.038).

Nested suffix-row/prefix-column QR factors are represented by two vectors
u=Q.T@g and v=h@inv(R), because Q.T@diag(x)@Q has diagonal y+u*v
and lower triangle u[i]*v[j]. Symmetry supplies the upper triangle.
Adding one row requires n-1 adjacent rotations and O(n) arithmetic.
The last v entry follows from the final diagonal, avoiding a structured solve.

In one bordering step initialize u=(g_new,u_old), v=(0,v_old).
At slot j set a=x_new-y[j], b=u[j+1]*v[j]-u[j]*v[j+1],
and (c,s)=(-b,a)/hypot(a,b). Rotate both generator pairs by
[[c,s],[-s,c]]. Finally v[-1] += (x_new-y[n-1])/u[-1].
The kernel equation (a,b)@(c,s)=0 is exactly the lower-triangular
displacement identity after the next QR rotation. Its new bottom diagonal
determines the last generator. Induction therefore produces the same QR
factorization as dense bordering, up to harmless diagonal signs.
The simplification a=x_new-y[j] follows from the invariant
d-u[j]*v[j]=x_new for the carried diagonal d, avoiding cancellation.
Every a is nonzero because row and column nodes are disjoint. The final
u[-1] is nonzero because [C[:,:-1],g] is a nonsingular Cauchy-Vandermonde
matrix; equivalently g has nonzero projection onto the last QR vector.

Classical arithmetic is O(k^2), mutable generator storage O(k), and the emitted
word has k(k-1)/2 rotations. Storing all angles uses O(k^2) memory. No dense
matrix is used to synthesize angles. Fixed precision can suffer generator
cancellation; numerical stability must be measured, not assumed from the
algebraic operation count. The default reversed columns keep the nested
submatrices close to the interlaced diagonal.
"""

from dataclasses import dataclass
import argparse
import time

import numpy as np

from fourier.amatrix import _a_matrix_from_contents_log
from fourier.decompositions import Givens


@dataclass
class GeneratorQR:
    gates: list[Givens]
    signs: np.ndarray
    column_order: np.ndarray
    generator_norm: float

    def matrix(self):
        k = len(self.signs)
        # Verification only: row updates avoid multiplying dense gate matrices.
        q = np.diag(self.signs)
        for gate in self.gates:
            c, s = np.cos(gate.theta), np.sin(gate.theta)
            ri, rj = q[gate.i].copy(), q[gate.j].copy()
            q[gate.i], q[gate.j] = c * ri - s * rj, s * ri + c * rj
        return q[:, np.argsort(self.column_order)]


def generator_qr(ac, rc, *, reverse_columns=True, dtype=np.longdouble):
    """Return fixed-word Givens factors using only rank-one generators.

    Compute the residue weights and Möbius generators in the same arithmetic
    precision as the recurrence. Output angles are float64 for interoperability.
    """
    ac, rc = np.asarray(ac, dtype=dtype), np.asarray(rc, dtype=dtype)
    k = len(ac)
    if ac.ndim != 1 or rc.ndim != 1 or k != len(rc) + 1 or k == 0:
        raise ValueError("expected k>=1 addable and k-1 removable contents")
    merged = np.empty(2 * k - 1, dtype=dtype)
    merged[::2], merged[1::2] = ac, rc
    if not np.all(np.isfinite(merged)) or not np.all(merged[:-1] > merged[1:]):
        raise ValueError("contents must be finite and strictly interlace")
    log_weights = np.array([
        np.sum(np.log(np.abs(rc - node)))
        - np.sum(np.log(np.abs(np.delete(ac, i) - node)))
        for i, node in enumerate(ac)
    ], dtype=dtype)
    weights = np.exp(log_weights - np.max(log_weights))
    weights /= np.sum(weights)
    alpha = np.sqrt(weights)
    eta = ac[0] + max(dtype(1), ac[0] - ac[-1])
    x = 1 / (ac - eta)
    y = np.r_[dtype(0), 1 / (rc - eta)]
    g = alpha / (ac - eta)
    order = np.arange(k)[::-1] if reverse_columns else np.arange(k)
    y = y[order]
    u = np.array([g[-1]], dtype=dtype)
    v = np.array([(x[-1] - y[0]) / u[0]], dtype=dtype)
    ops = []
    largest_generator = max(np.max(np.abs(u)), np.max(np.abs(v)))
    for n in range(2, k + 1):
        start = k - n
        u = np.r_[g[start], u]
        v = np.r_[dtype(0), v]
        for j in range(n - 1):
            ul, ur = u[j], u[j + 1]
            vl, vr = v[j], v[j + 1]
            a = x[start] - y[j]
            b = ur * vl - ul * vr
            rho = np.hypot(a, b)
            if rho == 0 or not np.isfinite(rho):
                raise ArithmeticError(f"generator breakdown at n={n}, j={j}")
            c, s = -b / rho, a / rho
            u[j], u[j + 1] = c * ul + s * ur, -s * ul + c * ur
            v[j], v[j + 1] = c * vl + s * vr, -s * vl + c * vr
            ops.append(Givens(start + j, start + j + 1, float(np.arctan2(s, c))))
        v[-1] += (x[start] - y[n - 1]) / u[-1]
        largest_generator = max(largest_generator, np.max(np.abs(u)), np.max(np.abs(v)))
    # For the full orthogonal matrix R is diagonal signs, and v=h/R.
    # eta>ac[0] makes every column generator h positive, so beta is unneeded.
    signs = np.sign(np.asarray(v, dtype=float))
    return GeneratorQR(list(reversed(ops)), signs, order, float(largest_generator))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128])
    parser.add_argument("--seed", type=int, default=20261022)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    print("family,k,rotations,seconds,max_error,fro_error,generator_max")
    for family in ["staircase", "random", "logwide"]:
        for k in args.sizes:
            gaps = (np.ones(2*k-2) if family == "staircase" else
                    rng.integers(1, 4, 2*k-2) if family == "random" else
                    np.exp(rng.uniform(-3, 3, 2*k-2)))
            contents = np.r_[0., np.cumsum(gaps)][::-1]
            ac, rc = contents[::2], contents[1::2]
            A = _a_matrix_from_contents_log(ac, rc)
            start = time.perf_counter()
            result = generator_qr(ac, rc)
            elapsed = time.perf_counter() - start
            error = result.matrix() - A
            print(f"{family},{k},{len(result.gates)},{elapsed:.6g},"
                  f"{np.max(np.abs(error)):.6g},{np.linalg.norm(error):.6g},"
                  f"{result.generator_norm:.6g}")


if __name__ == "__main__":
    main()
