"""Dense validation of a coherent row/column-overlap implementation of A.

This file validates the isometry and amplification identities, not a compiled
quantum circuit or its gate count. For any real orthogonal k-by-k matrix A,
its absolute row and column sums are at most sqrt(k). Define isometries

    P|i> = sum_j sqrt(|A_ij|/mu) |i,j,good> + row_padding_i,
    Q|j> = sum_i sign(A_ij) sqrt(|A_ij|/mu) |i,j,good> + col_padding_j.

The row and column padding sectors must be distinct. Then P.T Q = A/mu.
Choose m=ceil(pi*sqrt(k)/2) and mu=csc(pi/(4m+2)). This gives mu>=2sqrt(k),
so the padding probabilities have slack, and exact oblivious amplification

    (-R_Q R_P)^m Q = P A,
    R_P = 2 P P.T - I, R_Q = 2 Q Q.T - I.

The minus sign is necessary: it fixes the global phase, which matters when
the transform is controlled. The identity holds on every input superposition
and cleans the failure subspace without measurement or postselection.
This uses the standard oblivious-amplification identity of Berry, Childs,
Cleve, Kothari, Somma, arXiv:1312.1414v2, Lemma 3.6:
https://arxiv.org/html/1312.1414v2#S3

For A-matrices, already-computed content weights permit one row or column
to be assembled reversibly in O(k) scalar operations. A binary state-
preparation tree also has O(k) rotations; its data can be uncomputed because
it depends only on the preserved source index, not the prepared target.
One source-index lookup by a linear scan is O(k), not free QRAM. Thus this
identity reduces application to O(k**1.5) operations, suppressing precision
and address-control factors, IF the all-weight preprocessing is subquadratic.
The separate logarithmic-potential multipole experiment addresses that step.

The stronger ``a_matrix_positive_scaling`` construction below reduces the
normalization, and hence the amplification count, to O(log k) by exploiting
the A-matrix sign pattern. It requires O(log k) positive Cauchy matvecs for
preprocessing. If those and the weights are computed by fixed-word reversible
multipole arithmetic, the combined cost is k*polylog(k,L,1/eps), not k**1.5.
The dense routines here test the identities, not that compiled implementation.

A conservative precision budget is relative entry error eta=O(eps/k**1.5).
Every binary-tree subtree sum then has relative error O(eta). Computing each
angle as 2*atan2(sqrt(right),sqrt(left)) changes it by O(eta), including
very unbalanced branches; slack controls the padding branch. Telescoping the
O(k) tree rotations and O(sqrt(k)) amplification calls bounds full coherent
error by O(k**1.5*eta). Inactive zero leaves are fixed, not evaluated via 0/0.
This only incurs logarithmic precision overhead. It is deliberately more
conservative than an analysis grouping mutually exclusive tree controls.
For the scaled construction the same reasoning needs eta=O(eps/(k*log k)).
Finite positive scaling vectors cancel algebraically in P.T Q; they need
only sufficient accuracy to preserve the padding inequalities, not epsilon
accuracy themselves. An absolute scaling-vector error at most .01/sqrt(k)
is safely inside the chosen factor-two normalization slack.

This constructor explicitly stores O(k**3) dense isometry data to test the
mathematics; it is NOT that reversible O(k)-work state-preparation algorithm.
"""

from dataclasses import dataclass
import argparse

import numpy as np

from fourier.amatrix import _a_matrix_from_contents_log


def amplification_parameters(k: int) -> tuple[int, float]:
    """Return a fixed iteration count and normalization with exact peak phase."""
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer")
    steps = int(np.ceil(np.pi * np.sqrt(k) / 2))
    normalization = float(1 / np.sin(np.pi / (4 * steps + 2)))
    return steps, normalization


@dataclass
class OverlapEncoding:
    row_isometry: np.ndarray
    column_isometry: np.ndarray
    normalization: float

    @property
    def block(self) -> np.ndarray:
        return self.row_isometry.T @ self.column_isometry

    def amplify(self, steps: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the good block and full failure map after coherent OAA.

        Projection arithmetic avoids constructing the ambient D-by-D unitary.
        Columns represent all input basis states, so spectral norms test the
        worst superposition, rather than only individual success probabilities.
        """
        if not isinstance(steps, (int, np.integer)) or steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        p, q = self.row_isometry, self.column_isometry
        current = q.copy()
        for _ in range(steps):
            reflected_p = 2 * p @ (p.T @ current) - current
            current = reflected_p - 2 * q @ (q.T @ reflected_p)
        good = p.T @ current
        failure = current - p @ good
        return good, failure


def overlap_encoding(matrix, normalization=None, *, left_scaling=None,
                     right_scaling=None) -> OverlapEncoding:
    """Build exact row/column isometries for a real square test matrix.

    Positive scalings l,r replace the two good amplitudes by
    sqrt(|A_ij| r_j/(mu l_i)) and sign(A_ij)*sqrt(|A_ij| l_i/(mu r_j)).
    Their product remains A_ij/mu. These weighted sums, not the original
    unweighted sums, determine the required normalization.
    Weighted square-root-entry states are established quantum-walk machinery;
    see Childs, arXiv:0810.0312v3, Section 2, Eqs. (4)--(5). Our application
    replaces an exact Perron vector by a finite positive Neumann sum and slack.

    Unitarity is not required to form the block encoding. It is required for
    the exact fixed-angle amplification identity; perturbation tests use this
    distinction explicitly. The two private padding sectors are never merged.
    """
    matrix = np.asarray(matrix, dtype=float)
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]
            or matrix.shape[0] == 0 or not np.all(np.isfinite(matrix))):
        raise ValueError("expected a finite, nonempty, real square matrix")
    k = matrix.shape[0]
    if normalization is None:
        _, normalization = amplification_parameters(k)
    normalization = float(normalization)
    if not np.isfinite(normalization) or normalization <= 0:
        raise ValueError("normalization must be finite and positive")
    left = np.ones(k) if left_scaling is None else np.asarray(left_scaling, dtype=float)
    right = np.ones(k) if right_scaling is None else np.asarray(right_scaling, dtype=float)
    if (left.shape != (k,) or right.shape != (k,)
            or not np.all(np.isfinite(left)) or not np.all(np.isfinite(right))
            or np.min(left) <= 0 or np.min(right) <= 0):
        raise ValueError("scaling vectors must be finite and strictly positive")
    row_sums = np.abs(matrix) @ right / left
    column_sums = np.abs(matrix).T @ left / right
    if max(np.max(row_sums), np.max(column_sums)) > normalization:
        raise ValueError("normalization is smaller than an absolute row/column sum")
    p = np.zeros((k * k + 2 * k, k))
    q = np.zeros_like(p)
    p_magnitudes = np.sqrt(np.abs(matrix) * right[None, :] / left[:, None] / normalization)
    q_magnitudes = np.sqrt(np.abs(matrix) * left[:, None] / right[None, :] / normalization)
    signs = np.sign(matrix)
    for i in range(k):
        p[i * k:(i + 1) * k, i] = p_magnitudes[i]
        q[i * k:(i + 1) * k, :] = np.diag(signs[i] * q_magnitudes[i])
    for i in range(k):
        p[k * k + i, i] = np.sqrt(1 - row_sums[i] / normalization)
        q[k * k + k + i, i] = np.sqrt(1 - column_sums[i] / normalization)
    return OverlapEncoding(p, q, normalization)


@dataclass
class PositiveScaling:
    left: np.ndarray
    right: np.ndarray
    absolute_norm_bound: float
    neumann_steps: int
    amplification_steps: int
    normalization: float


def a_matrix_positive_scaling(matrix) -> PositiveScaling:
    """Dense reference for logarithmic-normalization A-matrix block encoding.

    This uses only the A-matrix sign pattern and orthogonality, not its poles.
    Let d=ceil(log2 k) and Q=A*diag(-1,1,...,1). Then Q is nonnegative
    strictly above the diagonal and nonpositive strictly below it. The signed
    strict off-diagonal part decomposes into d dyadic sibling levels. Each
    level is a direct sum of blocks [[0,Q_LR],[-Q_RL,0]], with norm at most
    one. Adding the absolute diagonal, also of norm at most one, gives
    ||abs(A)||_2 <= d+1. Positive identity-padding diagonal entries are allowed.

    Set B=abs(A), H=[[0,B],[B.T,0]], lambda=2(d+1), and
    v=sum_{t=0}^T (H/lambda)^t 1, with 2^(-T-1)*sqrt(2k)<=1/2.
    Then H v=lambda*(v-1+remainder)<=lambda*v, componentwise. Its two
    halves l,r satisfy B r<=lambda*l and B.T l<=lambda*r. Also
    1<=v_i<=2sqrt(2k). Normalizing at mu>=2lambda gives padding>=1/2.
    Both T and the number of amplification steps are O(log k).

    A fast implementation needs O(log k) positive Cauchy matvecs, for which
    1/|x-y| multipole evaluation is a near-linear-arithmetic candidate. This
    dense reference deliberately uses matrix products to test the identity.
    """
    matrix = np.asarray(matrix, dtype=float)
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]
            or matrix.shape[0] == 0 or not np.all(np.isfinite(matrix))):
        raise ValueError("expected a finite, nonempty, real square matrix")
    k = matrix.shape[0]
    # Check the convention on which the dyadic norm argument relies.
    ii, jj = np.indices((k, k))
    expected_sign = np.where((jj == 0) | (ii < jj), 1, -1)
    if np.any((ii != jj) & (matrix * expected_sign < 0)):
        raise ValueError("matrix does not have the A-matrix sign convention")
    if np.linalg.norm(matrix.T @ matrix - np.eye(k), 2) > 1e-8:
        raise ValueError("matrix must be orthogonal within reference precision")
    bound = float(int(np.ceil(np.log2(k))) + 1)
    lam = 2 * bound
    terms = int(np.ceil(np.log2(2 * np.sqrt(2 * k))))
    b = np.abs(matrix)
    u, w = np.ones(k), np.ones(k)
    left, right = u.copy(), w.copy()
    for _ in range(terms):
        u, w = b @ w / lam, b.T @ u / lam
        left += u
        right += w
    steps = int(np.ceil(np.pi * lam / 2))
    normalization = float(1 / np.sin(np.pi / (4 * steps + 2)))
    return PositiveScaling(left, right, bound, terms, steps, normalization)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--balanced", action="store_true",
                        help="use logarithmic normalization from positive scaling")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    print("family,k,amplification_steps,normalization,block_error,output_error,failure_norm")
    for family in ["staircase", "integer", "wide_integer"]:
        for k in args.sizes:
            gaps = (np.ones(2 * k - 2) if family == "staircase" else
                    rng.integers(1, 8, 2 * k - 2) if family == "integer" else
                    rng.integers(1, 10000, 2 * k - 2))
            merged = np.r_[0, np.cumsum(gaps)][::-1]
            matrix = _a_matrix_from_contents_log(merged[::2], merged[1::2])
            if args.balanced:
                scaling = a_matrix_positive_scaling(matrix)
                steps, normalization = scaling.amplification_steps, scaling.normalization
                encoded = overlap_encoding(matrix, normalization,
                                           left_scaling=scaling.left,
                                           right_scaling=scaling.right)
            else:
                steps, normalization = amplification_parameters(k)
                encoded = overlap_encoding(matrix, normalization)
            output, failure = encoded.amplify(steps)
            print(f"{family},{k},{steps},{normalization:.7g},"
                  f"{np.linalg.norm(normalization * encoded.block - matrix, 2):.5g},"
                  f"{np.linalg.norm(output - matrix, 2):.5g},"
                  f"{np.linalg.norm(failure, 2):.5g}")


if __name__ == "__main__":
    main()
