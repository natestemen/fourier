"""Structured spectral factorizations of A-matrices.

The Cauchy formula in :mod:`fourier.amatrix` hides a second exact
description.  If ``X = diag(addable_contents)`` and ``alpha`` is the first
column of the A-matrix, then the remaining columns diagonalize the
compression of ``X`` to ``alpha**perp``.  Equivalently, the A-matrix factors
through the orthogonal-polynomial (Jacobi) basis of the transition measure.

These identities do not by themselves give a short Givens circuit.  They do
replace an unstructured dense factorization by two structured polynomial
transforms, and make the staircase/Hahn special case explicit.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .amatrix import a_matrix_from_contents

__all__ = [
    "CayleyFactorization",
    "JacobiFactorization",
    "ProjectiveCauchyForm",
    "SymmetricContentFactorization",
    "arrowhead_from_contents",
    "cayley_factorization",
    "jacobi_factorization_from_contents",
    "orthogonal_polynomial_transform",
    "projective_cauchy_from_contents",
    "staircase_jacobi_coefficients",
    "symmetric_content_factorization",
]


FloatArray = npt.NDArray[np.float64]


def _validate_contents(ac: FloatArray, rc: FloatArray) -> tuple[FloatArray, FloatArray]:
    ac = np.asarray(ac, dtype=float)
    rc = np.asarray(rc, dtype=float)
    if ac.ndim != 1 or rc.ndim != 1 or len(ac) != len(rc) + 1:
        raise ValueError("expected k addable and k-1 removable contents")
    merged = np.empty(len(ac) + len(rc))
    merged[0::2] = ac
    merged[1::2] = rc
    if not np.all(merged[:-1] > merged[1:]):
        raise ValueError("contents must strictly interlace")
    return ac, rc


def _transition_weights(ac: FloatArray, rc: FloatArray) -> FloatArray:
    """Return alpha**2 stably from the interlaced content formula."""
    log_weights = np.array(
        [
            np.sum(np.log(np.abs(rc - x)))
            - np.sum(np.log(np.abs(np.delete(ac, i) - x)))
            for i, x in enumerate(ac)
        ]
    )
    weights = np.exp(log_weights - np.max(log_weights))
    return weights / np.sum(weights)


@dataclass(frozen=True)
class ProjectiveCauchyForm:
    """Scaled square Cauchy form of the *entire* A-matrix.

    The usual affine Cauchy form omits the constant column.  A Möbius change
    of nodes moves that column's projective node at infinity to a finite
    coordinate and recovers a square displacement-rank-one matrix.
    """

    row_nodes: FloatArray
    column_nodes: FloatArray
    row_scale: FloatArray
    column_scale: FloatArray

    @property
    def core(self) -> FloatArray:
        return 1.0 / (
            self.row_nodes[:, None] - self.column_nodes[None, :]
        )

    def matrix(self) -> FloatArray:
        return (
            self.row_scale[:, None]
            * self.core
            * self.column_scale[None, :]
        )

    @property
    def displacement(self) -> FloatArray:
        matrix = self.matrix()
        return (
            self.row_nodes[:, None] * matrix
            - matrix * self.column_nodes[None, :]
        )


def projective_cauchy_from_contents(
    ac: FloatArray, rc: FloatArray, eta: float | None = None
) -> ProjectiveCauchyForm:
    """Represent every column of A as one scaled square Cauchy matrix.

    The distinguished constant column is naturally a Cauchy node at
    infinity.  The Möbius coordinate ``u(t)=1/(t-eta)`` sends it to zero.
    In the resulting coordinates,

    ``diag(u) @ A - A @ diag(v) = row_scale @ column_scale.T``

    has rank one for the full square A-matrix.
    """
    ac, rc = _validate_contents(ac, rc)
    if eta is None:
        span = max(1.0, float(ac[0] - ac[-1]))
        eta = float(ac[0] + span)
    if np.any(np.isclose(ac, eta)) or np.any(np.isclose(rc, eta)):
        raise ValueError("eta must differ from every content")

    weights = _transition_weights(ac, rc)
    alpha = np.sqrt(weights)
    beta = np.array(
        [1.0 / np.linalg.norm(alpha / (ac - y)) for y in rc],
        dtype=float,
    )
    row_nodes = 1.0 / (ac - eta)
    column_nodes = np.r_[0.0, 1.0 / (rc - eta)]
    row_scale = alpha / (ac - eta)
    column_scale = np.r_[1.0, -beta / (rc - eta)]
    return ProjectiveCauchyForm(
        row_nodes=row_nodes,
        column_nodes=column_nodes,
        row_scale=row_scale,
        column_scale=column_scale,
    )


def arrowhead_from_contents(ac: FloatArray, rc: FloatArray) -> FloatArray:
    """Return the arrowhead ``H`` satisfying ``diag(ac) @ A = A @ H``.

    In exact arithmetic,

    ``H = A.T @ diag(ac) @ A``

    has diagonal ``(sum(alpha**2 * ac), *rc)`` and its only nonzero
    off-diagonal entries are ``H[0, 1:] = H[1:, 0] = beta``.  For genuine
    Young-diagram contents the leading diagonal entry is zero; retaining the
    weighted mean makes the identity invariant under translating all nodes.
    """
    ac, rc = _validate_contents(ac, rc)
    weights = _transition_weights(ac, rc)
    alpha = np.sqrt(weights)
    beta = np.array(
        [1.0 / np.linalg.norm(alpha / (ac - y)) for y in rc],
        dtype=float,
    )

    H = np.zeros((len(ac), len(ac)))
    H[0, 0] = float(weights @ ac)
    H[0, 1:] = beta
    H[1:, 0] = beta
    H[1:, 1:] = np.diag(rc)
    return H


def orthogonal_polynomial_transform(
    nodes: FloatArray, weights: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Build the weighted orthogonal-polynomial transform and Jacobi matrix.

    This is symmetric Lanczos applied to ``diag(nodes)`` with starting vector
    ``sqrt(weights)``.  The returned orthogonal matrix ``V`` has columns
    ``sqrt(weights_i) p_j(nodes_i)``, and ``J = V.T @ diag(nodes) @ V`` is
    tridiagonal.  Full reorthogonalization keeps this research implementation
    reliable for the moderately sized instances used in the experiments.
    """
    nodes = np.asarray(nodes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if nodes.ndim != 1 or weights.shape != nodes.shape:
        raise ValueError("nodes and weights must be one-dimensional and equally sized")
    if np.any(weights <= 0.0):
        raise ValueError("weights must be positive")
    weights = weights / np.sum(weights)

    k = len(nodes)
    V = np.zeros((k, k))
    diagonal = np.zeros(k)
    off_diagonal = np.zeros(max(0, k - 1))
    V[:, 0] = np.sqrt(weights)

    for j in range(k):
        residual = nodes * V[:, j]
        diagonal[j] = V[:, j] @ residual
        residual -= diagonal[j] * V[:, j]
        if j:
            residual -= off_diagonal[j - 1] * V[:, j - 1]

        if j + 1 == k:
            continue

        # The three-term recurrence is exact algebraically.  Reorthogonalize
        # only to suppress accumulated floating-point components.
        residual -= V[:, : j + 1] @ (V[:, : j + 1].T @ residual)
        off_diagonal[j] = np.linalg.norm(residual)
        if off_diagonal[j] <= 10.0 * np.finfo(float).eps * np.linalg.norm(nodes):
            raise np.linalg.LinAlgError("Lanczos breakdown before the full Jacobi basis")
        V[:, j + 1] = residual / off_diagonal[j]

    J = np.diag(diagonal)
    if k > 1:
        J += np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)
    return V, J


@dataclass(frozen=True)
class JacobiFactorization:
    """Exact structured factorization ``A = polynomial @ associated``.

    ``polynomial`` is the transition-measure orthogonal-polynomial transform.
    ``associated`` fixes its first coordinate and diagonalizes the trailing
    principal submatrix of ``jacobi``.  Thus the second factor is one
    dimension smaller and is itself an associated-polynomial transform.
    """

    polynomial: FloatArray
    associated: FloatArray
    jacobi: FloatArray
    tail_eigenvalues: FloatArray

    def matrix(self) -> FloatArray:
        return self.polynomial @ self.associated


@dataclass(frozen=True)
class CayleyFactorization:
    """Orthogonal matrix represented by a skew-symmetric Cayley generator.

    The fixed signs are free reflections in the repository's rotation-count
    model.  For the A-matrix sign convention they are ``(1, -1, ..., -1)``.
    """

    generator: FloatArray
    signs: FloatArray
    gauge_condition: float

    def matrix(self) -> FloatArray:
        k = len(self.signs)
        identity = np.eye(k)
        proper = np.linalg.solve(
            identity + self.generator, identity - self.generator
        )
        return proper @ np.diag(self.signs)


def cayley_factorization(A: FloatArray) -> CayleyFactorization:
    """Put an A-matrix in a stable, exactly skew Cayley chart.

    If ``D = diag(1, -1, ..., -1)`` and ``Q = A D``, then the usual content
    convention gives ``det(Q) = 1``.  When ``I + Q`` is nonsingular,

    ``S = (I - Q) (I + Q)^-1``

    is skew-symmetric and ``A = (I + S)^-1 (I - S) D``.  Approximating ``S``
    while preserving skew symmetry therefore preserves orthogonality exactly;
    this is useful for turning low-rank off-diagonal compression into a valid
    approximate circuit target.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    k = A.shape[0]
    signs = np.ones(k)
    signs[1:] = -1.0
    proper = A * signs[None, :]
    shifted = np.eye(k) + proper
    condition = float(np.linalg.cond(shifted))
    if not np.isfinite(condition):
        raise np.linalg.LinAlgError("the fixed sign gauge lies outside the Cayley chart")

    generator = np.linalg.solve(shifted.T, (np.eye(k) - proper).T).T
    # Project away roundoff.  This does not alter the exact mathematical
    # generator and makes downstream structured truncations exactly skew.
    generator = 0.5 * (generator - generator.T)
    return CayleyFactorization(
        generator=generator,
        signs=signs,
        gauge_condition=condition,
    )


@dataclass(frozen=True)
class SymmetricContentFactorization:
    """Parity split of an A-matrix with centrally symmetric contents.

    ``left_basis`` and ``right_basis`` are each products of disjoint
    pi/4 rotations.  In those bases the matrix is the direct sum of
    ``plus_block`` and ``minus_block``.
    """

    left_basis: FloatArray
    plus_block: FloatArray
    minus_block: FloatArray
    right_basis: FloatArray

    @property
    def rotation_count(self) -> int:
        """Count using dense Givens only inside the two half-size blocks."""
        p = self.plus_block.shape[0]
        q = self.minus_block.shape[0]
        k = p + q
        basis_rotations = k - 1
        return basis_rotations + p * (p - 1) // 2 + q * (q - 1) // 2

    def matrix(self) -> FloatArray:
        p = self.plus_block.shape[0]
        q = self.minus_block.shape[0]
        middle = np.zeros((p + q, p + q))
        middle[:p, :p] = self.plus_block
        middle[p:, p:] = self.minus_block
        return self.left_basis @ middle @ self.right_basis.T


def symmetric_content_factorization(
    ac: FloatArray, rc: FloatArray
) -> SymmetricContentFactorization:
    """Exactly split a centrally symmetric A-matrix into two parity blocks.

    This includes every staircase and, more generally, self-conjugate Young
    diagrams.  After a harmless common translation, central symmetry means

    ``ac[i] = -ac[k-1-i]`` and ``rc[j] = -rc[k-2-j]``.

    Row reversal then intertwines A with ``1 direct_sum (-tail_reversal)`` on
    the columns.  Pairwise pi/4 rotations diagonalize both involutions, so

    ``A = U @ (B_plus direct_sum B_minus) @ V.T``.

    Factoring the two blocks generically costs about half the dense Givens
    count, plus exactly ``k-1`` fixed pi/4 rotations for ``U`` and ``V``.
    """
    ac, rc = _validate_contents(ac, rc)
    center = 0.5 * (ac[0] + ac[-1])
    scale = max(1.0, float(np.max(np.abs(ac - center))))
    if not np.allclose(ac + ac[::-1], 2.0 * center, atol=1e-11 * scale):
        raise ValueError("addable contents are not centrally symmetric")
    if len(rc) and not np.allclose(
        rc + rc[::-1], 2.0 * center, atol=1e-11 * scale
    ):
        raise ValueError("removable contents do not share the same center")

    A = a_matrix_from_contents(ac, rc)
    k = len(ac)
    root_half = np.sqrt(0.5)

    left_basis = np.eye(k)
    for i in range(k // 2):
        j = k - 1 - i
        left_basis[np.ix_([i, j], [i, j])] = [
            [root_half, -root_half],
            [root_half, root_half],
        ]

    right_basis = np.eye(k)
    for i in range((k - 1) // 2):
        j = 1 + i
        ell = k - 1 - i
        right_basis[np.ix_([j, ell], [j, ell])] = [
            [root_half, root_half],
            [-root_half, root_half],
        ]

    middle = left_basis.T @ A @ right_basis
    p = (k + 1) // 2
    if not (
        np.linalg.norm(middle[:p, p:], ord=2) < 1e-10
        and np.linalg.norm(middle[p:, :p], ord=2) < 1e-10
    ):
        raise np.linalg.LinAlgError("parity bases did not block diagonalize A")

    return SymmetricContentFactorization(
        left_basis=left_basis,
        plus_block=middle[:p, :p],
        minus_block=middle[p:, p:],
        right_basis=right_basis,
    )


def jacobi_factorization_from_contents(
    ac: FloatArray, rc: FloatArray
) -> JacobiFactorization:
    """Factor the A-matrix determined by interlaced contents.

    Let ``V`` be the polynomial transform for the transition weights.  In the
    ``V`` basis, compressing ``diag(ac)`` to the complement of the constant
    polynomial simply removes the first row and column of the Jacobi matrix.
    Its eigenvalues are exactly ``rc``.  Diagonalizing that trailing Jacobi
    matrix therefore gives

    ``A = V @ (1 direct_sum W)``,

    up to free column signs, which are fixed here by the positive first row
    convention of :func:`fourier.amatrix.a_matrix_from_contents`.
    """
    ac, rc = _validate_contents(ac, rc)
    weights = _transition_weights(ac, rc)
    V, J = orthogonal_polynomial_transform(ac, weights)
    k = len(ac)

    associated = np.eye(k)
    if k > 1:
        eigenvalues, W = np.linalg.eigh(J[1:, 1:])
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        W = W[:, order]
        if not np.allclose(eigenvalues, rc, rtol=1e-10, atol=1e-10):
            raise np.linalg.LinAlgError(
                "trailing Jacobi spectrum does not match the removable contents"
            )

        tail = V[:, 1:] @ W
        signs = np.where(tail[0, :] < 0.0, -1.0, 1.0)
        associated[1:, 1:] = W * signs
    else:
        eigenvalues = np.empty(0)

    return JacobiFactorization(
        polynomial=V,
        associated=associated,
        jacobi=J,
        tail_eigenvalues=eigenvalues,
    )


def staircase_jacobi_coefficients(k: int) -> tuple[FloatArray, FloatArray]:
    """Closed Jacobi recurrence for the ``k``-by-``k`` staircase matrix.

    With ``N = k - 1``, the transition weights are the symmetric Hahn weights
    with parameters ``(-1/2, -1/2)``.  In the unscaled content coordinate the
    Jacobi diagonal vanishes and the positive off-diagonal coefficients are

    ``b_0**2 = N(N+1)/2``,

    ``b_j**2 = (N-j)(N+j+1)/4`` for ``1 <= j < N``.
    """
    if k < 1:
        raise ValueError("k must be positive")
    diagonal = np.zeros(k)
    if k == 1:
        return diagonal, np.empty(0)

    n = k - 1
    off_diagonal = np.empty(n)
    off_diagonal[0] = np.sqrt(n * (n + 1) / 2.0)
    if n > 1:
        j = np.arange(1, n, dtype=float)
        off_diagonal[1:] = 0.5 * np.sqrt((n - j) * (n + j + 1.0))
    return diagonal, off_diagonal
