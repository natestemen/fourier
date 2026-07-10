"""Two-qubit gate invariants of 4×4 A-matrices.

The headline fact (report.md, Finding 1): every 4-addable A-matrix has Weyl
coordinate a = π/4 — the maximally-entangling face of the Weyl chamber that
also contains CNOT and CZ — while (b, c) vary continuously over the family.
This module collects the tools used to establish and probe that fact:

- `weyl_coordinates`: the (a, b, c) KAK invariants, numerically via Qiskit.
- `kak_vector_symbolic`: the exact k-vector via Tucci's magic-basis SVD
  method, used for the symbolic proof on the generic-family matrices.
- `leakiness_rank` / `is_leaky`: the Peterson–Crooks–Smith leakiness test.
  A-matrices all turn out non-leaky (rank 3), ruling leakiness out as the
  explanation for a = π/4.
- `block_rotation_form`: the orthogonal conjugation of a 4-addable A-matrix
  to diag(1, −1) ⊕ R(θ), whose angle θ carries the (b, c) information.
"""

import numpy as np
import numpy.typing as npt
import sympy as sp

__all__ = [
    "weyl_coordinates",
    "kak_vector_symbolic",
    "leakiness_rank",
    "leaky_direction",
    "is_leaky",
    "block_rotation_form",
]


def weyl_coordinates(U: npt.NDArray) -> tuple[float, float, float]:
    """The Weyl-chamber coordinates (a, b, c) of a two-qubit unitary.

    Accepts any 4×4 unitary, including det = −1 orthogonal matrices (they are
    rescaled into SU(4) first, which leaves the invariants unchanged).
    """
    from qiskit.synthesis import TwoQubitWeylDecomposition

    U = np.asarray(U, dtype=complex)
    U = U / np.linalg.det(U) ** 0.25
    d = TwoQubitWeylDecomposition(U)
    return float(d.a), float(d.b), float(d.c)


# ── symbolic KAK via Tucci's magic-basis method ────────────────────────────────

_MAGIC_BASIS = (
    sp.Matrix(
        [
            [1, 0, 0, 1],
            [0, sp.I, sp.I, 0],
            [0, -1, 1, 0],
            [sp.I, 0, 0, -sp.I],
        ]
    )
    / sp.sqrt(2)
)

# Hadamard-like matrix Γ from Tucci (Eq. 33)
_GAMMA = sp.Matrix(
    [
        [1, 1, 1, 1],
        [1, 1, -1, -1],
        [1, -1, 1, -1],
        [1, -1, -1, 1],
    ]
)

_H2 = sp.Matrix([[1, 1], [1, -1]]) / sp.sqrt(2)


def kak_vector_symbolic(A: sp.Matrix) -> tuple[sp.Matrix, sp.Matrix]:
    """Exact KAK (Cartan) k-vector of a 4×4 unitary via Tucci's SVD method.

    Returns (k, θ) where k = (k₀, k₁, k₂, k₃) are the Cartan coefficients and
    θ the diagonal phases of e^{iΘ}.  Steps: conjugate into the magic basis,
    simultaneously diagonalize real and imaginary parts with QL = U·(H⊕H),
    QR = V·(H⊕H) from the SVD of the real part, then k = Γᵀθ/4.  Requires the
    real part to have full rank (true for the A-matrices this is used on).
    """
    M = _MAGIC_BASIS
    Xp = M.H * A * M

    XR = Xp.applyfunc(sp.re)
    UA, _, VA = XR.singular_value_decomposition()

    P = sp.diag(_H2, _H2)
    QL = UA * P
    QR = VA * P

    E = (QL.T * Xp * QR).applyfunc(sp.simplify)
    theta = sp.Matrix([sp.atan2(sp.im(E[i, i]), sp.re(E[i, i])) for i in range(4)])
    k = (_GAMMA.T / 4) * theta
    return sp.simplify(k), sp.simplify(theta)


# ── leakiness (Peterson, Crooks & Smith 2019) ──────────────────────────────────

_SX = np.array([[0, 1], [1, 0]], dtype=complex)
_SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
_SZ = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULIS = [_SX, _SY, _SZ]
_SU2_BASIS = [1j * p for p in _PAULIS]
_NONLOCAL = [np.kron(a, b) for a in _PAULIS for b in _PAULIS]


def _leakiness_system(U: npt.NDArray[np.complex128], direction: str) -> npt.NDArray[np.float64]:
    """The 18×3 real system whose null space is the set of leaky directions."""
    L = np.zeros((9, 3), dtype=complex)
    I2 = np.eye(2, dtype=complex)
    for k, ek in enumerate(_SU2_BASIS):
        X = np.kron(ek, I2) if direction == "left" else np.kron(I2, ek)
        Mk = U @ X @ U.conj().T
        for j, Gab in enumerate(_NONLOCAL):
            L[j, k] = np.trace(Mk @ (1j * Gab))
    return np.vstack([L.real, L.imag])


def leakiness_rank(U: npt.NDArray, direction: str = "left") -> tuple[int, float]:
    """Rank of the leakiness system and its smallest singular value.

    U is leaky in the given direction iff the rank is < 3, i.e. some nonzero
    h ∈ su(2) satisfies U·(h⊗I)·U† ∈ su(2)⊗I + I⊗su(2).  `direction` is
    "left" for h⊗I or "right" for I⊗h.  Reference points: identity and SWAP
    give rank 0, CZ gives rank 2, a generic gate gives rank 3.
    """
    sv = np.linalg.svd(_leakiness_system(np.asarray(U, dtype=complex), direction), compute_uv=False)
    tol = 1e-8 * float(sv[0]) if sv[0] > 0 else 1e-8
    return int(np.sum(sv > tol)), float(sv[-1])


def leaky_direction(U: npt.NDArray, direction: str = "left") -> npt.NDArray[np.float64] | None:
    """The leaky su(2) direction (α₁, α₂, α₃) — meaning h = Σ αₖ·iσₖ — if U is
    leaky in the given direction, else None.  For CZ this is iσ_z."""
    L = _leakiness_system(np.asarray(U, dtype=complex), direction)
    _, sv, Vt = np.linalg.svd(L)
    if sv[-1] < 1e-8 * sv[0]:
        return Vt[-1]
    return None


def is_leaky(U: npt.NDArray) -> bool:
    """True iff U is leaky in either direction."""
    return leakiness_rank(U, "left")[0] < 3 or leakiness_rank(U, "right")[0] < 3


# ── block-rotation normal form ─────────────────────────────────────────────────


def block_rotation_form(
    A: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], float, float]:
    """Orthogonal Q and angle θ with Qᵀ·A·Q = diag(1, −1) ⊕ R(θ).

    Exists for every 4-addable A-matrix because its spectrum is
    {+1, −1, e^{iθ}, e^{−iθ}}.  Returns (Q, θ, err) where err is the
    Frobenius distance of QᵀAQ from the exact block form — err > tol means A
    does not have the required spectrum.
    """
    w, V = np.linalg.eig(A)

    idx1 = int(np.argmin(np.abs(w - 1.0)))
    idxm1 = int(np.argmin(np.abs(w + 1.0)))

    v1 = V[:, idx1].real
    v1 /= np.linalg.norm(v1)
    v2 = V[:, idxm1].real
    v2 -= np.dot(v1, v2) * v1
    v2 /= np.linalg.norm(v2)

    # Orthonormal basis of the complementary (rotation) plane.
    P = np.outer(v1, v1) + np.outer(v2, v2)
    rest = [e - P @ e for e in np.eye(4)]
    rest = [v for v in rest if np.linalg.norm(v) > 1e-8]
    Q34, _ = np.linalg.qr(np.stack(rest, axis=1))
    v3, v4 = Q34[:, 0], Q34[:, 1]

    theta = float(np.arctan2(v4 @ A @ v3, v3 @ A @ v3))
    Q = np.column_stack([v1, v2, v3, v4])

    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    target = np.block(
        [[np.diag([1.0, -1.0]), np.zeros((2, 2))], [np.zeros((2, 2)), R]]
    )
    err = float(np.linalg.norm(Q.T @ A @ Q - target, ord="fro"))
    return Q, theta, err
