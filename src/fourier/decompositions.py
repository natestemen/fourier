"""Factorizations of orthogonal matrices into two-level rotations.

Three decompositions, in increasing order of structure exploited:

- `givens_factor`: the generic triangularization into k(k−1)/2 plane
  rotations — the O(k²) baseline every other method is measured against.
- `cs_factor`: the cosine–sine decomposition A = (U₁⊕U₂)·CS(θ)·(V₁ᵀ⊕V₂ᵀ),
  which matches Givens in gate count but has O(k) parallel depth, and is the
  entry point of the (open) O(k log k) recursion.
- `wei_di_fit`: the provably optimal 2-qubit synthesis of O(4) matrices —
  3 CNOT + 6 Ry for det = −1 (Wei & Di, arXiv:1203.0722), which covers every
  4-addable A-matrix.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.linalg import cossin
from scipy.optimize import differential_evolution, minimize

__all__ = [
    "Givens",
    "givens_factor",
    "givens_reconstruct",
    "CSDecomposition",
    "cs_factor",
    "cs_butterfly",
    "parallel_depth",
    "WeiDi",
    "wei_di_fit",
]


# ── Givens triangularization ───────────────────────────────────────────────────


@dataclass(frozen=True)
class Givens:
    """Rotation by θ in the (i, j) coordinate plane: acts as
    [[cos θ, −sin θ], [sin θ, cos θ]] on coordinates (i, j), identity elsewhere."""

    i: int
    j: int
    theta: float

    def embedded(self, k: int) -> npt.NDArray[np.float64]:
        """The full k×k matrix of this rotation."""
        M = np.eye(k)
        c, s = np.cos(self.theta), np.sin(self.theta)
        M[self.i, self.i] = c
        M[self.i, self.j] = -s
        M[self.j, self.i] = s
        M[self.j, self.j] = c
        return M

    def __repr__(self) -> str:
        return f"G({self.i}, {self.j}, {np.degrees(self.theta):+.4f}°)"


def givens_factor(
    A: npt.NDArray[np.float64],
) -> tuple[list[Givens], npt.NDArray[np.float64]]:
    """Factor an orthogonal A as  A = Gₙ · … · G₁ · diag(signs).

    Column-by-column reduction: for each column j, the sub-diagonal entries
    are zeroed by left rotations; orthogonality forces the remainder to
    diag(±1), returned as `signs`.  Uses ≤ k(k−1)/2 rotations.  Rotations by
    an exact multiple of 2π are dropped.
    """
    k = A.shape[0]
    M = A.copy()
    ops: list[Givens] = []  # left-multiplications driving M → diag(signs)

    for j in range(k - 1):
        for i in range(j + 1, k):
            if abs(M[i, j]) < 1e-12:
                continue
            theta = np.arctan2(M[i, j], M[j, j])
            c, s = np.cos(theta), np.sin(theta)
            row_j, row_i = M[j, :].copy(), M[i, :].copy()
            M[j, :] = c * row_j + s * row_i
            M[i, :] = -s * row_j + c * row_i
            ops.append(Givens(j, i, theta))

    signs = np.sign(np.diag(M))

    # Each op above is G(θ)ᵀ applied on the left; inverting the product turns
    # the list around without negating angles (Gᵀ(θ) = G(−θ) and we recorded θ
    # as the *zeroing* angle), so A = ops[-1]·…·ops[0]·diag(signs) reversed:
    gates = [g for g in reversed(ops) if abs(np.sin(g.theta)) > 1e-12 or np.cos(g.theta) < 0]
    return gates, signs


def givens_reconstruct(
    k: int, gates: list[Givens], signs: npt.NDArray[np.float64] | None = None
) -> npt.NDArray[np.float64]:
    """Multiply out  gates[0] applied first:  G_last · … · G_first · diag(signs)."""
    M = np.eye(k) if signs is None else np.diag(signs).astype(float)
    for g in gates:
        M = g.embedded(k) @ M
    return M


# ── cosine–sine decomposition ──────────────────────────────────────────────────


@dataclass(frozen=True)
class CSDecomposition:
    """A = (U₁ ⊕ U₂) · D(θ) · (V₁ᵀ ⊕ V₂ᵀ) where U₁, V₁ are p×p, U₂, V₂ are
    q×q (q = k−p ≥ p), and D(θ) rotates coordinate plane (i, q+i) by θᵢ for
    each of the p angles, fixing coordinates p…q−1 (LAPACK's convention).
    For p = q this is the familiar [[C, −S], [S, C]] block form.  Produced by
    `cs_factor`."""

    u1: npt.NDArray[np.float64]
    u2: npt.NDArray[np.float64]
    v1t: npt.NDArray[np.float64]
    v2t: npt.NDArray[np.float64]
    thetas: npt.NDArray[np.float64]

    @property
    def p(self) -> int:
        return self.u1.shape[0]

    @property
    def q(self) -> int:
        return self.u2.shape[0]

    def cs_block(self) -> npt.NDArray[np.float64]:
        """The middle factor D(θ)."""
        p, q = self.p, self.q
        c, s = np.cos(self.thetas), np.sin(self.thetas)
        D = np.eye(p + q)
        for i in range(p):
            D[i, i] = c[i]
            D[i, q + i] = -s[i]
            D[q + i, i] = s[i]
            D[q + i, q + i] = c[i]
        return D

    def matrix(self) -> npt.NDArray[np.float64]:
        """Reassemble the full matrix from the three factors."""
        p, q = self.p, self.q
        U = np.block([[self.u1, np.zeros((p, q))], [np.zeros((q, p)), self.u2]])
        Vt = np.block([[self.v1t, np.zeros((p, q))], [np.zeros((q, p)), self.v2t]])
        return U @ self.cs_block() @ Vt


def cs_factor(A: npt.NDArray[np.float64], p: int | None = None) -> CSDecomposition:
    """CS decomposition of an orthogonal k×k matrix, with square diagonal
    blocks of sizes p and k−p (default p = k//2; p must be ≤ k−p)."""
    k = A.shape[0]
    if p is None:
        p = k // 2
    if p > k - p:
        raise ValueError("cs_factor requires p <= k - p; transpose A to swap roles")
    (u1, u2), thetas, (v1t, v2t) = cossin(A, p=p, q=p, separate=True)
    return CSDecomposition(u1=u1, u2=u2, v1t=v1t, v2t=v2t, thetas=np.atleast_1d(thetas))


def cs_butterfly(
    A: npt.NDArray[np.float64],
) -> tuple[list[Givens], npt.NDArray[np.float64]]:
    """Factor an orthogonal A as  A = Gₙ · … · G₁ · diag(signs)  by recursive
    CS decomposition instead of triangularization.

    Each level splits the matrix in half: a layer of k/2 *disjoint* rotations
    (the CS angles) between recursive factorizations of the four sub-blocks.
    The total rotation count matches Givens at O(k²), but disjointness gives
    O(k) parallel depth instead of O(k²) — see `parallel_depth`.  Odd sizes
    fall back to plain Givens for that block.
    """
    ops: list[Givens | int] = []  # application order; an int is a sign flip
    _cs_butterfly_rec(A, list(range(A.shape[0])), ops)

    # Push sign flips to the right end of the product (= applied first):
    # Sᵢ·G(i,j,θ) = G(i,j,−θ)·Sᵢ, so a flip passing a rotation on exactly one
    # of its coordinates negates the angle.  Walking the product left-to-right
    # (reverse application order), each rotation is conjugated by the flips
    # dragged past it so far.
    signs = np.ones(A.shape[0])
    gates: list[Givens] = []
    for op in reversed(ops):
        if isinstance(op, int):
            signs[op] *= -1
        elif signs[op.i] * signs[op.j] < 0:
            gates.append(Givens(op.i, op.j, -op.theta))
        else:
            gates.append(op)
    gates.reverse()
    return gates, signs


def _cs_butterfly_rec(
    A: npt.NDArray[np.float64], idx: list[int], ops: list[Givens | int]
) -> None:
    k = A.shape[0]

    if k == 1:
        if A[0, 0] < 0:
            ops.append(idx[0])
        return

    if k == 2 or k % 2 != 0:
        gates, signs = givens_factor(A)
        ops.extend(idx[i] for i, s in enumerate(signs) if s < 0)
        ops.extend(Givens(idx[g.i], idx[g.j], g.theta) for g in gates)
        return

    cs = cs_factor(A)
    top, bot = idx[: cs.p], idx[cs.p :]

    _cs_butterfly_rec(cs.v1t, top, ops)
    _cs_butterfly_rec(cs.v2t, bot, ops)
    for i, theta in enumerate(cs.thetas):
        if abs(np.sin(theta)) > 1e-12:
            ops.append(Givens(top[i], bot[i], theta))
    _cs_butterfly_rec(cs.u1, top, ops)
    _cs_butterfly_rec(cs.u2, bot, ops)


def parallel_depth(gates: list[Givens]) -> int:
    """Circuit depth of the rotation sequence on all-to-all hardware:
    the longest path in the dependency DAG, where two rotations must be
    sequential iff they share a coordinate (disjoint plane rotations commute).

    For the column-major `givens_factor` output this is ≈ 2k; for
    `cs_butterfly` it is ≈ k — the butterfly's advantage is a constant
    factor in depth (the O(k log k) *gate count* remains open)."""
    last_layer: dict[int, int] = {}
    depth = 0
    for g in gates:
        d = max(last_layer.get(g.i, 0), last_layer.get(g.j, 0)) + 1
        last_layer[g.i] = last_layer[g.j] = d
        depth = max(depth, d)
    return depth


# ── Wei–Di optimal 2-qubit synthesis ───────────────────────────────────────────

# CNOT with control on qubit 0 / on qubit 1 (basis order |q1 q0⟩ row-major).
_CNOT = {
    0: np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=float),
    1: np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=float),
}


def _ry(theta: float) -> npt.NDArray[np.float64]:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]])


@dataclass(frozen=True)
class WeiDi:
    """Parameters of the Wei–Di circuit
        X = [Ry(θ₁)⊗Ry(θ₂)] · CNOT · [Ry(b)⊗Ry(a)] · CNOT · [Ry(θ₃)⊗Ry(θ₄)] [· CNOT]
    (the trailing CNOT only when det X = −1).  Gate cost: 6 Ry plus 2 CNOTs
    for det = +1, 3 CNOTs for det = −1 — both optimal.  Produced by `wei_di_fit`."""

    params: npt.NDArray[np.float64]  # (θ₁, θ₂, a, b, θ₃, θ₄)
    control: int  # kron factor controlling the CNOTs (0 = first/leftmost factor)
    det_fix: bool  # True ⇔ det = −1 ⇔ third CNOT present
    residual: float  # squared Frobenius error of the fit

    @property
    def n_cnots(self) -> int:
        return 3 if self.det_fix else 2

    def matrix(self) -> npt.NDArray[np.float64]:
        t1, t2, a, b, t3, t4 = self.params
        cnot = _CNOT[self.control]
        M = (
            np.kron(_ry(t1), _ry(t2))
            @ cnot
            @ np.kron(_ry(b), _ry(a))
            @ cnot
            @ np.kron(_ry(t3), _ry(t4))
        )
        return M @ cnot if self.det_fix else M


def _wei_di_loss(params, target, cnot, det_fix) -> float:
    t1, t2, a, b, t3, t4 = params
    M = (
        np.kron(_ry(t1), _ry(t2))
        @ cnot
        @ np.kron(_ry(b), _ry(a))
        @ cnot
        @ np.kron(_ry(t3), _ry(t4))
    )
    if det_fix:
        M = M @ cnot
    diff = target - M
    return float(np.sum(diff * diff))


def wei_di_fit(X: npt.NDArray[np.float64], restarts: int = 40, seed: int = 0) -> WeiDi:
    """Numerically extract Wei–Di parameters for X ∈ O(4).

    Global search (differential evolution) plus `restarts` local refinements,
    over both CNOT orientations.  A residual below ~1e-15 means the circuit
    reproduces X to machine precision; every 4-addable A-matrix tested to date
    does (report.md, Finding 5).
    """
    rng = np.random.default_rng(seed)
    det_fix = bool(np.linalg.det(X) < 0)
    bounds = [(-np.pi, np.pi)] * 6

    best: tuple[float, npt.NDArray[np.float64], int] | None = None
    for control, cnot in _CNOT.items():
        de = differential_evolution(
            _wei_di_loss,
            bounds,
            args=(X, cnot, det_fix),
            seed=int(rng.integers(1 << 30)),
            maxiter=300,
            tol=1e-12,
            popsize=8,
            mutation=(0.5, 1.5),
            recombination=0.7,
        )
        if best is None or de.fun < best[0]:
            best = (de.fun, de.x, control)

        for _ in range(restarts):
            res = minimize(
                _wei_di_loss,
                rng.uniform(-np.pi, np.pi, 6),
                args=(X, cnot, det_fix),
                method="L-BFGS-B",
                options={"maxiter": 2000, "ftol": 1e-20, "gtol": 1e-12},
            )
            if res.fun < best[0]:
                best = (res.fun, res.x, control)

    loss, params, control = best
    polish = minimize(
        _wei_di_loss,
        params,
        args=(X, _CNOT[control], det_fix),
        method="L-BFGS-B",
        options={"maxiter": 5000, "ftol": 1e-24, "gtol": 1e-14},
    )
    return WeiDi(
        params=polish.x, control=control, det_fix=det_fix, residual=float(polish.fun)
    )
