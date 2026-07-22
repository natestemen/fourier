"""Construction and structure of A-matrices.

The A-matrix A(λ) of a Young diagram λ with k addable cells is the k×k
orthogonal matrix whose entries encode the transition amplitudes of one step
of the symmetric-group branching rule between the irrep λ and its neighbours.

Convention (this module is the single source of truth for it):

- Row i ↔ addable cell aᵢ, ordered by content, descending.
- Column 0 is the "constant" column:  A[i,0] = √( f(λ+aᵢ) / (m·f(λ)) )
- Column j ≥ 1 ↔ removable cell rⱼ (content descending):
      A[i,j] = √( (m−1)·f(λ+aᵢ)·f(λ−rⱼ) / (m·f(λ)²) ) / (c(aᵢ) − c(rⱼ))
  where f(·) counts standard Young tableaux, m = |λ|+1, and c is the cell
  content x − y.

Structural facts (all verified in tests/test_amatrix.py):

- A(λ) is orthogonal; for k = 4 its eigenvalues are {+1, −1, e^{iθ}, e^{−iθ}},
  so det A = −1.
- Addable and removable contents strictly interlace on the integer line:
  ac[0] > rc[0] > ac[1] > rc[1] > … > ac[k−1].
- Cauchy factorization:  A[:,1:] = diag(α)·C·diag(β)  with
  C[i,j] = 1/(ac[i] − rc[j]) a Cauchy matrix and α = A[:,0].
- Displacement rank 1:  diag(ac)·C − C·diag(rc) = 1·1ᵀ, which is what makes
  the fast (O(k log² k)) mat-vec of `CauchyForm.matvec_fast` possible.
- For the staircase partition (k, k−1, …, 1) the Cauchy core is Toeplitz,
  so the mat-vec drops to O(k log k) via FFT (`CauchyForm.matvec_toeplitz`).
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import sympy as sp
from yungdiagram import Cell, YoungDiagram

__all__ = [
    "a_matrix",
    "a_matrix_symbolic",
    "a_matrix_from_contents",
    "a_matrix_generic4",
    "a_matrix_generic8",
    "addable_contents",
    "removable_contents",
    "staircase_a_matrix",
    "random_content_a_matrix",
    "CauchyForm",
    "cauchy_form",
]


def addable_contents(diagram: YoungDiagram) -> list[int]:
    """Contents of the addable cells, descending (the A-matrix row order)."""
    return sorted((c.content for c in diagram.addable_cells()), reverse=True)


def removable_contents(diagram: YoungDiagram) -> list[int]:
    """Contents of the removable cells, descending (A-matrix column order)."""
    return sorted((c.content for c in diagram.removable_cells()), reverse=True)


def a_matrix(diagram: YoungDiagram) -> npt.NDArray[np.float64]:
    """The k×k orthogonal A-matrix of `diagram` (k = number of addable cells)."""
    addable = diagram.addable_cells()
    removable = diagram.removable_cells()
    removable.insert(0, Cell(-1, -1))  # dummy cell for the constant column

    m = diagram.size + 1
    f = diagram.number_of_standard_tableaux()

    A = np.zeros((len(addable), len(removable)))
    for i, remove in enumerate(removable):
        for j, add in enumerate(addable):
            f_add = (diagram + add).number_of_standard_tableaux()
            if i == 0:
                A[j, i] = np.sqrt(f_add / (m * f))
            else:
                f_rem = (diagram - remove).number_of_standard_tableaux()
                A[j, i] = np.sqrt((m - 1) * f_add * f_rem / (m * f**2)) / (
                    add.content - remove.content
                )

    return A


def a_matrix_symbolic(diagram: YoungDiagram | list[int]) -> sp.Matrix:
    """Exact (sympy) A-matrix — same convention and entries as `a_matrix`,
    but with entries like √(5/12) kept symbolic."""
    if not isinstance(diagram, YoungDiagram):
        diagram = YoungDiagram(diagram)

    addable = diagram.addable_cells()
    removable = diagram.removable_cells()

    m = sp.Integer(diagram.size + 1)
    f = sp.Integer(diagram.number_of_standard_tableaux())

    A = sp.zeros(len(addable), len(removable) + 1)
    for i, add in enumerate(addable):
        f_add = sp.Integer((diagram + add).number_of_standard_tableaux())
        A[i, 0] = sp.sqrt(f_add / (m * f))
        for j, remove in enumerate(removable, start=1):
            f_rem = sp.Integer((diagram - remove).number_of_standard_tableaux())
            A[i, j] = sp.sqrt((m - 1) * f_add * f_rem / (m * f**2)) / (
                add.content - remove.content
            )

    return A


def a_matrix_from_contents(
    ac: npt.NDArray[np.float64], rc: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """The A-matrix determined by (possibly non-integer) interlaced contents
    ac[0] > rc[0] > ac[1] > … > ac[k−1].

    An A-matrix is fixed entirely by its content sequence:
        α_i² = ∏ⱼ(rc[j] − ac[i]) / ∏_{j≠i}(ac[j] − ac[i])
        β_j  = 1 / √( Σᵢ α_i² / (ac[i] − rc[j])² )
        A[i,0] = α_i,   A[i,j+1] = α_i·β_j / (ac[i] − rc[j])

    For the integer contents of a genuine diagram this reproduces
    `a_matrix(diagram)` exactly; for real-valued contents it extends the
    family continuously (used to test whether an arbitrary orthogonal block
    is an A-matrix of *any* content sequence).  If interlacing is violated
    the α² values are clamped at 0 and the result is not orthogonal.
    """
    ac = np.asarray(ac, dtype=float)
    rc = np.asarray(rc, dtype=float)
    k = len(ac)

    alpha2 = np.array(
        [
            np.prod(rc - ac[i]) / np.prod(np.delete(ac, i) - ac[i])
            for i in range(k)
        ]
    )
    alpha = np.sqrt(np.maximum(alpha2, 0.0))

    A = np.zeros((k, k))
    A[:, 0] = alpha
    for j in range(k - 1):
        diffs = ac - rc[j]
        col_norm2 = np.sum(alpha2 / diffs**2)
        if col_norm2 < 1e-30:
            return A  # degenerate contents; caller sees a large residual
        A[:, j + 1] = alpha / (diffs * np.sqrt(col_norm2))
    return A


def _a_matrix_from_contents_log(
    ac: npt.NDArray[np.float64], rc: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """`a_matrix_from_contents` for strictly interlaced contents of any size:
    α is computed in log space (the products overflow float64 beyond k ≈ 200)
    and β is recovered from column normalization."""
    k = len(ac)
    log_alpha2 = np.array(
        [
            np.sum(np.log(np.abs(rc - ac[i])))
            - np.sum(np.log(np.abs(np.delete(ac, i) - ac[i])))
            for i in range(k)
        ]
    )
    alpha = np.exp(0.5 * (log_alpha2 - log_alpha2.max()))
    alpha /= np.linalg.norm(alpha)

    A = np.zeros((k, k))
    A[:, 0] = alpha
    for j in range(k - 1):
        col = alpha / (ac - rc[j])
        A[:, j + 1] = col / np.linalg.norm(col)
    return A


def staircase_a_matrix(k: int) -> npt.NDArray[np.float64]:
    """The k×k staircase A-matrix — A(λ) for λ = (k−1, k−2, …, 1) — for any k.

    Built directly from the staircase's arithmetic contents (ac = m−2i,
    rc = m−1−2j with m = k−1) in log space, so large k is cheap where the
    tableaux-count route of `a_matrix` overflows.  Agrees with
    `a_matrix(staircase(k−1))` to machine precision."""
    m = k - 1
    ac = np.array([m - 2.0 * i for i in range(k)])
    rc = np.array([m - 1.0 - 2.0 * j for j in range(k - 1)])
    return _a_matrix_from_contents_log(ac, rc)


def random_content_a_matrix(k: int, rng: np.random.Generator) -> npt.NDArray[np.float64]:
    """A k×k A-matrix with random interlaced integer contents (gaps drawn
    from {1, 2, 3}) — a stand-in for the A-matrix of a generic large diagram
    in scaling experiments."""
    gaps = rng.integers(1, 4, size=2 * k - 2)
    seq = np.concatenate([[0.0], np.cumsum(gaps)])[::-1]
    return _a_matrix_from_contents_log(seq[0::2], seq[1::2])


# ── generic symbolic A-matrices, parametrized by block widths/heights ─────────
#
# A k-addable diagram is a staircase of k−1 rectangular blocks with widths
# w_1..w_{k−1} and heights h_1..h_{k−1} (all ≥ 1).  These builders express
# A(λ) as a function of those symbols, which is how the Weyl-chamber facts
# were proved for the whole family at once rather than diagram-by-diagram.


def _content(cell: tuple[sp.Expr, sp.Expr]) -> sp.Expr:
    x, y = cell
    return x - y


def _ratio_addable(a, addable, removable) -> sp.Expr:
    """f(λ+a) / (m·f(λ)) expressed through contents only (hook-walk identity)."""
    ca = _content(a)
    num = sp.prod(_content(r) - ca for r in removable)
    den = sp.prod(_content(x) - ca for x in addable if x != a)
    return num / den


def _ratio_removable(r, addable_r, removable_r) -> sp.Expr:
    """(m−1)·f(λ−r) / (m·f(λ)) through contents of λ−r (hook-walk identity)."""
    cr = _content(r)
    num = sp.prod(_content(x) - cr for x in addable_r if x != r)
    den = sp.prod(_content(x) - cr for x in removable_r)
    return num / den


def _generic_a_matrix(w: tuple[sp.Symbol, ...], h: tuple[sp.Symbol, ...]) -> sp.Matrix:
    """Symbolic A-matrix for the diagram with block widths w and heights h.

    Assumes every wᵢ ≥ 2 and hᵢ ≥ 2 so that removing a corner never merges
    two blocks or deletes a row (the generic stratum of the family).
    """
    n = len(w)  # number of blocks; the diagram has n+1 addable cells

    h_prefix = [sp.Integer(0)]
    for hi in h:
        h_prefix.append(h_prefix[-1] + hi)
    w_suffix = [sp.Integer(0)] * (n + 1)
    for idx in range(n - 1, -1, -1):
        w_suffix[idx] = w_suffix[idx + 1] + w[idx]

    addable = [(w_suffix[j], h_prefix[j]) for j in range(n)]
    addable.append((sp.Integer(0), h_prefix[n]))
    removable = [(w_suffix[j] - 1, h_prefix[j + 1] - 1) for j in range(n)]

    addable_ratios = [_ratio_addable(a, addable, removable) for a in addable]

    removable_ratios = []
    for j, rj in enumerate(removable):
        addable_rj = addable[: j + 1] + [rj] + addable[j + 1 :]
        xj, yj = rj
        removable_rj = removable[:j] + [(xj, yj - 1), (xj - 1, yj)] + removable[j + 1 :]
        removable_ratios.append(_ratio_removable(rj, addable_rj, removable_rj))

    A = sp.zeros(n + 1, n + 1)
    for i, a in enumerate(addable):
        A[i, 0] = sp.sqrt(addable_ratios[i])
        for j, r in enumerate(removable, start=1):
            A[i, j] = sp.sqrt(addable_ratios[i] * removable_ratios[j - 1]) / (
                _content(a) - _content(r)
            )
    return A


def a_matrix_generic4() -> tuple[sp.Matrix, tuple[sp.Symbol, ...]]:
    """Symbolic 4×4 A-matrix of the generic 4-addable diagram, with its
    (w_1, w_2, w_3, h_1, h_2, h_3) block symbols."""
    w = sp.symbols("w_1:4", integer=True, positive=True)
    h = sp.symbols("h_1:4", integer=True, positive=True)
    return _generic_a_matrix(w, h), (*w, *h)


def a_matrix_generic8() -> tuple[sp.Matrix, tuple[sp.Symbol, ...]]:
    """Symbolic 8×8 A-matrix of the generic 8-addable diagram, with its
    (w_1..w_7, h_1..h_7) block symbols."""
    w = sp.symbols("w_1:8", integer=True, positive=True)
    h = sp.symbols("h_1:8", integer=True, positive=True)
    return _generic_a_matrix(w, h), (*w, *h)


# ── Cauchy structure ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CauchyForm:
    """The factorization A[:,1:] = diag(alpha)·C·diag(beta) of an A-matrix,
    where C[i,j] = 1/(ac[i] − rc[j]).  Produced by `cauchy_form`."""

    ac: npt.NDArray[np.float64]  # addable contents, descending
    rc: npt.NDArray[np.float64]  # removable contents, descending
    alpha: npt.NDArray[np.float64]  # row scaling = constant column of A
    beta: npt.NDArray[np.float64]  # column scaling

    @property
    def core(self) -> npt.NDArray[np.float64]:
        """The Cauchy core C[i,j] = 1/(ac[i] − rc[j])."""
        return 1.0 / (self.ac[:, None] - self.rc[None, :])

    @property
    def displacement(self) -> npt.NDArray[np.float64]:
        """diag(ac)·C − C·diag(rc); equals the all-ones matrix (rank 1)."""
        C = self.core
        return np.diag(self.ac) @ C - C @ np.diag(self.rc)

    def matrix(self) -> npt.NDArray[np.float64]:
        """Reassemble the full A-matrix from the factorization."""
        A = np.empty((len(self.ac), len(self.rc) + 1))
        A[:, 0] = self.alpha
        A[:, 1:] = self.alpha[:, None] * self.core * self.beta[None, :]
        return A

    def matvec(self, v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """A @ v via the factorization (O(k²), no polynomial arithmetic)."""
        w = self.beta * v[1:]
        return self.alpha * (v[0] + self.core @ w)

    def matvec_fast(self, v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """A @ v in O(k log² k) arithmetic via partial fractions.

        Σⱼ wⱼ/(a − rⱼ) is P(a)/Q(a) with Q(t) = ∏ⱼ(t − rⱼ) and
        P(t) = Σⱼ wⱼ·Q(t)/(t − rⱼ); building P, Q and evaluating them at all
        addable contents is a multipoint-evaluation problem (Borodin–Moenck).
        This implementation demonstrates correctness of the representation;
        it uses numpy poly ops, so its practical constant is not tuned.
        """
        w = self.beta * v[1:]

        Q = np.array([1.0])
        for r in self.rc:
            Q = np.polymul(Q, np.array([1.0, -r]))

        P = np.zeros(len(self.rc))
        for j, r in enumerate(self.rc):
            Qj, _ = np.polydiv(Q, np.array([1.0, -r]))
            P = P + w[j] * Qj

        fvals = np.array([np.polyval(P, a) / np.polyval(Q, a) for a in self.ac])
        return self.alpha * (v[0] + fvals)

    def matvec_toeplitz(self, v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """A @ v in O(k log k) via FFT — valid only when the core is Toeplitz,
        i.e. for the staircase partition where C[i,j] = 1/(2(j−i)+1)."""
        C = self.core
        first_row, first_col = C[0, :], C[:, 0]

        w = self.beta * v[1:]
        # Standard circulant embedding of the (k × k−1) Toeplitz core.
        circ = np.concatenate([first_col, first_row[-1:0:-1]])
        w_padded = np.concatenate([w, np.zeros(len(first_col) - 1)])
        conv = np.fft.ifft(np.fft.fft(circ) * np.fft.fft(w_padded)).real
        return self.alpha * (v[0] + conv[: len(first_col)])


def cauchy_form(diagram: YoungDiagram) -> CauchyForm:
    """Compute the Cauchy factorization of A(diagram).

    alpha is read off the constant column and beta from the first row:
    β_j = A[0, j+1]·(ac[0] − rc[j]) / α_0.
    """
    A = a_matrix(diagram)
    ac = np.array(addable_contents(diagram), dtype=float)
    rc = np.array(removable_contents(diagram), dtype=float)
    alpha = A[:, 0].copy()
    beta = A[0, 1:] * (ac[0] - rc) / alpha[0]
    return CauchyForm(ac=ac, rc=rc, alpha=alpha, beta=beta)
