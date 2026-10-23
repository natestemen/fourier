r"""Independent direct-column/amplification audit for symmetric-group induction.

This is a small dense algebra verifier, NOT an efficient circuit compiler. It
does not import the streaming/Mackey recursion or any A-matrix implementation.
All phases and all auxiliary output coordinates are compared, not merely the
postselected success block.

Representation-theoretic source: Bruinsma--Grinko--Ozols,
https://arxiv.org/html/2608.28569v2, Definition 3.6 and Eq. (19) for the
induction entries, Lemma 2.4 for dimension ratios, and Appendix A.2.3 for
the cached Young-generator implementation. The direct erasure/amplification
combination below is a research construction and is not attributed to that
paper. Numerical tests are not a proof of a uniform finite-gate bound.

Construction and equal-dimensional promises
-----------------------------------------
Fix an external, unchanged partition lambda of M-1. The source basis (i,P),
1<=i<=M, P in SYT(lambda), and target basis (mu,Q), mu covering lambda, both
have N=M*d_lambda elements. Use the SAME physical registers throughout:
attenuation flag, arithmetic scratch, transversal index, and M tableau slots.
Initially the final tableau slot is star; finally the transversal register is
zero. The dense path alphabet below is the disjoint union of these star words
and full target tableaux, not a claim that their exponential enumeration is
part of a circuit. Source and target embeddings I,J have equal rank N.

B coherently prepares star -> sum_mu alpha_mu |mu\lambda>, where
alpha_mu^2=d_mu/(M*d_lambda), preserving P and lambda. V selects R_mu(t_i),
t_i=(i,...,M). H_M maps |0> to the uniform index state. Before attenuation,
W_0=(H_M^dagger tensor I) V B satisfies J^dagger W_0 I=U_Ind/sqrt(M).
The construction computes the index controls before erasing that register.

Choose public r=ceil(sqrt(M)), theta=pi/(4r+2), mu=csc(theta). Since mu>=sqrt(M),
a flag rotation with zero-amplitude sqrt(M)/mu gives a unitary W with
J^dagger W I=U_Ind/mu. Set R_I=2II^dagger-I and R_J=2JJ^dagger-I. Then

    (-W R_I W^dagger R_J)^r W I = J U_Ind.

Both reflections test ALL zero work, including the attenuation flag and
arithmetic scratch. They also test their respective valid tableau/index
promises. Validity need not scan every prefix: sort (row,label) records to
compute each label's within-row column, then use sorted joins to check that
each cell below the first row has an earlier-labeled cell immediately above.
Rows are already increasing by their within-row construction; the above-cell
checks establish column increase and the Young shape. A row histogram checks
the endpoint. Thus validity uses reversible fixed scans/sorts of O(M) words.
The minus sign in every iterate is required for coherent relative phases.
Arbitrary unitary extensions away from I do not matter: W R_I W^dagger is the
reflection through im(WI). If ||W_tilde I-WI||<=eta, the full clean-output
error is at most (4r+1)*eta, because the two image projectors differ by <=2eta.
This bound does NOT assume that the full unitary extensions are close.

Gate accounting, separate from this dense verifier
-------------------------------------------------
For k addable corners, k(k-1)/2<=M-1, so k=O(sqrt(M)). Extract the integer
corner contents by a fixed scan/sort. The residue expression for every
alpha_mu^2 has O(k) factors; computing all k coefficients by logarithmic
fixed-precision arithmetic costs O(k^2 polylog(M/eta))=O(M polylog(M/eta)).
Integer interlacing bounds alpha_mu below by an inverse polynomial, so O(log M
+ log(1/eta)) working precision, with ordinary guard bits, suffices. Controlled
state preparation and its inverse include erasure of coefficients/angles.

Build the tableau-column cache by sorting (row,label) records, computing each
within-row rank, and restoring label order: O(M polylog M) gates. Adjacent
Young generators update both row and column records, cost polylog(M/eta)
each, and maintain the cache. In descending j=M-1,...,1, apply generator s_j
controlled by i<=j. This implements all canonical t_i with M-1 gates, NOT
sum_i(M-i) gates. Rebuild/uncompute the cache from the final tableau. Thus W
costs O(M polylog(M/eta)), including coherent arithmetic and scratch cleanup;
the full reflections have the same bound. An induction costs
O(M^(3/2) polylog(M/eta)), and summing outer M gives O(n^(5/2) polylog(n/eps)).
Common eta=eps/O(n^(3/2)) suffices. This matches, rather than beats, exponent
5/2 and uses O(n polylog(n/eps)) workspace under straightforward reversible
implementations. No QRAM, postselection, or precomputed diagram oracle occurs.

After amplification, once per OUTER induction, compute mu from the resulting
tableau Q by the same row histogram, compute the unique added row a=mu\lambda,
and append a to the retained right path. Reconstruct lambda=mu-box(a) and erase
the old external lambda register; promote mu to the endpoint control. This
costs O(M polylog M), not an inner-loop charge, and is included in the bound.
The endpoint is not inferred from Q's last symbol: Q may have a different
penultimate shape. Approximation errors compose using ideal valid prefixes.

Two exact architecture obstructions
----------------------------------
(1) In any continuous RIGHT-adjacent-transposition walk visiting all left
cosets of S_(M-1), labels are t(M). In the initial one-line order, every pair
of tokens must cross: the initially left token eventually occupies slot M.
Each adjacent swap crosses one pair, hence >=binom(M,2) swaps are necessary.
Successive right-end extraction attains equality. This limits transversal
reordering/cancellation, NOT arbitrary circuits for representation matrices.

(2) Suppose normalized orthogonal induction columns retain their tag i and
one fixed normalized tag-erasure bra v is used. The resulting block is
U_Ind diag(conj(v_i)). A scalar block U_Ind/mu requires |v_i|=1/mu, hence
mu>=sqrt(M). Allowing per-column attenuation a_i<=1 still gives
|v_i|>=1/mu and the same bound. Weighted erasure alone cannot improve this
architecture; a better exponent requires a different embedding or a faster
controlled-transversal primitive.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from math import ceil, factorial, pi, sin, sqrt

import numpy as np
from scipy.linalg import null_space

Partition = tuple[int, ...]
Word = tuple[int, ...]


def partitions(n: int, ceiling: int | None = None) -> tuple[Partition, ...]:
    if n < 0:
        raise ValueError("negative size")
    if not n:
        return ((),)
    return tuple((first,) + tail
                 for first in range(min(n, n if ceiling is None else ceiling), 0, -1)
                 for tail in partitions(n - first, first))


def addable_rows(shape: Partition) -> tuple[int, ...]:
    return tuple(r for r in range(1, len(shape) + 2)
                 if r == 1 or shape[r - 2] > (shape[r - 1] if r <= len(shape) else 0))


def add_row(shape: Partition, row: int) -> Partition:
    if row not in addable_rows(shape):
        raise ValueError("invalid added row")
    result = list(shape) + ([0] if row == len(shape) + 1 else [])
    result[row - 1] += 1
    return tuple(result)


@lru_cache(maxsize=None)
def tableaux(shape: Partition) -> tuple[Word, ...]:
    if not shape:
        return ((),)
    result = []
    for r, length in enumerate(shape, 1):
        if length <= (shape[r] if r < len(shape) else 0):
            continue
        smaller = list(shape)
        smaller[r - 1] -= 1
        if not smaller[-1]:
            smaller.pop()
        result.extend(word + (r,) for word in tableaux(tuple(smaller)))
    return tuple(result)


def dimension(shape: Partition) -> int:
    """Exact hook formula, used only for independent small-size references."""
    hooks = 1
    for r, length in enumerate(shape, 1):
        for c in range(1, length + 1):
            hooks *= length - c + sum(other >= c for other in shape[r:]) + 1
    return factorial(sum(shape)) // hooks


def contents(word: Word) -> tuple[int, ...]:
    counts: dict[int, int] = {}
    result = []
    for row in word:
        counts[row] = counts.get(row, 0) + 1
        result.append(counts[row] - row)
    return tuple(result)


def shape_from_word(word: Word) -> Partition:
    """Reference for the local-neighbor validity criterion (no prefix scans).

    Python dictionaries here stand in for the fixed sorted joins described
    above; this routine is not claimed to be a reversible implementation.
    """
    if any(not isinstance(row, int) or not 1 <= row <= len(word) for row in word):
        raise ValueError("invalid row label")
    counts: dict[int, int] = {}
    cells = {}
    for row, label in sorted((row, label) for label, row in enumerate(word, 1)):
        counts[row] = counts.get(row, 0) + 1
        cells[(row, counts[row])] = label
    for (row, column), label in cells.items():
        if row > 1 and cells.get((row - 1, column), len(word) + 1) >= label:
            raise ValueError("word does not encode a standard Young tableau")
    return tuple(counts[r] for r in range(1, len(counts) + 1))


@dataclass(frozen=True)
class BoundaryWitness:
    endpoint: Partition
    added_row: int
    new_right_path: Word
    recovered_old_shape: Partition
    erased_old_shape: tuple[int, ...]


def outer_boundary(shape: Partition, output_word: Word, right_path: Word) -> BoundaryWitness:
    """Explicit endpoint transfer/old-control cleanup, once per induction."""
    if shape_from_word(right_path) != shape:
        raise ValueError("right path endpoint differs from external control")
    endpoint = shape_from_word(output_word)
    added = [r for r in addable_rows(shape) if add_row(shape, r) == endpoint]
    if len(added) != 1:
        raise ValueError("output endpoint does not cover external control")
    row = added[0]
    recovered = list(endpoint)
    recovered[row - 1] -= 1
    if recovered and not recovered[-1]:
        recovered.pop()
    recovered_shape = tuple(recovered)
    width = max(len(shape), len(recovered_shape))
    erased = tuple((shape[r] if r < len(shape) else 0) ^
                   (recovered_shape[r] if r < len(recovered_shape) else 0)
                   for r in range(width))
    return BoundaryWitness(endpoint, row, right_path + (row,), recovered_shape, erased)


@lru_cache(maxsize=None)
def young_generator(shape: Partition, j: int) -> np.ndarray:
    words = tableaux(shape)
    lookup = {word: i for i, word in enumerate(words)}
    if not 1 <= j < sum(shape):
        raise ValueError("generator outside shape")
    matrix = np.zeros((len(words), len(words)))
    for column, word in enumerate(words):
        cs = contents(word)
        axial = cs[j] - cs[j - 1]
        matrix[column, column] = 1 / axial
        swapped = word[:j - 1] + (word[j], word[j - 1]) + word[j + 1:]
        if swapped in lookup and swapped != word:
            matrix[lookup[swapped], column] = sqrt(max(0.0, 1 - axial ** -2))
    return matrix


def representation(shape: Partition, permutation: tuple[int, ...]) -> np.ndarray:
    """Reference via bubble sorting an arbitrary permutation, not induction."""
    if sorted(permutation) != list(range(1, sum(shape) + 1)):
        raise ValueError("invalid permutation")
    current = list(permutation)
    swaps = []
    while current != sorted(current):
        j = next(j for j in range(len(current) - 1) if current[j] > current[j + 1])
        current[j], current[j + 1] = current[j + 1], current[j]
        swaps.append(j + 1)
    matrix = np.eye(dimension(shape))
    for j in reversed(swaps):
        matrix = matrix @ young_generator(shape, j)
    return matrix


def transversal(m: int, i: int) -> tuple[int, ...]:
    if not 1 <= i <= m:
        raise ValueError("invalid transversal")
    return tuple(range(1, i)) + tuple(range(i + 1, m + 1)) + (i,)


def induction_bases(shape: Partition):
    m = sum(shape) + 1
    inputs = tuple((i, p) for i in range(1, m + 1) for p in tableaux(shape))
    outputs = tuple((add_row(shape, r), q) for r in addable_rows(shape)
                    for q in tableaux(add_row(shape, r)))
    assert len(inputs) == len(outputs) == m * dimension(shape)
    return inputs, outputs


def induction_reference(shape: Partition) -> np.ndarray:
    inputs, outputs = induction_bases(shape)
    m, d = sum(shape) + 1, dimension(shape)
    result = np.zeros((len(outputs), len(inputs)))
    for row, (mu, q) in enumerate(outputs):
        added = next(r for r in addable_rows(shape) if add_row(shape, r) == mu)
        labels = {word: j for j, word in enumerate(tableaux(mu))}
        for column, (i, p) in enumerate(inputs):
            result[row, column] = sqrt(dimension(mu) / (m * d)) * representation(
                mu, transversal(m, i))[labels[q], labels[p + (added,)]]
    return result


def _first_column_unitary(vector: np.ndarray, rng=None) -> np.ndarray:
    """A small dense completion; never used as a circuit complexity claim."""
    vector = np.asarray(vector, dtype=complex)
    if rng is None:
        first = np.eye(len(vector), dtype=complex)[:, 0]
        delta = first - vector
        if np.linalg.norm(delta) < 1e-14:
            return np.eye(len(vector), dtype=complex)
        return np.eye(len(vector)) - 2 * np.outer(delta, delta.conj()) / np.vdot(delta, delta)
    complement = null_space(vector.conj()[None, :])
    random = rng.normal(size=(len(vector) - 1,) * 2) + 1j * rng.normal(size=(len(vector) - 1,) * 2)
    rotation, _ = np.linalg.qr(random)
    return np.column_stack((vector, complement @ rotation))


@dataclass
class DirectInduction:
    shape: Partition
    steps: int
    normalization: float
    attenuation: float
    unitary: np.ndarray
    source: np.ndarray
    target: np.ndarray
    reference: np.ndarray
    path_dimension: int
    scratch_dimension: int

    @property
    def ambient_dimension(self) -> int:
        return len(self.unitary)

    def amplified_columns(self, unitary=None, source_projector_basis=None) -> np.ndarray:
        """Apply the actual two-reflection iteration, retaining ALL work."""
        w = self.unitary if unitary is None else unitary
        source = self.source if source_projector_basis is None else source_projector_basis
        reflected = w @ source
        state = w @ self.source
        for _ in range(self.steps):
            state = 2 * self.target @ (self.target.conj().T @ state) - state
            state = -(2 * reflected @ (reflected.conj().T @ state) - state)
        return state

    def inverse_columns(self, unitary=None) -> np.ndarray:
        w = self.unitary if unitary is None else unitary
        reflected = w @ self.source
        state = self.target.copy()
        for _ in range(self.steps):
            state = -(2 * reflected @ (reflected.conj().T @ state) - state)
            state = 2 * self.target @ (self.target.conj().T @ state) - state
        return w.conj().T @ state


def direct_induction(shape: Partition, *, steps: int | None = None,
                     completion_seed: int | None = None,
                     scratch_dimension: int = 2) -> DirectInduction:
    m, d = sum(shape) + 1, dimension(shape)
    steps = ceil(sqrt(m)) if steps is None else steps
    if steps < 0 or scratch_dimension < 1:
        raise ValueError("invalid amplification/scratch parameters")
    normalization = 1 / sin(pi / (4 * steps + 2))
    attenuation = sqrt(m) / normalization
    if attenuation > 1 + 1e-14:
        raise ValueError("normalization is smaller than sqrt(M)")
    inputs, outputs = induction_bases(shape)
    n = len(inputs)
    path_dimension = d + n
    words = tableaux(shape)
    output_index = {item: d + j for j, item in enumerate(outputs)}
    branch = np.eye(path_dimension, dtype=complex)
    rng = None if completion_seed is None else np.random.default_rng(completion_seed)
    for p_index, p in enumerate(words):
        indices = [p_index]
        amplitudes = [0.0]
        for r in addable_rows(shape):
            mu = add_row(shape, r)
            indices.append(output_index[(mu, p + (r,))])
            amplitudes.append(sqrt(dimension(mu) / (m * d)))
        branch[np.ix_(indices, indices)] = _first_column_unitary(np.array(amplitudes), rng)
    select = np.eye(m * path_dimension, dtype=complex)
    # Actual descending threshold-controlled Young-generator construction.
    for i in range(1, m + 1):
        path_action = np.eye(path_dimension)
        for j in range(m - 1, 0, -1):
            if i > j:
                continue
            gate = np.eye(path_dimension)
            for r in addable_rows(shape):
                mu = add_row(shape, r)
                indices = [output_index[(mu, q)] for q in tableaux(mu)]
                gate[np.ix_(indices, indices)] = young_generator(mu, j)
            path_action = gate @ path_action
        sl = slice((i - 1) * path_dimension, i * path_dimension)
        select[sl, sl] = path_action
    # Only its first column is specified; every other column is arbitrary.
    uniform = _first_column_unitary(np.ones(m) / sqrt(m), rng)
    base = np.kron(uniform.conj().T, np.eye(path_dimension)) @ select @ np.kron(np.eye(m), branch)
    failure = sqrt(max(0.0, 1 - attenuation ** 2))
    flag = np.array([[attenuation, -failure], [failure, attenuation]])
    w = np.kron(flag, np.kron(np.eye(scratch_dimension), base))
    source = np.zeros((len(w), n), dtype=complex)
    target = np.zeros_like(source)
    for column, (i, p) in enumerate(inputs):
        source[(i - 1) * path_dimension + words.index(p), column] = 1
    for column in range(n):
        target[d + column, column] = 1
    return DirectInduction(shape, steps, normalization, attenuation, w, source,
                           target, induction_reference(shape), path_dimension,
                           scratch_dimension)


def arbitrary_completion(probe: DirectInduction, seed: int) -> np.ndarray:
    """Change every unprescribed input column without changing W I."""
    source_indices = set(np.flatnonzero(np.any(probe.source, axis=1)))
    complement = [j for j in range(probe.ambient_dimension) if j not in source_indices]
    rng = np.random.default_rng(seed)
    random = rng.normal(size=(len(complement),) * 2) + 1j * rng.normal(size=(len(complement),) * 2)
    rotation, _ = np.linalg.qr(random)
    change = np.eye(probe.ambient_dimension, dtype=complex)
    change[np.ix_(complement, complement)] = rotation
    return probe.unitary @ change


def fourier_basis(n: int):
    return tuple((shape, q, r) for shape in partitions(n)
                 for q in tableaux(shape) for r in tableaux(shape))


def transversal_words(n: int):
    return tuple(product(*(range(1, m + 1) for m in range(1, n + 1))))


def permutation_from_word(word: tuple[int, ...]) -> tuple[int, ...]:
    result = ()
    for m, i in enumerate(word, 1):
        t = transversal(m, i)
        result = tuple(t[value - 1] for value in result + (m,))
    return result


def fourier_reference(n: int) -> np.ndarray:
    basis = fourier_basis(n)
    words = transversal_words(n)
    result = np.zeros((len(basis), len(words)))
    for column, word in enumerate(words):
        g = permutation_from_word(word)
        for row, (shape, q, r) in enumerate(basis):
            ts = tableaux(shape)
            result[row, column] = sqrt(dimension(shape) / factorial(n)) * representation(
                shape, g)[ts.index(q), ts.index(r)]
    return result


def fourier_by_amplified_induction(n: int) -> np.ndarray:
    """Compose verified clean blocks; ancilla cleanup is tested separately."""
    blocks = {}
    for m in range(1, n + 1):
        for shape in partitions(m - 1):
            probe = direct_induction(shape)
            blocks[shape] = probe.target.conj().T @ probe.amplified_columns()
    basis = fourier_basis(n)
    output_index = {item: j for j, item in enumerate(basis)}
    words = transversal_words(n)
    result = np.zeros((len(basis), len(words)), dtype=complex)
    for column, word in enumerate(words):
        state = {((), (), ()): 1.0 + 0j}
        for i in word:
            updated = {}
            for (shape, p, r), amplitude in state.items():
                inputs, outputs = induction_bases(shape)
                source_column = inputs.index((i, p))
                for row, (mu, q) in enumerate(outputs):
                    boundary = outer_boundary(shape, q, r)
                    assert boundary.endpoint == mu and not any(boundary.erased_old_shape)
                    key = (mu, q, boundary.new_right_path)
                    updated[key] = updated.get(key, 0) + amplitude * blocks[shape][row, source_column]
            state = updated
        for key, amplitude in state.items():
            result[output_index[key], column] = amplitude
    return result


def tight_right_adjacent_walk(m: int) -> tuple[int, ...]:
    """Visit all last-slot labels in exactly binom(M,2) right swaps."""
    if m < 1:
        raise ValueError("M must be positive")
    return tuple(j for i in range(m - 1, 0, -1) for j in range(i, m))


def amplification_resource_skeleton(n: int) -> tuple[int, int]:
    """Return W invocations and linear-word work; polylogs are suppressed."""
    if n < 1:
        raise ValueError("n must be positive")
    calls = sum(2 * ceil(sqrt(m)) + 1 for m in range(1, n + 1))
    work = sum(m * (4 * ceil(sqrt(m)) + 1) for m in range(1, n + 1))
    return calls, work


if __name__ == "__main__":
    for m in range(1, 5):
        errors = []
        for shape in partitions(m - 1):
            probe = direct_induction(shape)
            errors.append(np.linalg.norm(probe.amplified_columns() - probe.target @ probe.reference, 2))
        print(f"M={m}: maximum full-clean induction error {max(errors):.3e}")
    for n in range(1, 6):
        error = np.linalg.norm(fourier_by_amplified_induction(n) - fourier_reference(n), 2)
        print(f"S_{n}: full Fourier matrix error {error:.3e}")
