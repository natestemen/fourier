r"""A streaming Mackey induction map, with independent representation checks.

This is a small-state, dense numerical *verification* program, not an efficient
quantum simulator or a compiled reversible circuit.  The algebraic streaming
schedule uses one compact original-prefix shape, one endpoint-difference row
``token``, and one flag.  It never needs a second evolving endpoint shape.

Conventions follow Bruinsma--Grinko--Ozols, arXiv:2608.28569v2, Definitions
3.6, 5.16, 5.18 and Proposition 5.8.  The transversal is t_i=(i,...,n), with
t_n=e.  Row labels are one based; 0 is the star/inactive symbol.  The A block
has its constant column first, then removable rows in increasing row order.

Invariant immediately before a non-star embedding: the original prefix is
lambda=kappa+p, while the previously induced endpoint is kappa+a.  Equality
of these two shapes is exactly a=p.  On this diagonal subspace erase the
duplicate token a, apply A_lambda to p, and copy its output q back to a.
Off the diagonal the embedding is the one-dimensional identity: q=p and a
does not change.  The equality ancilla erases because its output value is
[a=q].  In particular a is NOT the last output path symbol in general.

The inactive branch is essential.  Initially word=(star,p_1,...,p_{n-1}),
token=star, flag=i.  At level m>1 update the original prefix with p_{m-1}
before any swap.  If token=star, swap the preceding star slot with the new
input symbol and decrement the flag.  Apply the embedding exactly when the
flag is 1.  This includes the star/star diagonal and hence the constant A
column.  At m=1 omit the update, swap, and decrement.  Thus i>n ends with
word=(p,star), flag=i-n+1 and unchanged endpoint, exactly Definition 5.18.

Every operation has a local inverse.  First A_lambda^T acts on the output
diagonal a=q (erase duplicate token, transform q, copy it back).  A recovered
star token identifies the earlier inactive branch; increment its flag and
undo its swap.  Remove the restored original row p from the prefix.  No
record of earlier shapes or branching decisions is retained.

At a complete induction boundary the original-prefix shape equals the
externally supplied original endpoint.  It can be erased against that
control; the token can be erased against the unique box between original
and output endpoints; flag=1 erases for valid inputs.  The numerical cleanup
witness below explicitly verifies these identities, but is not a gate-level
uncomputation implementation.

Proof of the full recursion invariant: after level m the compact prefix is
the ORIGINAL P_{m-1}, and all unprocessed symbols p_m,... are untouched.  If
i>m, the processed word is (p_1,...,p_{m-1},star), token=star, flag=i-m+1.
If i<=m, the processed word is a path q to P_{m-1}+a, token=a, flag=1.
The inactive swap proves the former case.  At i=m the star A column gives
the identity-transversal branch of Definition 3.6.  For i<m the preceding
output endpoint P_{m-2}+a and new original prefix P_{m-2}+p are precisely
the two shapes in Equation (66); their equal-shape block is A and their
unequal-shape block is identity.  Thus induction on m proves that this
stream computes the paper's induction map, including its invalid extension.
At the output, a=mu\lambda and q_m=mu\tilde-lambda, so a=q_m iff
lambda=tilde-lambda.  This proves the inverse branch test without consulting
discarded history, and linearity covers arbitrary coherent superpositions.

Use the same public capacity floor(sqrt(n)) for all prefixes within an
order-n induction, not an input-dependent allocation.  Compact updates,
row/index conversion, and a near-linear A each cost Õ(sqrt(n)); n levels
therefore cost Õ(n^(3/2)).  Even naive expanded/compact endpoint conversions
costing O(n sqrt(n)) at OUTER boundaries preserve this bound.  Summing the
outer inductions gives Õ(n^(5/2)) gates and Õ(n) qubits including the path.
Local A error epsilon/O(n^2) suffices for total error epsilon by telescoping.
This is a conditional circuit construction: the dense A/reference operations
here do not establish the efficient A implementation or its bit-precision
bound, and Python branches are controls on a fixed schedule, not adaptive
classical access to a quantum state.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from math import factorial, isqrt, sqrt
from typing import Iterable

import numpy as np

from fourier.amatrix import a_matrix_from_contents
from fourier.compact_shapes import FrobeniusShape
from fourier.diagrams import partitions

Partition = tuple[int, ...]
Word = tuple[int, ...]
STAR = 0


def addable_rows(shape: Partition) -> tuple[int, ...]:
    """Independent elementary row-list oracle, used only in references."""
    return tuple(
        r for r in range(1, len(shape) + 2)
        if r == 1 or (shape[r - 2] > (shape[r - 1] if r <= len(shape) else 0))
    )


def removable_rows(shape: Partition) -> tuple[int, ...]:
    return tuple(
        r for r in range(1, len(shape) + 1)
        if shape[r - 1] > (shape[r] if r < len(shape) else 0)
    )


def add_row(shape: Partition, row: int) -> Partition:
    if row not in addable_rows(shape):
        raise ValueError(f"row {row} is not addable to {shape}")
    result = list(shape)
    if row == len(shape) + 1:
        result.append(1)
    else:
        result[row - 1] += 1
    return tuple(result)


def remove_row(shape: Partition, row: int) -> Partition:
    if row not in removable_rows(shape):
        raise ValueError(f"row {row} is not removable from {shape}")
    result = list(shape)
    result[row - 1] -= 1
    if not result[-1]:
        result.pop()
    return tuple(result)


@lru_cache(maxsize=None)
def shape_of_word(word: Word) -> Partition:
    shape: Partition = ()
    for row in word:
        if row:
            shape = add_row(shape, row)
    return shape


@lru_cache(maxsize=None)
def tableaux(shape: Partition) -> tuple[Word, ...]:
    if not shape:
        return ((),)
    return tuple(
        word + (row,)
        for row in removable_rows(shape)
        for word in tableaux(remove_row(shape, row))
    )


def added_row(before: Partition, after: Partition) -> int:
    matches = [r for r in addable_rows(before) if add_row(before, r) == after]
    if len(matches) != 1:
        raise ValueError("endpoints do not differ by exactly one addable box")
    return matches[0]


@lru_cache(maxsize=None)
def reference_a(shape: Partition) -> tuple[tuple[int, ...], tuple[int, ...], np.ndarray]:
    """Dense residue formula used by the simulator, not resource accounting."""
    adds = addable_rows(shape)
    removes = removable_rows(shape)
    x = np.array([(shape[r - 1] if r <= len(shape) else 0) + 1 - r for r in adds])
    y = np.array([shape[r - 1] - r for r in removes])
    return adds, (STAR,) + removes, a_matrix_from_contents(x, y)


@dataclass(frozen=True)
class StreamState:
    prefix: FrobeniusShape
    token: int
    flag: int
    word: Word


@dataclass(frozen=True)
class InductionOutput:
    endpoint: Partition
    word: Word
    flag: int = 1


@dataclass(frozen=True)
class CleanupWitness:
    prefix_matches_control: bool
    token_xor: int
    flag_zero: int

    @property
    def clean(self) -> bool:
        return self.prefix_matches_control and self.token_xor == self.flag_zero == 0


@dataclass(frozen=True)
class StreamingResources:
    """Polynomial skeleton only: exclude logarithmic arithmetic constants."""

    outer_inductions: int
    embedding_calls: int
    padded_inner_slot_work: int
    naive_boundary_slot_work: int
    max_shape_slots: int


def streaming_resource_skeleton(n: int) -> StreamingResources:
    """Explicit fixed-capacity bounds; not gate counts of this dense verifier.

    Per outer m use D=floor(sqrt(m)) arm/leg slots and K=2D+1 corner
    slots.  Every one of m inner stages scans O(K) words (up to polylogs).
    Naive endpoint expansion/conversion also costs O(m K) only once at that
    outer boundary.  These sums are bounded by O(n^(5/2)), not O(n^3).
    """
    if n < 0:
        raise ValueError("n must be nonnegative")
    work = sum(m * (2 * isqrt(m) + 1) for m in range(1, n + 1))
    return StreamingResources(n, n * (n + 1) // 2, work, work, 2 * isqrt(n))


def _accumulate(items: Iterable[tuple[object, complex]]) -> dict:
    result = defaultdict(complex)
    for key, amplitude in items:
        result[key] += amplitude
    return {key: value for key, value in result.items() if abs(value) > 1e-15}


def initial_state(index: int, word: Word) -> StreamState:
    return StreamState(FrobeniusShape.from_partition((), max_boxes=len(word) + 1),
                       STAR, index, (STAR,) + word)


@lru_cache(maxsize=None)
def compact_a(shape: FrobeniusShape) -> tuple[tuple[int, ...], tuple[int, ...], np.ndarray]:
    """Only compact scans/corner extraction feed the local A interface.

    Constructing the dense matrix here is a small-system stand-in for the
    separately proposed near-linear A implementation.  No expanded partition
    is constructed or inspected in a streaming inner step.
    """
    adds = shape.corners("add")
    removes = shape.corners("remove")
    if not adds.clean or not removes.clean:
        raise AssertionError("corner extraction left scratch")
    return adds.rows, (STAR,) + removes.rows, a_matrix_from_contents(
        np.array(adds.contents), np.array(removes.contents)
    )


def embedding_step(state: StreamState, level: int, inverse: bool = False) -> dict[StreamState, complex]:
    """Diagonal token embedding and its inverse; equality workspace erases.

    This helper includes neither the prefix update nor inactive-branch swap.
    It acts as identity when flag!=1 or when token differs from the current
    symbol, as required by the detour identity in Definition 5.16.
    """
    current = state.word[level - 1]
    if state.flag != 1 or state.token != current:
        return {state: 1.0}
    adds, columns, matrix = compact_a(state.prefix)
    inputs, outputs, matrix = (adds, columns, matrix.T) if inverse else (columns, adds, matrix)
    if current not in inputs:
        raise ValueError("diagonal embedding symbol is not in the promised corner subspace")
    column = (state.prefix.row_to_corner(current, "add") if inverse
              else 0 if current == STAR else 1 + state.prefix.row_to_corner(current, "remove"))
    result = {}
    for output_index, (output, amplitude) in enumerate(zip(outputs, matrix[:, column])):
        recovered_row = (STAR if inverse and output_index == 0 else
                         state.prefix.corner_to_row(output_index - 1, "remove") if inverse else
                         state.prefix.corner_to_row(output_index, "add"))
        assert recovered_row == output
        word = state.word[: level - 1] + (output,) + state.word[level:]
        result[StreamState(state.prefix, output, state.flag, word)] = complex(amplitude)
        # The coherent equality bit is recomputed from the new registers.
        assert (state.token == current) == (output == word[level - 1])
    return result


def forward_step(state: StreamState, level: int) -> dict[StreamState, complex]:
    if not 1 <= level <= len(state.word):
        raise ValueError("level outside word")
    prefix, token, flag, word = state.prefix, state.token, state.flag, state.word
    if level > 1:
        prefix = prefix.add_box(word[level - 1])
        if token == STAR:
            if word[level - 2] != STAR or flag <= 1:
                raise ValueError("invalid inactive-branch invariant")
            updated = list(word)
            updated[level - 2], updated[level - 1] = updated[level - 1], updated[level - 2]
            word = tuple(updated)
            flag -= 1
    return embedding_step(StreamState(prefix, token, flag, word), level)


def inverse_step(state: StreamState, level: int) -> dict[StreamState, complex]:
    result = []
    for intermediate, amplitude in embedding_step(state, level, inverse=True).items():
        prefix, token, flag, word = intermediate.prefix, intermediate.token, intermediate.flag, intermediate.word
        if level > 1:
            if token == STAR:
                if word[level - 1] != STAR:
                    raise ValueError("inverse inactive branch did not recover star")
                flag += 1
                updated = list(word)
                updated[level - 2], updated[level - 1] = updated[level - 1], updated[level - 2]
                word = tuple(updated)
            prefix = prefix.remove_box(word[level - 1])
        result.append((StreamState(prefix, token, flag, word), amplitude))
    return _accumulate(result)


def evolve(state: dict[StreamState, complex], level: int, inverse: bool = False) -> dict[StreamState, complex]:
    step = inverse_step if inverse else forward_step
    return _accumulate(
        (output, amplitude * coefficient)
        for basis, amplitude in state.items()
        for output, coefficient in step(basis, level).items()
    )


def streaming_state(index: int, word: Word) -> dict[StreamState, complex]:
    if index < 1:
        raise ValueError("transversal indices start at one")
    shape_of_word(word)  # validate the promised tableau input
    state = {initial_state(index, word): 1.0 + 0j}
    for level in range(1, len(word) + 2):
        state = evolve(state, level)
    return state


def clean_output(state: StreamState, original_endpoint: Partition) -> tuple[InductionOutput, CleanupWitness]:
    output_shape = state.prefix.add_box(state.token) if state.token else state.prefix
    # External-boundary conversions are deliberately explicit.  Neither
    # from_partition nor to_partition is called inside the streaming loop.
    endpoint = output_shape.to_partition()
    external_shape = FrobeniusShape.from_partition(original_endpoint, max_boxes=state.prefix.max_boxes)
    recovered_token = output_shape.row_checksum - external_shape.row_checksum if state.flag == 1 else STAR
    witness = CleanupWitness(state.prefix.same_shape(external_shape), state.token ^ recovered_token,
                             state.flag - 1 if state.flag == 1 else 0)
    output = InductionOutput(endpoint, state.word, state.flag)
    if shape_of_word(state.word) != endpoint:
        raise AssertionError("output path and endpoint disagree")
    if not witness.clean:
        raise AssertionError("endpoint/token cleanup failed")
    return output, witness


@lru_cache(maxsize=None)
def streaming_column(index: int, word: Word) -> dict[InductionOutput, complex]:
    endpoint = shape_of_word(word)
    return _accumulate(
        (clean_output(state, endpoint)[0], amplitude)
        for state, amplitude in streaming_state(index, word).items()
    )


def inverse_streaming_column(output: InductionOutput, original_endpoint: Partition) -> dict[tuple[int, Word], complex]:
    """Rebuild boundary workspace from the clean output, then reverse locally.

    Unlike merely multiplying by a previously built matrix transpose, this
    exercises endpoint-token recovery and every reverse star/detour decision.
    All partition conversions occur once at this outer boundary.
    """
    n = sum(original_endpoint) + 1
    if len(output.word) != n or shape_of_word(output.word) != output.endpoint:
        raise ValueError("invalid output path")
    prefix = FrobeniusShape.from_partition(original_endpoint, max_boxes=n)
    endpoint = FrobeniusShape.from_partition(output.endpoint, max_boxes=n)
    token = endpoint.row_checksum - prefix.row_checksum if output.flag == 1 else STAR
    if (output.flag < 1 or (output.flag != 1 and output.endpoint != original_endpoint)
            or (output.flag == 1 and prefix.add_box(token) != endpoint)):
        raise ValueError("invalid output endpoint or flag")
    state = {StreamState(prefix, token, output.flag, output.word): 1.0 + 0j}
    for level in range(n, 0, -1):
        state = evolve(state, level, inverse=True)
    result = []
    for basis, amplitude in state.items():
        if basis.prefix.size or basis.token or basis.word[0] != STAR:
            raise AssertionError("inverse induction left workspace")
        result.append(((basis.flag, basis.word[1:]), amplitude))
    return _accumulate(result)


def induction_bases(shape: Partition, max_index: int | None = None):
    n = sum(shape) + 1
    max_index = n if max_index is None else max_index
    if max_index < n:
        raise ValueError("max_index must include all valid transversals")
    inputs = tuple((index, word) for index in range(1, max_index + 1) for word in tableaux(shape))
    outputs = tuple(
        InductionOutput(successor, word)
        for row in addable_rows(shape)
        for successor in (add_row(shape, row),)
        for word in tableaux(successor)
    ) + tuple(
        InductionOutput(shape, word + (STAR,), index - n + 1)
        for index in range(n + 1, max_index + 1) for word in tableaux(shape)
    )
    return inputs, outputs


def streaming_induction_matrix(shape: Partition, max_index: int | None = None) -> np.ndarray:
    inputs, outputs = induction_bases(shape, max_index)
    row_index = {output: j for j, output in enumerate(outputs)}
    matrix = np.zeros((len(outputs), len(inputs)), dtype=complex)
    for column, (index, word) in enumerate(inputs):
        for output, amplitude in streaming_column(index, word).items():
            matrix[row_index[output], column] += amplitude
    return matrix


@lru_cache(maxsize=None)
def tableau_contents(word: Word) -> tuple[int, ...]:
    lengths = defaultdict(int)
    contents = []
    for row in word:
        lengths[row] += 1
        contents.append(lengths[row] - row)
    return tuple(contents)


@lru_cache(maxsize=None)
def young_generator(shape: Partition, index: int) -> np.ndarray:
    """Direct Young orthogonal sigma_index; independent of all A blocks."""
    n = sum(shape)
    if not 1 <= index < n:
        raise ValueError("adjacent-transposition index outside shape")
    basis = tableaux(shape)
    position = {word: j for j, word in enumerate(basis)}
    matrix = np.zeros((len(basis), len(basis)))
    for column, word in enumerate(basis):
        contents = tableau_contents(word)
        distance = contents[index] - contents[index - 1]
        matrix[column, column] = 1.0 / distance
        swapped = word[: index - 1] + (word[index], word[index - 1]) + word[index + 1:]
        if swapped in position:
            matrix[position[swapped], column] += sqrt(max(0.0, 1 - 1 / distance**2))
        elif abs(distance) != 1:
            raise AssertionError("Young orthogonal invalid-tableau rule failed")
    return matrix


@lru_cache(maxsize=None)
def young_transversal(shape: Partition, level: int, index: int) -> np.ndarray:
    if not 1 <= index <= level <= sum(shape):
        raise ValueError("invalid transversal")
    matrix = np.eye(len(tableaux(shape)))
    for generator in range(index, level):
        matrix = matrix @ young_generator(shape, generator)
    return matrix


def direct_induction_matrix(shape: Partition, max_index: int | None = None) -> np.ndarray:
    """Definition 3.6, using representation matrices rather than recursion."""
    n = sum(shape) + 1
    inputs, outputs = induction_bases(shape, max_index)
    matrix = np.zeros((len(outputs), len(inputs)), dtype=complex)
    original_dimension = len(tableaux(shape))
    for column, (index, word) in enumerate(inputs):
        for row, output in enumerate(outputs):
            if index > n:
                matrix[row, column] = output == InductionOutput(shape, word + (STAR,), index - n + 1)
            elif output.flag == 1:
                successor_basis = tableaux(output.endpoint)
                extension = word + (added_row(shape, output.endpoint),)
                representation = young_transversal(output.endpoint, n, index)
                matrix[row, column] = sqrt(len(successor_basis) / (n * original_dimension)) * representation[
                    successor_basis.index(output.word), successor_basis.index(extension)
                ]
    return matrix


def transversal_words(n: int) -> tuple[Word, ...]:
    return tuple(product(*(range(1, level + 1) for level in range(1, n + 1))))


def fourier_basis(n: int) -> tuple[tuple[Partition, Word, Word], ...]:
    return tuple(
        (shape, left, right)
        for partition in partitions(n) for shape in (tuple(partition),)
        for left in tableaux(shape) for right in tableaux(shape)
    )


def streaming_fourier_column(indices: Word) -> dict[tuple[Partition, Word, Word], complex]:
    """Outer subgroup induction including the right-path extension T_m."""
    state = {((), (), ()): 1.0 + 0j}
    for level, index in enumerate(indices, 1):
        if not 1 <= index <= level:
            raise ValueError("invalid transversal word")
        state = _accumulate(
            ((output.endpoint, output.word, right + (added_row(shape, output.endpoint),)),
             amplitude * coefficient)
            for (shape, left, right), amplitude in state.items()
            for output, coefficient in streaming_column(index, left).items()
        )
    return state


def streaming_fourier_matrix(n: int) -> np.ndarray:
    rows = fourier_basis(n)
    row_index = {row: index for index, row in enumerate(rows)}
    columns = transversal_words(n)
    matrix = np.zeros((len(rows), len(columns)), dtype=complex)
    for column, indices in enumerate(columns):
        for output, amplitude in streaming_fourier_column(indices).items():
            matrix[row_index[output], column] += amplitude
    return matrix


def direct_fourier_matrix(n: int) -> np.ndarray:
    """Definition 3.1 on g=t_n ... t_1, independent Young generators."""
    rows = fourier_basis(n)
    columns = transversal_words(n)
    matrix = np.zeros((len(rows), len(columns)), dtype=complex)
    for column, indices in enumerate(columns):
        representations = {}
        for partition in partitions(n):
            shape = tuple(partition)
            basis = tableaux(shape)
            representation = np.eye(len(basis))
            for level, index in enumerate(indices, 1):
                representation = young_transversal(shape, level, index) @ representation
            representations[shape] = representation
        for row, (shape, left, right) in enumerate(rows):
            basis = tableaux(shape)
            matrix[row, column] = sqrt(len(basis) / factorial(n)) * representations[shape][basis.index(left), basis.index(right)]
    return matrix


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-n", type=int, default=5)
    args = parser.parse_args()
    for n in range(1, args.max_n + 1):
        actual = streaming_fourier_matrix(n)
        expected = direct_fourier_matrix(n)
        print(f"n={n} dimension={factorial(n)} max-entry-error={np.max(np.abs(actual-expected)):.3e} "
              f"orthogonality-error={np.linalg.norm(actual.conj().T@actual-np.eye(len(actual)), ord=np.inf):.3e}")


if __name__ == "__main__":
    main()
