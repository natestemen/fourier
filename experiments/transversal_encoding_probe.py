"""Fixed-schedule reversible conversion of permutations and cycle transversals.

The group convention is g = t_(i_n)^(n) ... t_(i_1)^(1), with
t_i^(m) = (i,i+1,...,m), acting on the left.  In one-line notation,

    i_m = rank(g_m among g_1,...,g_m).

Indeed g_n=i_n; multiplying by the inverse last cycle deletes that value
and order-compresses all remaining values.  Repeating gives prefix ranks.

FORWARD.  Bisect the original-index interval recursively.  At each public
node, sort its records by value, count left-child records in a prefix
scan, and add this count to the rank of each right-child record.  Unsort
immediately, using saved comparator bits.  Every pair j<i is counted
exactly once, at the lowest node whose left/right children separate j,i.
Starting ranks at one therefore gives the required prefix ranks.  Reverse
the same node operations with subtraction to erase the work after copying
the output.  Sorting keys never depend on the rank being accumulated.

INVERSE.  In each public value interval, maintain its records in decreasing
original-index order.  A record's current rank is its rank among records
in that interval with no larger original index.  Let L initially equal
floor(interval_size/2).  Scan in decreasing original-index order: rank<=L
assigns a record to the lower value half and decrements L; otherwise assign
it to the upper half and subtract L from its rank.  This is exactly the
partition of the remaining order statistic between the remaining lower
and upper values.  If lower capacity is zero no lower assignment occurs;
if upper capacity is zero every assignment is lower.  Consequently the
children have the public sizes floor(size/2) and ceil(size/2), including
for odd sizes.  Sorting by (branch, decreasing original index) puts both
children in their required order.  Recursively reach singleton intervals,
assign their values, and sort once by original index to produce g.

REVERSIBILITY.  Every sorting network stores one decision bit per
comparator; sorting is not treated as a reversible primitive without this
history.  In the inverse algorithm retain each record's branch at every
tree depth.  After copying the output, reverse the final sort, clear the
known singleton values, then undo nodes in reverse order.  Undo each
node's sort, scan its records in increasing original-index order, and
reconstruct L and the old rank.  The old comparison then erases the branch
bit.  Comparator bits are erased only after their swaps have been undone
and their original comparison has been recomputed.  This module checks
that all tapes and all computed branches are zero at the end.

GATE MODEL.  Every interval boundary, comparator pair, scan extent, and
recursion depth depends only on n.  Padding an interval of size s to the
next power of two uses fewer than 2s records and fixed validity masks.
Bitonic sorting costs O(s log^2(s+1)) comparators.  Summing over the fixed
balanced interval tree gives O(n log^3(n+1)) comparators; each comparison
and controlled record swap uses O(log(n+1)) Boolean/Toffoli gates because
the index, value, rank, and all branch bits together have that width.
Thus O(n log^4(n+1)) gates and O(n log^3(n+1)) work bits suffice, with
compute/copy/reverse and no QRAM.  Arithmetic is exact on the promise;
using fixed-width modular arithmetic supplies a unitary extension off it.

Both clean XOR oracles follow.  For the bijection f between equal-width
encodings, a clean in-place conversion is U_f; SWAP; U_(f^-1), producing
|g>|0> -> |f(g)>|0>.  Input validation here is a test-harness check, not
an extra algorithmic promise-testing circuit.  This is a reversible
classical reference simulation, not a compiled quantum-gate circuit.
"""

from dataclasses import dataclass
from functools import lru_cache


@dataclass
class _Record:
    index: int
    value: int
    rank: int
    branches: list[int]
    valid: bool = True


@dataclass
class _Counts:
    comparisons: int = 0
    scans: int = 0
    erased_decisions: int = 0


@dataclass(frozen=True)
class EncodingResult:
    outputs: tuple[int, ...]
    clean: bool
    comparator_applications: int
    scan_steps: int
    erased_decisions: int
    public_interval_sizes: tuple[int, ...]


@lru_cache(maxsize=None)
def bitonic_comparators(size):
    """Return public (left,right,ascending) pairs for a power-of-two size."""
    if size < 1 or size & (size - 1):
        raise ValueError("sorting-network size must be a positive power of two")
    pairs = []
    block = 2
    while block <= size:
        stride = block // 2
        while stride:
            for i in range(size):
                j = i ^ stride
                if j > i:
                    pairs.append((i, j, i & block == 0))
            stride //= 2
        block *= 2
    return tuple(pairs)


def _key(record, key):
    return (0,) + key(record) if record.valid else (1, record.index)


def _padding(size, depth):
    padded = 1 << (max(1, size) - 1).bit_length()
    return [_Record(i, 0, 0, [0] * depth, False) for i in range(padded - size)]


def _sort(records, start, size, key, depth, counts):
    working = records[start:start + size] + _padding(size, depth)
    tape = []
    for i, j, ascending in bitonic_comparators(len(working)):
        first, second = _key(working[i], key), _key(working[j], key)
        swap = first > second if ascending else first < second
        tape.append(int(swap))
        if swap:
            working[i], working[j] = working[j], working[i]
        counts.comparisons += 1
    assert all(r.valid for r in working[:size])
    assert all(not r.valid for r in working[size:])
    records[start:start + size] = working[:size]
    return tape


def _unsort(records, start, size, key, depth, tape, counts):
    working = records[start:start + size] + _padding(size, depth)
    pairs = bitonic_comparators(len(working))
    assert len(tape) == len(pairs)
    for t in reversed(range(len(pairs))):
        i, j, ascending = pairs[t]
        if tape[t]:
            working[i], working[j] = working[j], working[i]
        first, second = _key(working[i], key), _key(working[j], key)
        old_comparison = first > second if ascending else first < second
        tape[t] ^= int(old_comparison)
        assert tape[t] == 0
        counts.comparisons += 1
        counts.erased_decisions += 1
    assert all(r.valid for r in working[:size])
    assert all(not r.valid for r in working[size:])
    records[start:start + size] = working[:size]


def _nodes(start, size, lo=1, depth=0):
    if size <= 1:
        return
    left = size // 2
    yield start, size, lo, depth
    yield from _nodes(start, left, lo, depth + 1)
    yield from _nodes(start + left, size - left, lo + left, depth + 1)


def _initial_output(n, initial_output):
    result = (0,) * n if initial_output is None else tuple(initial_output)
    if len(result) != n or any(v < 0 for v in result):
        raise ValueError("initial output must contain n nonnegative words")
    return result


def _check_permutation(permutation):
    if sorted(permutation) != list(range(1, len(permutation) + 1)):
        raise ValueError("one-line input must be a permutation of 1,...,n")


def _check_code(code):
    if any(not 1 <= rank <= i for i, rank in enumerate(code, start=1)):
        raise ValueError("transversal index i_m must lie in 1,...,m")


def _forward_node(records, node, direction, counts):
    start, size, _, _ = node
    left_end = start + size // 2
    key = lambda r: (r.value, r.index)
    tape = _sort(records, start, size, key, 0, counts)
    left_count = 0
    for record in records[start:start + size]:
        if record.index <= left_end:
            left_count += 1
        else:
            record.rank += direction * left_count
        counts.scans += 1
    assert left_count == size // 2
    # This counter is the public left-child size, so a fixed XOR clears it.
    _unsort(records, start, size, key, 0, tape, counts)
    assert not any(tape)


def permutation_to_transversal(permutation, initial_output=None):
    """Clean XOR oracle for prefix ranks, using fixed sorting networks."""
    permutation = tuple(permutation)
    _check_permutation(permutation)
    n = len(permutation)
    before = _initial_output(n, initial_output)
    records = [_Record(i, value, 1, []) for i, value in enumerate(permutation, 1)]
    nodes = tuple(_nodes(0, n))
    counts = _Counts()
    for node in nodes:
        _forward_node(records, node, 1, counts)
    outputs = tuple(old ^ r.rank for old, r in zip(before, records))
    for node in reversed(nodes):
        _forward_node(records, node, -1, counts)
    # Index/rank constants and the copy of the unchanged input can be erased.
    clean = all((r.index, r.value, r.rank, r.branches) == (i, value, 1, [])
                for i, (value, r) in enumerate(zip(permutation, records), 1))
    return EncodingResult(outputs, clean, counts.comparisons, counts.scans,
                          counts.erased_decisions, tuple(node[1] for node in nodes))


def transversal_to_permutation(code, initial_output=None):
    """Clean XOR oracle for decoding, with explicitly erased sort tapes."""
    code = tuple(code)
    _check_code(code)
    n = len(code)
    before = _initial_output(n, initial_output)
    depth = max(0, (n - 1).bit_length())
    records = [_Record(i, 0, code[i - 1], [0] * depth) for i in range(n, 0, -1)]
    nodes = tuple(_nodes(0, n))
    histories = []
    counts = _Counts()
    for start, size, lo, level in nodes:
        remaining_left = size // 2
        assert all(a.index > b.index for a, b in
                   zip(records[start:start + size], records[start + 1:start + size]))
        for record in records[start:start + size]:
            branch = int(record.rank > remaining_left)
            record.branches[level] ^= branch
            if branch:
                record.rank -= remaining_left
            else:
                remaining_left -= 1
            counts.scans += 1
        assert remaining_left == 0
        key = lambda r, level=level: (r.branches[level], -r.index)
        tape = _sort(records, start, size, key, depth, counts)
        assert all(r.branches[level] == 0 for r in records[start:start + size // 2])
        assert all(r.branches[level] == 1 for r in records[start + size // 2:start + size])
        histories.append((start, size, lo, level, tape))
    # Ordered leaf intervals are [1,1],...,[n,n]; each contains one record.
    for value, record in enumerate(records, 1):
        assert record.rank == 1
        record.value ^= value
    final_key = lambda r: (r.index,)
    final_tape = _sort(records, 0, n, final_key, depth, counts) if n else []
    outputs = tuple(old ^ r.value for old, r in zip(before, records))
    if n:
        _unsort(records, 0, n, final_key, depth, final_tape, counts)
    for value, record in enumerate(records, 1):
        record.value ^= value
    for start, size, lo, level, tape in reversed(histories):
        key = lambda r, level=level: (r.branches[level], -r.index)
        _unsort(records, start, size, key, depth, tape, counts)
        remaining_left = 0
        for record in reversed(records[start:start + size]):
            branch = record.branches[level]
            if branch:
                record.rank += remaining_left
            else:
                remaining_left += 1
            old_comparison = int(record.rank > remaining_left)
            record.branches[level] ^= old_comparison
            assert record.branches[level] == 0
            counts.scans += 1
        assert remaining_left == size // 2
    clean = (not any(final_tape) and all(not any(h[-1]) for h in histories)
             and all((r.index, r.value, r.rank, r.branches)
                     == (i, 0, code[i - 1], [0] * depth)
                     for i, r in zip(range(n, 0, -1), records)))
    return EncodingResult(outputs, clean, counts.comparisons, counts.scans,
                          counts.erased_decisions, tuple(node[1] for node in nodes))


def canonical_cycle_product(code):
    """Independent quadratic test oracle using the actual cycle action."""
    code = tuple(code)
    _check_code(code)
    permutation = list(range(1, len(code) + 1))
    for m, index in enumerate(code, 1):
        permutation = [index if value == m else value + 1 if index <= value < m
                       else value for value in permutation]
    return tuple(permutation)
