"""Fixed-address reversible bookkeeping for direct amplified induction.

This is a word-level reversible simulator, NOT a quantum-gate compiler.  Every
sort is an explicit bitonic network with a retained decision bit per comparator;
every scan runs over public positions. No dictionary, data-dependent address,
library sort, or shape-dependent allocation occurs in the claimed schedules.
Python branches in word expressions stand for controlled fixed-width arithmetic.

TABLEAU ORACLE. For a public M-slot row word and active length t<=M, make M
data records and M sentinel records, one sentinel for each row 1,...,M. Sort
by (live,row,kind,label), with data before the corresponding sentinel. A
segmented prefix count assigns each data record its within-row column and
each sentinel its row's total. Sort by fixed origin to copy the M columns and
M-row histogram. Sort by (column,row,label) to check that each column starts
at row 1, has consecutive rows, and has strictly increasing labels. Together
with 1<=row<=M these conditions are equivalent to the active word being a
standard tableau: row labels increase by the within-row construction, and
the column checks enforce both the Young shape and column increase.

The three sorts and all annotations are reversed after copying the outputs.
Sorting keys are never changed before their corresponding inverse. Padding
records are kept physically until cleanup, not discarded on a validity claim.
Thus all tapes and work words erase on EVERY fixed-width input, including
zero, out-of-range, and non-tableau rows. Inactive positions are ignored;
the source predicate separately requires its final star slot to equal zero.

PREDICATES. Source membership requires i in 1,...,M, a valid M-1 prefix with
histogram lambda, a final star, attenuation flag zero, and every supplied work
bit zero. Target membership requires i=0, a valid M-box tableau with histogram
mu covering lambda, and the same full-work condition. Cover detection scans
padded row arrays: lambda and mu must be partitions of the public sizes and
their entries differ by exactly one +1. The added row is an output of this
scan, NOT the final left-tableau symbol. All of these are clean XOR oracles on
arbitrary output registers; malformed data produce false predicates and still
erase all internal scratch.

OUTER BOUNDARY. Compute mu from Q, copy the cover row a into the fresh right
slot, and erase old lambda by XOR with mu-box(a), obtained in a fixed scan.
The forward and inverse schedules explicitly uncompute all temporary outputs.
On the valid promise the old endpoint residue is zero. On malformed inputs
that residue is retained as an output, NOT asserted to be zero; the inverse
still restores every input and clears the computed endpoint. This is a total
reversible extension of the promised in-place boundary conversion.

COST. With b-bit input words, N=next_power_of_two(max(1,2M))<4M for M>0.
One tableau oracle uses six N-record network passes, each with
N log2(N)(log2(N)+1)/4 comparators, and O(N+M) reversible scan/copy operations.
Record and accumulator widths are O(b+log(M+1)); comparator pairs and all
other addresses are public. Hence O(M log^2(M+1)) word operations and
O(M log^2(M+1)*(b+log(M+1))) Boolean gates suffice. For b=O(log M), this is
O(M log^3 M) gates and O(M log^2 M) work bits, including retained swap tapes.
Predicates and boundary conversions invoke only a constant number of these
oracles plus fixed linear scans. W supplied work bits add O(W) gates. The
ledger is an operation skeleton, not an exact Toffoli/Clifford+T count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable


@dataclass
class Ledger:
    comparator_applications: int = 0
    erased_decisions: int = 0
    word_operations: int = 0
    scan_steps: int = 0
    peak_sort_tape_bits: int = 0
    record_slots: int = 0
    workspace_word_bound: int = 0
    arithmetic_bits: int = 1
    network_sizes: list[int] = field(default_factory=list)
    _live_sort_bits: int = 0

    def step(self, operations: int = 1) -> None:
        self.word_operations += operations
        self.scan_steps += 1

    def merge(self, other: Ledger) -> None:
        assert other._live_sort_bits == 0
        self.comparator_applications += other.comparator_applications
        self.erased_decisions += other.erased_decisions
        self.word_operations += other.word_operations
        self.scan_steps += other.scan_steps
        self.peak_sort_tape_bits = max(self.peak_sort_tape_bits, other.peak_sort_tape_bits)
        self.record_slots = max(self.record_slots, other.record_slots)
        self.workspace_word_bound = max(self.workspace_word_bound, other.workspace_word_bound)
        self.arithmetic_bits = max(self.arithmetic_bits, other.arithmetic_bits)
        self.network_sizes.extend(other.network_sizes)

    @property
    def schedule_signature(self):
        return (self.comparator_applications, self.erased_decisions, self.word_operations,
                self.scan_steps, self.peak_sort_tape_bits, self.record_slots,
                self.workspace_word_bound, self.arithmetic_bits, tuple(self.network_sizes))

    @property
    def word_operation_bound(self):
        # A comparator moves a constant number of words and computes one key.
        return 20 * self.comparator_applications + self.word_operations

    @property
    def workspace_bit_bound(self):
        return self.workspace_word_bound * self.arithmetic_bits + self.peak_sort_tape_bits


@dataclass(frozen=True)
class TableauResult:
    columns: tuple[int, ...]
    endpoint: tuple[int, ...]
    valid: int
    clean: bool
    ledger: Ledger


@dataclass(frozen=True)
class PredicateResult:
    value: int
    clean: bool
    ledger: Ledger


@dataclass(frozen=True)
class CoverResult:
    row: int
    valid: int
    clean: bool
    ledger: Ledger


@dataclass(frozen=True)
class WordArrayResult:
    values: tuple[int, ...]
    clean: bool
    ledger: Ledger


@dataclass(frozen=True)
class BoundaryResult:
    endpoint: tuple[int, ...]
    right_path: tuple[int, ...]
    old_endpoint_residue: tuple[int, ...]
    workspace_clean: bool
    ledger: Ledger


@dataclass(frozen=True)
class InverseBoundaryResult:
    old_endpoint: tuple[int, ...]
    right_prefix: tuple[int, ...]
    initial_new_row: int
    cleared_endpoint: tuple[int, ...]
    workspace_clean: bool
    ledger: Ledger


@dataclass
class _Record:
    origin: int = 0
    kind: int = 0                 # data=0, sentinel=1, padding=2
    live: int = 0
    row: int = 0
    label: int = 0
    prefix: int = 0
    bad: int = 0

    def fields(self):
        return (self.origin, self.kind, self.live, self.row, self.label, self.prefix, self.bad)

    def xor_initial(self, values):
        self.origin ^= values[0]
        self.kind ^= values[1]
        self.live ^= values[2]
        self.row ^= values[3]
        self.label ^= values[4]


def _width(m: int, word_bits: int | None) -> int:
    bits = max(1, m.bit_length()) if word_bits is None else word_bits
    if not isinstance(bits, int) or bits < max(1, m.bit_length()):
        raise ValueError("word width cannot encode the public capacity")
    return bits


def _words(values, count: int, bits: int, name: str):
    result = tuple(values)
    if len(result) != count or any(not isinstance(v, int) or not 0 <= v < 1 << bits
                                   for v in result):
        raise ValueError(f"{name} must contain {count} unsigned {bits}-bit words")
    return result


def _bit(value: int, name: str):
    if value not in (0, 1):
        raise ValueError(f"{name} must be a bit")
    return value


def _ledger(m: int, bits: int, extra_words: int = 0):
    return Ledger(arithmetic_bits=bits + max(1, m.bit_length()) + 3,
                  workspace_word_bound=20 * max(1, m) + 32 + extra_words)


@lru_cache(maxsize=None)
def _network(size: int):
    if size < 1 or size & (size - 1):
        raise ValueError("network size must be a positive power of two")
    result = []
    block = 2
    while block <= size:
        stride = block // 2
        while stride:
            for i in range(size):
                j = i ^ stride
                if j > i:
                    result.append((i, j, (i & block) == 0))
            stride //= 2
        block *= 2
    return tuple(result)


def _sort(records, key, ledger):
    pairs = _network(len(records))
    tape = [0] * len(pairs)
    for t, (i, j, ascending) in enumerate(pairs):
        before = key(records[i]), key(records[j])
        tape[t] ^= int(before[0] > before[1] if ascending else before[0] < before[1])
        if tape[t]:
            records[i], records[j] = records[j], records[i]
        ledger.comparator_applications += 1
    ledger.network_sizes.append(len(records))
    ledger._live_sort_bits += len(tape)
    ledger.peak_sort_tape_bits = max(ledger.peak_sort_tape_bits, ledger._live_sort_bits)
    return tape


def _unsort(records, key, tape, ledger):
    pairs = _network(len(records))
    assert len(tape) == len(pairs)
    for t in range(len(pairs) - 1, -1, -1):
        i, j, ascending = pairs[t]
        if tape[t]:
            records[i], records[j] = records[j], records[i]
        before = key(records[i]), key(records[j])
        tape[t] ^= int(before[0] > before[1] if ascending else before[0] < before[1])
        assert tape[t] == 0
        ledger.comparator_applications += 1
        ledger.erased_decisions += 1
    ledger.network_sizes.append(len(records))
    ledger._live_sort_bits -= len(tape)


def _initial_record(i, word, active):
    m = len(word)
    if i < m:                    # public positions, never a quantum address
        return (i, 0, int(i < active), word[i], i + 1)
    if i < 2 * m:
        return (i, 1, 1, i - m + 1, 0)
    return (i, 2, 0, 0, 0)


def _row_key(r):
    return (1 - r.live, r.row, r.kind, r.label, r.origin)


def _origin_key(r):
    return (r.origin,)


def _column_key(r):
    data = r.live and r.kind == 0
    return (0, r.prefix, r.row, r.label, r.origin) if data else (1, 0, 0, 0, r.origin)


def _prefix_value(records, i):
    current = records[i]
    if not current.live:
        return 0
    previous = records[i - 1] if i else None
    same = previous is not None and previous.live and previous.row == current.row
    return (previous.prefix if same else 0) + int(current.kind == 0)


def _column_bad(records, i, m):
    current = records[i]
    if not (current.live and current.kind == 0):
        return 0
    previous = records[i - 1] if i else None
    same = (previous is not None and previous.live and previous.kind == 0
            and previous.prefix == current.prefix)
    above = (current.row == previous.row + 1 and current.label > previous.label
             if same else current.row == 1)
    return int(not (1 <= current.row <= m and above))


def tableau_data_oracle(word, *, active_length=None, word_bits=None,
                        initial_columns=None, initial_endpoint=None, initial_valid=0):
    """Total clean XOR oracle; only hardware widths, not Young promises, reject."""
    word = tuple(word)
    m = len(word)
    bits = _width(m, word_bits)
    word = _words(word, m, bits, "row word")
    active = m if active_length is None else active_length
    if not isinstance(active, int) or not 0 <= active <= m:
        raise ValueError("active length is a public integer in 0,...,M")
    columns = list(_words((0,) * m if initial_columns is None else initial_columns,
                          m, bits, "column target"))
    endpoint = list(_words((0,) * m if initial_endpoint is None else initial_endpoint,
                           m, bits, "endpoint target"))
    valid = _bit(initial_valid, "validity target")
    capacity = 1 << (max(1, 2 * m) - 1).bit_length()
    ledger = _ledger(m, bits)
    ledger.record_slots = capacity
    ledger.workspace_word_bound = 7 * capacity + 4 * m + 32
    records = [_Record() for _ in range(capacity)]
    for i in range(capacity):
        records[i].xor_initial(_initial_record(i, word, active))
        ledger.step(5)
    row_tape = _sort(records, _row_key, ledger)
    for i in range(capacity):
        records[i].prefix ^= _prefix_value(records, i)
        ledger.step(8)
    origin_tape = _sort(records, _origin_key, ledger)
    for i in range(m):
        columns[i] ^= records[i].prefix
        endpoint[i] ^= records[m + i].prefix
        ledger.step(2)
    column_tape = _sort(records, _column_key, ledger)
    violations = 0
    for i in range(capacity):
        records[i].bad ^= _column_bad(records, i, m)
        violations += records[i].bad
        ledger.step(10)
    valid ^= int(violations == 0)
    ledger.word_operations += 1
    for i in range(capacity - 1, -1, -1):
        violations -= records[i].bad
        records[i].bad ^= _column_bad(records, i, m)
        assert records[i].bad == 0
        ledger.step(10)
    assert violations == 0
    _unsort(records, _column_key, column_tape, ledger)
    _unsort(records, _origin_key, origin_tape, ledger)
    for i in range(capacity - 1, -1, -1):
        records[i].prefix ^= _prefix_value(records, i)
        assert records[i].prefix == 0
        ledger.step(8)
    _unsort(records, _row_key, row_tape, ledger)
    for i in range(capacity):
        records[i].xor_initial(_initial_record(i, word, active))
        assert records[i].fields() == (0,) * 7
        ledger.step(5)
    clean = (ledger._live_sort_bits == 0 and not any(row_tape + origin_tape + column_tape)
             and all(r.fields() == (0,) * 7 for r in records))
    return TableauResult(tuple(columns), tuple(endpoint), valid, clean, ledger)


def _conjunction_xor(checks: list[Callable[[], bool]], initial: int, ledger: Ledger):
    """Compute/check/copy/reverse; checks only inspect unchanged public addresses."""
    flags = [0] * len(checks)
    # A fresh-prefix OR chain uses O(number of checks) Boolean gates/ancillas;
    # it does not silently use a log(W)-bit counter for W supplied work bits.
    failures = [0] * (len(checks) + 1)
    for i in range(len(checks)):
        flags[i] ^= int(checks[i]())
        failures[i + 1] ^= failures[i] | (1 - flags[i])
        ledger.step(4)
    result = initial ^ (1 - failures[-1])
    ledger.word_operations += 1
    for i in range(len(checks) - 1, -1, -1):
        failures[i + 1] ^= failures[i] | (1 - flags[i])
        flags[i] ^= int(checks[i]())
        assert flags[i] == failures[i + 1] == 0
        ledger.step(4)
    assert not any(failures)
    return result


def cover_oracle(old, new, *, word_bits=None, old_size=None,
                 initial_row=0, initial_valid=0):
    """Clean cover predicate/row XOR oracle on padded arrays, including malformed ones."""
    old, new = tuple(old), tuple(new)
    m = len(old)
    bits = _width(m, word_bits)
    old = _words(old, m, bits, "old shape")
    new = _words(new, m, bits, "new shape")
    old_size = m - 1 if old_size is None else old_size
    if not isinstance(old_size, int) or not 0 <= old_size <= m * m:
        raise ValueError("old size must be a public integer in 0,...,M^2")
    row_target = _words((initial_row,), 1, bits, "row target")[0]
    valid_target = _bit(initial_valid, "cover target")
    ledger = _ledger(m, bits)
    old_total = new_total = additions = weighted_row = 0
    for i in range(m):
        add = int(new[i] == old[i] + 1)
        old_total += old[i]
        new_total += new[i]
        additions += add
        weighted_row += (i + 1) * add
        ledger.step(8)
    checks = [lambda: old_total == old_size, lambda: new_total == old_size + 1,
              lambda: additions == 1]
    for i in range(m):
        checks.append(lambda i=i: (old[i] <= m and new[i] <= m
                                  and (i == 0 or (old[i - 1] >= old[i] and new[i - 1] >= new[i]))))
        checks.append(lambda i=i: new[i] == old[i] or new[i] == old[i] + 1)
    flag = _conjunction_xor(checks, 0, ledger)
    row_target ^= weighted_row if flag else 0
    valid_target ^= flag
    ledger.word_operations += 2
    flag = _conjunction_xor(checks, flag, ledger)
    assert flag == 0
    for i in range(m - 1, -1, -1):
        add = int(new[i] == old[i] + 1)
        old_total -= old[i]
        new_total -= new[i]
        additions -= add
        weighted_row -= (i + 1) * add
        ledger.step(8)
    clean = (old_total, new_total, additions, weighted_row, flag) == (0,) * 5
    return CoverResult(row_target, valid_target, clean, ledger)


def _erase_tableau_outputs(word, data, active, bits, ledger):
    erased = tableau_data_oracle(word, active_length=active, word_bits=bits,
                                 initial_columns=data.columns, initial_endpoint=data.endpoint,
                                 initial_valid=data.valid)
    ledger.merge(erased.ledger)
    assert erased.columns == erased.endpoint == (0,) * len(word) and erased.valid == 0
    assert erased.clean


def _full_work_checks(flag, work_bits):
    _bit(flag, "attenuation flag")
    work = tuple(_bit(v, "work bit") for v in work_bits)
    return [lambda: flag == 0] + [lambda i=i: work[i] == 0 for i in range(len(work))]


def source_predicate(word, old, index, *, attenuation_flag=0, work_bits=(),
                     word_bits=None, initial=0):
    """Source: valid lambda path, final star, i in 1..M, and all work zero."""
    word, old = tuple(word), tuple(old)
    m = len(word)
    if m < 1:
        raise ValueError("induction order must be positive")
    bits = _width(m, word_bits)
    word = _words(word, m, bits, "row word")
    old = _words(old, m, bits, "old shape")
    index = _words((index,), 1, bits, "index")[0]
    initial = _bit(initial, "predicate target")
    ledger = _ledger(m, bits, len(work_bits))
    data = tableau_data_oracle(word, active_length=m - 1, word_bits=bits)
    ledger.merge(data.ledger)
    checks = [lambda: data.valid == 1, lambda: word[-1] == 0, lambda: 1 <= index <= m]
    checks += [lambda i=i: data.endpoint[i] == old[i] for i in range(m)]
    checks += _full_work_checks(attenuation_flag, work_bits)
    value = _conjunction_xor(checks, initial, ledger)
    _erase_tableau_outputs(word, data, m - 1, bits, ledger)
    return PredicateResult(value, data.clean and ledger._live_sort_bits == 0, ledger)


def target_predicate(word, old, index, *, attenuation_flag=0, work_bits=(),
                     word_bits=None, initial=0):
    """Target: a covering endpoint, erased index, and every work register zero."""
    word, old = tuple(word), tuple(old)
    m = len(word)
    if m < 1:
        raise ValueError("induction order must be positive")
    bits = _width(m, word_bits)
    word = _words(word, m, bits, "row word")
    old = _words(old, m, bits, "old shape")
    index = _words((index,), 1, bits, "index")[0]
    initial = _bit(initial, "predicate target")
    ledger = _ledger(m, bits, len(work_bits))
    data = tableau_data_oracle(word, word_bits=bits)
    ledger.merge(data.ledger)
    cover = cover_oracle(old, data.endpoint, word_bits=bits)
    ledger.merge(cover.ledger)
    checks = [lambda: data.valid == 1, lambda: cover.valid == 1, lambda: index == 0]
    checks += _full_work_checks(attenuation_flag, work_bits)
    value = _conjunction_xor(checks, initial, ledger)
    erased_cover = cover_oracle(old, data.endpoint, word_bits=bits,
                                initial_row=cover.row, initial_valid=cover.valid)
    ledger.merge(erased_cover.ledger)
    assert erased_cover.row == erased_cover.valid == 0 and erased_cover.clean
    _erase_tableau_outputs(word, data, m, bits, ledger)
    return PredicateResult(value, data.clean and cover.clean and ledger._live_sort_bits == 0, ledger)


def endpoint_oracle(word, *, word_bits=None, initial=None):
    """Histogram-only clean XOR oracle; temporary cache and validity are erased."""
    word = tuple(word)
    m = len(word)
    bits = _width(m, word_bits)
    target = _words((0,) * m if initial is None else initial, m, bits, "endpoint target")
    ledger = _ledger(m, bits)
    data = tableau_data_oracle(word, word_bits=bits)
    ledger.merge(data.ledger)
    result = []
    for i in range(m):
        result.append(target[i] ^ data.endpoint[i])
        ledger.step()
    _erase_tableau_outputs(word, data, m, bits, ledger)
    return WordArrayResult(tuple(result), data.clean and ledger._live_sort_bits == 0, ledger)


def erase_old_endpoint_oracle(new, added_row, target, *, word_bits=None):
    """XOR mu-box(a) into target by a fixed scan; subtraction is modular off-promise."""
    new, target = tuple(new), tuple(target)
    m = len(new)
    bits = _width(m, word_bits)
    new = _words(new, m, bits, "new endpoint")
    target = list(_words(target, m, bits, "old endpoint target"))
    added_row = _words((added_row,), 1, bits, "added row")[0]
    ledger = _ledger(m, bits)
    scratch = [0] * m
    mask = (1 << bits) - 1
    for i in range(m):
        scratch[i] ^= (new[i] - int(added_row == i + 1)) & mask
        target[i] ^= scratch[i]
        scratch[i] ^= (new[i] - int(added_row == i + 1)) & mask
        assert scratch[i] == 0
        ledger.step(7)
    return WordArrayResult(tuple(target), not any(scratch), ledger)


def outer_boundary(old, left_word, right_prefix, *, word_bits=None, initial_new_row=0):
    """Promised clean conversion, with explicit reversible residual off-promise."""
    old, left_word, right_prefix = tuple(old), tuple(left_word), tuple(right_prefix)
    m = len(left_word)
    if m < 1:
        raise ValueError("induction order must be positive")
    bits = _width(m, word_bits)
    old = _words(old, m, bits, "old endpoint")
    left_word = _words(left_word, m, bits, "left word")
    right_prefix = _words(right_prefix, m - 1, bits, "right prefix")
    initial_new_row = _words((initial_new_row,), 1, bits, "new right slot")[0]
    ledger = _ledger(m, bits)
    endpoint = endpoint_oracle(left_word, word_bits=bits)
    ledger.merge(endpoint.ledger)
    cover = cover_oracle(old, endpoint.values, word_bits=bits)
    ledger.merge(cover.ledger)
    new_row = initial_new_row ^ cover.row
    ledger.word_operations += 1
    erased_cover = cover_oracle(old, endpoint.values, word_bits=bits,
                                initial_row=cover.row, initial_valid=cover.valid)
    ledger.merge(erased_cover.ledger)
    assert erased_cover.row == erased_cover.valid == 0 and erased_cover.clean
    residue = erase_old_endpoint_oracle(endpoint.values, new_row, old, word_bits=bits)
    ledger.merge(residue.ledger)
    clean = endpoint.clean and cover.clean and residue.clean and ledger._live_sort_bits == 0
    return BoundaryResult(endpoint.values, right_prefix + (new_row,), residue.values, clean, ledger)


def inverse_outer_boundary(endpoint, right_path, old_endpoint_residue, left_word, *, word_bits=None):
    """Undo the actual boundary schedule, including malformed promised-width data."""
    endpoint, right_path, residue, left_word = map(tuple,
        (endpoint, right_path, old_endpoint_residue, left_word))
    m = len(left_word)
    if m < 1:
        raise ValueError("induction order must be positive")
    bits = _width(m, word_bits)
    endpoint = _words(endpoint, m, bits, "new endpoint")
    right_path = _words(right_path, m, bits, "new right path")
    residue = _words(residue, m, bits, "old endpoint residue")
    left_word = _words(left_word, m, bits, "left word")
    ledger = _ledger(m, bits)
    recovered = erase_old_endpoint_oracle(endpoint, right_path[-1], residue, word_bits=bits)
    ledger.merge(recovered.ledger)
    cover = cover_oracle(recovered.values, endpoint, word_bits=bits)
    ledger.merge(cover.ledger)
    initial_new_row = right_path[-1] ^ cover.row
    ledger.word_operations += 1
    erased_cover = cover_oracle(recovered.values, endpoint, word_bits=bits,
                                initial_row=cover.row, initial_valid=cover.valid)
    ledger.merge(erased_cover.ledger)
    assert erased_cover.row == erased_cover.valid == 0 and erased_cover.clean
    erased_endpoint = endpoint_oracle(left_word, word_bits=bits, initial=endpoint)
    ledger.merge(erased_endpoint.ledger)
    clean = recovered.clean and cover.clean and erased_endpoint.clean and ledger._live_sort_bits == 0
    return InverseBoundaryResult(recovered.values, right_path[:-1], initial_new_row,
                                 erased_endpoint.values, clean, ledger)
