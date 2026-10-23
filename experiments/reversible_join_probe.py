"""Basis-state verifier for an oblivious, reversible batched source/query join.

Each valid source has a unique integer box ID and a fixed-width payload.
Each query asks for that payload, or zeros if no source has the ID. Records
are padded to a power of two and sorted by a fixed bitonic comparator network
with source records before queries of the same ID. A linear scan broadcasts
the latest source only to equal-ID queries. No addressed payload lookup is
used: all record accesses are fixed comparators or sequential scans.

The complete compute-copy-uncompute schedule is:
  sort; scan with carry history; unsort; copy fixed original query slots;
  sort; reverse scan; unsort.
Retained comparator bits let every sort be reversed and erased. Retained
scan history lets the carry and temporary query answers be erased. The output
is XORed into a separate register, so running the same join twice restores
an arbitrary initial output as well as all input and work registers.

This is a classical verifier on computational-basis records, NOT a quantum
circuit compiler. Python conditionals simulate controlled swaps/copies; a
coherent implementation must actually use reversible comparators, controlled
payload movement, and XORs while preserving input source registers. Testing
basis states alone does not independently verify phase-correct gate synthesis.
There are O(n log^2 n) fixed comparator applications and O(n) scan steps.
Gate costs must include the ID precision and all payload bits; payload access
is not a free QRAM oracle. Signed fixed-point FMM moments can be stored as
fixed-width two's-complement payload words.
"""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class SourceRecord:
    key: int
    payload: tuple[int, ...]
    valid: bool = True


@dataclass(frozen=True)
class QueryRecord:
    key: int
    valid: bool = True


@dataclass
class _WorkRecord:
    key: int
    payload: tuple[int, ...]
    valid: bool
    source: bool
    origin: int
    answer: tuple[int, ...]

    def sort_key(self):
        return (not self.valid, self.key if self.valid else 0,
                not self.source, self.origin)


@dataclass(frozen=True)
class JoinResult:
    outputs: tuple[tuple[int, ...], ...]
    padded_records: int
    comparators_per_sort: int
    sorting_passes: int
    scan_steps: int
    payload_words: int
    payload_bits: int
    clean: bool

    @property
    def comparator_applications(self):
        return self.comparators_per_sort * self.sorting_passes


def bitonic_comparators(size):
    """Fixed comparator wiring for an ascending bitonic sorting network."""
    if not isinstance(size, int) or size < 1 or size & (size - 1):
        raise ValueError("network size must be a positive power of two")
    network = []
    width = 2
    while width <= size:
        distance = width // 2
        while distance:
            for i in range(size):
                j = i ^ distance
                if j > i:
                    network.append((i, j, (i & width) == 0))
            distance //= 2
        width *= 2
    return tuple(network)


def _comparison(records, i, j, ascending):
    a, b = records[i].sort_key(), records[j].sort_key()
    return a > b if ascending else a < b


def _sort_with_tape(records, network):
    swaps = []
    for i, j, ascending in network:
        swapped = _comparison(records, i, j, ascending)
        swaps.append(swapped)
        if swapped:
            records[i], records[j] = records[j], records[i]
    return swaps


def _unsort_and_erase_tape(records, network, swaps):
    for t in range(len(network) - 1, -1, -1):
        i, j, ascending = network[t]
        if swaps[t]:
            records[i], records[j] = records[j], records[i]
        # Re-evaluating the predicate on the recovered pre-comparison records
        # clears the saved bit by XOR in an actual reversible implementation.
        recovered_predicate = _comparison(records, i, j, ascending)
        if recovered_predicate != swaps[t]:
            raise ArithmeticError("sort history does not match recovered records")
        swaps[t] ^= recovered_predicate


def _xor_words(a, b):
    return tuple(x ^ y for x, y in zip(a, b))


def reversible_batched_join(sources, queries, *, payload_words=1,
                           payload_bits=32, initial_output=None):
    """Compute a batched lookup into XOR output, then restore all work data.

    ``sources`` and ``queries`` already have their public padded slot counts;
    each slot's valid bit can depend on the diagram. Invalid slots remain in
    the fixed circuit. Duplicate valid source IDs are outside the input
    promise and are rejected during the sorted sequential scan.
    """
    sources, queries = tuple(sources), tuple(queries)
    if (not isinstance(payload_words, int) or payload_words < 1
            or not isinstance(payload_bits, int) or payload_bits < 1):
        raise ValueError("payload size and bit width must be positive integers")
    limit = 1 << payload_bits
    zeros = (0,) * payload_words

    def valid_payload(payload):
        return (len(payload) == payload_words
                and all(isinstance(x, int) and 0 <= x < limit for x in payload))

    for source in sources:
        if not isinstance(source.key, int) or source.key < 0 or not valid_payload(source.payload):
            raise ValueError("source IDs and payload words must be unsigned fixed-width integers")
    for query in queries:
        if not isinstance(query.key, int) or query.key < 0:
            raise ValueError("query IDs must be nonnegative integers")
    if initial_output is None:
        output = [zeros for _ in queries]
    else:
        output = [tuple(row) for row in initial_output]
        if len(output) != len(queries) or any(not valid_payload(row) for row in output):
            raise ValueError("initial output has invalid shape or payload words")

    size = 1 << (max(1, len(sources) + len(queries)) - 1).bit_length()
    initial = [
        _WorkRecord(s.key, tuple(s.payload), s.valid, True, i, zeros)
        for i, s in enumerate(sources)
    ]
    initial += [
        _WorkRecord(q.key, zeros, q.valid, False, len(sources) + i, zeros)
        for i, q in enumerate(queries)
    ]
    initial += [
        _WorkRecord(0, zeros, False, True, i, zeros)
        for i in range(len(initial), size)
    ]
    records = [replace(record) for record in initial]
    network = bitonic_comparators(size)
    swaps_first = _sort_with_tape(records, network)
    empty_carry = (False, 0, zeros)
    carry = empty_carry
    history = []
    for record in records:
        history.append(carry)
        if record.valid and record.source:
            if carry[0] and carry[1] == record.key:
                raise ValueError("valid source IDs must be unique")
            carry = (True, record.key, record.payload)
        elif record.valid and carry[0] and record.key == carry[1]:
            record.answer = _xor_words(record.answer, carry[2])

    # Preserve the scan tape/carry while moving answers back to fixed original
    # query slots. No scatter by an input-dependent query index is performed.
    _unsort_and_erase_tape(records, network, swaps_first)
    for i in range(len(queries)):
        output[i] = _xor_words(output[i], records[len(sources) + i].answer)

    swaps_second = _sort_with_tape(records, network)
    for i in range(size - 1, -1, -1):
        record = records[i]
        before = history[i]
        after = ((True, record.key, record.payload)
                 if record.valid and record.source else before)
        if after != carry:
            raise ArithmeticError("scan history does not match recovered carry")
        if record.valid and not record.source and before[0] and record.key == before[1]:
            record.answer = _xor_words(record.answer, before[2])
        carry = before
        history[i] = None
    _unsort_and_erase_tape(records, network, swaps_second)
    clean = (records == initial and carry == empty_carry
             and all(item is None for item in history)
             and not any(swaps_first) and not any(swaps_second))
    if not clean:
        raise ArithmeticError("join did not restore its inputs and temporary registers")
    return JoinResult(tuple(output), size, len(network), 4, 2 * size,
                      payload_words, payload_bits, clean)
