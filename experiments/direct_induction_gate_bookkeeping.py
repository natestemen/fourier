"""Bit-blasted, fixed-network bookkeeping for a small induction compiler.

Every data-dependent computation here is emitted through BooleanBuilder as
X/CX/CCX gates, not recorded as an opaque predicate. Only public capacities
choose Python control flow. The cache XOR and the two complete promise phases
have clean helper registers on *all* fixed-width inputs, including malformed
tableaux, dirty output caches, and dirty work registers.

For M rows of b bits, a fixed bitonic network moves O(M) constant-word records
in O(M log^2(M+1) b) gates. Immutable comparator outputs retain their histories;
reversing the builder erases them after an XOR copy or a phase. This uses the
same O(M log^2(M+1) b) conservative helper-bit upper bound, rather than the
tighter workspace of the in-place word simulator. There is no shape/tableau
enumeration or data-dependent memory access in this compiler. Compilation
and explicit gate counts are not an optimized fault-tolerant resource estimate.

Inputs are little-endian tuples of Wire objects (or literal bits for read-only
inputs). Each output-cache word consists of distinct Wire objects disjoint
from the row-word inputs. Helpers are allocated separately from every tested
work wire. Source/target phases are +1 on the complete valid promise and -1
on its complement. The shape is an unchanged coherent input, not a public
parameter or a classically selected circuit.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

from reversible_gate_ir import BooleanBuilder, Circuit, Wire


@dataclass(frozen=True)
class _Record:
    row: tuple
    label: tuple
    origin: tuple
    kind: tuple
    live: object
    column: tuple


def _shape_of_words(words, *, name):
    words = tuple(tuple(word) for word in words)
    m = len(words)
    if m < 1:
        raise ValueError(f"{name} must have positive public capacity")
    width = len(words[0])
    if width < max(1, m.bit_length()) or any(len(word) != width for word in words):
        raise ValueError(f"{name} must use equal word widths encoding 0,...,M")
    return words, m, width


def _bits_ok(bits):
    if any(not isinstance(bit, Wire) and bit not in (0, 1) for bit in bits):
        raise ValueError("read-only bits must be Wire objects or literal 0/1")


def _network(capacity):
    """Public comparator schedule; no record values are inspected in Python."""
    block = 2
    while block <= capacity:
        stride = block // 2
        while stride:
            for left in range(capacity):
                right = left ^ stride
                if right > left:
                    yield left, right, (left & block) == 0
            stride //= 2
        block *= 2


def _mux_record(builder, select, yes, no):
    return _Record(
        builder.mux(select, yes.row, no.row),
        builder.mux(select, yes.label, no.label),
        builder.mux(select, yes.origin, no.origin),
        builder.mux(select, yes.kind, no.kind),
        builder.mux(select, (yes.live,), (no.live,))[0],
        builder.mux(select, yes.column, no.column),
    )


def _sort(builder, records, key):
    records = list(records)
    for left, right, ascending in _network(len(records)):
        a, b = records[left], records[right]
        ka, kb = key(a), key(b)
        swap = builder.ult(kb, ka) if ascending else builder.ult(ka, kb)
        records[left] = _mux_record(builder, swap, b, a)
        records[right] = _mux_record(builder, swap, a, b)
    return records


def _tableau_data(builder, row_words, active_length, *, validity):
    """Build output expressions; the caller copies/phases then reverses history."""
    m, width = len(row_words), len(row_words[0])
    capacity = 1 << (2 * m - 1).bit_length()
    origin_width = max(1, (capacity - 1).bit_length())
    zero = builder.const(0, width)
    records = []
    for i in range(capacity):
        if i < m:
            row, label, kind, live = row_words[i], i + 1, 0, int(i < active_length)
        elif i < 2 * m:
            row, label, kind, live = builder.const(i - m + 1, width), 0, 1, 1
        else:
            row, label, kind, live = zero, 0, 2, 0
        records.append(_Record(row, builder.const(label, width),
                               builder.const(i, origin_width), builder.const(kind, 2),
                               live, zero))

    # Concatenating least-significant fields first gives the stated lexical keys.
    def row_key(record):
        return (record.origin + record.label + record.kind + record.row
                + (builder.bnot(record.live),))

    records = _sort(builder, records, row_key)
    counted = []
    for i, record in enumerate(records):
        if i:
            previous = counted[-1]
            same = builder.all_bits((record.live, previous.live,
                                     builder.eq(record.row, previous.row)))
            prior = builder.mux(same, previous.column, zero)
        else:
            prior = zero
        is_data = builder.eq(record.kind, builder.const(0, 2))
        count = builder.add(prior, (is_data,) + (0,) * (width - 1), width=width)
        count = builder.mux(record.live, count, zero)
        counted.append(_Record(record.row, record.label, record.origin,
                                record.kind, record.live, count))
    records = _sort(builder, counted, lambda record: record.origin)
    columns = tuple(record.column for record in records[:m])
    endpoint = tuple(record.column for record in records[m:2 * m])
    if not validity:
        return columns, endpoint, None

    def data_bit(record):
        return builder.band(record.live, builder.eq(record.kind, builder.const(0, 2)))

    def column_key(record):
        return (record.origin + record.label + record.row + record.column
                + (builder.bnot(data_bit(record)),))

    records = _sort(builder, records, column_key)
    checks = []
    for i, record in enumerate(records):
        current_data = data_bit(record)
        row_range = builder.all_bits((builder.ult(zero, record.row),
                                     builder.ule(record.row, builder.const(m, width))))
        first_row = builder.eq(record.row, builder.const(1, width))
        if i:
            previous = records[i - 1]
            same_column = builder.all_bits((data_bit(previous),
                                             builder.eq(record.column, previous.column)))
            next_row = builder.add(previous.row, builder.const(1, width), width=width)
            follows = builder.all_bits((builder.eq(record.row, next_row),
                                         builder.ult(previous.label, record.label)))
            above = builder.mux(same_column, (follows,), (first_row,))[0]
        else:
            above = first_row
        checks.append(builder.bor(builder.bnot(current_data), builder.band(row_range, above)))
    return columns, endpoint, builder.all_bits(checks)


def _append_tableau_xor(circuit, row_words, output_words, active_length, *, endpoint):
    row_words, m, width = _shape_of_words(row_words, name="row words")
    output_words = tuple(tuple(word) for word in output_words)
    if len(output_words) != m or any(len(word) != width for word in output_words):
        raise ValueError("output shape must match the row words")
    active = m if active_length is None else active_length
    if not isinstance(active, int) or not 0 <= active <= m:
        raise ValueError("active length must be a public integer in 0,...,M")
    inputs = tuple(bit for word in row_words for bit in word)
    targets = tuple(bit for word in output_words for bit in word)
    _bits_ok(inputs)
    if any(not isinstance(bit, Wire) for bit in targets):
        raise ValueError("XOR outputs must be wires")
    if len(set(targets)) != len(targets) or set(targets).intersection(inputs):
        raise ValueError("XOR output wires must be distinct and disjoint from input wires")
    circuit.reserve(bit for bit in inputs + targets if isinstance(bit, Wire))
    builder = BooleanBuilder(circuit)
    columns, histogram, _ = _tableau_data(builder, row_words, active, validity=False)
    result = histogram if endpoint else columns
    builder.copy_then_uncompute(tuple(bit for word in result for bit in word), targets)


def append_cache_xor(circuit: Circuit, row_words, cache_words, active_length=None):
    """Append |rows,cache> -> |rows,cache XOR columns(rows)> with clean helpers.

    Inactive output columns are zero. Input row values outside 1,...,M still
    get their within-row occurrence counts; no Young-tableau promise is used.
    """
    _append_tableau_xor(circuit, row_words, cache_words, active_length, endpoint=False)


def append_endpoint_xor(circuit: Circuit, row_words, endpoint_words, active_length=None):
    """Clean XOR of the M-row histogram; invalid row labels contribute to no bin.

    Counts use only the public active prefix. This is defined and reversible
    on all encoded input rows and arbitrary prior endpoint-target contents.
    """
    _append_tableau_xor(circuit, row_words, endpoint_words, active_length, endpoint=True)


def _cover_predicate(builder, old, new):
    """Exact one-box cover of sizes M-1,M on arbitrary unsigned word arrays."""
    m, width = len(old), len(old[0])
    total_width = width + max(1, m.bit_length())
    old_total = builder.const(0, total_width)
    new_total = builder.const(0, total_width)
    additions = builder.const(0, max(1, m.bit_length()))
    checks = []
    for i in range(m):
        old_total = builder.add(old_total, old[i] + (0,) * (total_width - width),
                                 width=total_width)
        new_total = builder.add(new_total, new[i] + (0,) * (total_width - width),
                                 width=total_width)
        plus_one = builder.add(old[i], builder.const(1, width), width=width)
        added = builder.band(builder.eq(new[i], plus_one), builder.ult(old[i], new[i]))
        additions = builder.add(additions, (added,) + (0,) * (len(additions) - 1),
                                 width=len(additions))
        checks.extend((builder.ule(old[i], builder.const(m, width)),
                       builder.ule(new[i], builder.const(m, width)),
                       builder.bor(builder.eq(old[i], new[i]), added)))
        if i:
            checks.extend((builder.ule(old[i], old[i - 1]),
                           builder.ule(new[i], new[i - 1])))
    checks.extend((builder.eq(old_total, builder.const(m - 1, total_width)),
                   builder.eq(new_total, builder.const(m, total_width)),
                   builder.eq(additions, builder.const(1, len(additions)))))
    return builder.all_bits(checks)


def append_cover_row_xor(circuit: Circuit, old_words, new_words, row_target,
                         valid_target=None):
    """XOR the unique added row (zero off-promise) and optionally the cover bit.

    Both shapes have M padded slots; the old/new total sizes must be M-1/M.
    All internal history cleans on malformed shapes and dirty XOR targets.
    A fixed scan replaces any lookup indexed by the quantum added-row value.
    """
    old_words, m, width = _shape_of_words(old_words, name="old shape")
    new_words = tuple(tuple(word) for word in new_words)
    row_target = tuple(row_target)
    if len(new_words) != m or any(len(word) != width for word in new_words):
        raise ValueError("old and new padded shapes must have matching widths")
    if len(row_target) < max(1, m.bit_length()):
        raise ValueError("added-row output must encode 0,...,M")
    inputs = tuple(bit for word in old_words + new_words for bit in word)
    targets = row_target + (() if valid_target is None else (valid_target,))
    _bits_ok(inputs)
    if any(not isinstance(bit, Wire) for bit in targets):
        raise ValueError("cover XOR targets must be wires")
    if len(set(targets)) != len(targets) or set(targets).intersection(inputs):
        raise ValueError("cover XOR targets must be distinct and disjoint from inputs")
    circuit.reserve(bit for bit in inputs + targets if isinstance(bit, Wire))
    builder = BooleanBuilder(circuit)
    valid = _cover_predicate(builder, old_words, new_words)
    row = (0,) * len(row_target)
    for i in range(m):
        added = builder.band(valid, builder.ult(old_words[i], new_words[i]))
        selected = builder.mux(added, builder.const(i + 1, len(row)), (0,) * len(row))
        row = tuple(builder.bxor(a, b) for a, b in zip(row, selected))
    result = row + (() if valid_target is None else (valid,))
    builder.copy_then_uncompute(result, targets)


def append_promise_phase(circuit: Circuit, shape_words, row_words, tag_words,
                         flag, work_wires, kind="source"):
    """Append +valid/-invalid using separately allocated, exactly cleaned helpers.

    ``tag_words`` is one little-endian tag word, despite the plural name retained
    from the compiler interface. The right tableau and future digits are not
    arguments and are spectators. All supplied work bits and the attenuation
    flag are tested. No tested work wire is borrowed as temporary storage.
    """
    row_words, m, width = _shape_of_words(row_words, name="row words")
    shape_words = tuple(tuple(word) for word in shape_words)
    tag_words, work_wires = tuple(tag_words), tuple(work_wires)
    if len(shape_words) != m or any(len(word) != width for word in shape_words):
        raise ValueError("shape words must match the row-word capacity and width")
    if len(tag_words) < max(1, m.bit_length()):
        raise ValueError("tag word must encode 0,...,M")
    if kind not in ("source", "target"):
        raise ValueError("kind must be source or target")
    inputs = (tuple(bit for words in (row_words, shape_words) for word in words for bit in word)
              + tag_words + (flag,) + work_wires)
    _bits_ok(inputs)
    # Column arithmetic may have released W after uncomputing its valid input.
    # A reflection must nevertheless test EVERY W basis state, including dirty
    # states. Temporarily reserve those physical wires so no helper aliases W.
    formerly_free = circuit.reserve(bit for bit in inputs if isinstance(bit, Wire))
    builder = BooleanBuilder(circuit)
    active = m - 1 if kind == "source" else m
    _, endpoint, valid_tableau = _tableau_data(builder, row_words, active, validity=True)
    checks = [valid_tableau, builder.bnot(flag)]
    checks.extend(builder.bnot(bit) for bit in work_wires)
    if kind == "source":
        checks.extend(builder.eq(endpoint[i], shape_words[i]) for i in range(m))
        checks.extend((builder.eq(row_words[-1], builder.const(0, width)),
                       builder.ult(builder.const(0, len(tag_words)), tag_words),
                       builder.ule(tag_words, builder.const(m, len(tag_words)))))
    else:
        checks.extend((_cover_predicate(builder, shape_words, endpoint),
                       builder.eq(tag_words, builder.const(0, len(tag_words)))))
    invalid = builder.bnot(builder.all_bits(checks))
    if isinstance(invalid, Wire):
        circuit.z(invalid)
    elif invalid:
        circuit.global_phase(pi)
    builder.uncompute()
    circuit.release(formerly_free)
