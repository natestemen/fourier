"""Fixed-capacity Young-shape primitives without addressed quantum memory.

``FrobeniusShape`` stores the lengths of the arms and legs of diagonal boxes.
For a public maximum size M there are D=floor(sqrt(M)) slots in each array,
each with O(log(M+1)) bits.  Active entries are strictly decreasing and all
inactive entries are zero.  Thus the representation is canonical.

The algorithms below are basis-state verifiers of *word-level reversible
schedules*, not a Toffoli/Clifford+T compiler.  Every shape-dependent choice
is a control on a fixed scan or comparator network; Python conditionals are
not permission to choose a different coherent circuit.  No array address is
computed from a quantum row or corner index.  Row lookup and box updates
use O(D) word operations; corner packing uses O(D log^2(D+1)) fixed
comparators.  Standard reversible O(poly(log M))-gate arithmetic therefore
gives O(sqrt(M) polylog(M)) gates and workspace for these primitives.

``xor_update`` computes the updated shape into arbitrary XOR output, then
explicitly reverses its row-lookup and arithmetic scratch.  ``clean_update``
uses that oracle and its inverse-direction oracle to erase the old shape,
leaving no box-edit history.  Its clean-output promise applies to valid
Young-graph edges: removing the same row reverses an addition.  It does not
claim that shape addition is a permutation on *all* bitstrings.  A reversible
XOR oracle has a total extension on invalid encodings, and the circuit may
leave garbage outside its promised input subspace.  ``edge_involution`` is
an explicit bijection on valid oriented edges, including the direction bit.

``from_partition`` and ``to_partition`` are convenience/test conversions;
they read or emit an expanded list of rows and are NOT used by the fast
primitives or counted as sublinear conversions.  Maintaining this encoding
throughout the QFT, rather than rebuilding it at each embedding, is essential.
"""

from dataclasses import dataclass
from math import isqrt
from typing import Literal


CornerKind = Literal["add", "remove"]


@dataclass(frozen=True)
class CornerRecord:
    valid: bool
    row: int = 0
    column: int = 0
    content: int = 0


ZERO_CORNER = CornerRecord(False)


@dataclass(frozen=True)
class CornerTable:
    """Canonical descending-content table with a public padded slot count."""

    records: tuple[CornerRecord, ...]
    count: int
    network_size: int
    comparator_applications: int
    clean: bool

    @property
    def valid_records(self) -> tuple[CornerRecord, ...]:
        """Convenience view, not a shape-dependent coherent allocation."""
        return tuple(record for record in self.records if record.valid)

    @property
    def contents(self) -> tuple[int, ...]:
        return tuple(record.content for record in self.records if record.valid)

    @property
    def rows(self) -> tuple[int, ...]:
        return tuple(record.row for record in self.records if record.valid)

    def row_to_index(self, row: int) -> int:
        result = matches = 0
        for index in range(len(self.records)):
            hit = self.records[index].valid and self.records[index].row == row
            result += index * int(hit)
            matches += int(hit)
        if matches != 1:
            raise ValueError("row is not a valid corner")
        return result

    def index_to_row(self, index: int) -> int:
        result = matches = 0
        for slot in range(len(self.records)):
            hit = self.records[slot].valid and index == slot
            result += self.records[slot].row * int(hit)
            matches += int(hit)
        if matches != 1:
            raise ValueError("index is not a valid corner")
        return result


def _bitonic_network(size: int) -> tuple[tuple[int, int, bool], ...]:
    """Public, input-independent comparator addresses."""
    result = []
    width = 2
    while width <= size:
        distance = width // 2
        while distance:
            for first in range(size):
                second = first ^ distance
                if second > first:
                    result.append((first, second, (first & width) == 0))
            distance //= 2
        width *= 2
    return tuple(result)


def _pack_corners(candidates: tuple[CornerRecord, ...]) -> CornerTable:
    """Sort/copy/unsort, explicitly erase all saved comparator decisions."""
    size = 1 << (max(1, len(candidates)) - 1).bit_length()
    initial = [(candidate, i) for i, candidate in enumerate(candidates)]
    initial += [(ZERO_CORNER, i) for i in range(len(initial), size)]
    work = initial.copy()
    network = _bitonic_network(size)

    def predicate(first, second, ascending):
        def key(item):
            record, origin = item
            return (not record.valid, -record.content, record.row, origin)
        left, right = key(work[first]), key(work[second])
        return left > right if ascending else left < right

    tape = []
    for first, second, ascending in network:
        swap = predicate(first, second, ascending)
        tape.append(swap)
        if swap:
            work[first], work[second] = work[second], work[first]
    # A fixed-position XOR into initially zero output, not addressed scatter.
    output = tuple(work[i][0] for i in range(len(candidates)))
    count = sum(int(record.valid) for record in output)
    for step in range(len(network) - 1, -1, -1):
        first, second, ascending = network[step]
        if tape[step]:
            work[first], work[second] = work[second], work[first]
        recovered = predicate(first, second, ascending)
        if recovered != tape[step]:
            raise ArithmeticError("corner-sort tape failed to reverse")
        tape[step] ^= recovered
    clean = work == initial and not any(tape)
    if not clean:
        raise ArithmeticError("corner extraction left sorting history")
    return CornerTable(output, count, size, 2 * len(network), clean)


@dataclass(frozen=True)
class FrobeniusShape:
    max_boxes: int
    diagonal: int
    arms: tuple[int, ...]
    legs: tuple[int, ...]

    def __post_init__(self):
        if not isinstance(self.max_boxes, int) or self.max_boxes < 0:
            raise ValueError("max_boxes must be a nonnegative public integer")
        if (not isinstance(self.diagonal, int)
                or not 0 <= self.diagonal <= self.capacity
                or len(self.arms) != self.capacity or len(self.legs) != self.capacity):
            raise ValueError("invalid diagonal count or fixed array capacity")
        for values in (self.arms, self.legs):
            for i in range(self.capacity):
                if not isinstance(values[i], int) or values[i] < 0:
                    raise ValueError("arms and legs must be nonnegative integers")
                if i >= self.diagonal and values[i] != 0:
                    raise ValueError("inactive slots must be zero")
                if i + 1 < self.diagonal and values[i] <= values[i + 1]:
                    raise ValueError("active arms and legs must strictly decrease")
        if self.size > self.max_boxes:
            raise ValueError("shape exceeds public max_boxes")

    @property
    def capacity(self) -> int:
        return isqrt(self.max_boxes)

    @property
    def word_bits(self) -> int:
        return max(1, self.max_boxes.bit_length())

    @property
    def size(self) -> int:
        return sum((1 + self.arms[i] + self.legs[i]) * int(i < self.diagonal)
                   for i in range(self.capacity))

    @property
    def row_checksum(self) -> int:
        """Sum of row numbers of every box, computed without expanded rows."""
        return sum(((i + 1) * (1 + self.arms[i] + self.legs[i])
                    + self.legs[i] * (self.legs[i] + 1) // 2)
                   * int(i < self.diagonal) for i in range(self.capacity))

    @classmethod
    def from_partition(cls, partition, *, max_boxes: int | None = None):
        """Expanded-input conversion; outside the fast coherent interface."""
        rows = tuple(partition)
        if (any(not isinstance(value, int) or value <= 0 for value in rows)
                or any(rows[i] < rows[i + 1] for i in range(len(rows) - 1))):
            raise ValueError("partition rows must be positive and nonincreasing")
        size = sum(rows)
        bound = size if max_boxes is None else max_boxes
        if not isinstance(bound, int) or bound < size:
            raise ValueError("max_boxes is smaller than partition size")
        capacity = isqrt(bound)
        diagonal = sum(value >= i + 1 for i, value in enumerate(rows))
        arms = tuple(rows[i] - i - 1 for i in range(diagonal))
        legs = tuple(sum(value >= i + 1 for value in rows) - i - 1
                     for i in range(diagonal))
        return cls(bound, diagonal, arms + (0,) * (capacity - diagonal),
                   legs + (0,) * (capacity - diagonal))

    def to_partition(self) -> tuple[int, ...]:
        """Expanded-output conversion; do not call inside fast primitives."""
        height = self.legs[0] + 1 if self.diagonal else 0
        return tuple(self.row_length(row) for row in range(1, height + 1))

    def to_words(self) -> tuple[int, ...]:
        return (self.diagonal,) + self.arms + self.legs

    @classmethod
    def from_words(cls, words, *, max_boxes: int):
        capacity = isqrt(max_boxes)
        words = tuple(words)
        if len(words) != 2 * capacity + 1:
            raise ValueError("wrong number of shape words")
        return cls(max_boxes, words[0], words[1:capacity + 1], words[capacity + 1:])

    def row_length(self, row: int) -> int:
        if not isinstance(row, int) or row < 1:
            raise ValueError("row must be a positive integer")
        top = bottom = 0
        for i in range(self.capacity):
            active = i < self.diagonal
            top += (self.arms[i] + i + 1) * int(active and row == i + 1)
            bottom += int(active and self.legs[i] + i + 1 >= row)
        return top if row <= self.diagonal else bottom

    def corners(self, kind: CornerKind = "add") -> CornerTable:
        if kind not in ("add", "remove"):
            raise ValueError("corner kind must be 'add' or 'remove'")
        candidates = []
        for i in range(self.capacity):
            active = i < self.diagonal
            arm, leg, rank = self.arms[i], self.legs[i], i + 1
            # Boundary selection is public, not a shape-dependent array read.
            next_arm = self.arms[i + 1] if i + 1 < self.capacity else 0
            next_leg = self.legs[i + 1] if i + 1 < self.capacity else 0
            if kind == "add":
                av = active and (i == 0 or self.arms[i - 1] >= arm + 2)
                lv = active and (i == 0 or self.legs[i - 1] >= leg + 2)
                ac = CornerRecord(True, rank, rank + arm + 1, arm + 1)
                lc = CornerRecord(True, rank + leg + 1, rank, -leg - 1)
            else:
                av = (active and arm > 0 and
                      (rank == self.diagonal or arm >= next_arm + 2))
                lv = (active and leg > 0 and
                      (rank == self.diagonal or leg >= next_leg + 2))
                ac = CornerRecord(True, rank, rank + arm, arm)
                lc = CornerRecord(True, rank + leg, rank, -leg)
            candidates.extend((ac if av else ZERO_CORNER, lc if lv else ZERO_CORNER))
        # Read the last active pair by scanning, not by addressed array access.
        last_arm = last_leg = 0
        for i in range(self.capacity):
            last_arm += self.arms[i] * int(i + 1 == self.diagonal)
            last_leg += self.legs[i] * int(i + 1 == self.diagonal)
        if kind == "add":
            valid = self.diagonal == 0 or last_arm > 0 and last_leg > 0
            rank = self.diagonal + 1
        else:
            valid = self.diagonal > 0 and last_arm == last_leg == 0
            rank = self.diagonal
        candidates.append(CornerRecord(True, rank, rank, 0) if valid else ZERO_CORNER)
        return _pack_corners(tuple(candidates))

    def row_to_corner(self, row: int, kind: CornerKind = "add") -> int:
        return clean_relabel(self, row, kind=kind, to_corner=True).label

    def corner_to_row(self, index: int, kind: CornerKind = "add") -> int:
        return clean_relabel(self, index, kind=kind, to_corner=False).label

    def same_shape(self, other: "FrobeniusShape") -> bool:
        """Fixed-scan equality, unlike a short-circuit Python tuple comparison."""
        return xor_equal(self, other).flag

    def add_box(self, row: int):
        return clean_update(self, row, 1).shape

    def remove_box(self, row: int):
        return clean_update(self, row, -1).shape


@dataclass(frozen=True)
class UpdateOracleResult:
    output_words: tuple[int, ...]
    scratch_words: tuple[int, ...]
    scan_slots: int
    output_word_xors: int
    clean: bool


def xor_update(shape: FrobeniusShape, row: int, direction: int,
               initial_output: tuple[int, ...] | None = None) -> UpdateOracleResult:
    """Fixed-scan XOR oracle for a valid edge, explicitly reverse scratch.

    The direction and array capacity are public circuit choices.  Row, shape,
    comparisons and updates are coherent controls.  Input-promise validation
    raises Python errors here; it is not a measurement inside the circuit.
    """
    if direction not in (-1, 1):
        raise ValueError("direction must be +1 or -1")
    length = shape.row_length(row)
    if direction == 1:
        valid = row == 1 or shape.row_length(row - 1) > length
        valid = valid and shape.size < shape.max_boxes
    else:
        valid = length > 0 and length > shape.row_length(row + 1)
    if not valid:
        raise ValueError("requested box edit is not a valid edge within max_boxes")
    output = list((0,) * (2 * shape.capacity + 1)
                  if initial_output is None else initial_output)
    limit = 1 << shape.word_bits
    if (len(output) != 2 * shape.capacity + 1
            or any(not isinstance(word, int) or not 0 <= word < limit for word in output)):
        raise ValueError("initial_output has invalid word count or width")

    # Compute shared row lookup once; doing a complete lookup separately for
    # each output word would introduce an avoidable quadratic cost.
    top = bottom = 0
    for i in range(shape.capacity):
        active = i < shape.diagonal
        top += (shape.arms[i] + i + 1) * int(active and row == i + 1)
        bottom += int(active and shape.legs[i] + i + 1 >= row)
    column = (top if row <= shape.diagonal else bottom) + int(direction == 1)
    above, below, diagonal = row < column, row > column, row == column

    temporary = shape.diagonal
    temporary += direction * int(diagonal)
    output[0] ^= temporary
    temporary -= direction * int(diagonal)
    temporary ^= shape.diagonal
    for i in range(shape.capacity):
        arm_delta = direction * int(above and row == i + 1)
        temporary ^= shape.arms[i]
        temporary += arm_delta
        output[i + 1] ^= temporary
        temporary -= arm_delta
        temporary ^= shape.arms[i]
        leg_delta = direction * int(below and column == i + 1)
        temporary ^= shape.legs[i]
        temporary += leg_delta
        output[shape.capacity + i + 1] ^= temporary
        temporary -= leg_delta
        temporary ^= shape.legs[i]
    # Reverse the shared lookup and flags.  The original shape and row are
    # unchanged, so the inverse uses the same controls with no saved history.
    above ^= row < column
    below ^= row > column
    diagonal ^= row == column
    column -= (top if row <= shape.diagonal else bottom) + int(direction == 1)
    for i in range(shape.capacity - 1, -1, -1):
        active = i < shape.diagonal
        bottom -= int(active and shape.legs[i] + i + 1 >= row)
        top -= (shape.arms[i] + i + 1) * int(active and row == i + 1)
    scratch = (top, bottom, column, temporary, int(above), int(below), int(diagonal))
    clean = not any(scratch)
    if not clean:
        raise ArithmeticError("box edit did not clean arithmetic scratch")
    return UpdateOracleResult(tuple(output), scratch, 3 * shape.capacity,
                              2 * shape.capacity + 1, clean)


@dataclass(frozen=True)
class CleanUpdateResult:
    shape: FrobeniusShape
    old_register: tuple[int, ...]
    oracle_calls: int
    scan_slots: int
    clean: bool


def clean_update(shape: FrobeniusShape, row: int, direction: int) -> CleanUpdateResult:
    """Compute new shape, swap, erase old shape by the inverse edge oracle."""
    forward = xor_update(shape, row, direction)
    updated = FrobeniusShape.from_words(forward.output_words, max_boxes=shape.max_boxes)
    reverse = xor_update(updated, row, -direction, shape.to_words())
    clean = forward.clean and reverse.clean and not any(reverse.output_words)
    if not clean:
        raise ArithmeticError("box edit retained the old shape or edit history")
    return CleanUpdateResult(updated, reverse.output_words, 2,
                             forward.scan_slots + reverse.scan_slots, clean)


def edge_involution(shape: FrobeniusShape, row: int, direction: int):
    """Bijective valid-edge map (shape,row,direction) -> (neighbor,row,-direction)."""
    return clean_update(shape, row, direction).shape, row, -direction


CompactShape = FrobeniusShape


@dataclass(frozen=True)
class EqualityResult:
    flag: bool
    mismatch_scratch: int
    word_comparisons: int
    clean: bool


def xor_equal(first: FrobeniusShape, second: FrobeniusShape,
              initial_flag: bool = False) -> EqualityResult:
    """Compute equality into an XOR flag, then erase the mismatch counter."""
    if first.max_boxes != second.max_boxes:
        raise ValueError("shape registers must have the same public capacity bound")
    left, right = first.to_words(), second.to_words()
    mismatches = 0
    for i in range(len(left)):
        mismatches += int(left[i] != right[i])
    flag = bool(initial_flag) ^ (mismatches == 0)
    for i in range(len(left) - 1, -1, -1):
        mismatches -= int(left[i] != right[i])
    return EqualityResult(flag, mismatches, 2 * len(left), mismatches == 0)


@dataclass(frozen=True)
class RelabelResult:
    label: int
    erased_input: int
    erased_table_words: tuple[int, ...]
    scan_slots: int
    corner_comparators: int
    clean: bool


def clean_relabel(shape: FrobeniusShape, label: int, *, kind: CornerKind = "add",
                  to_corner: bool = True) -> RelabelResult:
    """Bijective valid row/corner-index conversion, with its old label erased.

    Compute the new label by a full table scan, swap, then XOR the inverse
    lookup into the old-label register.  The controlling shape is unchanged.
    The preserved table can be uncomputed from it; no source-index history
    survives.  Invalid labels are outside the clean-output promise.
    """
    table = shape.corners(kind)
    if to_corner:
        output = table.row_to_index(label)
        erased = label ^ table.index_to_row(output)
    else:
        output = table.index_to_row(label)
        erased = label ^ table.row_to_index(output)
    # Re-run the same fixed extraction oracle into the table's XOR target.
    # This explicitly models erasure instead of relying on Python discarding
    # the object. Signed content words cancel under two's-complement XOR too.
    regenerated = shape.corners(kind)
    erased_table = []
    for i in range(len(table.records)):
        first, second = table.records[i], regenerated.records[i]
        erased_table.extend((int(first.valid) ^ int(second.valid),
                             first.row ^ second.row,
                             first.column ^ second.column,
                             first.content ^ second.content))
    clean = table.clean and regenerated.clean and erased == 0 and not any(erased_table)
    if not clean:
        raise ArithmeticError("corner relabeling retained its old input")
    return RelabelResult(output, erased, tuple(erased_table), 2 * len(table.records),
                         table.comparator_applications + regenerated.comparator_applications, clean)
