"""Exact basis-state checks for compact shape schedules and clean edits."""

from random import Random

import pytest

from fourier.compact_shapes import (
    CornerRecord,
    FrobeniusShape,
    clean_relabel,
    clean_update,
    edge_involution,
    xor_equal,
    xor_update,
)
from fourier.diagrams import partitions


def _reference_corners(rows, kind):
    if kind == "add":
        cells = [(i + 1, length + 1) for i, length in enumerate(rows)
                 if i == 0 or rows[i - 1] > length]
        cells.append((len(rows) + 1, 1))
    else:
        cells = [(i + 1, length) for i, length in enumerate(rows)
                 if i + 1 == len(rows) or length > rows[i + 1]]
    return tuple(CornerRecord(True, row, column, column - row) for row, column in cells)


def _reference_edit(rows, row, direction):
    result = list(rows)
    if row == len(result) + 1:
        assert direction == 1
        result.append(1)
    else:
        result[row - 1] += direction
        if result[row - 1] == 0:
            result.pop()
    return tuple(result)


def test_exhaustive_partitions_and_every_box_edge():
    """All 2,714 partitions through 20 boxes, including all removable edges."""
    seen = edges = 0
    for size in range(21):
        for rows in partitions(size):
            shape = FrobeniusShape.from_partition(rows, max_boxes=21)
            assert shape.to_partition() == tuple(rows)
            assert shape.size == size
            assert shape.row_checksum == sum((i + 1) * a for i, a in enumerate(rows))
            assert FrobeniusShape.from_words(shape.to_words(), max_boxes=21) == shape
            for kind, direction in (("add", 1), ("remove", -1)):
                table = shape.corners(kind)
                reference = _reference_corners(rows, kind)
                assert table.valid_records == reference
                assert table.count == len(reference)
                assert len(table.records) == 2 * shape.capacity + 1
                assert all(record == CornerRecord(False)
                           for record in table.records[table.count:])
                assert table.clean
                for index, record in enumerate(reference):
                    assert table.row_to_index(record.row) == index
                    assert table.index_to_row(index) == record.row
                    updated = clean_update(shape, record.row, direction)
                    assert updated.clean and not any(updated.old_register)
                    assert updated.shape.to_partition() == _reference_edit(rows, record.row, direction)
                    assert updated.shape.row_checksum - shape.row_checksum == direction * record.row
                    restored = clean_update(updated.shape, record.row, -direction)
                    assert restored.shape == shape and restored.clean
                    edges += 1
            seen += 1
    assert seen == 2714
    assert edges == 19246


@pytest.mark.parametrize("rows,bound", [([], 0), ([], 1), ([1], 2), ([2, 1], 4),
                                       ([4, 4, 4, 3], 16), ([5] * 5, 26),
                                       ([19], 20), ([1] * 19, 20)])
def test_padding_canonicality_row_lengths_and_word_roundtrip(rows, bound):
    shape = FrobeniusShape.from_partition(rows, max_boxes=bound)
    assert shape.arms[shape.diagonal:] == (0,) * (shape.capacity - shape.diagonal)
    assert shape.legs[shape.diagonal:] == (0,) * (shape.capacity - shape.diagonal)
    for row in range(1, bound + 3):
        expected = rows[row - 1] if row <= len(rows) else 0
        assert shape.row_length(row) == expected
    assert FrobeniusShape.from_words(shape.to_words(), max_boxes=bound) == shape


@pytest.mark.parametrize("rows,row,direction", [([], 1, 1), ([1], 1, -1),
                                               ([3, 3, 2], 3, 1),
                                               ([3, 3, 2], 2, -1),
                                               ([4, 4, 4, 3], 4, 1),
                                               ([4, 4, 4, 4], 4, -1)])
def test_arbitrary_xor_output_and_oriented_edge_involution(rows, row, direction):
    shape = FrobeniusShape.from_partition(rows, max_boxes=25)
    rng = Random(sum(rows) * 37 + row)
    initial = tuple(rng.randrange(1 << shape.word_bits)
                    for _ in range(2 * shape.capacity + 1))
    first = xor_update(shape, row, direction, initial)
    expected = clean_update(shape, row, direction).shape.to_words()
    assert first.output_words == tuple(a ^ b for a, b in zip(initial, expected))
    second = xor_update(shape, row, direction, first.output_words)
    assert second.output_words == initial
    assert first.clean and second.clean
    assert first.scratch_words == second.scratch_words == (0,) * 7
    neighbor, stable_row, opposite = edge_involution(shape, row, direction)
    assert edge_involution(neighbor, stable_row, opposite) == (shape, row, direction)


def test_fixed_counts_ignore_shape_and_corner_count():
    bounds = (1, 8, 16, 31, 64, 100)
    for bound in bounds:
        shapes = [FrobeniusShape.from_partition([], max_boxes=bound),
                  FrobeniusShape.from_partition([bound - 1] if bound > 1 else [], max_boxes=bound)]
        corner_counts = set()
        update_counts = set()
        for shape in shapes:
            for kind in ("add", "remove"):
                table = shape.corners(kind)
                corner_counts.add((table.network_size, table.comparator_applications))
                depth = table.network_size.bit_length() - 1
                assert table.comparator_applications == table.network_size * depth * (depth + 1) // 2
            result = clean_update(shape, 1, 1)
            update_counts.add((result.oracle_calls, result.scan_slots))
        assert len(corner_counts) == len(update_counts) == 1


def test_corner_wrappers_and_invalid_rows_or_indices():
    shape = FrobeniusShape.from_partition([3, 1], max_boxes=8)
    assert shape.corners().contents == (3, 0, -2)
    assert shape.corners("remove").contents == (2, -1)
    for kind in ("add", "remove"):
        for index, row in enumerate(shape.corners(kind).rows):
            assert shape.row_to_corner(row, kind) == index
            assert shape.corner_to_row(index, kind) == row
            forward = clean_relabel(shape, row, kind=kind)
            backward = clean_relabel(shape, index, kind=kind, to_corner=False)
            assert forward.label == index and backward.label == row
            assert forward.clean and backward.clean
            assert forward.erased_input == backward.erased_input == 0
            assert not any(forward.erased_table_words + backward.erased_table_words)
        with pytest.raises(ValueError):
            shape.row_to_corner(100, kind)
        with pytest.raises(ValueError):
            shape.corner_to_row(shape.corners(kind).count, kind)
    assert shape.add_box(2).remove_box(2) == shape
    with pytest.raises(ValueError):
        shape.corners("invalid")


def test_equality_fixed_scan_and_xor_cleanup():
    shapes = [FrobeniusShape.from_partition(rows, max_boxes=10)
              for rows in ([], [1], [3, 1], [4], [3, 1])]
    for first in shapes:
        for second in shapes:
            for initial in (False, True):
                result = xor_equal(first, second, initial)
                expected = first.to_partition() == second.to_partition()
                assert result.flag == (initial ^ expected)
                assert first.same_shape(second) == expected
                assert result.clean and result.mismatch_scratch == 0
                assert result.word_comparisons == 2 * (2 * first.capacity + 1)
                assert xor_equal(first, second, result.flag).flag == initial
    with pytest.raises(ValueError):
        xor_equal(shapes[0], FrobeniusShape.from_partition([], max_boxes=11))


@pytest.mark.parametrize("args", [
    (4, 0, (1, 0), (0, 0)),  # nonzero padding
    (4, 2, (0, 0), (1, 0)),  # non-strict arms
    (4, 1, (-1, 0), (0, 0)),
    (4, 1, (4, 0), (0, 0)),  # too many boxes
    (4, 3, (2, 1), (2, 1)),
    (4, 1, (0,), (0,)),
])
def test_bad_representations(args):
    with pytest.raises(ValueError):
        FrobeniusShape(*args)


def test_invalid_update_promises_and_output_words():
    shape = FrobeniusShape.from_partition([2, 2], max_boxes=5)
    with pytest.raises(ValueError):
        shape.add_box(2)
    with pytest.raises(ValueError):
        shape.remove_box(1)
    with pytest.raises(ValueError):
        shape.add_box(100)
    with pytest.raises(ValueError):
        shape.remove_box(0)
    with pytest.raises(ValueError):
        xor_update(shape, 1, 0)
    with pytest.raises(ValueError):
        xor_update(shape, 1, 1, (0,))
    with pytest.raises(ValueError):
        xor_update(shape, 1, 1, (1 << shape.word_bits,) * len(shape.to_words()))
    with pytest.raises(ValueError):
        shape.add_box(1).add_box(1)
    with pytest.raises(ValueError):
        FrobeniusShape.from_partition([1, 2])
    with pytest.raises(ValueError):
        FrobeniusShape.from_partition([2], max_boxes=1)
