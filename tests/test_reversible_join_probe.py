"""Basis-state checks of fixed wiring, missing IDs, and clean uncomputation."""

from pathlib import Path
from itertools import product
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from reversible_join_probe import (  # noqa: E402
    QueryRecord,
    SourceRecord,
    bitonic_comparators,
    reversible_batched_join,
)


def test_missing_source_duplicate_queries_invalid_slots_and_payload_zero():
    sources = [SourceRecord(9, (17, 2)), SourceRecord(3, (0, 0)),
               SourceRecord(3, (255, 3), valid=False), SourceRecord(1, (7, 8))]
    queries = [QueryRecord(9), QueryRecord(2), QueryRecord(3), QueryRecord(9),
               QueryRecord(1), QueryRecord(1, valid=False), QueryRecord(0)]
    joined = reversible_batched_join(sources, queries, payload_words=2, payload_bits=8)
    assert joined.outputs == ((17, 2), (0, 0), (0, 0), (17, 2), (7, 8), (0, 0), (0, 0))
    assert joined.clean
    assert joined.padded_records == 16
    assert joined.comparators_per_sort == 16 * 4 * 5 // 4
    assert joined.comparator_applications == 4 * joined.comparators_per_sort


@pytest.mark.parametrize("source_slots,query_slots", [(0, 0), (0, 5), (5, 0),
                                                       (1, 1), (7, 13), (32, 65)])
def test_random_records_and_xor_output_involution(source_slots, query_slots):
    rng = np.random.default_rng(893 + source_slots)
    ids = rng.choice(500, size=source_slots, replace=False)
    sources = [SourceRecord(int(key), tuple(map(int, rng.integers(0, 2**16, 3))),
                            valid=bool(rng.integers(0, 2))) for key in ids]
    queries = [QueryRecord(int(key), valid=bool(rng.integers(0, 2)))
               for key in rng.integers(0, 500, query_slots)]
    if source_slots and query_slots:
        queries[0] = QueryRecord(sources[0].key)
    before = tuple(tuple(map(int, row)) for row in rng.integers(0, 2**16, (query_slots, 3)))
    original_sources, original_queries = tuple(sources), tuple(queries)
    result = reversible_batched_join(sources, queries, payload_words=3,
                                    payload_bits=16, initial_output=before)
    # Quadratic test oracle is deliberately outside the fixed-work algorithm.
    for i, query in enumerate(queries):
        matching = [s for s in sources if s.valid and query.valid and s.key == query.key]
        wanted = matching[0].payload if matching else (0, 0, 0)
        assert result.outputs[i] == tuple(x ^ y for x, y in zip(before[i], wanted))
    reverse = reversible_batched_join(sources, queries, payload_words=3,
                                     payload_bits=16, initial_output=result.outputs)
    assert reverse.outputs == before
    assert result.clean and reverse.clean
    assert tuple(sources) == original_sources
    assert tuple(queries) == original_queries


def test_network_wiring_and_operation_counts_ignore_values_and_valid_bits():
    first = reversible_batched_join(
        [SourceRecord(8, (1,)), SourceRecord(2, (2,))],
        [QueryRecord(8), QueryRecord(2), QueryRecord(3)])
    second = reversible_batched_join(
        [SourceRecord(1, (255,), valid=False), SourceRecord(0, (0,), valid=False)],
        [QueryRecord(55, valid=False), QueryRecord(0), QueryRecord(0)])
    assert first.comparator_applications == second.comparator_applications
    assert first.scan_steps == second.scan_steps
    assert bitonic_comparators(first.padded_records) == bitonic_comparators(second.padded_records)


def test_bad_input_promises():
    with pytest.raises(ValueError, match="unique"):
        reversible_batched_join([SourceRecord(1, (1,)), SourceRecord(1, (2,))], [])
    with pytest.raises(ValueError):
        reversible_batched_join([SourceRecord(1, (256,))], [], payload_bits=8)
    with pytest.raises(ValueError):
        bitonic_comparators(3)


def test_small_basis_record_exhaustion():
    """Exhaust small valid masks and query addresses, not just random hits."""
    for first_valid, second_valid in product([False, True], repeat=2):
        sources = [SourceRecord(2, (3,), second_valid), SourceRecord(0, (1,), first_valid)]
        for keys in product(range(4), repeat=3):
            result = reversible_batched_join(sources, [QueryRecord(key) for key in keys],
                                            payload_bits=2)
            expected = tuple((1 if key == 0 and first_valid else
                              3 if key == 2 and second_valid else 0,) for key in keys)
            assert result.outputs == expected
            assert result.clean
