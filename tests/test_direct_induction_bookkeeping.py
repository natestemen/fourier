"""Independent references may use quadratic scans; claimed schedules may not."""

from itertools import product
from math import log2
from pathlib import Path
import random
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from alternative_qft_probe import add_row, addable_rows, partitions, tableaux  # noqa: E402
from direct_induction_bookkeeping import (  # noqa: E402
    cover_oracle,
    endpoint_oracle,
    erase_old_endpoint_oracle,
    inverse_outer_boundary,
    outer_boundary,
    source_predicate,
    tableau_data_oracle,
    target_predicate,
)


def reference_data(word, active=None):
    """Intentionally simple O(M^2) reference, not part of the resource claim."""
    m = len(word)
    active = m if active is None else active
    columns = tuple(sum(word[j] == word[i] for j in range(i + 1)) if i < active else 0
                    for i in range(m))
    endpoint = tuple(sum(row == r for row in word[:active]) for r in range(1, m + 1))
    valid = all(1 <= row <= m for row in word[:active])
    for length in range(1, active + 1):
        prefix = tuple(sum(row == r for row in word[:length]) for r in range(1, m + 1))
        valid &= all(prefix[r] >= prefix[r + 1] for r in range(m - 1))
    return columns, endpoint, int(valid)


def padded(shape, m):
    return tuple(shape) + (0,) * (m - len(shape))


def reference_cover(old, new, old_size=None):
    m = len(old)
    old_size = m - 1 if old_size is None else old_size
    valid = (sum(old) == old_size and sum(new) == old_size + 1
             and all(0 <= v <= m for v in old + new)
             and all(old[i] >= old[i + 1] and new[i] >= new[i + 1] for i in range(m - 1)))
    delta = tuple(b - a for a, b in zip(old, new))
    valid &= delta.count(1) == 1 and all(d in (0, 1) for d in delta)
    return (1 + delta.index(1), 1) if valid else (0, 0)


@pytest.mark.parametrize("m", range(4))
def test_all_encoded_small_words_and_every_active_length_have_clean_tapes(m):
    bits = max(2, m.bit_length())
    signatures = {}
    for word in product(range(1 << bits), repeat=m):
        for active in range(m + 1):
            result = tableau_data_oracle(word, active_length=active, word_bits=bits)
            assert (result.columns, result.endpoint, result.valid) == reference_data(word, active)
            assert result.clean
            signatures.setdefault(active, result.ledger.schedule_signature)
            assert result.ledger.schedule_signature == signatures[active]
    # Even changing the PUBLIC active length does not alter this padded schedule.
    assert len(set(signatures.values())) == 1


@pytest.mark.parametrize("m", [4, 5, 7, 8, 9, 16, 17, 31, 32, 33])
def test_random_malformed_words_and_arbitrary_xor_targets(m):
    rng = random.Random(20261023 + m)
    bits = m.bit_length() + 1
    signature = None
    for _ in range(30):
        word = tuple(rng.randrange(1 << bits) for _ in range(m))
        active = rng.randrange(m + 1)
        before_columns = tuple(rng.randrange(1 << bits) for _ in range(m))
        before_endpoint = tuple(rng.randrange(1 << bits) for _ in range(m))
        before_valid = rng.randrange(2)
        expected = reference_data(word, active)
        result = tableau_data_oracle(word, active_length=active, word_bits=bits,
                                     initial_columns=before_columns, initial_endpoint=before_endpoint,
                                     initial_valid=before_valid)
        assert result.columns == tuple(a ^ b for a, b in zip(before_columns, expected[0]))
        assert result.endpoint == tuple(a ^ b for a, b in zip(before_endpoint, expected[1]))
        assert result.valid == before_valid ^ expected[2]
        assert result.clean
        erased = tableau_data_oracle(word, active_length=active, word_bits=bits,
                                     initial_columns=result.columns, initial_endpoint=result.endpoint,
                                     initial_valid=result.valid)
        assert (erased.columns, erased.endpoint, erased.valid) == (
            before_columns, before_endpoint, before_valid)
        assert erased.clean
        signature = result.ledger.schedule_signature if signature is None else signature
        assert result.ledger.schedule_signature == signature


@pytest.mark.parametrize("m", range(1, 8))
def test_all_small_valid_tableaux_cache_and_padded_endpoint(m):
    for shape in partitions(m):
        for word in tableaux(shape):
            result = tableau_data_oracle(word)
            assert result.valid == 1 and result.clean
            assert result.endpoint == padded(shape, m)
            assert result.columns == reference_data(word)[0]
            histogram = endpoint_oracle(word)
            assert histogram.values == result.endpoint and histogram.clean


@pytest.mark.parametrize("m", range(2, 8))
def test_legal_adjacent_swaps_update_only_two_cached_coordinates(m):
    for shape in partitions(m):
        for word in tableaux(shape):
            cache = tableau_data_oracle(word).columns
            for j in range(m - 1):
                axial = cache[j + 1] - word[j + 1] - cache[j] + word[j]
                if abs(axial) <= 1:
                    continue
                changed_word = word[:j] + (word[j + 1], word[j]) + word[j + 2:]
                changed_cache = cache[:j] + (cache[j + 1], cache[j]) + cache[j + 2:]
                rebuilt = tableau_data_oracle(changed_word)
                assert rebuilt.valid == 1 and rebuilt.clean
                assert rebuilt.columns == changed_cache
                assert rebuilt.endpoint == padded(shape, m)


@pytest.mark.parametrize("m", [1, 2, 3, 5, 8, 17])
def test_cover_oracle_malformed_arrays_and_xor_cleanup(m):
    rng = random.Random(4183 + m)
    bits = max(2, m.bit_length() + 1)
    pairs = [(padded(shape, m), padded(add_row(shape, row), m))
             for shape in partitions(m - 1) for row in addable_rows(shape)]
    pairs += [(tuple(rng.randrange(1 << bits) for _ in range(m)),
               tuple(rng.randrange(1 << bits) for _ in range(m))) for _ in range(100)]
    signature = None
    for old, new in pairs:
        before_row, before_valid = rng.randrange(1 << bits), rng.randrange(2)
        expected_row, expected_valid = reference_cover(old, new)
        result = cover_oracle(old, new, word_bits=bits, initial_row=before_row,
                              initial_valid=before_valid)
        assert (result.row, result.valid) == (before_row ^ expected_row, before_valid ^ expected_valid)
        assert result.clean
        reverse = cover_oracle(old, new, word_bits=bits, initial_row=result.row,
                               initial_valid=result.valid)
        assert (reverse.row, reverse.valid) == (before_row, before_valid) and reverse.clean
        signature = result.ledger.schedule_signature if signature is None else signature
        assert result.ledger.schedule_signature == signature


@pytest.mark.parametrize("m", range(1, 7))
def test_source_and_target_predicates_on_all_valid_small_bases(m):
    for shape in partitions(m - 1):
        old = padded(shape, m)
        for word in tableaux(shape):
            for index in (1, m):
                source = source_predicate(word + (0,), old, index)
                assert source.value == 1 and source.clean
                assert source_predicate(word + (0,), old, 0).value == 0
        for row in addable_rows(shape):
            for word in tableaux(add_row(shape, row)):
                target = target_predicate(word, old, 0)
                assert target.value == 1 and target.clean
                assert target_predicate(word, old, 1).value == 0


@pytest.mark.parametrize("m", [1, 2, 3, 5, 9])
def test_predicates_match_references_and_clean_all_malformed_data(m):
    rng = random.Random(8721 + m)
    bits = max(2, m.bit_length() + 1)
    signatures = [None, None]
    for _ in range(100):
        word = tuple(rng.randrange(1 << bits) for _ in range(m))
        old = tuple(rng.randrange(1 << bits) for _ in range(m))
        index, before = rng.randrange(1 << bits), rng.randrange(2)
        work = tuple(rng.randrange(2) for _ in range(4))
        flag = rng.randrange(2)
        prefix = reference_data(word, m - 1)
        full = reference_data(word)
        clean_work = flag == 0 and not any(work)
        expected_source = int(prefix[2] and prefix[1] == old and word[-1] == 0
                              and 1 <= index <= m and clean_work)
        expected_target = int(full[2] and reference_cover(old, full[1])[1]
                              and index == 0 and clean_work)
        for j, (oracle, expected) in enumerate(((source_predicate, expected_source),
                                                (target_predicate, expected_target))):
            result = oracle(word, old, index, attenuation_flag=flag, work_bits=work,
                            word_bits=bits, initial=before)
            assert result.value == before ^ expected and result.clean
            reverse = oracle(word, old, index, attenuation_flag=flag, work_bits=work,
                             word_bits=bits, initial=result.value)
            assert reverse.value == before and reverse.clean
            signatures[j] = result.ledger.schedule_signature if signatures[j] is None else signatures[j]
            assert result.ledger.schedule_signature == signatures[j]


@pytest.mark.parametrize("kind", ["source", "target"])
def test_every_work_bit_and_the_attenuation_flag_participate_in_reflections(kind):
    old = (2, 0, 0)
    oracle, word, index = ((source_predicate, (1, 1, 0), 1) if kind == "source"
                           else (target_predicate, (1, 2, 1), 0))
    assert oracle(word, old, index, work_bits=(0,) * 19).value == 1
    for i in range(19):
        work = tuple(int(j == i) for j in range(19))
        result = oracle(word, old, index, work_bits=work)
        assert result.value == 0 and result.clean
    result = oracle(word, old, index, attenuation_flag=1, work_bits=(0,) * 19)
    assert result.value == 0 and result.clean


@pytest.mark.parametrize("m", range(1, 7))
def test_valid_outer_boundary_erases_old_shape_and_roundtrips(m):
    for shape in partitions(m - 1):
        old = padded(shape, m)
        right = tableaux(shape)[0]
        for row in addable_rows(shape):
            new_shape = add_row(shape, row)
            for left in tableaux(new_shape):
                result = outer_boundary(old, left, right)
                assert result.endpoint == padded(new_shape, m)
                assert result.right_path == right + (row,)
                assert result.old_endpoint_residue == (0,) * m
                assert result.workspace_clean
                inverse = inverse_outer_boundary(result.endpoint, result.right_path,
                                                  result.old_endpoint_residue, left)
                assert inverse.old_endpoint == old
                assert inverse.right_prefix == right and inverse.initial_new_row == 0
                assert inverse.cleared_endpoint == (0,) * m and inverse.workspace_clean


def test_boundary_added_row_is_not_the_last_left_symbol():
    result = outer_boundary((2, 0, 0), (1, 2, 1), (1, 1))
    assert result.right_path[-1] == 2
    assert result.right_path[-1] != (1, 2, 1)[-1]
    assert result.old_endpoint_residue == (0, 0, 0)


@pytest.mark.parametrize("m", [1, 2, 3, 5, 8, 17])
def test_malformed_boundary_keeps_residual_and_reverses_every_tape(m):
    rng = random.Random(371 + m)
    bits = max(2, m.bit_length() + 1)
    signature = None
    for _ in range(50):
        old = tuple(rng.randrange(1 << bits) for _ in range(m))
        left = tuple(rng.randrange(1 << bits) for _ in range(m))
        right = tuple(rng.randrange(1 << bits) for _ in range(m - 1))
        initial_row = rng.randrange(1 << bits)
        result = outer_boundary(old, left, right, word_bits=bits, initial_new_row=initial_row)
        inverse = inverse_outer_boundary(result.endpoint, result.right_path,
                                          result.old_endpoint_residue, left, word_bits=bits)
        assert inverse.old_endpoint == old and inverse.right_prefix == right
        assert inverse.initial_new_row == initial_row
        assert inverse.cleared_endpoint == (0,) * m
        assert result.workspace_clean and inverse.workspace_clean
        signature = result.ledger.schedule_signature if signature is None else signature
        assert result.ledger.schedule_signature == signature
        assert result.ledger.schedule_signature == inverse.ledger.schedule_signature
    explicit = outer_boundary((1,) * m, (0,) * m, (0,) * (m - 1), word_bits=bits)
    assert explicit.old_endpoint_residue == (1,) * m
    assert explicit.workspace_clean   # residue is an OUTPUT, not erased scratch


@pytest.mark.parametrize("m", [1, 2, 3, 9, 17])
def test_old_endpoint_xor_is_total_even_for_zero_length_selected_row(m):
    bits = m.bit_length() + 1
    new = (0,) * m
    target = tuple((i * 3) % (1 << bits) for i in range(m))
    result = erase_old_endpoint_oracle(new, 1, target, word_bits=bits)
    assert result.values[0] == target[0] ^ ((1 << bits) - 1)
    reverse = erase_old_endpoint_oracle(new, 1, result.values, word_bits=bits)
    assert reverse.values == target and reverse.clean and result.clean


@pytest.mark.parametrize("m", [0, 1, 2, 3, 7, 8, 9, 31, 32, 33, 127, 256])
def test_fixed_network_operation_ledger_is_near_linear(m):
    result = tableau_data_oracle((1,) * m)
    ledger = result.ledger
    n = 1 << (max(1, 2 * m) - 1).bit_length()
    levels = n.bit_length() - 1
    comparators = n * levels * (levels + 1) // 4
    assert ledger.record_slots == n
    assert ledger.network_sizes == [n] * 6
    assert ledger.comparator_applications == 6 * comparators
    assert ledger.erased_decisions == ledger.peak_sort_tape_bits == 3 * comparators
    assert ledger.word_operations == 46 * n + 2 * m + 1
    assert ledger.scan_steps == 6 * n + m
    assert ledger.word_operation_bound <= 1000 * max(1, m) * (1 + log2(m + 1))**2
    assert ledger.workspace_bit_bound <= 1000 * max(1, m) * (1 + log2(m + 1))**2
    assert result.clean


def test_harness_rejects_width_errors_not_malformed_tableaux():
    assert tableau_data_oracle((0, 3), word_bits=2).valid == 0
    with pytest.raises(ValueError):
        tableau_data_oracle((-1, 1))
    with pytest.raises(ValueError):
        tableau_data_oracle((4, 1), word_bits=2)
    with pytest.raises(ValueError):
        tableau_data_oracle((1, 1), active_length=3)
    with pytest.raises(ValueError):
        tableau_data_oracle((1, 1), initial_columns=(0,))
    with pytest.raises(ValueError):
        source_predicate((1,), (0,), 1, work_bits=(2,))
    with pytest.raises(ValueError):
        target_predicate((), (), 0)
