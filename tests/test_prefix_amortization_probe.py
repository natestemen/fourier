from fractions import Fraction
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from prefix_amortization_probe import (  # noqa: E402
    aligned_box_transport,
    alpha_squared,
    cache_cleanup_collision,
    corner_contents,
    long_many_corner_path,
    orthogonal_transport_correction,
    resolvent,
    staircase_transport,
)
from streaming_induction import add_row, addable_rows  # noqa: E402
from fourier.diagrams import partitions


@pytest.mark.parametrize("n", [2, 4, 8, 16, 31, 64, 127, 256])
def test_many_corner_prefix_witness(n):
    witness = long_many_corner_path(n)
    assert witness.staircase_size <= n // 2
    assert len(witness.word) == n
    assert set(witness.corner_counts[witness.staircase_size - 1:]) == {
        witness.staircase_rows + 1
    }
    assert witness.total_corner_records >= witness.persistent_tail_records


def test_exact_one_box_resolvent_and_residue_updates():
    for n in range(9):
        shapes = [()] if n == 0 else [tuple(p) for p in partitions(n)]
        for shape in shapes:
            old_x, old_y = corner_contents(shape)
            for row in addable_rows(shape):
                new_shape = add_row(shape, row)
                c = (shape[row - 1] if row <= len(shape) else 0) + 1 - row
                for z in (n + 4, n + 7, -n - 5):
                    update = Fraction((z - c)**2, (z - c)**2 - 1)
                    assert resolvent(new_shape, z) == resolvent(shape, z) * update
                new_x, new_y = corner_contents(new_shape)
                for x in set(old_x) & set(new_x):
                    assert abs(x - c) >= 2
                    assert alpha_squared(new_shape, x) == alpha_squared(shape, x) * Fraction(
                        (x - c)**2, (x - c)**2 - 1)
                assert all(abs(y - c) >= 2 for y in set(old_y) & set(new_y))


def test_prefix_cache_cannot_be_erased_by_output_controlled_cleanup():
    witness = cache_cleanup_collision()
    assert witness["left_prefix"] != witness["right_prefix"]
    assert np.isclose(abs(witness["left_amplitude"])**2, 3 / 32)
    assert np.isclose(abs(witness["right_amplitude"])**2, 1 / 32)


def test_one_box_transport_is_not_an_exact_rank_two_update():
    a, b, _, _ = staircase_transport(4, 0)
    singular_values = np.linalg.svd(b @ a.T - np.eye(4), compute_uv=False)
    assert np.allclose(singular_values, [0.1727643514, 0.1727643514,
                                         0.0002648376, 0.0002648376], atol=1e-10)
    assert singular_values[-1] > 2e-4


@pytest.mark.parametrize("k", [4, 8, 16, 32, 64])
@pytest.mark.parametrize("order", [0, 1, 2, 3, 5])
def test_constructive_low_rank_transport_bound(k, order):
    a, b, approximation, bound = staircase_transport(k, order)
    error = np.linalg.norm(b - a - approximation, ord=2)
    assert error <= bound + 2e-12
    assert np.linalg.matrix_rank(approximation, tol=1e-11) <= 2 * order + 3
    assert np.linalg.norm(b @ a.T - np.eye(k) - approximation @ a.T, ord=2) <= bound + 5e-12


@pytest.mark.parametrize("order", [0, 1, 3])
def test_aligned_transport_for_every_small_young_edge(order):
    seen_changes = set()
    for n in range(10):
        shapes = [()] if n == 0 else [tuple(p) for p in partitions(n)]
        for shape in shapes:
            for row in addable_rows(shape):
                result = aligned_box_transport(shape, row, order)
                seen_changes.add(result.new_size - result.old_size)
                assert result.exceptional_rank_bound <= 4
                assert np.linalg.matrix_rank(result.approximation, tol=1e-11) <= result.rank_bound
                assert np.linalg.norm(result.new - result.old - result.approximation,
                                      ord=2) <= result.bound + 2e-12
                identity = np.eye(len(result.old))
                assert np.linalg.norm(result.old @ result.old.T - identity, ord=2) < 2e-12
                assert np.linalg.norm(result.new @ result.new.T - identity, ord=2) < 2e-12
    assert seen_changes == {-1, 0, 1}


@pytest.mark.parametrize("d", [8, 16, 32])
@pytest.mark.parametrize("order", [0, 2, 5, 10])
def test_aligned_transport_large_growing_and_shrinking_corner_counts(d, order):
    staircase = tuple(range(d, 0, -1))
    fat_staircase = tuple(value for value in range(2*d, 0, -2) for _ in range(2))
    for shape, row, expected_change in [(staircase, d // 2, -1), (fat_staircase, 1, 1)]:
        result = aligned_box_transport(shape, row, order)
        assert result.new_size - result.old_size == expected_change
        assert result.exceptional_rank_bound == 4
        assert np.linalg.matrix_rank(result.approximation, tol=1e-10) <= result.rank_bound
        error = np.linalg.norm(result.new - result.old - result.approximation, ord=2)
        assert error <= result.bound + 5e-12


@pytest.mark.parametrize("order", [0, 1, 2, 5])
@pytest.mark.parametrize("kind", ["grow", "same", "shrink"])
def test_polar_correction_is_orthogonal_close_and_small_subspace(order, kind):
    d = 32
    staircase = tuple(range(d, 0, -1))
    fat_staircase = tuple(value for value in range(2*d, 0, -2) for _ in range(2))
    shape, row = {"grow": (fat_staircase, 1), "same": (staircase, 1),
                  "shrink": (staircase, d // 2)}[kind]
    result = aligned_box_transport(shape, row, order)
    correction, update = orthogonal_transport_correction(result)
    identity = np.eye(len(result.old))
    target = result.new @ result.old.T
    actual_error = np.linalg.norm(target - identity - update, ord=2)
    assert result.bound < 1
    assert np.linalg.norm(correction @ correction.T - identity, ord=2) < 2e-13
    assert np.linalg.norm(correction - target, ord=2) <= 2 * actual_error + 2e-12
    assert np.linalg.norm(correction - target, ord=2) <= 2 * result.bound + 2e-12
    # Span(ran R + ran R.T), followed by its numerical orthogonal complement.
    vectors, singular_values, _ = np.linalg.svd(np.concatenate([update, update.T], axis=1),
                                               full_matrices=True)
    support_dimension = np.count_nonzero(singular_values > 1e-11)
    assert support_dimension <= 2 * result.rank_bound
    complement = vectors[:, support_dimension:]
    assert np.linalg.norm((correction - identity) @ complement) < 1e-10
