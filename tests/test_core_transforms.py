"""Validation of the research implementations of core Givens elimination."""

from pathlib import Path
import sys

import numpy as np


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from quasiseparable_core import quasiseparable_core_factor  # noqa: E402
from ulv_circuit import factorization_error  # noqa: E402
from ulv_core_transforms import _compress_core  # noqa: E402


def _replay(A, ops):
    M = A.copy()
    for side, i, j, theta in ops:
        c, s = np.cos(theta), np.sin(theta)
        if side == "L":
            Mi, Mj = M[i, :].copy(), M[j, :].copy()
            M[i, :] = c * Mi - s * Mj
            M[j, :] = s * Mi + c * Mj
        elif side == "R":
            Mi, Mj = M[:, i].copy(), M[:, j].copy()
            M[:, i] = c * Mi + s * Mj
            M[:, j] = -s * Mi + c * Mj
        else:
            raise AssertionError(f"unexpected operation {side}")
    return M


def test_grassmann_compression_uses_r_times_d_rotations():
    rng = np.random.default_rng(4)
    for size, rank in [(7, 1), (8, 3), (11, 6)]:
        basis, _ = np.linalg.qr(rng.standard_normal((size, rank)))
        working = np.eye(size)
        ops = []
        order, count = _compress_core(
            working, basis, list(range(size)), ops
        )
        retired = order[: size - rank]
        transformed_basis = working @ basis

        assert count == rank * (size - rank)
        assert len(ops) == count
        assert np.linalg.norm(transformed_basis[retired, :]) < 2e-12
        assert np.allclose(
            transformed_basis.T @ transformed_basis,
            np.eye(rank),
            atol=2e-12,
        )


def test_exact_rank_one_quasiseparable_factorization_and_replay():
    rng = np.random.default_rng(8)
    k = 32
    A = np.eye(k)
    for i, theta in enumerate(rng.uniform(-1.0, 1.0, k - 1)):
        rotation = np.eye(k)
        c, s = np.cos(theta), np.sin(theta)
        rotation[np.ix_([i, i + 1], [i, i + 1])] = [[c, -s], [s, c]]
        A = rotation @ A

    assert max(
        np.linalg.matrix_rank(A[:split, split:], tol=1e-12)
        for split in range(1, k)
    ) == 1

    ops, final, count, steps = quasiseparable_core_factor(
        A, delta=1e-12, rank_bound=1
    )
    assert count < 2 * k + 2 * (1 + 1) ** 2
    assert factorization_error(final) < 2e-12
    assert np.allclose(_replay(A, ops), final, atol=2e-13)
    assert all(
        step.row_rotations == step.rank * step.retired
        and step.row_rotations + step.column_rotations
        <= 2 * step.retired
        for step in steps
    )
