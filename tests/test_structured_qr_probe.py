"""Validate the experimental generator-only coherent angle construction."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from structured_qr_probe import generator_qr  # noqa: E402
from fourier.amatrix import _a_matrix_from_contents_log


@pytest.mark.parametrize("k", [1, 2, 3, 4, 8, 17, 32, 64])
@pytest.mark.parametrize("family", ["uniform", "integer", "real"])
def test_content_formula_reconstructed_by_fixed_rotation_word(k, family):
    rng = np.random.default_rng(901 + k)
    gaps = (np.ones(2*k-2) if family == "uniform" else
            rng.integers(1, 20, size=2*k-2) if family == "integer" else
            np.exp(rng.uniform(-3, 3, size=2*k-2)))
    merged = np.r_[0., np.cumsum(gaps)][::-1]
    ac, rc = merged[::2], merged[1::2]
    result = generator_qr(ac, rc)
    reference = _a_matrix_from_contents_log(ac, rc)
    assert len(result.gates) == k * (k - 1) // 2
    assert all(g.j == g.i + 1 for g in result.gates)
    assert np.all(np.isin(result.signs, [-1., 1.]))
    np.testing.assert_allclose(result.matrix(), reference, atol=2e-10, rtol=0)
    np.testing.assert_allclose(result.matrix().T @ result.matrix(), np.eye(k),
                               atol=2e-13, rtol=0)


def test_invalid_content_sequences():
    for ac, rc in [([], []), ([1, 0], []), ([1, 0], [2]), ([1, 0], [0])]:
        with pytest.raises(ValueError):
            generator_qr(ac, rc)


@pytest.mark.parametrize("k", [4, 8, 16])
def test_reversed_interlacing_bounds_the_last_generator_divisor(k):
    """Check the analytic guard bound using independent dense suffix QR."""
    rng = np.random.default_rng(1200 + k)
    gaps = rng.integers(1, 20, size=2 * k - 2)
    merged = np.r_[0.0, np.cumsum(gaps)][::-1]
    ac, rc = merged[::2], merged[1::2]
    target = _a_matrix_from_contents_log(ac, rc)
    span, gap = max(1.0, ac[0] - ac[-1]), np.min(gaps)
    eta = ac[0] + span
    row_nodes = 1 / (ac - eta)
    poles = np.r_[0.0, 1 / (rc - eta)][::-1]
    generator = target[:, 0] / (ac - eta)
    assert np.min(target[:, 0] ** 2) >= gap**2 / span**2
    for n in range(2, k + 1):
        x, y, g = row_nodes[-n:], poles[:n - 1], generator[-n:]
        t = np.array([
            np.prod(node - y) / np.prod(node - np.delete(x, i))
            for i, node in enumerate(x)
        ])
        assert np.all(t > 0)
        assert np.isclose(np.sum(t), 1)
        q, _ = np.linalg.qr(target[-n:, ::-1][:, :n])
        last_generator = abs(q[:, -1] @ g)
        assert np.isclose(last_generator, 1 / np.linalg.norm(t / g))
        assert last_generator >= gap / (2 * span**2)


@pytest.mark.parametrize("scale,shift", [(0.03125, 123.5), (1024., -999.)])
def test_affine_content_changes_preserve_the_matrix(scale, shift):
    rng = np.random.default_rng(273)
    merged = np.r_[0., np.cumsum(rng.integers(1, 8, size=32))][::-1]
    ac, rc = merged[::2], merged[1::2]
    base = generator_qr(ac, rc)
    changed = generator_qr(scale * ac + shift, scale * rc + shift)
    assert [(g.i, g.j) for g in base.gates] == [(g.i, g.j) for g in changed.gates]
    np.testing.assert_allclose(changed.matrix(), base.matrix(), atol=2e-12, rtol=0)
