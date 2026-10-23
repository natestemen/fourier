"""Matrix-level checks; these are not resource estimates of compiled gates."""

from pathlib import Path
import sys

import numpy as np
import pytest
import mpmath as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from hilbert_block_encoding import (  # noqa: E402
    fractional_shift_matrix,
    integer_cotangent_probe,
    log_gamma_stirling_probe,
    offdiagonal_multipliers,
    rectangular_probe,
    rectangular_scales,
)
from fourier.amatrix import _a_matrix_from_contents_log


@pytest.mark.parametrize("k", [2, 4, 16, 64, 128])
@pytest.mark.parametrize("theta", [0.01, 0.1, 0.3, 0.5, 0.8, 0.99])
def test_rectangular_residues_and_error(k, theta):
    probe = rectangular_probe(k, theta, 128 * k)
    alpha, beta = probe.alpha, probe.beta
    assert np.isclose(np.linalg.norm(alpha), 1, atol=2e-13)
    assert np.max(alpha) <= max(alpha[0], alpha[-1]) * (1 + 1e-12)
    assert np.max(beta) <= np.e * np.sqrt((k - 1) * theta * (1 - theta))
    assert np.allclose(np.sum(beta**2), (k - 1) * k * theta * (1 - theta) / 2,
                       rtol=2e-12)
    assert np.linalg.norm(probe.approximate - probe.exact, 2) <= probe.kernel_error_bound + 2e-11
    assert probe.normalization_bound <= 1 + (3.5 * np.e + 1) * np.sqrt(k)


@pytest.mark.parametrize("delta", [-0.5, -0.3, -0.01, 1e-4, 0.1, 0.5])
def test_fractional_fourier_identity_and_removed_diagonal(delta):
    m = 32
    v = fractional_shift_matrix(m, delta)
    i, j = np.arange(m)[:, None], np.arange(m)[None, :]
    expected = np.sin(np.pi * delta) / (m * np.sin(np.pi * (j - i + delta) / m))
    assert np.max(np.abs(v.imag)) < 1e-13
    assert np.allclose(v.real, expected, atol=3e-13)
    assert np.linalg.norm(v.conj().T @ v - np.eye(m), 2) < 1e-12
    scaled = np.pi / np.sin(np.pi * delta) * v
    scaled -= np.eye(m) * np.pi / (m * np.sin(np.pi * delta / m))
    f = offdiagonal_multipliers(m, delta)
    assert np.linalg.norm(scaled, 2) == pytest.approx(np.max(np.abs(f)), abs=2e-10)
    assert np.max(np.abs(f)) < 7


def test_uniform_multiplier_bound_on_parameter_grid():
    for delta in np.r_[-np.geomspace(1e-6, 0.5, 60), np.geomspace(1e-6, 0.5, 60)]:
        assert np.max(np.abs(offdiagonal_multipliers(256, delta))) < 7


def test_generic_integer_cotangent_error_and_spectrum():
    ac = np.array([11, 8, 4, -1, -8], dtype=float)
    rc = np.array([10, 7, 1, -4], dtype=float)
    m = 1024
    approximate, normalization, bound = integer_cotangent_probe(ac, rc, m)
    exact = _a_matrix_from_contents_log(ac, rc)
    assert np.linalg.norm(approximate - exact, 2) <= bound
    assert normalization > 1
    offsets = np.arange(m)
    kernel = np.zeros(m)
    kernel[1:] = (np.pi / m) / np.tan(np.pi * offsets[1:] / m)
    expected = -1j * np.pi * (1 - 2 * offsets / m)
    expected[0] = 0
    assert np.allclose(np.fft.fft(kernel), expected, atol=1e-12)


@pytest.mark.parametrize("theta", [0, 1, np.nan])
def test_invalid_offsets(theta):
    with pytest.raises(ValueError):
        rectangular_scales(4, theta)


@pytest.mark.parametrize("z", ["1e-20", "0.1", "0.9", "1000000000.1"])
@pytest.mark.parametrize("bits", [20, 50, 100])
def test_fixed_loop_gamma_arithmetic(z, bits):
    approximate, remainder = log_gamma_stirling_probe(z, bits)
    with mp.workprec(4 * bits + 256):
        reference = mp.loggamma(mp.mpf(z))
        assert remainder < mp.mpf(2)**(-bits)
        assert abs(approximate - reference) < mp.mpf(2)**(-bits)
