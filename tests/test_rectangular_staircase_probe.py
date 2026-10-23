"""Independent checks of the experimental Fourier fast-family identity."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from rectangular_staircase_probe import (  # noqa: E402
    cotangent_contents, cotangent_kernel, fourier_circuit_matrix, cotangent_qft_circuit,
    perturbation_bound, rectangular_contents, rectangular_weights,
    rounded_cotangent_diagram, sufficient_rounding_scale,
)
from fourier.amatrix import _a_matrix_from_contents_log


@pytest.mark.parametrize("k", [2, 3, 4, 8, 17, 32, 64])
@pytest.mark.parametrize("rho", [0.13, 0.5, 0.87])
def test_two_fourier_identity(k, rho):
    x, y = cotangent_contents(k, rho)
    reference = _a_matrix_from_contents_log(x, y)
    circuit = fourier_circuit_matrix(k, rho)
    np.testing.assert_allclose(cotangent_kernel(k, rho), reference, atol=1e-12, rtol=0)
    np.testing.assert_allclose(circuit, reference, atol=1e-12, rtol=0)
    np.testing.assert_allclose(circuit.conj().T @ circuit, np.eye(k), atol=1e-12, rtol=0)


@pytest.mark.parametrize("k", [2, 4, 16, 64])
@pytest.mark.parametrize("rho", [0.1, 0.37, 0.9])
def test_rectangular_hahn_weights(k, rho):
    x, y = rectangular_contents(k, rho)
    reference = _a_matrix_from_contents_log(x, y)[:, 0] ** 2
    np.testing.assert_allclose(rectangular_weights(k, rho), reference, atol=2e-13, rtol=0)


@pytest.mark.parametrize("k", [4, 8, 16, 32])
@pytest.mark.parametrize("rho", [0.2, 0.5, 0.73])
def test_rounding_produces_genuine_diagram_and_certified_approximation(k, rho):
    epsilon = 1e-3
    scale = sufficient_rounding_scale(k, rho, epsilon)
    rounded = rounded_cotangent_diagram(k, rho, scale)
    assert np.sum(rounded.addable) == np.sum(rounded.removable)
    blocks = rounded.blocks()
    assert all(width > 0 and multiplicity > 0 for width, multiplicity in blocks)
    assert all(blocks[i][0] > blocks[i + 1][0] for i in range(len(blocks) - 1))
    assert sum(multiplicity for _, multiplicity in blocks) == -rounded.addable[-1]
    target = _a_matrix_from_contents_log(rounded.addable, rounded.removable)
    error = np.linalg.norm(target - cotangent_kernel(k, rho), 2)
    assert error <= rounded.error_bound + 2e-12
    assert rounded.error_bound <= epsilon * (1 + 1e-7)


def test_general_node_perturbation_certificate():
    rng = np.random.default_rng(1773)
    for k in [3, 7, 12]:
        merged = np.r_[0.0, np.cumsum(rng.uniform(1, 2, 2 * k - 2))][::-1]
        changed = merged + rng.uniform(-1e-4, 1e-4, len(merged))
        a = _a_matrix_from_contents_log(merged[::2], merged[1::2])
        b = _a_matrix_from_contents_log(changed[::2], changed[1::2])
        bound = perturbation_bound(k, np.min(merged[:-1] - merged[1:]),
                                   np.max(np.abs(changed - merged)))
        assert np.linalg.norm(a - b, 2) <= bound


@pytest.mark.parametrize("k", [2, 4, 8, 16])
def test_compiled_fixed_parameter_circuit_is_clean(k):
    from qiskit.quantum_info import Operator

    matrix = Operator(cotangent_qft_circuit(k, 0.37)).data[:, :k]
    np.testing.assert_allclose(matrix[:k], cotangent_kernel(k, 0.37), atol=3e-14, rtol=0)
    np.testing.assert_allclose(matrix[k:], 0, atol=3e-14, rtol=0)


@pytest.mark.parametrize("k,bits", [(2, 3), (4, 3), (8, 3)])
def test_compiled_coherent_parameter_circuit_preserves_all_sector_phases(k, bits):
    from qiskit.quantum_info import Operator

    size = k * 2**bits
    matrix = Operator(cotangent_qft_circuit(k, rho_bits=bits)).data[:, :size]
    for parameter in range(1, 2**bits):
        sector = slice(parameter * k, (parameter + 1) * k)
        expected = cotangent_kernel(k, parameter / 2**bits)
        np.testing.assert_allclose(matrix[sector, sector], expected, atol=3e-14, rtol=0)
        leakage = matrix[:, sector].copy()
        leakage[sector] = 0
        np.testing.assert_allclose(leakage, 0, atol=3e-14, rtol=0)
