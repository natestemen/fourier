"""Independent checks of the staircase circuit and its clean ancillas."""

from math import comb

import numpy as np
import pytest
from scipy.linalg import expm, hadamard

from fourier.amatrix import staircase_a_matrix
from fourier.staircase_lift import dicke_rotation, staircase_lift, symmetric_hadamard


@pytest.mark.parametrize("m", [0, 1, 2, 3, 4, 7, 8])
def test_krawtchouk_is_actual_tensor_hadamard_on_dicke_states(m):
    encoder = np.zeros((2**m, m + 1))
    for bitstring in range(2**m):
        weight = bitstring.bit_count()
        encoder[bitstring, weight] = 1 / np.sqrt(comb(m, weight))
    full_hadamard = hadamard(2**m) / np.sqrt(2**m)
    reference = encoder.T @ full_hadamard @ encoder
    assert np.allclose(symmetric_hadamard(m), reference, atol=2e-15, rtol=0)


@pytest.mark.parametrize("m", [1, 2, 5, 16, 32])
def test_su2_rotation_phase_convention(m):
    """Pin the phase needed when controlling an SU(2) Krawtchouk primitive."""
    w = np.arange(m)
    coupling = 0.5 * np.sqrt((w + 1) * (m - w))
    jy = np.diag(-1j * coupling, 1) + np.diag(1j * coupling, -1)
    rotation = expm(-0.5j * np.pi * jy)
    reference = rotation @ np.diag((-1.0) ** np.arange(m + 1))
    assert np.allclose(symmetric_hadamard(m), reference, atol=6e-15, rtol=0)


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 8, 16, 32, 64])
def test_lift_matches_independent_content_formula(k):
    lift = staircase_lift(k)
    result = lift.matrix()
    assert np.linalg.norm(result - staircase_a_matrix(k), ord=2) < 2e-12
    assert np.linalg.norm(result.T @ result - np.eye(k), ord=2) < 3e-14
    assert np.allclose(lift.lifted_matrix()[1::2], 0, atol=2e-15, rtol=0)


@pytest.mark.parametrize("k", [2, 3, 4, 5, 8, 17])
def test_constant_branch_flag_erases_for_the_full_input_space(k):
    """Simulate the branch flag explicitly, including both parities of N.

    For odd N, input |constant> and one removable input initially have the
    same weight label.  The retained flag makes the input map injective.
    Erasure is valid only after the controlled K and central sign fold.
    Testing the whole isometry verifies every input superposition at once.
    """
    lift = staircase_lift(k)
    n, m = k - 1, lift.n_qubits
    state = np.zeros((2, m + 1, k))
    state[0, n, 0] = 1
    for j in range(n):
        state[1, 2 * j + 1, j + 1] = (-1) ** j

    state[1] = lift.hadamard @ state[1]
    state *= lift.fold_signs[None, :, None]
    erased = np.zeros_like(state)
    for flag in range(2):
        for weight in range(m + 1):
            erased[flag ^ int(weight != n), weight] = state[flag, weight]
    assert np.allclose(erased[1], 0, atol=2e-15, rtol=0)
    assert np.allclose(erased[0], lift.prepared_columns(), atol=2e-15, rtol=0)

    output = lift.hadamard @ erased[0]
    assert np.allclose(output[1::2], 0, atol=2e-15, rtol=0)
    compressed = (-1.0) ** np.arange(k)[:, None] * output[::2]
    assert np.allclose(compressed, staircase_a_matrix(k), atol=2e-13, rtol=0)


@pytest.mark.parametrize("m", [1, 2, 3, 7, 8])
def test_sequential_dicke_encoder_has_uniform_amplitudes_and_zero_counter(m):
    for weight in range(m + 1):
        # (remaining-ones counter, emitted prefix) -> amplitude.
        branches = {(weight, 0): 1.0}
        for remaining_sites in range(m, 0, -1):
            updated = {}
            for (remaining_ones, prefix), amplitude in branches.items():
                cosine, sine = dicke_rotation(remaining_ones, remaining_sites)
                if cosine:
                    updated[remaining_ones, 2 * prefix] = amplitude * cosine
                if sine:
                    updated[remaining_ones - 1, 2 * prefix + 1] = amplitude * sine
            branches = updated
        assert len(branches) == comb(m, weight)
        for (remaining_ones, bitstring), amplitude in branches.items():
            assert remaining_ones == 0
            assert bitstring.bit_count() == weight
            assert np.isclose(amplitude, 1 / np.sqrt(comb(m, weight)), atol=2e-15)


@pytest.mark.parametrize("k", [0, -1, 2.5])
def test_invalid_dimension_rejected(k):
    with pytest.raises(ValueError, match="positive integer"):
        staircase_lift(k)


@pytest.mark.parametrize("args", [(1, 0), (-1, 3), (4, 3), (1.5, 3)])
def test_invalid_dicke_step_rejected(args):
    with pytest.raises(ValueError, match="expected integers"):
        dicke_rotation(*args)
