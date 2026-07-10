import numpy as np
from qiskit.quantum_info import Operator
from yungdiagram import YoungDiagram

from fourier.amatrix import a_matrix
from fourier.circuits import givens_circuit, transpiled_counts, wei_di_circuit
from fourier.decompositions import givens_factor, wei_di_fit


def test_wei_di_circuit_matches_matrix():
    A = a_matrix(YoungDiagram([3, 2, 1]))
    wd = wei_di_fit(A, restarts=10, seed=1)
    U = Operator(wei_di_circuit(wd)).data
    assert np.allclose(U.real, wd.matrix(), atol=1e-10)
    assert np.allclose(U.imag, 0, atol=1e-10)
    assert np.allclose(U.real, A, atol=1e-6)


def test_wei_di_gate_counts():
    A = a_matrix(YoungDiagram([3, 2, 1]))
    wd = wei_di_fit(A, restarts=10, seed=1)
    counts = dict(wei_di_circuit(wd).count_ops())
    assert counts == {"ry": 6, "cx": 3}


def test_givens_circuit_one_hot_action():
    """The one-hot-encoded circuit reproduces A on the one-hot subspace."""
    yd = YoungDiagram([3, 2, 1])
    A = a_matrix(yd)
    k = A.shape[0]
    gates, signs = givens_factor(A)
    qc = givens_circuit(k, gates, signs)
    U = Operator(qc).data

    # one-hot basis states: qubit i set ⇒ computational index 2^i
    idx = [1 << i for i in range(k)]
    sub = U[np.ix_(idx, idx)].real
    assert np.allclose(sub, A, atol=1e-10)


def test_transpiled_counts_runs():
    yd = YoungDiagram([2, 1])
    A = a_matrix(yd)
    gates, signs = givens_factor(A)
    qc = givens_circuit(A.shape[0], gates, signs)
    counts = transpiled_counts(qc)
    assert counts.get("cx", 0) >= 1
