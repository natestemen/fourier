"""Quantum-circuit realizations of the decompositions, and compiler benchmarks.

Encoding: a k×k A-matrix is realized on k qubits in the one-hot (unary)
encoding — basis state i of the matrix is qubit i being |1⟩ and the rest |0⟩.
A plane rotation G(i, j, θ) then acts only on the {|01⟩, |10⟩} subspace of
qubits (i, j), which is a two-qubit "Givens gate".  This is the encoding used
by all gate-count benchmarks in this project.
"""

import numpy as np
import numpy.typing as npt
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import QuantumRegister
from qiskit.circuit.library import UnitaryGate

from .decompositions import Givens, WeiDi

__all__ = [
    "givens_gate",
    "givens_circuit",
    "wei_di_circuit",
    "transpiled_counts",
    "bqskit_counts",
]


def givens_gate(theta: float) -> npt.NDArray[np.float64]:
    """The 4×4 unitary of a plane rotation in the one-hot encoding: identity
    on |00⟩ and |11⟩, rotation by θ mixing |01⟩ and |10⟩."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [1, 0, 0, 0],
            [0, c, -s, 0],
            [0, s, c, 0],
            [0, 0, 0, 1],
        ]
    )


def givens_circuit(
    k: int,
    gates: list[Givens],
    signs: npt.NDArray[np.float64] | None = None,
    labels: list[str] | None = None,
) -> QuantumCircuit:
    """A k-qubit one-hot-encoded circuit applying diag(signs) then the plane
    rotations in order.  `labels` names the qubit registers (default q0…)."""
    if labels is not None:
        qc = QuantumCircuit(*[QuantumRegister(1, name=lbl) for lbl in labels])
    else:
        qc = QuantumCircuit(k)

    if signs is not None:
        for i, s in enumerate(signs):
            if s < 0:
                qc.z(i)  # in one-hot encoding, diag entry −1 on state i is Z on qubit i

    for g in gates:
        U = givens_gate(g.theta)
        qc.append(
            UnitaryGate(U, label=f"G({g.i},{g.j})\n{np.degrees(g.theta):+.1f}°"),
            [g.i, g.j],
        )
    return qc


def wei_di_circuit(wd: WeiDi, label: str = "A") -> QuantumCircuit:
    """The 2-qubit Wei–Di circuit (6 Ry + 2 or 3 CNOTs) for a fitted O(4) gate.

    Operator(circuit) equals `wd.matrix()` exactly: the first kron factor of
    the matrix convention is Qiskit's qubit 1 (little-endian), and the
    det-fix CNOT — the rightmost matrix factor — is the first gate applied.
    """
    t1, t2, a, b, t3, t4 = wd.params
    ctrl, tgt = (1, 0) if wd.control == 0 else (0, 1)

    qc = QuantumCircuit(2, name=label)
    if wd.det_fix:
        qc.cx(ctrl, tgt)
    qc.ry(t3, 1)
    qc.ry(t4, 0)
    qc.cx(ctrl, tgt)
    qc.ry(b, 1)
    qc.ry(a, 0)
    qc.cx(ctrl, tgt)
    qc.ry(t1, 1)
    qc.ry(t2, 0)
    return qc


def transpiled_counts(
    qc: QuantumCircuit,
    basis_gates: tuple[str, ...] = ("u3", "cx"),
    optimization_level: int = 3,
) -> dict[str, int]:
    """Gate counts of `qc` after Qiskit transpilation to `basis_gates`."""
    compiled = transpile(
        qc, basis_gates=list(basis_gates), optimization_level=optimization_level
    )
    return dict(compiled.count_ops())


def bqskit_counts(qc: QuantumCircuit, optimization_level: int = 3) -> dict[str, int]:
    """Gate counts after BQSKit compilation (imported lazily — BQSKit is a
    heavy dependency and slow to load)."""
    from bqskit import compile as bqskit_compile
    from bqskit.ext import bqskit_to_qiskit, qiskit_to_bqskit

    compiled = bqskit_compile(qiskit_to_bqskit(qc), optimization_level=optimization_level)
    return dict(bqskit_to_qiskit(compiled).count_ops())
