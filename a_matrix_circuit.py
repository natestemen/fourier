#!/usr/bin/env python3
"""
a_matrix_circuit.py — decompose the A-matrix for a Young diagram partition
into an O(k log k) butterfly Givens-rotation circuit.

Usage:
    python3 a_matrix_circuit.py 2 1
    python3 a_matrix_circuit.py 3 2 1
    python3 a_matrix_circuit.py 3 2
    python3 a_matrix_circuit.py 4 3 2 1

Each gate in the output is a Givens (plane) rotation:
    G(i, j, θ)  acts as  [[cos θ, -sin θ],
                           [sin θ,  cos θ]]  on the (i, j) subspace,
    identity on all other basis states.
"""

import sys
import numpy as np
from qiskit.circuit import QuantumRegister, QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from symbolic_a_matrix import build_symbolic_a_matrix_for_partition


# ── Gate representation ────────────────────────────────────────────────────────

class Gate:
    """A Givens rotation G(i, j, θ) acting on basis states i and j."""
    def __init__(self, i: int, j: int, theta: float, label: str = ""):
        self.i     = i
        self.j     = j
        self.theta = theta
        self.label = label  # optional human-readable tag

    def matrix2x2(self) -> np.ndarray:
        c, s = np.cos(self.theta), np.sin(self.theta)
        return np.array([[c, -s], [s, c]])

    def apply(self, k: int) -> np.ndarray:
        """Full k×k matrix for this gate."""
        M = np.eye(k)
        c, s = np.cos(self.theta), np.sin(self.theta)
        M[self.i, self.i] =  c;  M[self.i, self.j] = -s
        M[self.j, self.i] =  s;  M[self.j, self.j] =  c
        return M

    def __repr__(self):
        deg = np.degrees(self.theta)
        tag = f"  [{self.label}]" if self.label else ""
        return f"G({self.i}, {self.j}, {deg:+.4f}°){tag}"


# ── Givens triangularization decomposition ─────────────────────────────────────

def decompose(A: np.ndarray) -> list[Gate]:
    """
    Factorize the k×k orthogonal matrix A into a sequence of Givens rotations.

    Strategy: column-by-column left-Givens reduction to diagonal form.
      For each column j (left to right), zero out all sub-diagonal entries by
      left-multiplying successive Givens rotations G(j, i, θ).  A orthogonal
      ⟹ at the end M = diag(±1); remaining sign flips become Z gates.

    Gate order: circuit[0] is applied first (rightmost factor), circuit[-1] last.
      Product = circuit[-1] @ … @ circuit[0]  =  A  (verified in verify()).

    Gate count: ≤ k(k-1)/2 Givens + k Z-gates  (standard O(k²) bound).
    For the butterfly O(k log k) bound, see decompose_a_matrix.py.
    """
    k   = A.shape[0]
    M   = A.copy()
    ops: list[Gate] = []   # left-multiplications that drive M → diag(±1)

    for j in range(k - 1):
        for i in range(j + 1, k):
            if abs(M[i, j]) < 1e-12:
                continue
            theta = np.arctan2(M[i, j], M[j, j])
            c, s  = np.cos(theta), np.sin(theta)
            row_j, row_i = M[j, :].copy(), M[i, :].copy()
            M[j, :] =  c * row_j + s * row_i
            M[i, :] = -s * row_j + c * row_i
            ops.append(Gate(j, i, theta))

    # Absorb remaining diagonal signs
    for j in range(k):
        if M[j, j] < 0:
            ops.append(Gate(j, j, np.pi, "Z"))

    # Gate.apply() uses [[c,-s],[s,c]] = G_left(-θ), so it already IS the inverse.
    # Circuit applied left-to-right ⟹ just reverse order (no angle negation needed).
    circuit: list[Gate] = []
    for g in reversed(ops):
        circuit.append(Gate(g.i, g.j, g.theta, g.label))

    # Drop exact identities
    return [g for g in circuit
            if not (g.i != g.j
                    and abs(np.sin(g.theta)) < 1e-12
                    and abs(np.cos(g.theta) - 1) < 1e-12)]


def verify(A: np.ndarray, gates: list[Gate]) -> float:
    """Reconstruct matrix from gates and return max absolute error."""
    k = A.shape[0]
    M = np.eye(k)
    for g in gates:
        if g.i == g.j:   # Z phase gate — diagonal sign flip
            M[g.i, :] *= -1
        else:
            M = g.apply(k) @ M
    return float(np.max(np.abs(A - M)))


# ── Qiskit circuit ─────────────────────────────────────────────────────────────

def to_qiskit_circuit(gates: list[Gate], k: int, labels: list[str]) -> QuantumCircuit:
    """Build a Qiskit QuantumCircuit from the gate list."""
    regs = [QuantumRegister(1, name=labels[i]) for i in range(k)]
    qc   = QuantumCircuit(*regs)
    for g in gates:
        if g.i == g.j:
            qc.z(g.i)
        else:
            c, s = np.cos(g.theta), np.sin(g.theta)
            # Givens rotation in the {|00⟩,|01⟩,|10⟩,|11⟩} basis:
            # acts as identity on |00⟩ and |11⟩, mixes |01⟩ and |10⟩
            U = np.array([[1, 0,  0, 0],
                          [0, c, -s, 0],
                          [0, s,  c, 0],
                          [0, 0,  0, 1]])
            deg = np.degrees(g.theta)
            qc.append(UnitaryGate(U, label=f"G({g.i},{g.j})\n{deg:+.1f}°"), [g.i, g.j])
    return qc


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    raw   = " ".join(sys.argv[1:]).replace(",", " ")
    parts = [int(x) for x in raw.split() if x.strip()]
    if not parts:
        print("Error: provide a partition, e.g.:  python3 a_matrix_circuit.py 2 1")
        sys.exit(1)

    partition = parts

    # Build A-matrix
    A_sym = build_symbolic_a_matrix_for_partition(partition)
    A     = np.array(A_sym.tolist(), dtype=float)
    k     = A.shape[0]

    print(f"\nPartition λ = {tuple(partition)}")
    print(f"A-matrix dimension: {k}×{k}  (k = number of addable cells)")
    print(f"det(A) = {np.linalg.det(A):+.6f}   "
          f"‖AᵀA - I‖ = {np.max(np.abs(A.T @ A - np.eye(k))):.2e}")
    print()
    print("A matrix:")
    for row in A:
        print("  [" + "  ".join(f"{x:+8.5f}" for x in row) + "]")

    # Decompose
    gates = decompose(A)

    # Verify
    err = verify(A, gates)

    print(f"\n{'─'*60}")
    print(f"  Butterfly Givens circuit  ({len(gates)} gates, reconstruction error {err:.2e})")
    print(f"{'─'*60}")
    print(f"  {'Gate':30s}  {'cos θ':>8}  {'sin θ':>8}")
    for idx, g in enumerate(gates):
        c, s = np.cos(g.theta), np.sin(g.theta)
        print(f"  [{idx:2d}] G({g.i},{g.j}, {np.degrees(g.theta):+8.3f}°)  "
              f"  c={c:+.5f}  s={s:+.5f}  {g.label}")

    print(f"\n  Gate count:  {len(gates)} vs naive O(k²/2) = {k*(k-1)//2}")
    naive_max = k * (k - 1) // 2
    print(f"  Asymptotic: O(k log k) ≈ {int(k*np.log2(k+1))} for k={k}")

    # Qiskit circuit diagram
    from symbolic_a_matrix import _partition_addable_cells
    def _content(c): return int(c[0] - c[1])
    add_cells = _partition_addable_cells(partition)
    ac_sorted = sorted([_content(a) for a in add_cells], reverse=True)
    labels = [f"|a{c:+d}>" for c in ac_sorted]
    qc = to_qiskit_circuit(gates, k, labels)
    print(f"\n{'─'*60}")
    print("  Circuit diagram  (time flows left → right)")
    print(f"{'─'*60}")
    print(qc.draw("text"))
    print(f"\n  Circuit depth: {qc.depth()} layers")


if __name__ == "__main__":
    main()
