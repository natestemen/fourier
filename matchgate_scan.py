#!/usr/bin/env python3
"""Check which 2-qubit Young-diagram Fourier matrices are matchgates."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from compute_matrix import A_matrix
from helper import find_yds_with_fixed_addable_cells


def _pauli_basis() -> tuple[list[np.ndarray], set[int]]:
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    c1 = np.kron(X, I)
    c2 = np.kron(Y, I)
    c3 = np.kron(Z, X)
    c4 = np.kron(Z, Y)
    C = [c1, c2, c3, c4]

    single = [I, X, Y, Z]
    basis = [np.kron(a, b) for a in single for b in single]
    c_indices: set[int] = set()
    for c in C:
        for i, P in enumerate(basis):
            if np.allclose(P, c):
                c_indices.add(i)
                break
    return basis, c_indices


def is_matchgate(U: np.ndarray, tol: float = 1e-8) -> bool:
    if U.shape != (4, 4):
        raise ValueError("Matchgate test expects a 4x4 unitary.")
    basis, c_indices = _pauli_basis()
    for c_index in c_indices:
        c = basis[c_index]
        M = U @ c @ U.conj().T
        for i, P in enumerate(basis):
            coeff = 0.25 * np.trace(P.conj().T @ M)
            if abs(coeff) > tol and i not in c_indices:
                return False
    return True


def _diagram_label(diagram) -> str:
    for attr in ("partition", "parts", "rows", "row_lengths"):
        if hasattr(diagram, attr):
            value = getattr(diagram, attr)
            value = value() if callable(value) else value
            try:
                return str(tuple(value))
            except TypeError:
                pass
    return str(diagram)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-diagram-size",
        type=int,
        required=True,
        help="Upper bound on Young diagram sizes to enumerate.",
    )
    parser.add_argument(
        "--max-diagrams",
        type=int,
        default=100,
        help="Maximum number of diagrams to test (default: 100).",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-8,
        help="Coefficient tolerance for the matchgate test (default: 1e-8).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/matchgate_scan.csv"),
        help="CSV output path (default: data/matchgate_scan.csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    num_qubits = 2
    addable_cells = 1 << num_qubits

    diagrams = find_yds_with_fixed_addable_cells(addable_cells, args.max_diagram_size)
    if not diagrams:
        raise SystemExit("No diagrams found for the requested size bounds.")
    diagrams = diagrams[: args.max_diagrams]

    rows: list[dict[str, str | int | float]] = []
    for idx, diagram in enumerate(diagrams):
        matrix = A_matrix(diagram)
        result = is_matchgate(matrix, tol=args.tol)
        rows.append(
            {
                "index": idx,
                "diagram": _diagram_label(diagram),
                "size": getattr(diagram, "size", None) if hasattr(diagram, "size") else "",
                "matchgate": result,
            }
        )
        print(f"{idx:03d} {rows[-1]['diagram']} -> {result}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "diagram", "size", "matchgate"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} results to {args.output}")


if __name__ == "__main__":
    main()
