#!/usr/bin/env python3
"""Compute Weyl chamber coefficients directly from Fourier matrices.

This script mirrors ``generate_data.py``'s CLI but skips compilation.  For
Each Young diagram matching the requested configuration it builds the Fourier
matrix via ``A_matrix``, verifies the unitary dimension, extracts the
TwoQubitWeylDecomposition (a, b, c) invariants, and writes one row per diagram
to a CSV file.
"""
from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable
from pathlib import Path
from typing import Sequence

import numpy as np
from qiskit.synthesis import TwoQubitWeylDecomposition

from compute_matrix import A_matrix
from helper import find_yds_with_fixed_addable_cells


def _diagram_label(diagram) -> str:
    candidate_attrs: Sequence[str] = ("partition", "parts", "rows", "row_lengths")
    for attr in candidate_attrs:
        if hasattr(diagram, attr):
            value = getattr(diagram, attr)
            value = value() if callable(value) else value
            if isinstance(value, Iterable):
                return str(tuple(value))
    if hasattr(diagram, "to_partition"):
        parts = diagram.to_partition()
        if isinstance(parts, Iterable):
            return str(tuple(parts))
    return str(diagram)


def _diagram_size(diagram) -> int:
    size_attr = getattr(diagram, "size", None)
    if size_attr is not None:
        return size_attr() if callable(size_attr) else int(size_attr)
    parts_attr = getattr(diagram, "partition", None)
    if parts_attr is not None:
        if callable(parts_attr):
            parts_attr = parts_attr()
        if isinstance(parts_attr, Iterable):
            return sum(int(p) for p in parts_attr)
    label = _diagram_label(diagram).strip()
    if label.startswith("(") and label.endswith(")"):
        inner = label[1:-1].strip()
        if inner:
            try:
                return sum(int(part.strip()) for part in inner.split(","))
            except ValueError:
                pass
    return 0


def _validate_matrix_shape(matrix: np.ndarray, num_qubits: int) -> None:
    rows, cols = matrix.shape
    expected = 1 << num_qubits
    if rows != cols:
        raise ValueError(
            f"Fourier matrix must be square; received {rows}x{cols} for {num_qubits} qubits."
        )
    if rows != expected:
        raise ValueError(
            f"Expected matrix dimension {expected} for {num_qubits} qubits, received {rows}."
        )


def _weyl_coefficients(matrix: np.ndarray) -> tuple[float, float, float]:
    if matrix.shape != (4, 4):
        raise ValueError("Weyl coefficients are only defined for two-qubit (4x4) unitaries.")
    matrix = np.asarray(matrix, dtype=np.complex128)
    decomposition = TwoQubitWeylDecomposition(matrix)
    return float(decomposition.a), float(decomposition.b), float(decomposition.c)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("num_qubits", type=int, help="Target number of logical qubits (must be 2).")
    parser.add_argument(
        "max_diagram_size",
        type=int,
        help="Upper bound on the Young diagram size to enumerate.",
    )
    parser.add_argument(
        "--max-diagrams",
        type=int,
        default=None,
        help="Optional cap on the number of diagrams to process.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory for the resulting CSV file (default: ./data).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_qubits != 2:
        raise SystemExit("This script currently supports only 2-qubit Fourier matrices (num_qubits=2).")

    addable_cells = 1 << args.num_qubits
    diagrams = find_yds_with_fixed_addable_cells(addable_cells, args.max_diagram_size)
    if args.max_diagrams is not None:
        diagrams = diagrams[: args.max_diagrams]
    if not diagrams:
        raise SystemExit("No diagrams matched the requested configuration.")

    rows: list[dict[str, str | float | int]] = []
    for entry_idx, diagram in enumerate(diagrams):
        matrix = A_matrix(diagram)
        _validate_matrix_shape(matrix, args.num_qubits)
        weyl_a, weyl_b, weyl_c = _weyl_coefficients(matrix)
        rows.append(
            {
                "entry": entry_idx,
                "diagram": _diagram_label(diagram),
                "size": diagram.size if hasattr(diagram, "size") else len(diagram.partition),
                "weyl_a": weyl_a,
                "weyl_b": weyl_b,
                "weyl_c": weyl_c,
            }
        )

    destination = args.output_dir / f"weyl_{addable_cells}_addable_size_{args.max_diagram_size}.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "entry", "diagram", "size", "weyl_a", "weyl_b", "weyl_c"])
        writer.writeheader()
        for idx, row in enumerate(rows):
            writer.writerow({"index": idx, **row})
    print(f"Wrote {len(rows)} Weyl rows to {destination}.")


if __name__ == "__main__":
    main()
