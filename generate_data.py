#!/usr/bin/env python3
"""Command-line tool for generating Fourier data sets from Young diagrams.

This script generalizes the exploratory code that previously lived in
``play.ipynb``.  Given a number of qubits, it enumerates all Young diagrams
with the matching number of addable cells (``2 ** num_qubits``), compiles the
corresponding Fourier matrices with BQSKit (fixed to the {U3, CNOT} gate set),
records the native U3 parameters for every single-qubit layer, and attaches the
TwoQubitWeylDecomposition (a, b, c) invariants computed directly from each
diagram's Fourier matrix.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Sequence

import numpy as np
from bqskit import Circuit, compile as bqskit_compile
from bqskit.compiler import GateSet, MachineModel
from bqskit.ir.gates import CNOTGate, U3Gate
from qiskit.synthesis import TwoQubitWeylDecomposition

from compute_matrix import A_matrix
from helper import find_yds_with_fixed_addable_cells
from itertools import islice


U3_CNOT_GATESET = GateSet([CNOTGate(), U3Gate()])
GATE_SET_LABEL = "u3_cnot"


class _ProgressBar:
    """Lightweight progress bar for CLI runs."""

    def __init__(self, total: int, description: str, width: int = 40) -> None:
        self.total = max(total, 1)
        self.description = description
        self.width = width
        self.count = 0

    def advance(self, step: int = 1) -> None:
        self.count = min(self.count + step, self.total)
        self._render()

    def _render(self) -> None:
        filled = int(self.width * self.count / self.total)
        bar = "#" * filled + "-" * (self.width - filled)
        percent = min(self.count / self.total, 1.0) * 100
        sys.stdout.write(
            f"\r{self.description}: |{bar}| {percent:5.1f}% ({self.count}/{self.total})"
        )
        sys.stdout.flush()

    def close(self) -> None:
        self.count = self.total
        self._render()
        sys.stdout.write("\n")


def _diagram_label(diagram) -> str:
    """Best-effort conversion of a ``YoungDiagram`` into a tuple string."""
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


def _weyl_coefficients(matrix: np.ndarray) -> tuple[float, float, float]:
    """Return the (a, b, c) Weyl chamber coordinates for a 4x4 unitary."""
    if matrix.shape != (4, 4):
        raise ValueError(
            "TwoQubitWeylDecomposition requires a 4x4 unitary (two qubits)."
        )
    matrix = np.asarray(matrix, dtype=np.complex128)
    decomposition = TwoQubitWeylDecomposition(matrix)
    return float(decomposition.a), float(decomposition.b), float(decomposition.c)


def _angles_from_operation(operation) -> tuple[float, float, float]:
    gate = getattr(operation, "gate", None)
    if not isinstance(gate, U3Gate):
        raise TypeError(f"Expected U3 gate; received {type(gate).__name__}.")
    params = getattr(operation, "params", None)
    if params is None or len(params) != 3:
        raise ValueError("U3 operation must expose exactly three parameters.")
    theta, phi, lam = (float(p) for p in params)
    return theta, phi, lam


def _collect_rows(
    entry_idx: int,
    diagram,
    circuit: Circuit,
    weyl: tuple[float, float, float] | None = None,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    label = _diagram_label(diagram)
    for layer, operation in circuit.operations_with_cycles():
        num_qudits = getattr(operation, "num_qudits", None)
        if num_qudits is None:
            num_qudits = len(getattr(operation, "location", []))
        if num_qudits != 1:
            continue
        theta, phi, lam = _angles_from_operation(operation)
        qubit = int(operation.location[0])
        rows.append(
            {
                "entry": entry_idx,
                "diagram": label,
                "layer": int(layer),
                "qubit": qubit,
                "theta": theta,
                "phi": phi,
                "lambda": lam,
                "cos_phi": math.cos(phi),
                "sin_phi": math.sin(phi),
                "cos_lambda": math.cos(lam),
                "sin_lambda": math.sin(lam),
                "cos_theta": math.cos(theta),
                "sin_theta": math.sin(theta),
                "weyl_a": weyl[0] if weyl else math.nan,
                "weyl_b": weyl[1] if weyl else math.nan,
                "weyl_c": weyl[2] if weyl else math.nan,
            }
        )
    return rows


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


def _write_csv(rows: list[dict[str, float | int | str]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index",
        "entry",
        "diagram",
        "layer",
        "qubit",
        "theta",
        "phi",
        "lambda",
        "cos_phi",
        "sin_phi",
        "cos_lambda",
        "sin_lambda",
        "cos_theta",
        "sin_theta",
        "weyl_a",
        "weyl_b",
        "weyl_c",
    ]
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rows):
            writer.writerow({"index": idx, **row})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("num_qubits", type=int, help="Target number of logical qubits.")
    parser.add_argument(
        "max_diagram_size",
        type=int,
        help="Upper bound on the Young diagram size to enumerate.",
    )
    parser.add_argument(
        "--max-diagrams",
        type=int,
        default=None,
        help="Optional cap on the number of Young diagrams to process.",
    )
    parser.add_argument(
        "--optimization-level",
        type=int,
        default=3,
        choices=[1, 2, 3, 4],
        help="BQSKit optimization level passed through to compile().",
    )
    parser.add_argument(
        "--synthesis-epsilon",
        type=float,
        default=1e-9,
        help="Numerical tolerance forwarded to BQSKit synthesis.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional PRNG seed for reproducible compilation runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Destination directory for the generated CSV file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    addable_cells = 1 << args.num_qubits
    if args.num_qubits != 2:
        sys.stderr.write(
            "Warning: Weyl chamber coefficients are only defined for two-qubit unitaries;"
            " columns will contain NaN values for this run.\n"
        )

    diagrams_iter = find_yds_with_fixed_addable_cells(addable_cells, args.max_diagram_size)
    if args.max_diagrams is not None:
        diagrams_iter = islice(diagrams_iter, args.max_diagrams)
    diagrams = list(diagrams_iter)
    if not diagrams:
        raise SystemExit("No diagrams matched the requested configuration.")

    model = MachineModel(args.num_qubits, gate_set=U3_CNOT_GATESET)
    csv_rows: list[dict[str, float | int | str]] = []

    progress = _ProgressBar(len(diagrams), "Compiling diagrams")
    for entry_idx, diagram in enumerate(diagrams):
        matrix = A_matrix(diagram)
        _validate_matrix_shape(matrix, args.num_qubits)
        if args.num_qubits == 2:
            weyl = _weyl_coefficients(matrix)
        else:
            weyl = (math.nan, math.nan, math.nan)

        circuit = Circuit.from_unitary(matrix)
        compiled = bqskit_compile(
            circuit,
            model,
            optimization_level=args.optimization_level,
            synthesis_epsilon=args.synthesis_epsilon,
            seed=args.seed,
        )
        csv_rows.extend(_collect_rows(entry_idx, diagram, compiled, weyl))
        progress.advance()
    progress.close()

    filename = f"{addable_cells}_addable_size_{args.max_diagram_size}_{GATE_SET_LABEL}.csv"
    destination = args.output_dir / filename
    _write_csv(csv_rows, destination)
    print(f"Wrote {len(csv_rows)} rows for {len(diagrams)} diagrams to {destination}.")


if __name__ == "__main__":
    main()
