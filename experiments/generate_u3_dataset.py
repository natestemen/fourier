#!/usr/bin/env python3
"""Generate the per-gate U3-parameter dataset for BQSKit-compiled A-matrices.

Question: which native U3 parameters does BQSKit's {U3, CNOT} synthesis pick
for each A-matrix, and how do they vary with the diagram's Weyl coordinates?

Supports report.md, Finding 1: the weyl_a column is π/4 for every 4-addable
diagram, while (weyl_b, weyl_c) vary across the family — this CSV is the raw
data behind the U3-parameter scatter/family plots.

Expected result: one CSV per size, columns identical to
data/4_addable_size_23_u3_cnot.csv, with weyl_a ≈ 0.785398 in every row of a
--num-qubits 2 run.

This merges the old generate_data.py with batch_generate.py's size sweep:
--sweep MIN:MAX compiles each max-size in the range separately, writing
data/<k>_addable_size_<size>_u3_cnot.csv and skipping already-written files
unless --force is given.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

from bqskit import Circuit, compile as bqskit_compile
from bqskit.compiler import GateSet, MachineModel
from bqskit.ir.gates import CNOTGate, U3Gate

from fourier import a_matrix, diagrams_with_addable_cells, weyl_coordinates

GATE_SET_LABEL = "u3_cnot"

FIELDNAMES = [
    "index", "entry", "diagram", "layer", "qubit",
    "theta", "phi", "lambda",
    "cos_phi", "sin_phi", "cos_lambda", "sin_lambda", "cos_theta", "sin_theta",
    "weyl_a", "weyl_b", "weyl_c",
]


def u3_rows(
    entry_idx: int,
    diagram_label: str,
    circuit: Circuit,
    weyl: tuple[float, float, float],
) -> list[dict]:
    """One CSV row per single-qubit (U3) gate of a compiled circuit."""
    rows = []
    for layer, op in circuit.operations_with_cycles():
        if op.num_qudits != 1:
            continue
        theta, phi, lam = (float(p) for p in op.params)
        rows.append(
            {
                "entry": entry_idx,
                "diagram": diagram_label,
                "layer": int(layer),
                "qubit": int(op.location[0]),
                "theta": theta,
                "phi": phi,
                "lambda": lam,
                "cos_phi": math.cos(phi),
                "sin_phi": math.sin(phi),
                "cos_lambda": math.cos(lam),
                "sin_lambda": math.sin(lam),
                "cos_theta": math.cos(theta),
                "sin_theta": math.sin(theta),
                "weyl_a": weyl[0],
                "weyl_b": weyl[1],
                "weyl_c": weyl[2],
            }
        )
    return rows


def generate_for_size(args, max_size: int) -> None:
    addable = 1 << args.num_qubits
    destination = (
        args.output_dir / f"{addable}_addable_size_{max_size}_{GATE_SET_LABEL}.csv"
    )
    if destination.exists() and not args.force:
        print(f"[skip] {destination} already exists.")
        return

    diagrams = list(diagrams_with_addable_cells(addable, max_size))
    if args.max_diagrams is not None:
        diagrams = diagrams[: args.max_diagrams]
    if not diagrams:
        print(f"[skip] no diagrams with {addable} addable cells and size ≤ {max_size}.")
        return

    model = MachineModel(args.num_qubits, gate_set=GateSet([CNOTGate(), U3Gate()]))
    rows: list[dict] = []
    for entry_idx, yd in enumerate(diagrams):
        A = a_matrix(yd)
        if args.num_qubits == 2:
            weyl = weyl_coordinates(A)
        else:
            weyl = (math.nan, math.nan, math.nan)

        compiled = bqskit_compile(
            Circuit.from_unitary(A),
            model,
            optimization_level=args.optimization_level,
            synthesis_epsilon=args.synthesis_epsilon,
            seed=args.seed,
        )
        rows.extend(u3_rows(entry_idx, str(yd.partition), compiled, weyl))
        print(f"\r  size {max_size}: compiled {entry_idx + 1}/{len(diagrams)}",
              end="", flush=True)
    print()

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        for idx, row in enumerate(rows):
            writer.writerow({"index": idx, **row})
    print(f"Wrote {len(rows)} rows for {len(diagrams)} diagrams to {destination}.")


def parse_sweep(spec: str) -> range:
    lo, _, hi = spec.partition(":")
    return range(int(lo), int(hi) + 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-qubits", type=int, default=2,
                        help="logical qubits; diagrams have 2^n addable cells (default 2)")
    parser.add_argument("--max-size", type=int, default=8,
                        help="largest diagram size for a single run (default 8)")
    parser.add_argument("--sweep", type=str, default=None, metavar="MIN:MAX",
                        help="generate one CSV per max-size in MIN..MAX (inclusive)")
    parser.add_argument("--max-diagrams", type=int, default=None,
                        help="cap on diagrams per size")
    parser.add_argument("--optimization-level", type=int, default=3,
                        choices=[1, 2, 3, 4],
                        help="BQSKit optimization level (default 3)")
    parser.add_argument("--synthesis-epsilon", type=float, default=1e-9,
                        help="BQSKit synthesis tolerance (default 1e-9)")
    parser.add_argument("--seed", type=int, default=None,
                        help="PRNG seed for reproducible compilations")
    parser.add_argument("--output-dir", type=Path, default=Path("data"),
                        help="destination directory (default data/)")
    parser.add_argument("--force", action="store_true",
                        help="recompute even if the target CSV exists")
    args = parser.parse_args()

    if args.num_qubits != 2:
        sys.stderr.write(
            "Warning: Weyl coordinates are only defined for two-qubit unitaries;"
            " those columns will be NaN for this run.\n"
        )

    sizes = parse_sweep(args.sweep) if args.sweep else [args.max_size]
    for size in sizes:
        generate_for_size(args, size)


if __name__ == "__main__":
    main()
