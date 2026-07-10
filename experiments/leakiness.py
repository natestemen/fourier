#!/usr/bin/env python3
"""Leakiness test (Peterson, Crooks & Smith 2019) for A-matrices.

Question: are A-matrices leaky entanglers — i.e. is their a = π/4 Weyl
coordinate explained by the gate-algebra property shared by CZ and iSWAP?
A gate U is leaky if some nonzero h ∈ su(2) satisfies
U·(h⊗I)·U† ∈ su(2)⊗I + I⊗su(2); the test is a rank computation on an 18×3
real linear system (`fourier.weyl.leakiness_rank`).

Supports report.md Finding 1 (final paragraph): with the default
--max-size 40 every 4-addable A-matrix has full rank 3 in both directions.

Expected result: 0 leaky out of all enumerated diagrams — a = π/4 is not
explained by leakiness.

Migrated from test_leakiness.py.  Behaviour changes: per-diagram results are
also written to data/leakiness.csv, and the Weyl (a, b, c) columns are
computed only for 4-addable matrices (printed as nan otherwise, where the
old script relied on a failing Qiskit call).
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from fourier import (
    a_matrix,
    diagrams_with_addable_cells,
    leakiness_rank,
    leaky_direction,
    weyl_coordinates,
)


def validate() -> None:
    """Check known leaky / non-leaky gates.

    Expected ranks: I⊗I and SWAP → 0, CZ → 2 (leaky direction iσ_z),
    iSWAP → 2, a random SU(4) gate → 3 (non-leaky)."""
    rng = np.random.default_rng(42)
    Q, _ = np.linalg.qr(rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4)))
    Q /= np.linalg.det(Q) ** 0.25

    gates: dict[str, np.ndarray] = {
        "I⊗I  ": np.eye(4, dtype=complex),
        "CZ   ": np.diag([1.0, 1.0, 1.0, -1.0]).astype(complex),
        "SWAP ": np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
        ),
        "iSWAP": np.array(
            [[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=complex
        ),
        "rnd  ": Q,
    }

    print("=== Validation on known gates ===")
    print(f"{'gate':<10} {'left rank':>10} {'min sv (L)':>12}   {'right rank':>10} {'min sv (R)':>12}")
    print("-" * 60)
    for name, U in gates.items():
        r_l, s_l = leakiness_rank(U, "left")
        r_r, s_r = leakiness_rank(U, "right")
        leaky_l = "leaky" if r_l < 3 else "non-leaky"
        leaky_r = "leaky" if r_r < 3 else "non-leaky"
        print(f"  {name}   {r_l} ({leaky_l:<9}) {s_l:>12.2e}   {r_r} ({leaky_r:<9}) {s_r:>12.2e}")
    print()

    v = leaky_direction(gates["CZ   "], "left")
    if v is not None:
        names = ["iσ_x", "iσ_y", "iσ_z"]
        dominant = names[int(np.argmax(np.abs(v)))]
        print(f"  CZ leaky direction: α = {np.round(v, 4)}  →  dominant: {dominant}")
        print("  (Expected: iσ_z, since [CZ, Rz⊗I] = 0 up to local phases)")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--addable", type=int, default=4, help="Number of addable cells (default: 4 = 2-qubit)"
    )
    parser.add_argument("--max-size", type=int, default=40)
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data"), help="Directory for the CSV."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.no_validate:
        validate()

    rows = []
    n_leaky_l = n_leaky_r = n_leaky_either = 0

    hdr = (
        f"{'Diagram':<22} {'a':>7} {'b':>7} {'c':>7}  {'L-rnk':>5} {'R-rnk':>5}"
        f"  {'leaky?':<8}  {'min_sv_L':>10} {'min_sv_R':>10}"
    )
    print(f"=== {args.addable}-addable A-matrices (max_size={args.max_size}) ===")
    print(hdr)
    print("-" * len(hdr))

    for diagram in diagrams_with_addable_cells(args.addable, args.max_size):
        A = a_matrix(diagram)

        r_l, s_l = leakiness_rank(A, "left")
        r_r, s_r = leakiness_rank(A, "right")
        leaky_l = r_l < 3
        leaky_r = r_r < 3

        a = b = c = math.nan
        if args.addable == 4:
            a, b, c = weyl_coordinates(A)

        leaky_str = (
            "L+R" if (leaky_l and leaky_r) else "L  " if leaky_l else "  R" if leaky_r else "no "
        )
        label = str(diagram.partition)
        print(
            f"{label:<22} {a:>7.4f} {b:>7.4f} {c:>7.4f}  "
            f"{r_l:>5} {r_r:>5}  {leaky_str:<8}  {s_l:>10.2e} {s_r:>10.2e}"
        )

        rows.append(
            {
                "diagram": label,
                "size": diagram.size,
                "weyl_a": a,
                "weyl_b": b,
                "weyl_c": c,
                "rank_left": r_l,
                "rank_right": r_r,
                "min_sv_left": s_l,
                "min_sv_right": s_r,
                "leaky": leaky_l or leaky_r,
            }
        )
        n_leaky_l += leaky_l
        n_leaky_r += leaky_r
        n_leaky_either += leaky_l or leaky_r

    print("-" * len(hdr))
    if not rows:
        raise SystemExit("No diagrams matched the requested configuration.")

    n_total = len(rows)
    print(f"\nResult ({n_total} diagrams):")
    print(f"  leaky (left  h⊗I):  {n_leaky_l}/{n_total}")
    print(f"  leaky (right I⊗h):  {n_leaky_r}/{n_total}")
    print(f"  leaky (either):     {n_leaky_either}/{n_total}")
    if n_leaky_either == n_total:
        print("\n  ALL A-matrices are leaky — consistent with the a=π/4 hypothesis.")
    elif n_leaky_either == 0:
        print("\n  NO A-matrices are leaky — a=π/4 does not imply leakiness here.")
    else:
        print("\n  Mixed result — leakiness does not hold universally for this family.")

    csv_path = args.output_dir / "leakiness.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {n_total} rows to {csv_path}.")


if __name__ == "__main__":
    main()
