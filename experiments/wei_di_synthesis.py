#!/usr/bin/env python3
"""Wei–Di optimal 2-qubit synthesis of every 4-addable A-matrix.

Question: does the provably optimal Wei–Di O(4) circuit template
(arXiv:1203.0722) reproduce every 4-addable A-matrix exactly?

Supports report.md, Finding 5: all 4-addable A-matrices have det = −1 and
synthesize as 3 CNOT + 6 Ry — the optimal count — to machine precision.

Expected result: every diagram has det = −1, fits with residual < 1e-12,
and costs 3 CNOTs; pass --qiskit to print each circuit.

The report scanned --max-size 15 (311 diagrams); the default here is 10
(33 diagrams at ≈1.5 s per fit) so a bare run finishes in about a minute.

Outputs: data/wei_di_synthesis.csv.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import ortho_group

from fourier import a_matrix, diagrams_with_addable_cells, wei_di_fit
from fourier.circuits import wei_di_circuit

PARAM_NAMES = ("theta1", "theta2", "a", "b", "theta3", "theta4")


def validate(restarts: int) -> None:
    """Sanity-check the fitter on random SO(4) and O(4) matrices."""
    print("=== Validation ===")
    rng = np.random.default_rng(7)

    X_pos = ortho_group.rvs(4, random_state=rng)
    if np.linalg.det(X_pos) < 0:
        X_pos[0] *= -1
    X_neg = ortho_group.rvs(4, random_state=rng)
    if np.linalg.det(X_neg) > 0:
        X_neg[0] *= -1

    for name, X in [("SO(4) det=+1", X_pos), ("O(4) det=-1", X_neg)]:
        wd = wei_di_fit(X, restarts=restarts)
        recon_err = float(np.max(np.abs(X - wd.matrix())))
        print(f"  {name}: residual={wd.residual:.2e}  n_CNOT={wd.n_cnots}  "
              f"recon_err={recon_err:.2e}  "
              f"{'OK' if wd.residual < 1e-8 else 'FAIL'}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-size", type=int, default=10,
                        help="largest diagram size (default 10; report used 15)")
    parser.add_argument("--restarts", type=int, default=40,
                        help="local-refinement restarts per fit (default 40)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tol", type=float, default=1e-6,
                        help="residual tolerance for 'success' (default 1e-6)")
    parser.add_argument("--no-validate", action="store_true",
                        help="skip the random-O(4) sanity check")
    parser.add_argument("--qiskit", action="store_true",
                        help="print the Qiskit circuit for each diagram")
    args = parser.parse_args()

    if not args.no_validate:
        validate(restarts=min(args.restarts, 20))

    hdr = (f"{'Diagram':<22} {'det':>4} {'residual':>10}  "
           + " ".join(f"{p:>7}" for p in ("θ₁", "θ₂", "a", "b", "θ₃", "θ₄"))
           + f"  {'CNOT':>4} {'ok?':>4}")
    print(f"=== Wei-Di synthesis of 4-addable A-matrices "
          f"(max_size={args.max_size}) ===")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    n_total = n_ok = n_cnot3 = 0
    for yd in diagrams_with_addable_cells(4, args.max_size):
        A = a_matrix(yd)
        det = float(np.linalg.det(A))
        wd = wei_di_fit(A, restarts=args.restarts, seed=args.seed)
        ok = wd.residual < args.tol
        label = str(yd.partition)

        print(f"{label:<22} {det:>+.0f} {wd.residual:>10.2e}  "
              + " ".join(f"{p:>7.4f}" for p in wd.params)
              + f"  {wd.n_cnots:>4} {'✓' if ok else '✗'}")

        if args.qiskit and ok:
            print(wei_di_circuit(wd, label=label).draw(output="text", fold=120))
            print()

        rows.append({
            "diagram": label,
            "det": det,
            "residual": wd.residual,
            **dict(zip(PARAM_NAMES, wd.params)),
            "n_cnots": wd.n_cnots,
        })
        n_total += 1
        n_ok += ok
        n_cnot3 += wd.n_cnots == 3

    csv_path = Path("data/wei_di_synthesis.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["diagram", "det", "residual", *PARAM_NAMES, "n_cnots"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print("-" * len(hdr))
    print(f"\nResult ({n_total} diagrams):")
    print(f"  det=+1 (2 CNOT): {n_total - n_cnot3}")
    print(f"  det=-1 (3 CNOT): {n_cnot3}")
    print(f"  Synthesis OK:    {n_ok}/{n_total}")
    if n_cnot3 == n_total:
        print("\n  All A-matrices have det=-1  →  optimal circuit = 3 CNOT + 6 Ry")
    if n_ok == n_total:
        print("  All decompositions succeeded to within tolerance.")
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
