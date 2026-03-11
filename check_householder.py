#!/usr/bin/env python3
"""Check whether A matrices are Householder reflections."""
from __future__ import annotations

import argparse

import numpy as np

from compute_matrix import A_matrix
from helper import find_yds_with_fixed_addable_cells


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--addable",
        type=int,
        default=4,
        help="Number of addable cells for diagrams to check.",
    )
    parser.add_argument("--max-size", type=int, default=30, help="Max diagram size to search.")
    parser.add_argument("--index", type=int, default=None, help="Optional index into diagram list.")
    parser.add_argument("--tol", type=float, default=1e-8, help="Numerical tolerance.")
    parser.add_argument("--report-all", action="store_true", help="Report all diagrams.")
    return parser.parse_args()


def _is_householder(A: np.ndarray, tol: float) -> tuple[bool, dict[str, float | int]]:
    n, m = A.shape
    if n != m:
        return False, {"reason": "non-square"}

    eye = np.eye(n)
    orth_err = float(np.max(np.abs(A.T @ A - eye)))
    sym_err = float(np.max(np.abs(A - A.T)))

    if orth_err > tol or sym_err > tol:
        return False, {
            "orth_err": orth_err,
            "sym_err": sym_err,
            "neg_eigs": 0,
            "pos_eigs": 0,
        }

    eigs = np.linalg.eigvalsh((A + A.T) / 2.0)
    neg = int(np.sum(np.abs(eigs + 1.0) <= tol))
    pos = int(np.sum(np.abs(eigs - 1.0) <= tol))
    householder = neg == 1 and pos == n - 1

    return householder, {
        "orth_err": orth_err,
        "sym_err": sym_err,
        "neg_eigs": neg,
        "pos_eigs": pos,
    }


def _format_partition(d) -> str:
    return str(getattr(d, "partition", d))


def main() -> None:
    args = parse_args()

    diagrams = list(find_yds_with_fixed_addable_cells(args.addable, args.max_size))
    if not diagrams:
        raise SystemExit(f"No diagrams found with {args.addable} addable cells.")

    if args.index is not None:
        if args.index < 0 or args.index >= len(diagrams):
            raise SystemExit(f"--index out of range (0..{len(diagrams)-1}).")
        diagrams = [diagrams[args.index]]

    matches: list[str] = []

    for d in diagrams:
        A = np.array(A_matrix(d), dtype=float)
        ok, info = _is_householder(A, args.tol)
        if ok:
            matches.append(_format_partition(d))
        if args.report_all:
            print(
                f"diagram={_format_partition(d)} householder={ok} "
                f"orth_err={info.get('orth_err', 0):.3e} "
                f"sym_err={info.get('sym_err', 0):.3e} "
                f"neg_eigs={info.get('neg_eigs', 0)} "
                f"pos_eigs={info.get('pos_eigs', 0)}"
            )

    print(f"\nHouseholder matches: {len(matches)}")
    for part in matches:
        print(f"  {part}")


if __name__ == "__main__":
    main()
