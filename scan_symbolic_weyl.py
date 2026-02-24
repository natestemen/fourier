#!/usr/bin/env python3
"""Scan valid (w1,h1,w2,h2,w3,h3) configs using symbolic A and scatter Weyl (a,b,c)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from qiskit.synthesis import TwoQubitWeylDecomposition

from symbolic_a_matrix import build_symbolic_a_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--w1-min", type=int, default=3)
    parser.add_argument("--w1-max", type=int, default=8)
    parser.add_argument("--w2-min", type=int, default=2)
    parser.add_argument("--w2-max", type=int, default=7)
    parser.add_argument("--w3-min", type=int, default=1)
    parser.add_argument("--w3-max", type=int, default=6)
    parser.add_argument("--h-min", type=int, default=1)
    parser.add_argument("--h-max", type=int, default=6)
    parser.add_argument(
        "--generic-only",
        action="store_true",
        default=True,
        help="Enforce extra constraints used by the symbolic AC/RC for lambda-r."
        " (default: true)",
    )
    parser.add_argument(
        "--allow-edge",
        action="store_false",
        dest="generic_only",
        help="Allow edge cases (may violate symbolic assumptions).",
    )
    parser.add_argument(
        "--require-unitary",
        action="store_true",
        default=True,
        help="Skip matrices that are not numerically unitary.",
    )
    parser.add_argument(
        "--no-require-unitary",
        action="store_false",
        dest="require_unitary",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on points.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/plots/symbolic_weyl_scan_bc.png"),
        help="Output path for the 2D b vs |c| scatter plot.",
    )
    return parser.parse_args()


def _valid_base(w1: int, w2: int, w3: int, h1: int, h2: int, h3: int) -> bool:
    if min(w1, w2, w3, h1, h2, h3) <= 0:
        return False
    if not (w1 > w2 > w3 >= 1):
        return False
    return True


def _valid_generic(w1: int, w2: int, w3: int, h1: int, h2: int, h3: int) -> bool:
    # Enforce extra constraints so lambda - r_j keeps the same 3-block shape
    # used in symbolic_a_matrix.py.
    if not _valid_base(w1, w2, w3, h1, h2, h3):
        return False
    if not (w1 >= w2 + 2 and w2 >= w3 + 2):
        return False
    if not (h1 >= 2 and h2 >= 2 and h3 >= 2):
        return False
    return True


def _is_unitary(mat: np.ndarray, tol: float = 1e-6) -> bool:
    eye = np.eye(mat.shape[0], dtype=complex)
    return np.allclose(mat.conj().T @ mat, eye, atol=tol)


def _family_points(
    n_max: int,
    step: int,
    A_func,
    require_unitary: bool,
) -> tuple[list[float], list[float]]:
    # Family: w1=n, w2=2, w3=1, h1=h2=h3=1.
    bs: list[float] = []
    cs: list[float] = []
    for n in range(3, n_max + 1, step):
        try:
            A_val = np.array(A_func(n, 2, 1, 1, 1, 1), dtype=complex)
        except ZeroDivisionError:
            continue
        if not np.isfinite(A_val).all():
            continue
        if require_unitary and not _is_unitary(A_val):
            continue
        try:
            decomp = TwoQubitWeylDecomposition(A_val)
        except Exception:
            continue
        bs.append(float(decomp.b))
        cs.append(abs(float(decomp.c)))
    return bs, cs


def _family_limit_point(A_sym: sp.Matrix, symbols: tuple[sp.Symbol, ...]) -> tuple[float, float] | None:
    # Limit as n -> infinity for family: w1=n, w2=2, w3=1, h1=h2=h3=1.
    w1, w2, w3, h1, h2, h3 = symbols
    A_family = A_sym.subs({w2: 2, w3: 1, h1: 1, h2: 1, h3: 1})
    A_lim = A_family.applyfunc(lambda expr: sp.limit(expr, w1, sp.oo))
    try:
        A_val = np.array(A_lim.evalf(), dtype=complex)
    except Exception:
        return None
    if not np.isfinite(A_val).all():
        return None
    try:
        decomp = TwoQubitWeylDecomposition(A_val)
    except Exception:
        return None
    return float(decomp.b), abs(float(decomp.c))


def main() -> None:
    args = parse_args()
    A_sym, symbols = build_symbolic_a_matrix()
    A_func = sp.lambdify(symbols, A_sym, "numpy")

    points: list[tuple[float, float, float, int, tuple[int, int, int, int, int, int]]] = []

    for w1 in range(args.w1_min, args.w1_max + 1):
        for w2 in range(args.w2_min, min(args.w2_max, w1 - 1) + 1):
            for w3 in range(args.w3_min, min(args.w3_max, w2 - 1) + 1):
                for h1 in range(args.h_min, args.h_max + 1):
                    for h2 in range(args.h_min, args.h_max + 1):
                        for h3 in range(args.h_min, args.h_max + 1):
                            if args.generic_only:
                                if not _valid_generic(w1, w2, w3, h1, h2, h3):
                                    continue
                            else:
                                if not _valid_base(w1, w2, w3, h1, h2, h3):
                                    continue

                            try:
                                A_val = np.array(
                                    A_func(w1, w2, w3, h1, h2, h3), dtype=complex
                                )
                            except ZeroDivisionError:
                                continue

                            if not np.isfinite(A_val).all():
                                continue

                            if args.require_unitary and not _is_unitary(A_val):
                                continue

                            try:
                                decomp = TwoQubitWeylDecomposition(A_val)
                            except Exception:
                                continue

                            size = w1 * h1 + w2 * h2 + w3 * h3
                            points.append(
                                (
                                    float(decomp.a),
                                    float(decomp.b),
                                    float(decomp.c),
                                    size,
                                    (w1, w2, w3, h1, h2, h3),
                                )
                            )

                            if args.limit and len(points) >= args.limit:
                                break
                        if args.limit and len(points) >= args.limit:
                            break
                    if args.limit and len(points) >= args.limit:
                        break
                if args.limit and len(points) >= args.limit:
                    break
            if args.limit and len(points) >= args.limit:
                break
        if args.limit and len(points) >= args.limit:
            break

    if not points:
        raise SystemExit("No valid points generated with the requested ranges.")

    bs = [p[1] for p in points]
    cs = [abs(p[2]) for p in points]
    sizes = [p[3] for p in points]
    fam_b, fam_c = _family_points(10000, 10, A_func, args.require_unitary)
    fam_limit = _family_limit_point(A_sym, symbols)

    fig, ax = plt.subplots()
    sc = ax.scatter(
        bs,
        cs,
        s=14,
        alpha=0.5,
        c=sizes,
        cmap="viridis",
        linewidths=0,
        edgecolors=None,
    )
    if fam_b and fam_c:
        ax.scatter(
            fam_b,
            fam_c,
            s=8,
            alpha=0.9,
            color="red",
            label="family: w1=n,w2=2,w3=1,h1=h2=h3=1 (n=3..10000 step 10)",
        )
    if fam_limit is not None:
        ax.scatter(
            [fam_limit[0]],
            [fam_limit[1]],
            s=80,
            marker="*",
            color="gold",
            edgecolors="black",
            linewidths=0.6,
            label="family limit n→∞",
        )
    ax.set_xlabel("b")
    ax.set_ylabel("|c|")
    ax.set_title("Weyl b vs |c| from symbolic A scan (a = pi/4)")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("diagram size (# cells)")
    if (fam_b and fam_c) or (fam_limit is not None):
        ax.legend()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
