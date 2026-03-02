#!/usr/bin/env python3
"""Plot 2D Weyl chamber scatter (b vs |c|) for symbolic scan plus four families."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
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
        help="Enforce extra constraints used by the symbolic AC/RC for lambda-r.",
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
    parser.add_argument("--scan-limit", type=int, default=0)
    parser.add_argument("--family-n-min", type=int, default=3)
    parser.add_argument("--family-n-max", type=int, default=10000)
    parser.add_argument("--family-step", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/plots/weyl_all_families.html"),
    )
    parser.add_argument(
        "--plotly-renderer",
        default="browser",
        help="Plotly renderer name (default: browser).",
    )
    return parser.parse_args()


def _is_unitary(mat: np.ndarray, tol: float = 1e-6) -> bool:
    eye = np.eye(mat.shape[0], dtype=complex)
    return np.allclose(mat.conj().T @ mat, eye, atol=tol)


def _valid_base(w1: int, w2: int, w3: int, h1: int, h2: int, h3: int) -> bool:
    if min(w1, w2, w3, h1, h2, h3) <= 0:
        return False
    if not (w1 > w2 > w3 >= 1):
        return False
    return True


def _valid_generic(w1: int, w2: int, w3: int, h1: int, h2: int, h3: int) -> bool:
    if not _valid_base(w1, w2, w3, h1, h2, h3):
        return False
    if not (w1 >= w2 + 2 and w2 >= w3 + 2):
        return False
    if not (h1 >= 2 and h2 >= 2 and h3 >= 2):
        return False
    return True


def _decomp(A_val: np.ndarray) -> tuple[float, float, float] | None:
    try:
        decomp = TwoQubitWeylDecomposition(A_val)
    except Exception:
        return None
    return float(decomp.a), float(decomp.b), float(decomp.c)


def _scan_points(args: argparse.Namespace, A_func) -> np.ndarray:
    points = []
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

                            abc = _decomp(A_val)
                            if abc is None:
                                continue
                            points.append(abc)

                            if args.scan_limit and len(points) >= args.scan_limit:
                                return np.array(points, dtype=float)
    return np.array(points, dtype=float)


def _family_series(A_func, require_unitary: bool, params_fn, n_min: int, n_max: int, step: int):
    xs, ys, zs = [], [], []
    for n in range(n_min, n_max + 1, step):
        params = params_fn(n)
        if params is None:
            continue
        w1, w2, w3, h1, h2, h3 = params
        try:
            A_val = np.array(A_func(w1, w2, w3, h1, h2, h3), dtype=complex)
        except ZeroDivisionError:
            continue
        if not np.isfinite(A_val).all():
            continue
        if require_unitary and not _is_unitary(A_val):
            continue
        abc = _decomp(A_val)
        if abc is None:
            continue
        a, b, c = abc
        xs.append(a)
        ys.append(b)
        zs.append(abs(c))
    return xs, ys, zs


def _family_limit_point(A_sym: sp.Matrix, symbols: tuple[sp.Symbol, ...], name: str) -> tuple[float, float, float] | None:
    w1, w2, w3, h1, h2, h3 = symbols
    if name == "gun":
        A_family = A_sym.subs({w2: 2, w3: 1, h1: 1, h2: 1, h3: 1})
        A_lim = A_family.applyfunc(lambda expr: sp.limit(expr, w1, sp.oo))
    elif name == "chocolate":
        A_family = A_sym.subs({w2: w1 - 1, w3: w1 - 2, h1: 1, h2: 1, h3: 1})
        A_lim = A_family.applyfunc(lambda expr: sp.limit(expr, w1, sp.oo))
    elif name == "uzi":
        A_family = A_sym.subs({w1: 3, w2: 2, w3: 1, h1: 1, h3: 1})
        A_lim = A_family.applyfunc(lambda expr: sp.limit(expr, h2, sp.oo))
    elif name == "F":
        A_family = A_sym.subs({w1: 3, w2: 2, w3: 1, h1: 1, h2: 1})
        A_lim = A_family.applyfunc(lambda expr: sp.limit(expr, h3, sp.oo))
    else:
        return None

    try:
        A_val = np.array(A_lim.evalf(), dtype=complex)
    except Exception:
        return None
    if not np.isfinite(A_val).all():
        return None
    abc = _decomp(A_val)
    if abc is None:
        return None
    a, b, c = abc
    return a, b, abs(c)


def _add_weyl_chamber_2d(fig: go.Figure) -> None:
    # 2D projection in (b, |c|): 0 <= b <= pi/4, 0 <= |c| <= b.
    pi4 = 0.25 * np.pi
    xs = [0, pi4, pi4, 0, 0]
    ys = [0, 0, pi4, 0, 0]
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color="black", width=2),
            name="Weyl Chamber",
            hoverinfo="skip",
            showlegend=False,
        )
    )


def main() -> None:
    args = parse_args()
    A_sym, symbols = build_symbolic_a_matrix()
    A_func = sp.lambdify(symbols, A_sym, "numpy")

    scan = _scan_points(args, A_func)

    # Family parameterizations
    def gun(n: int):
        return (n, 2, 1, 1, 1, 1)

    def chocolate(n: int):
        if n < 3:
            return None
        return (n, n - 1, n - 2, 1, 1, 1)

    def uzi(n: int):
        return (3, 2, 1, 1, n, 1)

    def f_family(n: int):
        return (3, 2, 1, 1, 1, n)

    families = {
        "gun": gun,
        "chocolate": chocolate,
        "uzi": uzi,
        "F": f_family,
    }

    fig = go.Figure()
    _add_weyl_chamber_2d(fig)

    if scan.size:
        fig.add_trace(
            go.Scatter(
                x=scan[:, 1],
                y=np.abs(scan[:, 2]),
                mode="markers",
                marker=dict(size=3, color="rgba(120,120,120,0.35)"),
                name="symbolic scan",
                hoverinfo="skip",
            )
        )

    colors = {
        "gun": "red",
        "chocolate": "blue",
        "uzi": "green",
        "F": "orange",
    }
    for name, fn in families.items():
        xs, ys, zs = _family_series(
            A_func,
            args.require_unitary,
            fn,
            args.family_n_min,
            args.family_n_max,
            args.family_step,
        )
        if not xs:
            continue
        fig.add_trace(
            go.Scatter(
                x=ys,
                y=zs,
                mode="markers",
                marker=dict(size=5, color=colors.get(name, "black"), opacity=0.85),
                name=f"{name} family",
            )
        )
        limit_point = _family_limit_point(A_sym, symbols, name)
        if limit_point is not None:
            fig.add_trace(
                go.Scatter(
                    x=[limit_point[1]],
                    y=[limit_point[2]],
                    mode="markers",
                    marker=dict(
                        size=9,
                        color=colors.get(name, "black"),
                        symbol="diamond",
                        line=dict(color="black", width=1),
                    ),
                    name=f"{name} limit",
                )
            )

    fig.update_layout(
        title="Weyl chamber (b vs |c|) with symbolic scan + families",
        xaxis=dict(title="b"),
        yaxis=dict(title="|c|"),
        legend=dict(itemsizing="constant"),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pio.renderers.default = args.plotly_renderer
    fig.write_html(args.output)
    fig.show()
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
