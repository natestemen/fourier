"""Weyl-chamber scatter of the generic 4-addable family with named-family overlays.

Question: where do 4-addable A-matrices live in the Weyl chamber — is a pinned
at π/4 across the whole (w1,w2,w3,h1,h2,h3) block-parameter space, and how do
the free coordinates (b, |c|) fill the chamber face?

Supports report.md, Finding 1: every 4-addable A-matrix has Weyl coordinate
a = π/4 while (b, c) vary continuously over the family.

Expected result: all scanned points have a = π/4; (b, |c|) fill a region of
the chamber face, with the four one-parameter families (gun, f, uzi,
chocolate-bar) tracing curves toward their symbolic n→∞ limit points.

Grid-scans valid (w1,w2,w3,h1,h2,h3) configurations of the generic symbolic
A-matrix (fourier.amatrix.a_matrix_generic4), overlays the four named
families and their exact limits, and writes:

  - data/plots/symbolic_weyl_scan_bc.png   (b vs |c| plane, scan colored by size)
  - data/symbolic_weyl_scan.csv            (a, b, c, size, block parameters)
  - data/plots/symbolic_weyl_scan_3d.html  (plotly (a, b, |c|) chamber; --html only)

Replaces scan_symbolic_weyl.py and plot_all_families_weyl.py (union of both:
the scan/CSV and matplotlib plane plot from the former, the four-family
overlays and interactive figure from the latter — the interactive figure is
now 3D so the a = π/4 pinning is visible). No plt.show()/fig.show().
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from fourier.amatrix import a_matrix_generic4
from fourier.weyl import weyl_coordinates

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
PLOT_DIR = DATA_DIR / "plots"

Params = tuple[int, int, int, int, int, int]
Symbols = tuple[sp.Symbol, ...]


@dataclass(frozen=True)
class Family:
    """A one-parameter family of generic 4-addable diagrams."""

    color: str
    params: Callable[[int], Params | None]  # n -> (w1, w2, w3, h1, h2, h3)
    # symbols -> (substitution fixing all but one symbol, the symbol sent to ∞)
    limit_subs: Callable[[Symbols], tuple[dict[sp.Symbol, object], sp.Symbol]]


FAMILIES: dict[str, Family] = {
    "gun": Family(
        color="red",
        params=lambda n: (n, 2, 1, 1, 1, 1),
        limit_subs=lambda s: ({s[1]: 2, s[2]: 1, s[3]: 1, s[4]: 1, s[5]: 1}, s[0]),
    ),
    "chocolate-bar": Family(
        color="blue",
        params=lambda n: (n, n - 1, n - 2, 1, 1, 1) if n >= 3 else None,
        limit_subs=lambda s: (
            {s[1]: s[0] - 1, s[2]: s[0] - 2, s[3]: 1, s[4]: 1, s[5]: 1},
            s[0],
        ),
    ),
    "uzi": Family(
        color="green",
        params=lambda n: (3, 2, 1, 1, n, 1),
        limit_subs=lambda s: ({s[0]: 3, s[1]: 2, s[2]: 1, s[3]: 1, s[5]: 1}, s[4]),
    ),
    "f": Family(
        color="orange",
        params=lambda n: (3, 2, 1, 1, 1, n),
        limit_subs=lambda s: ({s[0]: 3, s[1]: 2, s[2]: 1, s[3]: 1, s[4]: 1}, s[5]),
    ),
}


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
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restrict to the generic stratum (w1 >= w2+2, w2 >= w3+2, h_i >= 2) "
        "where the symbolic matrix's assumptions hold; --no-generic-only allows "
        "edge shapes.",
    )
    parser.add_argument(
        "--require-unitary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip matrices that are not numerically unitary.",
    )
    parser.add_argument("--scan-limit", type=int, default=0, help="Optional cap on scan points.")
    parser.add_argument("--family-n-min", type=int, default=3)
    parser.add_argument("--family-n-max", type=int, default=10000)
    parser.add_argument("--family-step", type=int, default=10)
    parser.add_argument(
        "--html",
        action="store_true",
        help="Also write an interactive plotly 3D (a, b, |c|) chamber plot.",
    )
    return parser.parse_args()


def is_unitary(mat: np.ndarray, tol: float = 1e-6) -> bool:
    return np.allclose(mat.conj().T @ mat, np.eye(mat.shape[0]), atol=tol)


def evaluate(A_func, params: Params, require_unitary: bool) -> np.ndarray | None:
    """The numeric matrix at `params`, or None if degenerate/non-unitary."""
    with np.errstate(invalid="ignore"):
        try:
            A = np.array(A_func(*params), dtype=complex)
        except ZeroDivisionError:  # coincident contents at edge shapes
            return None
    if not np.isfinite(A).all():
        return None
    if require_unitary and not is_unitary(A):
        return None
    return A


def weyl_or_none(A: np.ndarray) -> tuple[float, float, float] | None:
    # Non-unitary matrices (reachable with --no-require-unitary) can make the
    # Weyl decomposition fail; skip those points like the original scan did.
    try:
        return weyl_coordinates(A)
    except Exception:
        return None


def scan_points(args: argparse.Namespace, A_func) -> list[tuple]:
    """Rows (a, b, c, size, w1, w2, w3, h1, h2, h3) over the valid grid."""
    rows: list[tuple] = []
    for w1 in range(args.w1_min, args.w1_max + 1):
        for w2 in range(args.w2_min, min(args.w2_max, w1 - 1) + 1):
            for w3 in range(args.w3_min, min(args.w3_max, w2 - 1) + 1):
                for h1, h2, h3 in product(range(args.h_min, args.h_max + 1), repeat=3):
                    if min(w1, w2, w3, h1, h2, h3) <= 0:
                        continue
                    if args.generic_only and not (
                        w1 >= w2 + 2 and w2 >= w3 + 2 and min(h1, h2, h3) >= 2
                    ):
                        continue

                    A = evaluate(A_func, (w1, w2, w3, h1, h2, h3), args.require_unitary)
                    if A is None:
                        continue
                    abc = weyl_or_none(A)
                    if abc is None:
                        continue

                    size = w1 * h1 + w2 * h2 + w3 * h3
                    rows.append((*abc, size, w1, w2, w3, h1, h2, h3))
                    if args.scan_limit and len(rows) >= args.scan_limit:
                        return rows
    return rows


def family_series(
    A_func, params_fn, n_min: int, n_max: int, step: int, require_unitary: bool
) -> np.ndarray:
    """(a, b, |c|) rows along a named family."""
    rows = []
    for n in range(n_min, n_max + 1, step):
        params = params_fn(n)
        if params is None:
            continue
        A = evaluate(A_func, params, require_unitary)
        if A is None:
            continue
        abc = weyl_or_none(A)
        if abc is None:
            continue
        a, b, c = abc
        rows.append((a, b, abs(c)))
    return np.array(rows, dtype=float).reshape(-1, 3)


def family_limit(
    A_sym: sp.Matrix, symbols: Symbols, name: str
) -> tuple[float, float, float]:
    """(a, b, |c|) of the family's exact n→∞ limit matrix."""
    subs, var = FAMILIES[name].limit_subs(symbols)
    A_lim = A_sym.subs(subs).applyfunc(lambda expr: sp.limit(expr, var, sp.oo))
    a, b, c = weyl_coordinates(np.array(A_lim.evalf(), dtype=complex))
    return a, b, abs(c)


def plot_plane(
    scan: np.ndarray,
    families: dict[str, np.ndarray],
    limits: dict[str, tuple[float, float, float]],
    output: Path,
) -> None:
    """b vs |c| plane: chamber boundary, scan colored by size, family overlays."""
    fig, ax = plt.subplots()

    pi4 = 0.25 * np.pi
    ax.plot([0, pi4, pi4, 0], [0, 0, pi4, 0], color="black", linewidth=1)

    if scan.size:
        sc = ax.scatter(
            scan[:, 1],
            np.abs(scan[:, 2]),
            s=14,
            alpha=0.5,
            c=scan[:, 3],
            cmap="viridis",
            linewidths=0,
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("diagram size (# cells)")

    for name, series in families.items():
        color = FAMILIES[name].color
        ax.scatter(series[:, 1], series[:, 2], s=8, alpha=0.9, color=color, label=f"{name} family")
        _, b_lim, c_lim = limits[name]
        ax.scatter(
            [b_lim], [c_lim], s=80, marker="*", color=color,
            edgecolors="black", linewidths=0.6, label=f"{name} limit n→∞",
        )

    ax.set_xlabel("b")
    ax.set_ylabel("|c|")
    ax.set_title("Weyl b vs |c|: symbolic 4-addable scan + families (a = π/4)")
    ax.legend(fontsize=7)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plane plot to {output}")


def plot_chamber_3d(
    scan: np.ndarray,
    families: dict[str, np.ndarray],
    limits: dict[str, tuple[float, float, float]],
    output: Path,
) -> None:
    """Interactive (a, b, |c|) scatter inside the Weyl chamber wireframe."""
    import plotly.graph_objects as go

    pi4 = 0.25 * np.pi
    verts = np.array(
        [[0, 0, 0], [pi4, 0, 0], [pi4, pi4, 0], [pi4, pi4, pi4], [pi4, pi4, -pi4]]
    )
    edges = [(0, 1), (1, 2), (2, 3), (2, 4), (0, 2), (0, 3), (0, 4), (1, 3), (1, 4)]
    xs, ys, zs = [], [], []
    for i, j in edges:
        xs += [verts[i][0], verts[j][0], None]
        ys += [verts[i][1], verts[j][1], None]
        zs += [verts[i][2], verts[j][2], None]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(color="black", width=3),
            name="Weyl chamber", hoverinfo="skip", showlegend=False,
        )
    )
    if scan.size:
        fig.add_trace(
            go.Scatter3d(
                x=scan[:, 0], y=scan[:, 1], z=np.abs(scan[:, 2]),
                mode="markers",
                marker=dict(size=3, color="rgba(120,120,120,0.35)"),
                name="symbolic scan", hoverinfo="skip",
            )
        )
    for name, series in families.items():
        color = FAMILIES[name].color
        fig.add_trace(
            go.Scatter3d(
                x=series[:, 0], y=series[:, 1], z=series[:, 2],
                mode="markers",
                marker=dict(size=3, color=color, opacity=0.85),
                name=f"{name} family",
            )
        )
        a_lim, b_lim, c_lim = limits[name]
        fig.add_trace(
            go.Scatter3d(
                x=[a_lim], y=[b_lim], z=[c_lim],
                mode="markers",
                marker=dict(size=6, color=color, symbol="diamond",
                            line=dict(color="black", width=1)),
                name=f"{name} limit",
            )
        )
    fig.update_layout(
        title="Weyl chamber: symbolic 4-addable scan + families",
        scene=dict(xaxis_title="a", yaxis_title="b", zaxis_title="|c|", aspectmode="cube"),
        legend=dict(itemsizing="constant"),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output)
    print(f"Saved 3D chamber plot to {output}")


def main() -> None:
    args = parse_args()
    A_sym, symbols = a_matrix_generic4()
    A_func = sp.lambdify(symbols, A_sym, "numpy")

    rows = scan_points(args, A_func)
    if not rows:
        raise SystemExit("No valid scan points generated with the requested ranges.")
    scan = np.array([r[:4] for r in rows], dtype=float)

    a_vals = scan[:, 0]
    print(f"Scan: {len(rows)} points; max |a - pi/4| = {np.abs(a_vals - np.pi / 4).max():.3g}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / "symbolic_weyl_scan.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["weyl_a", "weyl_b", "weyl_c", "size", "w1", "w2", "w3", "h1", "h2", "h3"])
        writer.writerows(rows)
    print(f"Saved scan points to {csv_path}")

    families: dict[str, np.ndarray] = {}
    limits: dict[str, tuple[float, float, float]] = {}
    for name, family in FAMILIES.items():
        series = family_series(
            A_func, family.params, args.family_n_min, args.family_n_max,
            args.family_step, args.require_unitary,
        )
        if series.size == 0:
            continue
        families[name] = series
        limits[name] = family_limit(A_sym, symbols, name)
        a_lim, b_lim, c_lim = limits[name]
        print(f"{name} limit n→∞: a={a_lim:.6g} b={b_lim:.6g} |c|={c_lim:.6g}")

    plot_plane(scan, families, limits, PLOT_DIR / "symbolic_weyl_scan_bc.png")
    if args.html:
        plot_chamber_3d(scan, families, limits, PLOT_DIR / "symbolic_weyl_scan_3d.html")


if __name__ == "__main__":
    main()
