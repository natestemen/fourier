#!/usr/bin/env python3
"""Plot Weyl (a,b,c) for a 4-addable-cell family parameterized by (w1,h1,w2,h2,w3,h3)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from qiskit.synthesis import TwoQubitWeylDecomposition
from yungdiagram import YoungDiagram
from scipy.optimize import curve_fit

from compute_matrix import A_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--w1", type=int, default=6)
    parser.add_argument("--h1", type=int, default=2)
    parser.add_argument("--w2", type=int, default=4)
    parser.add_argument("--h2", type=int, default=1)
    parser.add_argument("--w3", type=int, default=2)
    parser.add_argument("--h3", type=int, default=1)
    parser.add_argument(
        "--vary",
        default="w1",
        help="Comma-separated parameters to sweep (e.g., w1,w2,h1).",
    )
    parser.add_argument("--start", type=int, default=5)
    parser.add_argument("--stop", type=int, default=12)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/plots/param_family_weyl.png"),
        help="Output path for the plot image.",
    )
    parser.add_argument(
        "--plotly-output",
        type=Path,
        default=Path("data/plots/param_family_weyl_3d.html"),
        help="Output HTML path for the Weyl chamber scatter plot.",
    )
    parser.add_argument(
        "--plotly-renderer",
        default="browser",
        help="Plotly renderer name (default: browser).",
    )
    parser.add_argument(
        "--abs",
        action="store_true",
        help="Plot |c| instead of c in both plots.",
    )
    return parser.parse_args()


def _build_partition(w1: int, h1: int, w2: int, h2: int, w3: int, h3: int) -> list[int]:
    return [w1] * h1 + [w2] * h2 + [w3] * h3


def _valid(w1: int, w2: int, w3: int, h1: int, h2: int, h3: int) -> bool:
    if min(w1, w2, w3, h1, h2, h3) <= 0:
        return False
    if not (w1 > w2 > w3):
        return False
    return True


def _adjust_widths(w1: int, w2: int, w3: int) -> tuple[int, int, int]:
    """Increase larger widths as needed to satisfy w1 > w2 > w3."""
    if w2 >= w1:
        w1 = w2 + 1
    if w3 >= w2:
        w2 = w3 + 1
        if w2 >= w1:
            w1 = w2 + 1
    return w1, w2, w3


def main() -> None:
    args = parse_args()
    values = list(range(args.start, args.stop + 1, args.step))
    vary_list = [v.strip() for v in args.vary.split(",") if v.strip()]
    valid_keys = {"w1", "h1", "w2", "h2", "w3", "h3"}
    for v in vary_list:
        if v not in valid_keys:
            raise SystemExit(f"Invalid --vary value: {v}")

    family_points: dict[str, list[tuple[int, float, float, float, tuple[int, ...]]]] = {}
    for vary in vary_list:
        points = []
        for value in values:
            params = {
                "w1": args.w1,
                "h1": args.h1,
                "w2": args.w2,
                "h2": args.h2,
                "w3": args.w3,
                "h3": args.h3,
            }
            params[vary] = value
            params["w1"], params["w2"], params["w3"] = _adjust_widths(
                params["w1"], params["w2"], params["w3"]
            )
            if not _valid(
                params["w1"],
                params["w2"],
                params["w3"],
                params["h1"],
                params["h2"],
                params["h3"],
            ):
                continue
            partition = _build_partition(**params)
            diagram = YoungDiagram(partition)
            matrix = A_matrix(diagram)
            decomp = TwoQubitWeylDecomposition(matrix)
            points.append(
                (value, float(decomp.a), float(decomp.b), float(decomp.c), tuple(partition))
            )
        if points:
            family_points[vary] = points

    if not family_points:
        raise SystemExit("No valid parameter values in the requested sweep.")


    fig, ax = plt.subplots()
    for vary, points in family_points.items():
        xs = [p[0] for p in points]
        b_vals = np.array([p[2] for p in points])
        c_vals = np.array([p[3] for p in points])

        if args.abs:
            c_vals = np.abs(c_vals)


        ax.plot(xs, b_vals, marker="o", label=f"{vary}: b")
        ax.plot(xs, c_vals, marker="o", label=f"{vary}: c")
    ax.set_xlabel("sweep value")
    ax.set_ylabel("Weyl components")
    ax.set_title(
        "Weyl components (|c|) vs parameter sweep" if args.abs else "Weyl components vs parameter sweep"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {args.output}")
    plt.show()

    pio.renderers.default = args.plotly_renderer
    _plot_weyl_chamber(family_points, args.plotly_output, abs_c=args.abs)


def _plot_weyl_chamber(
    families: dict[str, list[tuple[int, float, float, float, tuple[int, ...]]]],
    output: Path,
    abs_c: bool = False,
) -> None:
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
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(color="black", width=3),
            name="Weyl Chamber",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    colors = [
        "red",
        "blue",
        "green",
        "orange",
        "purple",
        "teal",
    ]
    for idx, (vary, points) in enumerate(families.items()):
        color = colors[idx % len(colors)]
        fig.add_trace(
            go.Scatter3d(
                x=[p[1] for p in points],
                y=[p[2] for p in points],
                z=[abs(p[3]) if abs_c else p[3] for p in points],
                mode="markers",
                marker=dict(size=5, color=color, opacity=0.7),
                name=f"{vary} family",
                hovertext=[
                    f"{vary}={p[0]} partition={p[4]} c={abs(p[3]) if abs_c else p[3]:.6f}"
                    for p in points
                ],
                hoverinfo="text",
            )
        )
    zlabel = "|c|" if abs_c else "c"
    title = "Weyl Chamber Scatter (|c|)" if abs_c else "Weyl Chamber Scatter (family sweep)"
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="a", yaxis_title="b", zaxis_title=zlabel, aspectmode="cube"),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output)
    fig.show()


if __name__ == "__main__":
    main()
