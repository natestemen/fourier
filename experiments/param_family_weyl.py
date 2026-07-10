"""Weyl coordinates of exact A-matrices while sweeping one block parameter at a time.

Question: holding a base 3-block diagram (w1,h1,w2,h2,w3,h3) fixed, how do the
Weyl coordinates (a, b, c) move as each chosen parameter is swept — computed
from the exact numeric A-matrix of the genuine diagram (fourier.amatrix.a_matrix),
not the generic symbolic parametrization, so edge shapes (h_i = 1, adjacent
widths) are handled exactly?

Supports report.md, Finding 1: a = π/4 for every 4-addable A-matrix; sweeping
any block parameter moves only (b, c).

Expected result: a stays at π/4 ≈ 0.7854 for every swept diagram while b and
|c| trace smooth curves in the sweep parameter.

For each parameter in --vary (comma-separated), the sweep value replaces that
parameter in the base configuration; widths are then bumped upward as needed
to restore w1 > w2 > w3 (same semantics as the original script). Outputs:

  - data/plots/param_family_weyl.png      (b and |c| vs sweep value, per parameter)
  - data/plots/param_family_weyl_3d.html  (plotly (a, b, |c|) chamber scatter)

Replaces plot_param_family.py. Behavior change: no plt.show()/fig.show(),
so no --plotly-renderer option.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from yungdiagram import YoungDiagram

from fourier.amatrix import a_matrix
from fourier.weyl import weyl_coordinates

REPO_ROOT = Path(__file__).resolve().parent.parent
PLOT_DIR = REPO_ROOT / "data" / "plots"

PARAM_KEYS = ("w1", "h1", "w2", "h2", "w3", "h3")

# (sweep value, a, b, c, partition)
Point = tuple[int, float, float, float, tuple[int, ...]]


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
        default=PLOT_DIR / "param_family_weyl.png",
        help="Output path for the plot image.",
    )
    parser.add_argument(
        "--plotly-output",
        type=Path,
        default=PLOT_DIR / "param_family_weyl_3d.html",
        help="Output HTML path for the Weyl chamber scatter plot.",
    )
    return parser.parse_args()


def adjust_widths(w1: int, w2: int, w3: int) -> tuple[int, int, int]:
    """Increase larger widths as needed to satisfy w1 > w2 > w3."""
    if w2 >= w1:
        w1 = w2 + 1
    if w3 >= w2:
        w2 = w3 + 1
        if w2 >= w1:
            w1 = w2 + 1
    return w1, w2, w3


def sweep(vary: str, values: list[int], base: dict[str, int]) -> list[Point]:
    """Weyl points of the exact A-matrix as `vary` runs over `values`."""
    points: list[Point] = []
    for value in values:
        params = dict(base)
        params[vary] = value
        params["w1"], params["w2"], params["w3"] = adjust_widths(
            params["w1"], params["w2"], params["w3"]
        )
        if min(params[k] for k in PARAM_KEYS) <= 0:
            continue
        partition = (
            [params["w1"]] * params["h1"]
            + [params["w2"]] * params["h2"]
            + [params["w3"]] * params["h3"]
        )
        a, b, c = weyl_coordinates(a_matrix(YoungDiagram(partition)))
        points.append((value, a, b, c, tuple(partition)))
    return points


def plot_series(families: dict[str, list[Point]], output: Path) -> None:
    fig, ax = plt.subplots()
    for vary, points in families.items():
        xs = [p[0] for p in points]
        ax.plot(xs, [p[2] for p in points], label=f"{vary}: b")
        ax.plot(xs, [abs(p[3]) for p in points], label=f"{vary}: |c|")
    ax.set_xlabel("sweep value")
    ax.set_ylabel("Weyl coordinate")
    ax.grid(True, alpha=0.3)
    ax.legend()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output}")


def plot_weyl_chamber(families: dict[str, list[Point]], output: Path) -> None:
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
            name="Weyl Chamber", hoverinfo="skip", showlegend=False,
        )
    )
    colors = ["red", "blue", "green", "orange", "purple", "teal"]
    for idx, (vary, points) in enumerate(families.items()):
        fig.add_trace(
            go.Scatter3d(
                x=[p[1] for p in points],
                y=[p[2] for p in points],
                z=[abs(p[3]) for p in points],
                mode="markers",
                marker=dict(size=5, color=colors[idx % len(colors)], opacity=0.7),
                name=f"{vary} family",
                hovertext=[
                    f"{vary}={p[0]} partition={p[4]} c={abs(p[3]):.6f}" for p in points
                ],
                hoverinfo="text",
            )
        )
    fig.update_layout(
        title="Weyl Chamber Scatter (|c|)",
        scene=dict(xaxis_title="a", yaxis_title="b", zaxis_title="|c|", aspectmode="cube"),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output)
    print(f"Saved 3D chamber plot to {output}")


def main() -> None:
    args = parse_args()
    values = list(range(args.start, args.stop + 1, args.step))
    vary_list = [v.strip() for v in args.vary.split(",") if v.strip()]
    for v in vary_list:
        if v not in PARAM_KEYS:
            raise SystemExit(f"Invalid --vary value: {v}")

    base = {k: getattr(args, k) for k in PARAM_KEYS}
    families = {vary: pts for vary in vary_list if (pts := sweep(vary, values, base))}
    if not families:
        raise SystemExit("No valid parameter values in the requested sweep.")

    for vary, points in families.items():
        a_vals = np.array([p[1] for p in points])
        print(
            f"{vary}: {len(points)} points, max |a - pi/4| = "
            f"{np.abs(a_vals - np.pi / 4).max():.3g}"
        )

    plot_series(families, args.output)
    plot_weyl_chamber(families, args.plotly_output)


if __name__ == "__main__":
    main()
