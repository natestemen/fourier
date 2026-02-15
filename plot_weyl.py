#!/usr/bin/env python3
"""Interactive Weyl-chamber scatter plot generated from scratch.

This script generates Fourier matrices from Young diagrams on the fly,
computes the TwoQubitWeylDecomposition (a, b, c) coordinates, and renders an
interactive 3-D scatter plot. Traces are color-coded by Young diagram so you
can rotate, zoom, and inspect clusters from the terminal (Plotly opens in a
browser window by default).
"""
from __future__ import annotations

import argparse
import math

import plotly.graph_objects as go
import plotly.io as pio
from qiskit.synthesis import TwoQubitWeylDecomposition

from compute_matrix import A_matrix
from helper import find_yds_with_fixed_addable_cells

PI_OVER_FOUR = 0.25 * 3.141592653589793
CHAMBER_VERTICES = (
    (0.0, 0.0, 0.0),
    (PI_OVER_FOUR, 0.0, 0.0),
    (PI_OVER_FOUR, PI_OVER_FOUR, 0.0),
    (PI_OVER_FOUR, PI_OVER_FOUR, PI_OVER_FOUR),
    (PI_OVER_FOUR, PI_OVER_FOUR, -PI_OVER_FOUR),
)
CHAMBER_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (2, 4),
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 3),
    (1, 4),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-diagram-size",
        type=int,
        required=True,
        help="Upper bound on Young diagram sizes to enumerate.",
    )
    parser.add_argument(
        "--max-diagrams",
        type=int,
        default=None,
        help="Optional cap on the number of diagrams to plot.",
    )
    parser.add_argument(
        "--renderer",
        default="browser",
        help="Plotly renderer name (default: browser).",
    )
    return parser.parse_args()


def load_rows(max_diagram_size: int, max_diagrams: int | None) -> tuple[
    dict[str, list[dict[str, float]]], tuple[float, float] | None
]:
    grouped: dict[str, list[dict[str, float]]] = {}
    min_color = math.inf
    max_color = -math.inf

    diagrams = find_yds_with_fixed_addable_cells(4, max_diagram_size)
    if not diagrams:
        raise SystemExit("No diagrams found for the requested size bounds.")
    if max_diagrams is not None:
        diagrams = diagrams[:max_diagrams]

    for idx, diagram in enumerate(diagrams):
        label = _diagram_label(diagram)
        metrics = _diagram_metrics(label)
        if not math.isnan(metrics["color"]):
            min_color = min(min_color, metrics["color"])
            max_color = max(max_color, metrics["color"])

        matrix = A_matrix(diagram)
        decomp = TwoQubitWeylDecomposition(matrix)
        point = {
            "a": float(decomp.a),
            "b": float(decomp.b),
            "c": float(decomp.c),
            "entry": idx,
            **metrics,
        }
        grouped.setdefault(label, []).append(point)

    color_range = None if math.isinf(min_color) else (min_color, max_color)
    return grouped, color_range


def build_scatter(
    diag: str,
    points: list[dict[str, float]],
    color_range: tuple[float, float] | None,
    show_colorbar: bool,
) -> go.Scatter3d:
    hover = [_hover_text(diag, p) for p in points]
    marker = dict(size=4, opacity=0.75)
    if color_range is not None:
        colors = [p.get("color", math.nan) for p in points]
        marker.update(
            color=colors,
            colorscale="Viridis",
            cmin=color_range[0],
            cmax=color_range[1],
            colorbar=dict(title="Symmetry deviation"),
            showscale=show_colorbar,
        )
    return go.Scatter3d(
        x=[p["a"] for p in points],
        y=[p["b"] for p in points],
        z=[p["c"] for p in points],
        mode="markers",
        marker=marker,
        name=str(diag),
        hovertext=hover,
        hoverinfo="text",
    )


def _hover_text(diagram: str, point: dict[str, float]) -> str:
    fields = [f"diagram={diagram}"]
    for label in ("entry", "size", "width", "height", "symmetry", "layer", "qubit"):
        if label in point and point[label] not in ("", None):
            fields.append(f"{label}={point[label]}")
    return "<br>".join(fields)


def _diagram_metrics(diagram: str) -> dict[str, float]:
    parts = _parse_partition(diagram)
    size_val = float(sum(parts)) if parts else math.nan
    width = max(parts) if parts else math.nan
    height = float(len(parts)) if parts else math.nan
    symmetry = _symmetry_deviation(parts)
    color_val = symmetry
    return {
        "size": size_val,
        "width": width,
        "height": height,
        "symmetry": symmetry,
        "color": color_val,
    }


def _parse_partition(diagram: str) -> tuple[int, ...] | None:
    label = (diagram or "").strip()
    if label.startswith("(") and label.endswith(")"):
        inner = label[1:-1].strip()
        if inner:
            try:
                return tuple(int(part.strip()) for part in inner.split(","))
            except ValueError:
                return None
    return None


def _diagram_label(diagram) -> str:
    for attr in ("partition", "parts", "rows", "row_lengths"):
        if hasattr(diagram, attr):
            value = getattr(diagram, attr)
            value = value() if callable(value) else value
            try:
                return str(tuple(value))
            except TypeError:
                pass
    return str(diagram)


def _symmetry_deviation(parts: tuple[int, ...] | None) -> float:
    if not parts:
        return math.nan

    transpose = _transpose_partition(parts)

    max_len = max(len(parts), len(transpose))
    padded_rows = list(parts) + [0] * (max_len - len(parts))
    padded_cols = list(transpose) + [0] * (max_len - len(transpose))

    deviation = sum(abs(r - c) for r, c in zip(padded_rows, padded_cols))
    size = sum(parts)

    return 1.0 - deviation / (2 * size)

def _transpose_partition(parts: tuple[int, ...]) -> tuple[int, ...]:
    max_height = max(parts)
    conjugate = []
    for k in range(1, max_height + 1):
        conjugate.append(sum(1 for part in parts if part >= k))
    return tuple(conjugate)


def build_weyl_edges() -> go.Scatter3d:
    xs, ys, zs = [], [], []
    for start, end in CHAMBER_EDGES:
        xs += [CHAMBER_VERTICES[start][0], CHAMBER_VERTICES[end][0], None]
        ys += [CHAMBER_VERTICES[start][1], CHAMBER_VERTICES[end][1], None]
        zs += [CHAMBER_VERTICES[start][2], CHAMBER_VERTICES[end][2], None]
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line=dict(color="black", width=3),
        name="Weyl Chamber",
        hoverinfo="skip",
        showlegend=False,
    )


def main() -> None:
    args = parse_args()
    pio.renderers.default = args.renderer
    data_by_diag, color_range = load_rows(args.max_diagram_size, args.max_diagrams)

    fig = go.Figure()
    fig.add_trace(build_weyl_edges())
    for idx, (diag, points) in enumerate(data_by_diag.items()):
        show_colorbar = color_range is not None and idx == 0
        fig.add_trace(build_scatter(diag, points, color_range, show_colorbar))

    fig.update_layout(
        title="Weyl Chamber Scatter (generated)",
        scene=dict(
            xaxis_title="a",
            yaxis_title="b",
            zaxis_title="c",
            aspectmode="cube",
        ),
        legend=dict(title="Diagram"),
    )
    fig.show()


if __name__ == "__main__":
    main()
