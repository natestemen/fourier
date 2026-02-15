#!/usr/bin/env python3
"""Interactive Weyl-chamber scatter plot for generated Fourier data.

This script loads either the dense per-layer CSVs from ``generate_data.py`` or
the compact ``generate_weyl.py`` outputs and renders an interactive 3-D
scatter plot of the (a, b, c) coordinates from the TwoQubitWeylDecomposition.
Traces are color-coded by Young diagram so you can rotate, zoom, and inspect
clusters from the terminal (Plotly opens in a browser window by default).
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio


DEFAULT_CSV = Path("data/weyl_4_addable_size_25.csv")
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
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Path to the generated CSV file (default: {DEFAULT_CSV}).",
    )
    parser.add_argument(
        "--renderer",
        default="browser",
        help="Plotly renderer name (default: browser).",
    )
    return parser.parse_args()


def load_rows(path: Path) -> tuple[dict[str, list[dict[str, float]]], tuple[float, float] | None]:
    grouped: dict[str, list[dict[str, float]]] = {}
    min_color = math.inf
    max_color = -math.inf
    with path.open() as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = {"weyl_a", "weyl_b", "weyl_c"} - fieldnames
        if missing:
            raise SystemExit(
                f"{path} is missing Weyl columns {missing}. "
                "Regenerate the data with the latest tooling."
            )
        for row in reader:
            diag = row.get("diagram", "unknown")
            metrics = _diagram_metrics(diag, row)
            if not math.isnan(metrics["color"]):
                min_color = min(min_color, metrics["color"])
                max_color = max(max_color, metrics["color"])
            point = {
                "a": float(row["weyl_a"]),
                "b": float(row["weyl_b"]),
                "c": float(row["weyl_c"]),
                **metrics,
            }
            for key in ("entry", "layer", "qubit"):
                if key in fieldnames:
                    point[key] = row.get(key, "")
            grouped.setdefault(diag, []).append(point)
    if not grouped:
        raise SystemExit(f"No rows found in {path}.")
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


def _diagram_metrics(diagram: str, row: dict[str, str]) -> dict[str, float]:
    parts = _parse_partition(diagram)
    size_val = _extract_size(row, parts)
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


def _extract_size(row: dict[str, str], parts: tuple[int, ...] | None) -> float:
    size_field = row.get("size")
    if size_field not in (None, ""):
        try:
            return float(size_field)
        except ValueError:
            pass
    if parts:
        return float(sum(parts))
    return math.nan


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


# def _symmetry_deviation(parts: tuple[int, ...] | None) -> float:
#     if not parts:
#         return math.nan
#     transpose = _transpose_partition(parts)
#     max_len = max(len(parts), len(transpose))
#     padded_rows = list(parts) + [0] * (max_len - len(parts))
#     padded_cols = list(transpose) + [0] * (max_len - len(transpose))
#     return float(sum(abs(r - c) for r, c in zip(padded_rows, padded_cols)))

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
    data_by_diag, color_range = load_rows(args.csv)

    fig = go.Figure()
    fig.add_trace(build_weyl_edges())
    for idx, (diag, points) in enumerate(data_by_diag.items()):
        show_colorbar = color_range is not None and idx == 0
        fig.add_trace(build_scatter(diag, points, color_range, show_colorbar))

    fig.update_layout(
        title=f"Weyl Chamber Scatter: {args.csv}",
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
