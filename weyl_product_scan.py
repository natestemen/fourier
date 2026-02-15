#!/usr/bin/env python3
"""Generate 2-qubit Fourier matrices, multiply a handful, and compare results."""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
from qiskit.synthesis import TwoQubitWeylDecomposition

from compute_matrix import A_matrix
from helper import find_yds_with_fixed_addable_cells


def _weyl_coefficients(matrix: np.ndarray) -> tuple[float, float, float]:
    matrix = np.asarray(matrix, dtype=np.complex128)
    decomp = TwoQubitWeylDecomposition(matrix)
    return float(decomp.a), float(decomp.b), float(decomp.c)


def _unitary_distance(U: np.ndarray, V: np.ndarray) -> float:
    vdot = np.vdot(V, U)
    phase = 1.0 if vdot == 0 else vdot / abs(vdot)
    diff = U - phase * V
    return float(np.linalg.norm(diff) / np.linalg.norm(V))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-diagram-size",
        type=int,
        required=True,
        help="Upper bound on Young diagram sizes to enumerate.",
    )
    parser.add_argument(
        "--num-matrices",
        type=int,
        default=2000,
        help="Number of matrices to generate (default: 2000).",
    )
    parser.add_argument(
        "--num-products",
        type=int,
        default=25,
        help="Number of random products to test (default: 25).",
    )
    parser.add_argument(
        "--match-tol",
        type=float,
        default=1e-6,
        help="Tolerance for matching products to precomputed matrices.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--plotly-output",
        type=Path,
        default=Path("data/weyl_products_3d.html"),
        help="Output HTML path for the interactive Weyl chamber plot.",
    )
    parser.add_argument(
        "--plotly-renderer",
        default="browser",
        help="Plotly renderer name (default: browser).",
    )
    parser.add_argument(
        "--output-products",
        type=Path,
        default=Path("data/weyl_products.csv"),
        help="Output CSV path for multiplied Weyl components.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    addable_cells = 4
    diagrams = find_yds_with_fixed_addable_cells(addable_cells, args.max_diagram_size)
    if not diagrams:
        raise SystemExit("No diagrams found for the requested size bounds.")
    if len(diagrams) < args.num_matrices:
        print(f"Warning: only {len(diagrams)} diagrams available; using all of them.")
    diagrams = diagrams[: args.num_matrices]

    matrices: list[np.ndarray] = []
    weyls: list[tuple[float, float, float]] = []
    for diagram in diagrams:
        matrix = A_matrix(diagram)
        matrices.append(matrix)
        a, b, c = _weyl_coefficients(matrix)
        assert np.isclose(a, np.pi / 4)
        weyls.append((a, b, c))

    product_entries: list[dict[str, float | int | str]] = []
    product_weyl_abc: list[tuple[float, float, float]] = []

    for idx in range(args.num_products):
        i, j = random.sample(range(len(matrices)), 2)
        product = matrices[i] @ matrices[j]
        prod_weyl = _weyl_coefficients(product)

        nearest_idx = None
        nearest_dist = None
        for k, matrix in enumerate(matrices):
            dist = _unitary_distance(product, matrix)
            if nearest_dist is None or dist < nearest_dist:
                nearest_dist = dist
                nearest_idx = k
        match = nearest_dist is not None and nearest_dist <= args.match_tol

        product_entries.append(
            {
                "index": idx,
                "left": i,
                "right": j,
                "weyl_a": prod_weyl[0],
                "weyl_b": prod_weyl[1],
                "weyl_c": prod_weyl[2],
                "nearest_index": nearest_idx if nearest_idx is not None else -1,
                "nearest_distance": nearest_dist if nearest_dist is not None else float("nan"),
                "match": match,
            }
        )
        product_weyl_abc.append(prod_weyl)
        print(
            f"product {idx}: a={prod_weyl[0]:.6f} b={prod_weyl[1]:.6f} c={prod_weyl[2]:.6f}"
        )

    args.output_products.parent.mkdir(parents=True, exist_ok=True)
    with args.output_products.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "index",
                "left",
                "right",
                "weyl_a",
                "weyl_b",
                "weyl_c",
                "nearest_index",
                "nearest_distance",
                "match",
            ],
        )
        writer.writeheader()
        writer.writerows(product_entries)

    pio.renderers.default = args.plotly_renderer
    _plot_weyl_chamber_3d(
        weyls,
        product_weyl_abc,
        args.plotly_output,
    )


def _plot_weyl_chamber_3d(
    base_weyls: list[tuple[float, float, float]],
    product_weyls: list[tuple[float, float, float]],
    output: Path,
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

    fig.add_trace(
        go.Scatter3d(
            x=[w[0] for w in base_weyls],
            y=[w[1] for w in base_weyls],
            z=[w[2] for w in base_weyls],
            mode="markers",
            marker=dict(size=3, opacity=0.4, color="rgba(0,0,150,0.5)"),
            name="fourier matrices",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[w[0] for w in product_weyls],
            y=[w[1] for w in product_weyls],
            z=[w[2] for w in product_weyls],
            mode="markers",
            marker=dict(size=5, color="red", opacity=0.4),
            name="multiplied matrices",
        )
    )

    fig.update_layout(
        title="Weyl Chamber: Precomputed vs Multiplied",
        scene=dict(xaxis_title="a", yaxis_title="b", zaxis_title="c", aspectmode="cube"),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output)
    fig.show()


if __name__ == "__main__":
    main()
