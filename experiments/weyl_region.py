#!/usr/bin/env python3
"""Render the Weyl chamber as a 3D region: π/4 ≥ a ≥ b ≥ |c|.

Question: what does the space of two-qubit gate equivalence classes look
like, and where is the a = π/4 face on which every 4-addable A-matrix lives?

Supports report.md Finding 1 ("Weyl Chamber — All 2-Qubit A-Matrices Have
a = π/4") as its illustration: the blue cap drawn at a = π/4 is exactly the
maximally-entangling face containing CNOT, CZ, and all 4-addable A-matrices.

Expected result: a PNG of the chamber wedge with its three boundary surfaces
(a = π/4 cap, b = a face, |c| = b cone) written to data/plots/weyl_region.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

PI4 = np.pi / 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--volume-grid",
        type=int,
        default=40,
        help="Grid points per axis for the interior scatter fill.",
    )
    parser.add_argument(
        "--surface-grid",
        type=int,
        default=50,
        help="Grid points per axis for the boundary surfaces.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/plots/weyl_region.png")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Interior fill: scatter of points satisfying |c| <= b <= a <= pi/4.
    N = args.volume_grid
    xs = np.linspace(0, PI4, N)
    ys = np.linspace(0, PI4, N)
    zs = np.linspace(-PI4, PI4, N)
    X, Y, Z = np.meshgrid(xs, ys, zs)
    mask = (X <= PI4) & (Y <= X) & (np.abs(Z) <= Y)
    ax.scatter(
        X[mask], Y[mask], Z[mask], c=X[mask], cmap="plasma", alpha=0.04, s=6, linewidths=0
    )

    # Boundary surfaces.
    alpha_surf = 0.25
    S = args.surface_grid

    # Surface 1: a = pi/4 (cap), for 0 <= b <= pi/4, |c| <= b.
    y1 = np.linspace(0, PI4, S)
    z1 = np.linspace(-PI4, PI4, S)
    Y1, Z1 = np.meshgrid(y1, z1)
    X1 = np.full_like(Y1, PI4)
    X1[~(np.abs(Z1) <= Y1)] = np.nan
    ax.plot_surface(X1, Y1, Z1, color="royalblue", alpha=alpha_surf)

    # Surface 2: b = a (wedge face), for 0 <= a <= pi/4, |c| <= a.
    x2 = np.linspace(0, PI4, S)
    z2 = np.linspace(-PI4, PI4, S)
    X2, Z2 = np.meshgrid(x2, z2)
    Y2 = X2.copy()
    mask2 = np.abs(Z2) <= Y2
    X2[~mask2] = np.nan
    Y2[~mask2] = np.nan
    Z2[~mask2] = np.nan
    ax.plot_surface(X2, Y2, Z2, color="tomato", alpha=alpha_surf)

    # Surfaces 3 & 4: c = +b and c = -b (cone faces), for 0 <= b <= a <= pi/4.
    x3 = np.linspace(0, PI4, S)
    y3 = np.linspace(0, PI4, S)
    X3, Y3 = np.meshgrid(x3, y3)
    Z3 = Y3.copy()
    mask3 = Y3 <= X3
    X3[~mask3] = np.nan
    Y3[~mask3] = np.nan
    Z3[~mask3] = np.nan
    ax.plot_surface(X3, Y3, Z3, color="mediumseagreen", alpha=alpha_surf)
    Z3n = -Y3.copy()
    Z3n[~mask3] = np.nan
    ax.plot_surface(X3, Y3, Z3n, color="mediumseagreen", alpha=alpha_surf)

    # Edges / ridge lines.
    t = np.linspace(0, PI4, 200)
    ax.plot([0, PI4], [0, 0], [0, 0], "k-", lw=1.5, alpha=0.7)  # spine c=b=0
    ax.plot(t, t, t, "k-", lw=1.5, alpha=0.7)  # upper ridge c=b=a
    ax.plot(t, t, -t, "k-", lw=1.5, alpha=0.7)  # lower ridge c=-b=-a
    ax.plot([PI4] * len(t), t, t, "k-", lw=1.2, alpha=0.5)  # cap edge, c=b
    ax.plot([PI4] * len(t), t, -t, "k-", lw=1.2, alpha=0.5)  # cap edge, c=-b
    ax.plot(t, t, np.zeros_like(t), "k--", lw=1, alpha=0.4)  # b=a, c=0

    ax.set_xlabel("x", fontsize=13)
    ax.set_ylabel("y", fontsize=13)
    ax.set_zlabel("z", fontsize=13)
    ax.set_title(r"Region: $\frac{\pi}{4} \geq x \geq y \geq |z|$", fontsize=15)

    ax.set_xticks([0, PI4])
    ax.set_xticklabels(["0", "π/4"])
    ax.set_yticks([0, PI4])
    ax.set_yticklabels(["0", "π/4"])
    ax.set_zticks([-PI4, 0, PI4])
    ax.set_zticklabels(["-π/4", "0", "π/4"])
    ax.set_xlim(0, PI4)
    ax.set_ylim(0, PI4)
    ax.set_zlim(-PI4, PI4)
    ax.set_box_aspect([1, 1, 2])  # x,y span pi/4; z spans pi/2

    legend_elements = [
        Patch(facecolor="royalblue", alpha=0.5, label=r"$x = \pi/4$"),
        Patch(facecolor="tomato", alpha=0.5, label=r"$y = x$"),
        Patch(facecolor="mediumseagreen", alpha=0.5, label=r"$|z| = y$"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=11)
    ax.view_init(elev=20, azim=-50)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
