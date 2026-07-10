#!/usr/bin/env python3
"""Map the region that fSim(θ, φ) sweeps out in the Weyl chamber.

fSim(θ, φ) = [[1,        0,        0,       0],
              [0,    cos θ,  -i sin θ,      0],
              [0, -i sin θ,     cos θ,      0],
              [0,        0,        0, e^{-iφ}]]

Question: which Weyl-chamber points does the fSim family reach?  This is
context for report.md Finding 1: all 4-addable A-matrices sit on the
a = π/4 face, and this sweep shows how a native two-parameter hardware gate
family covers the chamber (and that face) by comparison.

Expected result: a two-panel PNG (3D chamber and b-vs-|c| projection, colored
by θ) at data/plots/fsim_weyl.png; the fSim family fills a 2D sheet through
the chamber touching I, CNOT, and iSWAP.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fourier.weyl import weyl_coordinates

PI4 = np.pi / 4

# Weyl-chamber vertices and the tetrahedron edges connecting them.
WEYL_VERTS = {
    "I": (0, 0, 0),
    "CNOT": (PI4, 0, 0),
    "iSWAP": (PI4, PI4, 0),
    "SWAP": (PI4, PI4, PI4),
}
WEYL_EDGES = [
    ("I", "CNOT"),
    ("I", "iSWAP"),
    ("I", "SWAP"),
    ("CNOT", "iSWAP"),
    ("CNOT", "SWAP"),
    ("iSWAP", "SWAP"),
]


def fsim(theta: float, phi: float) -> np.ndarray:
    """4×4 fSim unitary in the computational basis {00, 01, 10, 11}."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [1, 0, 0, 0],
            [0, c, -1j * s, 0],
            [0, -1j * s, c, 0],
            [0, 0, 0, np.exp(-1j * phi)],
        ],
        dtype=complex,
    )


def draw_tetrahedron_3d(ax) -> None:
    for v0, v1 in WEYL_EDGES:
        p0, p1 = WEYL_VERTS[v0], WEYL_VERTS[v1]
        ax.plot(
            [p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
            color="black", lw=1, alpha=0.45,
        )


def draw_weyl_bc_boundary(ax) -> None:
    ax.plot([0, PI4], [0, 0], color="black", lw=1.2, alpha=0.7)
    ax.plot([0, PI4], [0, PI4], color="black", lw=1.2, alpha=0.7)
    ax.plot([PI4, PI4], [0, PI4], color="black", lw=1.2, alpha=0.7)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-theta", type=int, default=60, help="Grid points in θ ∈ [0, π/2]."
    )
    parser.add_argument(
        "--n-phi", type=int, default=80, help="Grid points in φ ∈ [0, 2π)."
    )
    parser.add_argument("--output", type=Path, default=Path("data/plots/fsim_weyl.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    thetas = np.linspace(0, np.pi / 2, args.n_theta)
    phis = np.linspace(0, 2 * np.pi, args.n_phi, endpoint=False)

    coords = np.array(
        [weyl_coordinates(fsim(theta, phi)) for theta in thetas for phi in phis]
    )
    a_flat, b_flat, c_flat = coords[:, 0], coords[:, 1], coords[:, 2]
    theta_color = np.repeat(thetas, args.n_phi)

    fig = plt.figure(figsize=(13, 5.5))
    ticks = [0, np.pi / 4, np.pi / 2]

    # 3D chamber view.
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    sc3 = ax3.scatter(
        a_flat, b_flat, c_flat,
        c=theta_color, cmap="plasma", s=5, alpha=0.35, linewidths=0,
    )
    draw_tetrahedron_3d(ax3)
    for name, (va, vb, vc) in WEYL_VERTS.items():
        ax3.scatter([va], [vb], [vc], color="black", s=40, zorder=5)
        ax3.text(va, vb, vc, f"  {name}", fontsize=7.5, zorder=6)
    ax3.set_xlabel("a", labelpad=4)
    ax3.set_ylabel("b", labelpad=4)
    ax3.set_zlabel("c", labelpad=4)
    ax3.set_title("FSim in Weyl chamber (3D)\ncolored by θ")
    cb3 = fig.colorbar(sc3, ax=ax3, fraction=0.03, pad=0.1, shrink=0.7)
    cb3.set_label("θ")
    cb3.set_ticks(ticks)
    cb3.set_ticklabels(["0", "π/4", "π/2"])

    # 2D projection: b vs |c|.
    ax2 = fig.add_subplot(1, 2, 2)
    sc2 = ax2.scatter(
        b_flat, np.abs(c_flat),
        c=theta_color, cmap="plasma", s=5, alpha=0.35, linewidths=0,
    )
    draw_weyl_bc_boundary(ax2)
    ax2.annotate("I", (0, 0), fontsize=8, ha="right")
    ax2.annotate("CNOT", (PI4, 0), fontsize=8, ha="left")
    ax2.annotate("iSWAP", (PI4, PI4), fontsize=8, ha="left")
    ax2.set_xlabel("b")
    ax2.set_ylabel("|c|")
    ax2.set_title("FSim Weyl projection (b vs |c|)\ncolored by θ")
    ax2.grid(True, alpha=0.3)
    cb2 = fig.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04)
    cb2.set_label("θ")
    cb2.set_ticks(ticks)
    cb2.set_ticklabels(["0", "π/4", "π/2"])

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
