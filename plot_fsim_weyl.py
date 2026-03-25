#!/usr/bin/env python3
"""Plot the region that FSim(θ, φ) sweeps out in the Weyl chamber.

FSim(θ, φ) = [[1,       0,          0,         0      ],
               [0,  cos θ,   -i sin θ,          0      ],
               [0, -i sin θ,   cos θ,            0      ],
               [0,       0,          0,   e^{-iφ}]]

θ ∈ [0, π/2] controls the XX+YY interaction strength.
φ ∈ [0, 2π]  controls the conditional ZZ phase on |11⟩.

Two subplots are produced:
  Left:  3D Weyl chamber (a, b, c) with the Weyl tetrahedron outline.
  Right: 2D projection (b vs |c|) with Weyl boundary.

Points are colored by θ.  Special gates are annotated.

Usage:
  python plot_fsim_weyl.py
  python plot_fsim_weyl.py --n-theta 100 --n-phi 120
  python plot_fsim_weyl.py --output my_plot.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)
from qiskit.synthesis import TwoQubitWeylDecomposition


# ---------------------------------------------------------------------------
# FSim gate
# ---------------------------------------------------------------------------

def fsim(theta: float, phi: float) -> np.ndarray:
    """4×4 FSim unitary in the computational basis {00, 01, 10, 11}."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [1,       0,       0,                0],
            [0,       c,  -1j*s,                0],
            [0,  -1j*s,       c,                0],
            [0,       0,       0, np.exp(-1j * phi)],
        ],
        dtype=complex,
    )


# ---------------------------------------------------------------------------
# Weyl coordinates
# ---------------------------------------------------------------------------

def weyl_coords(U: np.ndarray) -> tuple[float, float, float] | None:
    try:
        d = TwoQubitWeylDecomposition(U)
        return float(d.a), float(d.b), float(d.c)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Weyl chamber geometry helpers
# ---------------------------------------------------------------------------

_PI4 = np.pi / 4

# Vertices: Identity, CNOT, iSWAP, SWAP
_WEYL_VERTS = {
    "I":     (0,    0,    0   ),
    "CNOT":  (_PI4, 0,    0   ),
    "iSWAP": (_PI4, _PI4, 0   ),
    "SWAP":  (_PI4, _PI4, _PI4),
}

# All 6 edges of the tetrahedron
_WEYL_EDGES = [
    ("I", "CNOT"), ("I", "iSWAP"), ("I", "SWAP"),
    ("CNOT", "iSWAP"), ("CNOT", "SWAP"), ("iSWAP", "SWAP"),
]


def _draw_tetrahedron_3d(ax) -> None:
    for v0, v1 in _WEYL_EDGES:
        p0, p1 = _WEYL_VERTS[v0], _WEYL_VERTS[v1]
        ax.plot(
            [p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
            color="black", lw=1, alpha=0.45,
        )


def _draw_weyl_bc_boundary(ax) -> None:
    ax.plot([0, _PI4], [0,    0   ], color="black", lw=1.2, alpha=0.7)
    ax.plot([0, _PI4], [0,    _PI4], color="black", lw=1.2, alpha=0.7)
    ax.plot([_PI4, _PI4], [0, _PI4], color="black", lw=1.2, alpha=0.7)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--n-theta", type=int, default=60,
                   help="Grid points in θ ∈ [0, π/2]  (default 60)")
    p.add_argument("--n-phi",   type=int, default=80,
                   help="Grid points in φ ∈ [0, 2π)  (default 80)")
    p.add_argument("--output",  type=Path,
                   default=Path("data/plots/fsim_weyl.png"))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    thetas = np.linspace(0, np.pi / 2, args.n_theta)
    phis   = np.linspace(0, 2 * np.pi, args.n_phi, endpoint=False)

    a_arr = np.full((args.n_theta, args.n_phi), np.nan)
    b_arr = np.full_like(a_arr, np.nan)
    c_arr = np.full_like(a_arr, np.nan)

    for i, theta in enumerate(thetas):
        for j, phi in enumerate(phis):
            abc = weyl_coords(fsim(theta, phi))
            if abc is not None:
                a_arr[i, j], b_arr[i, j], c_arr[i, j] = abc

    mask = np.isfinite(a_arr)
    a_flat = a_arr[mask]
    b_flat = b_arr[mask]
    c_flat = c_arr[mask]
    theta_color = np.repeat(thetas[:, None], args.n_phi, axis=1)[mask]

    fig = plt.figure(figsize=(13, 5.5))

    # ---- 3D subplot -----------------------------------------------------------
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")

    sc3 = ax3.scatter(
        a_flat, b_flat, c_flat,
        c=theta_color, cmap="plasma",
        s=5, alpha=0.35, linewidths=0,
    )
    _draw_tetrahedron_3d(ax3)

    for name, (va, vb, vc) in _WEYL_VERTS.items():
        ax3.scatter([va], [vb], [vc], color="black", s=40, zorder=5)
        ax3.text(va, vb, vc, f"  {name}", fontsize=7.5, zorder=6)

    ax3.set_xlabel("a", labelpad=4)
    ax3.set_ylabel("b", labelpad=4)
    ax3.set_zlabel("c", labelpad=4)
    ax3.set_title("FSim in Weyl chamber (3D)\ncolored by θ")

    ticks = [0, np.pi / 4, np.pi / 2]
    cb3 = fig.colorbar(sc3, ax=ax3, fraction=0.03, pad=0.1, shrink=0.7)
    cb3.set_label("θ")
    cb3.set_ticks(ticks)
    cb3.set_ticklabels(["0", "π/4", "π/2"])

    # ---- 2D subplot -----------------------------------------------------------
    ax2 = fig.add_subplot(1, 2, 2)

    sc2 = ax2.scatter(
        b_flat, np.abs(c_flat),
        c=theta_color, cmap="plasma",
        s=5, alpha=0.35, linewidths=0,
    )
    _draw_weyl_bc_boundary(ax2)

    # Annotate the corners that appear in this projection
    ax2.annotate("I",     (0,    0   ), fontsize=8, ha="right")
    ax2.annotate("CNOT",  (_PI4, 0   ), fontsize=8, ha="left")
    ax2.annotate("iSWAP", (_PI4, _PI4), fontsize=8, ha="left")

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
    plt.show()


if __name__ == "__main__":
    main()
