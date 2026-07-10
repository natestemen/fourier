#!/usr/bin/env python3
"""Weyl-chamber scan of the 4-addable A-matrix family.

Question: which two-qubit KAK invariants (a, b, c) do the 4-addable
A-matrices realize?  Supports report.md Finding 1: every 4-addable A-matrix
lies on the maximally-entangling face a = π/4 of the Weyl chamber, while
(b, c) vary continuously over the family.

Expected result: max |a − π/4| < 1e-8 across all enumerated diagrams, with
the (b, c) points filling a region of the a = π/4 face.

Migrated from generate_weyl.py + plot_weyl.py.  Behaviour changes: the
scatter is colored by diagram size (the old plot colored by a partition
distance to (3, 2, 1)), the interactive hover labels and plt.show() are
gone, and the CSV always goes to data/weyl_scan.csv.
"""

import argparse
import csv
from itertools import islice
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from fourier import a_matrix, diagrams_with_addable_cells, weyl_coordinates

SURFACE = "#fcfcfb"
# Single-hue sequential ramp (light → dark) for the size coloring.
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-size",
        type=int,
        default=25,
        help="Upper bound on the Young diagram sizes to enumerate.",
    )
    parser.add_argument(
        "--max-diagrams",
        type=int,
        default=None,
        help="Optional cap on the number of diagrams to process.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory for the CSV; the plot goes to <output-dir>/plots.",
    )
    return parser.parse_args()


def plot_bc_scatter(rows: list[dict], destination: Path) -> None:
    """Scatter of (b, |c|) colored by diagram size, with the a = π/4 face
    of the Weyl chamber drawn as a reference triangle."""
    b = np.array([row["weyl_b"] for row in rows])
    c_abs = np.abs([row["weyl_c"] for row in rows])
    sizes = np.array([row["size"] for row in rows])

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    pi4 = np.pi / 4
    ax.plot([0, pi4, pi4, 0], [0, 0, pi4, 0], color="#898781", linewidth=1)

    sc = ax.scatter(b, c_abs, s=12, alpha=0.6, c=sizes, cmap=SEQ_BLUE, linewidths=0)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("diagram size")

    ax.set_xlabel("b")
    ax.set_ylabel("|c|")
    ax.set_title("Weyl (b, c) of 4-addable A-matrices (a = π/4)")
    ax.grid(True, color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)

    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    diagrams = diagrams_with_addable_cells(4, args.max_size)
    if args.max_diagrams is not None:
        diagrams = islice(diagrams, args.max_diagrams)

    rows = []
    for diagram in diagrams:
        a, b, c = weyl_coordinates(a_matrix(diagram))
        rows.append(
            {
                "diagram": str(diagram.partition),
                "size": diagram.size,
                "weyl_a": a,
                "weyl_b": b,
                "weyl_c": c,
            }
        )
    if not rows:
        raise SystemExit("No diagrams matched the requested configuration.")

    csv_path = args.output_dir / "weyl_scan.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} Weyl rows to {csv_path}.")

    plot_path = args.output_dir / "plots" / "weyl_scan.png"
    plot_bc_scatter(rows, plot_path)
    print(f"Saved (b, |c|) scatter to {plot_path}.")

    max_dev = max(abs(row["weyl_a"] - np.pi / 4) for row in rows)
    verdict = "PASS" if max_dev < 1e-8 else "FAIL"
    print(f"Finding 1 check: max |a − π/4| = {max_dev:.2e}  [{verdict}]")


if __name__ == "__main__":
    main()
