#!/usr/bin/env python3
"""Plot a grid of random 3-qubit (8-addable) A-matrix heatmaps.

Question: what do the entries of 8×8 A-matrices look like across different
diagrams — sign pattern, magnitude decay away from the constant column, and
the near-zero entries that the Cauchy factorization predicts?

Supports report.md Finding 3 (Cauchy structure): the heatmaps make the
diag(α)·C·diag(β) structure visible — entry magnitudes fall off with the
content gap c(aᵢ) − c(rⱼ).

Expected result: a PNG with up to 9 annotated heatmaps in a 3-column grid,
shared color scale, at data/plots/random_3qubit_matrices.png.  (Only 8
diagrams with 8 addable cells exist at the default --max-size 30, so the
default run plots all 8; raise --max-size for a genuine random sample.)

Behavior change vs. plot_random_3qubit_matrices.py: the old script ignored
--seed and always took the *first* `count` diagrams; this one actually
samples `count` diagrams at random (seed 0 = unseeded, as the old help text
promised).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fourier import a_matrix, diagrams_with_addable_cells


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-qubits", type=int, default=3, help="Number of qubits.")
    parser.add_argument(
        "--max-size", type=int, default=30, help="Max diagram size to search."
    )
    parser.add_argument(
        "--count", type=int, default=9, help="Number of diagrams to sample."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed (0 for unseeded)."
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/plots/random_3qubit_matrices.png")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    addable = 1 << args.num_qubits
    diagrams = list(diagrams_with_addable_cells(addable, args.max_size))
    if not diagrams:
        raise SystemExit(f"No diagrams found with {addable} addable cells.")
    if len(diagrams) < args.count:
        print(f"Warning: only {len(diagrams)} diagrams available; using all of them.")
        sampled = diagrams
    else:
        rng = random.Random(args.seed) if args.seed else random.Random()
        sampled = rng.sample(diagrams, args.count)

    matrices = [a_matrix(d) for d in sampled]
    labels = [str(d.partition) for d in sampled]

    all_vals = np.concatenate([m.flatten() for m in matrices])
    vmin = float(np.min(all_vals))
    vmax = float(np.max(all_vals))
    zero_thresh = max(1e-12, 1e-6 * max(abs(vmin), abs(vmax)))

    cols = 3
    rows = int(np.ceil(len(matrices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    axes = np.array(axes).reshape(rows, cols)

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="white")

    for idx, (mat, label) in enumerate(zip(matrices, labels)):
        ax = axes[idx // cols, idx % cols]
        mat_masked = np.ma.masked_where(np.abs(mat) <= zero_thresh, mat)
        im = ax.matshow(mat_masked, vmin=vmin, vmax=vmax, cmap=cmap)
        ax.text(
            0.98,
            0.98,
            label,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="black",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, linewidth=0),
        )
        for y in range(mat.shape[0]):
            for x in range(mat.shape[1]):
                ax.text(
                    x, y, f"{mat[y, x]:.2g}",
                    ha="center", va="center", fontsize=6, color="black",
                )
        ax.set_xticks([])
        ax.set_yticks([])

    for idx in range(len(matrices), rows * cols):
        axes[idx // cols, idx % cols].axis("off")

    # Square, evenly spaced grid; reserve space at right for the colorbar.
    fig.subplots_adjust(
        left=0.05, right=0.86, top=0.92, bottom=0.05, wspace=0.1, hspace=0.1
    )
    cax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cax)
    fig.suptitle(f"Random {args.num_qubits}-qubit A matrices")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
