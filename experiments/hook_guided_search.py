"""Hook-guided beam search for full k→2 Givens-reduction paths of A(λ).

Question: do hook-length ratios h₁:h₂ of λ predict which 2×2 rectangle
A-matrix the full (k−2)-step reduction chain of A(λ) can reach?  Candidate
targets are rectangles [q^p] with p:q a reduced ratio of two hook lengths;
a beam search over (pivot_row, target_col) reduction paths, scored by
column-alignment, looks for chains that land exactly on each target.

Supports report.md, "The Core Unsolved Problem: Recursive Decomposition"
(the "Hook-guided beam search" paragraph).

Expected result: inconsistent — some diagrams yield paths, most do not, and
no combinatorial rule ties successes to the hook lengths; a bare run (one
random 4-addable diagram) typically finds no path.

Found paths are appended to data/hook_guided_search.csv; the first path per
target is rendered to data/plots/ (disable with --no-plot).
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from yungdiagram import YoungDiagram

from fourier.amatrix import a_matrix
from fourier.diagrams import diagrams_with_addable_cells

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PLOT_DIR = DATA_DIR / "plots"


# ── Givens reduction step ──────────────────────────────────────────────────────


def _zero_column(
    M: np.ndarray, col: int, pivot_row: int, tol: float
) -> tuple[np.ndarray, bool]:
    """Left-multiply M by Givens rotations zeroing `col` except at `pivot_row`."""
    M = M.copy()
    for i in range(M.shape[0] - 1, -1, -1):
        if i == pivot_row:
            continue
        a, b = M[pivot_row, col], M[i, col]
        if abs(b) <= tol:
            continue
        r = math.hypot(a, b)
        c, s = a / r, b / r
        new_pivot = c * M[pivot_row, :] + s * M[i, :]
        new_i = -s * M[pivot_row, :] + c * M[i, :]
        M[pivot_row, :] = new_pivot
        M[i, :] = new_i
    success = all(abs(M[i, col]) <= tol for i in range(M.shape[0]) if i != pivot_row)
    return M, success


def reduce_step(M: np.ndarray, col: int, pivot_row: int, tol: float):
    """Zero `col` (pivot = pivot_row), delete that row and column.

    Returns (A_prime, success)."""
    M_rot, success = _zero_column(M, col, pivot_row, tol)
    return np.delete(np.delete(M_rot, pivot_row, axis=0), col, axis=1), success


# ── hook lengths and target enumeration ────────────────────────────────────────


def hook_lengths(partition: list[int]) -> list[int]:
    """All hook lengths of the diagram, ascending."""
    if not partition:
        return []
    conj = [sum(1 for r in partition if r > c) for c in range(partition[0])]
    hooks = []
    for i, row_len in enumerate(partition):
        for j in range(row_len):
            hooks.append((row_len - j - 1) + (conj[j] - i - 1) + 1)
    return sorted(hooks)


def hook_targets(partition: list[int]) -> list[tuple[int, int, np.ndarray]]:
    """Candidate (p, q, A_2x2) rectangle targets from hook-length ratios.

    Each candidate is the coprime rectangle [q^p] with p:q = h₁:h₂ for some
    pair of hook lengths; every rectangle has exactly 2 addable cells, so its
    A-matrix is 2×2.  Deduplicated by reduced ratio.
    """
    hooks = sorted(set(hook_lengths(partition)))
    seen: set[tuple[int, int]] = set()
    targets: list[tuple[int, int, np.ndarray]] = []
    for h1 in hooks:
        for h2 in hooks:
            g = math.gcd(h1, h2)
            p, q = h1 // g, h2 // g
            if (p, q) in seen:
                continue
            seen.add((p, q))
            targets.append((p, q, a_matrix(YoungDiagram([q] * p))))
    return targets


# ── beam search ────────────────────────────────────────────────────────────────


def _col_alignment(M: np.ndarray) -> float:
    """Mean of max|entry| per column ∈ [1/√k, 1]; higher = columns more
    concentrated on one row = easier to reduce further."""
    return float(np.max(np.abs(M), axis=0).mean())


def beam_search(
    A: np.ndarray, target: np.ndarray, beam_width: int, tol: float
) -> list[list[tuple[int, int]]]:
    """Reduction paths A (k×k) → target (2×2) found by beam search.

    At each of the k−2 steps, all m² (pivot_row, col) reductions are generated
    for each beam state, scored by column-alignment, and the top beam_width
    kept.  Returns the successful paths (possibly empty)."""
    k = A.shape[0]
    if k == 2:
        return [[]] if np.allclose(A, target, atol=tol) else []
    if k < 2:
        return []

    beam: list[tuple[np.ndarray, list]] = [(A, [])]
    for _ in range(k - 2):
        candidates: list[tuple[float, np.ndarray, list]] = []
        for M, path in beam:
            m = M.shape[0]
            for col in range(m):
                for pivot_row in range(m):
                    A_prime, success = reduce_step(M, col, pivot_row, tol)
                    if not success:
                        continue
                    candidates.append(
                        (_col_alignment(A_prime), A_prime, path + [(pivot_row, col)])
                    )
        if not candidates:
            return []
        candidates.sort(key=lambda x: -x[0])
        beam = [(mat, pth) for _, mat, pth in candidates[:beam_width]]

    return [path for M, path in beam if np.allclose(M, target, atol=tol)]


# ── visualization ──────────────────────────────────────────────────────────────


def _annotate(ax, mat: np.ndarray) -> None:
    for (i, j), val in np.ndenumerate(mat):
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="white")


def save_path_figure(
    A_orig: np.ndarray,
    path: list[tuple[int, int]],
    target: np.ndarray,
    rect_label: str,
    diagram_label: str,
    start_k: int,
    tol: float,
    out_path: Path,
) -> None:
    """Render the reduction chain next to the target 2×2 A-matrix."""
    chain = [A_orig]
    M = A_orig
    for pivot_row, col in path:
        M, _ = reduce_step(M, col, pivot_row, tol)
        chain.append(M)

    n_panels = len(chain) + 1
    fig, axes = plt.subplots(1, n_panels, figsize=(3.5 * n_panels, 4), constrained_layout=True)
    for i, (ax, mat) in enumerate(zip(axes[:-1], chain)):
        ax.matshow(mat, cmap="viridis")
        _annotate(ax, mat)
        if i == 0:
            title = f"A  ({start_k}-add)\n{diagram_label}"
        else:
            r, c = path[i - 1]
            title = f"step {i}\n(row={r}, col={c})"
        ax.set_title(title, fontsize=8)
        ax.set_xticks(range(mat.shape[1]))
        ax.set_yticks(range(mat.shape[0]))

    axes[-1].matshow(target, cmap="viridis")
    _annotate(axes[-1], target)
    axes[-1].set_title(f"target\n{rect_label}", fontsize=8)
    axes[-1].set_xticks([0, 1])
    axes[-1].set_yticks([0, 1])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ── reporting ──────────────────────────────────────────────────────────────────


def _path_str(path: list[tuple[int, int]]) -> str:
    return " -> ".join(f"(row={r},col={c})" for r, c in path)


def report_diagram(
    A: np.ndarray,
    diagram: YoungDiagram,
    start_k: int,
    results: list[tuple[int, int, np.ndarray, list[list]]],
    plot: bool,
    tol: float,
    csv_writer,
) -> int:
    """Print found paths for one diagram; return how many were found."""
    partition = list(diagram.partition)
    label = str(diagram.partition)
    hooks = hook_lengths(partition)
    hook_set = set(hooks)

    print(f"\n{'=' * 62}")
    print(f"Diagram: {label}  ({start_k}-addable)")
    print(f"  hooks:   {hooks}")
    print(f"  |λ|={diagram.size},  f^λ={diagram.number_of_standard_tableaux()}")
    print(f"  Reduction: {start_k}×{start_k} → 2×2  ({start_k - 2} steps)")
    print(f"{'=' * 62}")

    found = 0
    for p, q, A_target, paths in results:
        if not paths:
            continue
        rect_label = f"[{q}^{p}]  ({p}×{q})"
        in_hooks = [s for n, s in [(p, f"rows={p}✓"), (q, f"cols={q}✓")] if n in hook_set]
        hook_note = ("  hook: " + ", ".join(in_hooks)) if in_hooks else ""

        print(f"\n  Target {rect_label}  ratio {p}:{q}{hook_note}  ({len(paths)} path(s))")
        for path in paths:
            found += 1
            print(f"    {_path_str(path)}")
            csv_writer.writerow([label, p, q, _path_str(path)])
        if plot:
            out = PLOT_DIR / f"hook_guided_{'-'.join(map(str, partition))}_{p}x{q}.png"
            save_path_figure(A, paths[0], A_target, rect_label, label, start_k, tol, out)
            print(f"    saved {out}")

    if found == 0:
        print("  (no paths found — try increasing --beam)")
    return found


# ── CLI ────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--addable", type=int, default=4,
                   help="Addable cells of the starting diagram (default: 4).")
    p.add_argument("--max-size", type=int, default=30,
                   help="Max diagram size to search (default: 30).")
    p.add_argument("--beam", type=int, default=200,
                   help="Beam width: larger = more complete but slower (default: 200).")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed; 0 = random (default: 0).")
    p.add_argument("--index", type=int, default=None,
                   help="Index into starting diagram list (instead of random).")
    p.add_argument("--all", action="store_true",
                   help="Run on every starting diagram.")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip saving path figures.")
    p.add_argument("--tol", type=float, default=1e-8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(None if args.seed == 0 else args.seed)
    start_k = args.addable

    print(f"Loading {start_k}-addable diagrams (max_size={args.max_size}) …")
    diagrams = list(diagrams_with_addable_cells(start_k, args.max_size))
    if not diagrams:
        raise SystemExit(f"No {start_k}-addable diagrams found.")

    if args.all:
        selected = diagrams
    elif args.index is not None:
        if not (0 <= args.index < len(diagrams)):
            raise SystemExit(f"--index out of range (0..{len(diagrams) - 1})")
        selected = [diagrams[args.index]]
    else:
        selected = [rng.choice(diagrams)]

    print(f"Found {len(diagrams)} diagrams; running on {len(selected)}  "
          f"(beam={args.beam}, depth={start_k - 2})\n")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / "hook_guided_search.csv"

    total_found = 0
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["diagram", "target_p", "target_q", "path"])
        for diagram in selected:
            A = a_matrix(diagram)
            targets = hook_targets(list(diagram.partition))
            print(f"{diagram.partition}: {len(targets)} hook-derived target(s) …", flush=True)

            results = [
                (p, q, A_target, beam_search(A, A_target, args.beam, args.tol))
                for p, q, A_target in targets
            ]
            total_found += report_diagram(
                A, diagram, start_k, results, not args.no_plot, args.tol, writer
            )

    print(f"\n{'=' * 62}")
    print(f"SUMMARY: {total_found} path(s) found over {len(selected)} diagram(s).")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
