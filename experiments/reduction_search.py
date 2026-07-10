"""Exhaustive Givens-reduction search: does A(λ) reduce to a smaller A-matrix?

Question: for every (pivot_row, target_col) pair, zero target_col of A(λ)
onto pivot_row with Givens rotations, delete that row and column, and compare
the (k−1)×(k−1) remainder against the A-matrix of every (k−1)-addable diagram
in a bounded catalog.  Recurse to --depth.  This is the definitive negative
search behind the conjecture that the A-matrix family is closed under
removing one addable cell.

Supports report.md, "The Core Unsolved Problem: Recursive Decomposition"
(the "Exhaustive search" paragraph).

Expected result: occasional matches at depths 1 and 2 for specific diagrams,
but no systematic pattern; a bare run (one random 4-addable diagram,
depth 2) usually reports no matches at any depth.

Results are appended to data/reduction_search.csv; matched reduction chains
are rendered to data/plots/ (disable with --no-plot).
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


# ── hook lengths (report annotations only) ─────────────────────────────────────


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


def _rect_note(partition: list[int], hooks: list[int]) -> str:
    """Annotate a matched rectangle [q^p] with its p:q ratio and hook hits."""
    if len(set(partition)) != 1:
        return ""
    p, q = len(partition), partition[0]
    g = math.gcd(p, q)
    hook_set = set(hooks)
    in_hooks = [s for n, s in [(p, f"rows={p}✓"), (q, f"cols={q}✓")] if n in hook_set]
    hook_note = "  hook match: " + ", ".join(in_hooks) if in_hooks else ""
    return f"  ({p}×{q}, ratio {p // g}:{q // g}){hook_note}"


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
    """Zero `col` (pivot = pivot_row), then delete that row and column.

    Returns (A_prime, success, is_orthogonal)."""
    M_rot, success = _zero_column(M, col, pivot_row, tol)
    A_prime = np.delete(np.delete(M_rot, pivot_row, axis=0), col, axis=1)
    n = A_prime.shape[0]
    orth_ok = n == 0 or bool(np.allclose(A_prime.T @ A_prime, np.eye(n), atol=tol))
    return A_prime, success, orth_ok


# ── catalog and matching ───────────────────────────────────────────────────────

CatalogEntry = tuple[list[int], np.ndarray]  # (partition, A-matrix)


def build_catalog(min_k: int, max_k: int, max_size: int) -> dict[int, list[CatalogEntry]]:
    """{addable_count: [(partition, A)]} for counts in [min_k, max_k]."""
    catalog: dict[int, list[CatalogEntry]] = {}
    for k in range(max(1, min_k), max_k + 1):
        entries = [
            (list(d.partition), a_matrix(d))
            for d in diagrams_with_addable_cells(k, max_size)
        ]
        if entries:
            catalog[k] = entries
    return catalog


def find_matches(
    A_prime: np.ndarray,
    target_k: int,
    catalog: dict[int, list[CatalogEntry]],
    tol: float,
) -> list[tuple[list[int], float, np.ndarray]]:
    """(partition, op_norm, A) for every catalog A-matrix within tol of A_prime."""
    matches = []
    for partition, Ad in catalog.get(target_k, []):
        if Ad.shape != A_prime.shape:
            continue
        op_norm = float(np.linalg.norm(Ad - A_prime, 2))
        if op_norm <= tol:
            matches.append((partition, op_norm, Ad))
    return matches


# ── exhaustive recursive search ────────────────────────────────────────────────


def search(
    M: np.ndarray,
    current_k: int,
    catalog: dict[int, list[CatalogEntry]],
    tol: float,
    max_depth: int,
    depth: int = 0,
    path: list[tuple[int, int]] | None = None,
) -> list[dict]:
    """Try all (pivot_row, col) at this depth, record matches, and recurse."""
    if path is None:
        path = []
    if depth >= max_depth or current_k < 1:
        return []

    n = M.shape[0]
    results = []
    for col in range(n):
        for pivot_row in range(n):
            A_prime, success, orth_ok = reduce_step(M, col, pivot_row, tol)
            if not success:
                continue

            step_path = path + [(pivot_row, col)]
            target_k = current_k - 1
            results.append(
                {
                    "depth": depth + 1,
                    "path": step_path,
                    "target_k": target_k,
                    "orth_ok": orth_ok,
                    "matches": find_matches(A_prime, target_k, catalog, tol),
                }
            )
            if depth + 1 < max_depth and target_k >= 1:
                results.extend(
                    search(A_prime, target_k, catalog, tol, max_depth, depth + 1, step_path)
                )
    return results


# ── visualization ──────────────────────────────────────────────────────────────


def _annotate(ax, mat: np.ndarray) -> None:
    for (i, j), val in np.ndenumerate(mat):
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="white")


def _replay_path(A_orig: np.ndarray, path: list, tol: float) -> list[np.ndarray]:
    chain = [A_orig]
    M = A_orig
    for pivot_row, col in path:
        M, _, _ = reduce_step(M, col, pivot_row, tol)
        chain.append(M)
    return chain


def save_match_figure(
    A_orig: np.ndarray,
    path: list,
    matches: list[tuple[list[int], float, np.ndarray]],
    diagram_label: str,
    start_k: int,
    tol: float,
    out_path: Path,
) -> None:
    """Render the reduction chain and its matching diagrams side by side."""
    chain = _replay_path(A_orig, path, tol)

    panels = [chain[0]]
    titles = [f"A  ({start_k}-addable)\n{diagram_label}"]
    for i, mat in enumerate(chain[1:]):
        pr, c = path[i]
        panels.append(mat)
        titles.append(f"A' step {i + 1}\n(row={pr}, col={c})")
    for partition, op_norm, A_match in matches:
        panels.append(A_match)
        titles.append(f"match: {partition}\nop_norm={op_norm:.1e}")

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4), constrained_layout=True)
    if n == 1:
        axes = [axes]
    for ax, mat, title in zip(axes, panels, titles):
        ax.matshow(mat, cmap="viridis")
        _annotate(ax, mat)
        ax.set_title(title, fontsize=8)
        ax.set_xticks(range(mat.shape[1]))
        ax.set_yticks(range(mat.shape[0]))
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ── reporting ──────────────────────────────────────────────────────────────────


def _path_str(path: list) -> str:
    return " -> ".join(f"(row={r},col={c})" for r, c in path)


def report(
    A_orig: np.ndarray,
    diagram: YoungDiagram,
    start_k: int,
    results: list[dict],
    max_depth: int,
    tol: float,
    plot: bool,
    csv_writer,
) -> dict[int, int]:
    """Print per-depth summary, save match figures, write CSV rows.

    Returns {depth: number of matched reduction paths}."""
    partition = list(diagram.partition)
    label = str(diagram.partition)
    hooks = hook_lengths(partition)

    print(f"\n{'=' * 62}")
    print(f"Diagram: {label}  ({start_k}-addable)")
    print(f"  hooks:   {hooks}")
    print(f"  |λ|={diagram.size},  f^λ={diagram.number_of_standard_tableaux()}")
    print(f"{'=' * 62}")

    by_depth: dict[int, list] = {}
    for r in results:
        by_depth.setdefault(r["depth"], []).append(r)

    match_counts = dict.fromkeys(range(1, max_depth + 1), 0)
    fig_index = 0
    for d in range(1, max_depth + 1):
        entries = by_depth.get(d, [])
        matched = [e for e in entries if e["matches"]]
        match_counts[d] = len(matched)
        target_k = start_k - d
        print(
            f"\n  Depth {d}  ({start_k}-addable -> {target_k}-addable):  "
            f"{len(entries)} successful reductions,  {len(matched)} with a matching diagram"
        )
        for e in matched:
            for m_partition, norm, _ in e["matches"]:
                note = _rect_note(m_partition, hooks)
                print(f"    {_path_str(e['path'])}  =>  {m_partition}  op_norm={norm:.1e}{note}")
                csv_writer.writerow(
                    [label, d, _path_str(e["path"]), str(m_partition), f"{norm:.3e}"]
                )
            if plot:
                out = PLOT_DIR / f"reduction_search_{'-'.join(map(str, partition))}_{fig_index}.png"
                save_match_figure(A_orig, e["path"], e["matches"], label, start_k, tol, out)
                print(f"    saved {out}")
                fig_index += 1

    if not any(match_counts.values()):
        print("\n  (no matches at any depth)")
    return match_counts


# ── CLI ────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--addable", type=int, default=4,
                   help="Addable cells of the starting diagram (default: 4).")
    p.add_argument("--max-size", type=int, default=30,
                   help="Max diagram size for start list and catalog (default: 30).")
    p.add_argument("--depth", type=int, default=2,
                   help="Max number of reduction steps (default: 2).")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed; 0 = random (default: 0).")
    p.add_argument("--index", type=int, default=None,
                   help="Index into starting diagram list (instead of random).")
    p.add_argument("--all", action="store_true",
                   help="Run over every starting diagram, not just one.")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip saving match figures (useful with --all).")
    p.add_argument("--tol", type=float, default=1e-8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(None if args.seed == 0 else args.seed)

    start_k = args.addable
    min_k_cat = max(1, start_k - args.depth)

    print(f"Building diagram catalog  (addable {min_k_cat}..{start_k - 1}, max_size={args.max_size}) …")
    catalog = build_catalog(min_k_cat, start_k - 1, args.max_size)

    diagrams = list(diagrams_with_addable_cells(start_k, args.max_size))
    if not diagrams:
        raise SystemExit(f"No {start_k}-addable diagrams found (try a larger --max-size).")
    print(f"Found {len(diagrams)} starting diagram(s) with {start_k} addable cells.")

    if args.all:
        selected = diagrams
    elif args.index is not None:
        if not (0 <= args.index < len(diagrams)):
            raise SystemExit(f"--index out of range (0..{len(diagrams) - 1})")
        selected = [diagrams[args.index]]
    else:
        selected = [rng.choice(diagrams)]

    print(f"Running exhaustive search on {len(selected)} diagram(s),  depth={args.depth} …")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / "reduction_search.csv"

    totals = dict.fromkeys(range(1, args.depth + 1), 0)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["diagram", "depth", "path", "match", "op_norm"])
        for diagram in selected:
            A = a_matrix(diagram)
            results = search(A, start_k, catalog, args.tol, args.depth)
            counts = report(A, diagram, start_k, results, args.depth, args.tol,
                            not args.no_plot, writer)
            for d, c in counts.items():
                totals[d] += c

    print(f"\n{'=' * 62}")
    print(f"SUMMARY over {len(selected)} diagram(s):")
    for d in range(1, args.depth + 1):
        print(f"  Depth {d}: {totals[d]} reduction path(s) matched a ({start_k - d})-addable diagram")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
