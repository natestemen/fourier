#!/usr/bin/env python3
"""
Hook-guided A-matrix decomposition for Young diagrams.

Strategy:
  1. Compute hook lengths H of the starting k-addable diagram.
  2. Enumerate candidate 2×2 targets: rectangles [q^p] whose ratio p:q
     derives from a pair of hook lengths.
  3. For each target, find the full k→2 reduction path (k-2 Givens steps)
     using beam search.

Beam search is O(beam_width × k⁴) — practical for k up to ~20.
Exhaustive search is O(k^{2(k-2)}), intractable for k > ~6.

Usage:
    python3 hook_guided_reduce.py --addable 4
    python3 hook_guided_reduce.py --addable 8 --beam 100
    python3 hook_guided_reduce.py --all --addable 5 --no-plot
    python3 hook_guided_reduce.py --index 3 --addable 6
"""
from __future__ import annotations

import argparse
import math
import random

import numpy as np
import matplotlib.pyplot as plt

from yungdiagram import YoungDiagram
from compute_matrix import A_matrix
from helper import find_yds_with_fixed_addable_cells


# ── Givens helpers ─────────────────────────────────────────────────────────────

def _givens(a: float, b: float) -> tuple[float, float]:
    r = math.hypot(a, b)
    if r == 0.0:
        return 1.0, 0.0
    return a / r, b / r


def _zero_column(M: np.ndarray, col: int, pivot_row: int, tol: float) -> tuple[np.ndarray, bool]:
    M = M.copy()
    for i in range(M.shape[0] - 1, -1, -1):
        if i == pivot_row:
            continue
        a, b = M[pivot_row, col], M[i, col]
        if abs(b) <= tol:
            continue
        c, s = _givens(a, b)
        new_pivot = c * M[pivot_row, :] + s * M[i, :]
        new_i     = -s * M[pivot_row, :] + c * M[i, :]
        M[pivot_row, :] = new_pivot
        M[i, :]         = new_i
    success = all(abs(M[i, col]) <= tol for i in range(M.shape[0]) if i != pivot_row)
    return M, success


def _reduce(M: np.ndarray, col: int, pivot_row: int, tol: float):
    """Zero column col (pivot = pivot_row), delete row+col. Returns (M_rot, A', success, orth_ok)."""
    M_rot, success = _zero_column(M, col, pivot_row, tol)
    A_prime = np.delete(np.delete(M_rot, pivot_row, axis=0), col, axis=1)
    n = A_prime.shape[0]
    orth_ok = n == 0 or bool(np.allclose(A_prime.T @ A_prime, np.eye(n), atol=tol))
    return M_rot, A_prime, success, orth_ok


# ── Hook lengths ───────────────────────────────────────────────────────────────

def _hook_lengths(partition: list[int]) -> list[int]:
    if not partition:
        return []
    n_cols = partition[0]
    conj = [sum(1 for r in partition if r > c) for c in range(n_cols)]
    hooks = []
    for i, row_len in enumerate(partition):
        for j in range(row_len):
            hooks.append((row_len - j - 1) + (conj[j] - i - 1) + 1)
    return sorted(hooks)


# ── Hook-guided target enumeration ────────────────────────────────────────────

_rect_cache: dict[tuple[int, int], np.ndarray | None] = {}


def _rect_a_matrix(p: int, q: int) -> np.ndarray | None:
    """
    2×2 A-matrix for rectangle [q^p] (p rows of width q).
    Uses canonical (coprime) representative; cached.
    """
    g = math.gcd(p, q)
    key = (p // g, q // g)
    if key not in _rect_cache:
        rp, rq = key
        try:
            A = np.array(A_matrix(YoungDiagram([rq] * rp)), dtype=float)
            _rect_cache[key] = A if A.shape == (2, 2) else None
        except Exception:
            _rect_cache[key] = None
    return _rect_cache[key]


def _hook_targets(partition: list[int]) -> list[tuple[int, int, np.ndarray]]:
    """
    Enumerate candidate (p, q, A_2x2) targets derived from hook length pairs.
    Each candidate is a rectangle [q^p] whose ratio p:q = h1:h2 for some
    h1, h2 in the hook length set. Deduplicates by reduced ratio.
    Returns list of (p, q, A) using the canonical (coprime) representative.
    """
    hooks = sorted(set(_hook_lengths(partition)))
    seen: set[tuple[int, int]] = set()
    targets: list[tuple[int, int, np.ndarray]] = []

    for h1 in hooks:
        for h2 in hooks:
            g = math.gcd(h1, h2)
            p, q = h1 // g, h2 // g
            if (p, q) in seen:
                continue
            seen.add((p, q))
            A = _rect_a_matrix(p, q)
            if A is not None:
                targets.append((p, q, A))

    return targets


# ── Beam search ────────────────────────────────────────────────────────────────

def _col_alignment(M: np.ndarray) -> float:
    """
    Mean column-alignment score: average of max|entry| per column.
    Since A-matrix columns have unit norm, max|entry| ∈ [1/√k, 1].
    Higher = columns more concentrated on one row = easier to reduce further.
    """
    return float(np.max(np.abs(M), axis=0).mean())


def beam_search(
    A: np.ndarray,
    target: np.ndarray,
    beam_width: int,
    tol: float,
) -> list[list[tuple[int, int]]]:
    """
    Find reduction paths A (k×k) → target (2×2) using beam search.

    At each of the k-2 steps, all m² (pivot_row, col) reductions are generated
    for each beam state; they are scored by column-alignment and the top
    beam_width are kept for the next step.

    Returns a list of successful paths (may be empty if beam_width too small).
    Each path is a list of (pivot_row, col) tuples.
    """
    k = A.shape[0]
    if k == 2:
        return [[]] if np.allclose(A, target, atol=tol) else []
    if k < 2:
        return []

    beam: list[tuple[np.ndarray, list]] = [(A, [])]

    for _step in range(k - 2):
        candidates: list[tuple[float, np.ndarray, list]] = []
        for M, path in beam:
            m = M.shape[0]
            for col in range(m):
                for pivot_row in range(m):
                    _, A_prime, success, _ = _reduce(M, col, pivot_row, tol)
                    if not success:
                        continue
                    sc = _col_alignment(A_prime)
                    candidates.append((sc, A_prime, path + [(pivot_row, col)]))

        if not candidates:
            return []

        candidates.sort(key=lambda x: -x[0])
        beam = [(mat, pth) for _, mat, pth in candidates[:beam_width]]

    # Remaining items are all 2×2; check against target
    return [path for M, path in beam if np.allclose(M, target, atol=tol)]


# ── Visualization ──────────────────────────────────────────────────────────────

def _annotate(ax: plt.Axes, mat: np.ndarray) -> None:
    for (i, j), val in np.ndenumerate(mat):
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="white")


def _show_path(
    A_orig: np.ndarray,
    path: list[tuple[int, int]],
    target: np.ndarray,
    rect_label: str,
    diagram_label: str,
    start_k: int,
    tol: float,
) -> None:
    chain = [A_orig]
    M = A_orig
    for pivot_row, col in path:
        _, A_prime, _, _ = _reduce(M, col, pivot_row, tol)
        chain.append(A_prime)
        M = A_prime

    n_panels = len(chain) + 1
    fig, axes = plt.subplots(1, n_panels, figsize=(3.5 * n_panels, 4), constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

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
    plt.show()


# ── Reporting ──────────────────────────────────────────────────────────────────

def _path_str(path: list[tuple[int, int]]) -> str:
    return " -> ".join(f"(row={r},col={c})" for r, c in path)


def _report_diagram(
    A: np.ndarray,
    partition: list[int],
    diagram_label: str,
    start_k: int,
    results: list[tuple[int, int, np.ndarray, list[list]]],
    no_plot: bool,
    tol: float,
) -> int:
    hooks = _hook_lengths(partition)
    hook_set = set(hooks)
    n_boxes = sum(partition)
    f_lambda = math.factorial(n_boxes) // math.prod(hooks) if hooks else 1

    print(f"\n{'=' * 62}")
    print(f"Diagram: {diagram_label}  ({start_k}-addable)")
    print(f"  hooks:   {hooks}")
    print(f"  |λ|={n_boxes},  f^λ={f_lambda}")
    print(f"  Reduction: {start_k}×{start_k} → 2×2  ({start_k - 2} steps)")
    print(f"{'=' * 62}")

    found = 0
    for p, q, A_target, paths in results:
        if not paths:
            continue
        rect_label = f"[{q}^{p}]  ({p}×{q})"
        in_hooks = []
        if p in hook_set:
            in_hooks.append(f"rows={p}✓")
        if q in hook_set:
            in_hooks.append(f"cols={q}✓")
        hook_note = ("  hook: " + ", ".join(in_hooks)) if in_hooks else ""

        g = math.gcd(p, q)
        ratio_str = f"{p // g}:{q // g}"
        print(f"\n  Target {rect_label}  ratio {ratio_str}{hook_note}  ({len(paths)} path(s))")
        for path in paths:
            found += 1
            print(f"    {_path_str(path)}")
        if not no_plot:
            _show_path(A, paths[0], A_target, rect_label, diagram_label, start_k, tol)

    if found == 0:
        print("  (no paths found — try increasing --beam)")
    return found


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--addable",  type=int, default=4,
                   help="Addable cells of the starting diagram (default: 4).")
    p.add_argument("--max-size", type=int, default=30,
                   help="Max diagram size to search (default: 30).")
    p.add_argument("--beam",     type=int, default=200,
                   help="Beam width: larger = more complete but slower (default: 200).")
    p.add_argument("--seed",     type=int, default=0)
    p.add_argument("--index",    type=int, default=None)
    p.add_argument("--all",      action="store_true",
                   help="Run on every starting diagram.")
    p.add_argument("--no-plot",  action="store_true")
    p.add_argument("--tol",      type=float, default=1e-8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(None if args.seed == 0 else args.seed)
    start_k = args.addable

    print(f"Loading {start_k}-addable diagrams (max_size={args.max_size}) …")
    diagrams = list(find_yds_with_fixed_addable_cells(start_k, args.max_size))
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

    total_found = 0
    for diagram in selected:
        label     = str(getattr(diagram, "partition", diagram))
        partition = list(getattr(diagram, "partition", diagram))
        A         = np.array(A_matrix(diagram), dtype=float)

        targets = _hook_targets(partition)
        print(f"{label}: {len(targets)} hook-derived target(s) …", flush=True)

        results = []
        for p, q, A_target in targets:
            paths = beam_search(A, A_target, args.beam, args.tol)
            results.append((p, q, A_target, paths))

        total_found += _report_diagram(
            A, partition, label, start_k, results, args.no_plot, args.tol
        )

    if len(selected) > 1:
        print(f"\n{'=' * 62}")
        print(f"SUMMARY: {total_found} path(s) found over {len(selected)} diagrams.")


if __name__ == "__main__":
    main()
