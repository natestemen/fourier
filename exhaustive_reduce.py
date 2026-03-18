#!/usr/bin/env python3
"""
Exhaustively apply Givens column-zeroing reductions to a Young diagram's A-matrix.

For every (pivot_row, target_col) combination at each step, reduce the matrix by
zeroing target_col using Givens rotations on rows, then removing that row and column.
Check if the resulting A' matches the A-matrix of any (k-d)-addable Young diagram.
Recurse to depth --depth.

Usage:
    python3 exhaustive_reduce.py                      # random 4-addable diagram, depth 2
    python3 exhaustive_reduce.py --addable 5 --depth 3
    python3 exhaustive_reduce.py --all --addable 4    # every 4-addable diagram
    python3 exhaustive_reduce.py --index 2 --addable 4
"""
from __future__ import annotations

import argparse
import math
import random
from math import gcd

import numpy as np
import matplotlib.pyplot as plt

from compute_matrix import A_matrix
from helper import find_yds_with_fixed_addable_cells


# ── Hook length helpers ────────────────────────────────────────────────────────

def _hook_lengths(partition: list[int]) -> list[int]:
    """Return sorted (ascending) list of all hook lengths in the diagram."""
    if not partition:
        return []
    n_cols = partition[0]
    conj = [sum(1 for r in partition if r > c) for c in range(n_cols)]
    hooks = []
    for i, row_len in enumerate(partition):
        for j in range(row_len):
            hooks.append((row_len - j - 1) + (conj[j] - i - 1) + 1)
    return sorted(hooks)


def _parse_rect(label: str) -> tuple[int, int] | None:
    """Parse a partition label like '[7, 7, 7, 7]' into (n_rows, n_cols) if rectangular."""
    try:
        parts = [int(x) for x in label.strip("[]").split(",") if x.strip()]
        if not parts:
            return None
        if len(set(parts)) == 1:
            return len(parts), parts[0]
    except ValueError:
        pass
    return None


def _rect_note(n_rows: int, n_cols: int, hooks: list[int]) -> str:
    """Describe a matched rectangle's dimensions relative to hook lengths."""
    hook_set = set(hooks)
    g = gcd(n_rows, n_cols)
    ratio = f"{n_rows // g}:{n_cols // g}"
    in_hooks = []
    if n_rows in hook_set:
        in_hooks.append(f"rows={n_rows}✓")
    if n_cols in hook_set:
        in_hooks.append(f"cols={n_cols}✓")
    hook_note = "  hook match: " + ", ".join(in_hooks) if in_hooks else ""
    return f"  ({n_rows}×{n_cols}, ratio {ratio}){hook_note}"


# ── Givens helpers ─────────────────────────────────────────────────────────────

def _givens(a: float, b: float) -> tuple[float, float]:
    r = math.hypot(a, b)
    if r == 0.0:
        return 1.0, 0.0
    return a / r, b / r


def _zero_column(M: np.ndarray, col: int, pivot_row: int, tol: float) -> tuple[np.ndarray, bool]:
    """Left-multiply M by Givens rotations to zero column `col` everywhere except `pivot_row`."""
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
    """Zero `col` in M (pivot = pivot_row), then delete that row+col.
    Returns (M_rotated, A_prime, success, is_orthogonal)."""
    M_rot, success = _zero_column(M, col, pivot_row, tol)
    A_prime = np.delete(np.delete(M_rot, pivot_row, axis=0), col, axis=1)
    n = A_prime.shape[0]
    orth_ok = n == 0 or bool(np.allclose(A_prime.T @ A_prime, np.eye(n), atol=tol))
    return M_rot, A_prime, success, orth_ok


# ── Diagram catalog ────────────────────────────────────────────────────────────

def _build_catalog(min_k: int, max_k: int, max_size: int) -> dict[int, list]:
    """Return {addable_count: [diagrams]} for addable counts in [min_k, max_k]."""
    catalog: dict[int, list] = {}
    for k in range(max(1, min_k), max_k + 1):
        diags = list(find_yds_with_fixed_addable_cells(k, max_size))
        if diags:
            catalog[k] = diags
    return catalog


def _find_matches(A_prime: np.ndarray, target_k: int, catalog: dict, tol: float) -> list[tuple[str, float, np.ndarray]]:
    """Return list of (label, op_norm, A_match) for diagrams whose A-matrix matches A_prime."""
    if target_k not in catalog:
        return []
    matches = []
    for d in catalog[target_k]:
        Ad = np.array(A_matrix(d), dtype=float)
        if Ad.shape != A_prime.shape:
            continue
        op_norm = float(np.linalg.norm(Ad - A_prime, 2))
        if op_norm <= tol:
            matches.append((str(getattr(d, "partition", d)), op_norm, Ad))
    return matches


# ── Exhaustive recursive search ────────────────────────────────────────────────

def _search(
    M: np.ndarray,
    current_k: int,
    catalog: dict,
    tol: float,
    max_depth: int,
    depth: int = 0,
    path: list | None = None,
) -> list[dict]:
    """Try all (pivot_row, col) at this depth, check for matches, and recurse."""
    if path is None:
        path = []
    if depth >= max_depth or current_k < 1:
        return []

    n = M.shape[0]
    results = []

    for col in range(n):
        for pivot_row in range(n):
            _, A_prime, success, orth_ok = _reduce(M, col, pivot_row, tol)
            if not success:
                continue

            step_path = path + [(pivot_row, col)]
            target_k = current_k - 1
            matches = _find_matches(A_prime, target_k, catalog, tol)

            results.append({
                "depth":    depth + 1,
                "path":     step_path,
                "target_k": target_k,
                "A_prime":  A_prime,
                "orth_ok":  orth_ok,
                "matches":  matches,
            })

            # Recurse into A_prime
            if depth + 1 < max_depth and target_k >= 1:
                results.extend(
                    _search(A_prime, target_k, catalog, tol, max_depth, depth + 1, step_path)
                )

    return results


# ── Visualization ─────────────────────────────────────────────────────────────

def _annotate(ax: plt.Axes, mat: np.ndarray) -> None:
    for (i, j), val in np.ndenumerate(mat):
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="white")


def _replay_path(A_orig: np.ndarray, path: list, tol: float) -> list[np.ndarray]:
    """Return [A_orig, A'_1, A'_2, ...] by replaying the (pivot_row, col) reduction path."""
    chain = [A_orig]
    M = A_orig
    for pivot_row, col in path:
        _, A_prime, _, _ = _reduce(M, col, pivot_row, tol)
        chain.append(A_prime)
        M = A_prime
    return chain


def _show_match(
    A_orig: np.ndarray,
    path: list,
    matches: list[tuple[str, float, np.ndarray]],
    diagram_label: str,
    start_k: int,
    tol: float,
) -> None:
    """One figure per matched path: reduction chain + all matching diagrams side by side."""
    chain = _replay_path(A_orig, path, tol)

    # panels: each step in the chain, then each matching diagram
    panels = []
    titles = []
    panels.append(chain[0])
    titles.append(f"A  ({start_k}-addable)\n{diagram_label}")
    for i, mat in enumerate(chain[1:]):
        pr, c = path[i]
        panels.append(mat)
        titles.append(f"A' step {i+1}\n(row={pr}, col={c})")
    for label, op_norm, A_match in matches:
        panels.append(A_match)
        titles.append(f"match: {label}\nop_norm={op_norm:.1e}")

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
    plt.show()


# ── Reporting ──────────────────────────────────────────────────────────────────

def _path_str(path: list) -> str:
    return " -> ".join(f"(row={r},col={c})" for r, c in path)


def _report(
    A_orig: np.ndarray,
    diagram_label: str,
    partition: list[int],
    start_k: int,
    results: list[dict],
    max_depth: int,
    tol: float,
    no_plot: bool,
) -> None:
    hooks = _hook_lengths(partition)
    n = sum(partition)
    f_lambda = math.factorial(n) // math.prod(hooks) if hooks else 1

    print(f"\n{'=' * 62}")
    print(f"Diagram: {diagram_label}  ({start_k}-addable)")
    print(f"  hooks:   {hooks}")
    print(f"  |λ|={sum(partition)},  f^λ={f_lambda}")
    print(f"{'=' * 62}")

    by_depth: dict[int, list] = {}
    for r in results:
        by_depth.setdefault(r["depth"], []).append(r)

    any_match_overall = False
    for d in range(1, max_depth + 1):
        entries = by_depth.get(d, [])
        total    = len(entries)
        matched  = [e for e in entries if e["matches"]]
        target_k = start_k - d
        print(f"\n  Depth {d}  ({start_k}-addable -> {target_k}-addable):  "
              f"{total} successful reductions,  {len(matched)} with a matching diagram")
        for e in matched:
            any_match_overall = True
            for label, norm, _ in e["matches"]:
                note = ""
                dims = _parse_rect(label)
                if dims:
                    note = _rect_note(dims[0], dims[1], hooks)
                print(f"    {_path_str(e['path'])}  =>  {label}  op_norm={norm:.1e}{note}")
            if not no_plot:
                _show_match(A_orig, e["path"], e["matches"], diagram_label, start_k, tol)

    if not any_match_overall:
        print("\n  (no matches at any depth)")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--addable",  type=int, default=4,
                   help="Addable cells of the starting diagram (default: 4).")
    p.add_argument("--max-size", type=int, default=30,
                   help="Max diagram size to search (default: 30).")
    p.add_argument("--depth",    type=int, default=2,
                   help="Max number of reduction steps (default: 2).")
    p.add_argument("--seed",     type=int, default=0,
                   help="Random seed; 0 = random (default: 0).")
    p.add_argument("--index",    type=int, default=None,
                   help="Index into starting diagram list (instead of random).")
    p.add_argument("--all",      action="store_true",
                   help="Run over every starting diagram, not just one.")
    p.add_argument("--no-plot",  action="store_true",
                   help="Skip matshow visualizations (useful with --all).")
    p.add_argument("--tol",      type=float, default=1e-8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng  = random.Random(None if args.seed == 0 else args.seed)

    start_k   = args.addable
    min_k_cat = max(1, start_k - args.depth)

    print(f"Building diagram catalog  (addable {min_k_cat}..{start_k}, max_size={args.max_size}) …")
    catalog = _build_catalog(min_k_cat, start_k, args.max_size)

    if start_k not in catalog:
        raise SystemExit(f"No {start_k}-addable diagrams found (try a larger --max-size).")

    diagrams = catalog[start_k]
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

    match_counts: dict[int, int] = {d: 0 for d in range(1, args.depth + 1)}
    for diagram in selected:
        label     = str(getattr(diagram, "partition", diagram))
        partition = list(getattr(diagram, "partition", diagram))
        A         = np.array(A_matrix(diagram), dtype=float)
        results   = _search(A, start_k, catalog, args.tol, args.depth)
        _report(A, label, partition, start_k, results, args.depth, args.tol, args.no_plot)
        for r in results:
            if r["matches"]:
                match_counts[r["depth"]] += 1

    if len(selected) > 1:
        print(f"\n{'=' * 62}")
        print(f"SUMMARY over {len(selected)} diagrams:")
        for d in range(1, args.depth + 1):
            print(f"  Depth {d}: {match_counts[d]} reduction path(s) matched a ({start_k - d})-addable diagram")


if __name__ == "__main__":
    main()
