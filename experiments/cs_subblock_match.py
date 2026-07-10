"""Are the CS-decomposition sub-blocks of A(λ) themselves A-matrices?

Question: CS-decompose A(λ) = (U₁ ⊕ U₂)·CS(θ)·(V₁ᵀ ⊕ V₂ᵀ) via
fourier.decompositions.cs_factor and test whether any of {U₁, U₂, V₁, V₂}
— and simple variants (negation, transpose, row permutations for small
blocks) — equal the A-matrix of some smaller diagram.  If they did, the
O(k log k) CS recursion of report.md Finding 3 would close.

Two extensions beyond the catalog:
- --angle-scan N: for 2×2 sub-blocks, compare the rotation angle against the
  single-row [n] (θ = arctan √n) and single-column [1^m] families up to N.
- --fit: minimize ‖M − A(ac, rc)‖_F over real-valued interlaced contents via
  fourier.amatrix.a_matrix_from_contents, testing whether a sub-block is an
  A-matrix for ANY content sequence, not just those in the catalog.  (The
  library builder uses descending-content order — the canonical convention —
  and reproduces a_matrix(λ) exactly on the integer contents of a diagram;
  verified for A(3,2,1).)

Supports report.md, "Open Directions — 1. Recursive decomposition via CS
sub-blocks" and the CS paragraph of Finding 3.

Expected result: no catalog matches; 2×2 fits succeed only with non-integer
contents (trivial — every 2×2 rotation is an A-matrix of some real contents).

Behavior change vs the old cs_subblock_test.py: odd k now works (the library
takes square diagonal blocks p = k//2 ≤ q; the old script's hand-rolled
reconstruction crashed for odd k).

Per-sub-block results are appended to data/cs_subblock_match.csv.
"""

from __future__ import annotations

import argparse
import csv
import random
from itertools import permutations
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from yungdiagram import YoungDiagram

from fourier.amatrix import a_matrix, a_matrix_from_contents
from fourier.decompositions import cs_factor
from fourier.diagrams import diagrams_with_addable_cells

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ── angle-based scan for 2×2 sub-blocks ────────────────────────────────────────


def rotation_angle(M: np.ndarray) -> float:
    """Rotation angle θ ∈ [0, π/2) of a 2×2 orthogonal matrix, modulo the
    negation/transpose variants (which flip or reflect the angle)."""
    for V in [M, -M, M.T, -M.T]:
        c, s = V[0, 0], V[1, 0]
        if abs(c**2 + s**2 - 1) < 1e-6 and c >= 0 and s >= 0:
            return float(np.arctan2(s, c))
    return float(np.arctan2(M[1, 0], M[0, 0])) % (np.pi / 2)


def single_row_angle(n: int) -> float:
    """Angle of A([n]): θ = arctan √n (contents n, −1 addable; n−1 removable)."""
    return float(np.arctan(np.sqrt(n)))


def single_col_angle(m: int) -> float:
    """Angle of A([1^m]): θ = arctan(1/√m), by conjugate symmetry with [n]."""
    return float(np.arctan(1.0 / np.sqrt(m)))


def angle_scan(angle: float, n_max: int, tol_deg: float = 0.001) -> list[tuple[str, float]]:
    """(label, delta_degrees) for [n] / [1^m] family members whose A-matrix
    angle matches `angle` within tol_deg, for n, m up to n_max."""
    hits: list[tuple[str, float]] = []
    tol_rad = np.radians(tol_deg)
    for n in range(1, n_max + 1):
        for theta, lbl in [(single_row_angle(n), f"[{n}]"), (single_col_angle(n), f"[1^{n}]")]:
            # Check θ and π/2 − θ (neg/transpose variants flip the angle).
            for candidate in [theta, np.pi / 2 - theta]:
                delta = abs(angle - candidate)
                delta = min(delta, np.pi / 2 - delta)
                if delta < tol_rad:
                    hits.append((lbl, float(np.degrees(delta))))
    return hits


# ── continuous content fit ─────────────────────────────────────────────────────


def _contents_from_params(params: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Decode params = [center, log_gap_0, …] into interlaced descending
    contents ac[0] > rc[0] > ac[1] > … > ac[k−1]."""
    vals = np.empty(2 * k - 1)
    vals[0] = params[0]
    gaps = np.exp(params[1:])
    for i in range(1, 2 * k - 1):
        vals[i] = vals[i - 1] - gaps[i - 1]
    return vals[0::2], vals[1::2]


def _fit_loss(params: np.ndarray, M: np.ndarray) -> float:
    ac, rc = _contents_from_params(params, M.shape[0])
    A_fit = a_matrix_from_contents(ac, rc)
    return float(min(np.sum((M - V) ** 2) for V in [A_fit, -A_fit, A_fit.T, -A_fit.T]))


def fit_subblock(
    M: np.ndarray, n_restarts: int = 30
) -> tuple[float, tuple[np.ndarray, np.ndarray]]:
    """Minimize ‖M − A(ac, rc)‖²_F over continuous interlaced contents.

    Returns (best_loss, (ac, rc)) over n_restarts random initializations."""
    k = M.shape[0]
    rng = np.random.default_rng(42)
    best_loss = float("inf")
    best_params = None

    for _ in range(n_restarts):
        x0 = np.concatenate(
            [[rng.uniform(-k, k)], rng.normal(0.0, 0.5, size=2 * k - 2)]
        )
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            res = minimize(_fit_loss, x0, args=(M,), method="L-BFGS-B",
                           options={"maxiter": 500, "ftol": 1e-14, "gtol": 1e-10})
        loss = float(res.fun) if np.isfinite(res.fun) else float("inf")
        if best_params is None or loss < best_loss:
            best_loss = loss
            best_params = res.x

    return best_loss, _contents_from_params(best_params, k)


# ── catalog and matching ───────────────────────────────────────────────────────


def build_catalog(ks: list[int], max_size: int) -> dict[int, list[tuple[str, np.ndarray]]]:
    """{k: [(label, A-matrix)]} for each target addable count."""
    catalog: dict[int, list[tuple[str, np.ndarray]]] = {}
    for k in ks:
        entries = [
            (str(d.partition), a_matrix(d))
            for d in diagrams_with_addable_cells(k, max_size)
        ]
        if entries:
            catalog[k] = entries
    return catalog


_VARIANT_NAMES = ["direct", "negated", "transposed", "neg+transposed"]


def best_match(
    M: np.ndarray,
    target_k: int,
    catalog: dict[int, list[tuple[str, np.ndarray]]],
    tol: float,
    try_perms: bool,
) -> tuple[str | None, float, str]:
    """(matched_label, op_norm, variant_name) for the best catalog match, or
    (None, best_distance, '') if nothing is within tol.

    Variants: direct / negated / transposed / neg+transposed, each optionally
    under all row permutations when try_perms and the block is ≤ 4×4."""
    n = M.shape[0]
    if try_perms and n <= 4:
        perm_indices = list(permutations(range(n)))
    else:
        perm_indices = [tuple(range(n))]

    best_dist = float("inf")
    best_label = None
    best_vname = ""
    for label, Ad in catalog.get(target_k, []):
        if Ad.shape != M.shape:
            continue
        for vname, V in zip(_VARIANT_NAMES, [M, -M, M.T, -M.T]):
            for perm in perm_indices:
                dist = float(np.linalg.norm(Ad - V[list(perm), :], 2))
                if dist < best_dist:
                    best_dist = dist
                    if dist <= tol:
                        perm_str = f" perm={list(perm)}" if perm != tuple(range(n)) else ""
                        best_label = label
                        best_vname = vname + perm_str

    if best_dist <= tol:
        return best_label, best_dist, best_vname
    return None, best_dist, ""


# ── per-diagram report ─────────────────────────────────────────────────────────


def report_diagram(
    A: np.ndarray,
    label: str,
    k: int,
    catalog: dict[int, list[tuple[str, np.ndarray]]],
    args: argparse.Namespace,
    csv_writer,
) -> bool:
    """Print matching results for one diagram; return True on any match."""
    cs = cs_factor(A)
    p, q = cs.p, cs.q
    recon_err = float(np.max(np.abs(A - cs.matrix())))
    try_perms = not args.no_perm

    print(f"\n{'=' * 65}")
    print(f"λ = {label}  (k={k}, split p={p} / q={q})")
    print(f"  CS reconstruction error: {recon_err:.2e}")
    print(f"  CS angles (°): {np.round(np.degrees(cs.thetas), 3).tolist()}")

    found_any = False
    for name, block, sub_k in [
        ("U1", cs.u1, p), ("U2", cs.u2, q),
        ("V1", cs.v1t.T, p), ("V2", cs.v2t.T, q),
    ]:
        matched_label, dist, vname = best_match(block, sub_k, catalog, args.tol, try_perms)
        cat_size = len(catalog.get(sub_k, []))
        if matched_label is not None:
            found_any = True
            print(f"  {name} ({sub_k}×{sub_k})  MATCH → {matched_label}  "
                  f"[{vname}]  op_norm={dist:.2e}")
        else:
            perm_note = " (perms tried)" if try_perms and sub_k <= 4 else ""
            print(f"  {name} ({sub_k}×{sub_k})  no match{perm_note}  "
                  f"best dist to any of {cat_size} A-mats = {dist:.4f}")
        csv_writer.writerow(
            [label, name, sub_k, matched_label or "", vname, f"{dist:.6e}"]
        )

        if args.fit:
            fit_loss, (ac, rc) = fit_subblock(block, args.fit_restarts)
            frob_tol = args.tol * block.size
            if fit_loss <= frob_tol:
                ac_int = np.round(ac).astype(int)
                rc_int = np.round(rc).astype(int)
                integer = (np.max(np.abs(ac - ac_int)) < 0.01
                           and np.max(np.abs(rc - rc_int)) < 0.01)
                if sub_k == 2 and not integer:
                    print(f"    fit:  non-integer contents (trivial for 2×2)  "
                          f"ac={np.round(ac, 4).tolist()}  rc={np.round(rc, 4).tolist()}")
                else:
                    print(f"    fit:  MATCH  ‖loss‖_F={fit_loss:.2e}")
                    print(f"          ac={np.round(ac, 4).tolist()}")
                    print(f"          rc={np.round(rc, 4).tolist()}")
                    if integer:
                        print("          *** integer contents — may be a real diagram! ***")
            else:
                print(f"    fit:  no match  best ‖loss‖_F={fit_loss:.4f}  "
                      f"(frob_tol={frob_tol:.2e})")

        if args.angle_scan > 0 and sub_k == 2:
            theta = rotation_angle(block)
            hits = angle_scan(theta, args.angle_scan)
            if hits:
                for lbl, delta in hits:
                    print(f"    angle scan: θ={np.degrees(theta):.5f}°  →  {lbl}  Δ={delta:.5f}°")
            else:
                print(f"    angle scan: θ={np.degrees(theta):.5f}°  "
                      f"no [n]/[1^m] match up to n={args.angle_scan}")

    return found_any


# ── CLI ────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--addable", type=int, default=4,
                   help="Addable cells of the starting diagram (default: 4).")
    p.add_argument("--max-size", type=int, default=40,
                   help="Max diagram size for catalog search (default: 40).")
    p.add_argument("--all", action="store_true",
                   help="Run on every starting diagram.")
    p.add_argument("--index", type=int, default=None,
                   help="Index into the starting diagram list.")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for diagram selection (0 = random).")
    p.add_argument("--tol", type=float, default=1e-8,
                   help="Operator-norm tolerance for a match (default: 1e-8).")
    p.add_argument("--no-perm", action="store_true",
                   help="Skip row-permutation search (faster but less thorough).")
    p.add_argument("--angle-scan", type=int, default=0, metavar="N",
                   help="For 2×2 sub-blocks, scan single-row [n] and single-column "
                        "[1^n] families up to n=N for an angle match (default: 0 = off).")
    p.add_argument("--fit", action="store_true",
                   help="Continuously optimize over real-valued content parameters to "
                        "check if each sub-block is an A-matrix for ANY diagram.")
    p.add_argument("--fit-restarts", type=int, default=30, metavar="N",
                   help="Number of random restarts for the continuous fit (default: 30).")
    return p.parse_args()


def _pick_random(k: int, max_size: int, rng: random.Random) -> YoungDiagram:
    """One random k-addable diagram via reservoir sampling (streaming)."""
    chosen = None
    for i, d in enumerate(diagrams_with_addable_cells(k, max_size)):
        if rng.randint(0, i) == 0:
            chosen = d
    if chosen is None:
        raise SystemExit(f"No {k}-addable diagrams found (max_size={max_size}).")
    return chosen


def _pick_by_index(k: int, max_size: int, idx: int) -> YoungDiagram:
    """The idx-th k-addable diagram, by streaming."""
    for i, d in enumerate(diagrams_with_addable_cells(k, max_size)):
        if i == idx:
            return d
    raise SystemExit(f"--index {idx} out of range for {k}-addable diagrams up to size {max_size}.")


def main() -> None:
    args = parse_args()
    rng = random.Random(None if args.seed == 0 else args.seed)
    k = args.addable
    p, q = k // 2, k - k // 2

    sub_ks = sorted({p, q})
    print(f"Building sub-block catalog for {sub_ks}-addable diagrams (max_size={args.max_size}) …")
    catalog = build_catalog(sub_ks, args.max_size)
    for sk in sub_ks:
        print(f"  {sk}-addable: {len(catalog.get(sk, []))} entries")

    perm_note = " (+ row permutations for blocks ≤ 4×4)" if not args.no_perm else ""
    print(f"\n[variants: direct / negated / transposed{perm_note}]\n")

    if args.all:
        selected = list(diagrams_with_addable_cells(k, args.max_size))
    elif args.index is not None:
        selected = [_pick_by_index(k, args.max_size, args.index)]
    else:
        selected = [_pick_random(k, args.max_size, rng)]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / "cs_subblock_match.csv"

    n_matched = 0
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["diagram", "block", "sub_k", "match", "variant", "op_norm"])
        for d in selected:
            label = str(d.partition)
            if report_diagram(a_matrix(d), label, k, catalog, args, writer):
                n_matched += 1

    print(f"\n{'=' * 65}")
    print(f"SUMMARY: {n_matched} / {len(selected)} diagram(s) had at least one sub-block match.")
    if n_matched == 0:
        print("  → No sub-blocks are A-matrices (up to tol, sign, transpose, perms).")
        print("    Consider: --max-size <larger> to widen the catalog, or --tol <larger>.")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
