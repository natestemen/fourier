#!/usr/bin/env python3
"""
Test whether the CS-decomposition sub-blocks of A(λ) are themselves A-matrices.

For a k-addable partition λ, the CS decomposition gives:
  A(λ) = [U₁ ⊕ U₂] · CS(θ) · [V₁ᵀ ⊕ V₂ᵀ]

where U₁, U₂ are (p×p) and (q×q) orthogonal matrices (p = k//2, q = k−p).
This script checks whether any of {U₁, U₂, V₁, V₂} — and simple variants
(negation, transpose, row/col permutations for small blocks) — match the
A-matrix of some (p or q)-addable Young diagram.

For 2×2 sub-blocks, --angle-scan also checks large diagram families
(single-row [n], single-column [1^n], rectangle [q^p]) analytically to
see if the sub-block angle matches any of them up to large n.

--fit uses continuous optimization over real-valued content parameters to
determine if a sub-block is an A-matrix for ANY diagram, not just those
in the bounded catalog.  For each sub-block it minimizes
    ‖M - A(ac, rc)‖_F
over interleaved contents ac > rc[0] > ac[1] > … > rc[-1] > ac[-1],
starting from multiple random initialisations.

Usage:
    python3 cs_subblock_test.py                        # one random 4-addable diagram
    python3 cs_subblock_test.py --addable 6
    python3 cs_subblock_test.py --all --addable 4      # every 4-addable diagram
    python3 cs_subblock_test.py --index 2 --addable 4
    python3 cs_subblock_test.py --all --addable 4 --no-perm   # skip permutation search
    python3 cs_subblock_test.py --index 0 --angle-scan 5000   # scan [n] family up to n=5000
    python3 cs_subblock_test.py --fit                         # continuous optimisation
    python3 cs_subblock_test.py --fit --fit-restarts 50       # more random starts
"""
from __future__ import annotations

import argparse
import random
from itertools import permutations

import numpy as np
from scipy.linalg import cossin
from scipy.optimize import minimize

from compute_matrix import A_matrix
from helper import find_yds_with_fixed_addable_cells
from yungdiagram import YoungDiagram


# ── angle-based scan for 2×2 sub-blocks ───────────────────────────────────────

def _rotation_angle(M: np.ndarray) -> float | None:
    """Extract rotation angle θ ∈ [0, π/2) from a 2×2 orthogonal matrix
    (or its negation / transpose), returning None if M is not 2×2."""
    if M.shape != (2, 2):
        return None
    # Try the four sign/transpose variants and return the angle in [0, π/2)
    # that puts the matrix closest to [[c,-s],[s,c]] form.
    for V in [M, -M, M.T, -M.T]:
        c, s = V[0, 0], V[1, 0]
        if abs(c**2 + s**2 - 1) < 1e-6 and c >= 0 and s >= 0:
            return float(np.arctan2(s, c))
    # Fallback: just take the angle from the first variant
    return float(np.arctan2(M[1, 0], M[0, 0])) % (np.pi / 2)


def _2addable_angle(n: int) -> float:
    """Rotation angle of the A-matrix for the single-row partition [n].

    For [n]: two addable cells (content n and content -1), one removable (content n-1).
    The A-matrix is:
        [ sqrt(1/(n+1))     -sqrt(n/(n+1))  ]
        [ sqrt(n/(n+1))      sqrt(1/(n+1))  ]
    so θ = arctan(sqrt(n)).
    """
    return float(np.arctan(np.sqrt(n)))


def _col_angle(m: int) -> float:
    """Rotation angle for the single-column [1^m] partition (m rows of 1).

    For [1^m]: addable cells have contents 1 (top) and -m (bottom),
    removable content 0.  By symmetry with [n], θ = arctan(1/sqrt(m)).
    """
    return float(np.arctan(1.0 / np.sqrt(m))) if m > 0 else np.pi / 2


def angle_scan(
    angle: float,
    n_max: int,
    tol_deg: float = 0.001,
) -> list[tuple[str, float]]:
    """Scan the single-row [n] and single-column [1^m] families for n,m up to n_max,
    looking for diagrams whose A-matrix angle matches `angle` within tol_deg degrees.

    Returns list of (label, delta_degrees).
    """
    hits: list[tuple[str, float]] = []
    tol_rad = np.radians(tol_deg)

    for n in range(1, n_max + 1):
        for θ_fn, lbl in [(_2addable_angle, f"[{n}]"), (_col_angle, f"[1^{n}]")]:
            θ = θ_fn(n)
            # Check angle and π/2 - angle (because neg/transpose variants flip the angle)
            for candidate in [θ, np.pi / 2 - θ]:
                delta = abs(angle - candidate)
                delta = min(delta, np.pi / 2 - delta)  # mod symmetry
                if delta < tol_rad:
                    hits.append((lbl, float(np.degrees(delta))))

    return hits


# ── continuous fit ─────────────────────────────────────────────────────────────

def _amatrix_from_contents(ac: np.ndarray, rc: np.ndarray) -> np.ndarray:
    """Build a k×k A-matrix from real-valued contents ac (k addable) and rc (k-1 removable).

    Formula:
        alpha_i^2 = prod_{j}(rc_j - ac_i) / prod_{j≠i}(ac_j - ac_i)
        beta_j    = 1 / sqrt( sum_i alpha_i^2 / (ac_i - rc_j)^2 )
        A[i, 0]   = alpha_i
        A[i, j+1] = alpha_i * beta_j / (ac_i - rc_j)
    """
    k = len(ac)
    # alpha^2 — sign can be negative if interlacing is violated; clamp to 0 for safety
    alpha2 = np.array([
        np.prod(rc - ac[i]) / np.prod(np.delete(ac, i) - ac[i])
        for i in range(k)
    ])
    alpha = np.sqrt(np.maximum(alpha2, 0.0))

    A = np.zeros((k, k))
    A[:, 0] = alpha
    for j in range(k - 1):
        diffs = ac - rc[j]           # shape (k,)
        col_norm2 = np.sum(alpha2 / diffs**2)
        if col_norm2 < 1e-30:
            return A  # degenerate; caller will see large residual
        beta_j = 1.0 / np.sqrt(col_norm2)
        A[:, j + 1] = alpha * beta_j / diffs
    return A


def _fit_loss(params: np.ndarray, M: np.ndarray) -> float:
    """Frobenius loss ‖M - A(ac, rc)‖_F + soft interlacing penalty."""
    k = M.shape[0]
    # params encodes the 2k-1 interleaved values in *sorted descending* order
    # via softplus gaps: params = [center, log_gap_0, ..., log_gap_{2k-2}]
    center = params[0]
    gaps   = np.exp(params[1:])          # all positive
    # interleaved: ac[0], rc[0], ac[1], rc[1], ..., rc[k-2], ac[k-1]
    vals = np.empty(2 * k - 1)
    vals[0] = center
    for i in range(1, 2 * k - 1):
        vals[i] = vals[i - 1] - gaps[i - 1]

    ac = vals[0::2]   # indices 0, 2, 4, …  (k values)
    rc = vals[1::2]   # indices 1, 3, 5, …  (k-1 values)

    A_fit = _amatrix_from_contents(ac, rc)
    # Try the four sign/transpose variants and take the minimum loss
    best = min(
        np.sum((M - V) ** 2)
        for V in [A_fit, -A_fit, A_fit.T, -A_fit.T]
        if V.shape == M.shape
    )
    return float(best)


def fit_subblock(M: np.ndarray, n_restarts: int = 30) -> tuple[float, np.ndarray | None]:
    """Minimise ‖M - A(ac,rc)‖_F over continuous content parameters.

    Returns (best_frob_loss, best_ac_rc_pair) where best_ac_rc_pair is
    (ac, rc) as numpy arrays, or None if all restarts failed.
    """
    k = M.shape[0]
    n_params = 2 * k - 1   # center + 2k-2 gaps

    rng = np.random.default_rng(42)
    best_loss = float("inf")
    best_params = None

    for _ in range(n_restarts):
        # Random start: center near 0, gaps ~ Exp(1)
        center = rng.uniform(-k, k)
        log_gaps = rng.normal(0.0, 0.5, size=n_params - 1)
        x0 = np.concatenate([[center], log_gaps])

        try:
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                res = minimize(_fit_loss, x0, args=(M,), method="L-BFGS-B",
                               options={"maxiter": 500, "ftol": 1e-14, "gtol": 1e-10})
            if res.fun < best_loss:
                best_loss = res.fun
                best_params = res.x
        except Exception:
            continue

    if best_params is None:
        return best_loss, None

    # Recover ac, rc from best_params
    center = best_params[0]
    gaps   = np.exp(best_params[1:])
    vals   = np.empty(2 * k - 1)
    vals[0] = center
    for i in range(1, 2 * k - 1):
        vals[i] = vals[i - 1] - gaps[i - 1]
    ac = vals[0::2]
    rc = vals[1::2]
    return best_loss, (ac, rc)


# ── catalog ────────────────────────────────────────────────────────────────────

def build_catalog(ks: list[int], max_size: int) -> dict[int, list[tuple[str, np.ndarray]]]:
    """Return {k: [(label, A_matrix)]} for each target addable count."""
    catalog: dict[int, list[tuple[str, np.ndarray]]] = {}
    for k in ks:
        diags = list(find_yds_with_fixed_addable_cells(k, max_size))
        if diags:
            catalog[k] = [
                (str(getattr(d, "partition", d)), np.array(A_matrix(d), dtype=float))
                for d in diags
            ]
    return catalog


# ── matching ───────────────────────────────────────────────────────────────────

_VARIANT_NAMES = ["direct", "negated", "transposed", "neg+transposed"]


def _variants(M: np.ndarray) -> list[np.ndarray]:
    return [M, -M, M.T, -M.T]


def _best_match(
    M: np.ndarray,
    target_k: int,
    catalog: dict[int, list[tuple[str, np.ndarray]]],
    tol: float,
    try_perms: bool,
) -> tuple[str | None, float, str]:
    """
    Return (matched_label, op_norm, variant_name) for the best match, or
    (None, best_distance, '') if nothing is within tol.

    Checks: direct / negated / transposed / neg+transposed.
    If try_perms and k <= 4: also checks all row permutations of each variant.
    """
    if target_k not in catalog:
        return None, float("inf"), ""

    best_dist = float("inf")
    best_label = None
    best_vname = ""

    n = M.shape[0]
    perm_indices: list[tuple[int, ...]] = []
    if try_perms and n <= 4:
        perm_indices = list(permutations(range(n)))
    else:
        perm_indices = [tuple(range(n))]  # identity only

    for label, Ad in catalog[target_k]:
        if Ad.shape != M.shape:
            continue
        for vname, V in zip(_VARIANT_NAMES, _variants(M)):
            for perm in perm_indices:
                P = np.eye(n)[list(perm), :]
                candidate = P @ V
                dist = float(np.linalg.norm(Ad - candidate, 2))
                if dist < best_dist:
                    best_dist = dist
                    if dist <= tol:
                        perm_str = f" perm={list(perm)}" if perm != tuple(range(n)) else ""
                        best_label = label
                        best_vname = vname + perm_str

    if best_dist <= tol:
        return best_label, best_dist, best_vname
    return None, best_dist, ""


# ── CS decomposition ───────────────────────────────────────────────────────────

def cs_split(A: np.ndarray, p: int | None = None):
    """
    Compute CS decomposition A = [U1⊕U2] · CS · [V1ᵀ⊕V2ᵀ].

    Returns (U1, U2, V1, V2, thetas, recon_err).
    Note: V1 = Vt1.T, V2 = Vt2.T (scipy returns Vt = Vᵀ).
    """
    k = A.shape[0]
    if p is None:
        p = k // 2
    q = k - p
    (u1, u2), thetas, (vt1, vt2) = cossin(A, p=p, q=q, separate=True)

    # Verify reconstruction
    C = np.diag(np.cos(thetas))
    S = np.diag(np.sin(thetas))
    CS_block = np.block([[C, -S], [S, C]])
    U_block  = np.block([[u1, np.zeros((p, q))], [np.zeros((q, p)), u2]])
    Vt_block = np.block([[vt1, np.zeros((p, q))], [np.zeros((q, p)), vt2]])
    recon_err = float(np.max(np.abs(A - U_block @ CS_block @ Vt_block)))

    return u1, u2, vt1.T, vt2.T, thetas, recon_err


# ── per-diagram report ─────────────────────────────────────────────────────────

def report_diagram(
    A: np.ndarray,
    label: str,
    k: int,
    catalog: dict[int, list[tuple[str, np.ndarray]]],
    tol: float,
    try_perms: bool,
    angle_scan_n: int = 0,
    do_fit: bool = False,
    fit_restarts: int = 30,
) -> bool:
    """Print matching results for one diagram; return True if any sub-block matched."""
    p = k // 2
    q = k - p

    try:
        u1, u2, v1, v2, thetas, recon_err = cs_split(A, p)
    except Exception as e:
        print(f"  CS decomposition failed: {e}")
        return False

    print(f"\n{'=' * 65}")
    print(f"λ = {label}  (k={k}, split p={p} / q={q})")
    print(f"  CS reconstruction error: {recon_err:.2e}")
    print(f"  CS angles (°): {np.round(np.degrees(thetas), 3).tolist()}")

    found_any = False
    for name, block, sub_k in [
        ("U₁", u1, p), ("U₂", u2, q),
        ("V₁", v1, p), ("V₂", v2, q),
    ]:
        matched_label, dist, vname = _best_match(block, sub_k, catalog, tol, try_perms)
        cat_size = len(catalog.get(sub_k, []))
        if matched_label is not None:
            found_any = True
            print(f"  {name} ({sub_k}×{sub_k})  MATCH → {matched_label}  "
                  f"[{vname}]  op_norm={dist:.2e}")
        else:
            perm_note = " (perms tried)" if try_perms and sub_k <= 4 else ""
            print(f"  {name} ({sub_k}×{sub_k})  no match{perm_note}  "
                  f"best dist to any of {cat_size} A-mats = {dist:.4f}")

        # Continuous fit: optimise over real-valued content parameters
        if do_fit:
            fit_loss, ac_rc = fit_subblock(block, fit_restarts)
            frob_tol = tol * block.size  # scale: op-norm tol → Frobenius tol
            if fit_loss <= frob_tol:
                ac, rc = ac_rc
                # Check if contents are integer (would correspond to a real diagram)
                ac_int = np.round(ac).astype(int)
                rc_int = np.round(rc).astype(int)
                integer = (np.max(np.abs(ac - ac_int)) < 0.01
                           and np.max(np.abs(rc - rc_int)) < 0.01)
                if sub_k == 2 and not integer:
                    # Every 2×2 rotation is trivially achievable for some real ac, rc
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

        # Angle scan for 2×2 sub-blocks
        if angle_scan_n > 0 and sub_k == 2:
            θ = _rotation_angle(block)
            if θ is not None:
                hits = angle_scan(θ, angle_scan_n)
                if hits:
                    for lbl, Δ in hits:
                        print(f"    angle scan: θ={np.degrees(θ):.5f}°  →  {lbl}  Δ={Δ:.5f}°")
                else:
                    print(f"    angle scan: θ={np.degrees(θ):.5f}°  no [n]/[1^m] match up to n={angle_scan_n}")

    return found_any


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
                   help="For 2×2 sub-blocks, scan single-row [n] and single-column [1^n] "
                        "families up to n=N for an angle match (default: 0 = disabled).")
    p.add_argument("--fit", action="store_true",
                   help="Continuously optimise over real-valued content parameters "
                        "to check if each sub-block is an A-matrix for ANY diagram.")
    p.add_argument("--fit-restarts", type=int, default=30, metavar="N",
                   help="Number of random restarts for the continuous fit (default: 30).")
    return p.parse_args()


def _pick_random(k: int, max_size: int, rng: random.Random) -> YoungDiagram:
    """Pick one random k-addable diagram via reservoir sampling (never materializes the full list)."""
    chosen = None
    for i, d in enumerate(find_yds_with_fixed_addable_cells(k, max_size)):
        if rng.randint(0, i) == 0:
            chosen = d
    if chosen is None:
        raise SystemExit(f"No {k}-addable diagrams found (max_size={max_size}).")
    return chosen


def _pick_by_index(k: int, max_size: int, idx: int) -> YoungDiagram:
    """Return the idx-th k-addable diagram by streaming, without loading all of them."""
    for i, d in enumerate(find_yds_with_fixed_addable_cells(k, max_size)):
        if i == idx:
            return d
    raise SystemExit(f"--index {idx} out of range for {k}-addable diagrams up to size {max_size}.")


def main() -> None:
    args = parse_args()
    rng  = random.Random(None if args.seed == 0 else args.seed)
    k    = args.addable
    p, q = k // 2, k - k // 2
    try_perms = not args.no_perm

    sub_ks = sorted(set([p, q]))
    print(f"Building sub-block catalog for {sub_ks}-addable diagrams (max_size={args.max_size}) …")
    catalog = build_catalog(sub_ks, args.max_size)
    for sk in sub_ks:
        print(f"  {sk}-addable: {len(catalog.get(sk, []))} entries")

    perm_note = " (+ row permutations for blocks ≤ 4×4)" if try_perms else ""
    print(f"\n[variants: direct / negated / transposed{perm_note}]\n")

    n_matched = 0
    n_total   = 0

    if args.all:
        for d in find_yds_with_fixed_addable_cells(k, args.max_size):
            label = str(getattr(d, "partition", d))
            A     = np.array(A_matrix(d), dtype=float)
            n_total += 1
            if report_diagram(A, label, k, catalog, args.tol, try_perms,
                               args.angle_scan, args.fit, args.fit_restarts):
                n_matched += 1
    elif args.index is not None:
        d     = _pick_by_index(k, args.max_size, args.index)
        label = str(getattr(d, "partition", d))
        A     = np.array(A_matrix(d), dtype=float)
        n_total = 1
        if report_diagram(A, label, k, catalog, args.tol, try_perms,
                          args.angle_scan, args.fit, args.fit_restarts):
            n_matched = 1
    else:
        d     = _pick_random(k, args.max_size, rng)
        label = str(getattr(d, "partition", d))
        A     = np.array(A_matrix(d), dtype=float)
        n_total = 1
        if report_diagram(A, label, k, catalog, args.tol, try_perms,
                          args.angle_scan, args.fit, args.fit_restarts):
            n_matched = 1

    print(f"\n{'=' * 65}")
    print(f"SUMMARY: {n_matched} / {n_total} diagram(s) had at least one sub-block match.")
    if n_matched == 0:
        print("  → No sub-blocks are A-matrices (up to tol, sign, transpose, perms).")
        print("    Consider: --max-size <larger> to widen the catalog, or --tol <larger>.")


if __name__ == "__main__":
    main()
