"""Weyl coordinates b(n), |c(n)| along one-parameter families of 4-addable diagrams.

Question: how do the free Weyl coordinates (b, c) of a 4-addable A-matrix behave
along a one-parameter family of diagrams as the parameter n → ∞, and do they
converge to the Weyl coordinates of the symbolic limit matrix?

Supports report.md, Finding 1: every 4-addable A-matrix has a = π/4 while
(b, c) vary continuously over the family — this script traces that variation
along four named block-parameter families and fits the convergence rate.

Expected result: b(n) and |c(n)| converge to the symbolic n→∞ limit with a
power law c_inf + a/n^p, p ≈ 1, R² ≈ 1.

Families (block parameters (w1, w2, w3, h1, h2, h3) of the generic 4-addable
diagram; row lengths are the suffix sums of widths):

    gun:            (n, 2, 1, 1, 1, 1)
    f:              (3, 2, 1, 1, 1, n)
    uzi:            (3, 2, 1, 1, n, 1)
    chocolate-bar:  (n, n−1, n−2, 1, 1, 1)

Replaces plot_gun_family.py, plot_F_family.py, plot_uzi_family.py, and
plot_chocolate_bar_stair_family.py (line-for-line clones up to the family
substitution). Behavior change vs. those scripts: output paths are fixed to
data/plots/family_weyl_<name>.png and data/family_weyl_<name>.csv, and no
interactive plt.show().
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.optimize import curve_fit

from fourier.amatrix import a_matrix_generic4
from fourier.weyl import weyl_coordinates

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
PLOT_DIR = DATA_DIR / "plots"

Params = tuple[int, int, int, int, int, int]
Symbols = tuple[sp.Symbol, ...]


@dataclass(frozen=True)
class Family:
    """A one-parameter family of generic 4-addable diagrams."""

    label: str  # human-readable parametrization, used in the plot title
    n_min: int  # smallest n for which the family is defined / well-behaved
    params: Callable[[int], Params]  # n -> (w1, w2, w3, h1, h2, h3)
    # symbols -> (substitution fixing all but one symbol, the symbol sent to ∞)
    limit_subs: Callable[[Symbols], tuple[dict[sp.Symbol, object], sp.Symbol]]


def _gun_limit(s: Symbols) -> tuple[dict[sp.Symbol, object], sp.Symbol]:
    w1, w2, w3, h1, h2, h3 = s
    return {w2: 2, w3: 1, h1: 1, h2: 1, h3: 1}, w1


def _f_limit(s: Symbols) -> tuple[dict[sp.Symbol, object], sp.Symbol]:
    w1, w2, w3, h1, h2, h3 = s
    return {w1: 3, w2: 2, w3: 1, h1: 1, h2: 1}, h3


def _uzi_limit(s: Symbols) -> tuple[dict[sp.Symbol, object], sp.Symbol]:
    w1, w2, w3, h1, h2, h3 = s
    return {w1: 3, w2: 2, w3: 1, h1: 1, h3: 1}, h2


def _chocolate_limit(s: Symbols) -> tuple[dict[sp.Symbol, object], sp.Symbol]:
    w1, w2, w3, h1, h2, h3 = s
    return {w2: w1 - 1, w3: w1 - 2, h1: 1, h2: 1, h3: 1}, w1


FAMILIES: dict[str, Family] = {
    "gun": Family(
        label="w1=n, w2=2, w3=1, h1=h2=h3=1",
        n_min=50,
        params=lambda n: (n, 2, 1, 1, 1, 1),
        limit_subs=_gun_limit,
    ),
    "f": Family(
        label="w1=3, w2=2, w3=1, h1=1, h2=1, h3=n",
        n_min=1,
        params=lambda n: (3, 2, 1, 1, 1, n),
        limit_subs=_f_limit,
    ),
    "uzi": Family(
        label="w1=3, w2=2, w3=1, h1=1, h2=n, h3=1",
        n_min=1,
        params=lambda n: (3, 2, 1, 1, n, 1),
        limit_subs=_uzi_limit,
    ),
    "chocolate-bar": Family(
        label="w1=n, w2=n-1, w3=n-2, h1=h2=h3=1",
        n_min=3,
        params=lambda n: (n, n - 1, n - 2, 1, 1, 1),
        limit_subs=_chocolate_limit,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=sorted(FAMILIES), default="gun")
    parser.add_argument(
        "--all", action="store_true", help="Run every family (ignores --family)."
    )
    parser.add_argument(
        "--n-min",
        type=int,
        default=None,
        help="Smallest n (default: per-family; gun 50, f/uzi 1, chocolate-bar 3).",
    )
    parser.add_argument("--n-max", type=int, default=10000)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument(
        "--require-unitary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip parameter values whose matrix is not numerically unitary.",
    )
    return parser.parse_args()


def is_unitary(mat: np.ndarray, tol: float = 1e-6) -> bool:
    return np.allclose(mat.conj().T @ mat, np.eye(mat.shape[0]), atol=tol)


def family_series(
    A_func,
    family: Family,
    n_min: int,
    n_max: int,
    step: int,
    require_unitary: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(n, b(n), |c(n)|) arrays for the family, skipping degenerate n."""
    ns: list[int] = []
    bs: list[float] = []
    cs: list[float] = []
    with np.errstate(invalid="ignore"):
        for n in range(n_min, n_max + 1, step):
            A = np.array(A_func(*family.params(n)), dtype=complex)
            if not np.isfinite(A).all():
                continue
            if require_unitary and not is_unitary(A):
                continue
            _, b, c = weyl_coordinates(A)
            ns.append(n)
            bs.append(b)
            cs.append(abs(c))
    return np.asarray(ns, dtype=float), np.asarray(bs), np.asarray(cs)


def family_limit_matrix(A_sym: sp.Matrix, symbols: Symbols, family: Family) -> sp.Matrix:
    """The exact n→∞ limit of the family's symbolic A-matrix."""
    subs, var = family.limit_subs(symbols)
    return A_sym.subs(subs).applyfunc(lambda expr: sp.limit(expr, var, sp.oo))


def power_model(n: np.ndarray, c_inf: float, a: float, p: float) -> np.ndarray:
    return c_inf + a / (n**p)


def fit_stats(y: np.ndarray, yhat: np.ndarray, k_params: int) -> dict[str, float]:
    resid = y - yhat
    sse = float(np.sum(resid**2))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    rmse = float(np.sqrt(sse / len(y)))
    aic = float(len(y) * np.log(sse / len(y)) + 2 * k_params) if sse > 0 else float("-inf")
    return {"r2": r2, "rmse": rmse, "aic": aic}


def run_family(
    name: str,
    A_sym: sp.Matrix,
    symbols: Symbols,
    A_func,
    args: argparse.Namespace,
) -> None:
    family = FAMILIES[name]
    n_min = family.n_min if args.n_min is None else args.n_min

    n_arr, b_arr, c_arr = family_series(
        A_func, family, n_min, args.n_max, args.step, args.require_unitary
    )
    if n_arr.size == 0:
        raise SystemExit(f"{name}: no valid points computed for the requested range.")

    A_lim = family_limit_matrix(A_sym, symbols, family)
    A_lim_val = np.array(A_lim.evalf(), dtype=complex)
    _, b_lim, c_lim = weyl_coordinates(A_lim_val)
    c_lim = abs(c_lim)

    b_guess = (b_arr[-1], b_arr[0] - b_arr[-1], 1.0)
    b_params, _ = curve_fit(power_model, n_arr, b_arr, p0=b_guess, maxfev=20000)
    c_guess = (c_arr[-1], c_arr[0] - c_arr[-1], 1.0)
    c_params, _ = curve_fit(power_model, n_arr, c_arr, p0=c_guess, maxfev=20000)

    b_fit = power_model(n_arr, *b_params)
    c_fit = power_model(n_arr, *c_params)
    b_stats = fit_stats(b_arr, b_fit, 3)
    c_stats = fit_stats(c_arr, c_fit, 3)

    stem = f"family_weyl_{name.replace('-', '_')}"

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / f"{stem}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "weyl_b", "weyl_abs_c"])
        writer.writerows(zip(n_arr.astype(int), b_arr, c_arr))

    fig, ax = plt.subplots()
    ax.plot(n_arr, b_arr, marker="o", markersize=1.5, linewidth=1.0, color="tab:blue", label="b(n)")
    ax.plot(n_arr, c_arr, marker="o", markersize=1.5, linewidth=1.0, color="tab:orange", label="|c(n)|")
    ax.plot(n_arr, b_fit, linewidth=1.5, color="tab:blue", linestyle="--", label="fit: b_inf + a/n^p")
    ax.plot(n_arr, c_fit, linewidth=1.5, color="tab:orange", linestyle="--", label="fit: c_inf + a/n^p")
    ax.scatter([n_arr[-1]], [b_lim], marker="*", s=120, color="tab:blue", edgecolors="black", linewidths=0.6)
    ax.scatter([n_arr[-1]], [c_lim], marker="*", s=120, color="tab:orange", edgecolors="black", linewidths=0.6)
    ax.annotate("n→∞", xy=(n_arr[-1], b_lim), xytext=(6, 6), textcoords="offset points")
    ax.annotate("n→∞", xy=(n_arr[-1], c_lim), xytext=(6, 6), textcoords="offset points")
    ax.set_xlabel("n")
    ax.set_ylabel("value")
    ax.set_title(f"{name} family: {family.label}")
    ax.grid(True, alpha=0.3)
    ax.legend()

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = PLOT_DIR / f"{stem}.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"=== {name} family: {family.label} ===")
    print(f"Saved plot to {plot_path}")
    print(f"Saved series to {csv_path}")
    print("Limit matrix (n -> inf):")
    sp.pprint(A_lim, use_unicode=False)
    print(f"Limit Weyl: b={b_lim:.6g} |c|={c_lim:.6g}")
    print("Fit parameters:")
    print(f"  b(n) ~ b_inf + a/n^p:    b_inf={b_params[0]:.6g}, a={b_params[1]:.6g}, p={b_params[2]:.6g}")
    print(f"  |c(n)| ~ c_inf + a/n^p:  c_inf={c_params[0]:.6g}, a={c_params[1]:.6g}, p={c_params[2]:.6g}")
    print("Fit quality (higher R^2, lower RMSE/AIC is better):")
    print(f"  b 1/n^p: R^2={b_stats['r2']:.6g} RMSE={b_stats['rmse']:.6g} AIC={b_stats['aic']:.6g}")
    print(f"  c 1/n^p: R^2={c_stats['r2']:.6g} RMSE={c_stats['rmse']:.6g} AIC={c_stats['aic']:.6g}")


def main() -> None:
    args = parse_args()
    A_sym, symbols = a_matrix_generic4()
    A_func = sp.lambdify(symbols, A_sym, "numpy")

    names = sorted(FAMILIES) if args.all else [args.family]
    for name in names:
        run_family(name, A_sym, symbols, A_func, args)


if __name__ == "__main__":
    main()
