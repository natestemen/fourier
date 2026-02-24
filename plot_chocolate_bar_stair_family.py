#!/usr/bin/env python3
"""Plot b(n) and |c(n)| for chocolate bar stair family: w1=n,w2=n-1,w3=n-2,h1=h2=h3=1."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from qiskit.synthesis import TwoQubitWeylDecomposition
from scipy.optimize import curve_fit

from symbolic_a_matrix import build_symbolic_a_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-min", type=int, default=3)
    parser.add_argument("--n-max", type=int, default=10000)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument(
        "--require-unitary",
        action="store_true",
        default=True,
        help="Skip matrices that are not numerically unitary.",
    )
    parser.add_argument(
        "--no-require-unitary",
        action="store_false",
        dest="require_unitary",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/plots/chocolate_bar_stair_bc_vs_n.png"),
    )
    return parser.parse_args()


def _is_unitary(mat: np.ndarray, tol: float = 1e-6) -> bool:
    eye = np.eye(mat.shape[0], dtype=complex)
    return np.allclose(mat.conj().T @ mat, eye, atol=tol)


def _family_series(n_min: int, n_max: int, step: int, A_func, require_unitary: bool):
    ns: list[int] = []
    bs: list[float] = []
    cs: list[float] = []
    for n in range(n_min, n_max + 1, step):
        if n < 3:
            continue
        A_val = np.array(A_func(n, n - 1, n - 2, 1, 1, 1), dtype=complex)
        if require_unitary and not _is_unitary(A_val):
            continue
        decomp = TwoQubitWeylDecomposition(A_val)
        ns.append(n)
        bs.append(float(decomp.b))
        cs.append(abs(float(decomp.c)))
    return ns, bs, cs


def _family_limit(A_sym: sp.Matrix, symbols: tuple[sp.Symbol, ...]) -> tuple[float, float] | None:
    w1, w2, w3, h1, h2, h3 = symbols
    A_family = A_sym.subs({w2: w1 - 1, w3: w1 - 2, h1: 1, h2: 1, h3: 1})
    A_lim = A_family.applyfunc(lambda expr: sp.limit(expr, w1, sp.oo))
    try:
        A_val = np.array(A_lim.evalf(), dtype=complex)
    except Exception:
        return None
    if not np.isfinite(A_val).all():
        return None
    try:
        decomp = TwoQubitWeylDecomposition(A_val)
    except Exception:
        return None
    return float(decomp.b), abs(float(decomp.c))


def _power_model(n: np.ndarray, c_inf: float, a: float, p: float) -> np.ndarray:
    return c_inf + a / (n**p)


def _fit_stats(y: np.ndarray, yhat: np.ndarray, k_params: int) -> dict[str, float]:
    resid = y - yhat
    sse = float(np.sum(resid**2))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    rmse = float(np.sqrt(sse / len(y)))
    aic = float(len(y) * np.log(sse / len(y)) + 2 * k_params) if sse > 0 else float("inf")
    return {"r2": r2, "rmse": rmse, "aic": aic}


def main() -> None:
    args = parse_args()
    A_sym, symbols = build_symbolic_a_matrix()
    A_func = sp.lambdify(symbols, A_sym, "numpy")

    ns, bs, cs = _family_series(args.n_min, args.n_max, args.step, A_func, args.require_unitary)
    if not ns:
        raise SystemExit("No valid points computed for the requested range.")

    lim = _family_limit(A_sym, symbols)

    n_arr = np.asarray(ns, dtype=float)
    b_arr = np.asarray(bs, dtype=float)
    c_arr = np.asarray(cs, dtype=float)

    b_guess = (b_arr[-1], b_arr[0] - b_arr[-1], 1.0)
    b_params, _ = curve_fit(_power_model, n_arr, b_arr, p0=b_guess, maxfev=20000)

    c_guess = (c_arr[-1], c_arr[0] - c_arr[-1], 1.0)
    c_params, _ = curve_fit(_power_model, n_arr, c_arr, p0=c_guess, maxfev=20000)

    b_fit = _power_model(n_arr, *b_params)
    c_fit = _power_model(n_arr, *c_params)

    b_stats = _fit_stats(b_arr, b_fit, 3)
    c_stats = _fit_stats(c_arr, c_fit, 3)

    fig, ax = plt.subplots()
    ax.plot(n_arr, b_arr, marker="o", markersize=1.5, linewidth=1.0, color="tab:blue", label="b(n)")
    ax.plot(n_arr, c_arr, marker="o", markersize=1.5, linewidth=1.0, color="tab:orange", label="|c(n)|")
    ax.plot(n_arr, b_fit, linewidth=1.5, color="tab:blue", linestyle="--", label="fit: b_inf + a/n^p")
    ax.plot(n_arr, c_fit, linewidth=1.5, color="tab:orange", linestyle="--", label="fit: c_inf + a/n^p")

    if lim is not None:
        ax.scatter([args.n_max], [lim[0]], marker="*", s=120, color="tab:blue", edgecolors="black", linewidths=0.6)
        ax.scatter([args.n_max], [lim[1]], marker="*", s=120, color="tab:orange", edgecolors="black", linewidths=0.6)
        ax.annotate("n→∞", xy=(args.n_max, lim[0]), xytext=(6, 6), textcoords="offset points")
        ax.annotate("n→∞", xy=(args.n_max, lim[1]), xytext=(6, 6), textcoords="offset points")

    ax.set_xlabel("n")
    ax.set_ylabel("value")
    ax.set_title("Chocolate bar stair family: w1=n,w2=n-1,w3=n-2,h1=h2=h3=1")
    ax.grid(True, alpha=0.3)
    ax.legend()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {args.output}")
    print("Fit parameters:")
    print(f"  b(n) ~ b_inf + a/n^p:    b_inf={b_params[0]:.6g}, a={b_params[1]:.6g}, p={b_params[2]:.6g}")
    print(f"  |c(n)| ~ c_inf + a/n^p:  c_inf={c_params[0]:.6g}, a={c_params[1]:.6g}, p={c_params[2]:.6g}")
    print("Fit quality (higher R^2, lower RMSE/AIC is better):")
    print(f"  b 1/n^p: R^2={b_stats['r2']:.6g} RMSE={b_stats['rmse']:.6g} AIC={b_stats['aic']:.6g}")
    print(f"  c 1/n^p: R^2={c_stats['r2']:.6g} RMSE={c_stats['rmse']:.6g} AIC={c_stats['aic']:.6g}")
    plt.show()


if __name__ == "__main__":
    main()
