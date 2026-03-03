#!/usr/bin/env python3
"""Scan gun family, compile with bqskit opt level 4, and plot U3 theta vs n."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

import bqskit
from bqskit.ir.gates import RYGate, SqrtCNOTGate, U3Gate
from bqskit.compiler.machine import MachineModel

from symbolic_a_matrix import build_symbolic_a_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-min", type=int, default=3)
    parser.add_argument("--n-max", type=int, default=50)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("data/plots/gun_bqskit_theta.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    A_sym, symbols = build_symbolic_a_matrix()

    gate_set = {RYGate(), SqrtCNOTGate()}
    model = MachineModel(2, gate_set=gate_set)

    n_vals = []
    gate_indices = []
    thetas_flat = []

    for n in range(args.n_min, args.n_max + 1, args.step):
        print(n)
        subs = {
            symbols[0]: n,
            symbols[1]: 2,
            symbols[2]: 1,
            symbols[3]: 1,
            symbols[4]: 1,
            symbols[5]: 1,
        }
        yd_mat = np.array(sp.Matrix(A_sym.subs(subs)).evalf(), dtype=complex)
        circuit = bqskit.Circuit.from_unitary(yd_mat)
        compiled = bqskit.compile(circuit, optimization_level=4)#, model=model)
        found = False
        u3_index = 0
        for op in compiled:
            if isinstance(op.gate, U3Gate):
                found = True
                thetas_flat.append(float(op.params[0]) % 2*np.pi)
                n_vals.append(n)
                gate_indices.append(u3_index)
                u3_index += 1

        if not found:
            continue

    if not thetas_flat:
        raise SystemExit("No U3 thetas found.")

    fig, ax = plt.subplots()
    scatter = ax.scatter(n_vals, thetas_flat, c=gate_indices, cmap="viridis", s=14)
    ax.set_xlabel("n")
    ax.set_ylabel("theta")
    ax.set_title("Gun family: U3 theta vs n (bqskit opt=4)")
    ax.grid(True, alpha=0.3)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("U3 gate index")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
