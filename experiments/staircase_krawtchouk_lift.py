"""Verify the exact two-Krawtchouk staircase circuit, including leakage.

Run: .venv/bin/python experiments/staircase_krawtchouk_lift.py --sizes 4 16 64 128

Dense reference construction is classical verification, not a compiled
quantum circuit.  In particular, timings here do not measure coherent
angle generation.  Read fourier.staircase_lift for the circuit, proof,
branch-flag erasure, elementary arithmetic costs, and primitive interface.
"""

import argparse

import numpy as np

from fourier.amatrix import staircase_a_matrix
from fourier.staircase_lift import staircase_lift


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[4, 8, 16, 32, 64, 128])
    args = parser.parse_args()
    print("k,reference_error_2,orthogonality_error_2,odd_weight_leakage_2,flag_leakage_2")
    for k in args.sizes:
        lift = staircase_lift(k)
        matrix = lift.matrix()
        output = lift.lifted_matrix()
        error = np.linalg.norm(matrix - staircase_a_matrix(k), ord=2)
        orthogonality = np.linalg.norm(matrix.T @ matrix - np.eye(k), ord=2)
        odd = np.linalg.norm(output[1::2], ord=2) if k > 1 else 0.0
        flag = np.linalg.norm(lift.prepared_columns()[k - 1, 1:])
        print(f"{k},{error:.6e},{orthogonality:.6e},{odd:.6e},{flag:.6e}")


if __name__ == "__main__":
    main()
