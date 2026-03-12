#!/usr/bin/env python3
"""Compute and print the A matrix for the partition (2, 1)."""
from __future__ import annotations

import numpy as np

from compute_matrix import A_matrix
from yungdiagram import YoungDiagram


def main() -> None:
    diagram = YoungDiagram([2, 1])
    mat = A_matrix(diagram)

    # Print with stable, readable formatting.
    print("Partition (2, 1) A matrix:")
    print(np.array2string(np.array(mat, dtype=float), precision=8, suppress_small=True))


if __name__ == "__main__":
    main()
