from yungdiagram import YoungDiagram
from compute_matrix import A_matrix
import numpy as np

diagram = YoungDiagram([2, 1])
A = A_matrix(diagram)

print(f"Partition (2, 1) — size {diagram.size}")
print(f"Addable cells:   {diagram.addable_cells()}")
print(f"Removable cells: {diagram.removable_cells()}")
print(f"\nA matrix ({A.shape[0]}x{A.shape[1]}):")
print(np.array2string(A, precision=6, suppress_small=True))
