"""Circuit decompositions of A-matrices from the symmetric-group branching rule.

The public API is small; each module's docstring carries the math:

- `fourier.diagrams` — enumerating Young diagrams by addable-cell count
- `fourier.amatrix` — A-matrix construction and its Cauchy structure
- `fourier.decompositions` — Givens, cosine–sine, and Wei–Di factorizations
- `fourier.weyl` — Weyl-chamber invariants, leakiness, block-rotation form
- `fourier.circuits` — Qiskit realizations and compiler gate-count benchmarks
"""

from .amatrix import (
    CauchyForm,
    a_matrix,
    a_matrix_from_contents,
    a_matrix_generic4,
    a_matrix_generic8,
    a_matrix_symbolic,
    addable_contents,
    cauchy_form,
    random_content_a_matrix,
    removable_contents,
    staircase_a_matrix,
)
from .decompositions import (
    CSDecomposition,
    Givens,
    WeiDi,
    cs_butterfly,
    cs_factor,
    givens_factor,
    givens_reconstruct,
    parallel_depth,
    wei_di_fit,
)
from .diagrams import diagrams_with_addable_cells, partitions, staircase
from .weyl import (
    block_rotation_form,
    is_leaky,
    leakiness_rank,
    leaky_direction,
    weyl_coordinates,
)

__all__ = [
    "CauchyForm",
    "CSDecomposition",
    "Givens",
    "WeiDi",
    "a_matrix",
    "a_matrix_from_contents",
    "a_matrix_generic4",
    "a_matrix_generic8",
    "a_matrix_symbolic",
    "addable_contents",
    "block_rotation_form",
    "cauchy_form",
    "cs_factor",
    "cs_butterfly",
    "diagrams_with_addable_cells",
    "givens_factor",
    "givens_reconstruct",
    "is_leaky",
    "leakiness_rank",
    "leaky_direction",
    "parallel_depth",
    "partitions",
    "random_content_a_matrix",
    "removable_contents",
    "staircase",
    "staircase_a_matrix",
    "wei_di_fit",
    "weyl_coordinates",
]
