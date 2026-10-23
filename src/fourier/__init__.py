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
from .projective_transport import ProjectiveTransport, transport_at_removable
from .staircase_lift import KrawtchoukLift, staircase_lift, symmetric_hadamard
from .structured import (
    CayleyFactorization,
    JacobiFactorization,
    ProjectiveCauchyForm,
    SymmetricContentFactorization,
    arrowhead_from_contents,
    cayley_factorization,
    jacobi_factorization_from_contents,
    orthogonal_polynomial_transform,
    projective_cauchy_from_contents,
    staircase_jacobi_coefficients,
    symmetric_content_factorization,
)
from .weyl import (
    block_rotation_form,
    is_leaky,
    leakiness_rank,
    leaky_direction,
    weyl_coordinates,
)

__all__ = [
    "CauchyForm",
    "CayleyFactorization",
    "CSDecomposition",
    "Givens",
    "JacobiFactorization",
    "KrawtchoukLift",
    "ProjectiveCauchyForm",
    "ProjectiveTransport",
    "SymmetricContentFactorization",
    "WeiDi",
    "a_matrix",
    "a_matrix_from_contents",
    "a_matrix_generic4",
    "a_matrix_generic8",
    "a_matrix_symbolic",
    "addable_contents",
    "arrowhead_from_contents",
    "block_rotation_form",
    "cauchy_form",
    "cayley_factorization",
    "cs_factor",
    "cs_butterfly",
    "diagrams_with_addable_cells",
    "givens_factor",
    "givens_reconstruct",
    "is_leaky",
    "jacobi_factorization_from_contents",
    "leakiness_rank",
    "leaky_direction",
    "parallel_depth",
    "orthogonal_polynomial_transform",
    "projective_cauchy_from_contents",
    "partitions",
    "random_content_a_matrix",
    "removable_contents",
    "staircase",
    "staircase_a_matrix",
    "staircase_jacobi_coefficients",
    "staircase_lift",
    "symmetric_content_factorization",
    "symmetric_hadamard",
    "transport_at_removable",
    "wei_di_fit",
    "weyl_coordinates",
]
