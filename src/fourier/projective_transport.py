"""Transport A-matrix circuits between projectively equivalent contents.

Choose any removable node z=y[q], and change coordinates by t -> 1/(t-z).
The old column at infinity becomes the finite removable node 0; the column
at z becomes the new distinguished constant column. After sorting the new
nodes, the new A-matrix equals the old A-matrix up to signed permutations.
Thus a circuit for one member of this family works for every member, without
computing any new rotation angles.

Proof: the absolute value of the old z-column is the new constant column.
Writing u=1/(x-z), the ratio of old column infinity to the z-column is
1/(beta[q]*u). The ratio of old finite column j to the z-column is
beta[j]/(beta[q]*(z-y[j])*(u-1/(y[j]-z))). Row signs make the constant
column positive; column signs make each Cauchy scale positive. Orthogonality
and normalization uniquely fix these positive scales, giving exactly the
A-matrix for the transformed interlaced nodes.

The permutations consist of two reversed intervals and the moved constant
column. They depend only on (k,q), not on the node values, and can be applied
to a binary label by comparisons and subtraction. Coherent circuit overhead
is therefore poly(log k); it does not require a k-entry permutation lookup.
Recognizing whether arbitrary input contents belong to a fast family is a
separate task. This module constructs classical metadata in O(k) operations;
``matrix`` is only a dense numerical validation helper.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .structured import _validate_contents

__all__ = ["ProjectiveTransport", "transport_at_removable"]

FloatArray = npt.NDArray[np.float64]
IndexArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class ProjectiveTransport:
    """Signed row/column permutations and their transformed contents."""

    addable: FloatArray
    removable: FloatArray
    row_order: IndexArray
    column_order: IndexArray
    row_signs: FloatArray
    column_signs: FloatArray
    pole_index: int

    def matrix(self, original: FloatArray) -> FloatArray:
        """Apply the identity to a dense matrix, for validation only."""
        original = np.asarray(original)
        k = len(self.addable)
        if original.shape != (k, k):
            raise ValueError("original must be a k-by-k matrix")
        return (
            self.row_signs[:, None]
            * original[np.ix_(self.row_order, self.column_order)]
            * self.column_signs[None, :]
        )


def transport_at_removable(
    ac: FloatArray, rc: FloatArray, pole_index: int
) -> ProjectiveTransport:
    """Move a removable node to infinity, preserving A up to signed shuffles.

    Contents must be finite and strictly interlaced in descending order.
    The returned coordinates are real; positive affine rescaling leaves
    their A-matrix unchanged. Rational input coordinates admit a common
    integer scaling followed by the usual zero-content-sum translation,
    producing the corner contents of a genuine Young diagram.
    """
    ac, rc = _validate_contents(ac, rc)
    if not np.all(np.isfinite(ac)) or not np.all(np.isfinite(rc)):
        raise ValueError("contents must be finite")
    if (
        not isinstance(pole_index, (int, np.integer))
        or not 0 <= pole_index < len(rc)
    ):
        raise ValueError("pole_index must index a removable content")
    q, k = int(pole_index), len(ac)
    z = rc[q]
    # Reciprocal is decreasing on either side of its pole: reverse each side.
    rows = np.r_[np.arange(q, -1, -1), np.arange(k - 1, q, -1)]
    columns = np.r_[q + 1, np.arange(q, -1, -1), np.arange(k - 1, q + 1, -1)]
    new_ac = 1.0 / (ac[rows] - z)
    new_rc = np.zeros(k - 1)
    old_tail = columns[1:]
    finite = old_tail != 0
    new_rc[finite] = 1.0 / (rc[old_tail[finite] - 1] - z)
    _validate_contents(new_ac, new_rc)
    if not np.all(np.isfinite(new_ac)) or not np.all(np.isfinite(new_rc)):
        raise ValueError("reciprocal coordinates exceed floating-point range")
    return ProjectiveTransport(
        new_ac,
        new_rc,
        rows,
        columns,
        np.r_[np.ones(q + 1), -np.ones(k - q - 1)],
        np.r_[1.0, -np.ones(q), np.ones(k - q - 1)],
        q,
    )
