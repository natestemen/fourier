"""Checks for prefix amortization and one-box A-matrix transport.

These are classical exact/numerical probes, NOT a sublinear coherent circuit.
In particular, a low-rank matrix representation does not supply the controlled
state preparations needed to implement its dense vectors without QRAM.

One-box identity: if c is the added box content, the Young boundary resolvent
changes by f_new(z)/f_old(z) = (z-c)^2 / ((z-c)^2-1).  On persistent nodes,
put rho(z)=sqrt(1-(z-c)^(-2)).  Then

  A_new - A_old = diag(1/rho(x))
      [A_old diag(rho(y)) - diag(rho(x)) A_old]

on the common-row/common-tail-column block.  Persistent nodes have distance
at least two from c.  Truncating the binomial series for rho after p terms
gives a rank-at-most-2p divided difference, with spectral error at most
4/(3 sqrt(3)) * 4^(-p).  Copying the exceptional rows and columns exactly adds
only a constant to the rank.  The implementation below handles all Young
edges, including changing corner counts, by content alignment and padding.

If E_p approximates B-A with norm error eps<1 and rank r, R=E_p A.T has
the same rank and approximates B A.T-I.  The polar factor V of I+R is a true
orthogonal correction with ||V-B A.T||<=2 eps.  Indeed all singular values
of I+R lie within eps of one, hence ||V-(I+R)||<=eps.  Further, V is exactly
identity on (ran R + ran R.T)^perp, a codimension-at-most-2r subspace.  The
SVD below verifies this existence statement; it is NOT a fast circuit.

Exact constant rank is impossible even for the top-row staircase family.
After content translation, its persistent block differs by nonzero diagonal
scalings from L(x,y)=(rho(y)-rho(x))/(x-y), with x=-2,-4,... and y=-3,-5,... .
If r selected rows were dependent on more than 2r columns, clearing the r
simple poles would give rho(y) Q(y)=P(y), deg(P),deg(Q)<=r-1. Squaring gives
y^2 P(y)^2-(y^2-1)Q(y)^2=0 at more than 2r points, hence identically. This
contradicts the odd valuation at y=1 unless both polynomials vanish. The
partial fractions then force every dependence coefficient to vanish. Thus
rank(B-A)>=floor((k-3)/2), and rank(B A.T-I) has the same lower bound. Tiny
singular values must not be mistaken for exact zeroes.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isqrt, sqrt

import numpy as np

if __package__:
    from .streaming_induction import (
        InductionOutput, add_row, addable_rows, removable_rows,
        shape_of_word, streaming_column,
    )
else:
    from streaming_induction import (
        InductionOutput, add_row, addable_rows, removable_rows,
        shape_of_word, streaming_column,
    )
from fourier.amatrix import _a_matrix_from_contents_log


def corner_contents(shape: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    x = tuple((shape[r - 1] if r <= len(shape) else 0) + 1 - r
              for r in addable_rows(shape))
    y = tuple(shape[r - 1] - r for r in removable_rows(shape))
    return x, y


def resolvent(shape: tuple[int, ...], z: int) -> Fraction:
    x, y = corner_contents(shape)
    value = Fraction(1)
    for node in y:
        value *= z - node
    for node in x:
        value /= z - node
    return value


def alpha_squared(shape: tuple[int, ...], x: int) -> Fraction:
    adds, removes = corner_contents(shape)
    if x not in adds:
        raise ValueError("not an addable content")
    result = Fraction(1)
    for y in removes:
        result *= abs(x - y)
    for other in adds:
        if other != x:
            result /= abs(x - other)
    return result


@dataclass(frozen=True)
class PrefixWitness:
    n: int
    staircase_rows: int
    staircase_size: int
    word: tuple[int, ...]
    corner_counts: tuple[int, ...]

    @property
    def total_corner_records(self) -> int:
        return sum(self.corner_counts)

    @property
    def persistent_tail_records(self) -> int:
        return (self.n - self.staircase_size + 1) * (self.staircase_rows + 1)


def long_many_corner_path(n: int) -> PrefixWitness:
    """A path with >=sqrt(n)/constant corners for >=n/2 prefixes.

    Build the staircase d,...,1, with d(d+1)<=n, then add to row one.
    Every shape from the completed staircase onward has d+1 addable corners.
    Thus materializing all original-prefix corner lists takes Omega(n^1.5)
    words on this family, even before asking for coherent access or cleanup.
    """
    if n < 2:
        raise ValueError("n must be at least two")
    d = (isqrt(1 + 4 * n) - 1) // 2
    staircase_size = d * (d + 1) // 2
    word = tuple(r for r in range(1, d + 1) for _ in range(d - r + 1))
    word += (1,) * (n - staircase_size)
    shape: tuple[int, ...] = ()
    counts = []
    for row in word:
        shape = add_row(shape, row)
        counts.append(len(addable_rows(shape)))
    return PrefixWitness(n, d, staircase_size, word, tuple(counts))


def cache_cleanup_collision():
    """Two distinct original-prefix caches occur at the same output label.

    A cleanup controlled by the output computational label cannot unitarily
    map both distinct cache basis states to zero while preserving that label.
    This does NOT prohibit a more general cleanup that mixes output labels.
    """
    left = (1, 2, 1)
    right = (1, 1, 2)
    output = InductionOutput((3, 1), (1, 2, 1, 1))
    return {
        "left_input": left,
        "right_input": right,
        "left_prefix": shape_of_word(left[:2]),
        "right_prefix": shape_of_word(right[:2]),
        "output": output,
        "left_amplitude": streaming_column(1, left)[output],
        "right_amplitude": streaming_column(1, right)[output],
    }


@dataclass(frozen=True)
class AlignedTransport:
    old: np.ndarray
    new: np.ndarray
    approximation: np.ndarray
    bound: float
    order: int
    common_rows: int
    common_columns: int
    old_size: int
    new_size: int

    @property
    def exceptional_rank_bound(self):
        return 2 * len(self.old) - self.common_rows - self.common_columns

    @property
    def rank_bound(self):
        return min(len(self.old), 2 * self.order + self.exceptional_rank_bound)


def aligned_box_transport(shape: tuple[int, ...], row: int, order: int) -> AlignedTransport:
    """Handle all three possible changes in corner count: -1, 0, and +1.

    Pad both A matrices to K=max(k,k') with identity. Independently permute
    their rows to put persistent addable contents first, and their columns to
    put persistent removable contents first. Append the constant column and
    other columns afterward.  Padding indices are always exceptional.  This
    alignment is essential: comparing rank-indexed matrices without matching
    contents can introduce a long permutation that obscures the low-rank fact.

    If q in {0,1,2} new addable corners appear, k'=k-1+q. There are k-1
    common rows and k-3+q common tail columns. The counts of exceptional rows
    plus columns are respectively 4,3,4, hence always at most four.

    The approximation concerns B-A; multiplying on the right by A.T gives
    the same bound for B A.T-I. No SVD is used to construct the approximation.
    Its coefficients are cheap here only because this is a classical probe.
    """
    if order < 0:
        raise ValueError("order must be nonnegative")
    successor = add_row(shape, row)
    old_x, old_y = corner_contents(shape)
    new_x, new_y = corner_contents(successor)
    c = float((shape[row - 1] if row <= len(shape) else 0) + 1 - row)
    common_x = tuple(sorted(set(old_x) & set(new_x), reverse=True))
    common_y = tuple(sorted(set(old_y) & set(new_y), reverse=True))
    rx, cy = len(common_x), len(common_y)
    k, next_k = len(old_x), len(new_x)
    capacity = max(k, next_k)

    def align(x, y):
        original = _a_matrix_from_contents_log(np.array(x, float), np.array(y, float))
        matrix = np.eye(capacity)
        matrix[:len(x), :len(x)] = original
        row_order = ([x.index(node) for node in common_x]
                     + [j for j, node in enumerate(x) if node not in common_x]
                     + list(range(len(x), capacity)))
        column_order = ([1 + y.index(node) for node in common_y] + [0]
                        + [j + 1 for j, node in enumerate(y) if node not in common_y]
                        + list(range(len(x), capacity)))
        return original, matrix[np.ix_(row_order, column_order)]

    original_a, a = align(old_x, old_y)
    _, b = align(new_x, new_y)
    approximation = b - a
    approximation[:rx, :cy] = 0
    if rx and cy and order:
        u = np.array(common_x, float) - c
        v = np.array(common_y, float) - c
        rho_x = np.sqrt(1 - 1 / u**2)
        # Column cy is the aligned old constant column.
        alpha = a[:rx, cy]
        beta = np.array([original_a[0, old_y.index(y) + 1] * (old_x[0] - y)
                         / original_a[0, 0] for y in common_y])
        coefficient = Fraction(1)
        for m in range(1, order + 1):
            coefficient *= Fraction(2 * m - 3, 2 * m)
            # coefficient = (-1)^m binom(1/2,m)
            for power in range(1, 2 * m + 1):
                left = alpha / rho_x / u**power
                right = beta / v**(2 * m + 1 - power)
                approximation[:rx, :cy] += float(coefficient) * np.outer(left, right)
    bound = 4 / (3 * sqrt(3)) * 4.0**(-order)
    return AlignedTransport(a, b, approximation, bound, order, rx, cy, k, next_k)


def staircase_transport(k: int, order: int):
    """Compatibility wrapper; top-row staircase addition keeps the size k."""
    if k < 2:
        raise ValueError("require k>=2")
    result = aligned_box_transport(tuple(range(k - 1, 0, -1)), 1, order)
    return result.old, result.new, result.approximation, result.bound


def orthogonal_transport_correction(result: AlignedTransport):
    """SVD verification of a small-subspace *existence* result, not synthesis."""
    update = result.approximation @ result.old.T
    candidate = np.eye(len(result.old)) + update
    left, _, right_t = np.linalg.svd(candidate)
    correction = left @ right_t
    return correction, update


def main():
    for n in (16, 64, 256, 1024):
        witness = long_many_corner_path(n)
        print(f"n={n} corners={witness.staircase_rows+1} "
              f"tail_records={witness.persistent_tail_records} "
              f"all_records={witness.total_corner_records}")
    print("cache collision:", cache_cleanup_collision())
    for k in (4, 8, 16, 32, 64, 128):
        a, b, approximation, bound = staircase_transport(k, 5)
        singular_values = np.linalg.svd(b @ a.T - np.eye(k), compute_uv=False)
        error = np.linalg.norm(b - a - approximation, ord=2)
        print(f"k={k} rank_1e-6={sum(singular_values > 1e-6)} "
              f"rank_1e-12={sum(singular_values > 1e-12)} "
              f"order5_error={error:.3e} bound={bound:.3e}")


if __name__ == "__main__":
    main()
