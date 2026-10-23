"""Exact witnesses against a sublinear *fresh-input* coherent A primitive.

This is a gate-access obstruction, not a lower bound for the complete S_n
Fourier transform or for an amortized algorithm retaining parameter-dependent
workspace.  It does not assert a quantum query lower bound with free QRAM.

For m >= 1, k = 4m+1 and s in {0,1}^m, define

    x_i = 8m - 4i,
    y_j = x_j - a[j mod 4] + s[j//4] d[j mod 4],
    a = (2,2,1,3),  d = (1,-1,-1,1).

These are genuine Young corner lists.  They strictly interlace and have
sum(x) = sum(y) = 0.  Every member has N = 32m^2 + 3m boxes and content
span L = 16m.  Flipping one bit s_r from zero to one leaves the sum of
removable contents and their squared sum unchanged: the relevant adjacent
pair spacings are 4 and 6, so the squared-sum changes are +10 and -10.

For the first-column residue at x_(4r), that flip multiplies alpha^2 by

    (1/2) (7/6) (10/9) (14/15) = 49/81.

Thus alpha itself changes by a factor 7/9, independently of every other
bit.  The integer-interlacing bound alpha >= 1/L gives output-vector
separation >= 2/(9L), already on the input basis vector |0>.

Consequently, a fixed bounded-arity circuit that receives the corner words
as its only parameter-dependent input, starts all work in zero, preserves
the parameter words and implements A with clean error delta < 1/(9L)
must access at least one word from each disjoint four-word block.  Otherwise
the two parameter records in an unaccessed block induce the same output
on all remaining registers, contradicting the triangle inequality.  A
gate touching at most q qubits can access at most q blocks, so there are
at least ceil(m/q) gates.  The result includes any fresh preprocessing and
its cleanup, but cannot be multiplied by the number of calls if dependent
workspace is retained between calls.

More formally, let B_r be the union of the four removable-node words in
block r.  If no gate meets B_r, the circuit factors as I_(B_r) tensor V.
Take two promised basis parameter records differing only in s_r, and
strip their unchanged parameter labels from the two ideal output states.
The actual remaining output is the same vector v for both records, so
norm(A_s|0> - A_t|0>) <= norm(A_s|0> - v) + norm(v - A_t|0>) <= 2 delta.
The exact positive first-column coefficient above makes the left side at
least 2/(9L).  This contradiction applies when delta < 1/(9L), not at the
threshold.  Each bounded-arity gate meets at most q distinct B_r.  It
therefore implies ceil(m/q) gates, including ceil(m/2) for one-/two-qubit
gate sets.  No causal-cone or numerical-conditioning assumption is used.

The first-column convention is positive, as in the project's definition
of A; the 7/9 factor is not an undetermined eigenvector phase.  Translating
all nodes by 8m gives legal nonnegative words in [0,16m] and leaves A and
the disjoint-block argument unchanged.  Padding to K with k <= K < 2k
likewise leaves the witness unchanged, yielding Omega(K) on this sequence.
Here N is fixed across all 2^m records, k = Theta(sqrt(N)), and the needed
accuracy is only delta < 1/(144m) = Theta(N^(-1/2)).

The obstruction is not confined to the redundant corner-list encoding.
Restrict variable bits to r < floor(m/2) and fix the other bits at zero.
Every shape has Durfee side d = 4m.  In row block r, with W=8(m-r), the
eight row lengths are

    bit 0: (W,W,W-2,W-2,W-4,W-7,W-7,W-7),
    bit 1: (W,W-3,W-3,W-3,W-4,W-4,W-6,W-6).

These two blocks have the same height and area, and their changed cells
lie strictly to the right of the d-th column when r < floor(m/2).
Therefore bit r affects only the disjoint eight-arm-word block
(a_(8r+1),...,a_(8r+8)) in the canonical Frobenius record.  All d and b
words are fixed.  The same support proof gives ceil(floor(m/2)/q) gates
for a fresh Frobenius-controlled implementation, again Omega(k).  The
addable row corresponding to x_(4r) is always 8r+1, independently of the
bits; thus this witness also works for the star-column isometry with
output encoded by row number rather than addable-corner ordinal.

All routines below are exact classical verification helpers, not a quantum
implementation.  In particular, constructing a complete partition here
is not claimed to be a fast reversible shape conversion.
"""

from fractions import Fraction
from itertools import product
from math import prod


def hard_contents(bits):
    """Return exact integer addable/removable contents for a bit tuple."""
    bits = tuple(bits)
    if not bits or any(bit not in (0, 1) for bit in bits):
        raise ValueError("expected a nonempty tuple of zero/one bits")
    m = len(bits)
    offsets = (2, 2, 1, 3)
    changes = (1, -1, -1, 1)
    x = tuple(8 * m - 4 * i for i in range(4 * m + 1))
    y = tuple(x[j] - offsets[j % 4] + bits[j // 4] * changes[j % 4]
              for j in range(4 * m))
    return x, y


def partition_from_contents(x, y):
    """Reconstruct row blocks from centered, strictly interlaced contents."""
    if len(x) != len(y) + 1 or sum(x) != sum(y):
        raise ValueError("corner counts or centering are invalid")
    if any(not x[j] > y[j] > x[j + 1] for j in range(len(y))):
        raise ValueError("contents must strictly interlace")
    rows = []
    height = 0
    for upper, lower in zip(x, y):
        width = upper + height
        multiplicity = upper - lower
        if width <= 0 or rows and width >= rows[-1]:
            raise ValueError("contents do not form positive decreasing row blocks")
        rows.extend([width] * multiplicity)
        height += multiplicity
    if x[-1] != -height:
        raise ValueError("bottom corner does not match the row count")
    return tuple(rows)


def contents_from_partition(partition):
    """Independent direct corner enumeration of a nonempty partition."""
    if not partition or any(a < b for a, b in zip(partition, partition[1:])):
        raise ValueError("expected a nonempty nonincreasing partition")
    x, y = [], []
    for r, width in enumerate(partition, start=1):
        if r == 1 or partition[r - 2] > width:
            x.append(width + 1 - r)
        if r == len(partition) or partition[r] < width:
            y.append(width - r)
    x.append(-len(partition))
    return tuple(x), tuple(y)


def frobenius_from_partition(partition):
    """Return canonical (d, arms, legs), for independent support checks."""
    d = sum(width >= i for i, width in enumerate(partition, start=1))
    arms = tuple(partition[i - 1] - i for i in range(1, d + 1))
    legs = tuple(sum(width >= i for width in partition) - i for i in range(1, d + 1))
    return d, arms, legs


def exact_alpha_squared(x, y, i):
    """Compute a transition probability with rational arithmetic."""
    return Fraction(prod(abs(x[i] - z) for z in y),
                    prod(abs(x[i] - z) for h, z in enumerate(x) if h != i))


def changed_residue_ratio(bits, block):
    """Return alpha_(4r)^2 after/before flipping zero bit r to one."""
    bits = tuple(bits)
    if not 0 <= block < len(bits) or bits[block] != 0:
        raise ValueError("block must select a zero bit")
    changed = bits[:block] + (1,) + bits[block + 1:]
    x, y = hard_contents(bits)
    new_x, new_y = hard_contents(changed)
    return (exact_alpha_squared(new_x, new_y, 4 * block)
            / exact_alpha_squared(x, y, 4 * block))


def verify_family(m):
    """Exhaustively certify a small family and return a compact summary."""
    expected_size = 32 * m * m + 3 * m
    flips = 0
    for bits in product((0, 1), repeat=m):
        x, y = hard_contents(bits)
        partition = partition_from_contents(x, y)
        assert contents_from_partition(partition) == (x, y)
        assert frobenius_from_partition(partition)[0] == 4 * m
        assert sum(partition) == expected_size
        assert sum(v * v for v in x) - sum(v * v for v in y) == 2 * expected_size
        for r, bit in enumerate(bits):
            if bit == 0:
                assert changed_residue_ratio(bits, r) == Fraction(49, 81)
                assert exact_alpha_squared(x, y, 4 * r) >= Fraction(1, (16 * m) ** 2)
                flips += 1
    return {"bits": m, "k": 4 * m + 1, "boxes": expected_size,
            "span": 16 * m, "shapes": 2 ** m, "checked_flips": flips}


if __name__ == "__main__":
    for bit_count in range(1, 7):
        print(verify_family(bit_count))
