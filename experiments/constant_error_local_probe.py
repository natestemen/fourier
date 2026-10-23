"""Constant-error strengthening of the fresh-input local access obstruction.

This is not a lower bound for unrestricted full-QFT circuits.  It applies to
a fixed clean circuit whose only parameter-dependent inputs are fresh,
explicit corner/Frobenius words, with bounded-qubit-arity gates and no free
QRAM or parameter-dependent advice.  It cannot be charged independently
at repeated calls if correlated workspace is retained between calls.

PROOF OF A UNIFORM RESIDUE LOWER BOUND
Let K=4m and compare the hard family from sublinear_core_probe to midpoint
nodes y_j=x_j-2.  Its midpoint residue at index i is

    b_i = [binom(2i,i)/4^i] [binom(2(K-i),K-i)/4^(K-i)].

The elementary inequality binom(2j,j)/4^j >= 1/(2 sqrt(j)), j>=1,
follows inductively because (2j+1)^2 >= 4j(j+1).  Hence b_i >= 1/(2K)
at interior indices, and this also holds at the endpoints.

At selected index i=4r, block b, with t=b-r, multiplies the midpoint
residue by one of

    P_0(t) = 1 - 5/[(16t+10)(16t+14)],
    P_1(t) = 1 - 5/[(16t+2)(16t+6)].

Both lie in (0,1].  For t=0,-1 each is at least 7/12.  On the two
remaining rays t=j and t=-j-1, j>=1, each deficit is at most
5/(256 j^2).  The product of tail factors is at least one minus the
sum of deficits, hence at least 59/64, using sum_{j>=1} j^-2 <= 2.
Missing factors can only increase this bound.  Thus every hard record has

    alpha_(4r)^2 >= b_(4r) (7/12)^2 (59/64) > 1/(32m).

CONSTANT-ERROR ACCESS ARGUMENT
Flipping s_r changes alpha_(4r) from h_r to (7/9)h_r, with h_r depending
only on the other bits.  Conditional variance for uniform s_r is h_r^2/81,
at least 1/(2592m).  If U is a set of untouched disjoint input blocks,
fix the accessed bits and average over U.  Project actual output onto the
unchanged accessed parameter labels and zero workspace.  Its amplitude
v_r on selected output row r is independent of every untouched bit.
Orthogonality and total variance give

    delta^2 >= sum_(r in U) E |v_r-alpha_(4r)|^2 >= |U|/(2592m).

This projection argument allows approximate preservation of accessed
parameter registers; exact preservation by the implemented circuit is not
assumed.  For delta <= 1/144 at most floor(m/8) blocks can be untouched.

PADDING TO EVERY LARGE BOX COUNT
The unpadded size is N_m=32m^2+3m.  Add p boxes to the first row, whose
old width is W=8m.  At unchanged selected content z=x_(4r), r>=1,
put d=W-z=16r.  The residue multiplier is

    R = d(d+p-1)/[(d-1)(d+p)] = 1+p/[(d-1)(d+p)] >= 1.

This holds whether a new corner is created or not; canceled first-row
corner factors give the same expression.  It is independent of all bits,
so the 7/9 flip ratio is unchanged.  Vary only 1<=r<floor(m/2).  Their
disjoint eight-arm-word blocks remain untouched by padding; only the first
arm acquires the common addition p, while Durfee side and all legs stay
fixed.  Output row r is still 8r+1.  There are floor(m/2)-1 variable bits.
For delta<=1/144 and m>=16, at least m/4 blocks must be touched, so a
q-qubit gate circuit uses at least m/(4q) gates.  For any sufficiently
large n choose maximal m with N_m<=n; then p=n-N_m<64m+35 and
m=Theta(sqrt(n)).  This gives a constant-error Omega(sqrt(n)) fresh
Frobenius-controlled local lower bound for every sufficiently large n.

These routines are exact reference checks of identities used in the
proof, not quantum circuits, nor substitutes for the uniform proof above.
"""

from fractions import Fraction
from math import comb, isqrt, prod

from sublinear_core_probe import hard_contents, partition_from_contents


def midpoint_residue(m, index):
    """Exact transition residue for the all-midpoint reference diagram."""
    if m < 1 or not 0 <= index <= 4 * m:
        raise ValueError("expected m>=1 and an index in [0,4m]")
    k = 4 * m
    return Fraction(comb(2 * index, index) * comb(2 * (k - index), k - index),
                    4 ** k)


def block_ratio(relative_block, bit):
    """One four-node block's ratio to the midpoint residue at x_(4r)."""
    if bit not in (0, 1):
        raise ValueError("expected a zero/one bit")
    t = relative_block
    denominator = ((16 * t + 10) * (16 * t + 14) if bit == 0
                   else (16 * t + 2) * (16 * t + 6))
    return 1 - Fraction(5, denominator)


def residue_ratio_to_midpoints(bits, selected_block):
    """Exact product identity; no numerical square roots are used."""
    bits = tuple(bits)
    if not bits or not 0 <= selected_block < len(bits):
        raise ValueError("expected nonempty bits and a valid selected block")
    return prod(block_ratio(b - selected_block, bit)
                for b, bit in enumerate(bits))


def uniform_ratio_bound():
    """A rational lower bound greater than 1/4, uniform in all bits and m."""
    return Fraction(7, 12) ** 2 * Fraction(59, 64)


def padded_partition(bits, padding):
    """Classical verifier: append a fixed number of cells to the first row."""
    if padding < 0:
        raise ValueError("padding must be nonnegative")
    shape = partition_from_contents(*hard_contents(bits))
    return (shape[0] + padding,) + shape[1:]


def padding_residue_ratio(selected_block, padding):
    """Exact multiplier at a non-top selected addable content."""
    if selected_block < 1 or padding < 0:
        raise ValueError("expected a positive block and nonnegative padding")
    d = 16 * selected_block
    return Fraction(d * (d + padding - 1), (d - 1) * (d + padding))


def size_parameters(boxes):
    """Maximal hard-family size below boxes, with first-row padding."""
    if boxes < 35:
        raise ValueError("expected at least 35 boxes")
    m = (isqrt(9 + 128 * boxes) - 3) // 64
    while 32 * (m + 1) ** 2 + 3 * (m + 1) <= boxes:
        m += 1
    while 32 * m * m + 3 * m > boxes:
        m -= 1
    return m, boxes - (32 * m * m + 3 * m)


def forced_touched_blocks(m, error_squared=Fraction(1, 144 ** 2)):
    """Integer bound for the padded Frobenius family with bit zero fixed."""
    if m < 2 or error_squared < 0:
        raise ValueError("expected m>=2 and nonnegative squared error")
    missed_bound = 2592 * m * Fraction(error_squared)
    return max(0, m // 2 - 1 - missed_bound.numerator // missed_bound.denominator)
