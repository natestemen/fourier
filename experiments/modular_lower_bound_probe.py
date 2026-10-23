"""Exact witnesses for a restricted modular streaming lower bound.

This is NOT an unrestricted lower bound for the S_n Fourier transform.
It lower-bounds the sum of separately implemented fresh-input A modules
with a specified local accuracy contract.  Whole-QFT accuracy alone does
not imply that contract, nor does it prohibit fusion, amortization, or a
different factorization.

Every-size family
-----------------
For N >= 35 choose the largest m with B(m) = 32m^2 + 3m <= N, and put
t = N - B(m).  Then 0 <= t <= 64m+34.  In the hard family from
sublinear_core_probe, fix every bit except r=1,...,floor(m/2)-1, and add t
boxes to its first row.  There are b(N)=max(0,floor(m/2)-1) variable bits.
All resulting partitions have exactly N boxes and Durfee side 4m.

The first row always had length 8m, independently of these bits.  Padding
therefore changes only the first Frobenius arm, by the fixed amount t.
Each variable bit r still changes only its disjoint eight-arm-word block
8r+1,...,8r+8; every leg, the Durfee word, and all other arms are fixed.
When t>0, the corner lists replace x_0=8m by 8m+t and 8m-1, and append
the removable corner 8m+t-1 at the top.  All interior x_(4r) survive.
The corner ordinal increases by one, but its row remains 8r+1.

The extra corner factors are independent of every variable bit.  Thus
flipping bit r from zero to one still multiplies its positive first-column
coefficient alpha by exactly 7/9.  Integer interlacing gives alpha>=1/L
with L=16m+t <= 80m+34.  The two ideal outputs have Euclidean separation
at least 2/(9L).  A circuit of arity at most q, with fresh zero scratch,
whose only shape-dependent input is this canonical record, must touch
every variable arm block when its clean complete-output error delta is
less than 1/(9L).  Otherwise the untouched block factors out and the same
remaining actual output is within delta of two ideal outputs separated
by more than 2 delta.  The gate count is at least ceil(b(N)/q).

Since L<=114m and N>=32m^2, the uniform contract

    delta <= 1/(200 sqrt(N))

strictly implies delta<1/(9L).  We retain the sharper exact threshold in
the returned witness summaries.  Also m=sqrt(N/32)+O(1), so this yields
Omega(sqrt(N)) gates at EVERY sufficiently large size, not just a sparse
subsequence.  The padded records fit the existing public register bounds:
4m<=floor(sqrt(M)) and k<=4m+2<=2floor(sqrt(M))+2 for all M>=N+1.

Constant-error strengthening
----------------------------
The uniform residue estimate in constant_error_local_probe sharpens the
selected coefficient bound to alpha_(4r)^2 >= 1/(32m).  Briefly, midpoint
residues are at least 1/(8m); each actual residue is their product with
four-node-block factors whose total is at least
(7/12)^2 (59/64) > 1/4.  At selected block r>=1, first-row padding multiplies
the residue by d(d+t-1)/[(d-1)(d+t)] >= 1, where d=16r.  Thus this stronger
estimate survives padding to every N, with the 7/9 own-bit ratio intact.

If U is the set of untouched disjoint arm blocks, fix all other bits and
average uniformly over U.  The circuit factors as identity on U tensor a
fixed circuit on the remaining input.  Project its output onto the fixed
accessed parameter labels and zero workspace.  Its selected-row amplitudes
v_r are independent of all bits in U.  Orthogonality and conditional
variance therefore imply

    delta^2 >= sum_{r in U} E|v_r-alpha_(4r)|^2
            >= sum_{r in U} E Var_{s_r}(alpha_(4r))
            >= |U|/(2592m).

Indeed the own-bit values are h_r and 7h_r/9, with h_r depending only on
the other bits, so their variance is h_r^2/81 >= 1/(2592m).  Projection
permits approximate, rather than exact, preservation of the accessed
parameter registers.  For delta<=1/144, at most floor(m/8) blocks can be
untouched.  Hence at least

    c(N) = max(0, floor(m/2)-1-floor(m/8))

blocks must be touched, giving ceil(c(N)/q) gates at constant local error.

Restricted modular theorem
--------------------------
Suppose a compiled streaming QFT contains a separately counted A module
for each outer size M=2,...,n and each prefix size N=0,...,M-1.  Assume:

* each module obeys a full-promise clean error contract delta<=1/144 for
  EVERY valid N-box canonical Frobenius input, with its active control
  fixed to one and matrix label fixed to the star column;
* apart from that shape register, star label and fixed controls, all its
  incoming workspace is a shape-independent fixed state (zero suffices);
  it cannot read a redundant external tableau/path/cache;
* its gate cost includes all preprocessing and cleanup; no gates are
  shared, fused or cancelled across the counted module boundaries.

Then the total number of arity-q gates is at least

    sum_{N=35}^{n-1} (n-N) ceil(c(N)/q)
      = n^(5/2)/(40 sqrt(2) q) + O(n^2) = Omega(n^(5/2)).

Indeed each prefix N occurs at M=N+1,...,n and
c(N)=3sqrt(N)/(32 sqrt(2))+O(1), while
sum (n-N)sqrt(N)=(4/15)n^(5/2)+O(n^(3/2)).
The stronger size-dependent accuracy delta<=1/(200sqrt(N)) recovers the
all-block count ceil(b(N)/q) and leading constant 1/(30sqrt(2)q).
Control inactivity on some executions cannot remove gates from a fixed
module.  Different modules can have different witness shapes: full-promise
correctness, not simultaneous reachability of these shapes on one tableau
path, is the premise.  This is why the sum is valid here and why it would
be invalid as an unconditional lower bound for the complete QFT.

The current independent clean-core streaming architecture satisfies these
interface assumptions when implemented as prescribed.  Its error budget
epsilon/[4n(n+1)], 0<epsilon<=1, satisfies the constant local error bound
for n>=6, and the stronger size-dependent bound for n>=14.  The statement
does not force an optimizing compiler to
preserve the architecture.  These routines verify exact classical witness
identities and integer counts; they are not quantum implementations.
"""

from fractions import Fraction
from itertools import product
from math import isqrt

from sublinear_core_probe import (
    contents_from_partition,
    exact_alpha_squared,
    frobenius_from_partition,
    hard_contents,
    partition_from_contents,
)


def base_size(m):
    """Return the size of the unpadded m-block family."""
    if not isinstance(m, int) or m < 1:
        raise ValueError("m must be a positive integer")
    return 32 * m * m + 3 * m


def family_parameters(size):
    """Return public parameters for the family at any size >= 35."""
    if not isinstance(size, int) or size < 35:
        raise ValueError("size must be an integer at least 35")
    m = (isqrt(128 * size + 9) - 3) // 64
    padding = size - base_size(m)
    return {
        "boxes": size,
        "m": m,
        "padding": padding,
        "bits": max(0, m // 2 - 1),
        "span": 16 * m + padding,
        "k": 4 * m + 1 + (padding > 0),
        "durfee": 4 * m,
    }


def padded_partition(size, bits):
    """Return the N-box shape; bit i selects original block r=i+1."""
    parameters = family_parameters(size)
    bits = tuple(bits)
    if len(bits) != parameters["bits"] or any(bit not in (0, 1) for bit in bits):
        raise ValueError("wrong number of zero/one variable bits")
    full_bits = (0,) + bits + (0,) * (parameters["m"] - len(bits) - 1)
    partition = partition_from_contents(*hard_contents(full_bits))
    return (partition[0] + parameters["padding"],) + partition[1:]


def selected_corner(size, bit):
    """Return the fixed output corner ordinal and row for a variable bit."""
    parameters = family_parameters(size)
    if not isinstance(bit, int) or not 0 <= bit < parameters["bits"]:
        raise ValueError("bit index must select a variable family bit")
    block = bit + 1
    ordinal = 4 * block + (parameters["padding"] > 0)
    return ordinal, 8 * block + 1


def padded_residue_ratio(size, bits, bit):
    """Compute the exact squared-coefficient ratio for a zero-to-one flip."""
    bits = tuple(bits)
    ordinal, _ = selected_corner(size, bit)
    before = padded_partition(size, bits)
    if bits[bit] != 0:
        raise ValueError("selected bit must initially be zero")
    after_bits = bits[:bit] + (1,) + bits[bit + 1:]
    after = padded_partition(size, after_bits)
    return (exact_alpha_squared(*contents_from_partition(after), ordinal)
            / exact_alpha_squared(*contents_from_partition(before), ordinal))


def verify_size(size):
    """Exhaustively check a small N-box family, including support and span."""
    parameters = family_parameters(size)
    m, bit_count = parameters["m"], parameters["bits"]
    flips = 0
    for bits in product((0, 1), repeat=bit_count):
        shape = padded_partition(size, bits)
        x, y = contents_from_partition(shape)
        d, arms, legs = frobenius_from_partition(shape)
        assert sum(shape) == size
        assert d == 4 * m
        assert len(x) == parameters["k"]
        assert x[0] - x[-1] == parameters["span"]
        assert sum(x) == sum(y)
        assert all(x[j] > y[j] > x[j + 1] for j in range(len(y)))
        assert sum(z * z for z in x) - sum(z * z for z in y) == 2 * size
        for bit, value in enumerate(bits):
            if value:
                continue
            changed_bits = bits[:bit] + (1,) + bits[bit + 1:]
            changed_shape = padded_partition(size, changed_bits)
            new_x, _ = contents_from_partition(changed_shape)
            new_d, new_arms, new_legs = frobenius_from_partition(changed_shape)
            ordinal, row = selected_corner(size, bit)
            assert x == new_x
            assert x[ordinal] == shape[row - 1] + 1 - row
            assert new_x[ordinal] == changed_shape[row - 1] + 1 - row
            assert (new_d, new_legs) == (d, legs)
            changed = {i for i in range(d) if arms[i] != new_arms[i]}
            block = bit + 1
            assert changed
            assert changed <= set(range(8 * block, 8 * block + 8))
            assert padded_residue_ratio(size, bits, bit) == Fraction(49, 81)
            assert exact_alpha_squared(x, y, ordinal) >= Fraction(1, parameters["span"] ** 2)
            assert exact_alpha_squared(x, y, ordinal) >= Fraction(1, 32 * m)
            flips += 1
    assert (9 * parameters["span"]) ** 2 < 200 ** 2 * size
    return {
        **parameters,
        "shapes": 2 ** bit_count,
        "checked_flips": flips,
        "strict_error_threshold": Fraction(1, 9 * parameters["span"]),
    }


def module_gate_lower_bound(size, arity=2):
    """Fresh-module bound at delta<=1/(200sqrt(size)), not for any QFT."""
    if not isinstance(size, int) or size < 0:
        raise ValueError("size must be a nonnegative integer")
    if not isinstance(arity, int) or arity < 1:
        raise ValueError("arity must be a positive integer")
    bits = family_parameters(size)["bits"] if size >= 35 else 0
    return (bits + arity - 1) // arity


def constant_error_module_gate_lower_bound(size, arity=2,
                                         error_squared=Fraction(1, 144 ** 2)):
    """Fresh-module bound from variance; default clean error is at most 1/144."""
    module_gate_lower_bound(size, arity)  # Validate shared parameters.
    error_squared = Fraction(error_squared)
    if error_squared < 0:
        raise ValueError("squared error must be nonnegative")
    if size < 35:
        return 0
    parameters = family_parameters(size)
    missed_bound = 2592 * parameters["m"] * error_squared
    missed = missed_bound.numerator // missed_bound.denominator
    touched = max(0, parameters["bits"] - missed)
    return (touched + arity - 1) // arity


def _streaming_sum(n, arity, module_bound):
    """Sum identical per-size bounds over the prescribed fresh-call grid."""
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    module_gate_lower_bound(0, arity)  # Validate arity even for small n.
    total = 0
    m = 1
    while base_size(m) < n:
        first = base_size(m)
        last = min(n - 1, base_size(m + 1) - 1)
        count = last - first + 1
        multiplicities = count * n - count * (first + last) // 2
        total += multiplicities * module_bound(first, arity)
        m += 1
    return total


def streaming_gate_lower_bound(n, arity=2):
    """Sum the size-dependent-accuracy modular bound in O(sqrt(n)) arithmetic."""
    return _streaming_sum(n, arity, module_gate_lower_bound)


def constant_error_streaming_gate_lower_bound(n, arity=2):
    """Restricted count at constant per-module clean error delta<=1/144."""
    return _streaming_sum(n, arity, constant_error_module_gate_lower_bound)


if __name__ == "__main__":
    for block_count in (4, 8, 12):
        for pad in (0, 1, 64 * block_count + 34):
            print(verify_size(base_size(block_count) + pad))
    for group_size in (10_000, 100_000, 1_000_000):
        bound = streaming_gate_lower_bound(group_size)
        print({"n": group_size, "restricted_gate_lower_bound": bound,
               "ratio_to_n_power_2_5": bound / group_size ** 2.5,
               "constant_error_restricted_gate_lower_bound":
               constant_error_streaming_gate_lower_bound(group_size)})
