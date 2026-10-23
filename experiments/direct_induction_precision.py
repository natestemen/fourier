r"""Finite-precision certificate for direct-induction branching states.

This module is an exact-rational audit, not a quantum-circuit compiler.  It
supplies the missing elementary arithmetic lemma behind the direct-induction
``O(n**(5/2) polylog(n/epsilon))`` route in ``alternative_qft_probe.py``.

Let integer contents strictly interlace as

    x_0 > y_0 > x_1 > ... > y_{k-2} > x_{k-1}

and put

    p_i = prod_j |x_i-y_j| / prod_{h != i} |x_i-x_h|.

The important observation is the bounded-factor identity

    p_i = prod_{j<i} (y_j-x_i)/(x_j-x_i)
          prod_{j>=i} (x_i-y_j)/(x_i-x_{j+1}).                 (1)

Every factor in (1) lies strictly between zero and one.  Starting with the
``b``-bit dyadic integer ``N=2**b``, replace ``N`` after each factor ``a/d``
by ``floor(N*a/d)``.  If ``p_t`` is the exact partial product and
``q_t=N_t/2**b``, induction gives

    0 <= p_t-q_t < t*2**(-b),

because multiplication by the next factor contracts the old error and the
directed rounding contributes less than one ulp.  Thus no logarithm,
exponential, exponentially large product, or condition-number assumption is
needed.

Writing ``n_i`` for the final integers, ``T=sum_i n_i`` and
``q_i=n_i/T``, one has, whenever ``T>0``,

    sum_i |p_i-q_i| <= 2*k*(k-1)*2**(-b),
    ||sum_i (sqrt(p_i)-sqrt(q_i)) |i>||_2
        <= sqrt(2*k*(k-1)*2**(-b)).                            (2)

Moreover, if ``L=x_0-x_{k-1}``, telescoping the two sides of (1) gives
``p_i >= 1/L**2``.  Taking

    b >= log2(2*(k-1)*L**2)

keeps every genuine leaf positive.  Padded leaves remain exactly zero, so a
fixed power-of-two binary preparation tree has only three totalized cases:
positive siblings, a zero child (an exact endpoint rotation), or a zero
parent (an unreachable identity).  The positive-sibling angle

    theta = atan2(sqrt(R), sqrt(L))

is certified below using fixed-step integer square root and rational atan
series.  Range reduction makes every series argument at most 1/2, so the
number of terms is linear in the requested bit precision.  Angle scratch is
computed, used by a variable rotation, and uncomputed at every fixed tree
node.

For a public capacity ``K=O(sqrt(M))``, the masked factor schedule has
``K(K-1)=O(M)`` rational updates.  Each update acts on

    O(log M + log(1/eta))

bits and uses fixed multiplication/division.  Tree sums and rotations cost
``O(K polylog(M/eta))`` more.  Computing each product into scratch, copying
its final word, and reversing the recurrence uses ``O(K*b)`` parameter bits;
the surrounding tableau circuit already uses ``O(M polylog(M/eta))`` bits.
All parameter and angle registers can be erased from the unchanged content
record after state preparation, coherently across shape sectors.

The exact Fraction arctangent series below is a reference enclosure, not the
bounded-word reversible implementation: its rational denominators may grow.
The theorem uses fixed-point Horner evaluation with rounding at each step and
O(log c) guard bits for c requested fractional bits. Likewise, a c-bit angle
word means c controlled fixed rotations, each synthesized to its allocated
share of the error; it is not an uncharged variable-angle primitive.

Young-generator audit.  A nontrivial seminormal two-tableau block is

    [[1/d, sqrt(1-1/d**2)], [sqrt(1-1/d**2), -1/d]],

where ``d`` is a nonzero integer axial distance with ``|d|<=M``.  It is the
reflection ``H(theta)`` with ``cos(theta)=1/d``.  Approximating ``theta`` by
``tau`` changes the block in operator norm by at most ``tau``.  Endpoint
values ``d=+/-1`` are exact signs.  Hence all ``M-1`` threshold-controlled
generators in one direct-induction ``W`` have error at most ``M*tau``;
``O(log M + log(1/eta))`` angle bits suffice for total error ``eta``.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import atan2, cos, isqrt, pi, sin, sqrt
from typing import Iterable

import numpy as np


def _ceil_log2_fraction(value: Fraction) -> int:
    """Smallest nonnegative ``b`` with ``2**b >= value``, exactly."""
    value = Fraction(value)
    if value <= 1:
        return 0
    bits = max(0, value.numerator.bit_length() - value.denominator.bit_length())
    if (value.denominator << bits) < value.numerator:
        bits += 1
    return bits


def _power_of_two_at_least(value: int) -> int:
    if not isinstance(value, int) or value < 1:
        raise ValueError("capacity must be a positive integer")
    return 1 << (value - 1).bit_length()


def _validate_interlacing(
    addable: Iterable[int], removable: Iterable[int]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    x, y = tuple(addable), tuple(removable)
    if len(x) != len(y) + 1 or not x:
        raise ValueError("expected k addable and k-1 removable contents")
    if any(not isinstance(value, int) or isinstance(value, bool) for value in x + y):
        raise TypeError("contents must be ordinary integers")
    merged = tuple(value for pair in zip(x[:-1], y) for value in pair) + (x[-1],)
    if any(left <= right for left, right in zip(merged, merged[1:])):
        raise ValueError("contents must strictly interlace in descending order")
    return x, y


def bounded_weight_factors(
    addable: Iterable[int], removable: Iterable[int], index: int
) -> tuple[tuple[int, int], ...]:
    """Return the ``k-1`` factors in (1), each as ``(numerator, denominator)``."""
    x, y = _validate_interlacing(addable, removable)
    k = len(x)
    if not isinstance(index, int) or not 0 <= index < k:
        raise IndexError("weight index outside the addable contents")
    factors = []
    for j in range(index):
        factors.append((y[j] - x[index], x[j] - x[index]))
    for j in range(index, k - 1):
        factors.append((x[index] - y[j], x[index] - x[j + 1]))
    if len(factors) != k - 1 or any(not 0 < a < d for a, d in factors):
        raise AssertionError("interlacing did not produce bounded factors")
    return tuple(factors)


def exact_branch_weights(
    addable: Iterable[int], removable: Iterable[int]
) -> tuple[Fraction, ...]:
    """Return the exact branching probabilities from bounded products."""
    x, y = _validate_interlacing(addable, removable)
    result = []
    for i in range(len(x)):
        value = Fraction(1)
        for numerator, denominator in bounded_weight_factors(x, y, i):
            value *= Fraction(numerator, denominator)
        result.append(value)
    return tuple(result)


def downward_weight_numerators(
    addable: Iterable[int], removable: Iterable[int], fraction_bits: int
) -> tuple[int, ...]:
    """Compute all weights by the directed fixed-point recurrence.

    A circuit can implement every update with a fixed multiply/divide loop.
    Quotient and remainder scratch are uncomputed by compute-copy-reverse;
    this reference returns only the copied final words.
    """
    x, y = _validate_interlacing(addable, removable)
    if not isinstance(fraction_bits, int) or fraction_bits < 1:
        raise ValueError("fraction_bits must be a positive integer")
    scale = 1 << fraction_bits
    output = []
    for i in range(len(x)):
        word = scale
        for numerator, denominator in bounded_weight_factors(x, y, i):
            word = word * numerator // denominator
        output.append(word)
    return tuple(output)


@dataclass(frozen=True)
class BranchApproximation:
    exact: tuple[Fraction, ...]
    dyadic_numerators: tuple[int, ...]
    normalized: tuple[Fraction, ...]
    fraction_bits: int
    per_weight_error_bound: Fraction
    missing_mass: Fraction
    l1_error: Fraction
    state_error_bound_squared: Fraction


def branch_approximation(
    addable: Iterable[int], removable: Iterable[int], fraction_bits: int
) -> BranchApproximation:
    """Return exact audit data for the rounded-and-renormalized state."""
    x, y = _validate_interlacing(addable, removable)
    exact = exact_branch_weights(x, y)
    if sum(exact, Fraction()) != 1:
        raise ArithmeticError("residue weights failed to normalize exactly")
    numerators = downward_weight_numerators(x, y, fraction_bits)
    total = sum(numerators)
    if total <= 0:
        raise ArithmeticError("precision too small: every rounded weight vanished")
    scale = 1 << fraction_bits
    dyadic = tuple(Fraction(value, scale) for value in numerators)
    normalized = tuple(Fraction(value, total) for value in numerators)
    missing = 1 - sum(dyadic, Fraction())
    l1 = sum((abs(left - right) for left, right in zip(exact, normalized)), Fraction())
    per_weight = Fraction(max(0, len(x) - 1), scale)
    return BranchApproximation(
        exact=exact,
        dyadic_numerators=numerators,
        normalized=normalized,
        fraction_bits=fraction_bits,
        per_weight_error_bound=per_weight,
        missing_mass=missing,
        l1_error=l1,
        state_error_bound_squared=2 * len(x) * per_weight,
    )


@dataclass(frozen=True)
class BranchPrecisionBudget:
    corners: int
    content_span: int
    target_error: Fraction
    padded_capacity: int
    weight_fraction_bits: int
    angle_fraction_bits: int
    weight_state_error_bound_squared: Fraction
    tree_angle_error_bound: Fraction
    per_rotation_synthesis_error: Fraction

    @property
    def word_bits(self) -> int:
        return max(self.weight_fraction_bits, self.angle_fraction_bits) + max(
            self.content_span.bit_length(), self.padded_capacity.bit_length()
        )


def branch_precision_budget(
    corners: int, content_span: int, target_error=Fraction(1, 1000), *, capacity=None
) -> BranchPrecisionBudget:
    """Public sufficient precision for one clean branching preparation.

    The returned error split reserves less than half the target for weight
    rounding, less than one quarter for angle evaluation, and one quarter
    for finite-gate synthesis.  It intentionally has comfortable slack.
    """
    if not isinstance(corners, int) or corners < 1:
        raise ValueError("corners must be a positive integer")
    if not isinstance(content_span, int) or content_span < 1:
        raise ValueError("content_span must be a positive integer")
    eta = Fraction(target_error)
    if not 0 < eta <= 1:
        raise ValueError("target_error must lie in (0,1]")
    padded = _power_of_two_at_least(corners if capacity is None else capacity)
    if padded < corners:
        raise ValueError("capacity is smaller than the number of corners")

    # 2*k*2^(-b/2) <= eta/2, plus enough bits to keep every p_i>=L^-2
    # strictly positive after downward rounding.
    error_bits = 2 * _ceil_log2_fraction(Fraction(4 * corners, 1) / eta)
    positivity_bits = _ceil_log2_fraction(
        Fraction(2 * max(1, corners - 1) * content_span * content_span)
    )
    weight_bits = max(1, error_bits, positivity_bits)

    # The certified angle interval has width below 2^-c.  There are K-1
    # fixed tree nodes, so this leaves more than a factor-eight margin.
    angle_bits = max(1, _ceil_log2_fraction(Fraction(8 * padded, 1) / eta))
    weight_bound_sq = Fraction(2 * corners * max(0, corners - 1), 1 << weight_bits)
    tree_bound = Fraction(max(0, padded - 1), 1 << angle_bits)
    synthesis = eta / (4 * padded)
    return BranchPrecisionBudget(
        corners,
        content_span,
        eta,
        padded,
        weight_bits,
        angle_bits,
        weight_bound_sq,
        tree_bound,
        synthesis,
    )


@dataclass(frozen=True)
class RationalInterval:
    low: Fraction
    high: Fraction

    def __post_init__(self):
        if self.low > self.high:
            raise ValueError("interval endpoints are reversed")

    @property
    def width(self) -> Fraction:
        return self.high - self.low

    @property
    def midpoint(self) -> Fraction:
        return (self.low + self.high) / 2


def _atan_small_interval(value: Fraction, terms: int) -> RationalInterval:
    """Alternating-series enclosure for ``atan(value)``, ``0<=value<=1/2``."""
    value = Fraction(value)
    if not 0 <= value <= Fraction(1, 2) or terms < 1:
        raise ValueError("atan series requires 0<=value<=1/2 and terms>=1")
    partial = sum(
        ((-1) ** j * value ** (2 * j + 1) / (2 * j + 1) for j in range(terms)),
        Fraction(),
    )
    next_term = (-1) ** terms * value ** (2 * terms + 1) / (2 * terms + 1)
    return RationalInterval(min(partial, partial + next_term), max(partial, partial + next_term))


def _pi_over_four_interval(terms: int) -> RationalInterval:
    """Machin enclosure ``pi/4 = 4 atan(1/5) - atan(1/239)``."""
    first = _atan_small_interval(Fraction(1, 5), terms)
    second = _atan_small_interval(Fraction(1, 239), terms)
    return RationalInterval(4 * first.low - second.high, 4 * first.high - second.low)


def _atan_unit_interval(value: Fraction, terms: int) -> RationalInterval:
    """Rational enclosure for atan on [0,1] with geometric convergence."""
    value = Fraction(value)
    if not 0 <= value <= 1:
        raise ValueError("atan argument must lie in [0,1]")
    if value <= Fraction(1, 2):
        return _atan_small_interval(value, terms)
    reduced = (1 - value) / (1 + value)  # in [0,1/3]
    small = _atan_small_interval(reduced, terms)
    quarter = _pi_over_four_interval(terms)
    return RationalInterval(quarter.low - small.high, quarter.high - small.low)


def _sqrt_ratio_interval(numerator: int, denominator: int, bits: int) -> RationalInterval:
    """Enclose ``sqrt(numerator/denominator)`` by adjacent dyadics."""
    if not (isinstance(numerator, int) and isinstance(denominator, int)):
        raise TypeError("square-root ratio inputs must be integers")
    if not 0 <= numerator <= denominator or denominator <= 0 or bits < 1:
        raise ValueError("expected 0<=numerator<=denominator and positive bits")
    scale = 1 << bits
    radicand = numerator * scale * scale
    root = isqrt(radicand // denominator)
    while (root + 1) * (root + 1) * denominator <= radicand:
        root += 1
    while root * root * denominator > radicand:
        root -= 1
    low = Fraction(root, scale)
    if root * root * denominator == radicand:
        return RationalInterval(low, low)
    return RationalInterval(low, Fraction(root + 1, scale))


def certified_tree_angle(left_mass: int, right_mass: int, bits: int) -> RationalInterval:
    """Enclose ``atan2(sqrt(R),sqrt(L))`` in radians using rational work.

    Endpoint branches are totalized explicitly.  ``(0,0)`` denotes an
    unreachable padded subtree and returns the identity angle zero.
    """
    if any(not isinstance(value, int) for value in (left_mass, right_mass, bits)):
        raise TypeError("masses and bits must be integers")
    if left_mass < 0 or right_mass < 0 or bits < 1:
        raise ValueError("masses must be nonnegative and bits positive")
    if right_mass == 0:
        return RationalInterval(Fraction(), Fraction())
    terms = bits + 8
    quarter = _pi_over_four_interval(terms)
    half_pi = RationalInterval(2 * quarter.low, 2 * quarter.high)
    if left_mass == 0:
        return half_pi

    root_bits = bits + 8
    if right_mass <= left_mass:
        ratio = _sqrt_ratio_interval(right_mass, left_mass, root_bits)
        low = _atan_unit_interval(ratio.low, terms)
        high = _atan_unit_interval(ratio.high, terms)
        return RationalInterval(low.low, high.high)

    ratio = _sqrt_ratio_interval(left_mass, right_mass, root_bits)
    low = _atan_unit_interval(ratio.low, terms)
    high = _atan_unit_interval(ratio.high, terms)
    return RationalInterval(half_pi.low - high.high, half_pi.high - low.low)


@dataclass(frozen=True)
class TreeRotation:
    path: tuple[int, ...]
    left_mass: int
    right_mass: int
    angle: RationalInterval

    @property
    def parent_mass(self) -> int:
        return self.left_mass + self.right_mass


def fixed_tree_rotations(
    numerators: Iterable[int], angle_bits: int, *, capacity: int | None = None
) -> tuple[TreeRotation, ...]:
    """Return all ``K-1`` nodes of a fixed zero-padded preparation tree."""
    values = tuple(numerators)
    if not values or any(not isinstance(value, int) or value < 0 for value in values):
        raise ValueError("numerators must be a nonempty sequence of nonnegative integers")
    padded = _power_of_two_at_least(len(values) if capacity is None else capacity)
    if padded < len(values):
        raise ValueError("capacity is smaller than the numerator list")
    leaves = values + (0,) * (padded - len(values))
    prefix = [0]
    for value in leaves:
        prefix.append(prefix[-1] + value)

    output = []

    def visit(start: int, width: int, path: tuple[int, ...]) -> None:
        if width == 1:
            return
        half = width // 2
        left = prefix[start + half] - prefix[start]
        right = prefix[start + width] - prefix[start + half]
        output.append(TreeRotation(path, left, right, certified_tree_angle(left, right, angle_bits)))
        visit(start, half, path + (0,))
        visit(start + half, half, path + (1,))

    visit(0, padded, ())
    return tuple(output)


def tree_prepared_state(
    numerators: Iterable[int], *, angle_bits: int | None = None, capacity: int | None = None
) -> np.ndarray:
    """Dense reference state for the fixed tree, exact or certified-angle midpoint."""
    values = tuple(numerators)
    padded = _power_of_two_at_least(len(values) if capacity is None else capacity)
    rotations = fixed_tree_rotations(values, max(1, angle_bits or 8), capacity=padded)
    by_path = {rotation.path: rotation for rotation in rotations}
    state = np.zeros(padded)

    def visit(path: tuple[int, ...], start: int, width: int, amplitude: float) -> None:
        if width == 1:
            state[start] = amplitude
            return
        node = by_path[path]
        if node.parent_mass == 0 or node.right_mass == 0:
            theta = 0.0
        elif node.left_mass == 0:
            theta = pi / 2
        elif angle_bits is None:
            theta = atan2(sqrt(node.right_mass), sqrt(node.left_mass))
        else:
            theta = float(node.angle.midpoint)
        half = width // 2
        visit(path + (0,), start, half, amplitude * cos(theta))
        visit(path + (1,), start + half, half, amplitude * sin(theta))

    visit((), 0, padded, 1.0)
    return state


def young_reflection_block(axial_distance: int, *, angle_bits: int | None = None) -> np.ndarray:
    """Exact/display or dyadic-angle seminormal two-tableau block.

    For ``angle_bits`` not ``None``, the reflection angle is rounded to the
    nearest multiple of ``2**(-angle_bits)`` radians.  The operator error is
    at most ``2**(-angle_bits-1)``.  The ``|d|=1`` sign endpoints are exact.
    """
    if not isinstance(axial_distance, int) or axial_distance == 0:
        raise ValueError("axial_distance must be a nonzero integer")
    if axial_distance == 1:
        return np.array([[1.0, 0.0], [0.0, -1.0]])
    if axial_distance == -1:
        return np.array([[-1.0, 0.0], [0.0, 1.0]])
    reciprocal = 1.0 / axial_distance
    theta = atan2(sqrt(max(0.0, 1.0 - reciprocal * reciprocal)), reciprocal)
    if angle_bits is not None:
        if not isinstance(angle_bits, int) or angle_bits < 1:
            raise ValueError("angle_bits must be a positive integer")
        step = 2.0 ** (-angle_bits)
        theta = round(theta / step) * step
    return np.array([[cos(theta), sin(theta)], [sin(theta), -cos(theta)]])


def generator_angle_bits(level: int, target_error=Fraction(1, 1000)) -> int:
    """Sufficient bits for all ``level-1`` Young generators in one ``W``."""
    if not isinstance(level, int) or level < 1:
        raise ValueError("level must be a positive integer")
    eta = Fraction(target_error)
    if not 0 < eta <= 1:
        raise ValueError("target_error must lie in (0,1]")
    return max(1, _ceil_log2_fraction(Fraction(8 * level, 1) / eta))
