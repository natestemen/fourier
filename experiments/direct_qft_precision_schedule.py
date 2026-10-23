"""Integer-only precision witnesses for the direct-induction theorem.

This is an arithmetic reference, not a reversible-gate compiler. Unlike the
bounded small-instance gate prototype, its constants and guarded Horner steps
do not use machine floating point. All products are immediately reduced to a
fixed fractional width. Exact rational interval code is used only by tests.
The public schedule is sufficient, deliberately loose, and shape independent.
"""

from dataclasses import dataclass
from fractions import Fraction
from math import isqrt


def ceil_log2(value):
    value = Fraction(value)
    if value <= 0:
        raise ValueError("logarithm argument must be positive")
    if value <= 1:
        return 0
    bits = max(0, value.numerator.bit_length() - value.denominator.bit_length())
    return bits + int(value.denominator << bits < value.numerator)


def _power_two(value):
    return 1 << (value - 1).bit_length()


def guard_precision(bits):
    if not isinstance(bits, int) or isinstance(bits, bool) or bits < 1:
        raise ValueError("requested precision must be a positive integer")
    return bits + ceil_log2(bits + 2) + 12


def _atan_small(word, precision):
    """q-fractional signed Horner evaluation on 0 <= word/2**q <= 1/2."""
    scale = 1 << precision
    if not 0 <= 2 * word <= scale:
        raise ValueError("arctangent argument outside the reduced interval")
    square = word * word // scale
    total = 0
    for j in reversed(range(precision + 2)):
        coefficient = ((-1 if j % 2 else 1) * scale) // (2 * j + 1)
        total = coefficient + square * total // scale
    return word * total // scale


def _quarter_pi(precision):
    scale = 1 << precision
    return 4 * _atan_small(scale // 5, precision) - _atan_small(scale // 239, precision)


@dataclass(frozen=True)
class GuardedAngle:
    numerator: int
    fractional_bits: int
    requested_bits: int
    root_trials: int
    horner_terms: int

    @property
    def value(self):
        return Fraction(self.numerator, 1 << self.fractional_bits)

    @property
    def error_bound(self):
        return Fraction(64 * (self.fractional_bits + 1), 1 << self.fractional_bits)


def guarded_ratio_angle(left, right, bits):
    """Approximate atan2(sqrt(right), sqrt(left)) with an explicit guard budget.

    Zero parent means zero angle. In a circuit, one-sided masses may instead
    select exact endpoint gates. Root trials compare integers, including when
    the ratio is extremely small. This never forms growing rational powers.
    """
    if any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in (left, right)):
        raise ValueError("masses must be nonnegative integers")
    q = guard_precision(bits)
    scale = 1 << q
    if right == 0:
        return GuardedAngle(0, q, bits, 0, q + 2)
    quarter_pi = _quarter_pi(q)
    if left == 0:
        return GuardedAngle(2 * quarter_pi, q, bits, 0, q + 2)
    numerator, denominator = min(left, right), max(left, right)
    root = 0
    for position in reversed(range(q + 1)):
        trial = root | (1 << position)
        if trial * trial * denominator <= numerator << (2 * q):
            root = trial
    if 2 * root <= scale:
        small = _atan_small(root, q)
    else:
        reduced = (scale - root) * scale // (scale + root)
        small = quarter_pi - _atan_small(reduced, q)
    angle = 2 * quarter_pi - small if right > left else small
    return GuardedAngle(angle, q, bits, q + 1, q + 2)


@dataclass(frozen=True)
class ColumnPrecisionSchedule:
    level: int
    column_error: Fraction
    corner_capacity: int
    weight_bits: int
    angle_bits: int
    guarded_angle_bits: int
    angle_calls_bound: int
    public_rotation_slots_bound: int
    per_rotation_synthesis_error: Fraction

    @property
    def weight_state_error_bound_squared(self):
        k = self.corner_capacity
        return Fraction(2 * k * (k - 1), 1 << self.weight_bits)

    @property
    def angle_error_bound(self):
        return Fraction(self.angle_calls_bound, 1 << self.angle_bits)

    @property
    def synthesis_error_bound(self):
        return self.public_rotation_slots_bound * self.per_rotation_synthesis_error


def column_precision_schedule(level, column_error):
    """Public capacities for every coherent shape sector at one level."""
    if not isinstance(level, int) or isinstance(level, bool) or level < 2:
        raise ValueError("nontrivial induction level must be at least two")
    eta = Fraction(column_error)
    if not 0 < eta <= 1:
        raise ValueError("column error must lie in (0,1]")
    root = isqrt(2 * level)
    root += int(root * root < 2 * level)
    capacity = _power_two(root + 1)
    b = max(ceil_log2(2 * (capacity - 1) * (2 * level) ** 2),
            2 * ceil_log2(Fraction(16 * capacity) / eta))
    c = ceil_log2(Fraction(256 * (level + capacity + 1)) / eta)
    q = guard_precision(c)
    tag_nodes = _power_two(level + 1) - 1
    calls = (capacity - 1) + (level - 1) + tag_nodes + 1
    slots = ((capacity - 1) + (level - 1)) * (q + 2) + tag_nodes + 1
    return ColumnPrecisionSchedule(level, eta, capacity, b, c, q, calls, slots,
                                   eta / (16 * slots))


def full_qft_column_error(degree, channel_error):
    if not isinstance(degree, int) or isinstance(degree, bool) or degree < 1:
        raise ValueError("degree must be positive")
    epsilon = Fraction(channel_error)
    if not 0 < epsilon <= Fraction(1, 2):
        raise ValueError("channel error must lie in (0,1/2]")
    rounds = isqrt(degree)
    rounds += int(rounds * rounds < degree)
    return epsilon / (2 * degree * (4 * rounds + 1))
