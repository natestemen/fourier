"""Directed fixed-binary interval verifier for the coherent multipole schedule.

Unlike the floating reference, every endpoint here is an integer multiple of
2**(-bits). All rounding is outward. Logarithms use a bounded atanh series,
not a platform libm call. The reported enclosures include series truncation.

The data-access schedule uses padded segmented scans and the reversible
source/query join; it never looks up a payload through a Python dictionary.
Loop bounds depend only on public capacity, span, order and precision.
Python predicates represent controlled operations. This is NOT a compiled
quantum circuit. In particular, arithmetic compute/copy/uncompute is justified
by reversible simulation, whereas the join's cleanup is explicitly simulated.
Intervals themselves are a certification tool, not an extra quantum oracle.
"""

from dataclasses import dataclass
from math import isqrt

from reversible_join_probe import QueryRecord, SourceRecord, reversible_batched_join


@dataclass(frozen=True)
class BinaryInterval:
    """Closed interval [lo, hi]/2**bits with exact integer endpoints."""

    lo: int
    hi: int
    bits: int

    def __post_init__(self):
        if self.bits < 1 or self.lo > self.hi:
            raise ValueError("invalid binary interval")

    @classmethod
    def ratio(cls, numerator, denominator, bits):
        if denominator == 0:
            raise ZeroDivisionError
        if denominator < 0:
            numerator, denominator = -numerator, -denominator
        scaled = numerator << bits
        return cls(scaled // denominator, -((-scaled) // denominator), bits)

    @classmethod
    def integer(cls, value, bits):
        return cls(value << bits, value << bits, bits)

    def _coerce(self, other):
        if isinstance(other, int):
            other = self.integer(other, self.bits)
        if not isinstance(other, BinaryInterval) or self.bits != other.bits:
            raise ValueError("interval precisions differ")
        return other

    def __add__(self, other):
        other = self._coerce(other)
        return BinaryInterval(self.lo + other.lo, self.hi + other.hi, self.bits)

    __radd__ = __add__

    def __neg__(self):
        return BinaryInterval(-self.hi, -self.lo, self.bits)

    def __sub__(self, other):
        return self + (-self._coerce(other))

    def __mul__(self, other):
        other = self._coerce(other)
        products = [a * b for a in (self.lo, self.hi)
                    for b in (other.lo, other.hi)]
        scale = 1 << self.bits
        return BinaryInterval(min(products) // scale,
                              -((-max(products)) // scale), self.bits)

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self._coerce(other)
        if other.lo <= 0 <= other.hi:
            raise ZeroDivisionError("divisor interval includes zero")
        quotients = [self.ratio(a, b, self.bits)
                     for a in (self.lo, self.hi)
                     for b in (other.lo, other.hi)]
        return BinaryInterval(min(q.lo for q in quotients),
                              max(q.hi for q in quotients), self.bits)

    def widen(self, radius):
        """Enlarge by the nonnegative upper endpoint of ``radius``."""
        radius = self._coerce(radius)
        if radius.lo < 0:
            raise ValueError("negative uncertainty")
        return BinaryInterval(self.lo - radius.hi, self.hi + radius.hi, self.bits)

    @property
    def midpoint(self):
        return (self.lo + self.hi) / (2 << self.bits)

    @property
    def radius(self):
        return (self.hi - self.lo) / (2 << self.bits)


def _atanh_log(z, terms):
    """Enclose 2 atanh(z) for exact z in [0,1/3]."""
    zero = BinaryInterval.integer(0, z.bits)
    power, square, total = z, z * z, zero
    for j in range(terms):
        total = total + power / (2 * j + 1)
        power = power * square
    # Tail <= (9/4)*3**(-2R-1)/(2R+1); a simpler upper bound suffices.
    tail = BinaryInterval.ratio(3, 3 ** (2 * terms + 1), z.bits)
    result = 2 * total
    return BinaryInterval(result.lo, result.hi + tail.hi, z.bits)


def log_integer(value, *, bits, exponent_bound, terms=None):
    """Certified ln(value), with a public fixed range-reduction schedule.

    A coherent implementation obtains the leading bit by a fixed scan of
    exponent_bound+1 bits. Here bit_length simulates that classical function.
    """
    if not isinstance(value, int) or value < 1 or value.bit_length() - 1 > exponent_bound:
        raise ValueError("log argument outside public range")
    if terms is None:
        terms = bits + (exponent_bound + 1).bit_length() + 8
    if terms < 1:
        raise ValueError("positive series length required")
    exponent = value.bit_length() - 1
    base = 1 << exponent
    z = BinaryInterval.ratio(value - base, value + base, bits)
    ln2 = _atanh_log(BinaryInterval.ratio(1, 3, bits), terms)
    return _atanh_log(z, terms) + exponent * ln2


def exp_interval(value, *, argument_bound, terms=None):
    """Enclose exp(value) by fixed range reduction, Taylor and squaring.

    argument_bound is public; intermediate magnitudes are <= exp(bound).
    In the weight application bound=O(log(span)), so integer word widths
    remain logarithmic in the input span. This is not a generic constant-cost
    exponential oracle for exponentially large arguments.
    """
    if argument_bound < 1 or max(abs(value.lo), abs(value.hi)) > argument_bound << value.bits:
        raise ValueError("exponential argument exceeds public bound")
    if terms is None:
        terms = value.bits + 8
    if terms < 1:
        raise ValueError("positive series length required")
    squarings = argument_bound.bit_length() + 1
    one = BinaryInterval.integer(1, value.bits)
    # For reduced y<1/2 the positive Taylor tail is <=2**(-R).
    # This loose dyadic bound avoids a factorial with O(R log R) bits.
    tail = BinaryInterval.ratio(1, 1 << terms, value.bits)

    def endpoint_exp(endpoint):
        y = BinaryInterval.ratio(abs(endpoint), 1 << (value.bits + squarings), value.bits)
        term = total = one
        for j in range(1, terms + 1):
            term = term * y / j
            total = total + term
        total = BinaryInterval(total.lo, total.hi + tail.hi, value.bits)
        for _ in range(squarings):
            total = total * total
        return one / total if endpoint < 0 else total

    low, high = endpoint_exp(value.lo), endpoint_exp(value.hi)
    return BinaryInterval(low.lo, high.hi, value.bits)


def sqrt_interval(value):
    """Directed square root by fixed-width integer square-root arithmetic."""
    if value.lo < 0:
        raise ValueError("negative square-root argument")
    low_squared, high_squared = value.lo << value.bits, value.hi << value.bits
    low, high = isqrt(low_squared), isqrt(high_squared)
    return BinaryInterval(low, high + int(high * high != high_squared), value.bits)


def atan_unit_interval(value, *, terms=None):
    """Enclose atan(x) on [0,1], using one half-angle reduction.

    w=x/(1+sqrt(1+x*x)) <= sqrt(2)-1 < 1/2. Therefore the alternating
    series remainder after R terms, multiplied by two for the half-angle
    identity, has magnitude <=2**(-2R). All interval operations round out.
    """
    if value.lo < 0 or value.hi > 1 << value.bits:
        raise ValueError("atan input must lie in [0,1]")
    if terms is None:
        terms = value.bits + 8
    if terms < 1:
        raise ValueError("positive series length required")
    one = BinaryInterval.integer(1, value.bits)
    w = value / (one + sqrt_interval(one + value * value))
    square, power = w * w, w
    total = BinaryInterval.integer(0, value.bits)
    for j in range(terms):
        term = power / (2 * j + 1)
        total = total + (term if j % 2 == 0 else -term)
        power = power * square
    return (2 * total).widen(BinaryInterval.ratio(1, 1 << (2 * terms), value.bits))


def tree_angle_interval(left_mass, right_mass, *, bits=80):
    """Certify 2 atan2(sqrt(R),sqrt(L)) for exact nonnegative integer masses.

    Common dyadic scale cancels. Zero branches are exact; both-zero branches
    get a harmless zero angle. A circuit uses fixed public word widths, with
    these predicates controlling the operations rather than allocation.
    """
    if not isinstance(left_mass, int) or not isinstance(right_mass, int) or min(left_mass, right_mass) < 0:
        raise ValueError("masses must be nonnegative integers")
    if right_mass == 0:
        return BinaryInterval.integer(0, bits)
    pi_interval = 4 * atan_unit_interval(BinaryInterval.integer(1, bits))
    if left_mass == 0:
        return pi_interval
    ratio = BinaryInterval.ratio(min(left_mass, right_mass), max(left_mass, right_mass), bits)
    angle = 2 * atan_unit_interval(sqrt_interval(ratio))
    return angle if left_mass >= right_mass else pi_interval - angle


@dataclass(frozen=True)
class CertifiedPotential:
    values: tuple[BinaryInterval, ...]
    capacity: int
    levels: int
    order: int
    moment_terms: int
    evaluation_terms: int
    join_comparators: int
    join_scan_steps: int
    payload_bits: int
    joins_clean: bool
    truncation_radius: BinaryInterval


@dataclass(frozen=True)
class CertifiedWeights:
    alpha: tuple[BinaryInterval, ...]
    beta: tuple[BinaryInterval, ...]
    potentials: CertifiedPotential


@dataclass(frozen=True)
class CertifiedScaling:
    left: tuple[BinaryInterval, ...]
    right: tuple[BinaryInterval, ...]
    weights: CertifiedWeights
    vector_error: BinaryInterval
    padding_lower: BinaryInterval
    steps: int
    joins_clean: bool
    join_comparators: int


def certified_absolute_matvec(ac, rc, weights, vector, *, capacity, span_bound,
                             order, bits=80, transpose=False):
    """Enclose |A|v or |A|^T v, including identity padding.

    Weight intervals enclose the exact residues. Input vector entries are
    exact stored dyadic words; returning interval midpoints supplies a
    certified local error for a stable fixed-point Horner iteration.
    """
    ac, rc, vector = tuple(ac), tuple(rc), tuple(vector)
    k = len(ac)
    if len(vector) != capacity or any(v.bits != bits or v.lo != v.hi for v in vector):
        raise ValueError("expected a padded vector of point dyadic words")
    if any(v.lo < 0 or v.hi > (4 * capacity) << bits for v in vector):
        raise ValueError("vector outside public positive range")
    zero = BinaryInterval.integer(0, bits)
    merged = tuple(v for pair in zip(ac, rc) for v in pair) + (ac[-1],)
    charges = [zero] * (2 * k - 1)
    for i in range(k):
        if transpose:
            charges[2*i] = weights.alpha[i] * vector[i]
        elif i < k - 1:
            charges[2*i + 1] = weights.beta[i] * vector[i + 1]
    potentials = certified_potential(
        tuple(x - ac[-1] for x in merged[::-1]), tuple(charges[::-1]),
        capacity=2*capacity - 1, span_bound=span_bound,
        charge_bound=8 * capacity * span_bound, order=order, bits=bits, kernel="inverse")
    ordered = potentials.values[:2*k - 1][::-1]
    if transpose:
        first = sum((weights.alpha[i] * vector[i] for i in range(k)), zero)
        output = [first] + [weights.beta[j] * ordered[2*j + 1] for j in range(k - 1)]
    else:
        output = [weights.alpha[i] * (vector[0] + ordered[2*i]) for i in range(k)]
    output.extend(vector[k:])
    return tuple(output), potentials


def certified_balancing(ac, rc, *, capacity, span_bound, order, bits=80):
    """Certify stored positive scalings and padding for one basis sector.

    At each Horner iteration a fixed dyadic vector is sent to an interval
    matvec enclosing the EXACT H times that vector. Rounding its midpoint
    adds a certified component error. Since ||H/lambda||<=1/2, all such
    errors accumulate to <=2 sqrt(2K) times the largest step error.
    This is a rigorous instance certificate, not a uniform machine proof.
    """
    weights = certified_weights(ac, rc, capacity=capacity, span_bound=span_bound,
                                order=order, bits=bits)
    c = (capacity - 1).bit_length() + 1
    lam = 2*c
    steps = ((8 * capacity - 1).bit_length() + 1) // 2
    one = BinaryInterval.integer(1, bits)
    left = right = (one,) * capacity
    max_radius = 0
    comparators = weights.potentials.join_comparators
    clean = weights.potentials.joins_clean
    for _ in range(steps):
        next_left, stats_l = certified_absolute_matvec(
            ac, rc, weights, right, capacity=capacity, span_bound=span_bound,
            order=order, bits=bits)
        next_right, stats_r = certified_absolute_matvec(
            ac, rc, weights, left, capacity=capacity, span_bound=span_bound,
            order=order, bits=bits, transpose=True)
        rounded = []
        for value in next_left + next_right:
            value = one + value / lam
            midpoint = (value.lo + value.hi) // 2
            max_radius = max(max_radius, midpoint - value.lo, value.hi - midpoint)
            rounded.append(BinaryInterval(midpoint, midpoint, bits))
        left, right = tuple(rounded[:capacity]), tuple(rounded[capacity:])
        comparators += stats_l.join_comparators + stats_r.join_comparators
        clean &= stats_l.joins_clean and stats_r.joins_clean
    component = BinaryInterval(max_radius, max_radius, bits)
    error = 2 * sqrt_interval(BinaryInterval.integer(2 * capacity, bits)) * component
    if error.hi >= 1 << bits:
        raise ArithmeticError("precision insufficient to certify positive scaling")
    padding = one - (lam + error * (lam + 1) / (one - error)) / (2 * lam)
    return CertifiedScaling(left, right, weights, error, padding, steps, clean, comparators)


def certified_weights(ac, rc, *, capacity, span_bound, order, bits=80):
    """Certify residue weights of integer interlacing without dense products.

    capacity is the public number of addable slots, not branch-specific k.
    Returned arrays retain public capacity, with zero inactive entries.
    The variable-length inputs only encode the valid-prefix flags of this
    basis-state verifier; physical slots and operation counts are fixed.
    """
    ac, rc = tuple(ac), tuple(rc)
    k = len(ac)
    if k < 1 or k > capacity or len(rc) != k - 1:
        raise ValueError("wrong corner counts")
    merged = tuple(v for pair in zip(ac, rc) for v in pair) + (ac[-1],)
    if any(a <= b for a, b in zip(merged, merged[1:])):
        raise ValueError("contents must strictly interlace")
    positions = tuple(t - ac[-1] for t in merged[::-1])
    charges = tuple(-1 if i % 2 == 0 else 1 for i in range(2*k - 1))
    potential = certified_potential(positions, charges, capacity=2*capacity - 1,
                                    span_bound=span_bound, charge_bound=1,
                                    order=order, bits=bits)
    zero = BinaryInterval.integer(0, bits)
    alpha, beta = [], []
    # This prefix reversal only packages the returned certification object.
    # A coherent implementation instead carries original slot IDs and reverses
    # its fixed initial sorting network. Do NOT implement it by K repeated
    # linear scans; that would introduce a quadratic cost.
    ordered = list(potential.values[:2*k - 1])[::-1]
    bound = 2 * span_bound.bit_length() + 4
    for i in range(capacity):
        alpha.append(exp_interval(ordered[2*i] / 2, argument_bound=bound) if i < k else zero)
        if i < capacity - 1:
            beta.append(exp_interval(-ordered[2*i + 1] / 2, argument_bound=bound)
                        if i < k - 1 else zero)
    return CertifiedWeights(tuple(alpha), tuple(beta), potential)


def certified_potential(positions, charges, *, capacity, span_bound,
                        charge_bound, order, bits=80, kernel="log"):
    """Enclose sum_{s != t} charge[s]*kernel(|t-s|) for integer nodes.

    Valid points occupy a sorted prefix of ``capacity`` slots; all remaining
    slots are invalid zeros. Public span_bound >= max(position) >= 0.
    Each charge's absolute value must be <= public integer charge_bound.
    Six queries per target/level retain the same schedule for all sectors.
    """
    positions, charges = tuple(positions), tuple(charges)
    if (not isinstance(capacity, int) or capacity < 1 or len(positions) > capacity
            or len(charges) != len(positions) or not positions
            or span_bound < 1 or charge_bound < 1 or order < 1 or bits < 8
            or kernel not in ("log", "inverse")):
        raise ValueError("invalid public parameters or input sizes")
    if (any(not isinstance(x, int) or not 0 <= x <= span_bound for x in positions)
            or any(a >= b for a, b in zip(positions, positions[1:]))):
        raise ValueError("positions must be distinct sorted integers in range")
    count = len(positions)
    zero, one = (BinaryInterval.integer(v, bits) for v in (0, 1))
    charges = tuple(BinaryInterval.integer(q, bits) if isinstance(q, int) else q
                    for q in charges)
    if any(q.bits != bits or max(abs(q.lo), abs(q.hi)) > charge_bound << bits
           for q in charges):
        raise ValueError("charge outside public bound or wrong precision")
    points = positions + (0,) * (capacity - count)
    charges = charges + (zero,) * (capacity - count)
    valid = tuple(i < count for i in range(capacity))
    levels = span_bound.bit_length()
    exponent_bound = levels + 3
    ln2 = log_integer(2, bits=bits, exponent_bound=exponent_bound)
    # Normalized moments have magnitude <= capacity*charge_bound. Rounding
    # in powers may widen them; a generous polynomial envelope is validated.
    payload_bits = bits + (16 * capacity * charge_bound * (order + 1)).bit_length() + 2
    modulus, sign = 1 << payload_bits, 1 << (payload_bits - 1)

    def encode(value):
        if not -sign <= value < sign:
            raise OverflowError("moment exceeded its public word width")
        return value % modulus

    def decode(value):
        return value - modulus if value >= sign else value

    result = [zero for _ in range(capacity)]
    comparator_count = scan_count = 0
    clean = True
    for level in range(levels):
        width = 1 << level
        ids = [x // width for x in points]
        sources = []
        accumulator = [zero for _ in range(order + 1)]
        for i in range(capacity):
            start = i == 0 or not valid[i] or ids[i] != ids[i - 1]
            if start:
                accumulator = [zero for _ in range(order + 1)]
            z = BinaryInterval.ratio(2 * (points[i] % width) - width, width, bits)
            power = one
            for j in range(order + 1):
                accumulator[j] = accumulator[j] + charges[i] * power
                power = power * z
            last = valid[i] and (i == capacity - 1 or not valid[i + 1]
                                  or ids[i + 1] != ids[i])
            payload = tuple(encode(endpoint) for moment in accumulator
                            for endpoint in (moment.lo, moment.hi))
            sources.append(SourceRecord(ids[i], payload, last))

        queries, source_boxes = [], []
        for i in range(capacity):
            for offset in (-1, 0, 1):
                for child in (0, 1):
                    source = 2 * (ids[i] // 2 + offset) + child
                    active = valid[i] and source >= 0 and abs(source - ids[i]) > 1
                    queries.append(QueryRecord(max(0, source), active))
                    source_boxes.append(source)
        joined = reversible_batched_join(sources, queries,
                                        payload_words=2 * (order + 1),
                                        payload_bits=payload_bits)
        comparator_count += joined.comparator_applications
        scan_count += joined.scan_steps
        clean &= joined.clean
        for slot in range(6 * capacity):
            i = slot // 6
            active = queries[slot].valid
            # A safe dummy denominator makes inactive arithmetic well-defined.
            twice_distance = (2 * points[i] - (2 * source_boxes[slot] + 1) * width
                              if active else 3 * width)
            ratio = BinaryInterval.ratio(width, twice_distance, bits)
            moments = tuple(BinaryInterval(decode(joined.outputs[slot][2*j]),
                                           decode(joined.outputs[slot][2*j + 1]), bits)
                            for j in range(order + 1))
            if kernel == "log":
                log_distance = log_integer(abs(twice_distance), bits=bits,
                                           exponent_bound=exponent_bound) - ln2
                value = moments[0] * log_distance
            else:
                value = moments[0]
            power = one
            for j in range(1, order + 1):
                power = power * ratio
                if kernel == "log":
                    value = value - moments[j] * power / j
                else:
                    value = value + moments[j] * power
            if kernel == "inverse":
                value = value * BinaryInterval.ratio(2, abs(twice_distance), bits)
            result[i] = result[i] + (value if active else zero)

    if kernel == "inverse":
        for i in range(capacity - 1):
            adjacent = valid[i + 1] and points[i + 1] - points[i] == 1
            result[i] = result[i] + (charges[i + 1] if adjacent else zero)
            result[i + 1] = result[i + 1] + (charges[i] if adjacent else zero)
        tail = BinaryInterval.ratio(capacity * charge_bound, 3 ** (order + 1), bits)
    else:
        tail = BinaryInterval.ratio(3 * capacity * charge_bound,
                                    2 * (order + 1) * 3 ** (order + 1), bits)
    values = tuple(result[i].widen(tail) if valid[i] else zero for i in range(capacity))
    return CertifiedPotential(values, capacity, levels, order,
                              levels * capacity * (order + 1),
                              levels * 6 * capacity * (order + 1),
                              comparator_count, scan_count, payload_bits, clean, tail)
