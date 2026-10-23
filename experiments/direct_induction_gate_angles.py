"""Gate-level branching coefficients and rotations for direct induction.

This module is deliberately small and literal.  It compiles the corner-weight
formula and the binary state-preparation tree to :mod:`reversible_gate_ir`;
there is no instruction for arithmetic, no shape-to-angle oracle, and no
enumeration of Young diagrams.  All Boolean arithmetic emitted by the public
functions consists only of X, CX, and CCX gates.

The input corner contents are unsigned words.  A caller may therefore add one
public offset (for example ``M``) to every signed content before invoking this
module; all differences below are unchanged.  For descending, interlacing
active contents

    x[0] > y[0] > x[1] > ... > y[k-2] > x[k-1],

the ``i``-th probability is evaluated by the bounded product

    prod_{j<i} (y[j]-x[i])/(x[j]-x[i])
    prod_{j>=i} (x[i]-y[j])/(x[i]-x[j+1]).

Starting from ``2**fraction_bits``, every multiplication/division is rounded
down immediately.  Padding is handled coherently: a disabled factor is muxed
to ``1/1`` and a disabled output leaf is muxed to zero.  The loop topology is
therefore a function only of the public capacity and word widths.

For a mass pair ``(L,R)``, ``apply_ratio_rotation`` computes, uses, and erases
an angle approximating ``atan2(sqrt(R),sqrt(L))``.  A fixed-step trial square
root obtains ``sqrt(min(L,R)/max(L,R))``.  The argument is reduced to at most
one half, and atan is evaluated with separate positive and negative Taylor
sums, so no signed-arithmetic convention is hidden in the circuit.  The only
public real constants are pi and the bit increments of an Ry rotation.  The
``L=0<R`` endpoint is a direct controlled ``Ry(pi)``; ``R=0`` (including a
zero parent) is the identity.

This is a bounded small-instance compiler and audit harness, not an optimized
arithmetic library.  With capacity K, weight precision b, angle precision c,
and Boolean word width W, its fixed schedule contains K(K-1) multiply/divide
updates and, per tree node, O(c) fixed-width multiply/divide/compare
steps plus c+1 controlled public rotations.  Thus it witnesses polynomial
coherent angle generation without a precomputed shape table.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2, pi
from typing import Iterable, Sequence

from reversible_gate_ir import Bit, BooleanBuilder, Circuit, Wire, Word


@dataclass(frozen=True)
class ArithmeticReport:
    """Gate interval and public dimensions for one arithmetic oracle call."""

    gate_start: int
    gate_stop: int
    qubit_count: int
    capacity: int
    word_width: int

    @property
    def gate_count(self) -> int:
        return self.gate_stop - self.gate_start


@dataclass(frozen=True)
class WeightWords:
    """Clean copied fixed-point numerators returned by the weight oracle."""

    words: tuple[tuple[Wire, ...], ...]
    fraction_bits: int
    report: ArithmeticReport

    @property
    def capacity(self) -> int:
        return len(self.words)

    @property
    def word_width(self) -> int:
        return self.fraction_bits + 1

    @property
    def flat_wires(self) -> tuple[Wire, ...]:
        return tuple(bit for word in self.words for bit in word)


@dataclass(frozen=True)
class RatioRotationReport:
    """Public accounting for a computed mass-ratio rotation."""

    gate_start: int
    gate_stop: int
    qubit_count: int
    mass_width: int
    angle_fraction_bits: int
    angle_word_width: int
    atan_terms: int

    @property
    def gate_count(self) -> int:
        return self.gate_stop - self.gate_start


@dataclass(frozen=True)
class TreePreparationReport:
    """Accounting for a clean fixed-capacity binary preparation tree."""

    gate_start: int
    gate_stop: int
    qubit_count: int
    capacity: int
    mass_width: int
    angle_fraction_bits: int
    node_reports: tuple[RatioRotationReport, ...]

    @property
    def gate_count(self) -> int:
        return self.gate_stop - self.gate_start


@dataclass(frozen=True)
class BranchCompilation:
    """Accounting returned by :func:`compile_branch_state`."""

    weight_compute: ArithmeticReport
    tree: TreePreparationReport
    weight_erase: ArithmeticReport
    gate_start: int
    gate_stop: int
    qubit_count: int

    @property
    def gate_count(self) -> int:
        return self.gate_stop - self.gate_start


def _as_word(word: Iterable[Bit], name: str) -> Word:
    result = tuple(word)
    if not result:
        raise ValueError(f"{name} words must be nonempty")
    for bit in result:
        if not isinstance(bit, Wire) and bit not in (0, 1):
            raise ValueError(f"{name} contains a non-Boolean constant")
    return result


def _as_bit(bit: Bit, name: str) -> Bit:
    if not isinstance(bit, Wire) and bit not in (0, 1):
        raise ValueError(f"{name} must be a Wire or Boolean constant")
    return bit


def _resize(word: Iterable[Bit], width: int) -> Word:
    word = tuple(word)
    if width < 0:
        raise ValueError("negative word width")
    return (word + (0,) * width)[:width]


def _flatten(words: Iterable[Iterable[Bit]]) -> tuple[Bit, ...]:
    return tuple(bit for word in words for bit in word)


def _power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _validate_corner_registers(
    addable_words: Sequence[Iterable[Bit]],
    removable_words: Sequence[Iterable[Bit]],
    addable_active: Sequence[Bit],
    removable_active: Sequence[Bit],
) -> tuple[tuple[Word, ...], tuple[Word, ...], tuple[Bit, ...], tuple[Bit, ...]]:
    x = tuple(_as_word(word, "addable") for word in addable_words)
    y = tuple(_as_word(word, "removable") for word in removable_words)
    ax = tuple(_as_bit(bit, "addable activity") for bit in addable_active)
    ay = tuple(_as_bit(bit, "removable activity") for bit in removable_active)
    if not _power_of_two(len(x)):
        raise ValueError("addable capacity must be a positive power of two")
    if len(y) != len(x) - 1:
        raise ValueError("removable capacity must be addable capacity minus one")
    if len(ax) != len(x) or len(ay) != len(y):
        raise ValueError("activity-mask lengths do not match corner capacities")
    widths = {len(word) for word in x + y}
    if len(widths) > 1:
        raise ValueError("all corner words must have one common width")
    return x, y, ax, ay


def _validate_weight_targets(
    targets: Sequence[Iterable[Wire]], capacity: int, fraction_bits: int
) -> tuple[tuple[Wire, ...], ...]:
    if not isinstance(fraction_bits, int) or fraction_bits < 1:
        raise ValueError("fraction_bits must be a positive integer")
    result = tuple(tuple(word) for word in targets)
    width = fraction_bits + 1
    if len(result) != capacity or any(len(word) != width for word in result):
        raise ValueError("weight targets have the wrong shape")
    if any(not isinstance(bit, Wire) for word in result for bit in word):
        raise TypeError("weight targets must be circuit wires")
    if len(set(_flatten(result))) != capacity * width:
        raise ValueError("weight target wires must be distinct")
    return result


def xor_directed_weights(
    circuit: Circuit,
    addable_words: Sequence[Iterable[Bit]],
    removable_words: Sequence[Iterable[Bit]],
    addable_active: Sequence[Bit],
    removable_active: Sequence[Bit],
    targets: Sequence[Iterable[Wire]],
    *,
    fraction_bits: int,
) -> ArithmeticReport:
    """XOR all directed-product numerators into external target words.

    Calling this function twice with the same unchanged inputs and targets
    erases the targets.  All work internal to each call is reversed before it
    returns.  Correct branching values require the documented interlacing
    promise; the emitted Boolean circuit remains a total reversible map on
    malformed inputs.
    """

    x, y, ax, ay = _validate_corner_registers(
        addable_words, removable_words, addable_active, removable_active
    )
    if not isinstance(fraction_bits, int) or fraction_bits < 1:
        raise ValueError("fraction_bits must be a positive integer")
    outputs = _validate_weight_targets(targets, len(x), fraction_bits)
    input_wires = {
        bit for word in x + y for bit in word if isinstance(bit, Wire)
    } | {bit for bit in ax + ay if isinstance(bit, Wire)}
    if set(_flatten(outputs)) & input_wires:
        raise ValueError("weight targets must be disjoint from corner inputs and masks")

    start = len(circuit.gates)
    builder = BooleanBuilder(circuit)
    weight_width = fraction_bits + 1
    factor_width = len(x[0])
    one_factor = builder.const(1, factor_width)
    zero_weight = builder.const(0, weight_width)
    scale = builder.const(1 << fraction_bits, weight_width)
    computed: list[Word] = []

    for i in range(len(x)):
        accumulator: Word = scale
        for j in range(len(y)):
            if j < i:
                valid = builder.all_bits((ax[i], ax[j], ay[j]))
                raw_numerator = builder.sub(y[j], x[i], factor_width)
                raw_denominator = builder.sub(x[j], x[i], factor_width)
            else:
                valid = builder.all_bits((ax[i], ax[j + 1], ay[j]))
                raw_numerator = builder.sub(x[i], y[j], factor_width)
                raw_denominator = builder.sub(x[i], x[j + 1], factor_width)
            numerator = builder.mux(valid, raw_numerator, one_factor)
            denominator = builder.mux(valid, raw_denominator, one_factor)
            product = builder.mul(accumulator, numerator)
            division_width = max(weight_width, factor_width)
            quotient, _remainder = builder.divmod(
                product, denominator, width=division_width
            )
            accumulator = _resize(quotient, weight_width)
        computed.append(builder.mux(ax[i], accumulator, zero_weight))

    builder.copy_then_uncompute(_flatten(computed), _flatten(outputs))
    return ArithmeticReport(
        gate_start=start,
        gate_stop=len(circuit.gates),
        qubit_count=circuit.qubit_count,
        capacity=len(x),
        word_width=weight_width,
    )


def compute_directed_weights(
    circuit: Circuit,
    addable_words: Sequence[Iterable[Bit]],
    removable_words: Sequence[Iterable[Bit]],
    addable_active: Sequence[Bit],
    removable_active: Sequence[Bit],
    *,
    fraction_bits: int,
) -> WeightWords:
    """Allocate and return clean copied directed-product numerator words."""

    x, y, ax, ay = _validate_corner_registers(
        addable_words, removable_words, addable_active, removable_active
    )
    capacity = len(x)
    if not isinstance(fraction_bits, int) or fraction_bits < 1:
        raise ValueError("fraction_bits must be a positive integer")
    flat = circuit.allocate(
        capacity * (fraction_bits + 1), "branch_weight", fresh=True
    )
    words = tuple(
        flat[offset : offset + fraction_bits + 1]
        for offset in range(0, len(flat), fraction_bits + 1)
    )
    report = xor_directed_weights(
        circuit,
        x,
        y,
        ax,
        ay,
        words,
        fraction_bits=fraction_bits,
    )
    return WeightWords(words=words, fraction_bits=fraction_bits, report=report)


def _atan_term_count(angle_fraction_bits: int) -> int:
    # At |w|<=1/2, T terms leave <2^(-2T-1).  The extra two terms leave
    # comfortable room for the O(T) fixed-point truncations in this audit.
    return max(4, (angle_fraction_bits + 2) // 2 + 2)


def _build_ratio_angle(
    builder: BooleanBuilder,
    left_mass: Word,
    right_mass: Word,
    angle_fraction_bits: int,
) -> tuple[Word, Word, Bit, Bit, Bit]:
    """Build angle data, leaving all outputs owned by ``builder``.

    Returns ``(total_angle, positive_angle, both_positive, right_only,
    right_is_larger)``.  Angle words encode radians times ``2**c``.  The total
    word contains a rounded pi/2 at the right-only endpoint for diagnostics;
    the rotation compiler uses the exact public endpoint gate instead.
    """

    if angle_fraction_bits < 2:
        raise ValueError("angle_fraction_bits must be at least two")
    c = angle_fraction_bits
    scale = 1 << c
    angle_width = c + 1
    mass_width = max(len(left_mass), len(right_mass))
    left_mass = _resize(left_mass, mass_width)
    right_mass = _resize(right_mass, mass_width)

    left_positive = builder.any_bits(left_mass)
    right_positive = builder.any_bits(right_mass)
    both_positive = builder.band(left_positive, right_positive)
    right_only = builder.band(builder.bnot(left_positive), right_positive)
    right_is_larger = builder.ult(left_mass, right_mass)
    numerator = builder.mux(right_is_larger, left_mass, right_mass)
    denominator = builder.mux(right_is_larger, right_mass, left_mass)

    # q = floor(2**c * sqrt(numerator/denominator)).  Trying one root bit at
    # a time avoids logarithms, exponentials, or a data-indexed angle table.
    root: Word = builder.const(0, angle_width)
    rhs = (0,) * (2 * c) + numerator
    comparison_width = 2 * angle_width + mass_width
    rhs = _resize(rhs, comparison_width)
    for position in range(c, -1, -1):
        trial = root[:position] + (1,) + root[position + 1 :]
        square = builder.mul(trial, trial)
        lhs = _resize(builder.mul(square, denominator), comparison_width)
        accept = builder.ule(lhs, rhs)
        root = builder.mux(accept, trial, root)

    scale_word = builder.const(scale, angle_width)
    half_word = builder.const(scale // 2, angle_width)
    reduced = builder.ult(half_word, root)
    one_minus = builder.sub(scale_word, root, angle_width)
    one_plus = builder.add(scale_word, root, angle_width + 1)
    reduced_dividend = (0,) * c + one_minus
    reduced_word, _reduced_remainder = builder.divmod(
        reduced_dividend, one_plus, width=angle_width
    )
    argument = builder.mux(reduced, reduced_word, root)

    # atan(w) = w - w^3/3 + w^5/5 - ... .  Separate accumulators keep every
    # intermediate unsigned.  All powers retain c fractional bits.
    terms = _atan_term_count(c)
    square_full = builder.mul(argument, argument)
    argument_squared = _resize(square_full[c:], angle_width)
    sum_width = angle_width + max(2, ceil(log2(terms + 1)))
    positive_sum: Word = builder.const(0, sum_width)
    negative_sum: Word = builder.const(0, sum_width)
    power: Word = argument
    for term_index in range(terms):
        odd = 2 * term_index + 1
        divisor_width = max(angle_width, odd.bit_length())
        term, _term_remainder = builder.divmod(
            power, builder.const(odd, divisor_width), width=angle_width
        )
        term = _resize(term, sum_width)
        if term_index % 2:
            negative_sum = builder.add(negative_sum, term, sum_width)
        else:
            positive_sum = builder.add(positive_sum, term, sum_width)
        if term_index + 1 < terms:
            product = builder.mul(power, argument_squared)
            power = _resize(product[c:], angle_width)
    atan_reduced = builder.sub(positive_sum, negative_sum, sum_width)
    atan_reduced = _resize(atan_reduced, angle_width)

    quarter_pi = builder.const(round(pi * scale / 4), angle_width)
    reflected = builder.sub(quarter_pi, atan_reduced, angle_width)
    small_angle = builder.mux(reduced, reflected, atan_reduced)
    half_pi = builder.const(round(pi * scale / 2), angle_width)
    complementary = builder.sub(half_pi, small_angle, angle_width)
    positive_angle = builder.mux(right_is_larger, complementary, small_angle)

    zero = builder.const(0, angle_width)
    positive_or_zero = builder.mux(both_positive, positive_angle, zero)
    total_angle = builder.mux(right_only, half_pi, positive_or_zero)
    return (
        total_angle,
        positive_angle,
        both_positive,
        right_only,
        right_is_larger,
    )


def xor_ratio_angle_word(
    circuit: Circuit,
    left_mass_word: Iterable[Bit],
    right_mass_word: Iterable[Bit],
    targets: Iterable[Wire],
    *,
    angle_fraction_bits: int,
) -> ArithmeticReport:
    """XOR the computed fixed-point ratio angle into external targets.

    The output has ``angle_fraction_bits + 1`` little-endian bits and encodes
    radians times ``2**angle_fraction_bits``.  This diagnostic oracle is also
    useful for exact basis-state tests.  Calling it twice erases its output.
    """

    left = _as_word(left_mass_word, "left mass")
    right = _as_word(right_mass_word, "right mass")
    width = max(len(left), len(right))
    left, right = _resize(left, width), _resize(right, width)
    targets = tuple(targets)
    angle_width = angle_fraction_bits + 1
    if angle_fraction_bits < 2:
        raise ValueError("angle_fraction_bits must be at least two")
    if len(targets) != angle_width or any(not isinstance(bit, Wire) for bit in targets):
        raise ValueError("angle target has the wrong width or contains constants")
    input_wires = {bit for bit in left + right if isinstance(bit, Wire)}
    if input_wires & set(targets) or len(set(targets)) != len(targets):
        raise ValueError("angle targets must be distinct from each other and the inputs")

    start = len(circuit.gates)
    builder = BooleanBuilder(circuit)
    total_angle, *_ = _build_ratio_angle(
        builder, left, right, angle_fraction_bits
    )
    builder.copy_then_uncompute(total_angle, targets)
    return ArithmeticReport(
        gate_start=start,
        gate_stop=len(circuit.gates),
        qubit_count=circuit.qubit_count,
        capacity=1,
        word_width=angle_width,
    )


def compute_ratio_angle_word(
    circuit: Circuit,
    left_mass_word: Iterable[Bit],
    right_mass_word: Iterable[Bit],
    *,
    angle_fraction_bits: int,
) -> tuple[tuple[Wire, ...], ArithmeticReport]:
    """Allocate a clean copied output for :func:`xor_ratio_angle_word`."""

    if angle_fraction_bits < 2:
        raise ValueError("angle_fraction_bits must be at least two")
    left = _as_word(left_mass_word, "left mass")
    right = _as_word(right_mass_word, "right mass")
    targets = circuit.allocate(
        angle_fraction_bits + 1, "ratio_angle", fresh=True
    )
    report = xor_ratio_angle_word(
        circuit,
        left,
        right,
        targets,
        angle_fraction_bits=angle_fraction_bits,
    )
    return targets, report


def apply_ratio_rotation(
    circuit: Circuit,
    left_mass_word: Iterable[Bit],
    right_mass_word: Iterable[Bit],
    target: Wire,
    *,
    angle_fraction_bits: int,
    control: Bit = 1,
) -> RatioRotationReport:
    """Apply a clean computed ``Ry(2*atan2(sqrt(R),sqrt(L)))``.

    The mass registers and ``control`` are preserved.  Every angle, endpoint,
    and conjunction wire is erased before return.  ``L=0<R`` uses an exact
    public controlled ``Ry(pi)`` and ``R=0`` uses the identity.
    """

    left = _as_word(left_mass_word, "left mass")
    right = _as_word(right_mass_word, "right mass")
    width = max(len(left), len(right))
    left, right = _resize(left, width), _resize(right, width)
    control = _as_bit(control, "rotation control")
    if not isinstance(target, Wire):
        raise TypeError("rotation target must be a Wire")
    input_wires = {bit for bit in left + right if isinstance(bit, Wire)}
    if target in input_wires:
        raise ValueError("rotation target must be disjoint from the mass registers")
    if isinstance(control, Wire) and control == target:
        raise ValueError("rotation control and target must be distinct")
    if angle_fraction_bits < 2:
        raise ValueError("angle_fraction_bits must be at least two")

    start = len(circuit.gates)
    builder = BooleanBuilder(circuit)
    (_total_angle, positive_angle, both_positive, right_only,
     _right_is_larger) = _build_ratio_angle(
        builder, left, right, angle_fraction_bits
    )
    positive_control = builder.all_bits((control, both_positive))
    endpoint_control = builder.all_bits((control, right_only))

    scale = 1 << angle_fraction_bits
    for position, bit in enumerate(positive_angle):
        bit_control = builder.band(positive_control, bit)
        circuit.ry(2.0 * (1 << position) / scale, target, control=bit_control)
    circuit.ry(pi, target, control=endpoint_control)
    builder.uncompute()
    return RatioRotationReport(
        gate_start=start,
        gate_stop=len(circuit.gates),
        qubit_count=circuit.qubit_count,
        mass_width=width,
        angle_fraction_bits=angle_fraction_bits,
        angle_word_width=angle_fraction_bits + 1,
        atan_terms=_atan_term_count(angle_fraction_bits),
    )


def apply_weight_tree(
    circuit: Circuit,
    weight_words: Sequence[Iterable[Bit]],
    target_index: Sequence[Wire],
    *,
    angle_fraction_bits: int,
    control: Bit = 1,
) -> TreePreparationReport:
    """Prepare amplitudes proportional to square roots of integer weights.

    ``weight_words`` must have power-of-two public capacity K and a common
    width.  ``target_index`` is a zero-initialized, little-endian ``log2(K)``
    word.  The root acts on its most significant bit, and every lower node is
    controlled by the already prepared public prefix.  All mass, prefix, and
    angle work is uncomputed; weights and external control are preserved.
    """

    weights = tuple(_as_word(word, "weight") for word in weight_words)
    capacity = len(weights)
    if not _power_of_two(capacity):
        raise ValueError("weight capacity must be a positive power of two")
    widths = {len(word) for word in weights}
    if len(widths) != 1:
        raise ValueError("weight words must have one common width")
    target = tuple(target_index)
    depth_count = capacity.bit_length() - 1
    if len(target) != depth_count or any(not isinstance(bit, Wire) for bit in target):
        raise ValueError("target_index must be a Wire word of length log2(capacity)")
    if len(set(target)) != len(target):
        raise ValueError("target_index wires must be distinct")
    input_wires = {bit for word in weights for bit in word if isinstance(bit, Wire)}
    if input_wires & set(target):
        raise ValueError("weight and target-index registers must be disjoint")
    control = _as_bit(control, "tree control")
    if isinstance(control, Wire) and control in target:
        raise ValueError("tree control must be disjoint from target_index")
    if angle_fraction_bits < 2:
        raise ValueError("angle_fraction_bits must be at least two")

    start = len(circuit.gates)
    mass_width = next(iter(widths)) + depth_count
    mass_builder = BooleanBuilder(circuit)
    levels: list[tuple[Word, ...]] = [
        tuple(_resize(word, mass_width) for word in weights)
    ]
    while len(levels[-1]) > 1:
        previous = levels[-1]
        levels.append(tuple(
            mass_builder.add(previous[position], previous[position + 1], mass_width)
            for position in range(0, len(previous), 2)
        ))

    node_reports: list[RatioRotationReport] = []
    for depth in range(depth_count):
        child_level = levels[depth_count - depth - 1]
        bit_position = depth_count - depth - 1
        for node in range(1 << depth):
            prefix_builder = BooleanBuilder(circuit)
            if depth:
                prefix_word = target[depth_count - depth :]
                prefix_match = prefix_builder.eq(
                    prefix_word, prefix_builder.const(node, depth)
                )
                node_control = prefix_builder.all_bits((control, prefix_match))
            else:
                node_control = control
            node_reports.append(apply_ratio_rotation(
                circuit,
                child_level[2 * node],
                child_level[2 * node + 1],
                target[bit_position],
                angle_fraction_bits=angle_fraction_bits,
                control=node_control,
            ))
            prefix_builder.uncompute()

    mass_builder.uncompute()
    return TreePreparationReport(
        gate_start=start,
        gate_stop=len(circuit.gates),
        qubit_count=circuit.qubit_count,
        capacity=capacity,
        mass_width=mass_width,
        angle_fraction_bits=angle_fraction_bits,
        node_reports=tuple(node_reports),
    )


def compile_branch_state(
    circuit: Circuit,
    addable_words: Sequence[Iterable[Bit]],
    removable_words: Sequence[Iterable[Bit]],
    addable_active: Sequence[Bit],
    removable_active: Sequence[Bit],
    target_index: Sequence[Wire],
    *,
    fraction_bits: int,
    angle_fraction_bits: int,
    control: Bit = 1,
) -> BranchCompilation:
    """Compute weights, prepare the branch ordinal, and erase all weights.

    Only the content/activity inputs, external control, and prepared target
    remain live.  The internally allocated weight output is XOR-erased with a
    second invocation of the same fixed oracle and then released for reuse.
    """

    addable_words = tuple(tuple(word) for word in addable_words)
    removable_words = tuple(tuple(word) for word in removable_words)
    addable_active = tuple(addable_active)
    removable_active = tuple(removable_active)
    target_index = tuple(target_index)
    corner_wires = {
        bit
        for word in addable_words + removable_words
        for bit in word
        if isinstance(bit, Wire)
    } | {
        bit
        for bit in addable_active + removable_active
        if isinstance(bit, Wire)
    }
    if corner_wires & set(target_index):
        raise ValueError("branch target must be disjoint from all corner inputs")
    start = len(circuit.gates)
    weights = compute_directed_weights(
        circuit,
        addable_words,
        removable_words,
        addable_active,
        removable_active,
        fraction_bits=fraction_bits,
    )
    tree = apply_weight_tree(
        circuit,
        weights.words,
        target_index,
        angle_fraction_bits=angle_fraction_bits,
        control=control,
    )
    erase = xor_directed_weights(
        circuit,
        addable_words,
        removable_words,
        addable_active,
        removable_active,
        weights.words,
        fraction_bits=fraction_bits,
    )
    circuit.release(weights.flat_wires)
    return BranchCompilation(
        weight_compute=weights.report,
        tree=tree,
        weight_erase=erase,
        gate_start=start,
        gate_stop=len(circuit.gates),
        qubit_count=circuit.qubit_count,
    )


__all__ = [
    "ArithmeticReport",
    "BranchCompilation",
    "RatioRotationReport",
    "TreePreparationReport",
    "WeightWords",
    "apply_ratio_rotation",
    "apply_weight_tree",
    "compile_branch_state",
    "compute_directed_weights",
    "compute_ratio_angle_word",
    "xor_directed_weights",
    "xor_ratio_angle_word",
]
