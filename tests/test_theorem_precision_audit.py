"""Independent exact-rational audit of the bounded-word potential argument.

This is a mathematical reference, not a reversible implementation.  In
particular its dictionaries are only test conveniences; the theorem routes
the same records through fixed sorting networks.  Every comparison below
uses rational bounds, not floating logarithms.
"""

from fractions import Fraction
from math import factorial

import pytest


def _log_rational_bounds(value, terms):
    """Enclose log(value) for positive integers using an explicit tail."""
    exponent = value.bit_length() - 1
    base = 1 << exponent

    def series(z):
        return 2 * sum((z ** (2*j + 1) / (2*j + 1)
                        for j in range(terms)), Fraction())

    lower = series(Fraction(value - base, value + base))
    lower += exponent * series(Fraction(1, 3))
    return lower, lower + Fraction(exponent + 1, 3 ** (2*terms))


def _far_boxes(target, width):
    box = target // width
    return [2 * (box // 2 + offset) + child
            for offset in (-1, 0, 1) for child in (0, 1)
            if 2 * (box // 2 + offset) + child >= 0
            and abs(2 * (box // 2 + offset) + child - box) > 1]


def _rational_potential(positions, charges, span, order, log_terms, kernel):
    outputs = [Fraction() for _ in positions]
    largest_word = 1
    for level in range(span.bit_length()):
        width = 1 << level
        moments = {}
        for point, charge in zip(positions, charges):
            box = point // width
            z = Fraction(2 * (point % width) - width, width)
            entry = moments.setdefault(box, [Fraction() for _ in range(order+1)])
            for j in range(order+1):
                entry[j] += charge * z**j
        for i, point in enumerate(positions):
            for box in _far_boxes(point, width):
                moment = moments.get(box, [Fraction() for _ in range(order+1)])
                twice_distance = 2*point - (2*box+1)*width
                ratio = Fraction(width, twice_distance)
                assert abs(ratio) <= Fraction(1, 3)
                if kernel == "inverse":
                    value = sum((moment[j] * ratio**j for j in range(order+1)), Fraction())
                    value *= Fraction(2, abs(twice_distance))
                else:
                    logarithm = _log_rational_bounds(abs(twice_distance), log_terms)[0]
                    logarithm -= _log_rational_bounds(2, log_terms)[0]
                    value = moment[0] * logarithm
                    value -= sum((moment[j] * ratio**j / j for j in range(1, order+1)), Fraction())
                outputs[i] += value
                largest_word = max(largest_word, value.numerator.bit_length(),
                                   value.denominator.bit_length(),
                                   outputs[i].numerator.bit_length(),
                                   outputs[i].denominator.bit_length())
    if kernel == "inverse":
        for i, point in enumerate(positions):
            outputs[i] += sum((charge for source, charge in zip(positions, charges)
                               if abs(point-source) == 1), Fraction())
    return outputs, largest_word


@pytest.mark.parametrize("span", [1, 2, 3, 7, 16, 31])
def test_dyadic_interactions_partition_every_non_nearest_pair(span):
    for target in range(span+1):
        for source in range(span+1):
            count = sum(source // (1 << level) in _far_boxes(target, 1 << level)
                        for level in range(span.bit_length()))
            assert count == int(abs(target-source) >= 2)
        assert all(len(_far_boxes(target, 1 << level)) <= 3
                   for level in range(span.bit_length()))


def test_normalized_moments_have_common_dyadic_denominator_not_product_growth():
    fractional_bits = 11
    positions = [0, 1, 7, 16, 19, 31, 32, 63]
    charges = [Fraction(n, 1 << fractional_bits) for n in [-7, 9, 1023, -13, 2, 8, -1, 19]]
    for level in range(7):
        width = 1 << level
        for order in range(13):
            for box in {position // width for position in positions}:
                moment = sum((charge * Fraction(2*(point % width)-width, width)**order
                              for point, charge in zip(positions, charges)
                              if point // width == box), Fraction())
                common_denominator = 1 << (fractional_bits + level*order)
                assert common_denominator % moment.denominator == 0
                scaled = moment * common_denominator
                assert scaled.denominator == 1
                assert abs(scaled) <= len(positions) * common_denominator


@pytest.mark.parametrize("positions", [
    [0, 1, 2, 3, 4],
    [0, 1, 15, 16, 31],
    [0, 3, 127, 1024, 2**20+3],
])
def test_exact_rational_inverse_truncation_and_public_word_envelope(positions):
    charges = [Fraction(j+1, 8) for j in range(len(positions))]
    span, order, log_terms = max(positions), 10, 20
    approximate, largest_word = _rational_potential(
        positions, charges, span, order, log_terms, "inverse")
    for i, target in enumerate(positions):
        exact = sum((charge / abs(target-source)
                     for j, (source, charge) in enumerate(zip(positions, charges))
                     if i != j), Fraction())
        assert abs(approximate[i]-exact) <= exact / 3**(order+1)
        assert abs(approximate[i]-exact) <= len(positions) / Fraction(3**(order+1))
    # Deliberately loose, symbolic bit envelope: no numerical rounding model.
    depth = span.bit_length()
    word_bound = 4 * (6*depth+3) * (
        3 + (2*order+2)*(depth+4) + order*(order+1).bit_length()
        + 4*log_terms*(depth+4+(2*log_terms+1).bit_length())
        + len(positions).bit_length() + 2*(depth+5).bit_length() + 32)
    assert largest_word <= word_bound


@pytest.mark.parametrize("positions", [[0, 1, 2], [0, 7, 8, 31, 63], [0, 1, 127, 128, 2**16]])
def test_signed_log_truncation_against_independent_rational_enclosures(positions):
    charges = [Fraction(-1 if j % 2 == 0 else 1) for j in range(len(positions))]
    span, order, log_terms = max(positions), 10, 20
    approximate, _ = _rational_potential(positions, charges, span, order, log_terms, "log")
    count, depth = len(positions), span.bit_length()
    bound = Fraction(3*count, 2*(order+1)*3**(order+1))
    bound += Fraction(count*(depth+5), 3**(2*log_terms))
    for i, target in enumerate(positions):
        lower = upper = Fraction()
        for j, (source, charge) in enumerate(zip(positions, charges)):
            if i == j:
                continue
            low, high = _log_rational_bounds(abs(target-source), 40)
            lower += charge * (low if charge > 0 else high)
            upper += charge * (high if charge > 0 else low)
        # Stronger than interval overlap: every point in the direct enclosure
        # lies inside the multipole truncation ball.
        assert max(abs(approximate[i]-lower), abs(approximate[i]-upper)) <= bound


@pytest.mark.parametrize("merged", [[0], [0, 1, 2], [0, 1, 4, 15, 16], [0, 3, 127, 128, 2**20]])
def test_integer_interlacing_residue_conditioning_is_exact(merged):
    xnodes, ynodes = merged[::2], merged[1::2]
    span = max(1, merged[-1]-merged[0])
    alpha_squared = []
    for i, x in enumerate(xnodes):
        weight = Fraction(1)
        for y in ynodes:
            weight *= abs(x-y)
        for j, other in enumerate(xnodes):
            if i != j:
                weight /= abs(x-other)
        alpha_squared.append(weight)
        assert Fraction(1, span**2) <= weight <= 1
    assert sum(alpha_squared) == 1
    for y in ynodes:
        inverse_beta_squared = sum((weight / (x-y)**2
                                    for x, weight in zip(xnodes, alpha_squared)), Fraction())
        assert Fraction(1, span**2) <= inverse_beta_squared <= 1


def test_atanh_series_has_explicit_common_factorial_denominator():
    for numerator, denominator in [(0, 1), (1, 3), (31, 97)]:
        z = Fraction(numerator, denominator)
        for terms in [1, 2, 7, 19]:
            value = 2 * sum((z**(2*j+1) / (2*j+1) for j in range(terms)), Fraction())
            common = denominator**(2*terms-1) * factorial(2*terms-1)
            assert (value*common).denominator == 1
