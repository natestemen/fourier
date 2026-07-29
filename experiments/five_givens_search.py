"""Decisive k=4 search for exact five-Givens A-matrix charts.

The real-content closure of the 4-by-4 A-matrices has five effective
parameters, while SO(4) has dimension six.  Five rotations are therefore the
smallest plausible fixed-topology exact circuit (free diagonal signs are not
counted).  ``min_givens_count.py`` samples only five random words; this script
can enumerate all length-five words after elementary cancellation/commutation
pruning and can test a candidate chart by continuation and tangent spaces.

The default candidate was found by exhaustive search for the diagram
``(13, 8, 2)``.  A fit to one matrix is not a family-wide factorization, so
the continuation and tangent diagnostics are deliberately reported
separately.
"""

import argparse
import itertools

import numpy as np
from scipy.optimize import least_squares

from fourier.amatrix import a_matrix_from_contents


PLANES = tuple(itertools.combinations(range(4), 2))
CANDIDATE_WORD = ((0, 2), (1, 3), (0, 1), (2, 3), (0, 2))
FREE_SIGNS = np.array([1.0, 1.0, 1.0, -1.0])


def _rotation(plane: tuple[int, int], theta: float) -> tuple[np.ndarray, np.ndarray]:
    i, j = plane
    c, s = np.cos(theta), np.sin(theta)
    G = np.eye(4)
    dG = np.zeros((4, 4))
    G[i, i] = G[j, j] = c
    G[i, j], G[j, i] = -s, s
    dG[i, i] = dG[j, j] = -s
    dG[i, j], dG[j, i] = -c, c
    return G, dG


def product_and_jacobian(
    word: tuple[tuple[int, int], ...], angles: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rotations = [_rotation(plane, theta) for plane, theta in zip(word, angles)]
    prefix = [np.eye(4)]
    for G, _ in rotations:
        prefix.append(G @ prefix[-1])

    jacobian = np.empty((16, len(word)))
    left = np.eye(4)
    for index in range(len(word) - 1, -1, -1):
        G, dG = rotations[index]
        jacobian[:, index] = (left @ dG @ prefix[index]).ravel()
        left = left @ G
    return prefix[-1], jacobian


def fit_word(
    target: np.ndarray,
    word: tuple[tuple[int, int], ...],
    initial_angles: np.ndarray | None = None,
    restarts: int = 8,
    seed: int = 0,
) -> tuple[float, np.ndarray]:
    rng = np.random.default_rng(seed)
    starts = []
    if initial_angles is not None:
        starts.append(np.asarray(initial_angles, dtype=float))
    starts.extend(rng.uniform(-np.pi, np.pi, len(word)) for _ in range(restarts))

    best_residual = np.inf
    best_angles = np.zeros(len(word))
    for start in starts:
        result = least_squares(
            lambda theta: (product_and_jacobian(word, theta)[0] - target).ravel(),
            start,
            jac=lambda theta: product_and_jacobian(word, theta)[1],
            method="lm",
            ftol=1e-14,
            xtol=1e-14,
            gtol=1e-14,
            max_nfev=3000,
        )
        residual = np.linalg.norm(result.fun)
        if residual < best_residual:
            best_residual = float(residual)
            best_angles = result.x
    return best_residual, best_angles


def contents_from_gaps(gaps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build descending interlaced nodes without reversing the gap order."""
    gaps = np.asarray(gaps, dtype=float)
    if gaps.shape != (6,) or np.any(gaps <= 0.0):
        raise ValueError("a 4-by-4 family point needs six positive gaps")
    sequence = np.r_[np.cumsum(gaps[::-1])[::-1], 0.0]
    return sequence[0::2], sequence[1::2]


def target_from_gaps(gaps: np.ndarray) -> np.ndarray:
    ac, rc = contents_from_gaps(gaps)
    return a_matrix_from_contents(ac, rc) @ np.diag(FREE_SIGNS)


def canonical_words():
    """Enumerate words after adjacent cancellation and commutation pruning."""
    plane_order = {plane: index for index, plane in enumerate(PLANES)}
    for word in itertools.product(PLANES, repeat=5):
        valid = True
        for left, right in zip(word, word[1:]):
            if left == right:
                valid = False
                break
            if set(left).isdisjoint(right) and plane_order[left] > plane_order[right]:
                valid = False
                break
        if valid:
            yield word


def exhaustive_hits(
    target: np.ndarray, tolerance: float, restarts: int, seed: int
) -> list[tuple[tuple[tuple[int, int], ...], float, np.ndarray]]:
    hits = []
    for index, word in enumerate(canonical_words()):
        residual, angles = fit_word(
            target, word, restarts=restarts, seed=seed + index
        )
        if residual < tolerance:
            hits.append((word, residual, angles))
    return hits


def _skew_coordinates(tangent: np.ndarray, at: np.ndarray) -> np.ndarray:
    lie = tangent @ at.T
    return np.array(
        [lie[i, j] for i, j in itertools.combinations(range(4), 2)]
    )


def tangent_mismatch(gaps: np.ndarray, angles: np.ndarray) -> float:
    """Compare the five-dimensional family and circuit tangent spaces."""
    target = target_from_gaps(gaps)
    _, word_jacobian = product_and_jacobian(CANDIDATE_WORD, angles)
    circuit_tangents = np.column_stack(
        [
            _skew_coordinates(word_jacobian[:, j].reshape(4, 4), target)
            for j in range(5)
        ]
    )

    normalized = gaps / gaps[-1]
    family_tangents = np.empty((6, 5))
    step = 2e-6
    for j in range(5):
        plus = normalized.copy()
        minus = normalized.copy()
        plus[j] *= np.exp(step)
        minus[j] *= np.exp(-step)
        derivative = (target_from_gaps(plus) - target_from_gaps(minus)) / (2.0 * step)
        family_tangents[:, j] = _skew_coordinates(derivative, target)

    circuit_basis = np.linalg.qr(circuit_tangents)[0][:, :5]
    unexplained = family_tangents - circuit_basis @ (
        circuit_basis.T @ family_tangents
    )
    return float(np.linalg.norm(unexplained) / np.linalg.norm(family_tangents))


def continuation_trials(
    base_gaps: np.ndarray,
    base_angles: np.ndarray,
    trials: int,
    steps: int,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    residuals = []
    base = base_gaps / base_gaps[-1]
    for _ in range(trials):
        destination = np.exp(rng.uniform(-1.5, 1.5, 6))
        destination /= destination[-1]
        angles = base_angles.copy()
        residual = np.inf
        for amount in np.linspace(0.0, 1.0, steps + 1)[1:]:
            gaps = np.exp(
                (1.0 - amount) * np.log(base) + amount * np.log(destination)
            )
            residual, angles = fit_word(
                target_from_gaps(gaps),
                CANDIDATE_WORD,
                initial_angles=angles,
                restarts=0,
            )
        residuals.append(residual)
    return residuals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exhaustive", action="store_true")
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--continuation-trials", type=int, default=20)
    parser.add_argument("--continuation-steps", type=int, default=30)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    sequence = np.array([13.0, 12.0, 7.0, 6.0, 0.0, -1.0, -3.0])
    base_gaps = sequence[:-1] - sequence[1:]
    target = target_from_gaps(base_gaps)

    residual, angles = fit_word(
        target, CANDIDATE_WORD, restarts=args.restarts, seed=args.seed
    )
    print(f"candidate residual: {residual:.3e}")
    print(f"candidate angles:   {np.mod(angles + np.pi, 2 * np.pi) - np.pi}")
    print(f"tangent mismatch:   {tangent_mismatch(base_gaps, angles):.3e}")

    continuation = continuation_trials(
        base_gaps,
        angles,
        args.continuation_trials,
        args.continuation_steps,
        args.seed + 1,
    )
    print(
        "continuation:       "
        f"{sum(r < args.tol for r in continuation)}/{len(continuation)} exact, "
        f"median={np.median(continuation):.3e}, max={np.max(continuation):.3e}"
    )

    if args.exhaustive:
        hits = exhaustive_hits(target, args.tol, args.restarts, args.seed)
        print(f"exhaustive hits:     {len(hits)}")
        for word, hit_residual, hit_angles in hits[:20]:
            wrapped = np.mod(hit_angles + np.pi, 2 * np.pi) - np.pi
            print(f"  {word}: residual={hit_residual:.3e}, angles={wrapped}")


if __name__ == "__main__":
    main()
