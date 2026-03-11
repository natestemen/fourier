from typing import Generator
from yungdiagram import YoungDiagram


def partitions_exact_length(
    n: int, length: int, max_part: int | None = None
) -> Generator[list[int]]:
    """
    Generate partitions of n with exactly `length` parts.
    """
    if length == 0:
        if n == 0:
            yield []
        return

    if n < length:
        return  # minimal sum is 1+...+1

    if max_part is None:
        max_part = n

    for k in range(min(max_part, n - (length - 1)), 0, -1):
        for rest in partitions_exact_length(n - k, length - 1, k):
            yield [k] + rest


def distinct_partitions(n: int, k: int) -> Generator[list[int]]:
    """
    Generate partitions of n with exactly k distinct parts.
    """
    # subtract staircase (k-1, k-2, ..., 0)
    staircase_sum = k * (k - 1) // 2
    reduced_n = n - staircase_sum

    if reduced_n < k:
        return

    for mu in partitions_exact_length(reduced_n, k):
        yield [mu[i] + (k - 1 - i) for i in range(k)]


def partitions(n: int, max_part: int | None = None) -> Generator[list[int]]:
    """
    Generate all partitions of n in non-increasing order.
    """
    if n == 0:
        yield []
        return

    if max_part is None or max_part > n:
        max_part = n

    for k in range(max_part, 0, -1):
        for rest in partitions(n - k, k):
            yield [k] + rest


def find_yds_with_fixed_addable_cells(
    num_addable: int, max_size: int
) -> list[YoungDiagram]:

    if num_addable < 1:
        return []

    k = num_addable - 1
    yds: list[YoungDiagram] = []

    # minimal size for k distinct parts (1 + 2 + ... + k)
    min_size = k * (k + 1) // 2

    for n in range(min_size, max_size + 1):
        for p in partitions(n):
            if len(set(p)) == k:
                yds.append(YoungDiagram(p))

    return yds
