from typing import Generator
from yungdiagram import YoungDiagram

def partitions(n: int, max_part: int | None = None) -> Generator[list[int]]:
    if n == 0:
        yield []
        return
    if max_part is None:
        max_part = n
    for k in range(min(n, max_part), 0, -1):
        for rest in partitions(n - k, k):
            yield [k] + rest


def find_yds_with_fixed_addable_cells(num_addable: int, max_size: int) -> list[YoungDiagram]:
    if num_addable < 1:
        return []

    target_distinct = num_addable - 1
    yds: list[YoungDiagram] = []
    min_size = num_addable * (num_addable - 1) // 2

    for n in range(min_size, max_size + 1, 1):
        for p in partitions(n):
            if len(set(p)) == target_distinct:
                yds.append(YoungDiagram(p))

    return yds