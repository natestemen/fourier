import numpy as np
import math


class Cell:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

class YoungDiagram:
    def __init__(self, partition: list[int]):
        self.partition = partition
        self.cells = self._generate_cells()
    
    def _generate_cells(self) -> list[Cell]:
        cells = []
        for y, row_length in enumerate(self.partition):
            row = []
            for x in range(row_length):
                row.append(Cell(x, y))
            cells.append(row)
        return cells
    
    def __getitem__(self, index: tuple[int, int]) -> Cell:
        x, y = index
        return self.cells[y][x]
    
    def content(self, index: tuple[int, int]) -> int:
        cell = self[index]
        return cell.x - cell.y
    
    def addable_cells(self) -> list[Cell]:
        addable = []
        for row_index, row in enumerate(self.cells):
            last_cell = row[-1]
            if row_index == 0:
                addable.append(Cell(last_cell.x + 1, last_cell.y))
            else:
                above_row = self.cells[row_index - 1]
                if len(above_row) > len(row):
                    addable.append(Cell(last_cell.x + 1, last_cell.y))
        addable.append(Cell(0, last_cell.y + 1))
        return addable
            
    def draw_addable_cells(self) -> list[Cell]:
        """draws the young diagram with the addable cells marked with '+'.
        using the function addable_cells.
        """
        max_length = max(self.partition) + 1 if self.partition else 1
        diagram = np.full((len(self.partition) + 1, max_length), ' ')
        for y, row_length in enumerate(self.partition):
            for x in range(row_length):
                diagram[y, x] = '■'
        for cell in self.addable_cells():
            diagram[cell.y, cell.x] = '+'
        for row in diagram:
            print(' '.join(row))

    def removable_cells(self) -> list[Cell]:
        removable = []
        for row_index, row in enumerate(self.cells):
            last_cell = row[-1]
            if row_index == len(self.cells) - 1:
                removable.append(last_cell)
            else:
                below_row = self.cells[row_index + 1]
                if len(below_row) < len(row):
                    removable.append(last_cell)
        return removable
    
    def draw_removable_cells(self) -> list[Cell]:
        """draws the young diagram with the removable cells marked with a red square.
        the function removable_cells is used.
        """
        max_length = max(self.partition) if self.partition else 0
        diagram = np.full((len(self.partition), max_length), ' ')
        for y, row_length in enumerate(self.partition):
            for x in range(row_length):
                diagram[y, x] = '■'
        for cell in self.removable_cells():
            diagram[cell.y, cell.x] = '□'
        for row in diagram:
            print(' '.join(row))

    def draw(self):
        max_length = max(self.partition) if self.partition else 0
        diagram = np.full((len(self.partition), max_length), ' ')
        for y, row_length in enumerate(self.partition):
            for x in range(row_length):
                diagram[y, x] = '■'
        for row in diagram:
            print(' '.join(row))
    
    def hook_length(self, index: tuple[int, int]) -> int:
        x, y = index
        right = len(self.cells[y][x+1:])
        below = []
        for row in self.cells[y+1:]:
            if x < len(row):
                below.append(row[x])

        below = len(below)
        return right + below + 1

    def number_of_standard_tableaux(self) -> int:
        n = sum(self.partition)
        hook_lengths = []
        for y, row in enumerate(self.cells):
            for x, cell in enumerate(row):
                hook_lengths.append(self.hook_length((x, y)))
        product_of_hook_lengths = np.prod(hook_lengths)
        return int(math.factorial(n) / product_of_hook_lengths)
    