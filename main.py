import numpy as np
import numpy.typing as npt
import math


class Cell:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    @property
    def content(self) -> int:
        return self.x - self.y
    
    def __repr__(self):
        return f"Cell(x={self.x}, y={self.y})"

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
    
    @property
    def size(self) -> int:
        return sum(self.partition)
    
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
    
    def reachable_young_diagrams_by_addition(self) -> list['YoungDiagram']:
        diagrams = []
        for cell in self.addable_cells():
            new_partition = self.partition.copy()
            if cell.y == len(new_partition):
                new_partition.append(1)
            else:
                new_partition[cell.y] += 1
            diagrams.append(YoungDiagram(new_partition))
        return diagrams
            
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
    
    def reachable_young_diagrams_by_removal(self) -> list['YoungDiagram']:
        diagrams = []
        for cell in self.removable_cells():
            new_partition = self.partition.copy()
            new_partition[cell.y] -= 1
            if new_partition[cell.y] == 0:
                new_partition.pop()
            diagrams.append(YoungDiagram(new_partition))
        return diagrams
    
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
    
    def __add__(self, other: Cell) -> 'YoungDiagram':
        new_partition = self.partition.copy()
        if other.y == len(new_partition):
            new_partition.append(1)
        else:
            new_partition[other.y] += 1
        return YoungDiagram(new_partition)
    
    def __sub__(self, other: Cell) -> 'YoungDiagram':
        new_partition = self.partition.copy()
        new_partition[other.y] -= 1
        if new_partition[other.y] == 0:
            new_partition.pop()
        return YoungDiagram(new_partition)
    
    def matrix(self) -> npt.NDArray[np.float64]:
        addable = self.addable_cells()
        removable = self.removable_cells()
        removable.insert(0, Cell(-1, -1))  # Dummy cell for first column coefficients

        m = self.size + 1

        A = np.zeros((len(addable), len(removable)))
        for i, remove in enumerate(removable):
            for j, add in enumerate(addable):
                if i == 0:
                    A[j, i] = np.sqrt((self + add).number_of_standard_tableaux() / (m * self.number_of_standard_tableaux()))
                else:
                    numerator = (m - 1) * (self + add).number_of_standard_tableaux() * (self - remove).number_of_standard_tableaux()
                    denominator = m * self.number_of_standard_tableaux() ** 2

                    other_factor = add.content - remove.content
                    A[j, i] = np.sqrt(numerator / denominator) / other_factor

        return A

