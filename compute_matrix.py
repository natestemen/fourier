from yungdiagram import YoungDiagram, Cell

import numpy as np
import numpy.typing as npt


def A_matrix(diagram: YoungDiagram) -> npt.NDArray[np.float64]:
    addable = diagram.addable_cells()
    removable = diagram.removable_cells()
    removable.insert(0, Cell(-1, -1))  # Dummy cell for first column coefficients

    m = diagram.size + 1

    A = np.zeros((len(addable), len(removable)))
    for i, remove in enumerate(removable):
        for j, add in enumerate(addable):
            if i == 0:
                A[j, i] = np.sqrt(
                    (diagram + add).number_of_standard_tableaux()
                    / (m * diagram.number_of_standard_tableaux())
                )
            else:
                numerator = (
                    (m - 1)
                    * (diagram + add).number_of_standard_tableaux()
                    * (diagram - remove).number_of_standard_tableaux()
                )
                denominator = m * diagram.number_of_standard_tableaux() ** 2

                other_factor = add.content - remove.content
                A[j, i] = np.sqrt(numerator / denominator) / other_factor

    return A
