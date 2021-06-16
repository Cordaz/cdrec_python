import numpy as np
from typing import List, Tuple


def detect_missing_block(X: np.ndarray, na_value: float = np.nan) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int, int]]]:
    """Search for missing block in the input matrix `X` and returns
    the coordinates of each missing value in the matrix and the
    coordinates of each block (that is, the column, the first row
    and the size of the block).

    Parameters
    ----------
    X
        Input matrix of size `n` x `m`
    na_value
        The value representing the NA (default numpy nan)

    Returns
    -------
    miss_list
        List of pairs :math:`\left( i, j\right)`
        coordinates of a missing values
    miss_blocks
        List of triplets :math:`\left( j, i_{start}, size\right)`
        coordinates of a missing block (col, starting row and size)

    """
    assert len(X.shape) == 2, "ERROR: X must be a 2D array"
    # Init output
    miss_list = list()
    miss_blocks = list()

    # Loop over columns
    for j in range(X.shape[1]):
        is_block = False
        start = 0

        # Loop over rows
        for i in range(X.shape[0]):
            if (np.isnan(na_value) and np.isnan(X[i, j])) or (not np.isnan(na_value) and X[i, j] == na_value):
                # Found NA, if not in a block, init
                if not is_block:
                    is_block = True
                    start = i
                miss_list.append((i, j))
            else:
                # Found a not NA value, if was a block, end
                if is_block:
                    is_block = False
                    miss_blocks.append((j, start, i-start))
        # Loop over, check if open missing block
        if is_block:
            miss_blocks.append((j, start, X.shape[0] - start))

    return miss_list, miss_blocks
