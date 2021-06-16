"""Function to compute the Centroid Decomposition (CD) of a Matrix X (n x m).

"""
import numpy as np
from typing import Tuple
from copy import deepcopy

from src.issv import issv


def cd(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """

    Parameters
    ----------
    X
        Input matrix of size `n` x `m`

    Returns
    -------
    L
        Loading matrix of size `n` x `m`
    R
        Relevance matrix of size `m` x `m`

    """
    assert len(X.shape) == 2, "ERROR: X must be a 2D array"
    X_copy = deepcopy(X)

    # Init L, R
    L = np.zeros(X.shape)
    R = np.zeros((X.shape[1], X.shape[1]))

    for j in range(X_copy.shape[1]):
        Z = issv(X_copy)
        C_j = np.dot(X_copy.T, Z)
        R_j = C_j / np.linalg.norm(C_j, ord=2)
        L_j = np.dot(X_copy, R_j)
        X_copy = X_copy - np.outer(L_j, R_j.T)
        L[:, j] = L_j
        R[:, j] = R_j

    return L, R
