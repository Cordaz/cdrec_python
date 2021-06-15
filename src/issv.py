"""Incremental Scalable Sign Vector implementation.

"""
import numpy as np


def issv(X: np.ndarray) -> np.ndarray:
    """Compute the maximizing vector :math:`Z \in \left[ 1, -1\right]^n`
    s.t. it maximizes :math:`\Vert X^T \dot Z \Vert`

    Parameters
    ----------
    X
        Input matrix of size `n` x `m`

    Returns
    -------
    Z
        Sign Vector :math:`Z \in \left[ 1, -1\right]^n`

    """
    assert len(X.shape) == 2, "ERROR: X must be a 2D array"
    n = X.shape[0]

    # Init column vector Z
    Z = np.ones(n)

    V = _compute_initial_weight_vector(X, Z)

    p = _get_p(V, Z)

    while p >= 0:
        Z[p] = -1 * Z[p]  # Flipping

        for i in range(n):
            if i != p:
                # Update V vector
                V[i] = V[i] - 2 * np.dot(X[i, :], X[p, :].T)
        p = _get_p(V, Z)

    return Z


def _compute_initial_weight_vector(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Initialize the weight vector V computed as:

    .. math::

         v_i = z_i \left( z_i * X_{i*} \dot S - (X_{i*} \dot X_{i*}^T\right), v_i \in V, z_i \in Z


    .. math::

         S = \sum_{i=0}^{n-1} z_i * X_{i*}^T

    Parameters
    ----------
    X
        Input matrix with size :math:`n` x :math:`m`
    Z
        Sign vector (for the initialization is usually set to :math:`\left[ 1,\dots ,1\right]`

    Returns
    -------
    V
        Weight vector

    """
    # Init V
    V = np.zeros(X.shape[0])

    # Compute S
    S = np.zeros(X.shape[1])
    for i in range(X.shape[0]):
        S_ = Z[i] * X[i, :].T
        S = S + S_

    # Compute V element-wise
    for i in range(X.shape[0]):
        V[i] = Z[i] * (Z[i] * np.dot(X[i, :], S) - np.dot(X[i, :], X[i, :].T))

    return V


def _get_p(V: np.ndarray, Z: np.ndarray) -> int:
    """Retrieves the index of V such that :math:`v_i * z_i < 0` and
    :math:`v_i` has the largest absolute value.

    Parameters
    ----------
    V
        Weight vector
    Z
        Sign vector

    Returns
    -------
    p
        The index of V as described above.
        If there are no index that satisfies the condition, -1 is returned.

    """
    mask = V * Z < 0
    if not np.any(mask):
        return -1
    arg = np.argmax(np.absolute(V[mask]))
    p = np.arange(V.shape[0])[mask][arg]

    return p
