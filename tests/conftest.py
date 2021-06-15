"""The name of the module is recognized by pytest.
Global scope.

"""
import pytest
import numpy as np


@pytest.fixture
def X():
    return np.array([
        [6, 3, 3],
        [-2, 2, 2],
        [-7, 1, -5],
        [-3, 4, -1],
        [2, -4, 2]
    ])


@pytest.fixture
def V():
    return np.array([-57, 10, -46, 9, -54])


@pytest.fixture
def Z():
    return np.array([1, 1, 1, 1, 1])
