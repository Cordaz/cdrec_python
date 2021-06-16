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
def L():
    return np.array([
        [-5.27, 5, -1.1],
        [1.63, 2.19, 2.14],
        [8.27, -2.33, -1.1],
        [4.33, 2.67, -0.41],
        [-3.86, -2.48, 1.73]
    ])


@pytest.fixture
def R():
    return np.array([
        [-0.86, 0.19, -0.48],
        [0.34, 0.91, -0.25],
        [-0.39, 0.38, 0.84]
    ])


@pytest.fixture
def V():
    return np.array([-57, 10, -46, 9, -54])


@pytest.fixture
def Z():
    return np.array([1, 1, 1, 1, 1])
