import pytest
import numpy as np
from src import issv


def test_issv(X):
    assert (issv.issv(X) == np.array([-1, 1, 1, 1, -1])).all()


def test__compute_initial_weight_vector(X, Z, V):
    assert (issv._compute_initial_weight_vector(X, Z) == V).all()


def test__get_p(V, Z):
    assert issv._get_p(V, Z) == 0
    assert issv._get_p(V, np.array([-1, 1, 1, 1, 1])) == 4
    assert issv._get_p(V, np.array([-1, 1, 1, 1, -1])) == 2
    assert issv._get_p(V, np.array([-1, 1, -1, 1, -1])) == -1
