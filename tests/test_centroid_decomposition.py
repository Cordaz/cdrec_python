import numpy as np
import pytest
from src.centroid_decomposition import cd


def test_centroid_decomposition(X, L, R):
    L_, R_ = cd(X)

    assert (L_.round(decimals=2) == L).all()
    assert (R_.round(decimals=2) == R).all()
