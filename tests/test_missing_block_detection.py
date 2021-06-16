import pytest
import numpy as np

from src.missing_block_detection import detect_missing_block


def test_detect_missing_block(X, X_miss):
    miss_list, miss_blocks = detect_missing_block(X)

    assert not miss_list and not miss_blocks  # Both empty

    miss_list, miss_blocks = detect_missing_block(X_miss)

    expected_miss_list = [(2, 0), (3, 0)]
    expected_miss_blocks = [(0, 2, 2)]

    assert (np.array(miss_list) == np.array(expected_miss_list)).all()
    assert (np.array(miss_blocks) == np.array(expected_miss_blocks)).all()
