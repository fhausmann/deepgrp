"""Test deepgrp.mss."""
import numpy as np
import pytest

import deepgrp.mss as dgmss

# pylint: disable=missing-function-docstring


@pytest.mark.parametrize("min_mss_len", [0, 3, 10])
@pytest.mark.parametrize("xdrop_len", [-1, 0, 10])
def test_find_mss_labels(min_mss_len, xdrop_len):
    scores = np.array([1, 1, -1, 1, 1, -4, 1, 1, -10, 1, 1, -1, 1, 1],
                      dtype=np.float64)
    labels = np.array([1, 1, 0, 1, 1, 0, 2, 2, 0, 1, 1, 0, 2, 2], dtype=int)
    nof_labels = 3
    got = dgmss.find_mss_labels(scores, labels, nof_labels, min_mss_len,
                                xdrop_len)
    got = got.argmax(axis=1)
    expected = labels.copy()
    if min_mss_len == 0:
        expected[2] = 1
        expected[11] = 1
    np.testing.assert_equal(got, expected)
