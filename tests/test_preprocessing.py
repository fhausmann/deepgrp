"""Test deepgrp.preprocessing."""
import numpy as np
import pandas as pd
import pytest

import deepgrp.preprocessing as dgpreprocess

# pylint: disable=missing-function-docstring


def test_preprocess_y(tmp_path):

    data = [["chr1", 5, 10, 2, "X"], ["chr2", 6, 11, 5, "X"],
            ["chr1", 13, 15, 4, "X"], ["chr1", 16, 18, 7, "X"]]
    data = pd.DataFrame(data)
    filepath = tmp_path.joinpath("data.bed")
    data.to_csv(filepath, sep="\t", index=False, header=False)
    got = dgpreprocess.preprocess_y(filename=str(filepath),
                                    chromosom="chr1",
                                    length=20,
                                    repeats_to_search=[1, 2, 3, 4])
    expected = np.zeros((5, 20))
    expected[0, :5] = 1
    expected[2, 5:10] = 1
    expected[0, 10:13] = 1
    expected[4, 13:15] = 1
    expected[0, 15:] = 1
    np.testing.assert_equal(got, expected)


@pytest.mark.parametrize("start_n", (0, 10, 20))
@pytest.mark.parametrize("end_n", (0, 10, 20))
def test_drop_start_end_n(start_n, end_n):
    testdata = np.zeros((5, 100))
    if end_n == 0:
        testdata[1, start_n:] = 1
    else:
        testdata[1, start_n:-end_n] = 1
        testdata[4, -end_n:] = 1
    testdata[4, :start_n] = 1
    truelbl = np.arange(100).reshape((1, 100))
    got_x, got_y = dgpreprocess.drop_start_end_n(testdata, truelbl)
    assert got_y.shape == (1, 100 - start_n - end_n - 1)
    assert got_x.shape == (5, 100 - start_n - end_n - 1)
    np.testing.assert_equal(got_x.sum(axis=1),
                            [0, 100 - start_n - end_n - 1, 0, 0, 0])
    assert got_y[0, 0] == start_n
    assert got_y[0, -1] == 100 - end_n - 2
    np.testing.assert_equal(got_y[0, :-1] - got_y[0, 1:], -1)
