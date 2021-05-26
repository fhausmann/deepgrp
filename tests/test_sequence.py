"""Test deepgrp.sequence."""
import numpy as np
import pytest

import deepgrp.sequence as dgseq

# pylint: disable=missing-function-docstring


def test_one_hot_encode_dna_sequence():
    data = "NNNN" + "".join(
        np.random.choice(["A", "C", "G", "T", "N"], size=100, replace=True))
    data += "NNNN"
    startpos, one_hot_encoding = dgseq.one_hot_encode_dna_sequence(data)
    np.testing.assert_equal(one_hot_encoding.sum(axis=0), 1)
    tbl = str.maketrans({"A": "0", "C": "1", "G": "2", "T": "3", "N": "4"})
    expected = data.translate(tbl)
    while expected.startswith("4"):
        expected = expected[1:]
    while expected.endswith("4"):
        expected = expected[:-1]
    expected = np.array(list(expected)).astype(int)
    got = one_hot_encoding.argmax(axis=0)
    np.testing.assert_equal(got, expected)
    for i in range(startpos):
        assert data[i] == "N"
    assert data[startpos] != "N"


@pytest.mark.parametrize("begin", (3, 5, 50))
@pytest.mark.parametrize("startpos", [0, 10, 22, 33])
@pytest.mark.parametrize("endpos", [0, 44, 54, 62])
@pytest.mark.parametrize("label", [1, 2, 3])
def test_get_segments(begin, startpos, endpos, label):
    testdata = np.zeros(100, dtype=int)
    if endpos > 0:
        testdata[startpos:endpos] = label
    got_start, got_end, got_label = dgseq.get_segments(testdata, begin)
    expected_start = max(startpos, begin) if endpos > begin else 99
    expected_end = endpos if endpos > begin else 100
    expected_label = label if endpos > begin else 0
    assert got_start == expected_start
    assert got_end == expected_end
    assert got_label == expected_label


@pytest.mark.parametrize("stride", [1, 2, 3])
def test_get_max(stride):
    testdata = np.zeros((10, 100, 5), dtype=np.float32)
    testdata[:, 0, :] = 1.0
    output = np.zeros((10000, 5), dtype=np.float32)
    got = dgseq.get_max(output, testdata, stride=stride)
    for i in range(0, stride * 10, stride):
        np.testing.assert_equal(got[i], 1)
        got[i] -= 1.0
    np.testing.assert_equal(got, 0)
