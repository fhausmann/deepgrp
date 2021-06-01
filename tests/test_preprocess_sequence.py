"""Test preprocess_sequence script."""
import sys
import gzip
import numpy as np

from deepgrp._scripts import preprocess_sequence

# pylint: disable=missing-function-docstring


def test_parse_rm(tmp_path):
    filepath = tmp_path.joinpath("inputs.fa.gz.npz").resolve()
    inputs = tmp_path.joinpath("inputs.fa.gz").resolve()
    with gzip.open(inputs, "w") as file:
        file.write(">test\nACGTNACGTN\n".encode("utf-8"))
    sys.argv = ["preprocess_sequence", str(inputs)]
    assert not filepath.exists()
    preprocess_sequence.main()
    assert filepath.exists()
    got = np.load(filepath)
    expected = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]] * 2
    np.testing.assert_array_equal(got["fwd"], np.array(expected).T)
    assert got["hash"][0] == "ff8ed7aaa145d49602bf5fdf5e5b8338"
