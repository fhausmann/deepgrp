"""Test parse_rm script."""
import pathlib
import sys

import pandas as pd

from deepgrp._scripts import parse_rm

# pylint: disable=missing-function-docstring


def test_parse_rm(tmp_path):
    filepath = tmp_path.joinpath("results.bed").resolve()
    inputs = pathlib.Path(__file__).parent.joinpath(
        "test_parse_rm_input.out").resolve()
    sys.argv = ["parse_rm", "-o", str(filepath), str(inputs)]
    assert not filepath.exists()
    parse_rm.main()
    assert filepath.exists()
    expected = pd.read_csv(inputs.parent.joinpath("test_parse_rm_expect.bed"),
                           sep="\t",
                           header=None)
    got = pd.read_csv(filepath, sep="\t", header=None)
    pd.testing.assert_frame_equal(got, expected)
