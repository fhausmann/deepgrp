"""Test deepgrp.optimization."""

import pytest

import deepgrp.optimization as dgopt
import deepgrp.model as dgmodel

# pylint: disable=missing-function-docstring


@pytest.mark.parametrize("testcase, expected", [
    ({
        "somecustomstring": "hello"
    }, {
        "somecustomstring": "hello"
    }),
    ({
        "n_batches": 10
    }, {
        "n_batches": 10
    }),
    ({
        "units": 10
    }, {
        "units": 10
    }),
    ({
        "units": "10"
    }, {
        "units": 10
    }),
    ({
        "vecsize": "100"
    }, {
        "vecsize": 100
    }),
    ({
        "vecsize": "100"
    }, {
        "vecsize": 100
    }),
])
def test_update_options(testcase, expected):
    opt = dgmodel.Options()
    for key, value in testcase.items():
        try:
            assert opt[key] != expected[key]
            assert opt[key] != value
        except KeyError:
            pass
    got = dgopt._update_options(opt, testcase)  # pylint: disable=protected-access
    for key, value in expected.items():
        assert got[key] == value


def test_build_and_optimize():
    pytest.xfail("Not implemented")


def test_run_a_trial():
    pytest.xfail("Not implemented")
