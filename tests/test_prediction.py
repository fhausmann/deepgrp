"""Tests for deepgrp.prediction."""
import pytest
import numpy as np
import scipy.special

import deepgrp.prediction as dgpredict
import deepgrp.mss

#pylint: disable=no-self-use, missing-function-docstring, missing-class-docstring


@pytest.mark.parametrize("step_size", [2, 4])
@pytest.mark.parametrize("batch_size", [3, 10])
@pytest.mark.parametrize("vecsize", [20, 30])
def test_fetch_validation_batch(step_size, batch_size, vecsize):
    testdata = np.random.rand(5, 200)
    got = dgpredict.fetch_validation_batch(data=testdata,
                                           step_size=step_size,
                                           batch_size=batch_size,
                                           vecsize=vecsize)
    assert got.element_spec.shape.as_list() == [None, vecsize, 5]
    i = 0
    total = np.ceil((testdata.shape[1] - vecsize) / step_size)
    for tmp in got.as_numpy_iterator():
        expected_shape = (min(total, batch_size), vecsize, testdata.shape[0])
        assert tmp.shape == expected_shape
        for element in tmp:
            np.testing.assert_allclose(
                element, testdata.T[i * step_size:i * step_size + vecsize])
            i += 1
        total -= tmp.shape[0]
    assert total == 0


@pytest.mark.parametrize("expected_min_mss_len", [2, 4])
@pytest.mark.parametrize("expected_xdrop_len", [3, 10])
@pytest.mark.parametrize("n_classes", [3, 5])
def test_apply_mss(monkeypatch, expected_min_mss_len, expected_xdrop_len,
                   n_classes):

    testdata = np.random.rand(200, n_classes)
    test_labels = testdata.argmax(axis=1)
    expected_scores = np.minimum(testdata.max(axis=1) + 1e-6, 0.99)
    expected_scores = np.log(expected_scores / (1 - expected_scores))

    def _check_find_mss_labels(scores, results_classes, nof_labels,
                               min_mss_len, xdrop_len):
        assert nof_labels == n_classes
        np.testing.assert_array_equal(results_classes, test_labels)
        assert min_mss_len == expected_min_mss_len
        assert xdrop_len == expected_xdrop_len
        scores[test_labels == 0] /= -10
        np.testing.assert_allclose(scores, expected_scores)
        return "OK"

    monkeypatch.setattr(deepgrp.mss, "find_mss_labels", _check_find_mss_labels)
    opt = deepgrp.model.Options(min_mss_len=expected_min_mss_len,
                                xdrop_len=expected_xdrop_len)
    got = dgpredict.apply_mss(testdata, opt)
    assert got == "OK"


def test_softmax():
    testdata = np.random.rand(200, 10)
    got = dgpredict.softmax(testdata)
    expected = scipy.special.softmax(testdata, axis=1)
    np.testing.assert_allclose(got, expected)


def test_predict():
    pytest.xfail("Not implemented")


def test_predict_complete():
    pytest.xfail("Not implemented")


def test_calculate_multiclass_matthews_cc():
    pytest.xfail("Not implemented")


def test_calculate_metrics():
    pytest.xfail("Not implemented")


def test_confusion_matrix():
    pytest.xfail("Not implemented")


def test_filter_segments():
    pytest.xfail("Not implemented")
