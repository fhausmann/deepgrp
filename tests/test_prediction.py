"""Tests for deepgrp.prediction."""
import numpy as np
import pandas as pd
import pycm
import pytest
import scipy.special
import tensorflow as tf

import deepgrp.mss
import deepgrp.prediction as dgpredict
import deepgrp.preprocessing

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


@pytest.mark.parametrize("step_size", (1, 2))
def test_predict(step_size):
    class _Model:
        # pylint: disable=too-few-public-methods
        def predict_on_batch(self, _):
            tmp = np.zeros((4, 10, 3))
            tmp[:, 0, 1] = 1
            return tf.convert_to_tensor(tmp, dtype=tf.float32)

    testdata = (np.random.rand(4, 10, 5) for _ in range(3))
    got = dgpredict.predict(model=_Model(),
                            data=testdata,
                            results_shape=(50, 3),
                            step_size=step_size)
    np.testing.assert_array_equal(got.sum(axis=0), [0, 12, 0])
    for i in range(0, 12):
        i = i * step_size
        np.testing.assert_equal(got[i], [0, 1, 0])


@pytest.mark.parametrize("step_size", (10, 20))
@pytest.mark.parametrize("use_mss", (True, False))
def test_predict_complete(monkeypatch, step_size, use_mss, tmp_path,
                          randomword):

    opt = deepgrp.model.Options()
    testdata = np.zeros((100, 10))

    def _check_setup_prediction_from_options_checkpoint(options, logdir):
        assert logdir == tmp_path
        assert id(options) == id(opt)
        return randomword + "MODEL"

    monkeypatch.setattr(dgpredict, "setup_prediction_from_options_checkpoint",
                        _check_setup_prediction_from_options_checkpoint)

    def _check_fetch_validation_batch(data, steps, batchsize, vecsize):
        assert id(data) == id(testdata)
        assert steps == step_size
        assert batchsize == opt.batch_size
        assert vecsize == opt.vecsize
        return randomword + "DATA"

    monkeypatch.setattr(dgpredict, "fetch_validation_batch",
                        _check_fetch_validation_batch)

    def _check_predict(model, val_iterator, ouput_shape, steps):
        assert model == (randomword + "MODEL")
        assert val_iterator == (randomword + "DATA")
        assert ouput_shape == (4, 100)
        assert steps == step_size
        return randomword + "PREDICT"

    monkeypatch.setattr(dgpredict, "predict", _check_predict)

    def _check_apply_mss(data, options):
        assert id(opt) == id(options)
        assert data == randomword + "PREDICT"
        return randomword + "MSS"

    monkeypatch.setattr(dgpredict, "apply_mss", _check_apply_mss)

    def _check_softmax(data):
        assert data == randomword + "PREDICT"
        return randomword + "SOFTMAX"

    monkeypatch.setattr(dgpredict, "softmax", _check_softmax)

    data = deepgrp.preprocessing.Data(testdata, np.random.rand(100, 4))

    got = dgpredict.predict_complete(step_size=step_size,
                                     options=opt,
                                     logdir=tmp_path,
                                     data=data,
                                     use_mss=use_mss)
    if use_mss:
        assert got == randomword + "MSS"
    else:
        assert got == randomword + "SOFTMAX"


def test_calculate_metrics():
    truelbl = np.random.choice([0, 1, 2, 3], size=100, replace=True)
    predlbl = np.random.choice([0, 1, 2, 3], size=100, replace=True)
    got_cnf, got_stats = dgpredict.calculate_metrics(predlbl, truelbl)
    got_stats = pd.DataFrame(got_stats)
    expected = pycm.ConfusionMatrix(truelbl, predlbl)
    np.testing.assert_equal(got_cnf, expected.to_array())
    expected_stats = {
        k: expected.class_stat[k]  # pylint: disable=no-member
        for k in
        ["TPR", "TNR", "PPV", "NPV", "FPR", "FNR", "FDR", "ACC", "F1"]
    }
    expected_stats["MCC"] = expected.overall_stat["Overall MCC"]  # pylint: disable=no-member
    expected_stats["TotalACC"] = expected.overall_stat["Overall ACC"]  # pylint: disable=no-member
    expected_stats = pd.DataFrame(expected_stats)
    pd.testing.assert_frame_equal(got_stats, expected_stats)


def test_confusion_matrix():
    truelbl = np.random.choice([0, 1, 2, 3], size=100, replace=True)
    predlbl = np.random.choice([0, 1, 2, 3], size=100, replace=True)
    got = dgpredict.confusion_matrix(truelbl, predlbl)
    expected = pycm.ConfusionMatrix(truelbl, predlbl).to_array()
    np.testing.assert_equal(got, expected)


@pytest.mark.parametrize("min_len", (10, 20))
def test_filter_segments(min_len):
    segment_length = min_len * 2
    data = np.zeros(1000)
    data[110:110 + segment_length] = 1
    data[210 + segment_length:210 + 2 * segment_length] = 1
    expected = data.copy()
    data[0:min_len - 1] = 1
    data[120 + segment_length:120 + segment_length + min_len - 1] = 1
    data[(-min_len) + 1:] = 1
    dgpredict.filter_segments(data, min_len=min_len)
    np.testing.assert_equal(data, expected)
