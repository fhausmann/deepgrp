"""Test deepgrp.training."""
from typing import Callable, Iterator

import numpy as np
import pytest
import tensorflow as tf

import deepgrp.model
import deepgrp.preprocessing
import deepgrp.training as dgtrain

# pylint: disable=missing-function-docstring


@pytest.mark.parametrize("extra_callbacks", ([], ["CALLBACK"]))
@pytest.mark.parametrize("expect_steps_per_epoch", (1, 2))
@pytest.mark.parametrize("expect_n_epochs", (5, 10))
def test_training(extra_callbacks, tmp_path, expect_steps_per_epoch,
                  expect_n_epochs):
    testdata = deepgrp.preprocessing.Data(
        np.arange(5 * 1000).reshape((5, 1000)), np.zeros((3, 1000)))
    opt = deepgrp.model.Options(n_epochs=expect_n_epochs,
                                n_batches=expect_steps_per_epoch)

    class _Model:
        # pylint: disable=too-few-public-methods
        def fit(self, dataset, verbose, epochs, steps_per_epoch,
                validation_freq, shuffle, validation_data, validation_steps,
                callbacks):
            # pylint: disable=no-self-use, too-many-arguments
            assert isinstance(dataset, tf.data.Dataset)
            assert verbose == 0
            assert epochs == expect_n_epochs
            assert steps_per_epoch == expect_steps_per_epoch
            assert validation_freq == 1
            assert not shuffle
            assert isinstance(validation_data, tf.data.Dataset)
            assert validation_steps == 1
            assert isinstance(callbacks[0], tf.keras.callbacks.TensorBoard)
            assert isinstance(callbacks[1], tf.keras.callbacks.EarlyStopping)
            assert isinstance(callbacks[2], tf.keras.callbacks.ModelCheckpoint)
            if extra_callbacks:
                assert callbacks[3] == "CALLBACK"
            assert len(callbacks) == 3 + len(extra_callbacks)

    dgtrain.training(data=(testdata, testdata),
                     options=opt,
                     model=_Model(),
                     logdir=tmp_path,
                     extra_callbacks=extra_callbacks)


@pytest.mark.parametrize("vecsize", (10, 20))
def test_calc_indices(vecsize):
    testdata = np.zeros(100)
    testdata[53:65] = 1
    got = dgtrain._calc_indices(testdata, vecsize=vecsize)  # pylint: disable=protected-access
    expected = np.arange(53 - vecsize, 64)
    np.testing.assert_array_equal(got, expected)


def test_fetch_batch():
    np.random.seed(0)
    testdata = deepgrp.preprocessing.Data(
        np.arange(5 * 1000).reshape((5, 1000)), np.zeros((3, 1000)))
    testdata.truelbl[0, 200:] = 1
    testdata.truelbl[1, :100:] = 1
    testdata.truelbl[2, 100:200:] = 1

    opt = deepgrp.model.Options()
    opt.batch_size = 100
    opt.repeat_probability = 0.3
    got = dgtrain.fetch_batch(opt, testdata)
    assert isinstance(got, Callable)
    got = got()
    assert isinstance(got, Iterator)
    vals = []
    for _ in range(100):
        tmp = next(got)
        got_diff = tmp[0][:, :-1] - tmp[0][:, 1:]
        np.testing.assert_array_equal(got_diff, -1)
        n_greater = (tmp[1].sum(axis=1) > 0).mean(axis=0)
        vals.append(n_greater[1:])
    vals = np.array(vals).flatten()
    vals = vals[vals < 0.3]
    assert vals.size < 40
