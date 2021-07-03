"""Tests for deepgrp.model."""
#pylint: disable=no-self-use, missing-function-docstring, missing-class-docstring
import json
import pathlib
import os

import numpy as np
import pytest
import tensorflow as tf
import toml
from freezegun import freeze_time
# pylint: disable=no-name-in-module
from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.keras import keras_parameterized, testing_utils
# pylint: enable=no-name-in-module

from deepgrp import model


@freeze_time('19900305121503')
def test_create_logdir(tmp_path):
    opt = model.Options(project_root_dir=(str(tmp_path)))
    got = model.create_logdir(opt)
    assert got == str(tmp_path.joinpath("tf_logs", "run-19900305121503"))


_TESTCASES_OPTIONS = [({}, {
    'project_root_dir': '.',
    'repeats_to_search': [1, 2, 3, 4],
    'vecsize': 150,
    'n_epochs': 200,
    'n_batches': 250,
    'early_stopping_th': 10,
    'batch_size': 256,
    'repeat_probability': 0.3,
    'optimizer': 'RMSprop',
    'learning_rate': 0.001,
    'momentum': 0.9,
    'rho': 0.9,
    'epsilon': 1e-10,
    'rnn': 'GRU',
    'units': 32,
    'dropout': 0.25,
    'attention': False,
    'min_mss_len': 50,
    'xdrop_len': 50
}),
                      ({
                          'project_root_dir': 'test',
                          'repeats_to_search': [1, 2, 3],
                          'vecsize': 157,
                          'n_epochs': 206,
                          'n_batches': 256,
                          'early_stopping_th': 11,
                          'batch_size': 259,
                          'repeat_probability': 0.2,
                          'optimizer': 'Adam',
                          'learning_rate': 0.002,
                          'momentum': 0.8,
                          'rho': 0.8,
                          'epsilon': 1e-08,
                          'rnn': 'LSTM',
                          'units': 326,
                          'dropout': 0.256,
                          'attention': True,
                          'min_mss_len': 507,
                          'xdrop_len': 507,
                          'some_additional': 'test'
                      }, {
                          'project_root_dir': 'test',
                          'repeats_to_search': [1, 2, 3],
                          'vecsize': 157,
                          'n_epochs': 206,
                          'n_batches': 256,
                          'early_stopping_th': 11,
                          'batch_size': 259,
                          'repeat_probability': 0.2,
                          'optimizer': 'Adam',
                          'learning_rate': 0.002,
                          'momentum': 0.8,
                          'rho': 0.8,
                          'epsilon': 1e-08,
                          'rnn': 'LSTM',
                          'units': 326,
                          'dropout': 0.256,
                          'attention': True,
                          'min_mss_len': 507,
                          'xdrop_len': 507,
                          'some_additional': 'test'
                      })]


class TestOptions:
    @pytest.mark.parametrize('init_args, expected', _TESTCASES_OPTIONS)
    def test_init(self, init_args, expected):
        got = (model.Options)(**init_args)
        for attribute, value in expected.items():
            assert getattr(got, attribute) == value

    @pytest.mark.parametrize('init_args, expected', _TESTCASES_OPTIONS)
    def test_fromdict(self, init_args, expected):
        got = model.Options()
        got.fromdict(init_args)
        for attribute, value in expected.items():
            assert getattr(got, attribute) == value

    @pytest.mark.parametrize('init_args, expected', _TESTCASES_OPTIONS)
    def test_todict(self, init_args, expected):
        got = (model.Options)(**init_args).todict()
        assert got == expected

    @pytest.mark.parametrize('init_args, expected', _TESTCASES_OPTIONS)
    def test_to_toml(self, init_args, expected, tmp_path):
        with tmp_path.joinpath('testfile.toml').open('w') as (file):
            (model.Options)(**init_args).to_toml(file)
        got = toml.loads(tmp_path.joinpath('testfile.toml').read_text())
        assert got == expected

    @pytest.mark.parametrize('init_args, expected', _TESTCASES_OPTIONS)
    def test_from_toml(self, init_args, expected, tmp_path):
        tmp_path.joinpath('testfile.toml').write_text(toml.dumps(init_args))
        got = model.Options.from_toml(str(tmp_path.joinpath('testfile.toml')))
        for attribute, value in expected.items():
            assert getattr(got, attribute) == value


@pytest.mark.parametrize('optimizer', ['RMSprop', 'Adam', 'somethingelse'])
@pytest.mark.parametrize('learning_rate', [1e-05, 0.001])
@pytest.mark.parametrize('rho', [0.8, 0.9])
@pytest.mark.parametrize('momentum', [0.7, 0.6])
@pytest.mark.parametrize('epsilon', [1e-07, 1e-08])
def test_get_opimizer(optimizer, rho, momentum, epsilon, learning_rate):
    opt = model.Options(optimizer=optimizer,
                        rho=rho,
                        learning_rate=learning_rate,
                        momentum=momentum,
                        epsilon=epsilon)
    got = model._get_optimizer(opt)  # pylint: disable=protected-access
    if optimizer not in ('Adam', 'RMSprop'):
        assert got == optimizer
        return
    assert isinstance(got, tf.keras.optimizers.Optimizer)
    got = got.get_config()
    assert got["learning_rate"] == learning_rate
    assert got["epsilon"] == epsilon
    if optimizer == 'RMSprob':
        assert got["rho"] == rho
        assert got["momentum"] == "momentum"
    elif optimizer == "Adam":
        assert got["beta_2"] == rho
        assert got["beta_1"] == momentum


@pytest.mark.parametrize('rnn', ['GRU', 'LSTM', 'somethingelse'])
@pytest.mark.parametrize('units', [5, 10])
@pytest.mark.parametrize('dropout', [0.1, 0.2])
@pytest.mark.parametrize('attention', [True, False])
def test_get_brnn_layer(rnn, units, dropout, attention):
    opt = model.Options(rnn=rnn,
                        units=units,
                        dropout=dropout,
                        attention=attention)
    got = model._get_brnn_layer(opt)  # pylint: disable=protected-access
    assert isinstance(got, tf.keras.layers.RNN)
    if rnn == "LSTM":
        assert got.name == "BLSTM"
    else:
        assert got.name == "BGRU"
    output = got(tf.zeros((10, 6, 2)))
    if attention and not rnn == 'LSTM':
        assert len(output) == 2
        assert output[0].shape == (10, 6, units)
        assert output[1].shape == (10, units)
    else:
        assert output.shape == (10, 6, units)
    got = got.get_config()
    assert got["dropout"] == dropout
    assert got["return_sequences"]
    assert got["return_state"] == (attention if rnn != "LSTM" else False)


def test_get_dna_encoding():
    outputs = model._get_dna_encoding()  # pylint: disable=protected-access
    assert outputs == [3, 2, 1, 0, 4]


class TestReverseComplementLayer(keras_parameterized.TestCase):
    @keras_parameterized.run_all_keras_modes
    def test_basic(self):
        """Test layer creation."""
        testing_utils.layer_test((model.ReverseComplement),
                                 kwargs={'complements': [3, 2, 1, 0, 4]},
                                 input_shape=(1, 10, 5))

    @tf_test_util.run_in_graph_and_eager_modes
    def test_mmdpp_weights(self):
        """Test weights creation."""
        layer = model.ReverseComplement(complements=[3, 2, 1, 0, 4])
        layer.build((1, 10, 5))
        self.assertEqual(len(layer.trainable_weights), 0)
        self.assertEqual(len(layer.weights), 0)

    @keras_parameterized.run_all_keras_modes
    def test_output(self):
        """Test full run."""
        inputs = tf.keras.layers.Input(shape=(6, 5))
        add_layer = model.ReverseComplement(complements=[3, 2, 1, 0, 4])
        output = add_layer(inputs)
        self.assertEqual(output.shape.as_list(), [None, 6, 5])
        testmodel = tf.keras.models.Model(inputs, output)
        testmodel.run_eagerly = testing_utils.should_run_eagerly()
        if int(tf.__version__.split(".")[1]) < 3:
            should = testing_utils.should_run_tf_function()
            testmodel._experimental_run_tf_function = should  # pylint: disable=protected-access
        input_data = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0,
                                                 0], [0, 0, 1, 0, 0],
                               [0, 0, 0, 1, 0], [0, 0, 0, 0, 1],
                               [1, 0, 0, 0, 0]]).reshape((1, -1, 5))
        out = testmodel.predict(input_data)
        self.assertEqual(out.shape, (1, 6, 5))
        self.assertEqual(add_layer.compute_mask(inputs, [None, None]), None)
        with self.assertRaisesRegex(
                TypeError,
                'does not support masking, but was passed an input_mask'):
            add_layer.compute_mask(inputs, input_data)
        expected = np.array([[0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [1, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0], [0, 0, 1, 0, 0],
                             [0, 0, 0, 1, 0]]).reshape((1, -1, 5))
        np.testing.assert_equal(out, expected)

    @keras_parameterized.run_all_keras_modes
    def test_serialization(self):
        """Test serialization."""
        inputs = tf.keras.layers.Input(shape=(6, 5))
        add_layer = model.ReverseComplement(complements=[3, 2, 1, 0, 4])
        output = add_layer(inputs)
        self.assertEqual(output.shape.as_list(), [None, 6, 5])
        testmodel = tf.keras.models.Model(inputs, output)
        testmodel.run_eagerly = testing_utils.should_run_eagerly()
        if int(tf.__version__.split(".")[1]) < 3:
            should = testing_utils.should_run_tf_function()
            testmodel._experimental_run_tf_function = should  # pylint: disable=protected-access
        input_data = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0,
                                                 0], [0, 0, 1, 0, 0],
                               [0, 0, 0, 1, 0], [0, 0, 0, 0, 1],
                               [1, 0, 0, 0, 0]]).reshape((1, -1, 5))
        expect = testmodel.predict(input_data)
        model_config = testmodel.get_config()
        recovered_model = tf.keras.models.Model.from_config(model_config)
        output = recovered_model.predict(input_data)
        self.assertAllClose(output, expect)

@pytest.mark.parametrize("rnn", ("GRU", "LSTM"))
def test_create_model(rnn):
    got = model.create_model(model.Options(attention=True,
                                           rnn=rnn)).get_config()
    got = json.loads(json.dumps(got))
    tfversion = "{}{}".format(tf.__version__.split(".")[0],
                              tf.__version__.split(".")[1])
    jsonfile = pathlib.Path(__file__)
    jsonfile = os.path.splitext(jsonfile)[0]
    jsonfile += tfversion
    jsonfile += ".json"
    with open(jsonfile, 'r') as file:
        expected = json.load(file)[rnn]
    assert got == expected
