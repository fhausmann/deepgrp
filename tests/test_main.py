"""Test deepgrp.__main__."""

import pytest

import os
import sys
import numpy as np
import tensorflow as tf

import deepgrp.__main__ as dgparser
import deepgrp.model as dgmodel
import deepgrp.training as dgtrain
import deepgrp.preprocessing as dgpreprocess
import deepgrp.prediction as dgpred
from deepgrp import sequence as dgsequence

@pytest.fixture
def cmd_class_predict(tmpdir_factory):
    dummy_dir = tmpdir_factory.mktemp("dummy_files")
    fastafile = dummy_dir.join("chr_dummy.fa")
    fp = open(fastafile, "w")
    fp.write(">chr1\n" +
             "".join(np.random.choice(["N", "A", "C", "G", "T", "t"], size=(100))))
    fp.close()
    sys.argv = ["deepgrp", "predict", "model.hdf5", str(fastafile)]
    dgparsclass = dgparser.CommandLineParser()
    return dgparsclass.parse_args()

@pytest.fixture
def cmd_class_train(tmpdir_factory):
    # create dummy parameter file
    dummy_dir = tmpdir_factory.mktemp("dummy_files")
    dummy_logdir = tmpdir_factory.mktemp("dummy_logdir")
    parameter = dummy_dir.join("parameter")
    fp = open(parameter, "w")
    fp.close()
    # create training and validfile
    trainfile = dummy_dir.join("chr1.fa.gz.npz")
    validfile = dummy_dir.join("chr1.fa.npz")
    fwd = np.zeros((5, 100))
    np.savez(trainfile, fwd=fwd)
    np.savez(validfile, fwd=fwd)
    # create dummy parser
    sys.argv = ["deepgrp", "train", str(parameter),
                str(trainfile), str(validfile),
                "bedfile", "--logdir", str(dummy_logdir),
                "--modelfile", "model.hdf5"]
    dgparsclass = dgparser.CommandLineParser()
    return dgparsclass.parse_args()

@pytest.fixture
def cmd_class_read_fasta(tmpdir_factory):
    dummy_dir = tmpdir_factory.mktemp("dummy_files")
    fastafile = dummy_dir.join("chr_dummy.fa")
    fp = open(fastafile, "w")
    fp.write(">chr1\n" +
             "".join(np.random.choice(["N", "A", "C", "G", "T", "t"], size=(100))))
    fp.close()
    return fastafile


class TestCommandLineParser:
    @pytest.mark.xfail()
    def test_init(self):
        raise NotImplementedError()

    def test_predict(self, monkeypatch, cmd_class_predict, tmpdir):
        #helper funcions
        def load_model_dummy(model, custom_objects):
            assert model == "model.hdf5"
            assert isinstance(custom_objects, dict)
            #assert isinstance(custom_objects["ReverseComplement"], dgmodel.ReverseComplement)
            # create random dummy model
            inputs = tf.keras.Input(shape=(3,))
            x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
            outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            return model

        def _predict_dummy(dnasequence, model, options, step_size, use_mss):
            assert isinstance(dnasequence, str)
            assert isinstance(model, tf.keras.Model)
            assert isinstance(options, dgmodel.Options)
            assert isinstance(step_size, int)
            assert isinstance(use_mss, bool)
            return np.random.randint(low=0, high=1000, size=(100), dtype=np.long), \
                    np.random.randint(10)

        #monkeypatching
        monkeypatch.setattr(tf.keras.models, "load_model", load_model_dummy)
        monkeypatch.setattr(dgparser, "_predict", _predict_dummy)
        # create dummy opt
        opt = dgmodel.Options(project_root_dir=str(tmpdir),
                          n_batches=1,
                          n_epochs=1,
                          batch_size=10,
                          vecsize=10)
        cmd_class_predict.predict(cmd_class_predict.args, opt)


    def test_train(self, monkeypatch, cmd_class_train):
        # helper functions
        def get_toml_dummy(file):
            return dgmodel.Options()
        def from_dict_dummy(self, dictionary):
            return dgmodel.Options()

        def preprocess_y_dummy(filename, chromosom,
                                          length,
                                          repeats_to_search):
            assert isinstance(filename, str)
            assert chromosom == "chr1"
            assert isinstance(repeats_to_search, list)
            return "np.ndarray_PREPROCESS_Y"

        def drop_start_end_n_dummy(fwd, array):
            assert isinstance(fwd, np.ndarray)
            assert array == "np.ndarray_PREPROCESS_Y"
            return np.zeros((5, 100)), np.zeros((5, 100))

        def create_model_dummy(parameter):
            assert isinstance(parameter, dgmodel.Options)
            # create random dummy model
            inputs = tf.keras.Input(shape=(3,))
            x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
            outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            return model

        def training_dummy(data, options, model, logdir):
            assert isinstance(model, tf.keras.models.Model)
            assert isinstance(data[0], dgpreprocess.Data) and \
                    isinstance(data[1], dgpreprocess.Data)
            assert isinstance(options, dgmodel.Options)

        def model_save_dummy(self, filename):
            assert isinstance(self, tf.keras.Model)
            assert filename == "model.hdf5"



        #Overwriting functions
        monkeypatch.setattr(dgmodel.Options, "from_toml", get_toml_dummy)
        monkeypatch.setattr(dgmodel.Options, "todict", lambda _func, **unused: None)
        monkeypatch.setattr(dgmodel.Options, "fromdict", from_dict_dummy)

        #overwrite preprocessing functions
        monkeypatch.setattr(dgpreprocess, "preprocess_y", preprocess_y_dummy)
        monkeypatch.setattr(dgpreprocess, "drop_start_end_n", drop_start_end_n_dummy)

        # Training
        monkeypatch.setattr(dgmodel, "create_model", create_model_dummy)
        monkeypatch.setattr(dgtrain, "training", training_dummy)
        monkeypatch.setattr(tf.keras.Model, "save", model_save_dummy)
        cmd_class_train.train(cmd_class_train.args, dgmodel.Options())




    @pytest.mark.xfail()
    def test_run(self):
        raise NotImplementedError()

    @pytest.mark.xfail()
    def test_set_logging(self):
        raise NotImplementedError()

    @pytest.mark.xfail()
    def test_setup_tensorflow(self):
        raise NotImplementedError()

    @pytest.mark.xfail()
    def test_parse_args(self):
        raise NotImplementedError()


@pytest.mark.parametrize("mss_bool", (True, False))
def test_predict(monkeypatch, tmpdir, mss_bool):
    # helper functions:
    def dgpred_predict_dummy(model, data_iterator, output_shape, step_size):
        assert isinstance(model, tf.keras.Model)
        assert isinstance(data_iterator, tf.data.Dataset)
        assert isinstance(output_shape[0], int) and isinstance(output_shape[1], int)
        assert step_size == 3
        return np.ones((100, 5))

    def apply_mss_dummy(prediction, options):
        assert isinstance(prediction, np.ndarray)
        assert isinstance(options, dgmodel.Options)
        assert mss_bool
        return np.ones((100, 5))

    def softmax_dummy(prediction):
        assert isinstance(prediction, np.ndarray)
        assert not mss_bool
        return np.ones((100, 5))

    monkeypatch.setattr(dgpred, "predict", dgpred_predict_dummy)
    monkeypatch.setattr(dgpred, "apply_mss", apply_mss_dummy)
    monkeypatch.setattr(dgpred, "softmax", softmax_dummy)

    # variables to give for testing
    opt = dgmodel.Options(project_root_dir=str(tmpdir),
                          n_batches=1,
                          n_epochs=1,
                          batch_size=10,
                          vecsize=10)
    dnasequence = "".join(np.random.choice(["N", "A", "C", "G", "T"], size=(100)))
    model = dgmodel.create_model(opt)
    dgparser._predict(dnasequence=dnasequence,
                      model = model,
                      options = opt,
                      step_size = 3,
                      use_mss = mss_bool)

def test__read_multi_fasta(cmd_class_read_fasta):
    fp = open(cmd_class_read_fasta)
    for header, sequence in dgparser._read_multi_fasta(fp):
        assert header == "chr1"
        assert isinstance(sequence, str)
    fp.close()


