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
                "bedfile", "--logdir", str(dummy_logdir)]
    dgparsclass = dgparser.CommandLineParser()
    return dgparsclass.parse_args()

class TestCommandLineParser:
    @pytest.mark.xfail()
    def test_init(self):
        raise NotImplementedError()

    @pytest.mark.xfail()
    def test_predict(self):
        raise NotImplementedError()

    def test_train(self, monkeypatch, tmpdir, cmd_class_train):
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

        def training_dummy(data, options, model, logdir):
            #TODO
            assert 0
            assert(data[0] == "np.ndarray_PREPROCESS_Y" and data[1] == "np.ndarray_PREPROCESS_Y")
            assert()

        def create_model_dummy(parameter):
            assert isinstance(parameter, dgmodel.Options)
            assert 0
            assert(parameter == "FROM_TOML_OBJECT")

        #Overwrite parameters
        monkeypatch.setattr(dgmodel.Options, "from_toml", get_toml_dummy)
        monkeypatch.setattr(dgmodel.Options, "todict", lambda _func, **unused: None)
        monkeypatch.setattr(dgmodel.Options, "fromdict", from_dict_dummy)

        #TODO overwrite preprocessing functions
        monkeypatch.setattr(dgpreprocess, "preprocess_y", preprocess_y_dummy)

        # Training
        monkeypatch.setattr(dgmodel, "create_model", create_model_dummy)
        monkeypatch.setattr(dgtrain, "training", training_dummy)
        monkeypatch.setattr(tf.keras.Model, "save", lambda _func, **unused: None)
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


@pytest.mark.xfail()
def test_predict():
    raise NotImplementedError()


@pytest.mark.xfail()
def test__read_multi_fasta():
    raise NotImplementedError()
