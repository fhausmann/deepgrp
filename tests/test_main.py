"""Test deepgrp.__main__."""

import argparse
import pathlib
import sys

import numpy as np
import pytest
import tensorflow as tf
import toml

import deepgrp.__main__ as dgparser
import deepgrp.model as dgmodel
import deepgrp.prediction as dgpred
import deepgrp.preprocessing as dgpreprocess
import deepgrp.training as dgtrain
# pylint: disable=redefined-outer-name


@pytest.fixture
def dummyfasta(tmp_path):
    fastafile = tmp_path.joinpath("chr_dummy.fa")

    def create_fasta(n=1):
        sequences = {
            f"chr{i+1}":
            "".join(np.random.choice(["N", "A", "C", "G", "T"], size=(100)))
            for i in range(n)
        }
        fastafile.write_text("\n".join(
            (f">{header}\n{sequence}"
             for header, sequence in sequences.items())))
        return fastafile, sequences

    return create_fasta


class TestCommandLineParser:
    def test_init(self):
        parser = dgparser.CommandLineParser()
        assert isinstance(parser.parser, argparse.ArgumentParser)
        assert parser.threads == 1
        assert not parser.xla
        assert parser.verbose == 0
        assert parser.args is None

    def test_predict(self, monkeypatch, dummyfasta):

        fastafile, expected_sequence = dummyfasta(1)
        sys.argv = ["deepgrp", "predict", "model.hdf5", str(fastafile)]
        dgparsclass = dgparser.CommandLineParser().parse_args()

        #helper funcions
        def load_model_dummy(model, custom_objects):
            assert model == "model.hdf5"
            assert isinstance(custom_objects, dict)
            #assert isinstance(custom_objects["ReverseComplement"], dgmodel.ReverseComplement)
            # create random dummy model
            inputs = tf.keras.Input(shape=(3, ))
            x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
            outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            return model

        def _predict_dummy(dnasequence, model, options, step_size, use_mss):
            assert dnasequence == expected_sequence["chr1"]
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
        opt = dgmodel.Options(project_root_dir=str(fastafile.parent),
                              n_batches=1,
                              n_epochs=1,
                              batch_size=10,
                              vecsize=10)
        dgparsclass.predict(dgparsclass.args, opt)

    def test_train(self, monkeypatch, tmp_path):
        # create dummy parameter file
        dummy_dir = tmp_path.joinpath("dummy_files")
        dummy_logdir = tmp_path.joinpath("dummy_logdir")
        dummy_dir.mkdir()
        dummy_logdir.mkdir()
        parameter = dummy_dir.joinpath("parameter.toml")
        with parameter.open("w") as file:
            toml.dump({"dummy": "parameter"}, file)
        # create training and validfile
        trainfile = dummy_dir.joinpath("chr1.fa.gz.npz")
        validfile = dummy_dir.joinpath("chr1.fa.npz")
        fwd = np.zeros((5, 100))
        np.savez(trainfile, fwd=fwd)
        np.savez(validfile, fwd=fwd)
        # create dummy parser
        sys.argv = [
            "deepgrp", "train",
            str(parameter),
            str(trainfile),
            str(validfile), "bedfile", "--logdir",
            str(dummy_logdir), "--modelfile", "model.hdf5"
        ]
        dgparsclass = dgparser.CommandLineParser().parse_args()

        # helper functions
        def get_toml_dummy(file):
            assert pathlib.Path(file.name) == parameter
            return dgmodel.Options()

        def from_dict_dummy(_, dictionary):
            assert isinstance(dictionary, dict)
            return dgmodel.Options()

        def preprocess_y_dummy(filename, chromosom, length, repeats_to_search):
            assert isinstance(filename, str)
            assert filename == "bedfile"
            assert chromosom == "chr1"
            assert length == 100
            assert isinstance(repeats_to_search, list)
            return "np.ndarray_PREPROCESS_Y"

        def drop_start_end_n_dummy(fwd, array):
            assert isinstance(fwd, np.ndarray)
            assert array == "np.ndarray_PREPROCESS_Y"
            return np.zeros((5, 100)), np.zeros((5, 100))

        def create_model_dummy(parameter):
            assert isinstance(parameter, dgmodel.Options)
            # create random dummy model
            inputs = tf.keras.Input(shape=(3, ))
            x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
            outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            return model

        def training_dummy(data, options, model, logdir):
            assert pathlib.Path(logdir) == dummy_logdir
            assert isinstance(model, tf.keras.models.Model)
            assert isinstance(data[0], dgpreprocess.Data) and \
                    isinstance(data[1], dgpreprocess.Data)
            assert isinstance(options, dgmodel.Options)

        def model_save_dummy(self, filename):
            assert isinstance(self, tf.keras.Model)
            assert filename == "model.hdf5"

        #Overwriting functions
        monkeypatch.setattr(dgmodel.Options, "from_toml", get_toml_dummy)
        monkeypatch.setattr(dgmodel.Options, "fromdict", from_dict_dummy)

        #overwrite preprocessing functions
        monkeypatch.setattr(dgpreprocess, "preprocess_y", preprocess_y_dummy)
        monkeypatch.setattr(dgpreprocess, "drop_start_end_n",
                            drop_start_end_n_dummy)

        # Training
        monkeypatch.setattr(dgmodel, "create_model", create_model_dummy)
        monkeypatch.setattr(dgtrain, "training", training_dummy)
        monkeypatch.setattr(tf.keras.Model, "save", model_save_dummy)
        dgparsclass.train(dgparsclass.args, dgmodel.Options())

    @pytest.mark.parametrize("min_mss_length", [10, 15])
    @pytest.mark.parametrize("batch_size", [100, 150])
    @pytest.mark.parametrize("xdrop_len", [23, 15])
    @pytest.mark.parametrize("command", ["train", "predict"])
    def test_run(self, command, xdrop_len, batch_size, min_mss_length,
                 monkeypatch):
        parser = dgparser.CommandLineParser()
        parser.args = argparse.Namespace(command=command,
                                         xdrop_length=xdrop_len,
                                         batch_size=batch_size,
                                         min_mss_length=min_mss_length)

        def _check_function_call(called_command, got_args, got_opions):
            assert id(got_args) == id(parser.args)
            assert got_opions.min_mss_len == min_mss_length
            assert got_opions.batch_size == batch_size
            assert got_opions.xdrop_len == xdrop_len
            assert called_command == command

        monkeypatch.setattr(
            parser, "train", lambda *args, **kwargs: _check_function_call(
                "train", *args, **kwargs))
        monkeypatch.setattr(
            parser, "predict", lambda *args, **kwargs: _check_function_call(
                "predict", *args, **kwargs))
        parser.run()


@pytest.mark.parametrize("mss_bool", (True, False))
def test_predict(monkeypatch, tmpdir, mss_bool):
    # helper functions:
    def dgpred_predict_dummy(model, data_iterator, output_shape, step_size):
        assert isinstance(model, tf.keras.Model)
        assert isinstance(data_iterator, tf.data.Dataset)
        assert isinstance(output_shape[0], int) and isinstance(
            output_shape[1], int)
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
    dnasequence = "".join(
        np.random.choice(["N", "A", "C", "G", "T"], size=(100)))
    model = dgmodel.create_model(opt)
    dgparser._predict(  # pylint: disable=protected-access
        dnasequence=dnasequence,
        model=model,
        options=opt,
        step_size=3,
        use_mss=mss_bool)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_read_multi_fasta(dummyfasta, n):
    filename, expected_sequence = dummyfasta(n)

    with filename.open() as file:
        res = {
            header: sequence
            for header, sequence in dgparser._read_multi_fasta(file)  # pylint: disable=protected-access
        }
    assert res == expected_sequence
