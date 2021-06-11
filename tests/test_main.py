"""Test deepgrp.__main__."""

import pytest


class TestCommandLineParser:
    @pytest.mark.xfail()
    def test_init(self):
        raise NotImplementedError()

    @pytest.mark.xfail()
    def test_predict(self):
        raise NotImplementedError()

    def get_toml_dummy():
        # get dgmodel.Options that is initialised
        # maybe not necessary, depending on how preprocessing and training
        # are tested
        raise NotImplementedError()

    def rep_dg_train_training():
        # load model instead?
        return None

    @pytest.mark.xfail()
    def test_train(self):
        #Overwrite parameters
        monkeypatch.setattr(dgmodel.Options, "from_toml", get_toml_dummy())
        monkeypatch.setattr(parameter, "fromdict", lambda _func, **unused: None)

        #logdir
        monkeypatch.setattr(os, "mkdir", lambda _func, **unused: None)

        # preprocessing


        # training - skip training and load a model instead due to intense runtime
        monkeypatch.setattr(dgtrain, "training", rep_dg_train_training)
        monkeypatch.setattr(model, "save", lambda _func, **unused: None)



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
