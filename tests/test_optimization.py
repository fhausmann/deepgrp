"""Test deepgrp.optimization."""

import pickle

import hyperopt
import numpy as np
import pytest
import tensorflow as tf

import deepgrp.model as dgmodel
import deepgrp.optimization as dgopt
import deepgrp.preprocessing as dgprep
import deepgrp.prediction as dgpred

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


@pytest.mark.parametrize("is_failed", ("no", "loss", "exception"))
def test_build_and_optimize(tmpdir, monkeypatch, is_failed):

    opt = dgmodel.Options(project_root_dir=str(tmpdir),
                          n_batches=1,
                          n_epochs=1,
                          batch_size=10,
                          vecsize=10)
    train = dgprep.Data(fwd=np.zeros((5, 100)), truelbl=np.zeros((5, 100)))
    valid = dgprep.Data(fwd=np.zeros((5, 100)), truelbl=np.zeros((5, 100)))
    optdict = {"test": 10}
    mcc = np.nan if is_failed == "loss" else 1

    monkeypatch.setattr(dgpred, "calculate_metrics", lambda *_, **unused:
                        (None, {
                            "MCC": mcc
                        }))
    monkeypatch.setattr(dgpred, "predict_complete",
                        lambda *_, **unused: np.ones((100, 5)))
    monkeypatch.setattr(tf.keras.Model, "fit", lambda *_, **unused: None)
    if is_failed == "exception":

        def _func(*_, **unused):
            raise ValueError
    else:
        _func = lambda *_, **unused: None
    monkeypatch.setattr(dgpred, "filter_segments", _func)
    got = dgopt.build_and_optimize(train_data=train,
                                   val_data=valid,
                                   step_size=10,
                                   options=opt,
                                   options_dict=optdict)

    print(got["Metrics"])
    expected = hyperopt.STATUS_OK if is_failed == "no" else hyperopt.STATUS_FAIL
    assert got["status"] == expected
    expected = -1 if is_failed == "no" else np.inf
    assert got["loss"] == expected


@pytest.mark.parametrize("max_evals", (5, 10))
@pytest.mark.parametrize("file_is_present", (True, False))
def test_run_a_trial(tmp_path, file_is_present, max_evals):
    trials = hyperopt.Trials()
    filepath = tmp_path.joinpath("results.pkl")
    if file_is_present:
        filepath.write_bytes(pickle.dumps(trials))
    got = dgopt.run_a_trial(space={"test": hyperopt.hp.uniform("test", 1, 10)},
                            objective=lambda x: {
                                "loss": 5,
                                "status": hyperopt.STATUS_OK
                            },
                            project_root_dir=str(tmp_path.resolve()),
                            max_evals=max_evals)

    assert got == max_evals
