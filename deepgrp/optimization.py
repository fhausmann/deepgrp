"""deepgrp.optimization modul for optimisation with hyperopt"""

import logging
import pickle
import shutil
from os import PathLike, path
from typing import Any, Callable, Dict, Union

import numpy as np
import tensorflow as tf
from hyperopt import STATUS_FAIL, STATUS_OK, Trials, fmin, tpe
from tensorboard.plugins.hparams import api as hp
from tensorflow import summary as tfsum
from tensorflow.keras import backend as K  # pylint: disable=import-error

from deepgrp.model import Options, create_logdir, create_model
from deepgrp.prediction import (calculate_metrics, filter_segments,
                                predict_complete)
from deepgrp.preprocessing import Data
from deepgrp.training import training

_LOGGER = logging.getLogger(__name__)


def _update_options(options: Options, dictionary: Dict[str, Any]) -> Options:
    for key, value in dictionary.items():
        options[key] = value
    options.vecsize = int(options.vecsize)
    options.units = int(options.units)
    return options


def build_and_optimize(
        train_data: Data, val_data: Data, step_size: int, options: Options,
        options_dict: Dict[str, Union[str, float]]) -> Dict[str, Any]:
    """Builds an DeepGRP model with updated options,
    trains it and validates it. Used for hyperopt optimization.

    Args:
        train_data (deepgrp.preprocessing.Data): Training data.
        val_data (deepgrp.preprocessing.Data): Validation data.
        step_size (int): Window size for the final evaluation.
        options (Options): General hyperparameter.
        options_dict (Dict[str, Union[str, float]]): Varying hyperparameter.

    Returns:
        Dict[str, Any]: Dictionary with results (Hyperopt compatible).

    """

    options = _update_options(options, options_dict)
    logdir = create_logdir(options)

    def _train_test(model):  # pragma: no cover
        extra_callback = [hp.KerasCallback(logdir, options_dict)]
        training((train_data, val_data), options, model, logdir,
                 extra_callback)
        K.clear_session()
        predictions = predict_complete(step_size,
                                       options,
                                       logdir,
                                       val_data,
                                       use_mss=True)
        K.clear_session()
        is_not_na = np.logical_not(np.isnan(predictions[:, 0]))
        predictions_class = predictions[is_not_na].argmax(axis=1)
        filter_segments(predictions_class, options.min_mss_len)
        _, metrics = calculate_metrics(
            predictions_class, val_data.truelbl[:, is_not_na].argmax(axis=0))
        return metrics

    results = {
        'loss': np.inf,
        'Metrics': None,
        'options': options.todict(),
        'logdir': None,
        'status': STATUS_FAIL,
        'error': "",
    }
    try:
        K.clear_session()
        model = create_model(options)
        file_writer = tf.summary.create_file_writer(logdir)
        file_writer.set_as_default()
        metrics = _train_test(model)
        tfsum.scalar('MCC',
                     metrics['MCC'],
                     step=0,
                     description="Matthews correlation coefficient")
    except Exception as err:  # pylint: disable=broad-except
        _LOGGER.exception("Error occurred while training")
        results["error"] = str(err)
        results["status"] = STATUS_FAIL
        if results["logdir"]:
            shutil.rmtree(results["logdir"])
    else:
        results["logdir"] = logdir
        results["loss"] = -1 * metrics['MCC']

        results["status"] = STATUS_OK
        results["Metrics"] = metrics
        if np.isnan(results["loss"]):
            results["status"] = STATUS_FAIL
            results["loss"] = np.inf
    finally:
        file_writer.close()

    return results


def run_a_trial(space: Dict[str, Any], objective: Callable[[Dict[str, Any]],
                                                           Dict[str, Any]],
                project_root_dir: PathLike, max_evals: int) -> int:
    """Runs a hyperopt TPE meta optimisation step. Restores previous run,
     if `results.pkl` is available.

    Args:
        space (Dict[str, Any]): Hyperopt search space.
        objective (Callable[[Dict[str, Any]], Dict[str, Any]]):
            Function to minimize (usually partial function of
            `build_and_optimize`). Takes only one argument (dict) with
            new hyperparameters.
        project_root_dir (PathLike): Project root dir.
        max_evals (int): Maximum number of trials.

    Returns:
        int: Number of trials in `results.pkl`

    """
    nb_evals = max_evals

    _LOGGER.info("Attempt to resume a past training if it exists:")
    results_path = path.join(project_root_dir, "results.pkl")

    try:
        with open(results_path, "rb") as file:
            trials = pickle.load(file)
    except FileNotFoundError:
        trials = Trials()
        _LOGGER.info("Starting from scratch: new trials.")
    else:
        _LOGGER.warning("Found saved Trials! Loading...")
        max_evals = len(trials.trials) + nb_evals
        _LOGGER.info("Rerunning from %d trials to add another one.",
                     len(trials.trials))

    fmin(objective,
         space,
         algo=tpe.suggest,
         trials=trials,
         max_evals=max_evals)

    with open(results_path, "wb") as file:
        pickle.dump(trials, file)

    return len(trials.losses())
