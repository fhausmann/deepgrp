"""deepgrp.prediction module for prediction """

from typing import Dict, Tuple, Union
from os import PathLike
import numpy as np
import tensorflow as tf
from tensorflow import keras
# pytype: disable=import-error
# pylint: disable=import-error
from deepgrp.mss import find_mss_labels  # pylint: disable=no-name-in-module
from deepgrp.sequence import get_max
# pytype: enable=import-error
# pylint: enable=import-error
from deepgrp.model import create_model, Options
import deepgrp.preprocessing


def fetch_validation_batch(data: np.ndarray, step_size: int, batch_size: int,
                           vecsize: int) -> tf.data.Dataset:
    """Function to fetch a validation match (no randomization).

    Args:
        data (np.ndarray): Data with fwd numpy array.
        step_size (int): Window step size for the evaluation.
        batch_size (int): Batch size.
        vecsize (int): Size of one window.

    Returns:
        tf.data.Dataset: Dataset with input data adjusted for batch.

    """
    data = data.T

    def _fetch_data():
        for index in range(0, data.shape[0] - vecsize, step_size):
            yield data[index:index + vecsize].astype('float32')

    shape = tf.TensorShape([vecsize, data.shape[1]])
    dataset = tf.data.Dataset.from_generator(_fetch_data, tf.float32, shape)
    return dataset.batch(batch_size).prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)


def apply_mss(probs: np.ndarray, options: Options) -> np.ndarray:
    """Applies C-implemented Maximum segment algorithm to numpy array.

    Args:
        probs (np.ndarray): Probabilities (2dim).
        options (deepgrp.model.Options): Hyperparameter.

    Returns:
        np.ndarray: One-hot-encoded maximum segments (shape of probs).

    """
    nof_labels = probs.shape[1]
    results_classes = probs.argmax(axis=1)
    mins = probs.max(axis=1) + 1e-6
    mins[mins > 0.99] = 0.99
    t_scores = np.log(mins / (1 - mins))
    scores = np.where(results_classes > 0, t_scores,
                      -10 * t_scores).astype(float)
    return find_mss_labels(scores, results_classes, nof_labels,
                           options.min_mss_len, options.xdrop_len)


def softmax(array: np.ndarray) -> np.ndarray:
    """Compute softmax values for each sets of scores in `array`. """
    e_x = np.exp(array - np.max(array))
    return e_x / e_x.sum(axis=1, keepdims=True)


def setup_prediction_from_options_checkpoint(options: Options,
                                             logdir: PathLike) -> keras.Model:
    """Creates a DeepGRP model and loads latest weights from Checkpoint.

    Args:
        options (deepgrp.model.Options): Hyperparameter
        logdir (PathLike): Directory of the checkpoint.

    Returns:
        keras.Model: compiled functional DeepGRP model
                        with restored weights.

    """
    model = create_model(options)

    ckpt = tf.train.Checkpoint()
    manager = tf.train.CheckpointManager(ckpt, logdir, max_to_keep=None)
    model.load_weights(manager.latest_checkpoint).expect_partial()
    return model


def predict(model: keras.Model, data: tf.data.Dataset,
            results_shape: Tuple[int, int], step_size: int) -> np.ndarray:
    """Predict for a complete data using an Iterator over all data.

    Args:
        model (keras.Model): Model to use for prediction.
        data (tf.data.Dataset): Data to predict for.
        results_shape (Tuple[int,int]): shape of the output.
        step_size (int): Stepsize between two windows.

    Returns:
        np.ndarray: Predictions for complete sequence.

    """
    predictions = np.zeros(results_shape, dtype=np.float32)
    for i, batch in enumerate(data):
        vecsize = batch.shape[1]
        index = (i * batch.shape[0] * step_size)
        probas = model.predict_on_batch(batch)
        get_max(predictions[index:], probas.numpy(), step_size)
    return predictions


def predict_complete(step_size: int,
                     options: Options,
                     logdir: PathLike,
                     data: deepgrp.preprocessing.Data,
                     use_mss: bool = False) -> np.ndarray:
    """Restores a model and predict for sequence.

    Args:
        step_size (int): Window step size for the prediction.
        options (deepgrp.model.Options): Hyperparameter and other values
        logdir (PathLike): Directory of the checkpoint.
        data (deepgrp.preprocessing.Data): Data with fwd, cmp and truelbl
                                             to predict for
        use_mss (bool): Apply Maximum segment algorithm? (default=false)

    Returns:
        np.ndarray: Probabilities or one-hot encoded labels.

    """
    model = setup_prediction_from_options_checkpoint(options, logdir)
    val_iterator = fetch_validation_batch(data.fwd, step_size,
                                          options.batch_size, options.vecsize)
    ouput_shape = data.truelbl.shape[::-1]
    predictions = predict(model, val_iterator, ouput_shape, step_size)

    if use_mss:
        return apply_mss(predictions, options)
    return softmax(predictions)


def calculate_multiclass_matthews_cc(cnf_matrix: np.ndarray) -> float:
    """Calculates the R_K correlation coefficent, also called
        multi-class Matthews correlation coefficent from an confusion matrix.

    Args:
        cnf_matrix (np.ndarray): Confusion Matrix (n_classes x n_classes)

    Returns:
        float: R_K correlation coefficient
               (Multi-class Matthews correlation coefficent)

    """

    t_sum = cnf_matrix.sum(axis=1, dtype=float)
    p_sum = cnf_matrix.sum(axis=0, dtype=float)
    n_correct = np.trace(cnf_matrix, dtype=float)
    n_samples = p_sum.sum()
    cov_ytyp = n_correct * n_samples - np.dot(t_sum, p_sum)
    cov_ypyp = n_samples**2 - np.dot(p_sum, p_sum)
    cov_ytyt = n_samples**2 - np.dot(t_sum, t_sum)
    return cov_ytyp / np.sqrt(cov_ytyt * cov_ypyp)


def _calculate_metrics(
        cnf_matrix: np.ndarray) -> Dict[str, Union[np.ndarray, float]]:
    true_positive = np.diag(cnf_matrix).astype(float)
    false_positive = (cnf_matrix.sum(axis=0) - true_positive).astype(float)
    false_negative = (cnf_matrix.sum(axis=1) - true_positive).astype(float)
    true_negative = (
        cnf_matrix.sum() -
        (false_positive + false_negative + true_positive)).astype(float)
    metrics = {}
    # Sensitivity, hit rate, recall, or true positive rate
    metrics["TPR"] = true_positive / (true_positive + false_negative)
    # Specificity or true negative rate
    metrics["TNR"] = true_negative / (true_negative + false_positive)
    # Precision or positive predictive value
    metrics["PPV"] = true_positive / (true_positive + false_positive)
    # Negative predictive value
    metrics["NPV"] = true_negative / (true_negative + false_negative)
    # Fall out or false positive rate
    metrics["FPR"] = false_positive / (false_positive + true_negative)
    # False negative rate
    metrics["FNR"] = false_negative / (true_positive + false_negative)
    # False discovery rate
    metrics["FDR"] = false_positive / (true_positive + false_positive)
    # Accuracy
    metrics["ACC"] = (true_positive + true_negative) / \
        (true_positive + false_positive + false_negative + true_negative)
    # F1 -Score
    metrics["F1"] = 2 * metrics["TPR"] * \
        metrics["PPV"] / (metrics["TPR"] + metrics["PPV"])
    metrics['MCC'] = calculate_multiclass_matthews_cc(cnf_matrix)
    return metrics


def confusion_matrix(truelbl: np.ndarray,
                     predictedlbl: np.ndarray) -> np.ndarray:
    """Calculate confusion matrix from integer arrays.

    Args:
        truelbl (np.ndarray): Array with correct labels (integer).
        predictedlbl (np.ndarray): Array with predicted labels (integer).

    Returns:
        np.ndarray: Confusion Matrix

    """
    assert truelbl.size == predictedlbl.size
    n_classes = max(truelbl.max(), predictedlbl.max()) - min(
        truelbl.min(), predictedlbl.min()) + 1
    cnf = np.zeros((n_classes, n_classes), dtype=int)
    for i, j in zip(truelbl, predictedlbl):
        cnf[i, j] += 1
    return cnf


def calculate_metrics(
    predictions_class: np.ndarray, true_class: np.ndarray
) -> Tuple[np.ndarray, Dict[str, Union[np.ndarray, float]]]:
    """Calculated important metrics.

    Args:
        predictions_class (np.ndarray): Predicted labels.
        true_class (np.ndarray): True labels.

    Returns:
        Tuple[np.ndarray, Dict[str, Union[np.ndarray,float]]]:
            Confusion matrix as np.ndarray and dictionary with other metrics.

    """
    overall_acc = (true_class == predictions_class).sum() / true_class.shape[0]
    cnf_matrix = confusion_matrix(true_class, predictions_class)
    metrics = _calculate_metrics(cnf_matrix)
    metrics["TotalACC"] = overall_acc
    return cnf_matrix, metrics


def filter_segments(array: np.ndarray, min_len: int = 50) -> None:
    """Filter segments greater or equal to minimum length (inplace).

    Args:
        array (np.ndarray): Labels.
        min_len (int): Minumum length for one segment (default 50)
    """
    indices = np.where(array > 0)[0]
    next_idx = 0
    for idx in indices:
        if next_idx > idx:
            continue
        next_idx = idx + 1
        found = 1
        while next_idx < array.size and array[next_idx] == array[idx]:
            found += 1
            next_idx += 1
        if found < min_len:
            array[idx:next_idx] = 0
