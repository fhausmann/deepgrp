"""deepgrp.training modul for training"""

from typing import Callable, Iterator, Tuple, List, Optional
from os import path, PathLike
import numpy as np
import tensorflow as tf
from tensorflow import keras
# pylint: disable=import-error
import tensorflow.keras.callbacks as kc
# pylint: enable=import-error
from deepgrp import preprocessing
from deepgrp.model import Options


def training(data: Tuple[preprocessing.Data, preprocessing.Data],
             options: Options,
             model: keras.Model,
             logdir: PathLike,
             extra_callbacks: Optional[List[kc.Callback]] = None) -> None:
    """Runs training.

    Args:
        data (Tuple[preprocessing.Data, preprocessing.Data]):
            Training and the validation data.
        options (deepgrp.model.Options): Hyperparameter.
        model (keras.Model): Model to train.
        logdir (os.PathLike): Log / Checkpoint directory.
        extra_callbacks (Optional[List[kc.Callback]]): Additional callbacks.
    """
    n_repeats = len(options.repeats_to_search) + 1
    shapes = (tf.TensorShape([None, options.vecsize, data[0].fwd.shape[0]]),
              tf.TensorShape([None, options.vecsize, n_repeats]))
    dataset = tf.data.Dataset.from_generator(fetch_batch(options, data[0]),
                                             (tf.float32, tf.float32), shapes)
    dataset_val = tf.data.Dataset.from_generator(fetch_batch(options, data[1]),
                                                 (tf.float32, tf.float32),
                                                 shapes)

    callbacks = [
        kc.TensorBoard(log_dir=logdir,
                       histogram_freq=3,
                       write_graph=True,
                       write_images=True,
                       profile_batch=1,
                       update_freq='batch'),
        kc.EarlyStopping(monitor='val_loss',
                         min_delta=0,
                         patience=options.early_stopping_th,
                         verbose=0,
                         mode='auto',
                         baseline=None,
                         restore_best_weights=True),
        kc.ModelCheckpoint(path.join(logdir, '{epoch:002d}'),
                           monitor='val_loss',
                           verbose=0,
                           mode='min',
                           save_freq='epoch',
                           save_best_only=True,
                           save_weights_only=True)
    ]

    if extra_callbacks:
        callbacks += extra_callbacks

    model.fit(dataset,
              verbose=0,
              epochs=options.n_epochs,
              steps_per_epoch=options.n_batches,
              validation_freq=1,
              shuffle=False,
              validation_data=dataset_val,
              validation_steps=1,
              callbacks=callbacks)


def _calc_indices(array: np.ndarray, vecsize: int) -> np.ndarray:
    sums = array.cumsum()
    sums[vecsize:] = sums[vecsize:] - sums[:-vecsize]
    indices = np.where(sums > 0)[0] - vecsize
    indices = indices[indices > 0]
    return indices


def fetch_batch(
        options: Options, data: preprocessing.Data
) -> Callable[[], Iterator[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]]]:
    """Creates a function used for the creation of tensorflow Dataset.

    Args:
        options (Options): Hyperparameter.
        data (preprocessing.Data): Data for batch creation.

    Returns:
        Callable[[], Iterator[Tuple[Tuple[np.ndarray, np.ndarray],
                                    np.ndarray]]]:
            Function, which can used to create tf.Dataset
            with Dataset.from_generator.

    """
    one_class_size = int(options.batch_size * options.repeat_probability /
                         (data.truelbl.shape[0] - 1))

    batch_indices_total = (_calc_indices(data.truelbl[i], options.vecsize)
                           for i in range(1, data.truelbl.shape[0]))
    batch_indices = [
        idx for idx in batch_indices_total if idx.size > one_class_size
    ]
    filled = one_class_size * len(batch_indices)

    def get_batch():
        while True:
            fwd_batch = np.empty(
                (options.batch_size, options.vecsize, data.fwd.shape[0]),
                dtype='float32')
            y_batch = np.zeros(
                (options.batch_size, options.vecsize, data.truelbl.shape[0]),
                dtype='float32')
            indices = np.empty(options.batch_size, dtype=int)
            for i, bindex in enumerate(batch_indices):
                indices[one_class_size * i:one_class_size *
                        (i + 1)] = np.random.choice(bindex, one_class_size)
            indices[filled:] = np.random.randint(
                0,
                data.truelbl.shape[1] - options.vecsize,
                size=options.batch_size - filled)
            np.random.shuffle(indices)
            for i, index in enumerate(indices):
                fwd_batch[i] = data.fwd[:, index:index + options.vecsize].T
                y_batch[i] = data.truelbl[:, index:index + options.vecsize].T
            yield fwd_batch, y_batch

    return get_batch
