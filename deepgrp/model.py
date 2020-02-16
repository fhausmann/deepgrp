"""deepgrp.model modul for the creation of a model"""

from typing import Dict, List, Union, Any, TextIO
import os
from datetime import datetime
import toml
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as klayers


def create_logdir(options) -> str:
    """Standarized formating of logdirs.

    Args:
        options (Options): information about the projects root dir.

    Returns:
        str: standarized logdir path.

    """
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = os.path.join(options.project_root_dir, "tf_logs")
    logdir = os.path.join(root_logdir, "run-{}".format(now))
    return logdir


class Options:
    """Options holds hyperparameter and important
    information for training a DeepGRP model.

    Attributes:
        project_root_dir (str): Project root dir.
        repeats_to_search (List[int]): List with repeats to search for.
        vecsize (int): Window size of the model.
        n_epochs (int): Maximal number of epochs.
        n_batches (int): Number of batches per epoch.
        early_stopping_th (int): Number of steps for EarlyStopping.
        batch_size (int): Batch size.
        repeat_probability (float): Probability of repeats per batch.
        optimizer (str): Optimizer to use.
        learning_rate (float): Learning rate.
        momentum (float): Momentum (RMSprop) or beta_1 (Adam).
        rho (float): Rho (RMSprop) or beta_2 (Adam).
        epsilon (float): Small number for optimization.
        rnn (str): RNN type (GRU or LSTM).
        units (int): Number of hidden units.
        dropout (float): Dropout for the RNN.
        attention (bool): Use AdditiveAttention layer? (only with GRU).
        min_mss_len (int): Minimal lenght of a maximum scoring segment.
        xdrop_len (int): Length when MSS extension with decreasing
                         score is stopped.
    """

    # pylint: disable=too-many-instance-attributes
    attention: bool
    batch_size: int
    dropout: float
    early_stopping_th: int
    epsilon: float
    learning_rate: float
    min_mss_len: int
    momentum: float
    n_batches: int
    n_epochs: int
    optimizer: str
    project_root_dir: str
    repeat_probability: float
    repeats_to_search: List[int]
    rho: float
    rnn: str
    units: int
    vecsize: int
    xdrop_len: int

    def __init__(self, **kwargs) -> None:
        """Initialize Options with default values.

        Args:
            **kwargs: Additional/ changed attributes.
        """

        # General Parameter
        # Project root dir.
        self.project_root_dir = "."
        # List with repeats to search for.
        self.repeats_to_search = [1, 2, 3, 4]
        # Window size of the model.
        self.vecsize = 150
        # Maximal number of epochs.
        self.n_epochs = 200
        # Number of batches per epoch.
        self.n_batches = 250
        # Number of steps for EarlyStopping.
        self.early_stopping_th = 10
        # Batch size.
        self.batch_size = 256
        # Probability of repeats per batch.
        self.repeat_probability = 0.3

        # Optimzer Parameter
        # Optimizer to use.
        self.optimizer = "RMSprop"
        # Learning rate.
        self.learning_rate = 0.001
        # Momentum (RMSprop) or beta_1 (Adam).
        self.momentum = 0.9
        # Rho (RMSprop) or beta_2 (Adam).
        self.rho = 0.9
        # Small number for optimization.
        self.epsilon = 1e-10

        # Neural Network Parameter
        # RNN type (GRU or LSTM).
        self.rnn = "GRU"
        # Number of hidden units.
        self.units = 32
        # Dropout for the RNN.
        self.dropout = 0.25
        # Use AdditiveAttention layer? (only with GRU).
        self.attention = False

        # MSS Parameter
        # Minimal lenght of a maximum scoring segment.
        self.min_mss_len = 50
        # Length when MSS extension with decreasing score is stopped.
        self.xdrop_len = 50

        self.__dict__.update(kwargs)

        units = self.__dict__.pop("gru_units", None)
        dropout = self.__dict__.pop("gru_dropout", None)
        if units:
            self.units = units
        if dropout:
            self.dropout = dropout

    def __setitem__(self, key: str, item: Union[float, int, str]) -> None:
        key = key.replace("gru_", "")  # For compatibility issues
        self.__dict__[key] = item

    def __getitem__(self, key: str) -> Union[float, int, str]:
        key = key.replace("gru_", "")  # For compatibility issues
        return self.__dict__[key]

    def __str__(self) -> str:
        return str(self.__dict__)

    def todict(self) -> Dict[str, Union[float, int, str]]:
        """Creates a dict from Options.

        Returns:
            Dict[str, Union[float, int, str]]: Dict with all parameters and
                                               values in options
        """
        return self.__dict__.copy()

    def fromdict(self, dictionary: Dict[str, Union[float, int, str]]) -> None:
        """Updates an options object from dict.

        Args:
            dictionary (Dict[str, Union[float, int, str]]):
                Parameter and values to store in Options.
        """
        self.__dict__.update(dictionary)
        units = self.__dict__.pop("gru_units", None)
        dropout = self.__dict__.pop("gru_dropout", None)
        if units:
            self.units = units
        if dropout:
            self.dropout = dropout

    @classmethod
    def from_toml(cls, file: TextIO) -> 'Options':
        """Creates class from toml file

        Args:
            file (TextIO): File descriptor in toml format.

        Raises:
            TypeError: When anything other than file descriptor is passed

        Returns:
            [Options]: Options containing data from toml file
        """

        inputs = toml.load(file)
        return cls(**inputs)

    def to_toml(self, file: TextIO):
        """Writes options in class to file in toml format.

        Args:
            file (TextIO): Writable file descriptor.

        Raises:
            TypeError: When anything other than file descriptor is passed
        """
        toml.dump(self.__dict__, file)


def _get_optimizer(options: Options) -> tf.optimizers.Optimizer:
    if options.optimizer == "RMSprop":
        optimizer = tf.optimizers.RMSprop(learning_rate=options.learning_rate,
                                          momentum=options.momentum,
                                          rho=options.rho,
                                          epsilon=options.epsilon)
    elif options.optimizer == "Adam":
        optimizer = tf.optimizers.Adam(learning_rate=options.learning_rate,
                                       beta_1=options.momentum,
                                       beta_2=options.rho,
                                       epsilon=options.epsilon)
    else:
        optimizer = options.optimizer
    return optimizer


def _get_brnn_layer(options: Options) -> keras.layers.Layer:
    if options.rnn == 'LSTM':
        brnn = keras.layers.LSTM(units=options.units,
                                 dropout=options.dropout,
                                 name="BLSTM",
                                 return_sequences=True)
    else:
        brnn = keras.layers.GRU(units=options.units,
                                dropout=options.dropout,
                                name="BGRU",
                                return_state=options.attention,
                                return_sequences=True)
    return brnn


def _get_dna_encoding() -> List[int]:
    encoding = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    pairs = (('A', 'T'), ('T', 'A'), ('C', 'G'), ('G', 'C'), ('N', 'N'))
    complement = {encoding[k]: encoding[v] for k, v in pairs}
    return [v for _, v in sorted(complement.items())]


class ReverseComplement(tf.keras.layers.Layer):
    """Calculates reverse complement of the one hot encoded input based on
        dictionary containing complementary indices.

    Args:
        complements (List[int]): Complementary indices.
                                 Example: Value 1 at index 0 means
                                 row 0 gets row 1.

    Attributes:
        _indices (List[int]):Indices of the complement.
        _axis (type): Axis to be reversed, default: 1.

    """

    _indices: tf.Tensor
    _axis: tf.Tensor
    _expects_mask_arg: bool
    _expects_training_arg: bool

    def __init__(self, complements: List[int], **kwargs):
        super(ReverseComplement, self).__init__(**kwargs)
        self._indices = complements
        self._axis = [1]

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Calls the layer and returning the reverse complement.

        Args:
            inputs (tf.Tensor): Input one hot encoded sequence.
                                Shape: [None,n,len(`complements`)]

        Returns:
            tf.Tensor: Reverse complement of the input.

        """
        return tf.gather(tf.reverse(inputs, axis=self._axis),
                         self._indices,
                         axis=2)

    def get_config(self) -> Dict[str, Any]:
        """get_config for serialisation as JSON.

        Returns:
            Dict[str, Any]: Configuration of layer.

        """
        config = {'complements': self._indices}
        base_config = super(ReverseComplement, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def create_model(options: Options) -> keras.Model:
    """Creates a functional model and compiles it.

    Args:
        options (Options): Hyperparameter for building a DeepGRP model.

    Returns:
        keras.Model: Compiled functional keras model.

    """

    inputs_fwd = klayers.Input((options.vecsize, 5))
    inputs_rev = ReverseComplement(_get_dna_encoding())(inputs_fwd)
    brnn = _get_brnn_layer(options)

    if options.attention and isinstance(brnn, klayers.GRU):
        fwd, hidden_fwd = brnn(inputs_fwd)
        rev, hidden_rev = brnn(inputs_rev)
        hidden = klayers.Average()([hidden_fwd, hidden_rev])
        avg = klayers.Average()([fwd, rev])
        hidden = klayers.Reshape((1, options.units),
                                 input_shape=(options.units, ))(hidden)
        attention_result = klayers.AdditiveAttention()([hidden, avg])
        attention_result = klayers.Flatten()(attention_result)
        attention_result = klayers.RepeatVector(
            options.vecsize)(attention_result)
        avg = klayers.Concatenate()([attention_result, avg])
    else:
        fwd = brnn(inputs_fwd)
        rev = brnn(inputs_rev)
        avg = klayers.Average()([fwd, rev])

    logits = klayers.Dense(len(options.repeats_to_search) + 1,
                           name="FF",
                           activation=None)(avg)

    output = klayers.Softmax(axis=2)(logits)

    model = keras.Model(inputs=inputs_fwd, outputs=output)

    model.compile(optimizer=_get_optimizer(options),
                  loss=tf.losses.CategoricalCrossentropy())

    return model
