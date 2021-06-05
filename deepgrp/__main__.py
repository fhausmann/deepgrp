# !/usr/bin/env python3
"""DeepGRP main for prediction with pretrained model """
from typing import Iterator, Tuple, TextIO, List
import sys
import os
import argparse
import logging
import numpy as np
import tensorflow as tf
import deepgrp.prediction as dgpred
import deepgrp.model as dgmodel
import deepgrp.training as dgtrain
import deepgrp.preprocessing as dgpreprocess
from deepgrp import sequence as dgsequence

logging.basicConfig()
_LOG = logging.getLogger(__name__)


def _read_multi_fasta(filestream: TextIO) -> Iterator[Tuple[str, str]]:
    """Reads multi FASTA file.

    Args:
        filestream (TextIO): filestream from (multi-) FASTA file.

    Returns:
        Iterator[Tuple[str, str]]: Header and sequence

    """
    _LOG.debug("Reading FASTA file.")
    header = ""
    sequence: List[str] = []
    for line in filestream:
        line = line.strip()
        if line[0] == ">":
            if header:
                yield header, "".join(sequence)
            header = line[1:]
            sequence = []
        else:
            sequence.append(line.upper())
    if header:
        yield header, "".join(sequence)


def _predict(
    dnasequence: str,
    model: tf.keras.Model,
    options: dgmodel.Options,
    step_size: int,
    use_mss: bool,
) -> Tuple[np.ndarray, int]:
    """Runs a prediction for one sequence .

    Args:
        dnasequence (str): DNA sequence to predict for.
        model (tf.keras.Model): Model to use for prediction
        options (dgmodel.Options): Hyperparameter used for model
                                    creation and prediction
        step_size (int): Window step size.
        use_mss (bool): Use MSS algorithm.

    Returns:
        np.ndarray: predictions from TensorFlow and potential the MSS
        int: Start position of the prediction (== number of leading N's)

    """

    _LOG.debug("One hot encoding sequence.")
    start_pos, inputs = dgsequence.one_hot_encode_dna_sequence(dnasequence)
    _LOG.debug("Start prediction.")
    data_iterator = dgpred.fetch_validation_batch(inputs, step_size,
                                                  options.batch_size,
                                                  options.vecsize)
    output_shape = (inputs.shape[1], model.output_shape[2])
    prediction = dgpred.predict(model, data_iterator, output_shape, step_size)
    _LOG.debug("Finish prediction.")
    if use_mss:
        _LOG.debug("Applying MSS.")
        prediction = dgpred.apply_mss(prediction, options)
    else:
        prediction = dgpred.softmax(prediction)
    return np.asanyarray(prediction.argmax(axis=1)), start_pos


class CommandLineParser:
    """Commandline parser."""
    def __init__(self, **kwargs):
        kwargs.setdefault("prog", "deepgrp")
        kwargs.setdefault("formatter_class",
                          argparse.ArgumentDefaultsHelpFormatter)
        kwargs.setdefault("description",
                          "DeepGRP - Prediction of repetitive elements")

        self.parser = argparse.ArgumentParser(**kwargs)
        self.args = None
        self.threads = 1
        self.xla = False
        self.verbose = 0

        subparsers = self.parser.add_subparsers(help="sub-command help",
                                                dest="command")
        self.parser.add_argument(
            "--batch_size",
            "-b",
            type=int,
            default=256,
            help="Batch size to use for prediction with TensorFlow model ",
        )
        self.parser.add_argument(
            "--step_size",
            "-s",
            type=int,
            default=50,
            help="Window step size",
        )
        self.parser.add_argument(
            "--xdrop_length",
            "-x",
            type=int,
            default=50,
            help="XDrop parameter for MSS algorithm, ignored if "
            "--no_use_mss, disabled with values<0",
        )
        self.parser.add_argument(
            "--min_mss_length",
            "-l",
            type=int,
            default=50,
            help="Minimal length of maximum scoring segments, "
            "ignored if --no_use_mss",
        )
        self.parser.add_argument(
            "--threads",
            "-t",
            type=int,
            default=1,
            help="Number of threads (all=0)",
        )

        self.parser.add_argument("--xla",
                                 action="store_true",
                                 help="Enable XLA acceleration for TensorFlow")
        self.parser.add_argument("-v",
                                 "--verbose",
                                 action="count",
                                 default=0,
                                 help="Increase verbosity")

        train_subparser = subparsers.add_parser(
            name="train",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Train a deepgrp model",
        )

        train_subparser.add_argument("parameter",
                                     type=str,
                                     help="toml file with parameters")

        train_subparser.add_argument(
            "trainfile",
            type=str,
            help="Training data preprocessed with ´preprocess_sequence´",
        )
        train_subparser.add_argument(
            "validfile",
            type=str,
            help="Validation data preprocessed with ´preprocess_sequence´",
        )
        train_subparser.add_argument(
            "bedfile",
            type=str,
            help="Ground truth repeat annotation data.",
        )
        train_subparser.add_argument(
            "--logdir",
            type=str,
            default=".",
            help="Directory for the Tensorflow log files.",
        )
        train_subparser.add_argument(
            "--modelfile",
            type=str,
            default="model.hdf5",
            help="Output path for the model file.",
        )

        predict_subparser = subparsers.add_parser(
            name="predict",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="predict using a deepgrp model",
        )

        predict_subparser.add_argument("model",
                                       type=str,
                                       help="TensorFlow Model in HDF5 format")
        predict_subparser.add_argument("FASTA",
                                       nargs="+",
                                       type=str,
                                       help="Fasta input files")
        predict_subparser.add_argument("--output",
                                       type=str,
                                       default="-",
                                       help="Output filename")
        predict_subparser.add_argument(
            "--no_use_mss",
            "-m",
            action="store_true",
            help="Disable maximum scoring segment algorithm",
        )

    def parse_args(self) -> "CommandLineParser":
        """Parse command line arguments."""
        args = self.parser.parse_args()
        self.threads = args.threads
        self.verbose = args.verbose
        self.xla = args.xla
        self.args = args
        return self

    def setup_tensorflow(self) -> "CommandLineParser":
        """Set TensorFlow threads and XLA."""
        if self.threads > 0:
            inter_op_threads = max(1, self.threads // 2)
            intra_op_threads = max(1, inter_op_threads + self.threads % 2)
            tf.config.threading.set_inter_op_parallelism_threads(
                inter_op_threads)
            tf.config.threading.set_intra_op_parallelism_threads(
                intra_op_threads)
        if self.xla:
            os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
            tf.config.optimizer.set_jit(True)
        return self

    def set_logging(self) -> "CommandLineParser":
        """Setup logging."""
        loglevels = [logging.WARNING, logging.INFO, logging.DEBUG]
        loglevel = loglevels[min(len(loglevels) - 1, self.verbose)]
        _LOG.setLevel(level=loglevel)
        tf.debugging.set_log_device_placement(loglevel == logging.DEBUG)
        return self

    def run(self):
        """Run the command provided by the user."""
        options = dgmodel.Options(
            min_mss_len=self.args.min_mss_length,
            batch_size=self.args.batch_size,
            xdrop_len=self.args.xdrop_length,
        )
        getattr(self, self.args.command)(self.args, options)

    @staticmethod
    def predict(args: argparse.Namespace, options: dgmodel.Options):
        """Predict with deepgrp.

        Args:
            args (argparse.Namespace): Command line arguments.
                Includes: 'model', 'output', 'FASTA', 'no_use_mss',
                'step_size'
            options (dgmodel.Options): Default and user provided options.
        """

        _LOG.debug("Loading model %s!", args.model)
        model = tf.keras.models.load_model(
            args.model,
            custom_objects={
                "ReverseComplement": dgmodel.ReverseComplement,
            },
        )
        options.vecsize = model.input_shape[1]
        _LOG.info("Model loading finished successfully!")
        outstream = sys.stdout if args.output == "-" else open(
            args.output, "w")

        for filename in args.FASTA:
            _LOG.info("Processing %s", filename)
            try:
                filestream = sys.stdin if filename == "-" else open(
                    filename, "r")
                for header, dnasequence in _read_multi_fasta(filestream):
                    predictions, startpos = _predict(
                        dnasequence,
                        model,
                        options,
                        args.step_size,
                        use_mss=not args.no_use_mss,
                    )
                    for segment in dgsequence.yield_segments(
                            predictions, startpos):
                        if segment[2] > 0:
                            outstream.write("{}\t{}\t{}\t{}\t{}\n".format(
                                filename, header, *segment))
            finally:
                if filename != "-":
                    filestream.close()
        if args.output != "-":
            outstream.close()

    @staticmethod
    def train(args: argparse.Namespace, options: dgmodel.Options) -> None:
        """Train deepgrp.

        Args:
            args (argparse.Namespace): Command line arguments.
                Includes: 'parameter', 'trainfile', 'validfile', 'bedfile',
                'logdir', 'modelfile'
            options (dgmodel.Options): Default and user provided options.
        """
        with open(args.parameter, "r") as file:
            parameter = dgmodel.Options.from_toml(file)
        parameter.fromdict(options.todict())
        logdir = args.logdir

        # get which chromosome is used for training and which for validation
        train_chr = os.path.basename(args.trainfile).split('.')[0]
        val_chr = os.path.basename(args.validfile).split('.')[0]

        # check if logdir exists?
        if not os.path.isdir(logdir):
            # logdir is not valid
            sys.stderr.write('Given logdir is not a Directory\n')
            sys.exit(1)

        # load data
        _LOG.info("Loading in all data necessary from %s, %s, %s",
                  args.trainfile, args.validfile, args.bedfile)
        train_fwd = np.load(args.trainfile, allow_pickle=False)['fwd']
        val_fwd = np.load(args.validfile, allow_pickle=False)['fwd']

        # preprocess
        ## preprocess y (bedfile)
        y_train = dgpreprocess.preprocess_y(args.bedfile, train_chr,
                                            train_fwd.shape[1],
                                            parameter.repeats_to_search)
        y_val = dgpreprocess.preprocess_y(args.bedfile, val_chr,
                                          val_fwd.shape[1],
                                          parameter.repeats_to_search)

        ## preprocess training and validation data
        train_fwd, y_train = dgpreprocess.drop_start_end_n(train_fwd, y_train)
        val_fwd, y_val = dgpreprocess.drop_start_end_n(val_fwd, y_val)
        train_data = dgpreprocess.Data(train_fwd, y_train)
        val_data = dgpreprocess.Data(val_fwd, y_val)

        # training
        _LOG.info("Creating model for training")
        model = dgmodel.create_model(parameter)
        _LOG.info("Training Model")
        dgtrain.training((train_data, val_data), parameter, model, logdir)

        # save model in h5 format
        _LOG.info("Saving model as %s", args.modelfile)
        model.save(args.modelfile)


def main():
    """ Main function """
    CommandLineParser().parse_args().set_logging().setup_tensorflow().run()


if __name__ == "__main__":
    main()
