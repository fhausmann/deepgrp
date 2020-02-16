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
from deepgrp.sequence import one_hot_encode_dna_sequence, yield_segments  # pylint: disable=import-error, no-name-in-module

logging.basicConfig()
_LOG = logging.getLogger(__name__)


def read_multi_fasta(filestream: TextIO) -> Iterator[Tuple[str, str]]:
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
        if line[0] == '>':
            if header:
                yield header, ''.join(sequence)
            header = line[1:]
            sequence = []
        else:
            sequence.append(line.upper())
    if header:
        yield header, ''.join(sequence)


def predict(model: tf.keras.Model, inputs: np.ndarray,
            options: dgmodel.Options, step_size: int,
            use_mss: bool) -> np.ndarray:
    """Runs a prediction for one sequence .

    Args:
        model (tf.keras.Model): Model to use for prediction
        inputs (np.ndarray): One hot encoded sequence
        options (dgmodel.Options): Hyperparameter used for model
                                    creation and prediction
        step_size (int): Window step size.
        use_mss (bool): Use MSS algorithm.

    Returns:
        np.ndarray: predictions from TensorFlow and potential the MSS

    """
    _LOG.debug("Start prediction.")

    data_iterator = dgpred.fetch_validation_batch(inputs, step_size,
                                                  options.batch_size,
                                                  options.vecsize)
    output_shape = (inputs.shape[1], model.output_shape[2])
    prediction = dgpred.predict(model, data_iterator, output_shape, step_size)
    _LOG.debug("Finish prediction.")
    if use_mss:
        _LOG.debug("Applying MSS.")
        return dgpred.apply_mss(prediction, options)
    return dgpred.softmax(prediction)


def _main(args: 'argparse.Namespace'):
    if args.threads > 0:
        inter_op_threads = max(1, args.threads // 2)
        intra_op_threads = max(1, inter_op_threads + args.threads % 2)
        tf.config.threading.set_inter_op_parallelism_threads(inter_op_threads)
        tf.config.threading.set_intra_op_parallelism_threads(intra_op_threads)
    _LOG.debug("Loading model %s!", args.model)
    model = tf.keras.models.load_model(args.model,
                                       custom_objects={
                                           'ReverseComplement':
                                           dgmodel.ReverseComplement,
                                       })
    _LOG.info("Model loading finished successfully!")

    options = dgmodel.Options(min_mss_len=args.min_mss_length,
                              batch_size=args.batch_size,
                              xdrop_len=args.xdrop_length,
                              vecsize=model.input_shape[1])
    outstream = sys.stdout if args.output == '-' else open(args.output, 'w')

    for file in args.FASTA:
        if not os.access(file, os.R_OK):
            _LOG.warning("Could not read %s, skipping!", file)
            continue
        _LOG.info("Processing %s", file)
        filestream = sys.stdin if file == '-' else open(file, 'r')
        for header, sequence in read_multi_fasta(filestream):
            _LOG.debug("One hot encoding sequence.")
            start_pos, one_hot_sequence = one_hot_encode_dna_sequence(sequence)
            del sequence
            predictions = predict(model,
                                  one_hot_sequence,
                                  options,
                                  args.step_size,
                                  use_mss=not args.no_use_mss)
            del one_hot_sequence
            for segment in yield_segments(predictions.argmax(axis=1),
                                          start_pos):
                if segment[2] > 0:
                    outstream.write("{}\t{}\t{}\t{}\t{}\n".format(
                        file, header, *segment))
        if file != "-":
            filestream.close()
    if args.output != '-':
        outstream.close()


def main():
    """ Main function """
    parser = argparse.ArgumentParser(
        prog="deepgrp",
        description="DeepGRP - "
        "Prediction of repetitive elements using pretrained TensorFlow model "
        "with FASTA input files")

    parser.add_argument(
        '--batch_size',
        '-b',
        type=int,
        default=256,
        help="Batch size to use for prediction with TensorFlow model "
        "(default: 256)")
    parser.add_argument('--step_size',
                        '-s',
                        type=int,
                        default=50,
                        help="Window step size (default: 50)")
    parser.add_argument('--xdrop_length',
                        '-x',
                        type=int,
                        default=50,
                        help="XDrop parameter for MSS algorithm, ignored if "
                        "--no_use_mss, disabled with values<0 (default: 50)")
    parser.add_argument('--min_mss_length',
                        '-l',
                        type=int,
                        default=50,
                        help="Minimal length of maximum scoring segments, "
                        "ignored if --no_use_mss (default: 50)")
    parser.add_argument('--threads',
                        '-t',
                        type=int,
                        default=1,
                        help="Number of threads (default:1, all=0)")
    parser.add_argument('--no_use_mss',
                        '-m',
                        action='store_true',
                        help="Disable maximum scoring segment algorithm")
    parser.add_argument('--output',
                        type=str,
                        default='-',
                        help="Output filename (default: stdout)")
    parser.add_argument('--xla',
                        action='store_true',
                        help="Enable XLA acceleration for TensorFlow")
    parser.add_argument('-v',
                        '--verbose',
                        action='count',
                        default=0,
                        help="Increase verbositiy")
    parser.add_argument('model',
                        type=str,
                        help="TensorFlow Model in HDF5 format")
    parser.add_argument('FASTA', nargs='+', type=str, help="Fasta input files")

    args = parser.parse_args()

    loglevels = [logging.WARNING, logging.INFO, logging.DEBUG]
    loglevel = loglevels[min(len(loglevels) - 1, args.verbose)]
    _LOG.setLevel(level=loglevel)
    tf.debugging.set_log_device_placement(loglevel == logging.DEBUG)
    if args.xla:
        os.environ['TF_XLA_FLAGS'] = "--tf_xla_cpu_global_jit"
        tf.config.optimizer.set_jit(True)
    _main(args)


if __name__ == '__main__':
    main()
