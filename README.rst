==================================================================
DeepGRP - Deep learning for Genomic Repetitive element Prediction
==================================================================

|PyPI version fury.io|

.. |PyPI version fury.io| image:: https://badge.fury.io/py/deepgrp.svg
   :target: https://pypi.org/project/deepgrp/

DeepGRP is a python package used to predict genomic repetitive elements
with a deep learning model consisting of bidirectional gated recurrent units
with attention.
The idea of DeepGRP was initially based on `dna-nn`__, but was re-implemented
and extended using `TensorFlow`__ 2.1.
DeepGRP was tested for the prediction of HSAT2,3, alphoid, Alu
and LINE-1 elements.

.. __: https://github.com/lh3/dna-nn
.. __: https://www.tensorflow.org

Getting Started
===============

Installation
------------

For installation you can use the PyPI version with::

    pip install deepgrp

or install from this repository with::

    git clone https://github.com/fhausmann/deepgrp
    cd deepgrp
    pip install .

Additionally you can install the developmental version with `poetry`__::

    git clone https://github.com/fhausmann/deepgrp
    cd deepgrp
    poetry install

.. __: https://python-poetry.org/

Data preprocessing
------------------
For training and hyperparameter optimization the data have to be preprocessed.
For inference / prediction the FASTA sequences can directly be used and you
can skip this process.
The provided script `parse_rm` can be used to extract repeat annotations from
`RepeatMasker`__ annotations to a TAB seperated format by::

    parse_rm GENOME.fa.out > GENOME.bed

.. __: http://www.repeatmasker.org/

The FASTA sequences have to be converted to a one-hot-encoded representation,
which can be done with::

    preprocess_sequence FASTAFILE.fa.gz

`preprocess_sequence` creates a one-hot-encoded representation in numpy
compressed format in the same directory.


Hyperparameter optimization
---------------------------
For Hyperparameter optimization the github repository provides
a jupyter `notebook`__ which can be used.

.. __: https://github.com/fhausmann/deepgrp/blob/master/notebooks/DeepGRP.ipynb

Hyperparameter optimization is based on the `hyperopt`__ package.

.. __: https://github.com/hyperopt/hyperopt

Training
--------

Training of a model can be performed with the provided jupyter `notebook`__.

.. __: https://github.com/fhausmann/deepgrp/blob/master/notebooks/Training.ipynb

Prediction
----------
The prediction can be done with the deepgrp main function like::

    deepgrp <modelfile> <fastafile> [<fastafile>, ...]

where `<modelfile>` contains the trained model in `HDF5`__
format and `<fastafile>` is a (multi-)FASTA file containing DNA sequences.
Several FASTA files can be given at once.

.. __: https://www.tensorflow.org/tutorials/keras/save_and_load

Requirements
============
Requirements are listed in `pyproject.toml`__.

.. __: https://github.com/fhausmann/deepgrp/blob/master/pyproject.toml

Additionally for compiling C/Cython code, a C compiler should be installed.

Further information
===================
You can find material to reproduce
the results in the repository `deepgrp_reproducibility`__.

.. __: https://github.com/fhausmann/deepgrp_reproducibility
