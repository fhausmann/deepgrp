""" Build file for cython extensions """
from distutils.core import Extension
import numpy
from Cython.Build import cythonize

_EXTENSIONS = [
    Extension("deepgrp.mss",
              sources=["deepgrp/_mss/pymss.pyx", "./deepgrp/_mss/mss.c"],
              include_dirs=[numpy.get_include()] + ["./deepgrp"]),
    Extension("deepgrp.sequence",
              sources=["deepgrp/sequence.pyx"],
              include_dirs=[numpy.get_include()]),
]


def build(setup_kwargs):
    """
    This function is mandatory in order to build the extensions.
    """

    setup_kwargs.update({'ext_modules': cythonize(_EXTENSIONS)})
