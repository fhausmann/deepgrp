"""
Functions for processing sequences

"""

cimport numpy as np
cimport cython
import numpy as np
import deepgrp.preprocessing as DNAprep

cdef int*  ONEHOT = [
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0,
    4, 1, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 1, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4];

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef _one_hot_encode_dna_sequence(const unsigned char[:] sequence):
    """One hot encodes sequence, drops leading and trailing N's."""

    cdef long length = len(sequence)
    cdef long i, pos, startpos = 0

    while startpos < length and sequence[startpos]=='N':
        startpos+=1
    while length > 0 and sequence[length-1]=='N':
        length-=1

    cdef np.ndarray[np.int8_t, ndim=2] fwd = np.zeros((5, length-startpos), dtype=np.int8)

    for i in range(length-startpos):
        fwd[ONEHOT[<long>sequence[startpos+i]],i]=1
    return startpos, fwd

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def get_segments(np.ndarray[long,ndim=1] classes, long startpos):
    """Gets start, end and label of non-null segments from array of labels"""
    cdef long end
    cdef long length = classes.size
    cdef long currentlabel = classes[startpos]

    while startpos < length and currentlabel==0:
        startpos+=1
        currentlabel=classes[startpos]
    end = startpos+1

    while end<length and classes[end] == currentlabel:
        end+=1
    return startpos, end, currentlabel

def one_hot_encode_dna_sequence(sequence: str):
    """One hot encodes sequence, drops leading and trailing N's."""
    byte = sequence.encode('utf-8')
    return _one_hot_encode_dna_sequence(byte)


cdef extern from "maxcalc.h":
    float *_get_max(float *output, float *inputs, size_t dim0, size_t dim1,
               size_t stride, size_t batchsize)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def get_max(np.ndarray[np.float32_t, ndim=2, mode = 'c'] output,
               np.ndarray[np.float32_t, ndim=3, mode = 'c'] inputs,
               int stride):
    cdef float* in_buff = <float*> inputs.data
    cdef float* out_buff = <float*> output.data
    cdef unsigned int batchsize = inputs.shape[0]
    cdef unsigned int dim0  = inputs.shape[1]
    cdef unsigned int dim1  = inputs.shape[2]
    _get_max(out_buff, in_buff, dim0, dim1, stride, batchsize)
    return output


def yield_segments(classes: np.array, start_offset: int):
    """Converts numpy array of classes to Iterator over continuous segments."""
    i = 0
    while i < classes.size:
        start, end, label = get_segments(classes, i)
        i = end
        yield start + start_offset, end + start_offset, label
