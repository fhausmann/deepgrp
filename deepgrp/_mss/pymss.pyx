"""
Cython wrapper for Maximum Scoring segment algorithm

"""

import cython

import numpy as np
cimport numpy as np
cimport mss
cimport libc.math
cimport libc.stdlib

@cython.boundscheck(False)
@cython.wraparound(False)
def find_mss_labels(double[:] inputs not None, long[:] label not None,
                    int nof_labels, int min_mss_len, int xdrop_len):
  """
  Takes a numpy array as inputs and assigning the maximum scoring segments with labels

  param: array -- a 1-d numpy array of np.float64
  param: array -- a 1-d numpy array of np.int
  """
  cdef int length = inputs.shape[0]
  cdef np.ndarray[double, ndim=2, mode="c"] one_hot_output = np.zeros((length,nof_labels))
  _find_mss_labels(inputs,label,nof_labels,min_mss_len,xdrop_len,one_hot_output,length)
  return one_hot_output

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _find_mss_labels(double[:] inputs,
                      long[:] label,
                      int nof_labels, int min_mss_len, int xdrop_len,
                      double[:,:] one_hot_output,
                      int length):

  cdef int n_segments, i, max_idx
  cdef double xdrop, min_sc, s0
  cdef msseg_t *segs
  cdef long * labelcounts = <long*>libc.stdlib.malloc(nof_labels * sizeof(long))
  cdef long max_val, j
  cdef long pos = 0

  n_segments = 0

  s0 = libc.math.log( 0.99 / (1.0 -  0.99));

  if  xdrop_len > 0:
    xdrop = s0 * xdrop_len * 10.0
  else:
    xdrop = -1

  min_sc = s0 * min_mss_len;

  segs = mss_find_all(length, &inputs[0], min_sc, xdrop, &n_segments)

  for i in range(n_segments):
    for j in range(nof_labels):
      labelcounts[j]=0
    for j in range(segs[i].st,segs[i].en):
      labelcounts[label[j]]+=1
    max_idx = 1
    max_val = labelcounts[1]
    for j in range(2,nof_labels):
      if max_val<labelcounts[j]:
        max_idx = j
        max_val = labelcounts[j]
    for j in range(segs[i].st,segs[i].en):
      if label[j]==0:
        one_hot_output[j,max_idx]=1
      else:
        one_hot_output[j,label[j]]=1
    for j in range(pos,segs[i].st):
      one_hot_output[j,label[j]]=1
    pos = segs[i].en
  for j in range(pos,length):
    one_hot_output[j,label[j]]=1

  libc.stdlib.free(segs)
  libc.stdlib.free(labelcounts)
