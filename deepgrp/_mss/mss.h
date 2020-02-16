/**\file mss.h
 * Linear-time maximum scoring segment algorithm
 * Code by Heng Li (https://github.com/lh3/dna-nn/blob/master/mss.c)
 */

#ifndef DEEPGRP_C_MSS_H_
#define DEEPGRP_C_MSS_H_

#define MSS_FLOAT double

typedef struct {
  int st, en;
  MSS_FLOAT sc;
} msseg_t;

msseg_t *mss_find_all(int n, const MSS_FLOAT *S, MSS_FLOAT min_sc,
                      MSS_FLOAT xdrop, int *n_seg);

#endif  // DEEPGRP_C_MSS_H_
