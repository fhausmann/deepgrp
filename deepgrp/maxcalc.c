#include "maxcalc.h"

float *_get_max(float *output, float *inputs, size_t dim0, size_t dim1,
                size_t stride, size_t batchsize) {
  size_t total_vecsize = dim0 * dim1;
  size_t stepsize = stride * dim1;
  float *current_out = output;

  for (size_t batch_idx = 0; batch_idx < batchsize; ++batch_idx) {
    current_out += stepsize;
    for (size_t i = 0; i < total_vecsize; ++i) {
      if (inputs[i] > current_out[i]) {
        current_out[i] = inputs[i];
      }
    }
  }
  return output;
}
