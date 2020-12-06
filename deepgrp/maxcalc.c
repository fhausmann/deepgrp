#include "maxcalc.h"

#define MAX(a, b)                                                              \
  ({                                                                           \
    __typeof__(a) _a = (a);                                                    \
    __typeof__(b) _b = (b);                                                    \
    _a > _b ? _a : _b;                                                         \
  })

float *_get_max(float *output, float *inputs, size_t dim0, size_t dim1,
                size_t stride, size_t batchsize) {
  size_t total_vecsize = dim0 * dim1;
  size_t stepsize = stride * dim1;
  float *current_out = output;

  for (size_t batch_idx = 0; batch_idx < batchsize; ++batch_idx) {
    for (size_t i = 0; i < total_vecsize; ++i) {
      current_out[i] = MAX(current_out[i], inputs[i]);
    }
    inputs += total_vecsize;
    current_out += stepsize;
  }
  return output;
}

#undef MAX
