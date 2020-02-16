cdef extern from "mss.h":
  ctypedef struct msseg_t:
    int st
    int en
    double sc

  msseg_t *mss_find_all(int n, const double *S, double min_sc, double xdrop, int *n_seg);
