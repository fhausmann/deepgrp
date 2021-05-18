FROM tensorflow/tensorflow:2.1.0-gpu-py3 as base

ENV LC_ALL C.UTF-8
ENV TZ=Europe/Berlin
ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  NUMBA_CACHE_DIR=/tmp \
  POETRY_VIRTUALENVS_CREATE=false \
  OMP_NUM_THREADS=4 \
  NUMBA_NUM_THREADS=4
RUN ulimit -c 0
RUN pip install -U pip

FROM base as builder
RUN pip install -U poetry
RUN mkdir /build
WORKDIR /build
COPY ./ /build/
RUN pip install Cython
RUN poetry build --format wheel
RUN ls -l dist/

FROM base as deepgrp
RUN mkdir /deepgrp
WORKDIR /deepgrp
COPY --from=builder /build/dist/*.whl .
RUN pip install *.whl && rm *.whl
