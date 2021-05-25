from typing import Iterator, Tuple

import numpy as np


def get_max(output: np.ndarray, inputs: np.ndarray, stride: int) -> np.ndarray:
    ...


def get_segments(classes: np.ndarray, startpos: int) -> Tuple[int, int, int]:
    ...


def one_hot_encode_dna_sequence(sequence: str) -> np.ndarray:
    ...


def yield_segments(classes: np.array,
                   start_offset: int) -> Iterator[Tuple[int, int, int]]:
    ...
