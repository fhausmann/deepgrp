"""deepgrp.preprocessing module for preprocessing"""

from typing import Tuple, List, NamedTuple
import os
import pandas as pd
import numpy as np


def preprocess_y(filename: os.PathLike, chromosom: str, length: int,
                 repeats_to_search: List[int]) -> np.ndarray:
    """Encodes an space seperated file produced by script/parse_rm.py
    as np.ndarray.

    Args:
        filename (os.PathLike): Filename of the space seperated input file.
        chromosom (str): Chromosome name, like `chr2`.
        length (int): Lenght of the chromosom in bp.
        repeats_to_search (List[int]): Repeat numbers to search for.

    Returns:
        np.ndarray: One hot encoded annotations from file.

    """
    data = pd.read_csv(filename,
                       sep=r'\s+',
                       header=None,
                       index_col=False,
                       usecols=[0, 1, 2, 3])
    data.columns = ['chromosom', 'begin', 'end', 'repeatnumber']
    data = data[data.chromosom == chromosom]
    data.drop('chromosom', axis=1, inplace=True)

    bool_series = None
    for number in repeats_to_search:
        if bool_series is None:
            bool_series = (data.repeatnumber == number)
        else:
            bool_series |= (data.repeatnumber == number)
    data = data[bool_series]
    yarray = np.zeros((len(repeats_to_search) + 1, length), dtype=np.int8)

    def assign_to_y(row):
        yarray[row.repeatnumber, row.begin:row.end] = 1

    data.apply(assign_to_y, axis=1)
    yarray[0, yarray[1:].sum(axis=0) == 0] = 1
    del data
    return yarray


def drop_start_end_n(fwd: np.ndarray,
                     array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Drops leading and trailing N's of sequences.

    Args:
        fwd (np.ndarray): Forward sequence.
        cmp (np.ndarray): Complement sequence.
        array (np.ndarray): True annotation label of sequence.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Slice of the arguments.

    """
    start = 0
    sums = fwd[0:4].sum(axis=0)
    start = np.argmax(sums > 0)
    end = fwd.shape[1] - 1 - np.argmax(np.flip(sums) > 0)
    return fwd[:, start:end], array[:, start:end]


# Collection of forward one hot encoded sequence and true annotations
Data = NamedTuple("Data", [('fwd', np.ndarray), ('truelbl', np.ndarray)])
