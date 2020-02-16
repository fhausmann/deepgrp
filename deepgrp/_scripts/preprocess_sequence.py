#!/usr/bin/env python3
""" Script for creating one hot encode DNA sequences from FASTA"""
from typing import Tuple, BinaryIO, Dict
import argparse
import gzip
import hashlib
import sys
import numpy as np

_ENCODEDICT: Dict[str, int] = {
    'A': 0,
    'C': 1,
    'G': 2,
    'T': 3,
    'N': 4,
}


def fastaparser(filestream: BinaryIO) -> Tuple[str, str, str]:
    """Reads a fasta sequence and creates hash of sequence.

    Args:
        filestream (BinaryIO): Stream of input FASTA.

    Returns:
        Tuple[str, str, str]: Header, Hash and sequence.

    """
    sequence = list()
    hash_md5 = hashlib.md5()
    for line in filestream:
        line = line.strip()
        if line[0] == 62:  # ASCII '>'
            header = line[1:].decode()
        else:
            sequence.append(line.decode().upper())
            hash_md5.update(line)
    return header, str(hash_md5.hexdigest()), ''.join(sequence)


def main():
    """ main function """

    parser = argparse.ArgumentParser(
        description='Format fasta file to onehot encoded sequences')
    parser.add_argument('FASTAFILE', type=str, help='Fastafile (gzip)')
    parser.add_argument('--force',
                        action='store_true',
                        help='forces recreation even if files not changed')

    args = parser.parse_args()
    create_new = args.force
    try:
        infile = gzip.open(args.FASTAFILE, 'rb')  # type: BinaryIO
        _, hash_val, seq = fastaparser(infile)
        infile.close()
    except IOError:
        sys.stderr.write('Could not open file!\n')
        sys.exit(1)

    try:
        infile_hash = np.load(args.FASTAFILE + '.npz')["hash"]
        if hash_val != infile_hash[0]:
            create_new = True
    except (IOError, KeyError):
        create_new = True

    if create_new:
        seqenc = np.zeros((len(_ENCODEDICT), len(seq)), dtype=np.int8)
        seqenc[np.fromiter(([_ENCODEDICT[char] for char in seq]), np.int8),
               np.arange(seqenc.shape[1])] = 1  # pylint: disable=E1136
        np.savez_compressed(args.FASTAFILE,
                            fwd=seqenc,
                            hash=np.array([hash_val]))


if __name__ == '__main__':
    main()
