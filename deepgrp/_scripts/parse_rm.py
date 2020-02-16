#!/usr/bin/env python3
"""Script to parse Repeatmasker output"""
import argparse
import collections
import re
from typing import Dict, List, Pattern, Tuple, TextIO, Optional

_COMP_TBL: Dict[str, str] = str.maketrans({
    'A': 'T',
    'T': 'A',
    'C': 'G',
    'G': 'C'
})
_BASES: str = "ACGT"
_MOTIF0: str = "GGAAT"

_REPEATS: List[str] = [
    "HSATII",
    "ALR/Alpha",
    "SINE/Alu",
    "LINE/L1",
    "SINE/MIR",
    "LINE/L2",
    "LTR/ERV1",
    "LTR/ERVL",
    "LTR/ERVL-MaLR",
    "LTR/Gypsy",
]

_TYPES: collections.defaultdict = collections.defaultdict(
    int, **{k: v
            for v, k in enumerate(_REPEATS, 1)})

_REGEX1: Pattern[str] = re.compile(
    r"^\s*\d+\s+\S+\s+\S+\s+\S+\s+(\S+)\s+" +
    r"(\d+)\s+(\d+)\s+\S+\s+[+C]\s+(\S+)\s+(\S+)")
_REGEX2: Pattern[str] = re.compile(
    r"^\d+(\t\d+){4}\t(\S+)\t(\d+)\t(\d+)\t\S+\t[+-]\t(\S+)\t(\S+)\t(\S+)")


def get_rev_comp(motif: List[str]) -> List[str]:
    """Creates reverse complements."""
    return [mot[::-1].translate(_COMP_TBL) for mot in motif]


def rotate(motif: List[str]) -> List[str]:
    """Rotate Motifs"""
    motif_alt = []
    for i, _ in enumerate(motif):
        for j in range(1, len(motif[i])):
            motif_alt.append(motif[i][j:] + motif[i][0:j])
    return motif_alt


def mutate(motifs: List[str]) -> Dict[str, int]:
    """ Mutate motifs"""
    motif_mut_hash = {}
    for mot in motifs:
        for i, char in enumerate(mot):
            new_motif = list(mot)
            for j in _BASES:
                if char == j:
                    continue
                new_motif[i] = j
                motif_mut_hash[''.join(new_motif)] = 1
    return motif_mut_hash


class Repeat:  # pylint: disable=too-few-public-methods
    """ Struct to hold repeat information in"""
    ctg: Optional[str]
    end: Optional[int]
    fam: Optional[str]
    rep: str
    start: Optional[int]
    typ: int

    def __init__(self) -> None:
        self.ctg = None
        self.start = None
        self.end = None
        self.rep = ""
        self.fam = None
        self.typ = 0

    def __str__(self) -> str:
        return "{}\t{}\t{}\t{}\t{}\t{}".format(self.ctg, self.start, self.end,
                                               self.typ, self.rep, self.fam)


def _get_repeat(line: str) -> Repeat:
    match1 = _REGEX1.match(line)
    match2 = _REGEX2.match(line)
    rep = Repeat()
    if match1:
        rep.ctg = match1.group(1)
        rep.start = int(match1.group(2)) - 1
        rep.end = match1.group(3)
        rep.rep = match1.group(4)
        rep.fam = match1.group(5)
    elif match2:
        rep.ctg = match2.group(2)
        rep.start = match2.group(3)
        rep.end = match2.group(4)
        rep.rep = match2.group(5)
        if match2.group(6) == match2.group(7):
            rep.fam = match2.group(6)
        else:
            rep.fam = match2.group(6) + '/' + match2.group(7)
    rep.typ = _TYPES[rep.fam]
    if rep.typ == 0:
        rep.typ = _TYPES[rep.rep]
    return rep


def _check_motifs(motif: str, motif_mut_hash: Dict[str, int],
                  motif_hash: Dict[str, int]) -> Tuple[int, int]:
    count = 0
    count_mut = 0
    for j in range(0, len(motif), len(_MOTIF0)):
        smotif = motif[j:j + len(_MOTIF0)]
        if smotif in motif_hash:
            count += 1
        elif smotif in motif_mut_hash:
            count_mut += 1
    return count, count_mut


def read_repeatmasker(motif_hash: Dict[str, int], motif_mut_hash: Dict[str,
                                                                       int],
                      filestream: TextIO) -> None:
    """Reads Repeatmasker output and prints to stdout.

    Args:
        motif_hash (Dict[str, int]): Simple repeats to search for
        motif_mut_hash (Dict[str, int]): Mutated simple repeats to search for
        filestream (TextIO): Stream with Repeatmasker output
    """

    len0 = len(_MOTIF0)
    for line in filestream.readlines():
        repeat = _get_repeat(line)

        if repeat.typ == 0 and repeat.fam in ("Simple_repeat", "Satellite"):
            motif = re.match(r"^\(([ACGT]+)\)n", repeat.rep)
            if motif is not None:
                motif = motif.group(1)
                if motif in motif_hash:
                    repeat.typ = _TYPES['HSATII']
                elif len(motif) % len0 == 0:
                    count, count_mut = _check_motifs(motif, motif_mut_hash,
                                                     motif_hash)
                    if count > 0 and (count + count_mut) * len0 == len(motif):
                        repeat.typ = _TYPES['HSATII']
        if repeat.ctg and repeat.typ > 0:
            print(repeat)


def main():
    """ main function """
    parser = argparse.ArgumentParser(
        description='Reads Repeatmasker output to bed file (not all repeats!!)'
    )
    parser.add_argument('file',
                        type=argparse.FileType('r'),
                        help='Repeatmasker output')
    args = parser.parse_args()

    motiflist = [_MOTIF0]
    motiflist += get_rev_comp(motiflist)
    motiflist += rotate(motiflist)
    motif_mut_hash = mutate(motiflist)
    motif_hash = {m: k for k, m in enumerate(motiflist)}
    read_repeatmasker(motif_hash, motif_mut_hash, args.file)


if __name__ == '__main__':
    main()
