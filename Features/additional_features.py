# additional_features.py

from typing import List

def gc_content(seq: str) -> float:
    """计算序列的GC含量"""
    gc_count = seq.count('G') + seq.count('C')
    return gc_count / len(seq)

def sequence_length(seq: str) -> int:
    """计算序列的长度"""
    return len(seq)

def calculate_additional_features(seq: str) -> List[float]:
    gc = gc_content(seq)
    length = sequence_length(seq)
    return [gc, length]

