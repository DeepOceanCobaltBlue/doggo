from __future__ import annotations


def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))
