"""Pixel boost re-assembly: interleave the boost x boost swapped sub-grids without a mesh pattern.

Each sub-grid is swapped on its own, so the outputs differ slightly (and on some GPUs a batched run can return
them badly out of step). Interleaving outputs that differ shows up as a fine screen-door mesh over the face.
The fix: every sub-grid gets the SAME low frequencies (their average) and keeps only its own fine detail, and a
sub-grid that disagrees strongly with the others is reported so the caller can re-run it.
"""
from __future__ import annotations

import cv2
import numpy as np

MISMATCH = 6.0  # mean abs difference (0..255) of a blurred sub-grid from the median one; normal is ~1.5


def _low(y: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(y, (0, 0), sigma)


def mismatched(ys: np.ndarray, sigma: float = 2.0) -> list:
    """Indices of sub-grid outputs whose overall picture disagrees with the rest."""
    lows = np.stack([_low(y, sigma) for y in ys])
    med = np.median(lows, axis=0)
    return [k for k in range(len(ys)) if float(np.abs(lows[k] - med).mean()) > MISMATCH]


def assemble(shape, ys: np.ndarray, boost: int, sigma: float = 1.0) -> np.ndarray:
    """Interleave sub-grid outputs ys (boost*boost, n, n, 3) into one image of `shape`, mesh-free."""
    lows = np.stack([_low(y, sigma) for y in ys])
    common = lows.mean(0)
    out = np.empty(shape, np.float32)
    k = 0
    for i in range(boost):
        for j in range(boost):
            out[i::boost, j::boost] = ys[k] - lows[k] + common
            k += 1
    return np.clip(out, 0, 255)
