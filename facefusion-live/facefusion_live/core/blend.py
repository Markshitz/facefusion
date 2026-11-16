"""Masks, colour matching and ROI-restricted paste-back."""
from __future__ import annotations

import cv2
import numpy as np

from facefusion_live.core.align import transform_points

_box_cache: dict = {}


def box_mask(size: int, blur: float, padding: float = 0.0) -> np.ndarray:
    """Feathered rectangle mask in crop space (float32 0..1)."""
    key = (size, round(blur, 3), round(padding, 3))
    m = _box_cache.get(key)
    if m is None:
        amount = int(size * 0.5 * blur)
        area = max(amount // 2, 1)
        pad = max(area, int(size * padding))
        m = np.ones((size, size), np.float32)
        m[:pad, :] = 0
        m[-pad:, :] = 0
        m[:, :pad] = 0
        m[:, -pad:] = 0
        if amount > 0:
            m = cv2.GaussianBlur(m, (0, 0), amount * 0.25)
        _box_cache[key] = m
    return m


def soften(mask: np.ndarray, sigma: float = 5) -> np.ndarray:
    m = cv2.GaussianBlur(np.clip(mask, 0, 1).astype(np.float32), (0, 0), sigma)
    return (np.clip(m, 0.5, 1) - 0.5) * 2


def mouth_mask(lm68_crop: np.ndarray, size: int, grow: float = 0.15) -> np.ndarray:
    """Inner+outer lip polygon of the ORIGINAL face, in crop coordinates."""
    m = np.zeros((size, size), np.float32)
    pts = lm68_crop[48:60]
    c = pts.mean(0)
    pts = c + (pts - c) * (1 + grow)
    cv2.fillPoly(m, [pts.astype(np.int32)], 1.0)
    k = max(int(size * 0.02), 1)
    return cv2.GaussianBlur(m, (0, 0), k)


def color_transfer(src: np.ndarray, ref: np.ndarray, mask: np.ndarray, strength: float) -> np.ndarray:
    """Match LAB mean/std of src to ref inside mask. src/ref float32 BGR 0..255."""
    if strength <= 0:
        return src
    s = cv2.cvtColor(np.clip(src, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    r = cv2.cvtColor(ref.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    w = mask > 0.5
    if w.sum() < 64:
        return src
    sm, ss = s[w].mean(0), s[w].std(0) + 1e-3
    rm, rs = r[w].mean(0), r[w].std(0) + 1e-3
    t = (s - sm) * (rs / ss) + rm
    t = cv2.cvtColor(np.clip(t, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
    return src * (1 - strength) + t * strength


def paste_back(frame: np.ndarray, crop: np.ndarray, mask: np.ndarray, m: np.ndarray) -> None:
    """Composite `crop` (aligned by affine `m`) into `frame` in-place, only touching the face ROI."""
    size = crop.shape[0]
    inv = cv2.invertAffineTransform(m)
    corners = transform_points(np.array([[0, 0], [size, 0], [0, size], [size, size]], np.float32), inv)
    h, w = frame.shape[:2]
    x0, y0 = np.floor(corners.min(0)).astype(int) - 2
    x1, y1 = np.ceil(corners.max(0)).astype(int) + 2
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if x1 <= x0 or y1 <= y0:
        return
    inv_roi = inv.copy()
    inv_roi[:, 2] -= (x0, y0)
    rw, rh = x1 - x0, y1 - y0
    warped = cv2.warpAffine(crop, inv_roi, (rw, rh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    a = cv2.warpAffine(mask, inv_roi, (rw, rh), flags=cv2.INTER_LINEAR, borderValue=0)[:, :, None]
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = np.clip(roi + (warped - roi) * a, 0, 255).astype(np.uint8)
