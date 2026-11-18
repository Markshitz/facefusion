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


_fade_cache: dict = {}


def forehead_fade(size: int, amount: float) -> np.ndarray:
    """Vertical ramp (size x 1) that fades the top `amount` of the swap crop into the original.

    The swap crop ends mid-forehead; a short edge there shows as a horizontal line when the forehead is bare
    or lit differently. A long smooth fade hides that boundary without touching eyes, nose or mouth.
    """
    key = (size, round(amount, 3))
    r = _fade_cache.get(key)
    if r is None:
        t = np.clip((np.arange(size, dtype=np.float32) - size * 0.02) / max(size * amount, 1.0), 0, 1)
        r = (t * t * (3 - 2 * t))[:, None]
        _fade_cache[key] = r
    return r


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


def restore_mouth(frame: np.ndarray, original: np.ndarray, lm68: np.ndarray, region: str = "inner",
                  strength: float = 1.0) -> None:
    """Blend the original mouth back into `frame` (in-place), feathered, only around the mouth.

    region "inner": inside the inner lip line (60-67) -> real teeth, tongue and mouth opening.
    region "lips" : the outer lip line (48-59) -> the target's own lips as well.
    """
    inner = lm68[60:68].astype(np.float32)
    pts = (inner if region == "inner" else lm68[48:60]).astype(np.float32)
    width = float(np.linalg.norm(lm68[54] - lm68[48]))
    if width < 4:
        return
    grow = 0.12 if region == "inner" else 0.06
    c = pts.mean(0)
    pts = c + (pts - c) * (1 + grow)
    ci = inner.mean(0)
    inner = ci + (inner - ci) * 1.12
    pad = int(width * 0.35) + 4
    h, w = frame.shape[:2]
    x0, y0 = np.floor(pts.min(0)).astype(int) - pad
    x1, y1 = np.ceil(pts.max(0)).astype(int) + pad
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if x1 <= x0 or y1 <= y0:
        return
    m = np.zeros((y1 - y0, x1 - x0), np.float32)
    cv2.fillPoly(m, [(pts - (x0, y0)).astype(np.int32)], float(np.clip(strength, 0, 1)))
    cv2.fillPoly(m, [(inner - (x0, y0)).astype(np.int32)], 1.0)  # the mouth opening itself is always the original
    m = cv2.GaussianBlur(m, (0, 0), max(width * 0.04, 1.0))[:, :, None]
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (roi + (original[y0:y1, x0:x1].astype(np.float32) - roi) * m).astype(np.uint8)


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
