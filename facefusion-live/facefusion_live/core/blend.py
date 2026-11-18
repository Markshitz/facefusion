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


def frontalness(lm68: np.ndarray) -> float:
    """0 = full profile, 1 = facing the camera: balance of nose-to-jaw-edge distances on both sides."""
    a = float(np.linalg.norm(lm68[30] - lm68[2]))
    b = float(np.linalg.norm(lm68[30] - lm68[14]))
    return min(a, b) / max(a, b, 1e-6)


def mouth_landmarks_ok(lm_now: np.ndarray, lm_target: np.ndarray) -> bool:
    """The measured swapped mouth and the original mouth must roughly agree; otherwise a warp drags in garbage."""
    width = float(np.linalg.norm(lm_target[54] - lm_target[48]))
    if width < 6:
        return False
    shift = np.linalg.norm(lm_now[48:68] - lm_target[48:68], axis=1).max()
    return bool(shift < 0.6 * width)


PROFILE = 0.35  # frontalness below this: mouth landmarks are guesses, leave the mouth to the swap model


def face_hull_mask(lm68_crop: np.ndarray, size: int, grow: float = 0.04) -> np.ndarray:
    """Mask of the real face outline (jaw line + brows raised to the hairline) in crop space, feathered.

    On a turned head the square swap crop covers background in front of the profile; without this the swapped
    face is painted over it semi-transparently (a "ghost" face with the background showing through).
    """
    jaw = lm68_crop[0:17]
    brows = lm68_crop[26:16:-1]
    chin = lm68_crop[8]
    top = brows + (brows - chin) * 0.22  # forehead above the brows (the forehead fade blends it further)
    pts = np.vstack([jaw, top]).astype(np.float32)
    c = pts.mean(0)
    pts = c + (pts - c) * (1 + grow)
    m = np.zeros((size, size), np.float32)
    cv2.fillPoly(m, [cv2.convexHull(pts).astype(np.int32)], 1.0, cv2.LINE_AA)
    return cv2.GaussianBlur(m, (0, 0), max(size * 0.015, 1.0))


def _warp_triangle(src: np.ndarray, dst: np.ndarray, t_src: np.ndarray, t_dst: np.ndarray) -> None:
    """Affine-warp one triangle of `src` into `dst` (in-place)."""
    r1, r2 = cv2.boundingRect(t_src.astype(np.float32)), cv2.boundingRect(t_dst.astype(np.float32))
    if r1[2] < 1 or r1[3] < 1 or r2[2] < 1 or r2[3] < 1:
        return
    a = (t_src - r1[:2]).astype(np.float32)
    b = (t_dst - r2[:2]).astype(np.float32)
    patch = src[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    if patch.size == 0:
        return
    m = cv2.getAffineTransform(a, b)
    warped = cv2.warpAffine(patch, m, (r2[2], r2[3]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    mask = np.zeros((r2[3], r2[2]), np.float32)
    cv2.fillConvexPoly(mask, b.astype(np.int32), 1.0, cv2.LINE_AA)
    y0, x0 = r2[1], r2[0]
    roi = dst[y0:y0 + r2[3], x0:x0 + r2[2]]
    h, w = roi.shape[:2]
    if h == 0 or w == 0:
        return
    mk = mask[:h, :w, None]
    roi[:] = (roi * (1 - mk) + warped[:h, :w] * mk).astype(roi.dtype)


# mouth control points (2DFAN4 68 indices) and fixed anchors around them (nose base, cheeks, jaw)
_MOUTH = list(range(48, 68))
_ANCHORS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 31, 32, 33, 34, 35]


def warp_mouth(frame: np.ndarray, lm_now: np.ndarray, lm_target: np.ndarray) -> None:
    """Reshape the swapped mouth (landmarks lm_now, measured on `frame`) so its lips land on lm_target.

    Piecewise-affine warp over a Delaunay mesh of the lip points plus fixed anchors (nose base, jaw):
    the mouth opens and closes exactly like the original while keeping the swapped (source) lip texture.
    """
    src_pts = np.vstack([lm_now[_MOUTH], lm_now[_ANCHORS]]).astype(np.float32)
    dst_pts = np.vstack([lm_target[_MOUTH], lm_now[_ANCHORS]]).astype(np.float32)
    h, w = frame.shape[:2]
    x0, y0 = np.floor(np.minimum(src_pts.min(0), dst_pts.min(0))).astype(int) - 2
    x1, y1 = np.ceil(np.maximum(src_pts.max(0), dst_pts.max(0))).astype(int) + 2
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w - 1), min(y1, h - 1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return
    sub = cv2.Subdiv2D((int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)))
    index = {}
    for k, p in enumerate(dst_pts):
        q = (float(np.clip(p[0], x0, x1)), float(np.clip(p[1], y0, y1)))
        index[(round(q[0], 2), round(q[1], 2))] = k
        sub.insert(q)
    src = frame.copy()
    for t in sub.getTriangleList():
        tri = t.reshape(3, 2)
        ids = [index.get((round(float(x), 2), round(float(y), 2))) for x, y in tri]
        if None in ids:
            continue
        _warp_triangle(src, frame, src_pts[ids], dst_pts[ids])


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
