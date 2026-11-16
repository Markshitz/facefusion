"""Similarity alignment of faces to canonical 5-point templates."""
from __future__ import annotations

import cv2
import numpy as np

# Normalised (0..1) 5-point templates.
TEMPLATES = {
    "arcface_112": np.array([[0.34191607, 0.46157411], [0.65653393, 0.45983393], [0.50022500, 0.64050536],
                             [0.37097589, 0.82469196], [0.63151696, 0.82325089]], np.float32),
    "arcface_128": np.array([[0.36167656, 0.40387734], [0.63696719, 0.40235469], [0.50019687, 0.56044219],
                             [0.38710391, 0.72160547], [0.61507734, 0.72034453]], np.float32),
    "ffhq_512": np.array([[0.37691676, 0.46864664], [0.62285697, 0.46912813], [0.50123859, 0.61331904],
                          [0.39308822, 0.72541100], [0.61150205, 0.72490465]], np.float32),
}


def estimate_matrix(kps: np.ndarray, template: str, size: int) -> np.ndarray:
    dst = TEMPLATES[template] * size
    m, _ = cv2.estimateAffinePartial2D(kps.astype(np.float32), dst, method=cv2.LMEDS)
    if m is None:  # degenerate landmarks; fall back to least squares on 2 eyes + mouth
        m = cv2.getAffineTransform(kps[[0, 1, 3]].astype(np.float32), dst[[0, 1, 3]])
    return m.astype(np.float32)


def warp_face(img: np.ndarray, kps: np.ndarray, template: str, size: int):
    m = estimate_matrix(kps, template, size)
    crop = cv2.warpAffine(img, m, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return crop, m


def transform_points(pts: np.ndarray, m: np.ndarray) -> np.ndarray:
    return pts @ m[:, :2].T + m[:, 2]
