"""2DFAN4: 68 facial landmarks measured from the image (eyes, brows, nose, outer and inner lips, jaw).

Used for precise alignment (5 points derived from 68) and for the mouth mask, so the swap follows real lip
and head movement instead of the detector's coarse 5 points. Model: face-alignment (BSD-3), ONNX via FaceFusion.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from facefusion_live.core.align import transform_points
from facefusion_live.inference.session import Sessions

SIZE = 256


class FAN68:
    def __init__(self, sessions: Sessions, path: Path):
        self.sessions = sessions
        self.sess = sessions.get(path)
        self.inp = self.sess.get_inputs()[0].name

    def __call__(self, img: np.ndarray, bbox: np.ndarray) -> tuple[np.ndarray, float]:
        """Returns (68x2 landmarks in image coords, confidence 0..1)."""
        x1, y1, x2, y2 = bbox
        scale = 195.0 / max(x2 - x1, y2 - y1, 1.0)
        tx, ty = (SIZE - (x2 + x1) * scale) * 0.5, (SIZE - (y2 + y1) * scale) * 0.5
        m = np.array([[scale, 0, tx], [0, scale, ty]], np.float32)
        crop = cv2.warpAffine(img, m, (SIZE, SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        if lab[:, :, 0].mean() < 30:  # very dark faces: local contrast boost
            lab[:, :, 0] = cv2.createCLAHE(clipLimit=2).apply(lab[:, :, 0])
            crop = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        x = crop[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0  # RGB
        lm, heat = self.sessions.run(self.sess, {self.inp: np.ascontiguousarray(x)})
        pts = lm[0, :, :2] / 64.0 * SIZE
        pts = transform_points(pts, cv2.invertAffineTransform(m)).astype(np.float32)
        return pts, float(np.mean(np.amax(heat[0], axis=(1, 2))))


def five_from_68(lm: np.ndarray) -> np.ndarray:
    """Eye centres, nose tip and mouth corners (the alignment template's 5 points) from 68 landmarks."""
    return np.array([lm[36:42].mean(0), lm[42:48].mean(0), lm[30], lm[48], lm[54]], np.float32)


def mouth_open_ratio(lm: np.ndarray) -> float:
    """Inner-lip opening / mouth width (0 = closed)."""
    gap = np.linalg.norm(lm[[61, 62, 63]] - lm[[67, 66, 65]], axis=1).mean()
    return float(gap / (np.linalg.norm(lm[54] - lm[48]) + 1e-6))
