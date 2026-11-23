"""68-point landmarks estimated from the 5 detector points (fan_68_5, <1 MB, ~0.1 ms)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from facefusion_live.core.align import estimate_matrix, transform_points
from facefusion_live.inference.session import Sessions


class Landmark68From5:
    def __init__(self, sessions: Sessions, path: Path):
        self.sessions = sessions
        self.sess = sessions.get(path)

    def __call__(self, kps: np.ndarray) -> np.ndarray:
        m = estimate_matrix(kps, "ffhq_512", 1)
        pts = transform_points(kps, m).reshape(1, 5, 2).astype(np.float32)
        out = self.sessions.run(self.sess, {"input": pts})[0][0]
        inv = cv2.invertAffineTransform(m)
        return transform_points(out, inv).astype(np.float32)
