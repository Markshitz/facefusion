"""ArcFace identity embedding."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from facefusion_live.core.align import warp_face
from facefusion_live.inference.session import Sessions


class ArcFace:
    def __init__(self, sessions: Sessions, path: Path):
        self.sessions = sessions
        self.sess = sessions.get(path)
        self.input_name = self.sess.get_inputs()[0].name

    def embed(self, img: np.ndarray, kps: np.ndarray) -> np.ndarray:
        crop, _ = warp_face(img, kps, "arcface_112", 112)
        blob = cv2.dnn.blobFromImage(crop, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        e = self.sessions.run(self.sess, {self.input_name: blob})[0][0].astype(np.float32)
        return e / (np.linalg.norm(e) + 1e-8)
