"""Model-based masks: XSeg occlusion and BiSeNet face-region parsing."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from facefusion_live.core.blend import soften
from facefusion_live.inference.session import Sessions

# BiSeNet classes that belong to the face surface (excludes hair, hat, hands, background).
FACE_REGIONS = (1, 2, 3, 4, 5, 6, 10, 11, 12, 13)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class OcclusionMasker:
    def __init__(self, sessions: Sessions, path: Path):
        self.sessions = sessions
        self.sess = sessions.get(path)
        self.inp = self.sess.get_inputs()[0].name

    def __call__(self, crop: np.ndarray) -> np.ndarray:
        x = cv2.resize(crop, (256, 256)).astype(np.float32)[None] / 255.0
        y = self.sessions.run(self.sess, {self.inp: x})[0][0, :, :, 0]
        y = cv2.resize(np.clip(y, 0, 1), crop.shape[:2][::-1])
        return soften(y)


class RegionMasker:
    def __init__(self, sessions: Sessions, path: Path):
        self.sessions = sessions
        self.sess = sessions.get(path)
        self.inp = self.sess.get_inputs()[0].name

    def __call__(self, crop: np.ndarray, regions=FACE_REGIONS) -> np.ndarray:
        x = cv2.resize(crop, (512, 512))[:, :, ::-1].astype(np.float32) / 255.0
        x = ((x - MEAN) / STD).transpose(2, 0, 1)[None]
        y = self.sessions.run(self.sess, {self.inp: np.ascontiguousarray(x)})[0][0].argmax(0)
        m = np.isin(y, regions).astype(np.float32)
        m = cv2.resize(m, crop.shape[:2][::-1])
        return soften(m)
