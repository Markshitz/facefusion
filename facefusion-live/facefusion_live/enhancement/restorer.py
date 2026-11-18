"""GFPGAN / CodeFormer face restoration (512 px, FFHQ alignment)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from facefusion_live.core.align import warp_face
from facefusion_live.core.blend import box_mask, paste_back
from facefusion_live.inference.session import Sessions


class FaceRestorer:
    SIZE = 512

    def __init__(self, sessions: Sessions, path: Path, kind: str):
        self.sessions = sessions
        self.sess = sessions.get(path)
        self.kind = kind
        self.inputs = {i.name for i in self.sess.get_inputs()}
        shape = self.sess.get_inputs()[0].shape
        self.size = shape[2] if isinstance(shape[2], int) else 512

    def restore_crop(self, crop: np.ndarray, fidelity: float = 0.7) -> np.ndarray:
        x = (crop[:, :, ::-1].astype(np.float32) / 255.0 - 0.5) / 0.5
        feeds = {self.sess.get_inputs()[0].name: np.ascontiguousarray(x.transpose(2, 0, 1)[None])}
        if "weight" in self.inputs:
            feeds["weight"] = np.array(fidelity, dtype=np.float64)
        y = self.sessions.run(self.sess, feeds)[0][0]
        y = (np.clip(y, -1, 1) + 1) / 2
        return (y.transpose(1, 2, 0) * 255)[:, :, ::-1].astype(np.float32)

    def enhance(self, frame: np.ndarray, kps: np.ndarray, strength: float, fidelity: float = 0.7,
                region: np.ndarray | None = None) -> None:
        """In-place enhancement of the face at `kps`. `strength` blends with the unenhanced face
        so skin texture is not over-smoothed."""
        if strength <= 0:
            return
        crop, m = warp_face(frame, kps, "ffhq_512", self.size)
        out = self.restore_crop(crop, fidelity)
        mask = box_mask(self.size, 0.3, 0.05) * float(np.clip(strength, 0, 1))
        if region is not None:
            mask = mask * region
        paste_back(frame, out, mask, m)
