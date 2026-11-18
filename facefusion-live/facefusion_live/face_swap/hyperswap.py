"""HyperSwap 256 (FaceFusion, 2025): a newer identity-swap model working at 256 px.

Same interface as InSwapper. Inputs follow FaceFusion's reference pipeline: target = arcface_128-aligned
256 px crop, RGB, normalised to [-1, 1]; source = L2-normalised ArcFace embedding. Licence: ResearchRAIL.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from facefusion_live.face_swap import subgrid
from facefusion_live.inference.session import Sessions


class HyperSwapper:
    SIZE = 256

    def __init__(self, sessions: Sessions, path: Path):
        self.sessions = sessions
        self.sess = sessions.get(path)

    def source_latent(self, embedding: np.ndarray) -> np.ndarray:
        e = embedding.reshape(1, -1).astype(np.float32)
        return e / np.linalg.norm(e)

    def _run(self, crop_bgr: np.ndarray, latent: np.ndarray) -> np.ndarray:
        x = (crop_bgr[:, :, ::-1].astype(np.float32) / 255.0 - 0.5) / 0.5
        y = self.sessions.run(self.sess, {"target": np.ascontiguousarray(x.transpose(2, 0, 1)[None]), "source": latent})[0][0]
        y = np.clip(y.transpose(1, 2, 0) * 0.5 + 0.5, 0, 1)
        return (y * 255)[:, :, ::-1].astype(np.float32)

    def swap(self, crop_bgr: np.ndarray, latent: np.ndarray, boost: int = 1) -> np.ndarray:
        if boost == 1:
            return self._run(crop_bgr, latent)
        ys = np.stack([self._run(np.ascontiguousarray(crop_bgr[i::boost, j::boost]), latent)
                       for i in range(boost) for j in range(boost)])
        return subgrid.assemble(crop_bgr.shape, ys, boost)
