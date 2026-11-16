"""inswapper_128 identity swap with optional "pixel boost" super-sampling.

Pixel boost crops the target at N*128 px and swaps the N*N interleaved
128 px sub-grids separately, then re-interleaves them. This yields an
N*128 output with real detail instead of an upscaled 128 px patch.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from facefusion_live.inference.session import Sessions


def load_emap(path: Path) -> np.ndarray:
    cache = path.with_suffix(".emap.npy")
    if cache.exists():
        return np.load(cache)
    import onnx
    from onnx import numpy_helper
    model = onnx.load(str(path))
    emap = numpy_helper.to_array(model.graph.initializer[-1]).astype(np.float32)
    try:
        np.save(cache, emap)
    except OSError:
        pass
    return emap


class InSwapper:
    SIZE = 128

    def __init__(self, sessions: Sessions, path: Path):
        self.sessions = sessions
        self.sess = sessions.get(path)
        self.emap = load_emap(path)

    def source_latent(self, embedding: np.ndarray) -> np.ndarray:
        lat = embedding.reshape(1, -1).astype(np.float32) @ self.emap
        return (lat / np.linalg.norm(lat)).astype(np.float32)

    def _run(self, crop_bgr: np.ndarray, latent: np.ndarray) -> np.ndarray:
        x = crop_bgr[:, :, ::-1].astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        y = self.sessions.run(self.sess, {"target": np.ascontiguousarray(x), "source": latent})[0][0]
        return (np.clip(y.transpose(1, 2, 0), 0, 1) * 255)[:, :, ::-1]

    def swap(self, crop_bgr: np.ndarray, latent: np.ndarray, boost: int = 1) -> np.ndarray:
        """crop_bgr is (boost*128)^2 uint8 aligned with arcface_128. Returns float32 BGR 0..255."""
        if boost == 1:
            return self._run(crop_bgr, latent)
        out = np.empty(crop_bgr.shape, np.float32)
        for i in range(boost):
            for j in range(boost):
                out[i::boost, j::boost] = self._run(np.ascontiguousarray(crop_bgr[i::boost, j::boost]), latent)
        return out
