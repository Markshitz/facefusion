"""inswapper_128 identity swap with optional "pixel boost" super-sampling.

Pixel boost crops the target at N*128 px and swaps the N*N interleaved
128 px sub-grids separately, then re-interleaves them. This yields an
N*128 output with real detail instead of an upscaled 128 px patch.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from facefusion_live.face_swap import subgrid

from facefusion_live.inference.session import Sessions

log = logging.getLogger(__name__)


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


def fixed_batch_model(path: Path, batch: int) -> bytes:
    """inswapper with a fixed batch size, so the boost x boost sub-grids run as ONE call.

    Fixed (not dynamic) dims keep every shape constant, which DirectML requires; outputs are identical.
    """
    import onnx
    model = onnx.load(str(path))
    for v in list(model.graph.input) + list(model.graph.output):
        d = v.type.tensor_type.shape.dim[0]
        d.ClearField("dim_param")
        d.dim_value = batch
    return model.SerializeToString()


class InSwapper:
    SIZE = 128

    def __init__(self, sessions: Sessions, path: Path):
        self.sessions = sessions
        self.path = Path(path)
        self.sess = sessions.get(path)
        self._batched: dict = {1: self.sess}
        self._batch_ok = True  # set False if a batched run ever returns inconsistent sub-grids (seen on DirectML)
        self._batch_verified: set = set()  # batch sizes whose output was checked against single runs
        self.emap = load_emap(path)

    def _session(self, batch: int):
        s = self._batched.get(batch)
        if s is None:
            s = self.sessions.get(self.path, key=f"{self.path}#b{batch}", model_bytes=fixed_batch_model(self.path, batch))
            self._batched[batch] = s
        return s

    def source_latent(self, embedding: np.ndarray) -> np.ndarray:
        lat = embedding.reshape(1, -1).astype(np.float32) @ self.emap
        return (lat / np.linalg.norm(lat)).astype(np.float32)

    def _run_batch(self, crops_bgr: list, latent: np.ndarray) -> np.ndarray:
        """Swap N aligned 128 px crops in one inference call. Returns (N,128,128,3) float32 BGR 0..255."""
        n = len(crops_bgr)
        x = np.stack([c[:, :, ::-1] for c in crops_bgr]).astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        lat = np.repeat(latent.reshape(1, -1), n, axis=0).astype(np.float32)
        y = self.sessions.run(self._session(n), {"target": np.ascontiguousarray(x), "source": lat})[0]
        return (np.clip(y.transpose(0, 2, 3, 1), 0, 1) * 255)[..., ::-1]

    def swap(self, crop_bgr: np.ndarray, latent: np.ndarray, boost: int = 1) -> np.ndarray:
        """crop_bgr is (boost*128)^2 uint8 aligned with arcface_128. Returns float32 BGR 0..255.

        Pixel boost swaps the boost x boost interleaved 128 px sub-grids in a single batched call.
        """
        if boost == 1:
            return self._run_batch([crop_bgr], latent)[0]
        grid = [np.ascontiguousarray(crop_bgr[i::boost, j::boost]) for i in range(boost) for j in range(boost)]
        if self._batch_ok:
            ys = self._run_batch(grid, latent)
            bad = subgrid.mismatched(ys)
            n = len(grid)
            if not bad and n not in self._batch_verified:  # once per batch size: first and last item vs single runs
                self._batch_verified.add(n)
                bad = any(float(np.abs(self._run_batch([grid[k]], latent)[0] - ys[k]).mean()) > 2.0 for k in (0, n - 1))
            if bad:  # the batched GPU run went wrong: never batch again, redo one at a time
                log.warning("Batched pixel boost gave inconsistent results on this device; running sub-grids one by one")
                self._batch_ok = False
                ys = np.stack([self._run_batch([g], latent)[0] for g in grid])
        else:
            ys = np.stack([self._run_batch([g], latent)[0] for g in grid])
        return subgrid.assemble(crop_bgr.shape, ys, boost)
