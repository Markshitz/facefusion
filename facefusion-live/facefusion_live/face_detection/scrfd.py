"""SCRFD face detector (det_10g / scrfd_2.5g ONNX) with 5-point landmarks."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from facefusion_live.core.face import Face
from facefusion_live.inference.session import Sessions

STRIDES = (8, 16, 32)
NUM_ANCHORS = 2


def fixed_size_model(path: Path, size: int) -> bytes:
    """The detector with its input fixed to 1x3xSIZExSIZE.

    SCRFD computes its output reshapes from the (dynamic) input size. DirectML rejects those
    run-time shape computations ("Reshape ... The parameter is incorrect"); with a fixed input
    every shape becomes a constant, which all providers handle (and CPU runs slightly faster).
    """
    import onnx
    model = onnx.load(str(path))
    dims = model.graph.input[0].type.tensor_type.shape.dim
    for d, v in zip(dims, (1, 3, size, size)):
        d.ClearField("dim_param")
        d.dim_value = v
    for out in model.graph.output:  # let ONNX Runtime re-infer the output shapes
        out.type.tensor_type.ClearField("shape")
    return model.SerializeToString()


class SCRFD:
    def __init__(self, sessions: Sessions, path: Path):
        self.sessions = sessions
        self.path = Path(path)
        self._sessions: dict = {}
        self.sess = self._session(640)
        self.input_name = self.sess.get_inputs()[0].name
        self._anchor_cache: dict = {}

    def _session(self, size: int):
        s = self._sessions.get(size)
        if s is None:
            s = self.sessions.get(self.path, key=f"{self.path}#{size}", model_bytes=fixed_size_model(self.path, size))
            self._sessions[size] = s
        return s

    def _anchors(self, h: int, w: int, stride: int) -> np.ndarray:
        key = (h, w, stride)
        a = self._anchor_cache.get(key)
        if a is None:
            ys, xs = np.mgrid[:h, :w]
            a = (np.stack([xs, ys], -1).reshape(-1, 2) * stride).astype(np.float32)
            a = np.repeat(a, NUM_ANCHORS, axis=0)
            self._anchor_cache[key] = a
        return a

    def detect(self, img: np.ndarray, size: int = 640, threshold: float = 0.5, max_faces: int = 0) -> list[Face]:
        h, w = img.shape[:2]
        scale = size / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        # 32-aligned square canvas
        canvas = np.zeros((size, size, 3), np.uint8)
        canvas[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128, (size, size), (127.5, 127.5, 127.5), swapRB=True)
        size = int(size)
        outs = self.sessions.run(self._session(size), {self.input_name: blob})
        boxes, kpss, scores = [], [], []
        for i, stride in enumerate(STRIDES):
            sc = outs[i].reshape(-1)
            keep = np.where(sc >= threshold)[0]
            if keep.size == 0:
                continue
            fh, fw = size // stride, size // stride
            anc = self._anchors(fh, fw, stride)[keep]
            bb = outs[i + 3].reshape(-1, 4)[keep] * stride
            kp = outs[i + 6].reshape(-1, 10)[keep] * stride
            boxes.append(np.hstack([anc - bb[:, :2], anc + bb[:, 2:]]))
            kpss.append((kp.reshape(-1, 5, 2) + anc[:, None, :]))
            scores.append(sc[keep])
        if not boxes:
            return []
        boxes = np.vstack(boxes) / scale
        kpss = np.vstack(kpss) / scale
        scores = np.concatenate(scores)
        idx = cv2.dnn.NMSBoxes([(float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])) for b in boxes],
                               scores.tolist(), threshold, 0.4)
        idx = np.array(idx).reshape(-1)
        faces = [Face(bbox=boxes[i].astype(np.float32), kps=kpss[i].astype(np.float32), score=float(scores[i]))
                 for i in idx]
        faces.sort(key=lambda f: f.area, reverse=True)
        return faces[:max_faces] if max_faces else faces
