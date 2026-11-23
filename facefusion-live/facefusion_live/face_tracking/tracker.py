"""Multi-face tracker shared by video and live modes.

* Full-frame detection only every `detect_interval` frames (or when a track
  is lost); in between, each track is re-detected inside a small ROI around
  its last position (a 192 px detector pass instead of 640 px).
* Landmarks are smoothed with a One-Euro filter: strong smoothing when the
  head is still (kills jitter/flicker), little lag when it moves fast.
* Tracks keep a stable id and an identity embedding that is refreshed
  periodically, so face→source assignments stay consistent.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from facefusion_live.core.face import Face, iou


class OneEuro:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.02, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x: Optional[np.ndarray] = None
        self.dx: Optional[np.ndarray] = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, dt: float, scale: float = 1.0) -> np.ndarray:
        if self.x is None:
            self.x, self.dx = x.copy(), np.zeros_like(x)
            return x
        dx = (x - self.x) / dt / scale  # normalise speed by face size
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * np.abs(self.dx)
        a = 1.0 / (1.0 + (1.0 / (2 * math.pi * cutoff)) / dt)
        self.x = a * x + (1 - a) * self.x
        return self.x


@dataclass
class Track:
    id: int
    face: Face
    embedding: Optional[np.ndarray] = None
    missed: int = 0
    age: int = 0
    since_embed: int = 0
    filt: OneEuro = field(default_factory=OneEuro)


class FaceTracker:
    def __init__(self, engine, max_missed: int = 8, embed_every: int = 30, roi_size: int = 192):
        self.engine = engine
        self.max_missed = max_missed
        self.embed_every = embed_every
        self.roi_size = roi_size
        self.tracks: list[Track] = []
        self._next = 0
        self._frame = 0
        self.stats = {"full": 0, "roi": 0}
        self._img = None

    def reset(self) -> None:
        self.tracks.clear()
        self._frame = 0

    def _smooth(self, t: Track, f: Face, dt: float) -> None:
        s = self.engine.settings.smoothing
        if s <= 0:
            return
        size = max(f.bbox[2] - f.bbox[0], 1.0)
        # map smoothing 0..1 to One-Euro min_cutoff (Hz): low cutoff = heavier smoothing
        t.filt.min_cutoff = 0.4 + (1 - s) * 8.0
        t.filt.beta = 0.02
        f.kps = t.filt(f.kps.astype(np.float64), dt, size / 100.0).astype(np.float32)

    def _roi_detect(self, img: np.ndarray, t: Track) -> Optional[Face]:
        b = t.face.bbox
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        half = max(b[2] - b[0], b[3] - b[1]) * 1.1
        h, w = img.shape[:2]
        x0, y0 = int(max(cx - half, 0)), int(max(cy - half, 0))
        x1, y1 = int(min(cx + half, w)), int(min(cy + half, h))
        if x1 - x0 < 16 or y1 - y0 < 16:
            return None
        faces = self.engine.detector.detect(img[y0:y1, x0:x1], self.roi_size, self.engine.settings.det_threshold)
        if not faces:
            return None
        off = np.array([x0, y0], np.float32)
        best = max(faces, key=lambda f: f.score * (1 + iou(np.r_[f.bbox[:2] + off, f.bbox[2:] + off], b)))
        best.bbox = np.r_[best.bbox[:2] + off, best.bbox[2:] + off].astype(np.float32)
        best.kps = best.kps + off
        return best

    def update(self, img: np.ndarray, dt: float = 1 / 30, need_embedding: bool = True) -> list[Face]:
        """Return faces visible in this frame (with track ids, smoothed kps, embeddings)."""
        st = self.engine.settings
        self._img = img
        self._frame += 1
        interval = max(1, int(st.detect_interval)) if st.tracking else 1
        full = (not self.tracks) or self._frame % interval == 0 or all(t.missed for t in self.tracks)
        if full:
            self.stats["full"] += 1
            dets = self.engine.detect(img)
            unmatched = list(range(len(dets)))
            pairs = sorted(((iou(t.face.bbox, d.bbox), ti, di) for ti, t in enumerate(self.tracks)
                            for di, d in enumerate(dets)), reverse=True)
            used_t: set = set()
            for ov, ti, di in pairs:
                if ov < 0.3 or ti in used_t or di not in unmatched:
                    continue
                used_t.add(ti)
                unmatched.remove(di)
                self._assign(self.tracks[ti], dets[di], dt)
            for ti, t in enumerate(self.tracks):
                if ti not in used_t:
                    t.missed += 1
            for di in unmatched:
                self._refine(dets[di])
                t = Track(self._next, dets[di])
                self._next += 1
                dets[di].track_id = t.id
                self.tracks.append(t)
        else:
            self.stats["roi"] += 1
            for t in self.tracks:
                f = self._roi_detect(img, t)
                if f is None:
                    t.missed += 1
                else:
                    self._assign(t, f, dt)
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        visible = []
        for t in self.tracks:
            if t.missed:
                continue
            t.age += 1
            t.since_embed += 1
            if need_embedding and (t.embedding is None or t.since_embed >= self.embed_every):
                e = self.engine.embedder.embed(img, t.face.kps)
                # running average keeps identity stable across expressions
                t.embedding = e if t.embedding is None else _norm(0.7 * t.embedding + 0.3 * e)
                t.since_embed = 0
            t.face.embedding = t.embedding
            t.face.track_id = t.id
            visible.append(t.face)
        return visible

    def _refine(self, f: Face) -> None:
        """Precise 68-point landmarks (lips, eyes) for this frame, if the engine provides them."""
        refine = getattr(self.engine, "refine", None)
        if refine is not None and self._img is not None:
            refine(self._img, f)

    def _assign(self, t: Track, f: Face, dt: float) -> None:
        self._refine(f)
        self._smooth(t, f, dt)
        f.track_id = t.id
        f.embedding = t.embedding
        t.face = f
        t.missed = 0


def _norm(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-8)
