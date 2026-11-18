"""Face record shared by all stages."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Face:
    bbox: np.ndarray                 # (4,) x1 y1 x2 y2
    kps: np.ndarray                  # (5, 2) eyes, nose, mouth corners
    score: float
    embedding: Optional[np.ndarray] = None   # (512,) L2-normalised ArcFace
    landmarks68: Optional[np.ndarray] = None
    track_id: int = -1
    extra: dict = field(default_factory=dict)

    @property
    def area(self) -> float:
        w, h = self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1]
        return float(max(w, 0) * max(h, 0))

    @property
    def center(self) -> np.ndarray:
        return (self.bbox[:2] + self.bbox[2:]) / 2

    def similarity(self, other: "Face") -> float:
        if self.embedding is None or other.embedding is None:
            return 0.0
        return float(np.dot(self.embedding, other.embedding))


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0
